"""
agent.py
--------
MILESTONE 5: The Agent Loop

This is the "brain" of the system. Here's what it does in plain English:

    1. You give it a question (like "why did revenue drop?")
    2. It sends the question to OpenAI, along with the list of available tools
    3. OpenAI either:
       a) Answers directly with text → we're done
       b) Says "I need to call a tool" → we run the tool and send results back
    4. Step 3 repeats until OpenAI gives a final text answer (or we hit max rounds)

The @retry decorator on call_model means: if the API fails (timeout, rate limit),
wait a bit and try again automatically. This is the error handling from Milestone 4.

MILESTONE 10: ReAct Agent Architecture

Implements a ReAct-style KPI investigation agent:
- Reason → decide next action
- Act → call a tool (or retrieval helper)
- Observe → append result to memory
- Repeat until final answer

This extends the original tool-calling loop with:
1. Explicit planning / reasoning traces
2. Memory + state tracking across steps
3. Support for multi-step interactions
4. Optional RAG retrieval as an agent tool
"""

import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))  # Ensure current directory is in path for imports

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception_type

from src.tools import TOOL_DEFINITIONS, TOOL_IMPLEMENTATIONS
from src.rag_pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Load API key from .env file
# ---------------------------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Which model to use — you can change this
MODEL = "gpt-4o-mini"

selected_metrics = st.session_state.get("perf_metrics", [])
selected_groups = st.session_state.get("group_cols", [])

anomaly_query = (
    f"Analyze the KPI dataset. The anomaly appears to start around {anomaly_date}. "
    f"Focus on these performance metrics if relevant: {selected_metrics}. "
    f"Use these grouping columns if relevant: {selected_groups}. "
    f"Identify which metrics and segments are most affected. "
    f"Use the available tools to query the data and provide a structured finding."
)

# ===========================================================================
# MEMORY / STATE
# ==========================================================================

@dataclass
class AgentState:
    user_prompt: str
    current_step: int = 0
    completed_steps: List[str] = field(default_factory=list)
    thoughts: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None

    def record_thought(self, text: str) -> None:
        self.thoughts.append(text)

    def record_action(self, action_type: str, payload: Dict[str, Any]) -> None:
        self.actions.append({"type": action_type, "payload": payload})

    def record_observation(self, observation: Dict[str, Any]) -> None:
        self.observations.append(observation)

    def mark_step_complete(self, label: str) -> None:
        if label not in self.completed_steps:
            self.completed_steps.append(label)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_prompt": self.user_prompt,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "thoughts": self.thoughts,
            "actions": self.actions,
            "observations": self.observations,
            "final_answer": self.final_answer,
        }

# ===========================================================================
# API CALL WITH RETRY (Milestone 4, Task 3: Error Handling)
# ===========================================================================

@retry(
    wait=wait_exponential_jitter(initial=0.5, max=8),   # wait 0.5s, then 1s, 2s, 4s...
    stop=stop_after_attempt(4),                           # give up after 4 tries
    retry=retry_if_exception_type(Exception),             # retry on ANY error
)
def call_model(*, messages, tools=None, tool_choice="auto", max_tokens=900):
    """
    Call OpenAI API with automatic retry on failure.

    Why retry?
        APIs can fail randomly — server overload, network hiccup, rate limit.
        Instead of crashing, we wait and try again. The exponential backoff
        means we wait longer each time so we don't hammer the server.

    Parameters:
        messages:    The conversation so far (system prompt + user message + tool results)
        tools:       The tool definitions (the "menu")
        tool_choice: "auto" = model decides, "none" = force text response
        max_tokens:  Maximum length of response
    """
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    # Only include tools if we have them and aren't forcing "none"
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    return client.chat.completions.create(**kwargs)

# ===========================================================================
# RAG TOOL
# ===========================================================================

def rag_lookup(query: str, k: int = 4) -> Dict[str, Any]:
    """
    Lightweight retrieval tool for ReAct.
    """
    try:
        rag = RAGPipeline()
        rag.build_or_load()
        docs = rag.retrieve(query, k=k)

        results = []
        for d in docs:
            results.append({
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "content": d.page_content[:1200],
            })

        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": f"RAG lookup failed: {str(e)}"}


REACT_TOOL_DEFINITIONS = TOOL_DEFINITIONS + [
    {
        "type": "function",
        "function": {
            "name": "rag_lookup",
            "description": (
                "Retrieve relevant knowledge-base context for KPI investigation. "
                "Use this when business context, RCA patterns, or KPI definitions would help."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What context to retrieve from the knowledge base",
                    },
                    "k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Number of chunks to retrieve",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]

REACT_TOOL_IMPLEMENTATIONS = {
    **TOOL_IMPLEMENTATIONS,
    "rag_lookup": rag_lookup,
}


# ===========================================================================
# SYSTEM PROMPT
# ===========================================================================

REACT_SYSTEM_PROMPT = """
You are a ReAct-style KPI root-cause analysis agent.

Your role:
- Investigate KPI changes using iterative reasoning and tool use.
- Think step by step before deciding whether to act.
- Use tools to gather real evidence.
- Store progress in the conversation state.
- Stop when you have enough evidence to answer well.

Available capabilities:
1. query_kpi_data → pull KPI data from the dataset
2. compute_kpi_stats → compute deterministic math/statistics
3. rag_lookup → retrieve business context from the knowledge base

ReAct workflow:
- Thought: What do I need to know next?
- Action: Call a tool only when needed
- Observation: Use the tool result to update your understanding
- Repeat until the task is complete
- Final: Provide a structured answer

Rules:
- Do not guess numbers.
- Use tools for data and calculations.
- Use rag_lookup when context from the knowledge base would improve the explanation.
- Avoid repeating the same tool call unless parameters materially differ.
- Distinguish evidence from hypotheses.
- Final answer format:
  Summary
  Evidence
  Plausible explanations
  Recommendations
"""

THOUGHT_PROMPT_TEMPLATE = """
Given the user request and prior observations, decide the single best next step.

User request:
{user_prompt}

Current state summary:
{state_summary}

Return a short reasoning note for what should happen next.
"""
# ===========================================================================
# HELPERS
# ===========================================================================

def summarize_state(state: AgentState) -> str:
    last_obs = state.observations[-3:] if state.observations else []
    return json.dumps({
        "completed_steps": state.completed_steps,
        "recent_actions": state.actions[-3:],
        "recent_observations": last_obs,
    }, indent=2)[:3000]


def generate_next_thought(state: AgentState, debug: bool = False) -> str:
    prompt = THOUGHT_PROMPT_TEMPLATE.format(
        user_prompt=state.user_prompt,
        state_summary=summarize_state(state),
    )

    response = call_model(
        messages=[
            {"role": "system", "content": "You are a planning module for a ReAct agent."},
            {"role": "user", "content": prompt},
        ],
        tools=None,
        tool_choice="none",
        max_tokens=200,
    )

    thought = response.choices[0].message.content or ""
    state.record_thought(thought)

    if debug:
        print(f"\n[Thought {state.current_step}] {thought}")

    return thought


def execute_tool_call(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name not in REACT_TOOL_IMPLEMENTATIONS:
        return {"success": False, "error": f"Unknown tool: '{tool_name}'"}

    try:
        return REACT_TOOL_IMPLEMENTATIONS[tool_name](**tool_args)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool": tool_name,
            "args": tool_args,
        }


# ===========================================================================
# REACT AGENT LOOP
# ===========================================================================

def run_react_agent(
    user_prompt: str,
    system_prompt: str = REACT_SYSTEM_PROMPT,
    max_rounds: int = 6,
    debug: bool = True,
    return_state: bool = False,
):
    """
    ReAct loop with memory/state.
    """
    state = AgentState(user_prompt=user_prompt)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for round_num in range(max_rounds):
        state.current_step = round_num + 1

        # Step 1: explicit reasoning / planning
        thought = generate_next_thought(state, debug=debug)
        messages.append({
            "role": "system",
            "content": f"Internal planning note for this step:\n{thought}"
        })

        # Step 2: ask model whether to act or answer
        response = call_model(
            messages=messages,
            tools=REACT_TOOL_DEFINITIONS,
            tool_choice="auto",
            max_tokens=1000,
        )

        response_message = response.choices[0].message

        if debug:
            print(f"\n[Round {round_num + 1}] ", end="")
            if response_message.tool_calls:
                names = [tc.function.name for tc in response_message.tool_calls]
                print(f"TOOL CALLS → {names}")
            else:
                print("FINAL TEXT")

        # Final answer
        if not response_message.tool_calls:
            final_text = response_message.content or ""
            state.final_answer = final_text
            state.mark_step_complete("final_response")

            if return_state:
                return {"final_answer": final_text, "state": state.to_dict()}
            return final_text

        messages.append(response_message)

        # Execute tools
        for tool_call in response_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            state.record_action(tool_name, tool_args)

            if debug:
                print(f"  → Executing {tool_name} with args: {json.dumps(tool_args)[:300]}")

            result = execute_tool_call(tool_name, tool_args)

            state.record_observation({
                "tool": tool_name,
                "result": result,
            })
            state.mark_step_complete(tool_name)

            if debug:
                print(f"  ← success={result.get('success', 'n/a')}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    fallback = "[ReAct agent stopped] Reached maximum rounds without a final answer."
    state.final_answer = fallback

    if return_state:
        return {"final_answer": fallback, "state": state.to_dict()}
    return fallback


# ===========================================================================
# BACKWARD-COMPATIBLE WRAPPER
# ===========================================================================

def run_agent(system_prompt, user_prompt, max_rounds=6, debug=True, return_state=False):
    """
    Backward-compatible wrapper so app.py does not break.
    """
    return run_react_agent(
        user_prompt=user_prompt,
        system_prompt=system_prompt or REACT_SYSTEM_PROMPT,
        max_rounds=max_rounds,
        debug=debug,
        return_state=return_state,
    )
