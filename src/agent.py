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
from src.performance import (
    RESPONSE_CACHE, COST_TRACKER, ModelCascade, TASK_TIERS, default_verifier,
)

# ---------------------------------------------------------------------------
# Load API key from .env file
# ---------------------------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Which model to use — you can change this
MODEL = "gpt-4o-mini"

# Cache can be disabled for tests or non-idempotent workflows.
USE_RESPONSE_CACHE = os.getenv("KPI_DISABLE_CACHE", "0") != "1"

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
def _raw_call_model(*, model, messages, tools=None, tool_choice="auto",
                    max_tokens=900, temperature=1.0):
    """Transport layer — hits the OpenAI API with retry. No caching."""
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs)


def call_model(*, messages, tools=None, tool_choice="auto", max_tokens=900,
               model: str | None = None, tag: str = "agent",
               use_cache: bool | None = None):
    """
    Cache-aware wrapper around `_raw_call_model`.

    Returns a response object with the same shape the ReAct loop expects
    (`.choices[0].message.content` / `.tool_calls`). Also records token usage
    and dollar cost via the shared CostTracker.

    MILESTONE 14 additions vs. original call_model:
      - ResponseCache lookup by (model, messages, tools, max_tokens).
        Repeat queries within a session (or across sessions via disk cache)
        return in ~1ms and cost $0.
      - CostTracker.record_chat for every call — hits or misses are logged so
        cache hit-rate is measurable.
      - Optional per-call `model` override, used by ModelCascade to escalate.
    """
    model = model or MODEL
    use_cache = USE_RESPONSE_CACHE if use_cache is None else use_cache

    if use_cache:
        hit = RESPONSE_CACHE.get(model, messages, tools, 1.0, max_tokens)
        if hit is not None:
            COST_TRACKER.record_chat(model, hit.get("usage"), cached=True, tag=tag)
            return _CachedResponse(hit)

    response = _raw_call_model(
        model=model, messages=messages, tools=tools,
        tool_choice=tool_choice, max_tokens=max_tokens,
    )
    COST_TRACKER.record_chat(model, getattr(response, "usage", None),
                             cached=False, tag=tag)

    if use_cache:
        RESPONSE_CACHE.put(model, messages, tools, 1.0, max_tokens,
                           _serialize_response(response))
    return response


# ---- Cache serialization helpers ----

def _serialize_response(resp) -> Dict[str, Any]:
    """Strip an OpenAI response down to what the agent loop consumes."""
    choice = resp.choices[0]
    msg = choice.message
    tool_calls = []
    for tc in (msg.tool_calls or []):
        tool_calls.append({
            "id": tc.id,
            "type": tc.type,
            "function": {"name": tc.function.name,
                         "arguments": tc.function.arguments},
        })
    usage = getattr(resp, "usage", None)
    return {
        "content": msg.content,
        "tool_calls": tool_calls,
        "usage": {
            "prompt_tokens": _g(usage, "prompt_tokens"),
            "completion_tokens": _g(usage, "completion_tokens"),
        } if usage else None,
    }


def _g(o, k):
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


class _CachedMessage:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = [_CachedToolCall(tc) for tc in tool_calls] if tool_calls else None


class _CachedToolCall:
    def __init__(self, d):
        self.id = d["id"]
        self.type = d["type"]
        self.function = type("F", (), {
            "name": d["function"]["name"],
            "arguments": d["function"]["arguments"],
        })


class _CachedChoice:
    def __init__(self, entry):
        self.message = _CachedMessage(entry.get("content"),
                                      entry.get("tool_calls"))


class _CachedResponse:
    """Quacks like an OpenAI response so the agent loop needs no changes."""
    def __init__(self, entry: Dict[str, Any]):
        self.choices = [_CachedChoice(entry)]
        self.usage = entry.get("usage")

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
    """
    Cheap planning step — routed through a model cascade so the tiny nano
    model handles it unless it fails the verifier.
    """
    prompt = THOUGHT_PROMPT_TEMPLATE.format(
        user_prompt=state.user_prompt,
        state_summary=summarize_state(state),
    )
    messages = [
        {"role": "system", "content": "You are a planning module for a ReAct agent."},
        {"role": "user", "content": prompt},
    ]

    cascade = ModelCascade(tiers=TASK_TIERS["plan"], verifier=default_verifier)

    def _run(model_name: str):
        return call_model(
            messages=messages, tools=None, max_tokens=200,
            model=model_name, tag=f"plan:{model_name}",
        )

    response, used = cascade.run(_run)
    thought = response.choices[0].message.content or ""
    state.record_thought(thought)

    if debug:
        print(f"\n[Thought {state.current_step} via {used}] {thought}")

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
