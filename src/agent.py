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
"""

import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))  # Ensure current directory is in path for imports
import json
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception_type

from src.tools import TOOL_DEFINITIONS, TOOL_IMPLEMENTATIONS

# ---------------------------------------------------------------------------
# Load API key from .env file
# ---------------------------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Which model to use — you can change this
MODEL = "gpt-4o-mini"


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
# THE AGENT LOOP (Milestone 4, Task 2: Implement Tool Calling)
# ===========================================================================

def run_agent(system_prompt, user_prompt, max_rounds=4, debug=True):
    """
    Run the full agent loop.

    How it works (step by step):

        Round 0:
          → Send: [system prompt, user question] + tool menu
          ← Receive: model wants to call query_kpi_data(region="West", ...)
          → We run query_kpi_data locally, get real data
          → Send data back to model

        Round 1:
          ← Receive: model wants to call compute_kpi_stats(baseline=100, current=85)
          → We run compute_kpi_stats locally, get -15%
          → Send result back to model

        Round 2:
          ← Receive: "Revenue in the West region dropped 15%, primarily driven by..."
          → That's text! We're done. Return it.

    Parameters:
        system_prompt (str):  Instructions for the LLM (its "role")
        user_prompt (str):    The user's question
        max_rounds (int):     Safety limit — stop after this many tool calls
        debug (bool):         If True, print what's happening each round

    Returns:
        str: The final text answer from the model
    """

    # Build the initial conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for round_num in range(max_rounds):
        # ----- Call the model -----
        response = call_model(
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )

        response_message = response.choices[0].message

        if debug:
            print(f"\n[Round {round_num}] Model responded with: ", end="")
            if response_message.tool_calls:
                tool_names = [tc.function.name for tc in response_message.tool_calls]
                print(f"TOOL CALLS → {tool_names}")
            else:
                print("TEXT (final answer)")

        # ----- Check: did the model call any tools? -----
        if not response_message.tool_calls:
            # No tool calls = the model is giving us a final text answer
            return response_message.content or ""

        # ----- The model wants to use tools — execute them -----
        # First, add the model's response to the conversation history
        messages.append(response_message)

        for tool_call in response_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            if debug:
                print(f"  → Executing: {tool_name}({json.dumps(tool_args, indent=2)[:200]}...)")

            # --- Error handling (Milestone 4, Task 3) ---
            if tool_name not in TOOL_IMPLEMENTATIONS:
                # The model hallucinated a tool that doesn't exist
                result = {"success": False, "error": f"Unknown tool: '{tool_name}'"}
            else:
                try:
                    # Run the actual Python function
                    result = TOOL_IMPLEMENTATIONS[tool_name](**tool_args)
                except Exception as e:
                    # The function crashed — catch the error, don't crash the program
                    result = {
                        "success": False,
                        "error": str(e),
                        "tool": tool_name,
                        "args": tool_args,
                    }

            if debug:
                success = result.get("success", "?")
                error = result.get("error", "")
                if error:
                    print(f"  ← Result: success={success}, error={error}")
                else:
                    print(f"  ← Result: success={success}")

            # Send the tool result back to the model
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    # If we hit max rounds without a final answer
    return "[Agent stopped] Reached maximum tool call rounds without a final answer."