"""
test_tool_calling.py
--------------------
MILESTONE 4, Task 4: Test and Evaluate Tool Selection

This file tests whether the LLM:
  1. Picks the RIGHT tool for each question
  2. Sends the RIGHT parameters to the tool
  3. Knows when NOT to use a tool
  4. Handles errors gracefully

Run this file:  python tests/test_tool_calling.py
"""

import os
import sys
import json

# Add project root to path so we can import src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent import call_model, MODEL
from src.tools import TOOL_DEFINITIONS


# ===========================================================================
# TEST CASES
# ===========================================================================

TEST_CASES = [
    # ===========================================
    # EASY CASES (should get these right)
    # ===========================================
    {
        "id": "data_query",
        "description": "Clear data request — should call query_kpi_data",
        "prompt": "Pull revenue and orders for region='West' and device_type='Mobile' from 2024-04-01 to 2024-04-30.",
        "expected_tool": "query_kpi_data",
        "check_params": {
            "date_start": "2024-04-01",
            "date_end": "2024-04-30",
        },
    },
    {
        "id": "math_calculation",
        "description": "Explicit math request — should call compute_kpi_stats",
        "prompt": "Revenue went from 100000 to 85000. Compute the exact percent change.",
        "expected_tool": "compute_kpi_stats",
        "check_params": {
            "metric": "percent_change",
        },
    },
    {
        "id": "no_tool_needed",
        "description": "Summary from provided data — should NOT call any tool",
        "prompt": "Write an executive summary using only these findings: Revenue dropped 12%, mobile conversion fell 23%, West region marketing spend decreased 35%. Do not fetch any additional data.",
        "expected_tool": None,
        "check_params": {},
    },

    # ===========================================
    # TRICKY CASES (these are the interesting ones)
    # ===========================================
    {
        "id": "wrong_tool_bait",
        "description": "Sounds like math but actually needs data first — should call query_kpi_data",
        "prompt": "Calculate the total revenue for April 2024.",
        "expected_tool": "query_kpi_data",
        "check_params": {},
    },
    {
        "id": "ambiguous_no_dates",
        "description": "Vague request with no dates or specifics — should still pick query_kpi_data or ask for clarification",
        "prompt": "What happened with sales last quarter?",
        "expected_tool": "query_kpi_data",
        "check_params": {},
    },
    {
        "id": "invalid_metric",
        "description": "Asks for a metric that doesn't exist — should call tool (and get an error back)",
        "prompt": "Show me the vibe_score and satisfaction_rating for April 2024.",
        "expected_tool": "query_kpi_data",
        "check_params": {},
    },
    {
        "id": "multi_step_comparison",
        "description": "Requires pulling data for TWO periods then comparing — should call query_kpi_data (not compute_kpi_stats first)",
        "prompt": "Did mobile conversion drop more than desktop conversion after April 20th compared to before?",
        "expected_tool": "query_kpi_data",
        "check_params": {},
    },
    {
        "id": "math_disguised_as_data",
        "description": "Numbers are already provided — should use compute_kpi_stats, NOT query data",
        "prompt": "Before April 20, average daily revenue was $42,500. After April 20, it was $36,800. What is the percent change?",
        "expected_tool": "compute_kpi_stats",
        "check_params": {
            "metric": "percent_change",
        },
    },
    {
        "id": "general_knowledge",
        "description": "Business question that doesn't need tools at all",
        "prompt": "What are common reasons e-commerce revenue drops suddenly?",
        "expected_tool": None,
        "check_params": {},
    },
    {
        "id": "mixed_signals",
        "description": "Mentions numbers AND asks to investigate — could go either way but should query data",
        "prompt": "Revenue seems to have dropped about 15% in late April. Can you check what actually happened across regions?",
        "expected_tool": "query_kpi_data",
        "check_params": {},
    },
]


# ===========================================================================
# TEST RUNNER
# ===========================================================================

def run_tool_selection_test(test_case):
    """
    Run a single test: send the prompt to OpenAI with tools available,
    and check what tool (if any) the model chose.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a KPI analysis assistant. Use tools when you need real data or calculations.",
        },
        {"role": "user", "content": test_case["prompt"]},
    ]

    response = call_model(
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
    )

    message = response.choices[0].message
    tool_calls = message.tool_calls or []

    # --- Check: Did the model call the expected tool? ---
    if test_case["expected_tool"] is None:
        # We expected NO tool call
        passed = len(tool_calls) == 0
        actual_tool = None
        actual_args = {}
    else:
        # We expected a specific tool
        if len(tool_calls) > 0:
            actual_tool = tool_calls[0].function.name
            actual_args = json.loads(tool_calls[0].function.arguments)
            passed = actual_tool == test_case["expected_tool"]
        else:
            actual_tool = None
            actual_args = {}
            passed = False

    # --- Check: Did the parameters look right? ---
    param_checks = {}
    for key, expected_value in test_case.get("check_params", {}).items():
        actual_value = actual_args.get(key)
        param_checks[key] = {
            "expected": expected_value,
            "actual": actual_value,
            "match": str(actual_value) == str(expected_value),
        }

    return {
        "test_id": test_case["id"],
        "description": test_case["description"],
        "expected_tool": test_case["expected_tool"],
        "actual_tool": actual_tool,
        "tool_correct": passed,
        "param_checks": param_checks,
        "actual_args": actual_args,
    }


def run_all_tests():
    """Run all test cases and print results."""

    print("=" * 70)
    print(f"TOOL CALLING TESTS — Model: {MODEL}")
    print("=" * 70)

    results = []
    pass_count = 0

    for test in TEST_CASES:
        print(f"\n--- Test: {test['id']} ---")
        print(f"    Prompt: {test['prompt'][:80]}...")
        print(f"    Expected tool: {test['expected_tool']}")

        try:
            result = run_tool_selection_test(test)
            results.append(result)

            if result["tool_correct"]:
                pass_count += 1
                print(f"    ✅ PASS — Model called: {result['actual_tool']}")
            else:
                print(f"    ❌ FAIL — Model called: {result['actual_tool']} (expected {test['expected_tool']})")

            # Show parameter checks
            for param_name, check in result["param_checks"].items():
                status = "✅" if check["match"] else "❌"
                print(f"    {status} Param '{param_name}': expected={check['expected']}, actual={check['actual']}")

        except Exception as e:
            print(f"     ERROR: {e}")
            results.append({"test_id": test["id"], "tool_correct": False, "error": str(e)})

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"RESULTS: {pass_count}/{len(TEST_CASES)} tests passed")
    print(f"Tool Selection Accuracy: {pass_count / len(TEST_CASES) * 100:.0f}%")
    print("=" * 70)

    return results


# ===========================================================================
# RUN
# ===========================================================================

if __name__ == "__main__":
    run_all_tests()