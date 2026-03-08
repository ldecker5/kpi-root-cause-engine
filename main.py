"""
main.py
-------
Run this to test the full agent end-to-end.

Usage:
    python main.py

Make sure you have:
    1. A .env file with OPENAI_API_KEY=sk-your-key
    2. A CSV file in data/sample_ecommerce_kpi_data.csv
    3. Installed requirements: pip install -r requirements.txt
"""

from src.agent import run_agent
from src.prompts import AGENT_SYSTEM_PROMPT


def main():
    # ------------------------------------------------------------------
    # Example 1: Simple data query
    # The agent should call query_kpi_data, then summarize the results
    # ------------------------------------------------------------------
    print("=" * 70)
    print("EXAMPLE 1: Simple Data Query")
    print("=" * 70)

    result = run_agent(
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_prompt=(
            "Pull revenue and orders for region='West' and device_type='Mobile' "
            "from 2024-04-01 to 2024-04-30."
        ),
        debug=True,
    )
    print("\n📋 FINAL ANSWER:")
    print(result)

    # ------------------------------------------------------------------
    # Example 2: Investigation (requires multiple tool calls)
    # The agent should query data, compute stats, then explain
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Full Investigation")
    print("=" * 70)

    result = run_agent(
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_prompt=(
            "Revenue appears to have dropped around April 20, 2024. "
            "Investigate what caused this. Check different regions and "
            "device types. Provide an executive summary of your findings."
        ),
        debug=True,
    )
    print("\n📋 FINAL ANSWER:")
    print(result)

    # ------------------------------------------------------------------
    # Example 3: Math only (should use compute_kpi_stats)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Math Calculation")
    print("=" * 70)

    result = run_agent(
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_prompt="Revenue went from 125000 to 108000. What is the exact percent change?",
        debug=True,
    )
    print("\n📋 FINAL ANSWER:")
    print(result)


if __name__ == "__main__":
    main()
