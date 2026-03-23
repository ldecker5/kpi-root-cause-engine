"""
tools.py
--------
MILESTONE 5: Tool/Function Calling

This file has TWO parts:
  1. TOOL IMPLEMENTATIONS — the actual Python functions that do real work
  2. TOOL DEFINITIONS — the "menu" we hand to OpenAI so it knows what's available

Think of it this way:
  - The DEFINITIONS are the menu at a restaurant (what you CAN order)
  - The IMPLEMENTATIONS are the kitchen (what actually happens when you order)
"""

import json
import pandas as pd
from src.data_loader import load_data

# ---------------------------------------------------------------------------
# Load the dataset once when this module is imported
# ---------------------------------------------------------------------------
_ACTIVE_DF = None
_DIMENSION_KEYS = []
_METRIC_KEYS = []


def set_active_dataframe(df: pd.DataFrame):
    global _ACTIVE_DF, _DIMENSION_KEYS, _METRIC_KEYS

    _ACTIVE_DF = df.copy()

    _DIMENSION_KEYS = [
        c for c in _ACTIVE_DF.columns
        if c != "date" and not pd.api.types.is_numeric_dtype(_ACTIVE_DF[c])
    ]

    _METRIC_KEYS = [
        c for c in _ACTIVE_DF.columns
        if c != "date" and pd.api.types.is_numeric_dtype(_ACTIVE_DF[c])
    ]


def initialize_default_data():
    global _ACTIVE_DF, _DIMENSION_KEYS, _METRIC_KEYS
    result = load_data()
    _ACTIVE_DF = result["df"]
    _DIMENSION_KEYS = result["dimension_keys"]
    _METRIC_KEYS = result["metric_keys"]

initialize_default_data()
# ===========================================================================
# PART 1: TOOL IMPLEMENTATIONS (the actual functions)
# ===========================================================================
"""
TOOL #1: Query the KPI dataset.

    What it does:
        Filters the dataset by date range and dimensions (like region, device_type),
        then aggregates the requested metrics.

    When the LLM would call this:
        "Show me revenue and orders for the West region in April 2024"

    Parameters:
        source (str):      Dataset identifier (we only have one, but good practice)
        date_start (str):  Start date like "2024-04-01"
        date_end (str):    End date like "2024-04-30"
        dimensions (dict): Filters like {"region": "West", "device_type": "Mobile"}
        metrics (list):    Which numbers to return, like ["revenue", "orders"]
        agg (str):         How to aggregate — "sum" or "mean"
        limit (int):       Max rows to return in preview

    Returns:
        dict with success status, aggregates, and a preview of the data
    """
def query_kpi_data(source, date_start, date_end, metrics, dimensions=None, agg="sum", limit=200):
    global _ACTIVE_DF, _DIMENSION_KEYS, _METRIC_KEYS

    if _ACTIVE_DF is None:
        return {"success": False, "error": "No active dataset loaded."}

    ds = pd.to_datetime(date_start)
    de = pd.to_datetime(date_end)

    if dimensions is None:
        dimensions = {}
    if not isinstance(dimensions, dict):
        dimensions = {}

    dff = _ACTIVE_DF[(_ACTIVE_DF["date"] >= ds) & (_ACTIVE_DF["date"] <= de)].copy()

    for k, v in dimensions.items():
        if k not in _DIMENSION_KEYS:
            return {"success": False, "error": f"Invalid dimension: {k}. Valid dimensions: {_DIMENSION_KEYS}"}
        dff = dff[dff[k].astype(str) == str(v)]

    if dff.empty:
        return {"success": True, "rows_returned": 0, "aggregates": {}, "preview": []}

    for m in metrics:
        if m not in _METRIC_KEYS:
            return {"success": False, "error": f"Invalid metric: {m}. Valid metrics: {_METRIC_KEYS}"}

    if agg == "sum":
        aggregates = {m: float(dff[m].sum()) for m in metrics}
    elif agg == "mean":
        aggregates = {m: float(dff[m].mean()) for m in metrics}
    else:
        return {"success": False, "error": "agg must be 'sum' or 'mean'"}

    preview = (
        dff.sort_values("date")
        .head(limit)
        .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
        .to_dict(orient="records")
    )

    return {
        "success": True,
        "rows_returned": int(len(dff)),
        "aggregates": aggregates,
        "preview": preview,
    }

def compute_kpi_stats(metric, baseline_value, current_value):
    """
    TOOL #2: Compute basic KPI statistics.

    What it does:
        Performs deterministic math calculations that the LLM shouldn't
        guess at (LLMs are bad at math — let Python do it).

    When the LLM would call this:
        "Revenue went from 100 to 85. What's the percent change?"

    Parameters:
        metric (str):          What to calculate — currently supports "percent_change"
        baseline_value (float): The "before" number
        current_value (float):  The "after" number

    Returns:
        dict with the calculated result
    """
    if metric == "percent_change":
        if baseline_value == 0:
            return {"success": False, "error": "baseline_value cannot be 0 (division by zero)"}
        pct = (current_value - baseline_value) / baseline_value
        return {
            "success": True,
            "percent_change": round(pct, 4),
            "percent_change_formatted": f"{pct * 100:.2f}%",
            "direction": "increase" if pct > 0 else "decrease" if pct < 0 else "no change",
        }

    return {"success": False, "error": f"Unsupported metric: {metric}. Supported: ['percent_change']"}


# ===========================================================================
# PART 2: TOOL DEFINITIONS (the "menu" for OpenAI)
#
# This is what OpenAI's API needs to understand what tools are available.
# It's basically a JSON description of each function's inputs.
# ===========================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_kpi_data",
            "description": (
                "Query the KPI dataset by date range and dimension filters. "
                "Returns aggregated metrics and a preview of matching rows. "
                "Use this when you need to look at actual data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Dataset name (use 'ecommerce_kpi')",
                    },
                    "date_start": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format",
                    },
                    "date_end": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format",
                    },
                    "dimensions": {
                        "type": "object",
                        "description": (
                            f"Filter by dimension columns. "
                            f"Available dimensions: {_DIMENSION_KEYS}. "
                            f"Example: {{\"region\": \"West\", \"device_type\": \"Mobile\"}}"
                        ),
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string", "enum": _METRIC_KEYS},
                        "description": f"Which metrics to return. Available: {_METRIC_KEYS}",
                    },
                    "agg": {
                        "type": "string",
                        "enum": ["sum", "mean"],
                        "description": "Aggregation method: 'sum' or 'mean'. Default: 'sum'",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "description": "Max rows in preview. Default: 200",
                    },
                },
                "required": ["source", "date_start", "date_end", "metrics"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_kpi_stats",
            "description": (
                "Compute deterministic KPI statistics like percent change. "
                "Use this instead of doing math yourself — it's more accurate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["percent_change"],
                        "description": "Which calculation to perform",
                    },
                    "baseline_value": {
                        "type": "number",
                        "description": "The 'before' value (e.g., last month's revenue)",
                    },
                    "current_value": {
                        "type": "number",
                        "description": "The 'after' value (e.g., this month's revenue)",
                    },
                },
                "required": ["metric", "baseline_value", "current_value"],
                "additionalProperties": False,
            },
        },
    },
]


# ===========================================================================
# PART 3: TOOL DISPATCH MAP
#
# This connects tool NAMES to actual FUNCTIONS.
# When the LLM says "call query_kpi_data", we look up the function here.
# ===========================================================================

TOOL_IMPLEMENTATIONS = {
    "query_kpi_data": query_kpi_data,
    "compute_kpi_stats": compute_kpi_stats,
}
