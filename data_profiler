"""
data_profiler.py
----------------
Helpers for profiling uploaded KPI datasets and making the app more flexible.

Used by app.py to:
- summarize schema
- recommend exclusions
- choose dynamic defaults
- score dataset compatibility
- detect wide-format patterns
- suggest anomaly dates
"""

from __future__ import annotations

import re
from typing import Any
import pandas as pd


def _safe_sample_values(series: pd.Series, n: int = 3) -> str:
    vals = series.dropna().astype(str).unique().tolist()[:n]
    return ", ".join(vals)


def infer_semantic_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    nunique = series.nunique(dropna=True)
    ratio = nunique / max(len(series), 1)
    if nunique <= 20 or ratio < 0.2:
        return "categorical"
    return "text/id-like"


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        missing_pct = round(s.isna().mean() * 100, 1)
        nunique = int(s.nunique(dropna=True))
        dtype = str(s.dtype)
        semantic_type = infer_semantic_type(s)

        rows.append({
            "column": col,
            "dtype": dtype,
            "semantic_type": semantic_type,
            "missing_pct": missing_pct,
            "unique_values": nunique,
            "sample_values": _safe_sample_values(s),
        })

    return pd.DataFrame(rows)


def detect_id_like_columns(df: pd.DataFrame) -> list[str]:
    flagged = []
    n = len(df)

    for col in df.columns:
        col_l = col.lower()
        s = df[col]

        if any(tok in col_l for tok in ["id", "uuid", "record", "index", "rownum"]):
            flagged.append(col)
            continue

        nunique = s.nunique(dropna=True)
        if n > 0 and nunique / n > 0.95 and not pd.api.types.is_datetime64_any_dtype(s):
            flagged.append(col)

    return sorted(set(flagged))


def detect_constant_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if df[col].nunique(dropna=True) <= 1]


def detect_high_missing_columns(df: pd.DataFrame, threshold: float = 0.9) -> list[str]:
    return [col for col in df.columns if df[col].isna().mean() >= threshold]


def detect_wide_format_patterns(df: pd.DataFrame) -> dict[str, Any]:
    pattern_hits = []
    regexes = [
        r".+_(us|usa|emea|apac|latam|west|east|north|south)$",
        r".+_(mobile|desktop|tablet)$",
        r".+_(q[1-4]|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)$",
    ]

    for col in df.columns:
        for rgx in regexes:
            if re.match(rgx, col.lower()):
                pattern_hits.append(col)
                break

    return {
        "likely_wide_format": len(pattern_hits) >= 3,
        "matching_columns": pattern_hits[:15],
    }


def suggest_default_metrics(df: pd.DataFrame, max_metrics: int = 4) -> list[str]:
    scored = []

    for col in df.columns:
        s = df[col]
        if col == "date":
            continue
        if not pd.api.types.is_numeric_dtype(s):
            continue

        completeness = 1 - s.isna().mean()
        variance = float(s.var()) if s.notna().sum() > 1 else 0.0
        nunique = s.nunique(dropna=True)

        if nunique <= 1:
            continue

        score = completeness * 1000 + min(variance, 1e9) * 1e-6
        scored.append((col, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:max_metrics]]


def suggest_default_groups(df: pd.DataFrame, max_groups: int = 3) -> list[str]:
    scored = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        if col == "date":
            continue
        if pd.api.types.is_numeric_dtype(s):
            continue

        nunique = s.nunique(dropna=True)
        if nunique <= 1:
            continue
        if nunique > min(30, max(10, n * 0.5)):
            continue

        completeness = 1 - s.isna().mean()
        score = completeness * 100 + max(0, 30 - nunique)
        scored.append((col, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:max_groups]]


def validate_dataset_for_analysis(
    df: pd.DataFrame,
    date_col: str,
    metric_cols: list[str],
    group_cols: list[str],
) -> dict[str, list[str]]:
    errors = []
    warnings = []

    if date_col not in df.columns:
        errors.append("The selected date column was not found in the cleaned dataset.")

    if len(metric_cols) == 0:
        errors.append("Select at least one numeric performance metric.")

    if "date" in df.columns:
        unique_dates = df["date"].nunique(dropna=True)
        if unique_dates < 5:
            warnings.append("The dataset has fewer than 5 distinct dates, so time-series analysis may be unreliable.")

    if len(df) < 20:
        warnings.append("The dataset is very small; results may not be stable.")

    for col in metric_cols:
        if col not in df.columns:
            errors.append(f"Selected metric '{col}' is not available.")
            continue
        non_null = df[col].notna().mean()
        if non_null < 0.5:
            warnings.append(f"Metric '{col}' has more than 50% missing values.")

    id_like = set(detect_id_like_columns(df))
    for col in group_cols:
        if col in id_like:
            warnings.append(f"Grouping column '{col}' looks like an ID field and may not produce meaningful segments.")

    return {"errors": errors, "warnings": warnings}


def infer_date_frequency(date_series: pd.Series) -> str:
    s = pd.Series(pd.to_datetime(date_series, errors="coerce").dropna()).sort_values().drop_duplicates()
    if len(s) < 3:
        return "unknown"

    diffs = s.diff().dropna().dt.days
    if diffs.empty:
        return "unknown"

    median_gap = diffs.median()

    if median_gap <= 1.5:
        return "daily"
    if median_gap <= 8:
        return "weekly-ish"
    if median_gap <= 32:
        return "monthly-ish"
    return "irregular"


def suggest_anomaly_dates(df: pd.DataFrame, metric_cols: list[str], top_n: int = 5) -> list[str]:
    if "date" not in df.columns or len(metric_cols) == 0:
        return []

    daily = df.groupby("date")[metric_cols].mean(numeric_only=True).sort_index()
    if len(daily) < 6:
        return []

    change_scores = pd.Series(0.0, index=daily.index)

    for col in metric_cols:
        series = daily[col].astype(float)
        pct_change = series.pct_change().replace([float("inf"), float("-inf")], pd.NA).fillna(0).abs()
        change_scores = change_scores.add(pct_change, fill_value=0)

    top_dates = change_scores.sort_values(ascending=False).head(top_n).index
    return [pd.Timestamp(d).date().isoformat() for d in top_dates]


def score_dataset_compatibility(
    df: pd.DataFrame,
    date_col: str | None,
    metric_cols: list[str],
    group_cols: list[str],
) -> dict[str, Any]:
    score = 0
    reasons = []

    if date_col and "date" in df.columns and df["date"].notna().mean() > 0.8:
        score += 25
        reasons.append("usable date column")
    else:
        reasons.append("weak or missing date column")

    unique_dates = df["date"].nunique(dropna=True) if "date" in df.columns else 0
    if unique_dates >= 20:
        score += 20
        reasons.append("enough time periods")
    elif unique_dates >= 5:
        score += 10
        reasons.append("limited but usable time periods")

    if len(metric_cols) >= 2:
        score += 20
        reasons.append("multiple numeric KPIs")
    elif len(metric_cols) == 1:
        score += 10
        reasons.append("one numeric KPI")

    if len(group_cols) >= 1:
        score += 15
        reasons.append("has grouping columns")

    high_missing = len(detect_high_missing_columns(df))
    if high_missing == 0:
        score += 10
        reasons.append("low missingness")
    elif high_missing <= 2:
        score += 5
        reasons.append("moderate missingness")

    wide = detect_wide_format_patterns(df)
    if not wide["likely_wide_format"]:
        score += 10
        reasons.append("appears tidy enough for analysis")
    else:
        reasons.append("may need reshaping from wide to long format")

    if score >= 80:
        label = "Strong fit"
    elif score >= 55:
        label = "Moderate fit"
    else:
        label = "Low fit"

    return {"score": score, "label": label, "reasons": reasons}
