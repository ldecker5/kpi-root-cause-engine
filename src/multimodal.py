"""
multimodal.py
-------------
MILESTONE 7: Multimodal KPI Analysis

Adds vision capabilities to the KPI root cause engine:
  1. KPIChartGenerator  — matplotlib charts from the KPI DataFrame
  2. MultimodalAnalyzer — vision analysis of charts / dashboard screenshots
  3. CrossModalChecker  — consistency test between vision and data findings

Typical flow
------------
  gen   = KPIChartGenerator(df)
  path  = gen.plot_metric_timeseries("revenue", anomaly_start="2024-04-20")
  viz   = MultimodalAnalyzer()
  vis   = viz.extract_anomaly_from_chart(path, metric="revenue")
  data  = viz.build_data_summary(df, "revenue", anomaly_start="2024-04-20")
  check = CrossModalChecker()
  check.report(data, vis)
"""

import base64
import os
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive — safe in scripts / CI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
VISION_MODEL = "gpt-4o-mini"      # vision-capable; used for chart analysis
TEXT_MODEL   = "gpt-4.1-nano"     # consistent with rest of project

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_PALETTE = {
    "Mobile":  "#2196F3",
    "Desktop": "#FF9800",
    "East":    "#4CAF50",
    "West":    "#F44336",
    "North":   "#9C27B0",
    "South":   "#FF5722",
}
_ANOMALY_COLOUR = "#FF000033"   # translucent red


# ===========================================================================
# 1. KPI Chart Generator
# ===========================================================================

class KPIChartGenerator:
    """
    Generate matplotlib charts from the KPI DataFrame and save them as PNGs.

    All methods return the absolute path to the saved file.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str | Path = None):
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])

        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "charts"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def plot_metric_timeseries(
        self,
        metric: str,
        *,
        split_by: str | None = None,
        filter_by: dict | None = None,
        anomaly_start: str | None = "2024-04-20",
        anomaly_end: str | None = None,
        title: str | None = None,
        filename: str | None = None,
    ) -> Path:
        """
        Time-series line chart of `metric` aggregated by date.

        Args:
            metric:        Column to plot (e.g. "revenue", "conversion_rate").
            split_by:      Optional dimension to split into separate lines
                           ("region" or "device_type").
            filter_by:     Optional dict of column→value filters applied before
                           aggregation (e.g. {"device_type": "Mobile"}).
            anomaly_start: ISO date string — draws a red vertical line and
                           shades the anomaly window.
            anomaly_end:   ISO end date for the anomaly shading (default: end
                           of data).
            title:         Chart title (auto-generated if None).
            filename:      PNG output filename (auto-generated if None).

        Returns:
            Path to the saved PNG file.
        """
        df = self._filter(filter_by)
        fig, ax = plt.subplots(figsize=(12, 5))

        if split_by:
            for val, grp in df.groupby(split_by):
                ts = grp.groupby("date")[metric].mean().reset_index()
                colour = _PALETTE.get(str(val), None)
                ax.plot(ts["date"], ts[metric], label=str(val),
                        linewidth=2, color=colour)
            ax.legend(title=split_by, loc="upper left")
        else:
            ts = df.groupby("date")[metric].mean().reset_index()
            ax.plot(ts["date"], ts[metric], linewidth=2, color="#1976D2")

        self._add_anomaly_markers(ax, df, anomaly_start, anomaly_end)
        self._style_axes(ax, metric, title)

        filename = filename or self._auto_filename(metric, split_by, filter_by)
        return self._save(fig, filename)

    def plot_dimension_comparison(
        self,
        metric: str,
        dimension: str,
        *,
        before_end: str = "2024-04-19",
        after_start: str = "2024-04-20",
        filename: str | None = None,
    ) -> Path:
        """
        Side-by-side bar chart: average `metric` per `dimension` value,
        before vs. after the anomaly date.

        Returns:
            Path to the saved PNG file.
        """
        df = self.df.copy()
        df["date"] = pd.to_datetime(df["date"])
        before = df[df["date"] <= before_end].groupby(dimension)[metric].mean()
        after  = df[df["date"] >= after_start].groupby(dimension)[metric].mean()

        categories = sorted(set(before.index) | set(after.index))
        x = np.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        bars_b = ax.bar(x - width / 2,
                        [before.get(c, 0) for c in categories],
                        width, label="Before Apr 20", color="#42A5F5", alpha=0.85)
        bars_a = ax.bar(x + width / 2,
                        [after.get(c, 0) for c in categories],
                        width, label="After Apr 20",  color="#EF5350", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_',' ').title()} by {dimension}: Before vs After Apr 20 Anomaly")
        ax.legend()
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        plt.tight_layout()

        filename = filename or f"bar_{metric}_by_{dimension}.png"
        return self._save(fig, filename)

    def plot_multimodal_overview(
        self,
        anomaly_start: str = "2024-04-20",
        filename: str = "multimodal_overview.png",
    ) -> Path:
        """
        2×2 dashboard: revenue, conversion_rate, marketing_spend, sessions
        each split by device_type — gives the vision model the full picture.
        """
        metrics = ["revenue", "conversion_rate", "marketing_spend", "sessions"]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(
            "KPI Dashboard — Anomaly Window Highlighted (Apr 20, 2024+)",
            fontsize=14, fontweight="bold",
        )

        for ax, metric in zip(axes.flat, metrics):
            for val, grp in self.df.groupby("device_type"):
                ts = grp.groupby("date")[metric].mean().reset_index()
                ax.plot(ts["date"], ts[metric],
                        label=str(val), color=_PALETTE.get(str(val)),
                        linewidth=1.8)
            self._add_anomaly_markers(ax, self.df, anomaly_start, None)
            self._style_axes(ax, metric)
            ax.legend(fontsize=8)

        plt.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter(self, filter_by: dict | None) -> pd.DataFrame:
        df = self.df
        if filter_by:
            for col, val in filter_by.items():
                df = df[df[col] == val]
        return df

    def _add_anomaly_markers(self, ax, df, anomaly_start, anomaly_end):
        if anomaly_start is None:
            return
        start_ts = pd.Timestamp(anomaly_start)
        end_ts   = pd.Timestamp(anomaly_end) if anomaly_end else df["date"].max()
        ax.axvline(start_ts, color="red", linewidth=1.4,
                   linestyle="--", label="Anomaly start")
        ax.axvspan(start_ts, end_ts, color=_ANOMALY_COLOUR)

    def _style_axes(self, ax, metric, title=None):
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title or metric.replace("_", " ").title())
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)

    def _auto_filename(self, metric, split_by, filter_by):
        parts = [metric]
        if split_by:
            parts.append(f"by_{split_by}")
        if filter_by:
            for k, v in filter_by.items():
                parts.append(f"{k}_{v}".replace(" ", "_"))
        return "_".join(parts) + ".png"

    def _save(self, fig, filename: str) -> Path:
        path = self.output_dir / filename
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Chart saved → {path}")
        return path


# ===========================================================================
# 2. Multimodal Analyzer
# ===========================================================================

def _encode_image(path: str | Path) -> str:
    """Return base64-encoded PNG/JPEG for the OpenAI vision API."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class MultimodalAnalyzer:
    """
    Vision-based analysis of KPI charts and dashboard screenshots.
    Uses gpt-4o-mini with image_url content blocks.
    """

    def __init__(self, vision_model: str = VISION_MODEL):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vision_model = vision_model

    # ------------------------------------------------------------------
    # Core vision call
    # ------------------------------------------------------------------

    def _vision_call(self, image_path: str | Path, prompt: str) -> str:
        b64 = _encode_image(image_path)
        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def describe_chart(self, image_path: str | Path) -> str:
        """
        General description of what's happening in the chart.
        Useful for dashboard screenshots or exploratory analysis.
        """
        prompt = (
            "You are a data analyst reviewing a business KPI chart. "
            "Describe what you see in detail:\n"
            "- What metric(s) are plotted?\n"
            "- What is the overall trend?\n"
            "- Are there any visible spikes, drops, or inflection points?\n"
            "- Approximately when do changes occur?\n"
            "- What dimensions or groups are shown (if any)?\n"
            "Be specific about dates and magnitudes if they are visible on the axes."
        )
        return self._vision_call(image_path, prompt)

    def extract_anomaly_from_chart(
        self, image_path: str | Path, metric: str = "the KPI"
    ) -> dict:
        """
        Ask the vision model to extract a structured anomaly description
        from the chart.  Returns a dict with keys:
            anomaly_detected, anomaly_date, magnitude, affected_segments,
            trend_before, trend_after, raw_text
        """
        prompt = f"""You are analyzing a time-series chart of {metric} for an e-commerce business.

Examine the chart carefully and answer these questions:
1. Is there a visible anomaly (sudden drop or spike)? (yes/no)
2. Approximately when does it start? (give a date or month if visible)
3. How large is the change? (rough percentage or absolute, based on the axis)
4. Which segments or groups are most affected (if the chart shows multiple lines)?
5. What was the trend BEFORE the anomaly?
6. What is the trend AFTER the anomaly?

Format your response EXACTLY as:
ANOMALY_DETECTED: <yes/no>
ANOMALY_DATE: <date or "unknown">
MAGNITUDE: <e.g. "~20% drop" or "unknown">
AFFECTED_SEGMENTS: <e.g. "Mobile more than Desktop" or "all segments equally">
TREND_BEFORE: <description>
TREND_AFTER: <description>
NOTES: <any other observations>"""

        raw = self._vision_call(image_path, prompt)

        # Parse structured fields
        parsed = {"raw_text": raw}
        field_map = {
            "ANOMALY_DETECTED":  "anomaly_detected",
            "ANOMALY_DATE":      "anomaly_date",
            "MAGNITUDE":         "magnitude",
            "AFFECTED_SEGMENTS": "affected_segments",
            "TREND_BEFORE":      "trend_before",
            "TREND_AFTER":       "trend_after",
            "NOTES":             "notes",
        }
        for label, key in field_map.items():
            for line in raw.splitlines():
                if line.strip().startswith(label + ":"):
                    parsed[key] = line.split(":", 1)[1].strip()
                    break
            else:
                parsed[key] = "unknown"

        return parsed

    def analyze_dashboard_screenshot(self, image_path: str | Path) -> str:
        """
        Analyze a user-uploaded dashboard screenshot.
        Extracts KPI readings, anomalies, and suggested investigations.
        """
        prompt = (
            "You are a senior data analyst reviewing an e-commerce BI dashboard screenshot.\n\n"
            "Extract and report:\n"
            "1. All visible KPI metrics and their current values / trends\n"
            "2. Any metrics that appear anomalous (unusual levels or changes)\n"
            "3. Time period shown\n"
            "4. Dimensions visible (regions, device types, product categories, etc.)\n"
            "5. Suggested root-cause hypotheses based solely on what you can see\n\n"
            "Format as a structured analysis report."
        )
        return self._vision_call(image_path, prompt)

    def analyze_multimodal(
        self, text_summary: str, image_path: str | Path
    ) -> str:
        """
        Combined text + vision analysis.
        Useful when you already have a data-derived text summary but also
        want the model to cross-check it against the chart.
        """
        prompt = (
            "You are a KPI analyst with access to both a written data summary "
            "and a visual chart.\n\n"
            f"DATA SUMMARY:\n{text_summary}\n\n"
            "INSTRUCTIONS:\n"
            "1. Describe what you see in the chart.\n"
            "2. Identify points where the chart CONFIRMS the data summary.\n"
            "3. Identify any DISCREPANCIES between the chart and the summary.\n"
            "4. State your overall confidence that the summary accurately "
            "describes the chart (Low / Medium / High), with a brief justification."
        )
        return self._vision_call(image_path, prompt)

    def build_data_summary(
        self,
        df: pd.DataFrame,
        metric: str,
        anomaly_start: str = "2024-04-20",
        split_by: str | None = None,
    ) -> str:
        """
        Compute a concise text summary of `metric` before vs. after
        `anomaly_start` directly from the DataFrame.
        This is the 'ground truth' text side for cross-modal comparison.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        cutoff = pd.Timestamp(anomaly_start)

        before = df[df["date"] < cutoff]
        after  = df[df["date"] >= cutoff]

        def _pct_change(b, a):
            if b == 0:
                return "N/A"
            return f"{(a - b) / b * 100:+.1f}%"

        lines = [f"Data-derived summary for [{metric}] (anomaly start: {anomaly_start})"]
        lines.append(f"  Overall: {before[metric].mean():.4g} → {after[metric].mean():.4g} "
                     f"({_pct_change(before[metric].mean(), after[metric].mean())})")

        if split_by and split_by in df.columns:
            lines.append(f"  By {split_by}:")
            for val in sorted(df[split_by].unique()):
                b_val = before[before[split_by] == val][metric].mean()
                a_val = after[after[split_by] == val][metric].mean()
                lines.append(f"    {val}: {b_val:.4g} → {a_val:.4g} ({_pct_change(b_val, a_val)})")

        return "\n".join(lines)


# ===========================================================================
# 3. Cross-Modal Consistency Checker
# ===========================================================================

class CrossModalChecker:
    """
    Uses an LLM to compare data-derived text findings against vision findings
    and produce a structured consistency report.
    """

    def __init__(self, model: str = TEXT_MODEL):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def check_consistency(
        self, text_findings: str, vision_findings: str | dict
    ) -> dict:
        """
        Ask the LLM to evaluate how well `vision_findings` aligns with
        `text_findings` (the ground-truth data summary).

        Returns:
            {
                "score":          float 0–1,
                "label":          "High" | "Medium" | "Low",
                "matching":       list[str],
                "discrepancies":  list[str],
                "explanation":    str,
                "raw":            str,
            }
        """
        if isinstance(vision_findings, dict):
            vision_text = vision_findings.get("raw_text", str(vision_findings))
        else:
            vision_text = vision_findings

        prompt = f"""You are evaluating consistency between two sources of information
about the same e-commerce KPI dataset.

SOURCE A — Data analysis (ground truth from raw numbers):
{text_findings}

SOURCE B — Vision analysis (from looking at a chart of the same data):
{vision_text}

Evaluate:
1. What facts from Source A are CONFIRMED by Source B?
2. What facts from Source A are CONTRADICTED or MISSED by Source B?
3. Are there things in Source B that are NOT in Source A?
4. Overall consistency score on a scale of 0–10.

Format your response EXACTLY as:
SCORE: <integer 0-10>
MATCHING_POINTS:
- <point 1>
- <point 2>
DISCREPANCIES:
- <discrepancy 1>
EXTRA_IN_VISION:
- <anything vision found that data didn't mention>
EXPLANATION: <2-3 sentence summary>"""

        raw = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
        ).choices[0].message.content

        return self._parse_consistency(raw)

    def report(self, text_findings: str, vision_findings: str | dict) -> None:
        """Run check_consistency and pretty-print the results."""
        result = self.check_consistency(text_findings, vision_findings)
        score = result["score"]
        label = result["label"]

        icon = "✅" if score >= 0.7 else ("⚠️" if score >= 0.4 else "❌")
        print(f"\n{icon}  Cross-Modal Consistency: {label} ({score:.0%})")
        print(f"   {result['explanation']}")

        if result["matching"]:
            print("\n   Confirmed by vision:")
            for pt in result["matching"]:
                print(f"     ✓ {pt}")
        if result["discrepancies"]:
            print("\n   Discrepancies / missed:")
            for pt in result["discrepancies"]:
                print(f"     ✗ {pt}")

    def _parse_consistency(self, raw: str) -> dict:
        score_raw = 5
        matching, discrepancies, extra, explanation = [], [], [], ""
        current = None

        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("SCORE:"):
                try:
                    score_raw = int(stripped.split(":", 1)[1].strip())
                except ValueError:
                    score_raw = 5
            elif stripped.startswith("MATCHING_POINTS:"):
                current = "matching"
            elif stripped.startswith("DISCREPANCIES:"):
                current = "discrepancies"
            elif stripped.startswith("EXTRA_IN_VISION:"):
                current = "extra"
            elif stripped.startswith("EXPLANATION:"):
                explanation = stripped.split(":", 1)[1].strip()
                current = None
            elif stripped.startswith("- ") and current:
                item = stripped[2:].strip()
                if current == "matching":
                    matching.append(item)
                elif current == "discrepancies":
                    discrepancies.append(item)
                elif current == "extra":
                    extra.append(item)

        score = score_raw / 10.0
        if score >= 0.7:
            label = "High"
        elif score >= 0.4:
            label = "Medium"
        else:
            label = "Low"

        return {
            "score":         score,
            "label":         label,
            "matching":      matching,
            "discrepancies": discrepancies,
            "extra":         extra,
            "explanation":   explanation,
            "raw":           raw,
        }
