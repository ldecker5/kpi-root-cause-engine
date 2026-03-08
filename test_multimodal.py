"""
test_multimodal.py
------------------
MILESTONE 7: Cross-Modal Consistency Tests

Tests three cross-modal scenarios using the KPI sample dataset:

  Test 1 — Revenue timeseries (split by device_type)
            Vision should spot the Mobile-specific drop after Apr 20.

  Test 2 — Conversion rate (Mobile only)
            Vision should identify the ~22% drop in conversion rate.

  Test 3 — Marketing spend (split by region)
            Vision should identify the West region marketing cut.

  Full dashboard — 2×2 overview chart
            Vision describes the full picture; RAG + vision gives a
            richer root-cause explanation than text alone.

Run:
    python test_multimodal.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_data
from src.multimodal import KPIChartGenerator, MultimodalAnalyzer, CrossModalChecker
from src.rag_pipeline import RAGPipeline

ANOMALY_START = "2024-04-20"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("  MILESTONE 7 — MULTIMODAL CROSS-MODAL CONSISTENCY TESTS")
print("=" * 70)

df, dims, metrics = load_data()
print(f"\nDataset: {len(df)} rows | Dimensions: {dims} | Metrics: {metrics}\n")

gen   = KPIChartGenerator(df)
viz   = MultimodalAnalyzer()
check = CrossModalChecker()


# ===========================================================================
# Helper: run one full test scenario
# ===========================================================================

def run_test(
    test_num: int,
    name: str,
    metric: str,
    split_by: str | None,
    filter_by: dict | None,
):
    print(f"\n{'─' * 70}")
    print(f"  TEST {test_num}: {name}")
    print(f"{'─' * 70}")

    # 1. Generate chart
    print("\n[1] Generating chart ...")
    chart_path = gen.plot_metric_timeseries(
        metric,
        split_by=split_by,
        filter_by=filter_by,
        anomaly_start=ANOMALY_START,
    )

    # 2. Data-derived ground truth
    print("[2] Computing data summary ...")
    data_summary = viz.build_data_summary(df, metric, ANOMALY_START, split_by)
    print(f"\n    {data_summary.replace(chr(10), chr(10) + '    ')}")

    # 3. Vision analysis
    print("\n[3] Running vision analysis ...")
    vision_result = viz.extract_anomaly_from_chart(chart_path, metric=metric)

    print(f"\n    Vision findings:")
    for field in ["anomaly_detected", "anomaly_date", "magnitude",
                  "affected_segments", "trend_before", "trend_after"]:
        print(f"      {field:20s}: {vision_result.get(field, 'n/a')}")

    # 4. Cross-modal consistency check
    print("\n[4] Cross-modal consistency check ...")
    check.report(data_summary, vision_result)

    return chart_path, data_summary, vision_result


# ===========================================================================
# Test 1 — Revenue by device_type
# ===========================================================================

path_t1, data_t1, vis_t1 = run_test(
    1,
    "Revenue over time — split by device_type",
    metric="revenue",
    split_by="device_type",
    filter_by=None,
)

# ===========================================================================
# Test 2 — Conversion rate, Mobile only
# ===========================================================================

path_t2, data_t2, vis_t2 = run_test(
    2,
    "Conversion rate — Mobile only",
    metric="conversion_rate",
    split_by=None,
    filter_by={"device_type": "Mobile"},
)

# ===========================================================================
# Test 3 — Marketing spend by region
# ===========================================================================

path_t3, data_t3, vis_t3 = run_test(
    3,
    "Marketing spend — split by region",
    metric="marketing_spend",
    split_by="region",
    filter_by=None,
)

# ===========================================================================
# Full 2×2 dashboard — vision describes the complete picture
# ===========================================================================

print(f"\n{'─' * 70}")
print(f"  FULL DASHBOARD — multimodal overview chart + RAG")
print(f"{'─' * 70}")

print("\n[1] Generating 2×2 dashboard chart ...")
dash_path = gen.plot_multimodal_overview(ANOMALY_START)

print("\n[2] Vision: general chart description ...")
description = viz.describe_chart(dash_path)
print(f"\n    {description[:600]}{'...' if len(description) > 600 else ''}")

print("\n[3] Vision: multimodal analysis (data summary + chart) ...")
full_data_summary = (
    "Anomaly start: 2024-04-20\n"
    + viz.build_data_summary(df, "revenue",          ANOMALY_START, "device_type") + "\n"
    + viz.build_data_summary(df, "conversion_rate",  ANOMALY_START, "device_type") + "\n"
    + viz.build_data_summary(df, "marketing_spend",  ANOMALY_START, "region")
)
mm_analysis = viz.analyze_multimodal(full_data_summary, dash_path)
print(f"\n    {mm_analysis[:700]}{'...' if len(mm_analysis) > 700 else ''}")

# ===========================================================================
# RAG + Vision: enrich RAG prompt with vision context
# ===========================================================================

print(f"\n{'─' * 70}")
print(f"  RAG + VISION INTEGRATION — enriched root-cause explanation")
print(f"{'─' * 70}")

# Build an anomaly summary that combines data analysis + vision observations
vision_enriched_summary = (
    "Dataset columns: date, revenue, orders, conversion_rate, marketing_spend, "
    "sessions, region, device_type.\n\n"
    "Anomaly: Revenue dropped starting 2024-04-20.\n\n"
    "Data findings:\n"
    f"{viz.build_data_summary(df, 'revenue', ANOMALY_START, 'device_type')}\n"
    f"{viz.build_data_summary(df, 'conversion_rate', ANOMALY_START, 'device_type')}\n"
    f"{viz.build_data_summary(df, 'marketing_spend', ANOMALY_START, 'region')}\n\n"
    "Visual chart analysis:\n"
    f"{vis_t1.get('raw_text', '')[:400]}"
)

print("\n[1] Loading RAG pipeline (using existing vector DB if available) ...")
rag = RAGPipeline()
try:
    rag.build_or_load()
    print("\n[2] Generating RAG + vision root-cause explanation ...")
    rag_result = rag.generate_rag_response(vision_enriched_summary, k=5)
    print("\n=== RAG + VISION RESPONSE ===\n")
    print(rag_result["response"])
    print("\nSources retrieved:")
    for i, d in enumerate(rag_result["retrieved_docs"], 1):
        print(f"  [{i}] {d.metadata.get('source')} (page {d.metadata.get('page')})")
except FileNotFoundError as e:
    print(f"\n  [SKIP] RAG pipeline skipped — {e}")
    print("  Add PDFs to knowledge_base/ and re-run to enable RAG + vision.")

# ===========================================================================
# Summary table
# ===========================================================================

print(f"\n\n{'=' * 70}")
print(f"  CROSS-MODAL CONSISTENCY SUMMARY")
print(f"{'=' * 70}")
print(f"  {'Test':<45} {'Vision anomaly?':<18} {'Date found?'}")
print(f"  {'-' * 68}")

for test_name, vis in [
    ("Revenue by device_type",           vis_t1),
    ("Conversion rate (Mobile only)",     vis_t2),
    ("Marketing spend by region",         vis_t3),
]:
    detected = vis.get("anomaly_detected", "?").lower()
    date_str = vis.get("anomaly_date", "?")
    ok_icon  = "✅" if "yes" in detected else "❌"
    date_ok  = "✅" if any(tok in date_str.lower() for tok in ["apr", "2024", "april", "20"]) else "❌"
    print(f"  {ok_icon} {test_name:<44} {detected:<18} {date_ok} {date_str}")

print(f"\n  Charts saved to: {gen.output_dir}/")
print(f"\n  All tests complete.\n")
