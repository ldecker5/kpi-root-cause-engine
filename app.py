"""

"""

import os
import sys
import json
import time
import uuid
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd

from src.data_loader import clean_column_name, normalize_columns, infer_date_column
from src.tools import set_active_dataframe
from src.security import (
    validate_input,
    sanitize_input,
    filter_output,
    check_api_key_safety,
    rate_limiter,
    get_session_id,
)

from src.data_profiler import (
    profile_dataframe,
    detect_id_like_columns,
    detect_constant_columns,
    detect_high_missing_columns,
    detect_wide_format_patterns,
    suggest_default_metrics,
    suggest_default_groups,
    validate_dataset_for_analysis,
    infer_date_frequency,
    suggest_anomaly_dates,
    score_dataset_compatibility,
)

# ── Wizard navigation ─────────────────────────────────────────

if "setup_step" not in st.session_state:
    st.session_state["setup_step"] = 1

TOTAL_STEPS = 5

def go_next_step():
    if st.session_state["setup_step"] < TOTAL_STEPS:
        st.session_state["setup_step"] += 1

def go_prev_step():
    if st.session_state["setup_step"] > 1:
        st.session_state["setup_step"] -= 1

# App
def app_log(event_type, **payload):
    record = {
        "ts": datetime.utcnow().isoformat(),
        "event": event_type,
        **payload,
    }
    print(json.dumps(record))

def load_data_from_df(raw_df, selected_date_col):
    df, rename_map = normalize_columns(raw_df.copy())
    selected_date_col = clean_column_name(selected_date_col)

    if selected_date_col not in df.columns:
        raise ValueError(
            f"Selected date column '{selected_date_col}' not found after normalization. "
            f"Available columns: {list(df.columns)}"
        )

    df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors="coerce")
    df = df.dropna(subset=[selected_date_col]).copy()

    if df.empty:
        raise ValueError("No valid rows remain after parsing the selected date column.")

    if selected_date_col != "date":
        df = df.rename(columns={selected_date_col: "date"})

    dimension_keys = [
        c for c in df.columns
        if c != "date" and not pd.api.types.is_numeric_dtype(df[c])
    ]

    metric_keys = [
        c for c in df.columns
        if c != "date" and pd.api.types.is_numeric_dtype(df[c])
    ]

    return {
        "df": df,
        "dimension_keys": dimension_keys,
        "metric_keys": metric_keys,
        "column_mapping": rename_map,
    }


def build_dynamic_anomaly_summary(df, anomaly_ts, selected_metrics, selected_groups):
    before = df[df["date"] < anomaly_ts]
    after = df[df["date"] >= anomaly_ts]

    lines = [f"Anomaly analysis starting {anomaly_ts.date()}."]

    for m in selected_metrics:
        if m not in df.columns:
            continue

        b = before[m].mean() if len(before) else 0
        a = after[m].mean() if len(after) else 0
        pct = ((a - b) / b * 100) if b else 0
        lines.append(f"  {m}: {b:.3g} → {a:.3g} ({pct:+.1f}%)")

        for grp in selected_groups[:2]:
            if grp in df.columns:
                top_vals = df[grp].dropna().astype(str).unique()[:4]
                for val in top_vals:
                    bd = before[before[grp].astype(str) == str(val)][m].mean() if len(before) else 0
                    ad = after[after[grp].astype(str) == str(val)][m].mean() if len(after) else 0
                    p2 = ((ad - bd) / bd * 100) if bd else 0
                    lines.append(f"    [{grp}={val}] {bd:.3g} → {ad:.3g} ({p2:+.1f}%)")

    return "\n".join(lines)

# ── Make src/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="KPI Root Cause Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Font */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  code, pre, .stCode          { font-family: 'IBM Plex Mono', monospace; }

  /* Dark top bar */
  .top-bar {
    background: #0f172a;
    color: #f8fafc;
    padding: 1.2rem 2rem;
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .top-bar h1 { margin: 0; font-size: 1.5rem; font-weight: 600; letter-spacing: -0.02em; }
  .top-bar p  { margin: 0; font-size: 0.85rem; color: #94a3b8; }

  /* Metric cards */
  .metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    padding: 1rem 1.25rem;
    text-align: center;
  }
  .metric-card .value { font-size: 1.8rem; font-weight: 600; color: #0f172a; }
  .metric-card .label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-card .delta { font-size: 0.85rem; margin-top: 0.2rem; }
  .metric-card .delta.negative { color: #ef4444; }
  .metric-card .delta.positive { color: #22c55e; }

  /* Step badges */
  .step-badge {
    background: #0f172a;
    color: #f8fafc;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-right: 0.4rem;
  }

  /* RAG response box */
  .rag-box {
    background: #f0f9ff;
    border-left: 3px solid #0ea5e9;
    padding: 1rem 1.25rem;
    border-radius: 0 0.4rem 0.4rem 0;
    white-space: pre-wrap;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #1e293b;
  }
  .baseline-box {
    background: #fafafa;
    border-left: 3px solid #94a3b8;
    padding: 1rem 1.25rem;
    border-radius: 0 0.4rem 0.4rem 0;
    white-space: pre-wrap;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #1e293b;
  }

  /* Security alert boxes */
  .security-blocked {
    background: #fef2f2;
    border-left: 3px solid #ef4444;
    padding: 0.8rem 1.25rem;
    border-radius: 0 0.4rem 0.4rem 0;
    font-size: 0.9rem;
    color: #1e293b;
  }
  .security-passed {
    background: #f0fdf4;
    border-left: 3px solid #22c55e;
    padding: 0.8rem 1.25rem;
    border-radius: 0 0.4rem 0.4rem 0;
    font-size: 0.9rem;
    color: #1e293b;
  }

  /* Consistency badge */
  .badge-high   { background:#dcfce7; color:#15803d; padding:0.2rem 0.7rem; border-radius:999px; font-weight:600; font-size:0.85rem; }
  .badge-medium { background:#fef9c3; color:#854d0e; padding:0.2rem 0.7rem; border-radius:999px; font-weight:600; font-size:0.85rem; }
  .badge-low    { background:#fee2e2; color:#b91c1c; padding:0.2rem 0.7rem; border-radius:999px; font-weight:600; font-size:0.85rem; }

  /* Hide Streamlit chrome */
  #MainMenu, footer { visibility: hidden; }
  .stDeployButton   { display: none; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] { gap: 1rem; border-bottom: 2px solid #e2e8f0; }
  .stTabs [data-baseweb="tab"]      { font-weight: 600; font-size: 0.9rem; padding-bottom: 0.5rem; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #0f172a; }
  section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
  section[data-testid="stSidebar"] .stButton > button {
    background: #1e40af; border: none; color: white !important;
    width: 100%; border-radius: 0.4rem; font-weight: 600; padding: 0.6rem;
  }
  section[data-testid="stSidebar"] .stButton > button:hover { background: #1d4ed8; }
</style>
""", unsafe_allow_html=True)


# ── Top bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <span style="font-size:2rem">📊</span>
  <div>
    <h1>KPI Root Cause Analysis Engine</h1>
    <p>Anomaly detection · RAG-powered explanations · Vision analysis · Tool calling · 🔒 Security hardened</p>
  </div>
</div>
""", unsafe_allow_html=True)

setup_steps = [
    "Upload Data",
    "Configure Columns",
    "Data Readiness",
    "Analysis Settings",
    "Run Analysis",
]

current_step = st.session_state["setup_step"]

st.info(
    """
**How to use Oops AI**

1. Upload a dataset with a date column and KPI metrics  
2. Configure metrics and segment columns  
3. Review data readiness and suggested anomaly dates  
4. Run the full analysis in Step 5
"""
)

st.markdown("### Setup Progress")

progress_cols = st.columns(len(setup_steps))
for i, step_name in enumerate(setup_steps, start=1):
    label = f"**➡ {step_name}**" if i == current_step else step_name
    progress_cols[i - 1].markdown(label)

jump_cols = st.columns(len(setup_steps))
for i, step_name in enumerate(setup_steps, start=1):
    disabled = False

    # prevent jumping forward before required setup exists
    if i >= 2 and st.session_state.get("raw_df") is None:
        disabled = True
    if i >= 3 and st.session_state.get("df") is None:
        disabled = True

    jump_cols[i - 1].button(
        ("➡ Current" if i == current_step else f"Go to {i}"),
        key=f"jump_step_{i}",
        use_container_width=True,
        disabled=disabled,
        on_click=lambda target=i: st.session_state.update({"setup_step": target}),
    )

name_cols = st.columns(len(setup_steps))
for i, step_name in enumerate(setup_steps, start=1):
    name_cols[i - 1].caption(step_name)
# ═══════════════════════════════════════════════════════════════════════════════
# FULL-PAGE SETUP WIZARD
# ═══════════════════════════════════════════════════════════════════════════════

api_key = st.secrets.get("OPENAI_API_KEY", None)
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

df = None
raw_df = st.session_state.get("raw_df")
pdf_files = st.session_state.get("pdf_files", None)
dimension_keys = st.session_state.get("resolved_dimension_keys", [])
metric_keys = st.session_state.get("resolved_metric_keys", [])

step = st.session_state["setup_step"]

# ── STEP 1: Upload data ────────────────────────────────────────────────────────
if step == 1:
    st.subheader("Step 1 — Upload Data")

    st.info(
    """
    **What data works best with Oops AI**
    
    Oops AI analyzes **time-series KPI datasets**.
    ✔ One **date column** (e.g., daily, weekly, or monthly)
    ✔ At least one **numeric KPI metric** (e.g., revenue, sales, conversions)
    ✔ Optional segment columns (e.g., region, category, device, etc.)
    
    Examples: sales dashboards, marketing metrics, operational KPIs.
    
    Avoid uploading datasets that only contain IDs or text fields.
    You will select the **date column, KPI metrics, and segment columns** in the next step.
    """
    )

    data_source = st.radio(
        "Choose data",
        ["Use sample dataset", "Upload your own file"],
        index=0,
        key="wizard_data_source",
    )

    if data_source == "Use sample dataset":
        sample_path = Path(__file__).parent / "data" / "sample_ecommerce_kpi_data.csv"
        if sample_path.exists():
            sample_df = pd.read_csv(sample_path)
        
            previous_raw_df = st.session_state.get("raw_df")
            is_new_dataset = (
                previous_raw_df is None
                or list(previous_raw_df.columns) != list(sample_df.columns)
                or len(previous_raw_df) != len(sample_df)
            )
        
            raw_df = sample_df
            st.session_state["raw_df"] = raw_df
        
            if is_new_dataset:
                for key in [
                    "df",
                    "selected_analysis_columns",
                    "selected_metrics",
                    "selected_groups",
                    "resolved_dimension_keys",
                    "resolved_metric_keys",
                    "profile_df",
                    "recommended_exclusions",
                    "compatibility",
                    "anomaly_candidates",
                    "date_freq",
                    "wide_info",
                    "agent_result",
                    "rag_result",
                    "rag_baseline",
                    "rag_mode",
                    "rag_error",
                    "rag_pipeline",
                    "vision_result",
                    "data_summary",
                    "consistency",
                    "analysis_ran",
                ]:
                    st.session_state.pop(key, None)
        
            st.success(f"Loaded sample dataset: {len(raw_df):,} rows")
        else:
            st.error("sample_ecommerce_kpi_data.csv not found in data/")
    else:
        uploaded = st.file_uploader(
            "Upload data file",
            type=["csv", "xlsx", "xls", "tsv"],
            key="wizard_upload",
        )
        if uploaded:
            try:
                suffix = Path(uploaded.name).suffix.lower()
                if suffix == ".csv":
                    raw_df = pd.read_csv(uploaded)
                elif suffix == ".tsv":
                    raw_df = pd.read_csv(uploaded, sep="\t")
                elif suffix in [".xlsx", ".xls"]:
                    raw_df = pd.read_excel(uploaded)
                else:
                    st.error("Unsupported file type. Please upload CSV, TSV, or Excel.")
                    raw_df = None

                if raw_df is not None:
                    previous_raw_df = st.session_state.get("raw_df")
                    is_new_dataset = (
                        previous_raw_df is None
                        or list(previous_raw_df.columns) != list(raw_df.columns)
                        or len(previous_raw_df) != len(raw_df)
                    )
                
                    st.session_state["raw_df"] = raw_df
                
                    if is_new_dataset:
                        for key in [
                            "df",
                            "selected_analysis_columns",
                            "selected_metrics",
                            "selected_groups",
                            "resolved_dimension_keys",
                            "resolved_metric_keys",
                            "profile_df",
                            "recommended_exclusions",
                            "compatibility",
                            "anomaly_candidates",
                            "date_freq",
                            "wide_info",
                            "agent_result",
                            "rag_result",
                            "rag_baseline",
                            "rag_mode",
                            "rag_error",
                            "rag_pipeline",
                            "vision_result",
                            "data_summary",
                            "consistency",
                            "analysis_ran",
                        ]:
                            st.session_state.pop(key, None)
                
                    st.success(f"Uploaded file: {len(raw_df):,} rows")
            except Exception as e:
                st.error(f"Could not read uploaded file: {e}")

    if raw_df is not None:
        st.dataframe(raw_df.head(20), use_container_width=True)

    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col3:
        st.button("Next ➜", on_click=go_next_step, disabled=raw_df is None)
        
# ── STEP 2: Configure columns ──────────────────────────────────────────────────
elif step == 2:
    st.subheader("Step 2 — Configure Columns")

    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        st.warning("Upload a dataset first.")
        st.button("⬅ Back", on_click=go_prev_step)
        st.stop()

    preview_df, _ = normalize_columns(raw_df.copy())
    inferred_date = infer_date_column(preview_df)
    all_cols = list(preview_df.columns)

    if len(all_cols) > 0:
        default_date_idx = all_cols.index(inferred_date) if inferred_date in all_cols else 0
        selected_date_col = st.selectbox(
            "Date column",
            all_cols,
            index=default_date_idx,
            key="wizard_date_col",
        )
    else:
        selected_date_col = None

    if selected_date_col is not None:
        try:
            loaded = load_data_from_df(raw_df, selected_date_col)
            df = loaded["df"]
            dimension_keys = loaded["dimension_keys"]
            metric_keys = loaded["metric_keys"]

            set_active_dataframe(df)

            profile_df = profile_dataframe(df)
            id_like_cols = detect_id_like_columns(df)
            constant_cols = detect_constant_columns(df)
            high_missing_cols = detect_high_missing_columns(df, threshold=0.9)
            wide_info = detect_wide_format_patterns(df)
            compatibility = score_dataset_compatibility(df, "date", metric_keys, dimension_keys)
            date_freq = infer_date_frequency(df["date"])

            recommended_exclusions = sorted(set(
                [c for c in id_like_cols if c != "date"] +
                [c for c in constant_cols if c != "date"] +
                [c for c in high_missing_cols if c != "date"]
            ))

            analysis_options = [c for c in df.columns if c != "date"]
            
            default_analysis_columns = [
                c for c in analysis_options if c not in recommended_exclusions
            ]
            
            saved_analysis_defaults = st.session_state.get(
                "selected_analysis_columns",
                default_analysis_columns
            )
            
            valid_analysis_defaults = [c for c in saved_analysis_defaults if c in analysis_options]
            
            selected_analysis_columns = st.multiselect(
                "Include columns",
                options=analysis_options,
                default=valid_analysis_defaults,
                key="wizard_analysis_cols",
            )
            
            allowed_metric_options = [c for c in metric_keys if c in selected_analysis_columns]
            allowed_group_options = [c for c in dimension_keys if c in selected_analysis_columns]

            dynamic_metric_defaults = [c for c in suggest_default_metrics(df) if c in allowed_metric_options]
            dynamic_group_defaults = [c for c in suggest_default_groups(df) if c in allowed_group_options]
            
            saved_metric_defaults = st.session_state.get(
                "selected_metrics",
                dynamic_metric_defaults or allowed_metric_options[:min(3, len(allowed_metric_options))]
            )
            saved_group_defaults = st.session_state.get(
                "selected_groups",
                dynamic_group_defaults
            )
            
            valid_metric_defaults = [c for c in saved_metric_defaults if c in allowed_metric_options]
            valid_group_defaults = [c for c in saved_group_defaults if c in allowed_group_options]
            
            selected_metrics = st.multiselect(
                "Performance metrics",
                options=allowed_metric_options,
                default=valid_metric_defaults,
                key="wizard_metrics",
            )
            
            selected_groups = st.multiselect(
                "Segment columns",
                options=allowed_group_options,
                default=valid_group_defaults,
                key="wizard_groups",
            )
            
            st.session_state["selected_analysis_columns"] = selected_analysis_columns
            st.session_state["selected_metrics"] = selected_metrics
            st.session_state["selected_groups"] = selected_groups

            validation = validate_dataset_for_analysis(df, "date", selected_metrics, selected_groups)
            for err in validation["errors"]:
                st.error(err)
            for warn in validation["warnings"]:
                st.warning(warn)

            anomaly_candidates = suggest_anomaly_dates(df, selected_metrics, top_n=5)

            st.session_state["df"] = df
            st.session_state["selected_analysis_columns"] = selected_analysis_columns
            st.session_state["selected_metrics"] = selected_metrics
            st.session_state["selected_groups"] = selected_groups
            st.session_state["resolved_dimension_keys"] = dimension_keys
            st.session_state["resolved_metric_keys"] = metric_keys
            st.session_state["profile_df"] = profile_df
            st.session_state["recommended_exclusions"] = recommended_exclusions
            st.session_state["compatibility"] = compatibility
            st.session_state["anomaly_candidates"] = anomaly_candidates
            st.session_state["date_freq"] = date_freq
            st.session_state["wide_info"] = wide_info

            for key in [
                "agent_result",
                "rag_result",
                "rag_baseline",
                "rag_mode",
                "rag_error",
                "rag_pipeline",
                "vision_result",
                "data_summary",
                "consistency",
            ]:
                st.session_state.pop(key, None)
            
            st.session_state["analysis_ran"] = False

        except Exception as e:
            st.error(
                f"Could not prepare this dataset for analysis: {e}. "
                "Make sure your file has one usable date column and at least one numeric KPI column."
            )
            st.session_state["df"] = None

    st.divider()

    col1, col2, col3 = st.columns([1,1,1])
    
    with col1:
        if st.session_state["setup_step"] > 1:
            st.button("⬅ Back", on_click=go_prev_step)
    
    with col3:
        if st.session_state["setup_step"] < TOTAL_STEPS:
            st.button("Next ➜", on_click=go_next_step)

# ── STEP 3: Data readiness ─────────────────────────────────────────────────────
elif step == 3:
    st.subheader("Step 3 — Data Readiness")

    df = st.session_state.get("df")
    compatibility = st.session_state.get("compatibility")
    profile_df = st.session_state.get("profile_df")
    recommended_exclusions = st.session_state.get("recommended_exclusions", [])
    anomaly_candidates = st.session_state.get("anomaly_candidates", [])
    date_freq = st.session_state.get("date_freq", "unknown")
    wide_info = st.session_state.get("wide_info", {"likely_wide_format": False})

    if df is None or compatibility is None:
        st.warning("Configure your columns first.")
        st.button("⬅ Back", on_click=go_prev_step)
        st.stop()

    st.metric("Readiness Score", f'{compatibility["score"]}/100', compatibility["label"])

    for reason in compatibility["reasons"]:
        st.markdown(f"- {reason}")

    st.caption(f"Inferred date frequency: {date_freq}")

    if recommended_exclusions:
        st.info(f"Recommended exclusions: {recommended_exclusions}")

    if wide_info.get("likely_wide_format"):
        st.warning("This file may be in wide format. Oops AI works best with long-format time-series data.")

    if anomaly_candidates:
        st.caption(f"Suggested anomaly dates: {', '.join(anomaly_candidates)}")

    with st.expander("Schema preview", expanded=False):
        st.dataframe(profile_df, use_container_width=True)

    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.button("⬅ Back", on_click=go_prev_step)
    
    with col3:
        st.button("Next ➜", on_click=go_next_step)

# ── STEP 4: Analysis settings ──────────────────────────────────────────────────
elif step == 4:
    st.subheader("Step 4 — Analysis Settings")

    candidate_dates = st.session_state.get("anomaly_candidates", [])
    default_anomaly_value = pd.Timestamp("2024-04-20")

    if candidate_dates:
        suggested_default = pd.Timestamp(candidate_dates[0])
        use_suggested = st.checkbox("Use suggested anomaly date", value=True, key="wizard_use_suggested")
        anomaly_date = st.date_input(
            "Anomaly start date",
            value=suggested_default if use_suggested else default_anomaly_value,
            key="wizard_anomaly_date",
        )
        st.caption(f"Top candidate dates: {', '.join(candidate_dates[:5])}")
    else:
        anomaly_date = st.date_input(
            "Anomaly start date",
            value=default_anomaly_value,
            key="wizard_anomaly_date",
        )

    top_k = st.slider("RAG top-k chunks", 3, 10, 5, key="wizard_top_k")
    run_vision = st.checkbox("Enable vision analysis", value=True, key="wizard_run_vision")
    debug_mode = st.checkbox("Debug mode", value=False, key="wizard_debug_mode")

    pdf_files = st.file_uploader(
        "Upload PDFs for RAG (optional)",
        type=["pdf"],
        accept_multiple_files=True,
        key="wizard_pdf_files",
        help="Powers the RAG explanation. Skip to use LLM-only mode."
    )

    st.session_state["anomaly_date"] = anomaly_date
    st.session_state["top_k"] = top_k
    st.session_state["run_vision"] = run_vision
    st.session_state["debug_mode"] = debug_mode
    st.session_state["pdf_files"] = pdf_files

    for key in [
        "agent_result",
        "rag_result",
        "rag_baseline",
        "rag_mode",
        "rag_error",
        "rag_pipeline",
        "vision_result",
        "data_summary",
        "consistency",
    ]:
        st.session_state.pop(key, None)
    
    st.session_state["analysis_ran"] = False

    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.button("⬅ Back", on_click=go_prev_step)
    
    with col3:
        st.button("Next ➜", on_click=go_next_step)

# ── STEP 5: Ready to run ───────────────────────────────────────────────────────
elif step == 5:
    st.subheader("Step 5 — Run Analysis")

    df = st.session_state.get("df")
    selected_metrics = st.session_state.get("selected_metrics", [])
    selected_groups = st.session_state.get("selected_groups", [])
    anomaly_date = st.session_state.get("anomaly_date", pd.Timestamp("2024-04-20"))
    top_k = st.session_state.get("top_k", 5)
    run_vision = st.session_state.get("run_vision", True)
    debug_mode = st.session_state.get("debug_mode", False)
    pdf_files = st.session_state.get("pdf_files", None)

    st.write("Review your setup and run the analysis.")
    st.markdown(f"**Metrics:** {selected_metrics if selected_metrics else 'None selected'}")
    st.markdown(f"**Segment columns:** {selected_groups if selected_groups else 'None selected'}")
    st.markdown(f"**Anomaly date:** {anomaly_date}")

    session_id = get_session_id(st.session_state)
    run_clicked = st.button("🚀 Run Full Analysis", use_container_width=True)
    if run_clicked:
        rate_ok, rate_msg = rate_limiter.check(session_id)
    else:
        rate_ok, rate_msg = True, "ok"

    run_btn = run_clicked and rate_ok
    if run_clicked:
        st.session_state["analysis_ran"] = True
    
    if run_clicked and not rate_ok:
        st.warning(rate_msg)

    if "request_id" not in st.session_state:
        st.session_state["request_id"] = str(uuid.uuid4())

    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.button("⬅ Back", on_click=go_prev_step)

# Make values available below
df = st.session_state.get("df")
selected_metrics = st.session_state.get("selected_metrics", [])
selected_groups = st.session_state.get("selected_groups", [])
anomaly_date = st.session_state.get("anomaly_date", pd.Timestamp("2024-04-20"))
top_k = st.session_state.get("top_k", 5)
run_vision = st.session_state.get("run_vision", True)
debug_mode = st.session_state.get("debug_mode", False)
pdf_files = st.session_state.get("pdf_files", None)
run_btn = locals().get("run_btn", False)

analysis_ran = st.session_state.get("analysis_ran", False)
        
# ── Guard: only show results after setup is complete ───────────────────────────
if step < 5:
    st.info("Complete setup to run analysis.")
    st.stop()

if not analysis_ran and not run_btn:
    st.info("Click **Run Full Analysis** in Step 5 to generate results.")
    st.stop()
    
if df is None:
    st.info("Complete setup first.")
    st.stop()

if len(st.session_state.get("selected_metrics", [])) == 0:
    st.warning("Choose at least one numeric performance metric before running analysis.")
    st.stop()

if not os.environ.get("OPENAI_API_KEY"):
    st.warning(
        "⚠️ No OpenAI API key found. In deployment, add OPENAI_API_KEY to Streamlit secrets. "
        "For local testing, add one before running analysis."
    )
    st.stop()
    
# Lazy imports (only after API key is set)
try:
    from src.agent import run_agent
    from src.multimodal import KPIChartGenerator, MultimodalAnalyzer, CrossModalChecker
    from src.rag_pipeline import RAGPipeline
    from src.prompts import REACT_AGENT_SYSTEM_PROMPT
    IMPORTS_OK = True
except ImportError as e:
    st.error(f"Import error: {e}\n\nMake sure you've installed requirements:\n"
             "`pip install -r requirements.txt`")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Anomaly Detection",
    "🧠 Root Cause (RAG)",
    "👁️ Vision Analysis",
    "📋 Data Explorer",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### <span class='step-badge'>STEP 1</span> Anomaly Detection via Tool Calling", unsafe_allow_html=True)
    st.caption("The agent queries the dataset using tool calls to locate statistical anomalies.")

    # KPI summary cards
    anomaly_ts = pd.Timestamp(anomaly_date)
    before = df[df["date"] < anomaly_ts]
    after = df[df["date"] >= anomaly_ts]

    selected_metrics = st.session_state.get("selected_metrics", [])
    selected_groups = st.session_state.get("selected_groups", [])
    numeric_cols = df.select_dtypes("number").columns.tolist()

    summary_metrics = [c for c in selected_metrics if c in numeric_cols]

    if len(summary_metrics) == 0:
        st.info("No performance metrics selected. Go back to setup and choose one or more numeric metrics.")
    else:
        cols = st.columns(len(summary_metrics))
        for col, m in zip(cols, summary_metrics):
            b_mean = before[m].mean() if len(before) else 0
            a_mean = after[m].mean() if len(after) else 0
            pct = (a_mean - b_mean) / b_mean * 100 if b_mean else 0
            sign = "▼" if pct < 0 else "▲"
            cls = "negative" if pct < 0 else "positive"
            col.markdown(f"""
            <div class="metric-card">
              <div class="value">{a_mean:,.2f}</div>
              <div class="label">{m.replace('_',' ').title()}</div>
              <div class="delta {cls}">{sign} {abs(pct):.1f}% vs pre-anomaly</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("##### Time-series charts with anomaly window")
    with st.spinner("Generating charts..."):
        chart_gen = KPIChartGenerator(df, output_dir=tempfile.mkdtemp())

        plot_pairs = []
        for metric in summary_metrics[:4]:
            split_by = selected_groups[0] if len(selected_groups) > 0 else None
            plot_pairs.append((metric, split_by))

        if len(plot_pairs) == 0:
            st.info("No charts generated because no valid metrics were selected.")
        else:
            chart_cols = st.columns(2)

            for i, (metric, split_by) in enumerate(plot_pairs):
                with chart_cols[i % 2]:
                    try:
                        p = chart_gen.plot_metric_timeseries(
                            metric,
                            split_by=split_by,
                            anomaly_start=str(anomaly_date)
                        )
                        caption = metric.replace("_", " ").title()
                        if split_by:
                            caption += f" by {split_by.replace('_', ' ').title()}"
                        st.image(str(p), caption=caption, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not plot {metric}: {e}")

            if len(selected_groups) > 0 and len(summary_metrics) > 0:
                try:
                    p_compare = chart_gen.plot_dimension_comparison(
                        summary_metrics[0],
                        selected_groups[0],
                        before_end=str(anomaly_date - pd.Timedelta(days=1)),
                        after_start=str(anomaly_date)
                    )
                    st.image(
                        str(p_compare),
                        caption=f"{summary_metrics[0].replace('_', ' ').title()} before vs after by {selected_groups[0].replace('_', ' ').title()}",
                        use_container_width=True
                    )
                except Exception as e:
                    st.warning(f"Could not generate comparison chart: {e}")

    # Tool-calling agent
    st.markdown("---")
    st.markdown("##### 🤖 Agent tool-calling output")

    if run_btn or "agent_result" not in st.session_state:
        with st.spinner("Agent is querying the dataset via tool calls..."):
            request_id = st.session_state["request_id"]
            step_start = time.time()

            try:
                selected_metrics = st.session_state.get("selected_metrics", [])
                selected_groups = st.session_state.get("selected_groups", [])

                anomaly_query = (
                    f"Analyze the KPI dataset. The anomaly appears to start around {anomaly_date}. "
                    f"Focus on these performance metrics if relevant: {selected_metrics}. "
                    f"Use these grouping columns if relevant: {selected_groups}. "
                    f"Identify which metrics and segments are most affected. "
                    f"Use the available tools to query the data and provide a structured finding."
                )

                app_log(
                    "agent_analysis_started",
                    request_id=request_id,
                    selected_metrics=selected_metrics,
                    selected_groups=selected_groups,
                )

                agent_result = run_agent(
                    system_prompt=REACT_AGENT_SYSTEM_PROMPT,
                    user_prompt=anomaly_query,
                    debug=False,
                    return_state=True,
                )
                st.session_state["agent_result"] = agent_result

                state = agent_result.get("state", {})
                app_log(
                    "agent_analysis_finished",
                    request_id=request_id,
                    elapsed_sec=round(time.time() - step_start, 3),
                    actions=len(state.get("actions", [])),
                    observations=len(state.get("observations", [])),
                    completed_steps=state.get("completed_steps", []),
                )
            except Exception as e:
                app_log(
                    "agent_analysis_error",
                    request_id=request_id,
                    elapsed_sec=round(time.time() - step_start, 3),
                    error_type=type(e).__name__,
                    error=str(e),
                )
                st.session_state["agent_result"] = f"Agent error: {e}"
    result = st.session_state.get("agent_result", "")
    if result:
        raw_answer = result.get("final_answer", "") if isinstance(result, dict) else str(result)

        # ── SECURITY: Filter agent output for sensitive data ───────────────
        filtered_answer, findings = filter_output(raw_answer)
        if findings:
            st.warning(f"🔒 Output filtered — sensitive data redacted: {', '.join(findings)}")

        st.markdown(
            f"<div class='rag-box'>{filtered_answer}</div>",
            unsafe_allow_html=True
        )

        if debug_mode and isinstance(result, dict):
            with st.expander("Debug trace"):
                st.json({
                    "request_id": st.session_state.get("request_id"),
                    "selected_metrics": st.session_state.get("selected_metrics", []),
                    "selected_groups": st.session_state.get("selected_groups", []),
                    "agent_state": result.get("state", {}),
                })


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RAG Root Cause
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### <span class='step-badge'>STEP 2</span> RAG-Powered Root Cause Explanation", unsafe_allow_html=True)
    st.caption("Retrieves relevant business context from the knowledge base to ground the LLM's explanation.")

    # Build anomaly summary from data
    selected_metrics = st.session_state.get("selected_metrics", [])
    selected_groups = st.session_state.get("selected_groups", [])

    anomaly_summary = build_dynamic_anomaly_summary(
        df,
        anomaly_ts,
        selected_metrics,
        selected_groups
    )

    with st.expander("📄 Anomaly summary sent to the LLM", expanded=False):
        st.code(anomaly_summary, language="text")

    if run_btn or "rag_result" not in st.session_state:
        request_id = st.session_state["request_id"]
        step_start = time.time()

        rag_docs_dir = Path(tempfile.mkdtemp())
        knowledge_base_dir = Path(__file__).parent / "knowledge_base"

        if pdf_files:
            for pdf in pdf_files:
                with open(rag_docs_dir / pdf.name, "wb") as f:
                    f.write(pdf.read())
            docs_source = rag_docs_dir
        elif knowledge_base_dir.exists() and any(knowledge_base_dir.glob("*.pdf")):
            docs_source = knowledge_base_dir
        else:
            docs_source = None

        with st.spinner("Building RAG pipeline and retrieving context..."):
            try:
                rag = RAGPipeline(
                    docs_dir=docs_source or rag_docs_dir,
                    persist_dir=str(Path(tempfile.mkdtemp()) / "chroma_db"),
                )
                if docs_source:
                    rag.build_or_load()
                    rag_result = rag.generate_rag_response(anomaly_summary, k=top_k)
                    baseline = rag.generate_baseline_response(anomaly_summary)
                    st.session_state["rag_result"] = rag_result
                    st.session_state["rag_baseline"] = baseline
                    st.session_state["rag_mode"] = "rag"
                    st.session_state["rag_pipeline"] = rag

                    app_log(
                        "rag_completed",
                        request_id=request_id,
                        mode="rag",
                        elapsed_sec=round(time.time() - step_start, 3),
                        top_k=top_k,
                    )
                else:
                    baseline = rag.generate_baseline_response(anomaly_summary)
                    st.session_state["rag_baseline"] = baseline
                    st.session_state["rag_mode"] = "baseline"
                    st.session_state["rag_pipeline"] = rag

                    app_log(
                        "rag_completed",
                        request_id=request_id,
                        mode="baseline",
                        elapsed_sec=round(time.time() - step_start, 3),
                    )
            except Exception as e:
                st.session_state["rag_mode"] = "error"
                st.session_state["rag_error"] = str(e)

                app_log(
                    "rag_error",
                    request_id=request_id,
                    elapsed_sec=round(time.time() - step_start, 3),
                    error_type=type(e).__name__,
                    error=str(e),
                )

    mode = st.session_state.get("rag_mode", "none")

    if mode == "rag":
        rag_result = st.session_state["rag_result"]
        baseline   = st.session_state.get("rag_baseline", "")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 🔵 RAG-augmented response")
            st.caption("Grounded in retrieved knowledge base documents")

            # ── SECURITY: Filter RAG output ────────────────────────────────
            raw_rag = rag_result["response"]
            filtered_rag, rag_findings = filter_output(raw_rag)
            if rag_findings:
                st.warning(f"🔒 Output filtered: {', '.join(rag_findings)}")
            st.markdown(f"<div class='rag-box'>{filtered_rag}</div>", unsafe_allow_html=True)

            st.markdown("**Sources retrieved:**")
            for i, doc in enumerate(rag_result["retrieved_docs"], 1):
                st.markdown(f"- [{i}] `{doc.metadata.get('source')}` page {doc.metadata.get('page')}")

        with col_b:
            st.markdown("##### ⚪ Baseline (no RAG)")
            st.caption("LLM with no retrieval — for comparison")

            # ── SECURITY: Filter baseline output ───────────────────────────
            filtered_baseline, _ = filter_output(baseline)
            st.markdown(f"<div class='baseline-box'>{filtered_baseline}</div>", unsafe_allow_html=True)

    elif mode == "baseline":
        st.info("💡 No PDFs found — running in baseline (no RAG) mode. Add PDFs in Step 4 to enable RAG.")
        baseline = st.session_state.get("rag_baseline", "")

        # ── SECURITY: Filter baseline output ──────────────────────────────
        filtered_baseline, _ = filter_output(baseline)
        st.markdown("##### LLM Root Cause Explanation")
        st.markdown(f"<div class='baseline-box'>{filtered_baseline}</div>", unsafe_allow_html=True)

    elif mode == "error":
        st.error(f"RAG error: {st.session_state.get('rag_error')}")

    else:
        st.info("Use the setup flow above, then click **Run Full Analysis** in Step 5.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECURITY: Free-form follow-up question box
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("##### 💬 Ask a follow-up question")
    st.caption("Ask anything about the anomaly. Inputs are validated before reaching the LLM.")

    user_question = st.text_input(
        "Your question",
        placeholder="e.g. Which region was most affected by the revenue drop?",
        key="followup_input",
        label_visibility="collapsed",
    )

    ask_btn = st.button("Ask", key="ask_followup")

    if ask_btn and user_question:
        # Step 1: Rate limit check
        session_id = get_session_id(st.session_state)
        rate_ok, rate_msg = rate_limiter.check(session_id)
        if not rate_ok:
            st.warning(rate_msg)
        else:
            # Step 2: Input validation (injection / jailbreak check)
            is_safe, reason = validate_input(user_question)
            if not is_safe:
                st.markdown(
                    f"<div class='security-blocked'>🚫 <strong>Blocked:</strong> {reason}</div>",
                    unsafe_allow_html=True
                )
            else:
                # Step 3: Sanitize and send to LLM
                clean_question = sanitize_input(user_question)
                st.markdown(
                    f"<div class='security-passed'>✅ <strong>Input validated</strong> — sending to LLM...</div>",
                    unsafe_allow_html=True
                )

                rag_pipeline = st.session_state.get("rag_pipeline")
                if rag_pipeline is None:
                    st.warning("Run the full analysis first to initialize the RAG pipeline.")
                else:
                    with st.spinner("Generating answer..."):
                        try:
                            context = f"Dataset anomaly context:\n{anomaly_summary}\n\nUser question: {clean_question}"
                            followup_response = rag_pipeline.generate_baseline_response(context)

                            # Step 4: Filter output before displaying
                            filtered_followup, followup_findings = filter_output(followup_response)
                            if followup_findings:
                                st.warning(f"🔒 Output filtered: {', '.join(followup_findings)}")

                            st.markdown(
                                f"<div class='rag-box'>{filtered_followup}</div>",
                                unsafe_allow_html=True
                            )
                        except Exception as e:
                            st.error(f"Error generating answer: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Vision Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### <span class='step-badge'>STEP 3</span> Vision Analysis & Cross-Modal Consistency", unsafe_allow_html=True)
    st.caption("GPT-4o-mini reads the chart image and its findings are compared against data-derived ground truth.")

    if not run_vision:
        st.info("Vision analysis is disabled. Enable it in Step 4 of setup.")
    else:
        analyzer = MultimodalAnalyzer()
        checker = CrossModalChecker()

        selected_metrics = st.session_state.get("selected_metrics", [])
        selected_groups = st.session_state.get("selected_groups", [])

        vision_metric = selected_metrics[0] if len(selected_metrics) > 0 else None
        vision_group = selected_groups[0] if len(selected_groups) > 0 else None

        if vision_metric is None:
            st.info("Select at least one performance metric during setup to run vision analysis.")
        else:
            with st.spinner("Generating chart for vision analysis..."):
                try:
                    chart_gen = KPIChartGenerator(df, output_dir=tempfile.mkdtemp())
                    overview_path = chart_gen.plot_metric_timeseries(
                        vision_metric,
                        split_by=vision_group,
                        anomaly_start=str(anomaly_date)
                    )
                    st.image(
                        str(overview_path),
                        caption=f"{vision_metric.replace('_', ' ').title()}" + (
                            f" by {vision_group.replace('_', ' ').title()}" if vision_group else ""
                        ),
                        use_container_width=True
                    )
                except Exception as e:
                    overview_path = None
                    st.error(f"Could not generate chart for vision analysis: {e}")

            if overview_path is not None and (run_btn or "vision_result" not in st.session_state):
                with st.spinner("GPT-4o-mini analyzing chart..."):
                    request_id = st.session_state["request_id"]
                    step_start = time.time()

                    try:
                        vis = analyzer.extract_anomaly_from_chart(overview_path, metric=vision_metric)
                        data_summary = analyzer.build_data_summary(
                            df,
                            vision_metric,
                            str(anomaly_date),
                            vision_group
                        )
                        consistency = checker.check_consistency(data_summary, vis)
                        st.session_state["vision_result"] = vis
                        st.session_state["data_summary"] = data_summary
                        st.session_state["consistency"] = consistency

                        app_log(
                            "vision_completed",
                            request_id=request_id,
                            elapsed_sec=round(time.time() - step_start, 3),
                            vision_metric=vision_metric,
                            vision_group=vision_group,
                            consistency_score=consistency.get("score") if consistency else None,
                        )
                    except Exception as e:
                        st.session_state["vision_result"] = {"raw_text": f"Error: {e}"}
                        st.session_state["consistency"] = None

                        app_log(
                            "vision_error",
                            request_id=request_id,
                            elapsed_sec=round(time.time() - step_start, 3),
                            error_type=type(e).__name__,
                            error=str(e),
                        )

        vis = st.session_state.get("vision_result", {})
        consistency = st.session_state.get("consistency")

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("##### 👁️ Vision model findings")
            fields = [
                ("anomaly_detected", "Anomaly detected?"),
                ("anomaly_date", "Detected date"),
                ("magnitude", "Magnitude"),
                ("affected_segments", "Affected segments"),
                ("trend_before", "Trend before"),
                ("trend_after", "Trend after"),
            ]
            for key, label in fields:
                val = vis.get(key, "—")
                st.markdown(f"**{label}:** {val}")

        with col_r:
            st.markdown("##### ✅ Cross-modal consistency")
            if consistency:
                score = consistency["score"]
                label = consistency["label"]
                badge_cls = f"badge-{label.lower()}"
                st.markdown(
                    f"Consistency: <span class='{badge_cls}'>{label} ({score:.0%})</span>",
                    unsafe_allow_html=True
                )
                st.markdown(f"_{consistency.get('explanation', '')}_")

                if consistency.get("matching"):
                    st.markdown("**✓ Confirmed by vision:**")
                    for pt in consistency["matching"]:
                        st.markdown(f"- {pt}")

                if consistency.get("discrepancies"):
                    st.markdown("**✗ Discrepancies / missed:**")
                    for pt in consistency["discrepancies"]:
                        st.markdown(f"- {pt}")
            else:
                st.info("Run analysis to see consistency results.")

        st.markdown("---")
        st.markdown("##### Upload your own dashboard screenshot")
        uploaded_img = st.file_uploader("Upload a dashboard image", type=["png", "jpg", "jpeg"])
        if uploaded_img:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(uploaded_img.read())
                tmp_path = tmp.name
            st.image(tmp_path, caption="Your dashboard", use_container_width=True)
            if st.button("Analyze uploaded image"):
                with st.spinner("Analyzing..."):
                    desc = analyzer.analyze_dashboard_screenshot(tmp_path)

                # ── SECURITY: Filter vision output ─────────────────────────
                filtered_desc, _ = filter_output(desc)
                st.markdown("**Analysis:**")
                st.markdown(f"<div class='rag-box'>{filtered_desc}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### 📋 Data Explorer")
    st.caption("Inspect cleaned data, schema, and compatibility checks before analysis.")

    compatibility = st.session_state.get("compatibility")
    if compatibility:
        st.metric("Readiness Score", f'{compatibility["score"]}/100', compatibility["label"])
        st.write("Why:", ", ".join(compatibility["reasons"]))

    profile_df = st.session_state.get("profile_df")
    if profile_df is not None:
        st.markdown("##### Schema summary")
        st.dataframe(profile_df, use_container_width=True)

    recommended_exclusions = st.session_state.get("recommended_exclusions", [])
    if recommended_exclusions:
        st.markdown("##### Recommended exclusions")
        st.write(recommended_exclusions)

    st.markdown("##### Cleaned dataset preview")
    st.dataframe(df.head(100), use_container_width=True)

    # Filters
    dimension_keys = st.session_state.get("resolved_dimension_keys", [])
    metric_keys = st.session_state.get("resolved_metric_keys", [])
    selected_analysis_columns = st.session_state.get("selected_analysis_columns", [])

    allowed_dimension_keys = [c for c in dimension_keys if c in selected_analysis_columns]
    allowed_metric_keys = [c for c in metric_keys if c in selected_analysis_columns]

    filters = {}
    filter_cols = st.columns(3)

    for i, col_name in enumerate(allowed_dimension_keys[:3]):
        opts = ["All"] + sorted(df[col_name].dropna().astype(str).unique().tolist())
        sel = filter_cols[i].selectbox(col_name.replace("_", " ").title(), opts)
        if sel != "All":
            filters[col_name] = sel

    if len(allowed_metric_keys) == 0:
        st.info("No numeric metrics available to summarize.")
        metric_view = None
    else:
        metric_view = st.selectbox("Metric to summarize", allowed_metric_keys)

    fdf = df.copy()
    for col_name, val in filters.items():
        fdf = fdf[fdf[col_name].astype(str) == str(val)]

    if metric_view is not None:
        summary = fdf.groupby("date")[metric_view].mean().reset_index()
        summary.columns = ["date", f"avg_{metric_view}"]
        st.line_chart(summary.set_index("date"))

    st.dataframe(fdf.head(200), use_container_width=True)

    # Download
    csv_bytes = fdf.to_csv(index=False).encode()
    st.download_button("⬇️ Download filtered data", csv_bytes,
                       "filtered_kpi.csv", "text/csv")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#94a3b8;font-size:0.78rem'>"
    "KPI Root Cause Analysis Engine · Milestone 12 · "
    "Built with Streamlit + OpenAI GPT-4o-mini + LangChain + ChromaDB · 🔒 Security hardened"
    "</p>",
    unsafe_allow_html=True,
)
