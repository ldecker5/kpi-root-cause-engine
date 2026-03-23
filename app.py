"""
app.py
------
MILESTONE 9: MVP — KPI Root Cause Analysis Engine
Streamlit interface that wires together:
  • Data loading (sample CSV or upload)
  • Anomaly detection via tool-calling agent
  • RAG-powered root cause explanation
  • Chart generation + GPT-4o-mini vision analysis
  • Cross-modal consistency check

Run:
    streamlit run app.py
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

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
  }
  .baseline-box {
    background: #fafafa;
    border-left: 3px solid #94a3b8;
    padding: 1rem 1.25rem;
    border-radius: 0 0.4rem 0.4rem 0;
    white-space: pre-wrap;
    font-size: 0.9rem;
    line-height: 1.6;
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
    <p>Anomaly detection · RAG-powered explanations · Vision analysis · Tool calling</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Configuration & Data Loading
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key = st.text_input("OpenAI API Key", type="password",
                            placeholder="sk-...",
                            help="Your key is never stored.")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("### 📁 Data Source")

    data_source = st.radio("Choose data", ["Use sample dataset", "Upload your own CSV"],
                           index=0)

    df = None

    if data_source == "Use sample dataset":
        sample_path = Path(__file__).parent / "data" / "sample_ecommerce_kpi_data.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path, parse_dates=["date"])
            st.success(f"✓ Loaded {len(df):,} rows")
        else:
            st.error("sample_ecommerce_kpi_data.csv not found in data/")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded, parse_dates=["date"])
            st.success(f"✓ {uploaded.name} — {len(df):,} rows")

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")
    pdf_files = st.file_uploader("Upload PDFs (optional)",
                                 type=["pdf"], accept_multiple_files=True,
                                 help="Powers the RAG explanation. Skip to use LLM-only mode.")

    st.markdown("---")
    st.markdown("### 🔧 Settings")
    anomaly_date = st.date_input("Anomaly start date",
                                 value=pd.Timestamp("2024-04-20"))
    top_k = st.slider("RAG top-k chunks", 3, 10, 5)
    run_vision = st.checkbox("Enable vision analysis", value=True)

    st.markdown("---")
    run_btn = st.button("🚀 Run Full Analysis", use_container_width=True)


# ── Guard: need data + API key ─────────────────────────────────────────────────
if df is None:
    st.info("👈 Load your data in the sidebar to get started.")
    st.stop()

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("⚠️ Enter your OpenAI API key in the sidebar to run the analysis.")
    st.stop()


# ── Lazy imports (only after API key is set) ───────────────────────────────────
try:
    from src.data_loader import load_data
    from src.agent import run_agent
    from src.multimodal import KPIChartGenerator, MultimodalAnalyzer, CrossModalChecker
    from src.rag_pipeline import RAGPipeline
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
    after  = df[df["date"] >= anomaly_ts]

    numeric_cols = df.select_dtypes("number").columns.tolist()
    summary_metrics = [c for c in ["revenue", "conversion_rate", "marketing_spend", "orders"]
                       if c in numeric_cols]

    cols = st.columns(len(summary_metrics))
    for col, m in zip(cols, summary_metrics):
        b_mean = before[m].mean() if len(before) else 0
        a_mean = after[m].mean()  if len(after)  else 0
        pct = (a_mean - b_mean) / b_mean * 100 if b_mean else 0
        sign = "▼" if pct < 0 else "▲"
        cls  = "negative" if pct < 0 else "positive"
        col.markdown(f"""
        <div class="metric-card">
          <div class="value">{a_mean:,.2f}</div>
          <div class="label">{m.replace('_',' ').title()}</div>
          <div class="delta {cls}">{sign} {abs(pct):.1f}% vs pre-anomaly</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart
    st.markdown("##### Time-series charts with anomaly window")
    with st.spinner("Generating charts..."):
        chart_gen = KPIChartGenerator(df, output_dir=tempfile.mkdtemp())

        c1, c2 = st.columns(2)
        with c1:
            p = chart_gen.plot_metric_timeseries(
                "revenue", split_by="device_type",
                anomaly_start=str(anomaly_date))
            st.image(str(p), caption="Revenue by Device Type", use_container_width=True)

        with c2:
            p2 = chart_gen.plot_metric_timeseries(
                "conversion_rate", split_by="device_type",
                anomaly_start=str(anomaly_date))
            st.image(str(p2), caption="Conversion Rate by Device Type", use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            p3 = chart_gen.plot_metric_timeseries(
                "marketing_spend", split_by="region",
                anomaly_start=str(anomaly_date))
            st.image(str(p3), caption="Marketing Spend by Region", use_container_width=True)

        with c4:
            p4 = chart_gen.plot_dimension_comparison(
                "revenue", "device_type",
                before_end=str(anomaly_date - pd.Timedelta(days=1)),
                after_start=str(anomaly_date))
            st.image(str(p4), caption="Revenue Before vs After Anomaly", use_container_width=True)

    # Tool-calling agent
    st.markdown("---")
    st.markdown("##### 🤖 Agent tool-calling output")

    if run_btn or "agent_result" not in st.session_state:
        with st.spinner("Agent is querying the dataset via tool calls..."):
            try:
                anomaly_query = (
                    f"Analyze the KPI dataset. The anomaly appears to start around "
                    f"{anomaly_date}. Identify which metrics and segments are most affected. "
                    f"Use the available tools to query the data and provide a structured finding."
                )
                agent_result = run_agent(
                    system_prompt=REACT_AGENT_SYSTEM_PROMPT,
                    user_prompt=anomaly_query,
                    debug=False,
                    return_state=True,
                )
                st.session_state["agent_result"] = agent_result
            except Exception as e:
                st.session_state["agent_result"] = f"Agent error: {e}"

    result = st.session_state.get("agent_result", "")
    if result:
        st.markdown(
            f"<div class='rag-box'>{result.get('final_answer', '')}</div>",
            unsafe_allow_html=True
        )
    
        with st.expander("ReAct Memory / State"):
            st.json(result.get("state", {}))

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RAG Root Cause
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### <span class='step-badge'>STEP 2</span> RAG-Powered Root Cause Explanation", unsafe_allow_html=True)
    st.caption("Retrieves relevant business context from the knowledge base to ground the LLM's explanation.")

    # Build anomaly summary from data
    def build_anomaly_summary(df, anomaly_ts):
        before = df[df["date"] < anomaly_ts]
        after  = df[df["date"] >= anomaly_ts]
        lines = [f"Anomaly detected starting {anomaly_ts.date()}."]
        for m in ["revenue", "conversion_rate", "marketing_spend"]:
            if m not in df.columns:
                continue
            b, a = before[m].mean(), after[m].mean()
            pct = (a - b) / b * 100 if b else 0
            lines.append(f"  {m}: {b:.3g} → {a:.3g} ({pct:+.1f}%)")
            if "device_type" in df.columns:
                for dev in df["device_type"].unique():
                    bd = before[before["device_type"]==dev][m].mean()
                    ad = after[after["device_type"]==dev][m].mean()
                    p2 = (ad - bd) / bd * 100 if bd else 0
                    lines.append(f"    [{dev}] {bd:.3g} → {ad:.3g} ({p2:+.1f}%)")
        return "\n".join(lines)

    anomaly_summary = build_anomaly_summary(df, anomaly_ts)

    with st.expander("📄 Anomaly summary sent to the LLM", expanded=False):
        st.code(anomaly_summary, language="text")

    if run_btn or "rag_result" not in st.session_state:
        # Handle PDFs
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
                    baseline   = rag.generate_baseline_response(anomaly_summary)
                    st.session_state["rag_result"]  = rag_result
                    st.session_state["rag_baseline"] = baseline
                    st.session_state["rag_mode"]    = "rag"
                else:
                    baseline = rag.generate_baseline_response(anomaly_summary)
                    st.session_state["rag_baseline"] = baseline
                    st.session_state["rag_mode"]    = "baseline"
            except Exception as e:
                st.session_state["rag_mode"] = "error"
                st.session_state["rag_error"] = str(e)

    mode = st.session_state.get("rag_mode", "none")

    if mode == "rag":
        rag_result = st.session_state["rag_result"]
        baseline   = st.session_state.get("rag_baseline", "")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 🔵 RAG-augmented response")
            st.caption("Grounded in retrieved knowledge base documents")
            st.markdown(f"<div class='rag-box'>{rag_result['response']}</div>",
                        unsafe_allow_html=True)

            st.markdown("**Sources retrieved:**")
            for i, doc in enumerate(rag_result["retrieved_docs"], 1):
                st.markdown(f"- [{i}] `{doc.metadata.get('source')}` page {doc.metadata.get('page')}")

        with col_b:
            st.markdown("##### ⚪ Baseline (no RAG)")
            st.caption("LLM with no retrieval — for comparison")
            st.markdown(f"<div class='baseline-box'>{baseline}</div>",
                        unsafe_allow_html=True)

    elif mode == "baseline":
        st.info("💡 No PDFs found — running in baseline (no RAG) mode. Upload PDFs in the sidebar to enable RAG.")
        baseline = st.session_state.get("rag_baseline", "")
        st.markdown("##### LLM Root Cause Explanation")
        st.markdown(f"<div class='baseline-box'>{baseline}</div>", unsafe_allow_html=True)

    elif mode == "error":
        st.error(f"RAG error: {st.session_state.get('rag_error')}")

    else:
        st.info("Click **Run Full Analysis** in the sidebar to generate the root cause explanation.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Vision Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### <span class='step-badge'>STEP 3</span> Vision Analysis & Cross-Modal Consistency", unsafe_allow_html=True)
    st.caption("GPT-4o-mini reads the chart image and its findings are compared against data-derived ground truth.")

    if not run_vision:
        st.info("Vision analysis is disabled. Enable it in the sidebar.")
    else:
        analyzer = MultimodalAnalyzer()
        checker  = CrossModalChecker()

        # Generate overview chart
        with st.spinner("Generating overview dashboard chart..."):
            overview_path = chart_gen.plot_multimodal_overview(str(anomaly_date))

        st.image(str(overview_path), caption="2×2 KPI Dashboard (sent to vision model)",
                 use_container_width=True)

        if run_btn or "vision_result" not in st.session_state:
            with st.spinner("GPT-4o-mini analyzing chart..."):
                try:
                    vis = analyzer.extract_anomaly_from_chart(overview_path, metric="revenue")
                    data_summary = analyzer.build_data_summary(
                        df, "revenue", str(anomaly_date), "device_type")
                    consistency = checker.check_consistency(data_summary, vis)
                    st.session_state["vision_result"]    = vis
                    st.session_state["data_summary"]     = data_summary
                    st.session_state["consistency"]      = consistency
                except Exception as e:
                    st.session_state["vision_result"] = {"raw_text": f"Error: {e}"}
                    st.session_state["consistency"]   = None

        vis = st.session_state.get("vision_result", {})
        consistency = st.session_state.get("consistency")

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("##### 👁️ Vision model findings")
            fields = [
                ("anomaly_detected",  "Anomaly detected?"),
                ("anomaly_date",      "Detected date"),
                ("magnitude",         "Magnitude"),
                ("affected_segments", "Affected segments"),
                ("trend_before",      "Trend before"),
                ("trend_after",       "Trend after"),
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
                    unsafe_allow_html=True)
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
        uploaded_img = st.file_uploader("Upload a dashboard image", type=["png","jpg","jpeg"])
        if uploaded_img:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(uploaded_img.read())
                tmp_path = tmp.name
            st.image(tmp_path, caption="Your dashboard", use_container_width=True)
            if st.button("Analyze uploaded image"):
                with st.spinner("Analyzing..."):
                    desc = analyzer.analyze_dashboard_screenshot(tmp_path)
                st.markdown("**Analysis:**")
                st.markdown(f"<div class='rag-box'>{desc}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### 📋 Raw Dataset Explorer")
    st.caption(f"{len(df):,} rows · {len(df.columns)} columns")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    filters = {}

    for col, label in [("region", "Region"), ("device_type", "Device Type")]:
        if col in df.columns:
            col_ref = col_f1 if col == "region" else col_f2
            opts = ["All"] + sorted(df[col].dropna().unique().tolist())
            sel = col_ref.selectbox(label, opts)
            if sel != "All":
                filters[col] = sel

    metric_view = col_f3.selectbox("Metric to summarize",
                                   [c for c in ["revenue","conversion_rate","marketing_spend","orders","sessions"]
                                    if c in df.columns])

    fdf = df.copy()
    for col, val in filters.items():
        fdf = fdf[fdf[col] == val]

    # Summary by date
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
    "KPI Root Cause Analysis Engine · Milestone 9 MVP · "
    "Built with Streamlit + OpenAI GPT-4o-mini + LangChain + ChromaDB"
    "</p>",
    unsafe_allow_html=True,
)
