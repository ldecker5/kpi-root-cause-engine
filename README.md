# KPI Root-Cause Analysis Engine

An LLM-powered system that automatically investigates why business KPIs changed.

## Project Structure

```
kpi_root_cause_engine/
│
├── .env                     ← Your OpenAI API key (you create this)
├── requirements.txt         ← Python packages to install
├── main.py                  ← Run this to test the full agent
│
├── src/
│   ├── data_loader.py       ← Loads CSV, identifies columns
│   ├── tools.py             ← Tool definitions + implementations (Milestone 4)
│   ├── agent.py             ← Agent loop - talks to OpenAI, executes tools (Milestone 4)
│   ├── prompts.py           ← Prompt templates from Milestone 3
│   └── evaluation.py        ← Evaluation functions from Milestone 3
│
├── data/
│   └── sample_ecommerce_kpi_data.csv
│
├── tests/
│   └── test_tool_calling.py ← Tool selection accuracy tests (Milestone 4)
│
└── notebooks/               ← Your original Colab notebooks (for reference)
```

## Setup

1. Open this folder in VS Code
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Mac) or `venv\Scripts\activate` (Windows)
4. Install packages: `pip install -r requirements.txt`
5. Create `.env` file with: `OPENAI_API_KEY=sk-your-key-here`
6. Run: `python main.py`
```
## Streamlit Cloud Deployment

Entrypoint: `app.py`

Required secret:
```toml
OPENAI_API_KEY = "sk-your-key-here"
