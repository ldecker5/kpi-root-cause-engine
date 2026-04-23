# Oops AI – KPI Root Cause Analysis Engine

## Overview
Oops AI is an interactive analytics system that detects anomalies in time-series KPI data and automatically investigates potential root causes. The system combines statistical analysis with large language model (LLM) reasoning to generate clear, executive-ready explanations of performance changes.

The application is designed for business analytics workflows where teams need to quickly understand why a KPI moved and what factors may be driving the change.

Typical use cases include:
- revenue drops  
- conversion rate shifts  
- marketing performance changes  
- operational KPI monitoring  

Users upload a KPI dataset, configure metrics and segments, and the system performs automated anomaly analysis and root-cause investigation.

## Key Features
### Time-series KPI anomaly detection
Automatically identifies significant KPI shifts relative to historical performance.

### Segment-level root cause analysis
Investigates KPI changes across dimensions such as region, product category, device type, or channel.

### AI-generated explanations
Uses Retrieval-Augmented Generation (RAG) and LLM reasoning to produce clear business explanations.

### Interactive analysis workflow
Step-by-step interface for dataset upload, configuration, validation, and analysis.

### Executive-ready outputs
Visualizations, summary statistics, and narrative insights suitable for decision makers.


## System Architecture
The system integrates statistical analysis, knowledge retrieval, and LLM reasoning.
```
User Interface (Streamlit)
        ↓
Dataset Validation & Configuration
        ↓
KPI Analysis Engine
        ↓
Segment Root Cause Detection
        ↓
RAG Knowledge Retrieval
        ↓
LLM Explanation Generation
```

## Project Structure

```
kpi_root_cause_engine/

├── app.py                   ← Streamlit application (UI and workflow)
├── main.py                  ← CLI entrypoint for running the agent pipeline
├── requirements.txt         ← Python dependencies
├── .env                     ← OpenAI API key (created locally)

├── src/
│   ├── data_loader.py       ← Dataset ingestion and schema detection
│   ├── tools.py             ← Analytical tools exposed to the LLM
│   ├── agent.py             ← LLM agent orchestration and tool execution
│   ├── prompts.py           ← Prompt templates and reasoning structure
│   └── evaluation.py        ← Evaluation utilities and testing metrics

├── data/
│   └── sample_ecommerce_kpi_data.csv

├── tests/
│   └── test_tool_calling.py ← Tests for tool selection and execution

└── notebooks/
    └── development notebooks used during earlier milestones
```

---

# AI Techniques Used

## Prompt Engineering
Structured prompts guide the model to generate KPI explanations and executive summaries.

## Retrieval-Augmented Generation (RAG)
Domain knowledge documents are embedded and retrieved to ground explanations in KPI analysis patterns.

## Tool Calling
The LLM agent can call analytical tools to compute KPI statistics and segment comparisons.

## Agent Workflow

The system follows a multi-step reasoning process:

1. Detect KPI anomaly  
2. Identify potential drivers  
3. Retrieve relevant knowledge  
4. Generate explanation and recommendations  

---

# Running the Project Locally

### 1. Clone the repository
```
git clone <repo-url>
cd kpi-root-cause-engine
```

### 2. Create a virtual environment
```python -m venv venv```

### 3. Activate the environment
Mac / Linux
```source venv/bin/activate```

Windows
```venv\Scripts\activate```

### 4. Install dependencies
```pip install -r requirements.txt```


### 5. Add OpenAI API key
Create a `.env` file:
```OPENAI_API_KEY=sk-your-key-here```

### 6. Run the application
```streamlit run app.py```

---

# Streamlit Cloud Deployment

The application is deployed on **Streamlit Community Cloud**.

Entrypoint:
```app.py```

Required secret:

```toml
OPENAI_API_KEY = "sk-your-key-here"
```
# Authors
Oops AI
DSBA-6010 – LLM Systems Project

