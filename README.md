# 📊 DataLens AI

DataLens AI is a Streamlit web app that lets you upload any CSV dataset and ask questions about it in plain English. Gemini interprets your question, but every number in the answer is calculated by pandas — not hallucinated by the LLM.

## Overview

Upload a CSV → get an instant automated profile (shape, missing values, duplicates, column types, descriptive stats) → then ask natural-language questions like *"Which department has the highest average salary?"* and get back a written answer, a data table, and (when appropriate) a chart.

## Features

- CSV upload with validation (encoding issues, empty files, oversized files all handled gracefully)
- Automatic dataset profiling: rows, columns, missing values, duplicates, numeric vs. categorical columns, `describe()` stats
- Natural-language question box ("Ask Your Data")
- AI query understanding: Gemini converts your question into a structured operation (mean, groupby, top-N, filter, correlation, frequency, missing values, describe)
- All actual calculations run in pandas — the LLM never invents numbers and never executes arbitrary code
- Automatic chart selection with Plotly (bar, line, histogram, scatter, heatmap)
- Friendly error handling everywhere (no raw tracebacks shown to users)
- Sample datasets included so you can try it immediately

## Tech Stack

| Layer | Choice |
|---|---|
| UI | Streamlit |
| Data analysis | Pandas, NumPy |
| Visualization | Plotly |
| AI | Google Gemini (`google-genai` SDK, model `gemini-2.5-flash`) |
| Config | python-dotenv |
| Testing | pytest |

## Architecture

```
User
  ↓
Streamlit UI (app.py)
  ↓
CSV Upload (src/data_loader.py)
  ↓
Dataset Profiler (src/profiler.py)
  ↓
Natural Language Query (src/query_engine.py)
  ↓
AI Query Understanding (src/ai_client.py → Gemini)
  ↓
Controlled Analysis Engine (src/analyzer.py)
  ↓
Pandas (actual calculation)
  ↓
Visualization (src/visualizer.py → Plotly)
  ↓
AI Explanation (src/ai_client.py → Gemini, wording only)
  ↓
Result shown to user
```

The key design decision: **Gemini decides *what* to calculate, pandas decides *the answer*.** The LLM returns a small JSON spec (operation, target column, group-by column, filter, etc.), which `analyzer.py` executes using a fixed, whitelisted set of pandas operations. This means the LLM can never invent a number and can never run arbitrary code.

MIT
