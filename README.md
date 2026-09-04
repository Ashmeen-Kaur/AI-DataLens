# 📊 DataLens AI

**Ask questions. Discover insights. Understand your data.**

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

## Demo

Live demo: _[add your deployed Streamlit Community Cloud URL here]_

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

## Project Structure

```
DataLens-AI/
├── app.py                   # Streamlit UI
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # CSV loading & validation
│   ├── profiler.py          # Automatic dataset profiling
│   ├── query_engine.py      # Orchestrates the full question pipeline
│   ├── analyzer.py          # Controlled pandas operations (the "safe execution" layer)
│   ├── visualizer.py        # Plotly chart selection
│   ├── ai_client.py         # Gemini API wrapper
│   └── utils.py             # Shared helpers (fuzzy column matching, formatting)
├── sample_data/
│   ├── students.csv
│   ├── sales.csv
│   └── employees.csv
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_analyzer.py
    └── test_profiler.py
```

## Installation

### Windows Setup

```bash
git clone YOUR_REPOSITORY_URL
cd DataLens-AI
python -m venv .venv
```

Activate the virtual environment (Windows PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

If you get a script-execution error in PowerShell, either run this once:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

...or use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up your environment file:

```bash
copy .env.example .env
```

Open `.env` in VS Code and paste your Gemini API key after `GEMINI_API_KEY=` (see the **API Key** section below for how to get one).

## Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey).
2. Sign in with a Google account.
3. Click **Create API Key**.
4. Copy the key into your `.env` file:
   ```
   GEMINI_API_KEY=paste_your_key_here
   ```
5. Never commit `.env` to Git — it's already listed in `.gitignore`.

## Running Locally

```bash
streamlit run app.py
```

Streamlit will print a local URL, typically:

```
http://localhost:8501
```

Open that URL in your browser. The app works immediately for upload and profiling — you only need the API key once you use "Ask Your Data".

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again inside the activated virtual environment. |
| "No Gemini API key found" | Make sure `.env` exists (not just `.env.example`) and contains a real key, then restart `streamlit run app.py`. |
| PowerShell won't activate venv | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or use Command Prompt's `activate.bat`. |
| Blank/broken chart | Some questions don't need a chart — this is expected. If a chart should appear but doesn't, try rephrasing the question with an explicit column name. |
| "Could not reach the Gemini API" | Check your internet connection and that the API key hasn't expired or hit a quota limit. |
| Upload fails with encoding error | Re-save the CSV as UTF-8 from Excel/Google Sheets and re-upload. |

## GitHub Setup

```bash
git init
git add .
git commit -m "Initial commit: DataLens AI"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

`.env` is excluded via `.gitignore` and will never be committed — only `.env.example` (with a placeholder) is tracked.

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (see above).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** → **From existing repo**.
4. Select your `DataLens-AI` repository and branch.
5. Set the main file path to `app.py`.
6. Before deploying, open **Advanced settings → Secrets** and add:
   ```
   GEMINI_API_KEY = "your_real_key_here"
   ```
7. Click **Deploy**.
8. Once live, open the app URL, upload a sample CSV, and ask a question to confirm the Gemini key works in the deployed environment.

(The deployment process on Streamlit Community Cloud may be updated over time — check the **Secrets** section of your app's settings if the location has moved.)

## Running Tests

```bash
pytest tests/ -v
```

Tests cover CSV loading, profiling, and the analyzer's pandas operations — none of them require a Gemini API key.

## Limitations

- Query understanding depends on Gemini correctly mapping a question to one of a fixed set of operations; very ambiguous or multi-part questions may not parse correctly.
- Only single-column filters are supported (no compound `AND`/`OR` conditions yet).
- Designed for datasets that fit comfortably in memory (tested up to tens of thousands of rows).

## License

MIT
