"""
app.py
------
DataLens AI - Natural Language Data Analyst

A Streamlit app that lets a user upload a CSV, see an automatic profile
of it, and ask plain-English questions that get answered using a
controlled pandas analysis pipeline (see src/query_engine.py).
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.data_loader import load_csv
from src.profiler import profile_dataset
from src.query_engine import answer_question
from src.utils import DataLensError
from src.visualizer import auto_chart_for_column

load_dotenv()

st.set_page_config(
    page_title="DataLens AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_QUESTIONS = [
    "What is the average salary?",
    "Which department has the highest average salary?",
    "Show the top 10 employees by salary.",
    "How many employees are in each department?",
    "What is the correlation between experience and salary?",
    "Are there any missing values?",
]


def init_session_state() -> None:
    if "df" not in st.session_state:
        st.session_state.df = None
    if "filename" not in st.session_state:
        st.session_state.filename = None
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (question, QueryAnswer)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📊 DataLens AI")
        st.caption("Your AI-powered data analyst")

        st.markdown("### Upload your CSV dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                df = load_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.filename = uploaded_file.name
                st.success(f"Loaded **{uploaded_file.name}**")
            except DataLensError as exc:
                st.error(str(exc))

        st.markdown("---")

        if st.session_state.get("df") is not None:
            df = st.session_state.df
            st.markdown("### Dataset Information")
            st.write(f"**File:** {st.session_state.filename}")
            st.write(f"**Rows:** {df.shape[0]:,}")
            st.write(f"**Columns:** {df.shape[1]}")
            with st.expander("Column names"):
                st.write(", ".join(map(str, df.columns)))

        st.markdown("---")
        st.markdown("### Example Questions")
        for q in EXAMPLE_QUESTIONS:
            st.caption(f"• {q}")

        st.markdown("---")
        st.markdown("### Settings")
        api_key_present = bool(os.environ.get("GEMINI_API_KEY", "").strip())
        if api_key_present:
            st.success("Gemini API key detected")
        else:
            st.warning(
                "No Gemini API key found. Upload and preview still work, "
                "but 'Ask Your Data' requires a key in your .env file."
            )


def render_hero() -> None:
    st.markdown("# 📊 DataLens AI")
    st.markdown("##### Ask questions. Discover insights. Understand your data.")
    st.markdown("*Your AI-powered data analyst*")
    st.divider()


def render_dataset_overview(df: pd.DataFrame) -> None:
    st.markdown("## Dataset Overview")

    profile = profile_dataset(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{profile.n_rows:,}")
    col2.metric("Columns", profile.n_columns)
    col3.metric("Missing Values", f"{profile.missing_values_total:,}")
    col4.metric("Duplicate Rows", f"{profile.duplicate_rows:,}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Numerical Columns**")
        st.write(", ".join(profile.numerical_columns) if profile.numerical_columns else "None")
    with col_b:
        st.markdown("**Categorical Columns**")
        st.write(", ".join(profile.categorical_columns) if profile.categorical_columns else "None")

    tab1, tab2, tab3 = st.tabs(["Data Preview", "Column Information", "Descriptive Statistics"])
    with tab1:
        st.dataframe(df.head(20), use_container_width=True)
    with tab2:
        st.dataframe(profile.column_info, use_container_width=True)
    with tab3:
        st.dataframe(profile.describe, use_container_width=True)

    with st.expander("Quick column chart"):
        chosen_col = st.selectbox("Choose a column to visualize", options=list(df.columns))
        if chosen_col:
            fig = auto_chart_for_column(df, chosen_col)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)


def render_ask_your_data(df: pd.DataFrame) -> None:
    st.markdown("## Ask Your Data")

    question = st.text_input(
        "Ask a question about your dataset...",
        placeholder="e.g. Which department has the highest average salary?",
        label_visibility="collapsed",
    )
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and question:
        with st.spinner("Analyzing your question..."):
            try:
                answer = answer_question(df, question)
                st.session_state.history.insert(0, (question, answer))
            except DataLensError as exc:
                st.error(str(exc))
            except Exception:  # noqa: BLE001 - never show a raw traceback
                st.error(
                    "Something went wrong while analyzing your question. "
                    "Please try rephrasing it or check your dataset."
                )

    if st.session_state.history:
        st.markdown("### Analysis Result")
        latest_question, latest_answer = st.session_state.history[0]
        _render_answer(latest_question, latest_answer)

        if len(st.session_state.history) > 1:
            with st.expander(f"Previous questions ({len(st.session_state.history) - 1})"):
                for past_q, past_a in st.session_state.history[1:]:
                    st.markdown(f"**Q: {past_q}**")
                    _render_answer(past_q, past_a, show_headers=False)
                    st.markdown("---")


def _render_answer(question: str, answer, show_headers: bool = True) -> None:
    if show_headers:
        st.markdown("#### Answer")
    st.write(answer.answer_text)

    if answer.result_df is not None and not getattr(answer.result_df, "empty", False):
        if show_headers:
            st.markdown("#### Data")
        st.dataframe(answer.result_df, use_container_width=True)

    if answer.chart is not None:
        if show_headers:
            st.markdown("#### Visualization")
        st.plotly_chart(answer.chart, use_container_width=True)


def render_empty_state() -> None:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
    st.markdown("Don't have a dataset handy? Try one of the sample files in `sample_data/`:")
    st.markdown(
        "- **students.csv** - student performance and placement data\n"
        "- **sales.csv** - order-level sales transactions\n"
        "- **employees.csv** - employee salary and attrition data"
    )


def main() -> None:
    init_session_state()
    render_sidebar()
    render_hero()

    df = st.session_state.get("df")
    if df is None:
        render_empty_state()
        return

    render_dataset_overview(df)
    st.divider()
    render_ask_your_data(df)


if __name__ == "__main__":
    main()
