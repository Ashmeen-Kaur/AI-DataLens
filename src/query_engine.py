"""
query_engine.py
---------------
Ties together the full "Ask Your Data" pipeline:

    question -> ai_client.understand_query -> analyzer.run_analysis
             -> ai_client.generate_explanation -> visualizer.build_chart

This is the only module app.py needs to call for a natural-language
question; it hides the multi-step orchestration described in the
project's design principle (LLM interprets, pandas calculates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from src import ai_client
from src.analyzer import run_analysis
from src.utils import DataLensError
from src.visualizer import build_chart


@dataclass
class QueryAnswer:
    """Everything the UI needs to render one natural-language answer."""

    answer_text: str
    result_df: Optional[pd.DataFrame]
    chart: Optional[go.Figure]
    query_spec: dict


def answer_question(df: pd.DataFrame, question: str) -> QueryAnswer:
    """Run the full pipeline for one user question and return a
    ready-to-display answer.

    Args:
        df: The uploaded dataset.
        question: The user's natural-language question.

    Raises:
        DataLensError: Propagated from any stage (missing API key,
            unsupported question, invalid column, etc) so app.py can
            show one consistent friendly error message.
    """
    if not question or not question.strip():
        raise DataLensError("Please type a question about your data first.")

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    spec = ai_client.understand_query(question, list(df.columns), dtypes)

    result = run_analysis(df, spec)

    explanation = ai_client.generate_explanation(question, result.summary)

    chart = build_chart(result) if spec.get("visualization", "none") != "none" or result.chart_hint != "none" else None

    return QueryAnswer(
        answer_text=explanation,
        result_df=result.result_df,
        chart=chart,
        query_spec=spec,
    )
