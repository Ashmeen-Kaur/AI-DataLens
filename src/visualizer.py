"""
visualizer.py
-------------
Builds a Plotly figure from an AnalysisResult, choosing the chart type
that the analyzer already decided on (chart_hint). This module never
decides *whether* to chart something on its own beyond a couple of
sensible fallbacks -- that decision was made upstream based on the
question and the operation performed.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analyzer import AnalysisResult


def build_chart(result: AnalysisResult) -> Optional[go.Figure]:
    """Return a Plotly figure for the given analysis result, or None if
    no chart is appropriate.
    """
    hint = result.chart_hint
    df = result.result_df
    cols = result.chart_columns or {}

    if hint == "none" or df is None:
        return None

    try:
        if hint == "bar":
            return _bar_chart(df, cols)
        if hint == "line":
            return _line_chart(df, cols)
        if hint == "histogram":
            return _histogram(df, cols)
        if hint == "scatter":
            return _scatter_chart(df, cols)
        if hint == "heatmap":
            return _heatmap(result.raw_value if isinstance(result.raw_value, pd.DataFrame) else df)
    except Exception:  # noqa: BLE001 - never let a chart error break the answer
        return None

    return None


def _bar_chart(df: pd.DataFrame, cols: dict) -> go.Figure:
    x = cols.get("x") or df.columns[0]
    y = cols.get("y") or df.columns[1]
    fig = px.bar(df, x=x, y=y, title=f"{y} by {x}", color=x)
    fig.update_layout(showlegend=False)
    return fig


def _line_chart(df: pd.DataFrame, cols: dict) -> go.Figure:
    x = cols.get("x") or df.columns[0]
    y = cols.get("y") or df.columns[1]
    return px.line(df, x=x, y=y, title=f"{y} over {x}")


def _histogram(df: pd.DataFrame, cols: dict) -> go.Figure:
    x = cols.get("x") or df.select_dtypes(include="number").columns[0]
    return px.histogram(df, x=x, title=f"Distribution of {x}")


def _scatter_chart(df: pd.DataFrame, cols: dict) -> go.Figure:
    x = cols.get("x") or df.columns[0]
    y = cols.get("y") or df.columns[1]
    return px.scatter(df, x=x, y=y, title=f"{y} vs {x}")


def _heatmap(matrix: pd.DataFrame) -> go.Figure:
    if not isinstance(matrix, pd.DataFrame):
        return None
    numeric = matrix.select_dtypes(include="number") if not matrix.empty else matrix
    fig = px.imshow(
        numeric,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap",
    )
    return fig


def auto_chart_for_column(df: pd.DataFrame, column: str) -> Optional[go.Figure]:
    """Standalone helper used by the 'Dataset Overview' tab to show a
    quick distribution chart for a single column the user picks, outside
    of the natural-language query flow.
    """
    if column not in df.columns:
        return None
    if pd.api.types.is_numeric_dtype(df[column]):
        return px.histogram(df, x=column, title=f"Distribution of {column}")
    counts = df[column].value_counts().head(20).reset_index()
    counts.columns = [column, "Count"]
    return px.bar(counts, x=column, y="Count", title=f"Frequency of {column}")
