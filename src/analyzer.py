"""
analyzer.py
-----------
The controlled analysis engine.

This is the most important module in the project from a "reliability"
and "security" point of view. It takes the small JSON query spec
produced by the LLM (see ai_client.py) and executes ONE of a fixed,
whitelisted set of pandas operations against the user's dataframe.

The LLM never runs code and never sees raw data values -- it only
picks which of these named operations to run and on which columns.
This module is the only place that actually touches numbers, so all
numerical answers come directly from pandas, not from the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from src.utils import DataLensError, find_best_column_match, format_number

MAX_TOP_N = 100
DEFAULT_TOP_N = 10


@dataclass
class AnalysisResult:
    """Everything the UI needs to render one answer."""

    summary: str                     # plain-text summary for the LLM/UI
    result_df: Optional[pd.DataFrame] = None
    chart_hint: str = "none"         # bar / line / histogram / scatter / heatmap / none
    chart_columns: Optional[Dict[str, str]] = None  # e.g. {"x": "department", "y": "salary"}
    raw_value: Any = None


def run_analysis(df: pd.DataFrame, spec: Dict[str, Any]) -> AnalysisResult:
    """Execute the operation described by ``spec`` against ``df``.

    Args:
        df: The user's uploaded dataset.
        spec: A validated query spec dict (see ai_client.understand_query).

    Returns:
        An AnalysisResult with a summary, optional table, and chart hint.

    Raises:
        DataLensError: If the operation is unsupported or references
            columns that don't exist in the dataset.
    """
    operation = spec.get("operation", "unknown")
    columns = list(df.columns)

    target = find_best_column_match(spec.get("target_column"), columns)
    group_by = find_best_column_match(spec.get("group_by_column"), columns)

    handlers = {
        "mean": _basic_stat,
        "median": _basic_stat,
        "min": _basic_stat,
        "max": _basic_stat,
        "sum": _basic_stat,
        "count": _basic_stat,
        "std": _basic_stat,
        "groupby": _groupby,
        "filter": _filter,
        "sort": _top_n,
        "top_n": _top_n,
        "frequency": _frequency,
        "correlation": _correlation,
        "missing_values": _missing_values,
        "describe": _describe,
    }

    if operation not in handlers:
        raise DataLensError(
            "I couldn't understand that as a data question. Try asking "
            "about an average, a comparison between groups, a top-N list, "
            "or a count -- for example: 'What is the average salary?'"
        )

    return handlers[operation](df=df, spec=spec, target=target, group_by=group_by, operation=operation)


def _require_numeric(df: pd.DataFrame, column: Optional[str]) -> str:
    if column is None:
        raise DataLensError(
            "I couldn't figure out which column you're asking about. "
            "Try mentioning the exact column name."
        )
    if column not in df.columns:
        raise DataLensError(f"Column '{column}' was not found in the dataset.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise DataLensError(f"'{column}' is not a numeric column, so this calculation isn't possible.")
    return column


def _basic_stat(df, spec, target, group_by, operation, **_) -> AnalysisResult:
    """mean / median / min / max / sum / count / std on a single column."""
    if operation == "count":
        # 'count' can apply to any column (counts non-null rows), numeric or not.
        column = target or (df.columns[0] if len(df.columns) else None)
        if column is None or column not in df.columns:
            raise DataLensError("I couldn't figure out which column to count.")
        value = int(df[column].count())
        return AnalysisResult(
            summary=f"The count of non-missing values in '{column}' is {format_number(value)}.",
            raw_value=value,
        )

    column = _require_numeric(df, target)
    stat_funcs = {
        "mean": "mean", "median": "median", "min": "min",
        "max": "max", "sum": "sum", "std": "std",
    }
    value = getattr(df[column], stat_funcs[operation])()
    label = {"mean": "average", "median": "median", "min": "minimum",
              "max": "maximum", "sum": "sum", "std": "standard deviation"}[operation]
    return AnalysisResult(
        summary=f"The {label} of '{column}' is {format_number(value)}.",
        raw_value=float(value),
    )


def _groupby(df, spec, target, group_by, operation, **_) -> AnalysisResult:
    """Aggregate a numeric column by a categorical group (e.g. average
    salary per department), sorted descending so the top group is obvious.
    """
    column = _require_numeric(df, target)
    if group_by is None:
        raise DataLensError(
            "I couldn't figure out which column to group by. Try mentioning "
            "the category (e.g. 'by department')."
        )
    if group_by not in df.columns:
        raise DataLensError(f"Column '{group_by}' was not found in the dataset.")

    grouped = (
        df.groupby(group_by)[column]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={column: f"Average {column}"})
    )

    if grouped.empty:
        raise DataLensError("No data was available to group.")

    top_row = grouped.iloc[0]
    summary = (
        f"'{top_row[group_by]}' has the highest average {column} at "
        f"{format_number(top_row[f'Average {column}'])}."
    )
    return AnalysisResult(
        summary=summary,
        result_df=grouped,
        chart_hint="bar",
        chart_columns={"x": group_by, "y": f"Average {column}"},
        raw_value=grouped,
    )


def _frequency(df, spec, target, group_by, operation, **_) -> AnalysisResult:
    """Count of rows per category (e.g. number of students per department)."""
    column = group_by or target
    if column is None or column not in df.columns:
        raise DataLensError(
            "I couldn't figure out which column to count by category."
        )

    counts = (
        df[column]
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="Count")
    )
    summary = f"Here is the number of rows for each unique value of '{column}'."
    return AnalysisResult(
        summary=summary,
        result_df=counts,
        chart_hint="bar",
        chart_columns={"x": column, "y": "Count"},
        raw_value=counts,
    )


def _top_n(df, spec, target, group_by, operation, **_) -> AnalysisResult:
    """Top/bottom N rows sorted by a numeric column."""
    column = _require_numeric(df, target)
    n = spec.get("n") or DEFAULT_TOP_N
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = DEFAULT_TOP_N
    n = max(1, min(n, MAX_TOP_N))

    ascending = bool(spec.get("sort_ascending", False))
    result_df = df.sort_values(by=column, ascending=ascending).head(n).reset_index(drop=True)

    direction = "lowest" if ascending else "top"
    summary = f"Here are the {direction} {n} rows sorted by '{column}'."
    return AnalysisResult(summary=summary, result_df=result_df, raw_value=result_df)


def _filter(df, spec, target, group_by, operation, **_) -> AnalysisResult:
    """Filter rows by a simple 'column OP value' condition."""
    filter_column = find_best_column_match(spec.get("filter_column"), list(df.columns))
    op = spec.get("filter_operator")
    value = spec.get("filter_value")

    if filter_column is None or filter_column not in df.columns:
        raise DataLensError(
            "I couldn't figure out which column to filter on. Try being "
            "more specific, e.g. 'salary above 50000'."
        )
    if op not in {">", "<", ">=", "<=", "==", "!="}:
        raise DataLensError("I couldn't understand the filter condition.")

    series = df[filter_column]
    # Coerce the filter value to a number if the column is numeric.
    if pd.api.types.is_numeric_dtype(series):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise DataLensError(f"'{value}' is not a valid number to compare against '{filter_column}'.")

    ops = {
        ">": series > value, "<": series < value,
        ">=": series >= value, "<=": series <= value,
        "==": series == value, "!=": series != value,
    }
    mask = ops[op]
    result_df = df[mask].reset_index(drop=True)

    summary = f"Found {len(result_df)} rows where '{filter_column}' {op} {value}."
    return AnalysisResult(summary=summary, result_df=result_df, raw_value=len(result_df))


def _correlation(df, spec, target, group_by, operation, **_) -> AnalysisResult:
    """Correlation between two numeric columns, or a full correlation
    heatmap if only one (or no) column was specified.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        raise DataLensError("There aren't at least two numeric columns to correlate.")

    col_a = target
    col_b = group_by  # second column is reused from the group_by slot

    if col_a and col_b and col_a in numeric_df.columns and col_b in numeric_df.columns:
        value = numeric_df[col_a].corr(numeric_df[col_b])
        summary = f"The correlation between '{col_a}' and '{col_b}' is {value:.2f}."
        return AnalysisResult(
            summary=summary,
            raw_value=float(value),
            chart_hint="scatter",
            chart_columns={"x": col_a, "y": col_b},
        )

    corr_matrix = numeric_df.corr().round(2)
    summary = "Here is the correlation matrix across all numeric columns."
    return AnalysisResult(
        summary=summary,
        result_df=corr_matrix.reset_index().rename(columns={"index": ""}),
        chart_hint="heatmap",
        raw_value=corr_matrix,
    )


def _missing_values(df, spec, target, group_by, operation, **_) -> AnalysisResult:
    from src.profiler import missing_values_report

    report = missing_values_report(df)
    if report.empty:
        summary = "Good news -- there are no missing values in this dataset."
    else:
        summary = f"{len(report)} column(s) have missing values."
    return AnalysisResult(summary=summary, result_df=report if not report.empty else None, raw_value=report)


def _describe(df, spec, target, group_by, operation, **_) -> AnalysisResult:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        described = df.describe(include="all").fillna("")
    else:
        described = numeric_df.describe().round(2)
    summary = "Here is a statistical summary of the dataset."
    return AnalysisResult(summary=summary, result_df=described.reset_index(), raw_value=described)
