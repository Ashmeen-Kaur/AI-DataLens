"""
profiler.py
-----------
Automatic dataset profiling: shape, dtypes, missing values, duplicates,
numerical/categorical column detection, and descriptive statistics.

This module does pure pandas analysis with no dependency on Streamlit or
the AI client, so it's easy to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass
class DatasetProfile:
    """Container for the profiling results of a dataframe."""

    n_rows: int
    n_columns: int
    missing_values_total: int
    duplicate_rows: int
    numerical_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    column_info: pd.DataFrame = field(default_factory=pd.DataFrame)
    describe: pd.DataFrame = field(default_factory=pd.DataFrame)


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Compute a full profile of the given dataframe.

    Args:
        df: The dataset to profile.

    Returns:
        A DatasetProfile with shape, missing values, duplicates, column
        type breakdown, per-column info table, and descriptive stats.
    """
    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

    column_info = _build_column_info(df)

    # describe() on numeric columns; fall back to all columns if there are
    # no numeric ones so the user still sees something useful.
    if numerical_columns:
        describe = df[numerical_columns].describe().round(2)
    else:
        describe = df.describe(include="all").fillna("")

    return DatasetProfile(
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        missing_values_total=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        column_info=column_info,
        describe=describe,
    )


def _build_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-column summary table: name, dtype, missing, unique."""
    rows = []
    for col in df.columns:
        rows.append(
            {
                "Column": col,
                "Data Type": str(df[col].dtype),
                "Missing Values": int(df[col].isna().sum()),
                "Unique Values": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the columns that actually have missing values,
    sorted from most to least missing. Used by the 'missing values'
    natural-language query.
    """
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return missing.rename("Missing Count").reset_index().rename(columns={"index": "Column"})
