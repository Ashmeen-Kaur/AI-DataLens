"""
Tests for src/analyzer.py.

These tests call run_analysis() directly with hand-built query specs,
bypassing the LLM entirely -- so they never require an API key and
verify that the actual pandas calculations are correct.
"""

import pandas as pd
import pytest

from src.analyzer import run_analysis
from src.utils import DataLensError


def make_employees_df():
    return pd.DataFrame({
        "department": ["Engineering", "Engineering", "HR", "Sales", "Sales", "HR"],
        "salary": [80000, 90000, 50000, 60000, 65000, 52000],
        "experience": [5, 7, 2, 3, 4, 2],
    })


def test_mean_operation():
    df = make_employees_df()
    spec = {"operation": "mean", "target_column": "salary", "group_by_column": None,
            "filter_column": None, "filter_operator": None, "filter_value": None,
            "sort_ascending": False, "n": None, "visualization": "none"}
    result = run_analysis(df, spec)
    assert result.raw_value == pytest.approx(df["salary"].mean())


def test_groupby_operation_finds_highest_group():
    df = make_employees_df()
    spec = {"operation": "groupby", "target_column": "salary", "group_by_column": "department",
            "filter_column": None, "filter_operator": None, "filter_value": None,
            "sort_ascending": False, "n": None, "visualization": "bar"}
    result = run_analysis(df, spec)
    assert result.result_df.iloc[0]["department"] == "Engineering"


def test_top_n_operation():
    df = make_employees_df()
    spec = {"operation": "top_n", "target_column": "salary", "group_by_column": None,
            "filter_column": None, "filter_operator": None, "filter_value": None,
            "sort_ascending": False, "n": 2, "visualization": "none"}
    result = run_analysis(df, spec)
    assert len(result.result_df) == 2
    assert result.result_df.iloc[0]["salary"] == 90000


def test_bottom_n_operation():
    df = make_employees_df()
    spec = {"operation": "top_n", "target_column": "salary", "group_by_column": None,
            "filter_column": None, "filter_operator": None, "filter_value": None,
            "sort_ascending": True, "n": 1, "visualization": "none"}
    result = run_analysis(df, spec)
    assert result.result_df.iloc[0]["salary"] == 50000


def test_filter_operation():
    df = make_employees_df()
    spec = {"operation": "filter", "target_column": None, "group_by_column": None,
            "filter_column": "salary", "filter_operator": ">", "filter_value": 60000,
            "sort_ascending": False, "n": None, "visualization": "none"}
    result = run_analysis(df, spec)
    assert len(result.result_df) == 3  # 80000, 90000, 65000


def test_frequency_operation():
    df = make_employees_df()
    spec = {"operation": "frequency", "target_column": None, "group_by_column": "department",
            "filter_column": None, "filter_operator": None, "filter_value": None,
            "sort_ascending": False, "n": None, "visualization": "bar"}
    result = run_analysis(df, spec)
    total = result.result_df["Count"].sum()
    assert total == len(df)


def test_correlation_operation():
    df = make_employees_df()
    spec = {"operation": "correlation", "target_column": "salary", "group_by_column": "experience",
            "filter_column": None, "filter_operator": None, "filter_value": None,
            "sort_ascending": False, "n": None, "visualization": "scatter"}
    result = run_analysis(df, spec)
    assert -1.0 <= result.raw_value <= 1.0


def test_missing_column_raises_error():
    df = make_employees_df()
    spec = {"operation": "mean", "target_column": "does_not_exist_at_all",
            "group_by_column": None, "filter_column": None, "filter_operator": None,
            "filter_value": None, "sort_ascending": False, "n": None, "visualization": "none"}
    with pytest.raises(DataLensError):
        run_analysis(df, spec)


def test_non_numeric_column_for_mean_raises_error():
    df = make_employees_df()
    spec = {"operation": "mean", "target_column": "department", "group_by_column": None,
            "filter_column": None, "filter_operator": None, "filter_value": None,
            "sort_ascending": False, "n": None, "visualization": "none"}
    with pytest.raises(DataLensError):
        run_analysis(df, spec)


def test_unknown_operation_raises_error():
    df = make_employees_df()
    spec = {"operation": "unknown", "target_column": None, "group_by_column": None,
            "filter_column": None, "filter_operator": None, "filter_value": None,
            "sort_ascending": False, "n": None, "visualization": "none"}
    with pytest.raises(DataLensError):
        run_analysis(df, spec)
