"""Tests for src/profiler.py. No API key required."""

import numpy as np
import pandas as pd

from src.profiler import missing_values_report, profile_dataset


def make_sample_df():
    return pd.DataFrame({
        "department": ["Eng", "Eng", "HR", "Sales", "HR"],
        "salary": [80000, 82000, 55000, np.nan, 58000],
        "experience": [5, 6, 2, 3, 4],
    })


def test_profile_dataset_shape():
    df = make_sample_df()
    profile = profile_dataset(df)
    assert profile.n_rows == 5
    assert profile.n_columns == 3


def test_profile_dataset_missing_values():
    df = make_sample_df()
    profile = profile_dataset(df)
    assert profile.missing_values_total == 1


def test_profile_dataset_duplicate_detection():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    profile = profile_dataset(df)
    assert profile.duplicate_rows == 1


def test_profile_numerical_and_categorical_columns():
    df = make_sample_df()
    profile = profile_dataset(df)
    assert "salary" in profile.numerical_columns
    assert "experience" in profile.numerical_columns
    assert "department" in profile.categorical_columns


def test_missing_values_report():
    df = make_sample_df()
    report = missing_values_report(df)
    assert "salary" in report["Column"].values
    assert report[report["Column"] == "salary"]["Missing Count"].iloc[0] == 1


def test_missing_values_report_empty_when_no_missing():
    df = pd.DataFrame({"a": [1, 2, 3]})
    report = missing_values_report(df)
    assert report.empty
