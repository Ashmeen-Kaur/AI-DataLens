"""Tests for src/data_loader.py. No API key required."""

import io

import pandas as pd
import pytest

from src.data_loader import load_csv
from src.utils import DataLensError


def test_load_valid_csv():
    csv_content = "a,b,c\n1,2,3\n4,5,6\n"
    df = load_csv(io.StringIO(csv_content))
    assert df.shape == (2, 3)
    assert list(df.columns) == ["a", "b", "c"]


def test_load_csv_from_path(tmp_path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text("x,y\n1,2\n3,4\n")
    df = load_csv(str(file_path))
    assert df.shape == (2, 2)


def test_empty_csv_raises_error():
    with pytest.raises(DataLensError):
        load_csv(io.StringIO(""))


def test_none_file_raises_error():
    with pytest.raises(DataLensError):
        load_csv(None)


def test_csv_with_only_headers_raises_error():
    with pytest.raises(DataLensError):
        load_csv(io.StringIO("col1,col2\n"))


def test_drops_fully_empty_unnamed_columns():
    csv_content = "a,b,\n1,2,\n3,4,\n"
    df = load_csv(io.StringIO(csv_content))
    assert "a" in df.columns and "b" in df.columns
    assert not any(str(c).startswith("Unnamed") for c in df.columns)
