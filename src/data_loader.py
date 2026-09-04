"""
data_loader.py
--------------
Handles loading and validating a user-uploaded CSV file into a pandas
DataFrame. Keeping this isolated makes it easy to unit test without
needing Streamlit or an API key.
"""

from __future__ import annotations

import io
from typing import Union

import pandas as pd

from src.utils import DataLensError

# Reject files larger than this to keep the app responsive on free hosting.
MAX_FILE_SIZE_MB = 50


def load_csv(file: Union[str, io.BytesIO, io.StringIO]) -> pd.DataFrame:
    """Load a CSV file (path or file-like object) into a DataFrame.

    Tries a couple of common encodings before giving up, and raises a
    friendly ``DataLensError`` for any problem so the UI layer never has
    to deal with raw pandas/Python exceptions.

    Args:
        file: A file path, or a file-like object (e.g. from
            ``st.file_uploader``).

    Returns:
        A validated, non-empty pandas DataFrame.

    Raises:
        DataLensError: If the file is missing, too large, empty, has no
            columns, or cannot be parsed as CSV with any supported encoding.
    """
    if file is None:
        raise DataLensError("No file was provided. Please upload a CSV file.")

    # If it's an uploaded file object, check its size first.
    size_bytes = getattr(file, "size", None)
    if size_bytes is not None and size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise DataLensError(
            f"This file is too large ({size_bytes / (1024 * 1024):.1f} MB). "
            f"Please upload a CSV smaller than {MAX_FILE_SIZE_MB} MB."
        )

    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    last_error: Exception | None = None

    for encoding in encodings_to_try:
        try:
            # Reset stream position in case a previous attempt consumed it.
            if hasattr(file, "seek"):
                file.seek(0)
            df = pd.read_csv(file, encoding=encoding)
            return _validate_dataframe(df)
        except DataLensError:
            # Validation errors are not encoding problems; re-raise directly.
            raise
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise DataLensError(
                "The uploaded file is empty or has no readable data. "
                "Please upload a valid CSV file."
            ) from exc
        except pd.errors.ParserError as exc:
            raise DataLensError(
                "This file doesn't look like a valid CSV. Please check the "
                "formatting and try again."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - convert anything else too
            last_error = exc
            continue

    raise DataLensError(
        "Could not read this file with any supported text encoding "
        "(tried UTF-8, Latin-1, and CP1252). Please re-save the CSV in a "
        "standard encoding and try again."
    ) from last_error


def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run basic sanity checks on a freshly-loaded dataframe."""
    if df is None or df.empty:
        raise DataLensError(
            "The uploaded CSV has no rows. Please upload a dataset that "
            "contains data."
        )

    if len(df.columns) == 0:
        raise DataLensError("The uploaded CSV has no columns.")

    # Drop fully-empty "Unnamed" columns pandas sometimes creates from
    # trailing commas in the source file.
    unnamed_all_null = [
        col for col in df.columns
        if str(col).startswith("Unnamed") and df[col].isna().all()
    ]
    if unnamed_all_null:
        df = df.drop(columns=unnamed_all_null)

    return df
