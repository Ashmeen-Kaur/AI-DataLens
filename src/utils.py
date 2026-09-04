"""
utils.py
--------
Small shared helper functions used across the DataLens AI project.

Keeping these in one place avoids duplicating logic (e.g. fuzzy column
matching, number formatting) in multiple modules.
"""

from __future__ import annotations

import difflib
from typing import Optional, Sequence


class DataLensError(Exception):
    """Base exception for all DataLens AI specific errors.

    Using a custom exception lets app.py catch anything that goes wrong
    inside our own analysis pipeline and show a friendly message, while
    still letting truly unexpected errors bubble up during development.
    """


def find_best_column_match(name: Optional[str], columns: Sequence[str]) -> Optional[str]:
    """Map a possibly-imprecise column name (as guessed by the LLM) to an
    actual column in the dataframe.

    The LLM sometimes returns a column name that is close but not an exact
    match (different case, small typo, singular/plural, etc). Rather than
    trusting the LLM blindly, we validate its suggestion against the real
    columns and pick the closest match. If nothing is close enough, we
    return None so the caller can handle the "column not found" case.

    Args:
        name: The column name suggested by the LLM (may be None).
        columns: The actual column names present in the dataframe.

    Returns:
        The best-matching real column name, or None if no reasonable
        match exists.
    """
    if not name:
        return None

    # Exact match (case-insensitive) is preferred.
    lowered = {c.lower(): c for c in columns}
    if name.lower() in lowered:
        return lowered[name.lower()]

    # Try normalizing spaces/underscores.
    normalized = {c.lower().replace(" ", "_"): c for c in columns}
    key = name.lower().replace(" ", "_")
    if key in normalized:
        return normalized[key]

    # Fall back to fuzzy matching.
    matches = difflib.get_close_matches(name, columns, n=1, cutoff=0.6)
    if matches:
        return matches[0]

    matches_lower = difflib.get_close_matches(name.lower(), list(lowered.keys()), n=1, cutoff=0.6)
    if matches_lower:
        return lowered[matches_lower[0]]

    return None


def format_number(value: float) -> str:
    """Format a number nicely for display in natural-language answers.

    Whole numbers are shown without decimals; everything else is rounded
    to 2 decimal places. Large numbers get comma separators.
    """
    if value is None:
        return "N/A"
    try:
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"
    except (ValueError, TypeError):
        return str(value)


def safe_truncate(df_rows: int, limit: int = 500) -> int:
    """Return a safe row limit to avoid rendering huge tables in the UI."""
    return min(df_rows, limit)
