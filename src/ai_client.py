"""
ai_client.py
------------
Thin wrapper around the Google Gemini API (current official SDK:
``google-genai``, imported as ``from google import genai``).

This module has exactly two AI responsibilities, on purpose:

1. ``understand_query``  -> turn a natural-language question into a
   small structured JSON "query spec" (which column, which operation,
   whether to chart it, etc). The LLM never sees or touches the actual
   data values, only the column names/types -- so it cannot hallucinate
   numbers.
2. ``generate_explanation`` -> turn an already-computed pandas result
   into a friendly sentence. By the time this is called, every number
   has already been calculated by pandas in analyzer.py.

The LLM is never asked to perform the calculation itself, and it is
never allowed to execute arbitrary code -- see analyzer.py for the
controlled, whitelisted operations it can request.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from src.utils import DataLensError

# Current recommended Gemini SDK (replaces the deprecated
# `google-generativeai` package). See: https://ai.google.dev/gemini-api/docs/sdks
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover - exercised only if package missing
    raise DataLensError(
        "The 'google-genai' package is not installed. Run "
        "'pip install -r requirements.txt' and try again."
    ) from exc

MODEL_NAME = "gemini-3.6-flash"

QUERY_SPEC_KEYS = {
    "operation",
    "target_column",
    "group_by_column",
    "filter_column",
    "filter_operator",
    "filter_value",
    "sort_ascending",
    "n",
    "visualization",
}

VALID_OPERATIONS = {
    "mean", "median", "min", "max", "sum", "count", "std",
    "groupby", "filter", "sort", "top_n", "frequency",
    "correlation", "missing_values", "describe", "unknown",
}

VALID_VISUALIZATIONS = {"bar", "line", "histogram", "scatter", "heatmap", "none"}


def _get_client() -> "genai.Client":
    """Create a Gemini API client from the GEMINI_API_KEY env variable.

    Raises:
        DataLensError: If no API key is configured.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise DataLensError(
            "No Gemini API key found. Add GEMINI_API_KEY to your .env file "
            "(see .env.example) to enable natural-language questions."
        )
    return genai.Client(api_key=api_key)


def understand_query(
    question: str,
    columns: List[str],
    dtypes: Dict[str, str],
) -> Dict[str, Any]:
    """Ask Gemini to interpret a natural-language question about the
    dataset and return a small structured "query spec".

    Only column names and their data types are sent to the model --
    never the actual row values -- which keeps the request small and
    avoids leaking dataset contents unnecessarily.

    Args:
        question: The user's natural-language question.
        columns: List of column names in the uploaded dataset.
        dtypes: Mapping of column name -> pandas dtype string.

    Returns:
        A dict describing the requested operation. See analyzer.py for
        how this spec is consumed. Unknown/unsupported questions come
        back with operation == "unknown".

    Raises:
        DataLensError: If the API key is missing, the request fails, or
            the model's response cannot be parsed as valid JSON.
    """
    client = _get_client()

    system_instruction = (
        "You are a query-understanding engine for a data analysis tool. "
        "You NEVER calculate answers yourself. Your only job is to read a "
        "user's natural-language question about a dataset and translate it "
        "into a strict JSON object describing which pandas operation to run. "
        "Only use column names from the provided list -- if you are unsure, "
        "pick the closest matching column name. "
        "Respond with ONLY a raw JSON object, no markdown fences, no prose.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "operation": one of ["mean","median","min","max","sum","count","std",\n'
        '                       "groupby","filter","sort","top_n","frequency",\n'
        '                       "correlation","missing_values","describe","unknown"],\n'
        '  "target_column": string or null (the numeric column to analyze),\n'
        '  "group_by_column": string or null (categorical column to group by),\n'
        '  "filter_column": string or null,\n'
        '  "filter_operator": one of [">","<",">=","<=","==","!="] or null,\n'
        '  "filter_value": number or string or null,\n'
        '  "sort_ascending": true or false,\n'
        '  "n": integer or null (for top/bottom N questions),\n'
        '  "visualization": one of ["bar","line","histogram","scatter","heatmap","none"]\n'
        "}\n\n"
        "Guidelines:\n"
        "- 'average/mean X' -> operation mean, target_column X\n"
        "- 'X per category' or 'highest average X by Y' -> operation groupby, "
        "target_column X, group_by_column Y\n"
        "- 'top 10 by X' / 'lowest 5 X' -> operation top_n, target_column X, n set, "
        "sort_ascending false for top/highest, true for lowest/bottom\n"
        "- 'how many rows have X > 50000' -> operation filter\n"
        "- 'number of rows per category' / 'count per department' -> operation frequency, "
        "group_by_column set\n"
        "- 'correlation between X and Y' -> operation correlation, target_column X, "
        "group_by_column Y (reused as second column)\n"
        "- 'missing values' -> operation missing_values\n"
        "- 'summarize' / 'describe the data' -> operation describe\n"
        "- Only set visualization to something other than 'none' if a chart genuinely "
        "helps answer the question.\n"
        "- If the question cannot be answered with the available columns, use "
        'operation "unknown".'
    )

    dtype_lines = ", ".join(f"{col} ({dtypes.get(col, 'unknown')})" for col in columns)
    prompt = (
        f"Available columns and types: {dtype_lines}\n\n"
        f"User question: {question}"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
    except Exception as exc:  # noqa: BLE001 - surface any API/network failure
        raise DataLensError(
            "Could not reach the Gemini API. Please check your internet "
            "connection and API key, then try again."
        ) from exc

    raw_text = (response.text or "").strip()
    spec = _parse_query_spec(raw_text)
    return spec


def _parse_query_spec(raw_text: str) -> Dict[str, Any]:
    """Parse and sanity-check the JSON returned by the model."""
    cleaned = raw_text.strip()
    # Defensive cleanup in case the model wraps the JSON in code fences anyway.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise DataLensError(
            "The AI returned a response that couldn't be understood. "
            "Please try rephrasing your question."
        ) from exc

    if not isinstance(data, dict):
        raise DataLensError("The AI response was not in the expected format.")

    # Fill in any missing keys with safe defaults so downstream code
    # never has to guess about missing dictionary keys.
    spec: Dict[str, Any] = {
        "operation": data.get("operation") if data.get("operation") in VALID_OPERATIONS else "unknown",
        "target_column": data.get("target_column"),
        "group_by_column": data.get("group_by_column"),
        "filter_column": data.get("filter_column"),
        "filter_operator": data.get("filter_operator"),
        "filter_value": data.get("filter_value"),
        "sort_ascending": bool(data.get("sort_ascending", False)),
        "n": data.get("n"),
        "visualization": data.get("visualization") if data.get("visualization") in VALID_VISUALIZATIONS else "none",
    }
    return spec


def generate_explanation(question: str, result_summary: str) -> str:
    """Ask Gemini to phrase an already-computed result as a friendly
    natural-language sentence.

    Args:
        question: The original user question.
        result_summary: A short, plain-text description of the pandas
            result (already fully computed -- the LLM cannot change the
            numbers, only the wording).

    Returns:
        A short natural-language explanation. Falls back to the raw
        summary if the API call fails, so the app still shows an answer.
    """
    try:
        client = _get_client()
    except DataLensError:
        # No API key: just return the computed summary as-is.
        return result_summary

    prompt = (
        "You are explaining a data analysis result to a user in one or two "
        "short sentences. Do NOT invent, change, or round any numbers beyond "
        "what is given below -- use them exactly as provided.\n\n"
        f"User's question: {question}\n"
        f"Computed result: {result_summary}\n\n"
        "Write a friendly, concise natural-language answer."
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.3),
        )
        text = (response.text or "").strip()
        return text if text else result_summary
    except Exception:  # noqa: BLE001 - never let explanation failures break the app
        return result_summary
