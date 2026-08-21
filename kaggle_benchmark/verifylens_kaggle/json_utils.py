"""
verifylens_kaggle/json_utils.py
---------------------------------
Shared JSON parsing + DocumentFields validation helpers.
Copied from the parse/validate logic in src/document/ocr_llm_extractor.py.

These are used by both the LLM extractor and the VLM extractor in this
Kaggle package (mirroring the production import structure).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


# ── DocumentFields (lightweight copy, no Pydantic dependency) ────────────────
# The Kaggle package avoids the full production schema.py (which imports from
# pydantic and the full src tree). Instead we use a simple dataclass-style
# namedtuple with the same field names.

DOCUMENT_FIELD_KEYS = {"name", "dob", "doc_number", "doc_type", "gender", "address"}


def make_empty_fields() -> Dict[str, Optional[str]]:
    """Return a dict with all DocumentFields keys set to None."""
    return {k: None for k in DOCUMENT_FIELD_KEYS}


# ── JSON parsing ─────────────────────────────────────────────────────────────


def parse_json(raw: str):
    """
    Robustly extract a JSON object from a model output string.

    Returns
    -------
    (parsed_dict, json_valid, parse_error)
    """
    text = raw.strip()

    # 1. Direct parse
    try:
        return json.loads(text), True, None
    except json.JSONDecodeError:
        pass

    # 2. Strip Markdown code fences
    fenced = re.sub(r"```(?:json)?\s*(.*?)```", r"\1", text, flags=re.DOTALL).strip()
    try:
        return json.loads(fenced), True, None
    except json.JSONDecodeError:
        pass

    # 3. Extract first {...} block (handles surrounding explanation text)
    match = re.search(r"\{[^{}]*\}", fenced, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()), True, None
        except json.JSONDecodeError as e:
            return None, False, f"JSON block found but unparseable: {e}"

    return None, False, f"No JSON object found in output: {repr(text[:200])}"


def validate_fields(parsed: Optional[dict]) -> Dict[str, Optional[str]]:
    """
    Map a raw parsed dict to a DocumentFields dict.
    Unknown keys are silently ignored; missing keys become None.
    Empty strings are treated as None (matching Pydantic validator in production).
    """
    fields = make_empty_fields()
    if not parsed:
        return fields

    for k in DOCUMENT_FIELD_KEYS:
        v = parsed.get(k, None)
        if isinstance(v, str) and not v.strip():
            v = None
        fields[k] = v

    return fields
