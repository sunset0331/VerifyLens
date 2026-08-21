"""
verifylens_kaggle/normalizer.py
--------------------------------
COPIED FROM: src/evaluation/normalizer.py
Keep synchronized with production normalizer.

If the production normalizer changes, update this file too.
"""

import re
from typing import Optional, Any


def normalize_string(value: Any) -> Optional[str]:
    """
    Apply shared, strict text normalization layer before comparison.
    Identical to the production normalizer in src/evaluation/normalizer.py.
    """
    if value is None:
        return None

    s = str(value)

    # Lowercase
    s = s.lower()

    # Replace common date separators with slash for consistency
    # (assuming DD/MM/YYYY is the ground truth format in our dataset)
    s = re.sub(r'(\d{2})-(\d{2})-(\d{4})', r'\1/\2/\3', s)
    s = re.sub(r'(\d{2})\.(\d{2})\.(\d{4})', r'\1/\2/\3', s)

    # Strip whitespace
    s = s.strip()

    # Normalize repeated spaces to a single space
    s = re.sub(r'\s+', ' ', s)

    if s == "" or s == "none" or s == "null":
        return None

    return s


def normalize_dict(data: dict) -> dict:
    """Normalize all string values in a dictionary."""
    normalized = {}
    for k, v in data.items():
        normalized[k] = normalize_string(v)
    return normalized
