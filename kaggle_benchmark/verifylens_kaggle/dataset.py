"""
verifylens_kaggle/dataset.py
------------------------------
Standalone benchmark dataset loader for Kaggle.
Does NOT import from src/ — zero production dependencies.

Loads benchmark.jsonl and resolves image paths relative to the
benchmark directory root.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional


def load_benchmark(
    jsonl_path: str,
    limit: Optional[int] = None,
    skip_ids: Optional[set] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Load the VerifyLens KYC benchmark dataset from JSONL.

    Parameters
    ----------
    jsonl_path : str
        Path to benchmark.jsonl
    limit : int, optional
        Stop after this many samples (useful for smoke tests).
    skip_ids : set, optional
        Set of sample IDs to skip (used for checkpoint resume).

    Yields
    ------
    dict with keys:
        id            : str  (e.g. "00042")
        image_path    : str  (relative, e.g. "images/00042_pan.jpg")
        document_type : str  (e.g. "pan", "aadhaar", "passport")
        ground_truth  : dict (name, dob, doc_number, doc_type, gender, address)
        image_abs_path: Path (absolute, resolved against jsonl parent dir)
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    base_dir = path.parent
    skip_ids = skip_ids or set()
    count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            sample_id = sample["id"]

            if sample_id in skip_ids:
                continue

            if limit is not None and count >= limit:
                break

            # Resolve absolute image path
            sample["image_abs_path"] = base_dir / sample["image_path"]
            yield sample
            count += 1


def count_benchmark(jsonl_path: str) -> Dict[str, int]:
    """
    Count total samples and per-document-type counts without loading images.

    Returns
    -------
    dict: {"total": N, "aadhaar": N, "pan": N, "passport": N}
    """
    path = Path(jsonl_path)
    counts: Dict[str, int] = {"total": 0}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            counts["total"] += 1
            dt = sample.get("document_type", "unknown")
            counts[dt] = counts.get(dt, 0) + 1

    return counts


def validate_benchmark_images(jsonl_path: str) -> Dict[str, Any]:
    """
    Check that every image referenced in the JSONL actually exists on disk.

    Returns
    -------
    dict: {"ok": bool, "missing": [list of missing paths], "total": N}
    """
    path = Path(jsonl_path)
    base_dir = path.parent
    missing: List[str] = []
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            total += 1
            img_path = base_dir / sample["image_path"]
            if not img_path.exists():
                missing.append(str(img_path))

    return {"ok": len(missing) == 0, "missing": missing, "total": total}
