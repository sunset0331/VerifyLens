"""
document/ocr.py
----------------
PaddleOCR wrapper that extracts raw text blocks from a document image
and applies doc-type-specific regex heuristics for field parsing.

Returns structured fields: name, dob, doc_number (best-effort).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.utils.image_utils import load_image, ImageInput


# ── Regex patterns for common Indian ID fields ──────────────────────────────

_DOB_PATTERNS = [
    re.compile(r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b"),  # DD/MM/YYYY
    re.compile(r"\b(\d{4}[/\-\.]\d{2}[/\-\.]\d{2})\b"),  # YYYY/MM/DD
    re.compile(r"\b(DOB|Date of Birth)[:\s]+([0-9/\-\.]+)", re.IGNORECASE),
]

_AADHAAR_PATTERN = re.compile(r"\b(\d{4}\s\d{4}\s\d{4})\b")
_PAN_PATTERN = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
_PASSPORT_PATTERN = re.compile(r"\b([A-Z][0-9]{7})\b")


class OCRExtractor:
    """
    Thin wrapper around PaddleOCR with field extraction post-processing.

    Lazy-loads PaddleOCR on first use to avoid import-time cost.
    Falls back gracefully if paddleocr is not installed.

    Example
    -------
    >>> ocr = OCRExtractor()
    >>> result = ocr.extract("path/to/aadhaar.jpg", doc_type="aadhaar")
    >>> print(result["doc_number"])   # "1234 5678 9012"
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        self._lang = lang
        self._use_gpu = use_gpu
        self._engine = None  # lazy init

    def _get_engine(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR  # type: ignore
                self._engine = PaddleOCR(
                    use_angle_cls=True,
                    lang=self._lang,
                    use_gpu=self._use_gpu,
                    show_log=False,
                )
            except ImportError:
                raise ImportError(
                    "PaddleOCR is not installed. Run: pip install paddleocr paddlepaddle"
                )
        return self._engine

    def get_raw_text_blocks(self, image: ImageInput) -> List[Tuple[str, float]]:
        """
        Run OCR and return list of (text, confidence) tuples.

        Parameters
        ----------
        image : ImageInput
            Document image in any supported format.

        Returns
        -------
        List of (text_string, confidence_score) sorted by reading order (top→bottom).
        """
        img = load_image(image)
        img_np = np.array(img)

        engine = self._get_engine()
        result = engine.ocr(img_np, cls=True)

        blocks = []
        if result and result[0]:
            for line in result[0]:
                _, (text, conf) = line
                if text.strip():
                    blocks.append((text.strip(), float(conf)))
        return blocks

    def extract(
        self, image: ImageInput, doc_type: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        """
        Run OCR and extract structured fields using regex heuristics.

        Parameters
        ----------
        image : ImageInput
        doc_type : str, optional
            One of "aadhaar", "pan", "passport", "driving_license".
            If provided, applies doc-specific patterns first.

        Returns
        -------
        dict with keys: raw_text, name, dob, doc_number, confidence
        """
        blocks = self.get_raw_text_blocks(image)
        full_text = " ".join(text for text, _ in blocks)
        avg_conf = sum(c for _, c in blocks) / len(blocks) if blocks else 0.0

        fields: Dict[str, Optional[str]] = {
            "raw_text": full_text,
            "name": None,
            "dob": None,
            "doc_number": None,
            "ocr_confidence": round(avg_conf, 3),
        }

        # ── DOB extraction ──
        for pat in _DOB_PATTERNS:
            m = pat.search(full_text)
            if m:
                fields["dob"] = m.group(1) if len(m.groups()) == 1 else m.group(2)
                break

        # ── Document number extraction ──
        if doc_type == "aadhaar" or doc_type is None:
            m = _AADHAAR_PATTERN.search(full_text)
            if m:
                fields["doc_number"] = m.group(1)

        if fields["doc_number"] is None and (doc_type == "pan" or doc_type is None):
            m = _PAN_PATTERN.search(full_text)
            if m:
                fields["doc_number"] = m.group(1)

        if fields["doc_number"] is None and (doc_type == "passport" or doc_type is None):
            m = _PASSPORT_PATTERN.search(full_text)
            if m:
                fields["doc_number"] = m.group(1)

        # ── Name extraction (heuristic: line after "Name" keyword) ──
        name_match = re.search(r"(?:Name|नाम)[:\s]+([A-Z][a-z]+(?: [A-Z][a-z]+)+)", full_text)
        if name_match:
            fields["name"] = name_match.group(1)

        return fields
