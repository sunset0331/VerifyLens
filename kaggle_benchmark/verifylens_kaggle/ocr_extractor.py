"""
verifylens_kaggle/ocr_extractor.py
-------------------------------------
GPU-compatible PaddleOCR wrapper for Kaggle benchmark.

This mirrors the OCR logic from src/document/ocr.py with two differences:
  1. Uses use_gpu=True when CUDA is available (production uses CPU on Mac).
  2. Standalone — does not import from src/.

OCR semantics are identical to production:
  - Same PaddleOCR API (paddleocr >= 2.7.0)
  - Same regex patterns for DOB, Aadhaar, PAN, Passport number extraction
  - Same raw_text construction (join all detected text blocks)

PaddleOCR on Kaggle: uses 'gpu:0' device string when CUDA is available.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ── Regex patterns (identical to src/document/ocr.py) ───────────────────────

_DOB_PATTERNS = [
    re.compile(r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b"),       # DD/MM/YYYY
    re.compile(r"\b(\d{4}[/\-\.]\d{2}[/\-\.]\d{2})\b"),       # YYYY/MM/DD
    re.compile(r"\b(DOB|Date of Birth)[:\s]+([0-9/\-\.]+)", re.IGNORECASE),
]

_AADHAAR_PATTERN = re.compile(r"\b(\d{4}\s\d{4}\s\d{4})\b")
_PAN_PATTERN = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
_PASSPORT_PATTERN = re.compile(r"\b([A-Z][0-9]{7})\b")


class OCRExtractor:
    """
    GPU-compatible PaddleOCR wrapper.

    Lazy-loads PaddleOCR on first use. On Kaggle, use_gpu=True routes
    computation to the CUDA GPU. On Mac (local tests), use_gpu=False.

    Parameters
    ----------
    lang : str
        PaddleOCR language code (default: 'en').
    use_gpu : bool
        If True, uses GPU (requires paddlepaddle-gpu). Detected automatically.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = True):
        self._lang = lang
        self._use_gpu = use_gpu
        self._engine = None  # lazy init

    def _get_engine(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR  # type: ignore
            except ImportError:
                raise ImportError(
                    "PaddleOCR is not installed. Run:\n"
                    "  pip install paddlepaddle-gpu paddleocr\n"
                    "On Kaggle, paddlepaddle-gpu is available as a package."
                )

            device = "gpu:0" if self._use_gpu else "cpu"
            print(f"[OCRExtractor] Initializing PaddleOCR (device={device}, lang={self._lang})")
            self._engine = PaddleOCR(
                use_textline_orientation=True,
                lang=self._lang,
                device=device,
            )
            print("[OCRExtractor] PaddleOCR ready.")
        return self._engine

    def get_raw_text_blocks(self, image: Image.Image) -> List[Tuple[str, float]]:
        """
        Run OCR and return (text, confidence) tuples sorted by reading order.
        """
        img_np = np.array(image.convert("RGB"))
        engine = self._get_engine()
        result = engine.predict(img_np)

        blocks: List[Tuple[str, float]] = []
        if result and len(result) > 0:
            res_dict = result[0]
            texts = res_dict.get("rec_texts", [])
            scores = res_dict.get("rec_scores", [])
            for text, conf in zip(texts, scores):
                if text.strip():
                    blocks.append((text.strip(), float(conf)))
        return blocks

    def extract(
        self, image: Image.Image, doc_type: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        """
        Run OCR and extract structured fields using regex heuristics.

        Identical logic to src/document/ocr.py OCRExtractor.extract().

        Parameters
        ----------
        image : PIL.Image.Image
        doc_type : str, optional
            Hint for document-type-specific patterns.

        Returns
        -------
        dict with keys: raw_text, name, dob, doc_number, ocr_confidence,
                        ocr_latency_ms
        """
        t0 = time.perf_counter()
        blocks = self.get_raw_text_blocks(image)
        ocr_latency_ms = (time.perf_counter() - t0) * 1000

        full_text = " ".join(text for text, _ in blocks)
        avg_conf = sum(c for _, c in blocks) / len(blocks) if blocks else 0.0

        fields: Dict[str, Optional[str]] = {
            "raw_text": full_text,
            "name": None,
            "dob": None,
            "doc_number": None,
            "ocr_confidence": round(avg_conf, 3),
            "ocr_latency_ms": round(ocr_latency_ms, 1),
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

        if fields["doc_number"] is None and (
            doc_type == "passport" or doc_type is None
        ):
            m = _PASSPORT_PATTERN.search(full_text)
            if m:
                fields["doc_number"] = m.group(1)

        # ── Name extraction (heuristic: line after "Name" keyword) ──
        name_match = re.search(
            r"(?:Name|नाम)[:\s]+([A-Z][a-z]+(?: [A-Z][a-z]+)+)", full_text
        )
        if name_match:
            fields["name"] = name_match.group(1)

        return fields
