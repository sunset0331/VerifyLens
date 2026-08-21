"""
src/document/extractor_router.py
----------------------------------
Factory that selects and returns the appropriate DocumentExtractor
based on the configured extraction_mode.

Supported modes
---------------
  "ocr_llm"  →  OCRLLMExtractor  (Approach 1: image → OCR → text LLM → JSON)
  "vlm"      →  VLMExtractor     (Approach 2: image → VLM → JSON)

Usage
-----
    from src.document.extractor_router import get_extractor
    extractor = get_extractor(mode="vlm")
    result = extractor.extract(pil_image)

The router performs lazy imports so that loading Qwen2.5 (text) does not
incur the cost of loading Qwen2.5-VL and vice versa.
"""

from __future__ import annotations

from typing import Optional

from src.document.base_extractor import DocumentExtractor

SUPPORTED_MODES = ("ocr_llm", "vlm")


def get_extractor(
    mode: str,
    *,
    vlm_model: Optional[str] = None,
    vlm_max_new_tokens: int = 256,
    vlm_temperature: float = 0.0,
    vlm_resize_max_px: int = 1120,
    ocr_llm_model: Optional[str] = None,
    ocr_llm_adapter_path: Optional[str] = None,
    ocr_llm_max_tokens: int = 128,
) -> DocumentExtractor:
    """
    Instantiate and return the extractor for the requested mode.

    Parameters
    ----------
    mode : str
        "ocr_llm" or "vlm".
    vlm_model : str, optional
        Override the default VLM model path.
    ocr_llm_model : str, optional
        Override the default OCR+LLM model path.
    ocr_llm_adapter_path : str, optional
        Override the LoRA adapter path for Approach 1.
    ... (other kwargs forwarded to the selected extractor)

    Returns
    -------
    DocumentExtractor
        Fully initialized extractor, ready to call .extract(image).

    Raises
    ------
    ValueError
        If mode is not one of the supported strings.
    """
    mode = mode.strip().lower()

    if mode == "vlm":
        from src.document.vlm.vlm_extractor import VLMExtractor, _DEFAULT_VLM_MODEL

        return VLMExtractor(
            model_path=vlm_model or _DEFAULT_VLM_MODEL,
            max_new_tokens=vlm_max_new_tokens,
            temperature=vlm_temperature,
            resize_max_px=vlm_resize_max_px,
        )

    elif mode == "ocr_llm":
        from src.document.ocr_llm_extractor import (
            OCRLLMExtractor,
            _DEFAULT_MODEL,
            _DEFAULT_ADAPTER,
        )

        return OCRLLMExtractor(
            model_path=ocr_llm_model or _DEFAULT_MODEL,
            adapter_path=ocr_llm_adapter_path or _DEFAULT_ADAPTER,
            max_tokens=ocr_llm_max_tokens,
        )

    else:
        raise ValueError(
            f"Unknown extraction mode: '{mode}'. "
            f"Supported modes: {SUPPORTED_MODES}"
        )
