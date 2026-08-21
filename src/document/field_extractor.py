"""
document/field_extractor.py
-----------------------------
DEPRECATED: This class is kept for backward compatibility only.

The class `VLMFieldExtractor` was misleadingly named — it is NOT a
Vision-Language Model. It is a text-only LLM (Qwen2.5-1.5B) that
receives raw OCR text, not a document image.

The canonical implementation has moved to:
    src/document/ocr_llm_extractor.py  →  OCRLLMExtractor

The new genuine VLM (Approach 2) is at:
    src/document/vlm_extractor.py      →  VLMExtractor

This file re-exports VLMFieldExtractor as a deprecation shim so that
existing callers (test_vlm_only.py, etc.) continue to work unchanged.
"""

from __future__ import annotations

import warnings

from src.document.ocr_llm_extractor import (
    OCRLLMExtractor as _OCRLLMExtractor,
    _DEFAULT_MODEL,
    _DEFAULT_ADAPTER,
)


class VLMFieldExtractor(_OCRLLMExtractor):
    """
    DEPRECATED — use OCRLLMExtractor from src.document.ocr_llm_extractor.

    This is a text-only LLM extractor (Approach 1: OCR → text LLM → JSON).
    For the genuine VLM image-based extractor (Approach 2), use VLMExtractor
    from src.document.vlm_extractor.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        adapter_path: str = _DEFAULT_ADAPTER,
    ):
        warnings.warn(
            "VLMFieldExtractor is deprecated and will be removed in a future version. "
            "Use OCRLLMExtractor (Approach 1 — OCR + text LLM) or "
            "VLMExtractor (Approach 2 — true Vision-Language Model) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            model_path=model_path,
            adapter_path=adapter_path,
        )

    def extract_all(self, ocr_text: str) -> dict:
        """
        DEPRECATED shim — wraps OCRLLMExtractor for backward compatibility.

        Note: This method accepts raw OCR text (not a PIL Image).
        The new DocumentExtractor interface uses .extract(image) instead.
        """
        warnings.warn(
            "extract_all(ocr_text) is deprecated. "
            "Use .extract(pil_image) via the DocumentExtractor interface.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Reproduce original behavior: call the LLM on already-extracted text
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        from src.document.ocr_llm_extractor import (
            _SYSTEM_PROMPT,
            _EXTRACTION_QUESTION,
            _parse_json,
        )

        if not ocr_text or not ocr_text.strip():
            return {}

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Document OCR text:\n{ocr_text}\n\nQuestion: {_EXTRACTION_QUESTION}",
            },
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            raw_output = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=128,
                verbose=False,
                sampler=make_sampler(temp=0),
            )
            parsed, _, _ = _parse_json(raw_output)
            return parsed or {}
        except Exception as e:
            print(f"[VLMFieldExtractor] Extraction failed: {e}")
            return {}
