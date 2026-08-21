"""
src/document/ocr_llm_extractor.py
-----------------------------------
Approach 1: Document Image → PaddleOCR → OCR text → Qwen2.5-1.5B (+ LoRA) → JSON

This is the renamed and refactored version of the original VLMFieldExtractor.
The class was misleadingly named "VLM" because it only receives OCR text —
it is not a Vision-Language Model.

Now correctly named OCRLLMExtractor.

A backward-compatible alias `VLMFieldExtractor` is preserved in
field_extractor.py with a deprecation warning.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from src.document.base_extractor import DocumentExtractor
from src.document.schema import DocumentFields, ExtractionResult
from src.document.ocr import OCRExtractor
from src.utils.config import config


# ── Extraction prompt (identical to original for continuity) ─────────────────

_SYSTEM_PROMPT = (
    "You are a document intelligence assistant specializing in Indian identity documents. "
    "You will receive the OCR-extracted text from an identity document and a question. "
    "Respond ONLY with a valid JSON object containing the requested field(s). "
    'Example: {"name": "Ravi Sharma"} or {"dob": "23/04/1990"}. '
    "If a field is not found in the text, use null as the value."
)

_EXTRACTION_QUESTION = "Extract all key fields from this document as JSON."

_DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
_DEFAULT_ADAPTER = "checkpoints/verifylens-adapter"


class OCRLLMExtractor(DocumentExtractor):
    """
    Approach 1 document extractor.

    Pipeline:  PIL Image → PaddleOCR → raw text → Qwen2.5-1.5B+LoRA → JSON

    The text LLM receives OCR text, NOT the image directly.
    The LoRA adapter is loaded when available; falls back to the base model
    with an explicit flag in the result (extractor_fallback=True).

    Parameters
    ----------
    model_path : str
        HuggingFace model ID or local path for the MLX quantized model.
    adapter_path : str
        Path to the trained MLX LoRA adapter weights.
    max_tokens : int
        Maximum tokens to generate.
    """

    _EXTRACTION_MODE = "ocr_llm"

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        adapter_path: str = _DEFAULT_ADAPTER,
        max_tokens: int = 128,
    ):
        self._model_path = model_path
        self._adapter_path = adapter_path
        self._max_tokens = max_tokens
        self._using_adapter = False

        # Lazy-loaded
        self._ocr: Optional[OCRExtractor] = None
        self._model = None
        self._tokenizer = None

        self._load_llm()

    def _load_llm(self) -> None:
        """Load the MLX text model once at initialization."""
        from mlx_lm import load

        if self._adapter_path:
            adapter_path_obj = Path(self._adapter_path)
            if not adapter_path_obj.exists():
                raise FileNotFoundError(
                    f"Adapter '{self._adapter_path}' not found. "
                    f"LoRA fine-tuning must be run to generate the adapter."
                )
            print(f"[OCRLLMExtractor] Loading model: {self._model_path}")
            print(f"[OCRLLMExtractor] Loading adapter: {self._adapter_path}")
            self._model, self._tokenizer = load(
                self._model_path, adapter_path=self._adapter_path
            )
            self._using_adapter = True
        else:
            print(f"[OCRLLMExtractor] Loading BASE model ONLY: {self._model_path}")
            self._model, self._tokenizer = load(self._model_path)
            self._using_adapter = False

    def _get_ocr(self) -> OCRExtractor:
        if self._ocr is None:
            self._ocr = OCRExtractor()
        return self._ocr

    # ── DocumentExtractor interface ──────────────────────────────────────────

    @property
    def extraction_mode(self) -> str:
        return self._EXTRACTION_MODE

    @property
    def model_name(self) -> str:
        suffix = f"+adapter" if self._using_adapter else "+base_only"
        return f"{self._model_path}{suffix}"

    def extract(self, image: Image.Image) -> ExtractionResult:
        """
        Full Approach 1 pipeline: image → OCR → text LLM → DocumentFields.
        """
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        # ── Step 1: OCR ──────────────────────────────────────────────────────
        try:
            ocr_result = self._get_ocr().extract(image)
            raw_text = ocr_result.get("raw_text", "")
        except Exception as e:
            return ExtractionResult(
                fields=DocumentFields(),
                extraction_mode=self._EXTRACTION_MODE,
                model=self.model_name,
                base_model=self._model_path,
                adapter_loaded=self._using_adapter,
                json_valid=False,
                extractor_fallback=not self._using_adapter,
                error=f"OCR failed: {e}",
            )

        if not raw_text.strip():
            return ExtractionResult(
                fields=DocumentFields(),
                extraction_mode=self._EXTRACTION_MODE,
                model=self.model_name,
                base_model=self._model_path,
                adapter_loaded=self._using_adapter,
                json_valid=False,
                extractor_fallback=not self._using_adapter,
                error="OCR produced no text",
            )

        # ── Step 2: Build prompt ──────────────────────────────────────────────
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Document OCR text:\n{raw_text}\n\nQuestion: {_EXTRACTION_QUESTION}"
                ),
            },
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # ── Step 3: Generate ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            raw_output = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self._max_tokens,
                verbose=False,
                sampler=make_sampler(temp=0),
            )
        except Exception as e:
            return ExtractionResult(
                fields=DocumentFields(),
                extraction_mode=self._EXTRACTION_MODE,
                model=self.model_name,
                base_model=self._model_path,
                adapter_loaded=self._using_adapter,
                json_valid=False,
                extractor_fallback=not self._using_adapter,
                error=f"LLM generation failed: {e}",
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        # ── Step 4: Parse + validate ──────────────────────────────────────────
        parsed, json_valid, parse_error = _parse_json(raw_output)
        fields = _validate_fields(parsed)

        return ExtractionResult(
            fields=fields,
            extraction_mode=self._EXTRACTION_MODE,
            model=self.model_name,
            base_model=self._model_path,
            adapter_loaded=self._using_adapter,
            json_valid=json_valid,
            extractor_fallback=not self._using_adapter,
            latency_ms=round(latency_ms, 1),
            parse_error=parse_error,
            raw_output=raw_output,
        )


# ── Shared JSON helpers (used by both extractors) ────────────────────────────

def _parse_json(raw: str) -> tuple[Optional[dict], bool, Optional[str]]:
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


def _validate_fields(parsed: Optional[dict]) -> DocumentFields:
    """
    Map a raw parsed dict to DocumentFields using Pydantic validation.
    Unknown keys are silently ignored; missing keys become null.
    """
    if not parsed:
        return DocumentFields()

    # Only pass known schema keys
    known_keys = set(DocumentFields.model_fields.keys())
    filtered = {k: v for k, v in parsed.items() if k in known_keys}
    return DocumentFields(**filtered)
