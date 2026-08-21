"""
src/document/vlm_extractor.py
-------------------------------
Approach 2: Document Image → VLM (Qwen2.5-VL) → JSON

The VLM receives the ORIGINAL DOCUMENT IMAGE.
It does NOT call PaddleOCR.
It does NOT receive OCR text.

Selected model: mlx-community/Qwen2.5-VL-2B-Instruct-4bit
  - True multimodal: processes pixel data directly
  - 4-bit quantized for Apple Silicon (M-series)
  - Excellent document/layout understanding
  - ~1.5 GB on disk, fits in 8 GB unified memory alongside other models
  - Uses mlx-vlm framework (same MLX backend as Approach 1)
  - Can be fine-tuned later with Unsloth (HF-compatible) when a GPU is available

Architecture for future fine-tuning (Unsloth path):
  - Export to HF-compatible format from MLX
  - Fine-tune with Unsloth on GPU
  - Re-quantize to 4-bit MLX for inference
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

from PIL import Image

from src.document.base_extractor import DocumentExtractor
from src.document.schema import DocumentFields, ExtractionResult

# Shared JSON helpers (defined in ocr_llm_extractor, imported here)
from src.document.ocr_llm_extractor import _parse_json, _validate_fields


# ── Model configuration ───────────────────────────────────────────────────────

_DEFAULT_VLM_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

# ── Extraction prompt ──────────────────────────────────────────────────────────
# Strict, structured prompt designed for Qwen2.5-VL's chat format.
# We ask for JSON only — no preamble, no markdown, no explanation.

_EXTRACTION_SYSTEM = (
    "You are an identity-document information extraction system. "
    "Analyze the provided document image and extract visible information only. "
    "Return ONLY a valid JSON object — no markdown, no explanation, no preamble. "
    "Do not infer, guess, or fabricate any values."
)

_EXTRACTION_USER = (
    "Extract the following fields from this identity document image. "
    "Return a JSON object with exactly these keys:\n\n"
    "  name       – full name of the holder\n"
    "  dob        – date of birth (DD/MM/YYYY)\n"
    "  doc_number – document ID number\n"
    "  doc_type   – document type (e.g. Aadhaar Card, PAN Card, Passport)\n"
    "  gender     – gender as printed\n"
    "  address    – address if present on the document\n\n"
    "Use null for any field that is not visible or readable in the image.\n"
    'Return only the JSON object. Example: {"name": "Ravi Kumar", "dob": "14/09/1990", '
    '"doc_number": "1234 5678 9012", "doc_type": "Aadhaar Card", "gender": "Male", "address": null}'
)


class VLMExtractor(DocumentExtractor):
    """
    Approach 2 document extractor.

    Pipeline:  PIL Image → Qwen2.5-VL-2B-Instruct (4-bit, MLX) → JSON

    The VLM receives the original document image directly.
    PaddleOCR is NOT called at any point in this pipeline.

    Parameters
    ----------
    model_path : str
        mlx-community model ID or local path for Qwen2.5-VL.
    max_new_tokens : int
        Maximum tokens to generate (default 256 — enough for 6 JSON fields).
    temperature : float
        Sampling temperature. 0 = greedy/deterministic.
    resize_max_px : int
        Maximum side length (px) for the image sent to the VLM.
        Qwen2.5-VL handles variable resolution natively, but very large
        images slow inference. 1120 is a good balance for A4 documents.
    """

    _EXTRACTION_MODE = "vlm"

    def __init__(
        self,
        model_path: str = _DEFAULT_VLM_MODEL,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        resize_max_px: int = 1120,
    ):
        self._model_path = model_path
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._resize_max_px = resize_max_px

        # Loaded once on init
        self._model = None
        self._processor = None

        self._load_model()

    def _load_model(self) -> None:
        """Load Qwen2.5-VL model + processor once at initialization."""
        from mlx_vlm import load

        print(f"[VLMExtractor] Loading model: {self._model_path}")
        print(
            "[VLMExtractor] This downloads ~1.5 GB on first run — "
            "subsequent runs use the local cache."
        )
        self._model, self._processor = load(self._model_path)
        print(f"[VLMExtractor] Model ready.")

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Ensure image is RGB and resize if any side exceeds resize_max_px.
        Aspect ratio is always preserved.
        """
        img = image.convert("RGB")
        w, h = img.size
        max_side = max(w, h)
        if max_side > self._resize_max_px:
            scale = self._resize_max_px / max_side
            img = img.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )
        return img

    # ── DocumentExtractor interface ──────────────────────────────────────────

    @property
    def extraction_mode(self) -> str:
        return self._EXTRACTION_MODE

    @property
    def model_name(self) -> str:
        return self._model_path

    def extract(self, image: Image.Image) -> ExtractionResult:
        """
        Approach 2 pipeline: image → VLM → DocumentFields.

        The image is passed directly to Qwen2.5-VL.
        No OCR is performed at any point.
        """
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        # ── Step 1: Preprocess image ─────────────────────────────────────────
        try:
            img = self._preprocess_image(image)
        except Exception as e:
            return ExtractionResult(
                fields=DocumentFields(),
                extraction_mode=self._EXTRACTION_MODE,
                model=self.model_name,
                base_model=self._model_path,
                adapter_loaded=False,
                json_valid=False,
                error=f"Image preprocessing failed: {e}",
            )

        # ── Step 2: Build multimodal prompt ──────────────────────────────────
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": _EXTRACTION_USER},
                    ],
                }
            ]
            # apply_chat_template handles the model-specific chat format
            formatted_prompt = apply_chat_template(
                self._processor,
                config=self._model.config if hasattr(self._model, "config") else None,
                prompt=_EXTRACTION_USER,
                num_images=1,
            )
        except Exception as e:
            return ExtractionResult(
                fields=DocumentFields(),
                extraction_mode=self._EXTRACTION_MODE,
                model=self.model_name,
                base_model=self._model_path,
                adapter_loaded=False,
                json_valid=False,
                error=f"Prompt construction failed: {e}",
            )

        # ── Step 3: Generate ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            output = generate(
                self._model,
                self._processor,
                image=img,
                prompt=formatted_prompt,
                max_tokens=self._max_new_tokens,
                temperature=self._temperature,
                verbose=False,
            )
            # generate() may return a string or a GenerationResult object
            if hasattr(output, "text"):
                raw_output = output.text
            elif isinstance(output, str):
                raw_output = output
            else:
                raw_output = str(output)
        except Exception as e:
            return ExtractionResult(
                fields=DocumentFields(),
                extraction_mode=self._EXTRACTION_MODE,
                model=self.model_name,
                base_model=self._model_path,
                adapter_loaded=False,
                json_valid=False,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                error=f"VLM generation failed: {e}",
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
            adapter_loaded=False,
            json_valid=json_valid,
            extractor_fallback=False,   # VLM has no fallback path
            latency_ms=round(latency_ms, 1),
            parse_error=parse_error,
            raw_output=raw_output,
        )
