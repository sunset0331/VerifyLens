"""
verifylens_kaggle/vlm_extractor.py
-------------------------------------
Approach 2 (VLM) for Kaggle GPU benchmark.

Pipeline: PIL Image → Qwen2.5-VL-3B-Instruct (BF16, CUDA) → JSON

IMPORTANT — COMPATIBILITY NOTE
================================
Production uses:
  Model     : mlx-community/Qwen2.5-VL-3B-Instruct-4bit
  Runtime   : mlx-vlm (Apple MLX, Apple Silicon only)
  Precision : 4-bit MLX quantization

This Kaggle implementation uses:
  Model     : Qwen/Qwen2.5-VL-3B-Instruct
  Runtime   : HuggingFace Transformers + qwen-vl-utils + CUDA
  Precision : BF16 (bfloat16)

These are the SAME model family and weights (Qwen2.5-VL-3B).
They are NOT bit-identical due to:
  - Different precision (4-bit MLX vs BF16 full)
  - Different inference framework (mlx-vlm vs HuggingFace Transformers)
  - Different image preprocessing pipelines

The benchmark report explicitly states this difference.
Results from this Kaggle run SHOULD NOT be claimed to be equivalent
to Mac MLX results without caveats. They benchmark the same model
family through a GPU-compatible implementation.

EXTRACTION PROMPT
=================
Identical to production (src/document/vlm/vlm_extractor.py):
  _EXTRACTION_USER — same field list, same JSON format instructions
  Same temperature=0.0 (greedy)
  Same max_new_tokens=256
  Same resize_max_px=1120
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, Optional

import torch
from PIL import Image

from verifylens_kaggle.json_utils import parse_json, validate_fields, make_empty_fields

# ── Exact same prompt as production (src/document/vlm/vlm_extractor.py) ─────

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

# Kaggle model — same family, BF16 full precision
_KAGGLE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
_KAGGLE_DTYPE = "bfloat16"

# Same as production
_RESIZE_MAX_PX = 1120
_MAX_NEW_TOKENS = 256
_TEMPERATURE = 0.0


class VLMExtractor:
    """
    Approach 2 VLM extractor for Kaggle GPU benchmark.

    Pipeline: PIL Image → Qwen2.5-VL-3B-Instruct (BF16, CUDA) → JSON

    The model is loaded ONCE at construction and reused for all samples.
    Processing is sequential (batch_size=1) for memory safety.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID.
    max_new_tokens : int
        Max tokens to generate (default 256).
    temperature : float
        0.0 = greedy/deterministic.
    resize_max_px : int
        Maximum side length for image preprocessing (same as production).
    """

    def __init__(
        self,
        model_id: str = _KAGGLE_MODEL_ID,
        max_new_tokens: int = _MAX_NEW_TOKENS,
        temperature: float = _TEMPERATURE,
        resize_max_px: int = _RESIZE_MAX_PX,
    ):
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._resize_max_px = resize_max_px
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = None
        self._processor = None

        self._load_model()

    def _load_model(self) -> None:
        """Load Qwen2.5-VL model and processor once."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"[VLMExtractor] Loading model: {self._model_id}")
        print(f"[VLMExtractor] Device: {self._device}, dtype: {_KAGGLE_DTYPE}")
        print(
            "[VLMExtractor] Downloading ~7 GB on first run — "
            "subsequent runs use the Kaggle model cache."
        )

        self._processor = AutoProcessor.from_pretrained(
            self._model_id, trust_remote_code=True
        )

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",  # PyTorch built-in, no flash-attn package needed
        )
        self._model.eval()
        print(f"[VLMExtractor] Model ready on {self._device}.")

    @property
    def model_info(self) -> Dict[str, str]:
        return {
            "model_id": self._model_id,
            "dtype": _KAGGLE_DTYPE,
            "device": self._device,
            "production_model": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
            "production_runtime": "mlx-vlm (Apple MLX)",
            "kaggle_model": self._model_id,
            "kaggle_runtime": "HuggingFace Transformers + CUDA",
            "comparability_note": (
                "Same model family and weights (Qwen2.5-VL-3B). "
                "NOT bit-identical: production is 4-bit MLX, Kaggle is BF16 CUDA. "
                "Results indicate model-family capability on GPU, not Mac parity."
            ),
        }

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Ensure image is RGB and resize if any side exceeds resize_max_px.
        Aspect ratio preserved. Identical logic to production VLMExtractor.
        """
        img = image.convert("RGB")
        w, h = img.size
        max_side = max(w, h)
        if max_side > self._resize_max_px:
            scale = self._resize_max_px / max_side
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img

    def extract(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run Approach 2 pipeline on a single image.

        Returns
        -------
        dict with keys:
            prediction      : dict (DocumentFields)
            json_valid      : bool
            latency_ms      : float
            ocr_latency_ms  : None (VLM does not use OCR)
            llm_latency_ms  : None
            parse_error     : str | None
            error           : str | None
            raw_output      : str | None
        """
        result: Dict[str, Any] = {
            "prediction": make_empty_fields(),
            "json_valid": False,
            "latency_ms": None,
            "ocr_latency_ms": None,   # VLM does not call OCR
            "llm_latency_ms": None,
            "parse_error": None,
            "error": None,
            "raw_output": None,
        }

        # ── Step 1: Preprocess image ─────────────────────────────────────────
        try:
            img = self._preprocess_image(image)
        except Exception as e:
            result["error"] = f"Image preprocessing failed: {e}"
            return result

        # ── Step 2: Build multimodal messages ────────────────────────────────
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": _EXTRACTION_USER},
                    ],
                }
            ]

            # Use qwen-vl-utils if available for proper image processing
            try:
                from qwen_vl_utils import process_vision_info
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self._processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            except ImportError:
                # Fallback: use processor directly with the PIL image
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._processor(
                    text=[text],
                    images=[img],
                    return_tensors="pt",
                    padding=True,
                )

            inputs = inputs.to(self._device)

        except Exception as e:
            result["error"] = f"Input preparation failed: {e}"
            return result

        # ── Step 3: Generate (CUDA sync timing, no grad, no hidden states) ──
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                if self._device == "cuda":
                    torch.cuda.synchronize()

                gen_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,          # greedy
                    temperature=None,
                    top_p=None,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict_in_generate=False,
                )

                if self._device == "cuda":
                    torch.cuda.synchronize()

            latency_ms = (time.perf_counter() - t0) * 1000

            # Decode only new tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], gen_ids)
            ]
            raw_output = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            # Explicit cleanup — do not accumulate tensors across 200 samples
            del gen_ids, generated_ids_trimmed, inputs
            if self._device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        except torch.cuda.OutOfMemoryError:
            # Re-raise so the benchmark runner handles OOM correctly
            raise
        except Exception as e:
            result["error"] = f"VLM generation failed: {e}"
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return result

        # ── Step 4: Parse + validate ──────────────────────────────────────────
        parsed, json_valid, parse_error = parse_json(raw_output)
        fields = validate_fields(parsed)

        result.update(
            {
                "prediction": fields,
                "json_valid": json_valid,
                "latency_ms": round(latency_ms, 1),
                "parse_error": parse_error,
                "raw_output": raw_output,
            }
        )

        return result
