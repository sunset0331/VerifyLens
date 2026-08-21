"""
verifylens_kaggle/llm_extractor.py
-------------------------------------
Approach 1 (BASE model only) for Kaggle GPU benchmark.

Pipeline: PIL Image → PaddleOCR (GPU) → OCR text → Qwen2.5-1.5B (CUDA) → JSON

IMPORTANT — COMPATIBILITY NOTE
================================
Production uses:
  Model   : mlx-community/Qwen2.5-1.5B-Instruct-4bit
  Runtime : mlx-lm (Apple MLX)
  Adapter : checkpoints/verifylens-adapter (MLX LoRA format)

This Kaggle implementation uses:
  Model   : Qwen/Qwen2.5-1.5B-Instruct
  Runtime : HuggingFace Transformers + CUDA
  Adapter : NONE — see LoRA section below

WHY THE ADAPTER CANNOT BE LOADED ON KAGGLE
============================================
The existing LoRA adapter was trained with mlx-lm against the MLX 4-bit
quantized model (mlx-community/Qwen2.5-1.5B-Instruct-4bit).

mlx-lm uses its own internal layer naming convention for LoRA weights.
The HuggingFace PEFT library expects a different naming scheme and a
different base model (full precision, not MLX quantized).

Direct loading of the MLX adapter safetensors into HuggingFace PEFT
would result in shape mismatches and incorrect results.

CONVERSION REQUIREMENT:
  To evaluate the LoRA adapter on Kaggle, a conversion utility would
  need to:
  1. Map MLX layer names → HuggingFace layer names
  2. Convert 4-bit MLX tensors → float32/BF16
  3. Export in HuggingFace PEFT adapter format

  This conversion is non-trivial and out of scope for this package.
  When --model lora is requested, the benchmark stops and prints this
  explanation rather than silently using base model or fabricating results.

WHAT THIS FILE IMPLEMENTS:
  --model base  : Qwen2.5-1.5B-Instruct in BF16 on CUDA, no adapter.
                  This evaluates the base model's zero-shot extraction
                  capability via the same OCR+LLM pipeline.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
from PIL import Image

from verifylens_kaggle.ocr_extractor import OCRExtractor
from verifylens_kaggle.json_utils import parse_json, validate_fields, make_empty_fields

# ── Exact same prompts as production (src/document/ocr_llm_extractor.py) ────

_SYSTEM_PROMPT = (
    "You are a document intelligence assistant specializing in Indian identity documents. "
    "You will receive the OCR-extracted text from an identity document and a question. "
    "Respond ONLY with a valid JSON object containing the requested field(s). "
    'Example: {"name": "Ravi Sharma"} or {"dob": "23/04/1990"}. '
    "If a field is not found in the text, use null as the value."
)

_EXTRACTION_QUESTION = "Extract all key fields from this document as JSON."

# Kaggle model — HuggingFace full-precision equivalent
_KAGGLE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
_KAGGLE_DTYPE = "bfloat16"


class LLMExtractor:
    """
    Approach 1 (base model, no LoRA) for Kaggle GPU benchmark.

    Pipeline: PIL Image → GPU PaddleOCR → OCR text → Qwen2.5-1.5B (BF16, CUDA) → JSON

    Loaded once at construction. Never re-loads per sample.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID.
    max_new_tokens : int
        Max tokens to generate (matches production default of 128).
    temperature : float
        0.0 = greedy/deterministic (matches production).
    """

    def __init__(
        self,
        model_id: str = _KAGGLE_MODEL_ID,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ):
        self._model_id = model_id
        self._adapter_path = adapter_path
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = None
        self._tokenizer = None
        self._ocr: Optional[OCRExtractor] = None

        self._load_model()

    def _load_model(self) -> None:
        """Load model, tokenizer, and optional PEFT adapter once."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Determine if we need peft
        if self._adapter_path:
            try:
                from peft import PeftModel
            except ImportError:
                raise RuntimeError("peft library is required to load adapter_path.")

        print(f"[LLMExtractor] Loading model: {self._model_id}")
        if self._adapter_path:
            print(f"[LLMExtractor] Loading PEFT adapter: {self._adapter_path}")
        print(f"[LLMExtractor] Device: {self._device}, dtype: {_KAGGLE_DTYPE}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, trust_remote_code=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        if self._adapter_path:
            self._model = PeftModel.from_pretrained(base_model, self._adapter_path)
        else:
            self._model = base_model
            
        self._model.eval()
        print(f"[LLMExtractor] Model ready on {self._device}.")

    def _get_ocr(self) -> OCRExtractor:
        if self._ocr is None:
            use_gpu = self._device == "cuda"
            self._ocr = OCRExtractor(use_gpu=use_gpu)
        return self._ocr

    @property
    def model_info(self) -> Dict[str, str]:
        if self._adapter_path:
            return {
                "model_id": self._model_id,
                "dtype": _KAGGLE_DTYPE,
                "device": self._device,
                "adapter": self._adapter_path,
                "production_equivalent": "mlx-community/Qwen2.5-1.5B-Instruct-4bit via mlx-lm + LoRA adapter",
                "comparability_note": (
                    "LoRA adapter is loaded via PEFT. Results reflect the "
                    "fine-tuned production model capability."
                ),
            }
        return {
            "model_id": self._model_id,
            "dtype": _KAGGLE_DTYPE,
            "device": self._device,
            "adapter": "none",
            "production_equivalent": "mlx-community/Qwen2.5-1.5B-Instruct-4bit via mlx-lm + LoRA adapter",
            "comparability_note": (
                "Base model only — LoRA adapter not loaded. "
                "Results reflect zero-shot capability of the base model, "
                "NOT the fine-tuned production model."
            ),
        }

    def extract(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run Approach 1 pipeline on a single image.

        Returns
        -------
        dict with keys:
            prediction      : dict (DocumentFields)
            json_valid      : bool
            latency_ms      : float (total: OCR + LLM)
            ocr_latency_ms  : float
            llm_latency_ms  : float
            parse_error     : str | None
            error           : str | None
            raw_output      : str | None
        """
        result: Dict[str, Any] = {
            "prediction": make_empty_fields(),
            "json_valid": False,
            "latency_ms": None,
            "ocr_latency_ms": None,
            "llm_latency_ms": None,
            "parse_error": None,
            "error": None,
            "raw_output": None,
        }

        t_total_start = time.perf_counter()

        # ── Step 1: OCR ──────────────────────────────────────────────────────
        try:
            ocr_result = self._get_ocr().extract(image)
            raw_text = ocr_result.get("raw_text", "")
            result["ocr_latency_ms"] = ocr_result.get("ocr_latency_ms")
        except Exception as e:
            result["error"] = f"OCR failed: {e}"
            result["latency_ms"] = round((time.perf_counter() - t_total_start) * 1000, 1)
            return result

        if not raw_text.strip():
            result["error"] = "OCR produced no text"
            result["latency_ms"] = round((time.perf_counter() - t_total_start) * 1000, 1)
            return result

        # ── Step 2: Build prompt (identical to production) ───────────────────
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Document OCR text:\n{raw_text}\n\nQuestion: {_EXTRACTION_QUESTION}",
            },
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # ── Step 3: Tokenize and generate (CUDA, with sync timing) ──────────
        try:
            inputs = self._tokenizer(
                prompt_text, return_tensors="pt", truncation=True, max_length=2048
            ).to(self._device)

            t_llm_start = time.perf_counter()

            with torch.inference_mode():
                if self._device == "cuda":
                    torch.cuda.synchronize()

                gen_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,          # greedy (temperature=0)
                    temperature=None,
                    top_p=None,
                    pad_token_id=self._tokenizer.eos_token_id,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict_in_generate=False,
                )

                if self._device == "cuda":
                    torch.cuda.synchronize()

            llm_latency_ms = (time.perf_counter() - t_llm_start) * 1000

            # Decode only new tokens (strip the input prompt)
            new_ids = gen_ids[0][inputs["input_ids"].shape[1]:]
            raw_output = self._tokenizer.decode(new_ids, skip_special_tokens=True)

            # Explicit cleanup
            del gen_ids, new_ids, inputs
            if self._device == "cuda":
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            # Re-raise so the benchmark runner can handle OOM
            raise
        except Exception as e:
            result["error"] = f"LLM generation failed: {e}"
            result["latency_ms"] = round((time.perf_counter() - t_total_start) * 1000, 1)
            return result

        # ── Step 4: Parse + validate ──────────────────────────────────────────
        parsed, json_valid, parse_error = parse_json(raw_output)
        fields = validate_fields(parsed)

        total_latency_ms = (time.perf_counter() - t_total_start) * 1000

        result.update(
            {
                "prediction": fields,
                "json_valid": json_valid,
                "latency_ms": round(total_latency_ms, 1),
                "ocr_latency_ms": result["ocr_latency_ms"],
                "llm_latency_ms": round(llm_latency_ms, 1),
                "parse_error": parse_error,
                "raw_output": raw_output,
            }
        )

        return result
