"""
document/field_extractor.py
-----------------------------
VLM-based structured field extraction using an MLX-LM fine-tuned
Qwen2.5-1.5B-Instruct model with a LoRA adapter.

Given raw OCR text from a document, returns a structured JSON answer.
Used as a second-pass extractor after OCR.

The model is loaded natively via MLX to run efficiently on Apple
Silicon unified memory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


# ── Prompt template ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a document intelligence assistant specializing in Indian identity documents. "
    "You will receive the OCR-extracted text from an identity document and a question. "
    "Respond ONLY with a valid JSON object containing the requested field(s). "
    'Example: {"name": "Ravi Sharma"} or {"dob": "23/04/1990"}. '
    "If a field is not found in the text, use null as the value."
)


class VLMFieldExtractor:
    """
    Fine-tuned MLX Qwen2.5-1.5B field extractor for identity documents.

    Parameters
    ----------
    model_path : str
        HuggingFace model ID or local path.
        Defaults to 'mlx-community/Qwen2.5-1.5B-Instruct-4bit'.
    adapter_path : str
        Path to the trained MLX LoRA adapter.
        Defaults to 'checkpoints/verifylens-adapter'.

    Example
    -------
    >>> extractor = VLMFieldExtractor()
    >>> fields = extractor.extract_all("Document OCR text: ...")
    >>> print(fields)
    {"name": "Ravi Sharma", "dob": "23/04/1990", ...}
    """

    def __init__(
        self,
        model_path: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        adapter_path: str = "checkpoints/verifylens-adapter",
    ):
        print(f"[VLMFieldExtractor] Loading base model: {model_path}")
        print(f"[VLMFieldExtractor] Loading adapter: {adapter_path}")
        
        # Load the model and tokenizer once and reuse them
        if Path(adapter_path).exists():
            self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        else:
            print(f"[WARNING] Adapter path '{adapter_path}' not found! Loading base model only.")
            self.model, self.tokenizer = load(model_path)

    def _parse_json_output(self, raw: str) -> Dict[str, Any]:
        """Extract the first JSON object from model output."""
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Try to extract JSON from surrounding text
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def extract_all(self, ocr_text: str) -> Dict[str, Optional[str]]:
        """
        Extract all standard fields from a document's OCR text in a single pass.

        Parameters
        ----------
        ocr_text : str
            The raw text extracted from the document by PaddleOCR.

        Returns
        -------
        dict with keys: name, dob, doc_number, doc_type, gender, address
        """
        if not ocr_text or not ocr_text.strip():
            return {}

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Document OCR text:\n{ocr_text}\n\nQuestion: Extract all key fields from this document as JSON."
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            raw_output = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=128,
                verbose=False,
                sampler=make_sampler(temp=0),  # greedy decoding: deterministic, reproducible
            )
            return self._parse_json_output(raw_output)
        except Exception as e:
            print(f"[VLMFieldExtractor] Extraction failed: {e}")
            return {}
