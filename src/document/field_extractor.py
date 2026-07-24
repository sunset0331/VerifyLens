"""
document/field_extractor.py
-----------------------------
VLM-based structured field extraction using a QLoRA fine-tuned
Qwen2-VL-2B-Instruct model.

Given a document image + a question, returns a structured JSON answer.
Used as a second-pass extractor after OCR, especially for name and
address fields where regex fails.

The model is loaded with 4-bit quantization (BitsAndBytes) to run
on consumer hardware (< 6GB VRAM / unified memory).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)

from src.utils.image_utils import load_image, ImageInput
from src.utils.config import config


# ── Prompt template ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a document intelligence assistant. "
    "Given an image of an identity document and a question, "
    "respond with ONLY a valid JSON object containing the answer. "
    'Example: {"name": "Utkarsh Gaur"} or {"dob": "15/08/1998"}. '
    "If the field is not visible, respond with null for the value."
)

FIELD_QUESTIONS = {
    "name": "What is the full name printed on this document?",
    "dob": "What is the date of birth on this document? Return in DD/MM/YYYY format.",
    "doc_number": "What is the document ID number or Aadhaar/PAN/Passport number?",
    "doc_type": "What type of identity document is this?",
    "gender": "What is the gender of the person on this document?",
    "address": "What is the address printed on this document, if any?",
}


class VLMFieldExtractor:
    """
    Fine-tuned Qwen2-VL field extractor for identity documents.

    Parameters
    ----------
    model_path : str or Path
        HuggingFace model ID or local checkpoint path.
        Defaults to the fine-tuned checkpoint from training_config.yaml,
        falls back to the base model if checkpoint not found.
    device : str, optional
        "cuda", "mps", or "cpu". Auto-detected if not specified.

    Example
    -------
    >>> extractor = VLMFieldExtractor()
    >>> fields = extractor.extract_all("path/to/aadhaar.jpg")
    >>> print(fields)
    {"name": "Ravi Sharma", "dob": "23/04/1990", "doc_number": "1234 5678 9012", ...}
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        # Resolve model path: prefer fine-tuned checkpoint, fall back to base
        cfg = config.vlm
        if model_path is None:
            checkpoint_dir = Path("checkpoints/qwen2vl-verifylens")
            model_path = str(checkpoint_dir) if checkpoint_dir.exists() else cfg.name

        print(f"[VLMFieldExtractor] Loading model from: {model_path}")
        self._load_model(str(model_path), cfg)

    def _load_model(self, model_path: str, cfg):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.use_nested_quant,
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

    @torch.no_grad()
    def ask(
        self,
        image: ImageInput,
        question: str,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Ask a free-form question about the document image.

        Returns raw model output string (JSON expected but not enforced).
        """
        img = load_image(image)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question},
                ],
            },
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_input],
            images=[img],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # Trim the input tokens from output
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def _parse_json_output(self, raw: str) -> Optional[Any]:
        """Extract the first JSON object from model output."""
        # Try direct parse
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
        return None

    def extract_field(
        self, image: ImageInput, field: str
    ) -> Optional[str]:
        """
        Extract a single field from a document image.

        Parameters
        ----------
        field : str
            One of: name, dob, doc_number, doc_type, gender, address
        """
        if field not in FIELD_QUESTIONS:
            raise ValueError(f"Unknown field '{field}'. Choose from: {list(FIELD_QUESTIONS)}")

        raw = self.ask(image, FIELD_QUESTIONS[field])
        parsed = self._parse_json_output(raw)

        if isinstance(parsed, dict):
            return parsed.get(field) or next(iter(parsed.values()), None)
        return raw if raw else None

    def extract_all(self, image: ImageInput) -> Dict[str, Optional[str]]:
        """
        Extract all standard fields from a document image.

        Makes one VLM call per field. For production, consider batching.

        Returns
        -------
        dict with keys: name, dob, doc_number, doc_type, gender, address
        """
        results = {}
        for field in FIELD_QUESTIONS:
            results[field] = self.extract_field(image, field)
        return results
