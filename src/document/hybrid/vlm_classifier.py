"""
src/document/hybrid/vlm_classifier.py
-------------------------------------
VLM Document Classifier for Approach 3 (Hybrid).
Receives a raw PIL image and returns only a classification JSON.
Must NOT receive OCR text.
"""

import json
import re
import time
from typing import Optional

from PIL import Image

_DEFAULT_VLM_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

_CLASSIFICATION_USER = (
    "Classify the identity document shown in the image.\n\n"
    "Return ONLY JSON:\n\n"
    "{\n"
    '  "doc_type": "..."\n'
    "}\n\n"
    "Allowed values:\n"
    "- PAN Card\n"
    "- Aadhaar Card\n"
    "- Passport\n\n"
    "If the document cannot be classified reliably, return null.\n\n"
    "Do not invent another category."
)

class VLMDocumentClassifier:
    """
    Classifies a document image into a strict schema.
    Uses mlx-vlm with Qwen2.5-VL.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_VLM_MODEL,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        resize_max_px: int = 1120,
    ):
        self._model_path = model_path
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._resize_max_px = resize_max_px

        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self) -> None:
        from mlx_vlm import load

        print(f"[VLMClassifier] Loading model: {self._model_path}")
        self._model, self._processor = load(self._model_path)
        print(f"[VLMClassifier] Model ready.")

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGB")
        w, h = img.size
        max_side = max(w, h)
        if max_side > self._resize_max_px:
            scale = self._resize_max_px / max_side
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img

    def classify(self, image: Image.Image) -> tuple[Optional[str], float, Optional[str]]:
        """
        Runs VLM classification on the image.

        Returns:
            (doc_type, latency_ms, error_msg)
        """
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        try:
            img = self._preprocess_image(image)
        except Exception as e:
            return None, 0.0, f"Image preprocessing failed: {e}"

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": _CLASSIFICATION_USER},
                    ],
                }
            ]
            formatted_prompt = apply_chat_template(
                self._processor,
                config=self._model.config if hasattr(self._model, "config") else None,
                prompt=_CLASSIFICATION_USER,
                num_images=1,
            )
        except Exception as e:
            return None, 0.0, f"Prompt construction failed: {e}"

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
            if hasattr(output, "text"):
                raw_output = output.text
            elif isinstance(output, str):
                raw_output = output
            else:
                raw_output = str(output)
        except Exception as e:
            return None, round((time.perf_counter() - t0) * 1000, 1), f"VLM generation failed: {e}"
        
        latency_ms = (time.perf_counter() - t0) * 1000

        # Parse output
        doc_type, error = self._parse_classification(raw_output)
        return doc_type, round(latency_ms, 1), error

    def _parse_classification(self, raw: str) -> tuple[Optional[str], Optional[str]]:
        text = raw.strip()

        # 1. Direct parse
        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # 2. Markdown fence
            fenced = re.sub(r"```(?:json)?\s*(.*?)```", r"\1", text, flags=re.DOTALL).strip()
            try:
                parsed = json.loads(fenced)
            except json.JSONDecodeError:
                # 3. Brace extract
                match = re.search(r"\{[^{}]*\}", fenced, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                    except json.JSONDecodeError as e:
                        return None, f"JSON block found but unparseable: {e}"
                else:
                    return None, f"No JSON object found: {repr(text[:200])}"

        if parsed is not None:
            doc_type = parsed.get("doc_type")
            if not doc_type:
                return None, None
            
            dt_lower = str(doc_type).lower()
            if "pan" in dt_lower:
                return "PAN Card", None
            elif "aadhaar" in dt_lower or "aadhar" in dt_lower or "aadhare" in dt_lower:
                return "Aadhaar Card", None
            elif "passport" in dt_lower:
                return "Passport", None
            else:
                return None, f"Invalid doc_type value: {doc_type}"

        return None, "Unexpected parsing failure"
