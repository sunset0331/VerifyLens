"""
src/document/hybrid/hybrid_extractor.py
---------------------------------------
Approach 3 Hybrid Extractor.
Implements DocumentExtractor interface.
"""

from __future__ import annotations

import time
from typing import Optional

from PIL import Image

from src.document.base_extractor import DocumentExtractor
from src.document.schema import DocumentFields, ExtractionResult
from src.document.ocr_llm_extractor import OCRLLMExtractor
from src.document.hybrid.vlm_classifier import VLMDocumentClassifier
from src.document.hybrid.merger import merge

class HybridExtractor(DocumentExtractor):
    """
    Hybrid extractor combining VLM classification and OCR+LoRA extraction.
    """

    _EXTRACTION_MODE = "hybrid"

    def __init__(
        self,
        vlm_model_path: str,
        vlm_max_tokens: int,
        vlm_temperature: float,
        vlm_resize_max_px: int,
        ocr_llm_model_path: str,
        ocr_llm_adapter_path: str,
        ocr_llm_max_tokens: int,
    ):
        self._vlm_model_path = vlm_model_path
        
        # [CRITICAL FIX] macOS Metal + fork() deadlock prevention:
        # PaddleOCR uses multiprocessing/fork internally during initialization.
        # If MLX initializes the Metal GPU backend before PaddleOCR forks, the child process deadlocks.
        # Therefore, we MUST eagerly initialize PaddleOCR before loading any MLX models.
        from src.document.ocr import OCRExtractor
        _ = OCRExtractor()._get_engine()
        
        # Instantiate VLM Classifier (Loads MLX)
        self._vlm = VLMDocumentClassifier(
            model_path=vlm_model_path,
            max_new_tokens=vlm_max_tokens,
            temperature=vlm_temperature,
            resize_max_px=vlm_resize_max_px,
        )

        # Instantiate OCR+LoRA Extractor
        self._ocr_llm = OCRLLMExtractor(
            model_path=ocr_llm_model_path,
            adapter_path=ocr_llm_adapter_path,
            max_tokens=ocr_llm_max_tokens,
        )

    @property
    def extraction_mode(self) -> str:
        return self._EXTRACTION_MODE

    @property
    def model_name(self) -> str:
        return f"Hybrid({self._vlm_model_path} + {self._ocr_llm.model_name})"

    def extract(self, image: Image.Image) -> ExtractionResult:
        """
        Executes hybrid pipeline:
        1. VLM Classification
        2. OCR+LoRA Extraction
        3. Merge
        """
        t0 = time.perf_counter()
        
        # 1. VLM Classification
        vlm_doc_type, vlm_latency_ms, vlm_error = self._vlm.classify(image)
        
        # 2. OCR+LoRA Extraction
        ocr_result = self._ocr_llm.extract(image)
        
        # 3. Merge
        total_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        
        final_result = merge(
            ocr_result=ocr_result,
            vlm_doc_type=vlm_doc_type,
            vlm_latency_ms=vlm_latency_ms,
            total_latency_ms=total_latency_ms,
        )
        
        # Append detailed latency metadata if not existing
        # (ExtractionResult model might reject unknown kwargs in earlier pydantic, 
        # so we ensure it's returned via the schema or log it.)
        
        return final_result
