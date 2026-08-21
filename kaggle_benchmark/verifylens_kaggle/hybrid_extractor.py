"""
verifylens_kaggle/hybrid_extractor.py
-------------------------------------
Approach 3 (Hybrid: VLM doc_type + OCR+LoRA identity fields) for Kaggle GPU benchmark.

Pipeline:
1. Image → VLM (Qwen2.5-VL) → JSON (extracts doc_type)
2. Image → OCR (PaddleOCR) → LLM (Qwen2.5-1.5B + PEFT adapter) → JSON (extracts identity fields)
3. Merge results

Loaded once at construction. Never re-loads per sample to avoid OOM.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from PIL import Image

from verifylens_kaggle.llm_extractor import LLMExtractor
from verifylens_kaggle.vlm_extractor import VLMExtractor
from verifylens_kaggle.json_utils import make_empty_fields


class HybridExtractor:
    """
    Approach 3 for Kaggle GPU benchmark.
    """

    def __init__(
        self,
        vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        llm_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        llm_adapter_path: str = "adapters/verifylens-adapter-peft",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ):
        self._vlm_model_id = vlm_model_id
        self._llm_model_id = llm_model_id
        self._llm_adapter_path = llm_adapter_path
        
        # Instantiate the two extractors
        print("[HybridExtractor] Initializing VLM for doc_type classification...")
        self._vlm = VLMExtractor(
            model_id=vlm_model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        print(f"[HybridExtractor] Initializing OCR+LLM with PEFT ({llm_adapter_path}) for identity fields...")
        self._llm = LLMExtractor(
            model_id=llm_model_id,
            adapter_path=llm_adapter_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
    @property
    def model_info(self) -> Dict[str, str]:
        vlm_info = self._vlm.model_info
        llm_info = self._llm.model_info
        
        return {
            "mode": "hybrid",
            "vlm_model_id": vlm_info["model_id"],
            "llm_model_id": llm_info["model_id"],
            "llm_adapter": llm_info["adapter"],
            "vlm_device": vlm_info["device"],
            "llm_device": llm_info["device"],
            "production_equivalent": "VLM classification + OCR+LoRA extraction",
            "comparability_note": "Hybrid approach using PEFT adapter for LLM and VLM for classification."
        }
        
    def extract(self, image: Image.Image) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "prediction": make_empty_fields(),
            "json_valid": False,
            "latency_ms": None,
            "vlm_latency_ms": None,
            "llm_latency_ms": None,
            "ocr_latency_ms": None,
            "error": None,
            "parse_error": None,
        }
        
        t_start = time.perf_counter()
        
        # 1. Run VLM
        try:
            vlm_res = self._vlm.extract(image)
        except Exception as e:
            result["error"] = f"VLM failed: {e}"
            result["latency_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            return result
            
        # 2. Run OCR+LLM
        try:
            llm_res = self._llm.extract(image)
        except Exception as e:
            result["error"] = f"LLM failed: {e}"
            result["latency_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            return result
            
        # 3. Merge
        vlm_pred = vlm_res.get("prediction", {}) or {}
        llm_pred = llm_res.get("prediction", {}) or {}
        
        merged_pred = make_empty_fields()
        
        # Doc Type from VLM
        if "document_type" in vlm_pred:
            merged_pred["document_type"] = vlm_pred["document_type"]
            
        # Identity fields from LLM
        for field in ["name", "dob", "document_number"]:
            if field in llm_pred:
                merged_pred[field] = llm_pred[field]
                
        # Both must be valid for the whole to be considered strictly valid json format
        json_valid = vlm_res.get("json_valid", False) and llm_res.get("json_valid", False)
        
        # Error propagation: if either VLM or LLM had an internal error that didn't throw an exception but returned an error string, capture it
        combined_error = None
        if vlm_res.get("error"):
            combined_error = f"VLM Error: {vlm_res['error']}"
        if llm_res.get("error"):
            llm_err = f"LLM Error: {llm_res['error']}"
            combined_error = f"{combined_error} | {llm_err}" if combined_error else llm_err
            
        result.update({
            "prediction": merged_pred,
            "json_valid": json_valid,
            "latency_ms": round((time.perf_counter() - t_start) * 1000, 1),
            "vlm_latency_ms": vlm_res.get("latency_ms"),
            "llm_latency_ms": llm_res.get("llm_latency_ms"),
            "ocr_latency_ms": llm_res.get("ocr_latency_ms"),
            "error": combined_error,
            "parse_error": f"VLM: {vlm_res.get('parse_error')} | LLM: {llm_res.get('parse_error')}",
            "raw_output": f"VLM:\n{vlm_res.get('raw_output')}\n\nLLM:\n{llm_res.get('raw_output')}"
        })
        
        return result
