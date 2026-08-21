"""
Tests for Approach 3 Hybrid Extractor.
"""
from unittest.mock import patch, MagicMock
from PIL import Image

from src.document.hybrid.merger import merge
from src.document.schema import ExtractionResult, DocumentFields
from src.document.extractor_router import get_extractor

def test_merger_success():
    ocr_result = ExtractionResult(
        fields=DocumentFields(name="John Doe", dob="01/01/1990", doc_number="123", doc_type="Wrong Type"),
        extraction_mode="ocr_llm",
        model="Qwen2.5-1.5B",
        base_model="Qwen2.5",
        adapter_loaded=True,
        json_valid=True,
        extractor_fallback=False,
    )
    
    vlm_doc_type = "PAN Card"
    
    merged = merge(ocr_result, vlm_doc_type, 100.0, 500.0)
    
    assert merged.fields.name == "John Doe"
    assert merged.fields.doc_type == "PAN Card"
    assert merged.extraction_mode == "hybrid"
    assert merged.json_valid is True

def test_merger_vlm_failure():
    ocr_result = ExtractionResult(
        fields=DocumentFields(name="John Doe", dob="01/01/1990", doc_number="123", doc_type="Wrong Type"),
        extraction_mode="ocr_llm",
        model="Qwen2.5-1.5B",
        base_model="Qwen2.5",
        adapter_loaded=True,
        json_valid=True,
        extractor_fallback=False,
    )
    
    merged = merge(ocr_result, None, 100.0, 500.0)
    
    # VLM failed, so doc_type should be null (overwriting the OCR one, as VLM has precedence even in failure, per prompt: "doc_type = null")
    assert merged.fields.doc_type is None

def test_router_hybrid():
    with patch("src.document.hybrid.vlm_classifier.VLMDocumentClassifier._load_model"), \
         patch("src.document.ocr_llm_extractor.OCRLLMExtractor._load_llm"):
         
        extractor = get_extractor("hybrid")
        assert extractor.extraction_mode == "hybrid"
        assert "Hybrid" in extractor.model_name
