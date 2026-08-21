"""
src/document/hybrid/merger.py
-----------------------------
Merges OCR+LoRA extraction result with VLM doc_type classification.
"""

from typing import Optional
import json

from src.document.schema import ExtractionResult, DocumentFields

def merge(
    ocr_result: ExtractionResult,
    vlm_doc_type: Optional[str],
    vlm_latency_ms: float,
    total_latency_ms: float,
) -> ExtractionResult:
    """
    Merges VLM classification with OCR+LoRA extraction.
    
    Rules:
    - name, dob, doc_number, gender, address come from OCR+LoRA.
    - doc_type comes from VLM.
    - If VLM classification fails (vlm_doc_type is None), preserve the VLM doc_type as None (or fallback to OCR? 
      "If OCR+LoRA fails: preserve the VLM doc_type if available. If VLM classification fails: doc_type = null").
    """
    # Create a new fields object with OCR data
    fields = DocumentFields(
        name=ocr_result.fields.name,
        dob=ocr_result.fields.dob,
        doc_number=ocr_result.fields.doc_number,
        gender=ocr_result.fields.gender,
        address=ocr_result.fields.address,
        doc_type=vlm_doc_type if vlm_doc_type is not None else None
    )

    # Note: If ocr_result failed completely (fields are all None), we still preserve vlm_doc_type as requested.

    # Build comprehensive metadata
    # The OCRLLMExtractor records its own latency inside ocr_result.latency_ms (this is the LLM part).
    # But wait, OCRLLMExtractor.latency_ms is just the LLM generation time. 
    # To keep things clean, we will just use the ocr_result.latency_ms as ocr_llm_latency_ms,
    # or rely on the hybrid extractor to measure the total time.

    # Modify raw output to reflect the merge for debugging
    try:
        merged_raw = fields.model_dump()
        raw_output_str = json.dumps(merged_raw)
    except Exception:
        raw_output_str = "{}"

    # We determine json_valid if both the OCR output was mostly valid and we successfully merged
    # Actually, the requirement says "Do not treat gender/address as meaningful... keep them in schema".
    # json_valid should represent if we successfully parsed the expected JSON structure.
    # Since we strictly control the merge, the final output is structurally valid.
    
    return ExtractionResult(
        fields=fields,
        extraction_mode="hybrid",
        model="Hybrid(Qwen2.5-VL + Qwen2.5-1.5B-LoRA)",
        base_model=ocr_result.base_model, # Track the base model used by OCR
        adapter_loaded=ocr_result.adapter_loaded,
        json_valid=ocr_result.json_valid, # Consider the OCR JSON validity as the primary structure validity
        extractor_fallback=ocr_result.extractor_fallback,
        latency_ms=total_latency_ms,
        parse_error=ocr_result.parse_error,
        raw_output=raw_output_str,
    )
