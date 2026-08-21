"""
api/models.py
-------------
Pydantic schemas for the FastAPI server.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class VerificationResponse(BaseModel):
    """Overall response for the KYC pipeline."""

    status: str = Field(..., description="'success' or 'error'")
    verdict: str = Field(..., description="'APPROVED', 'REJECTED', or 'MANUAL_REVIEW'")
    confidence: float = Field(..., description="Overall confidence score (0 to 1)")

    # Document classification
    doc_type: Optional[str] = Field(None, description="Detected document type")
    doc_scores: Optional[Dict[str, float]] = Field(
        None, description="Confidence for each document class"
    )

    # Extracted document fields (from whichever extractor was active)
    extracted_fields: Optional[Dict[str, Optional[str]]] = Field(
        None, description="Structured fields extracted from the document"
    )

    # Extraction transparency — always present so the caller knows exactly
    # which model produced the result and whether anything went wrong
    extraction_mode: Optional[str] = Field(
        None, description="'ocr_llm' (Approach 1) or 'vlm' (Approach 2)"
    )
    extractor_fallback: Optional[bool] = Field(
        None,
        description="True if OCR+LLM path ran without its LoRA adapter",
    )
    model_used: Optional[str] = Field(
        None, description="Exact model identifier that produced extracted_fields"
    )
    extraction_json_valid: Optional[bool] = Field(
        None, description="Whether the extractor returned parseable JSON"
    )
    extraction_latency_ms: Optional[float] = Field(
        None, description="Extraction inference latency in milliseconds"
    )
    extraction_error: Optional[str] = Field(
        None, description="Non-null if extraction failed entirely"
    )

    # Face match details
    face_match: Optional[bool] = Field(None, description="Did the face match the selfie?")
    face_score: Optional[float] = Field(
        None, description="Cosine similarity score for face matching"
    )

    error_message: Optional[str] = Field(
        None, description="Error details if status is 'error'"
    )
