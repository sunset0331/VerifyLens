"""
api/models.py
-------------
Pydantic schemas for the FastAPI server.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class VerificationResponse(BaseModel):
    """Overall response for the KYC pipeline."""
    status: str = Field(..., description="'success' or 'error'")
    verdict: str = Field(..., description="'APPROVED', 'REJECTED', 'MANUAL_REVIEW'")
    confidence: float = Field(..., description="Overall confidence score (0 to 1)")
    
    # Document details
    doc_type: Optional[str] = Field(None, description="Detected document type")
    doc_scores: Optional[Dict[str, float]] = Field(None, description="Confidence for each document class")
    
    # Extracted fields
    extracted_fields: Optional[Dict[str, Optional[str]]] = Field(None, description="Merged fields from OCR and VLM")
    vlm_confidence: Optional[str] = Field(None, description="Any VLM metadata")

    # Face match details
    face_match: Optional[bool] = Field(None, description="Did the face match the selfie?")
    face_score: Optional[float] = Field(None, description="Cosine similarity score for face matching")
    
    error_message: Optional[str] = Field(None, description="Error details if status is error")
