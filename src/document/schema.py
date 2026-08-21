"""
src/document/schema.py
-----------------------
Common document extraction schema shared by BOTH extraction approaches:

  Approach 1:  Document Image → PaddleOCR → text LLM → DocumentFields
  Approach 2:  Document Image → VLM        → DocumentFields

Every extractor must validate its output against this schema.
Missing or unreadable fields must be null, not fabricated.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class DocumentFields(BaseModel):
    """
    Structured identity-document fields extracted from a document image.

    All fields are Optional[str].  A null value explicitly means the field
    was not found or could not be reliably read — it is NOT the same as
    the extractor failing to run.

    Supported document types (Indian ID documents):
        aadhaar, pan, passport, driving_license
    """

    name: Optional[str] = Field(None, description="Full name of the document holder")
    dob: Optional[str] = Field(None, description="Date of birth (DD/MM/YYYY preferred)")
    doc_number: Optional[str] = Field(None, description="Primary document identifier number")
    doc_type: Optional[str] = Field(None, description="Document type label, e.g. 'Aadhaar Card'")
    gender: Optional[str] = Field(None, description="Gender as stated on the document")
    address: Optional[str] = Field(None, description="Address as printed on the document (if present)")

    @field_validator("*", mode="before")
    @classmethod
    def empty_string_to_none(cls, v: Any) -> Any:
        """Treat empty strings as null rather than letting them propagate."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Return fields as a plain dict (null values included)."""
        return self.model_dump()

    def non_null_fields(self) -> Dict[str, str]:
        """Return only fields that are not null."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def is_empty(self) -> bool:
        """True if every field is null (extraction produced nothing useful)."""
        return all(v is None for v in self.model_dump().values())


class ExtractionResult(BaseModel):
    """
    Wrapper returned by every extractor.

    Contains the structured fields plus extraction metadata.
    This keeps DocumentFields clean while giving the API enough
    information to be transparent about what actually happened.
    """

    fields: DocumentFields = Field(
        default_factory=DocumentFields,
        description="Extracted and schema-validated document fields",
    )
    extraction_mode: str = Field(
        ..., description="'ocr_llm' or 'vlm' — which extractor produced this result"
    )
    model: str = Field(..., description="Exact model identifier used")
    base_model: Optional[str] = None
    adapter_loaded: Optional[bool] = None
    json_valid: bool = Field(
        ..., description="Whether the raw model output was valid parseable JSON"
    )
    extractor_fallback: bool = Field(
        False,
        description="True if a fallback path was taken (e.g. adapter missing)",
    )
    latency_ms: Optional[float] = Field(
        None, description="Inference latency in milliseconds (model only, not load time)"
    )
    parse_error: Optional[str] = Field(
        None, description="JSON parse error message if json_valid is False"
    )
    error: Optional[str] = Field(
        None,
        description="Non-null if extraction failed entirely (e.g. model crash)",
    )
    raw_output: Optional[str] = Field(
        None, description="Raw string output from the model (for debugging)"
    )
