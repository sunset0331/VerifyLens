"""
src/document/base_extractor.py
--------------------------------
Abstract base class for both document extraction approaches.

Approach 1 (OCRLLMExtractor): image → PaddleOCR → text LLM → DocumentFields
Approach 2 (VLMExtractor):    image → VLM → DocumentFields

Both subclasses must:
  - accept a PIL Image
  - return an ExtractionResult
  - never load models inside the extraction method
  - be safe to call repeatedly with the same loaded model
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from src.document.schema import ExtractionResult


class DocumentExtractor(ABC):
    """
    Abstract base for all document field extractors.

    Subclasses implement `extract()` to accept a PIL Image and return
    an ExtractionResult containing DocumentFields plus metadata.
    """

    @abstractmethod
    def extract(self, image: Image.Image) -> ExtractionResult:
        """
        Extract identity fields from a document image.

        Parameters
        ----------
        image : PIL.Image.Image
            The original document image in RGB mode.

        Returns
        -------
        ExtractionResult
            Structured fields + extraction metadata.
            On total failure, result.error is non-null and result.fields
            contains all-null DocumentFields.
        """
        ...

    @property
    @abstractmethod
    def extraction_mode(self) -> str:
        """Returns 'ocr_llm' or 'vlm'."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the exact model identifier used by this extractor."""
        ...
