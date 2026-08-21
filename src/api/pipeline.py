"""
api/pipeline.py
---------------
End-to-end Verification Pipeline that orchestrates all ML models.

Extraction mode is controlled by config.extraction.mode:
  "ocr_llm"  →  OCRLLMExtractor  (Approach 1: image → PaddleOCR → text LLM → JSON)
  "vlm"      →  VLMExtractor     (Approach 2: image → VLM → JSON)

Face verification is independent of the extraction mode and unchanged.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from PIL import Image

from src.document.classifier import DocumentClassifier
from src.document.extractor_router import get_extractor
from src.face.detector import FaceDetector
from src.face.embedder import FaceEmbedder
from src.face.matcher import FaceMatcher
from src.utils.config import config


class VerificationPipeline:
    """
    Singleton orchestrator for all ML models.
    Lazy-loads models on first request to speed up server boot time.

    Document extraction is routed via extraction_mode:
      "ocr_llm"  →  OCRLLMExtractor  (Approach 1)
      "vlm"      →  VLMExtractor     (Approach 2)

    The extraction mode is read from config at first call to
    _init_document_models().  To switch modes, restart the server.
    """

    def __init__(self):
        self._doc_classifier = None
        self._extractor = None           # DocumentExtractor (OCRLLMExtractor or VLMExtractor)
        self._face_detector = None
        self._face_embedder = None
        self._face_matcher = FaceMatcher()
        self.config = config

    def _init_document_models(self):
        if not self._doc_classifier:
            self._doc_classifier = DocumentClassifier()
        if not self._extractor:
            mode = getattr(getattr(self.config, "extraction", None), "mode", "ocr_llm")
            print(f"[Pipeline] Initializing document extractor: mode='{mode}'")
            self._extractor = get_extractor(mode=mode)
            print(f"[Pipeline] Document extractor ready: {self._extractor.model_name}")

    def _init_face_models(self):
        if not self._face_detector:
            self._face_detector = FaceDetector()
        if not self._face_embedder:
            self._face_embedder = FaceEmbedder()

    async def run(self, id_image: Image.Image, selfie_image: Image.Image) -> Dict[str, Any]:
        """
        Run the full KYC pipeline on the given document and selfie.

        Returns
        -------
        dict with keys:
            status, verdict, confidence, doc_type, doc_scores,
            extracted_fields, extraction_mode, extractor_fallback, model_used,
            face_match, face_score, error_message
        """
        loop = asyncio.get_event_loop()

        # 1. Initialize models (blocking, only happens once)
        self._init_document_models()
        self._init_face_models()

        # 2. Document Classification
        doc_type, doc_scores = await loop.run_in_executor(
            None, self._doc_classifier.classify, id_image
        )

        # 3. Start Face Verification concurrently (independent of document extraction)
        face_task = loop.run_in_executor(
            None,
            self._face_matcher.verify_pipeline,
            self._face_detector,
            self._face_embedder,
            id_image,
            selfie_image,
        )

        # 4. Document Field Extraction
        #    BOTH approaches receive the original id_image.
        #    Approach 1 (ocr_llm): internally calls OCR then passes text to LLM.
        #    Approach 2 (vlm):     passes the image directly to the VLM.
        extraction_result = await loop.run_in_executor(
            None, self._extractor.extract, id_image
        )

        # 5. Await Face Verification
        face_result = await face_task

        # 6. Build final fields from ExtractionResult
        #    No fusion with OCR fallback here — each approach stands alone.
        extracted_fields = extraction_result.fields.to_dict()

        # 7. Verdict logic
        face_match = face_result.get("match", False)
        face_score = face_result.get("score", 0.0)

        if not face_match:
            verdict = "REJECTED"
        elif extracted_fields.get("doc_number") and extracted_fields.get("name"):
            verdict = "APPROVED"
        else:
            verdict = "MANUAL_REVIEW"

        overall_confidence = face_score if face_match else (1.0 - face_score)

        return {
            "status": "success",
            "verdict": verdict,
            "confidence": overall_confidence,
            "doc_type": doc_type,
            "doc_scores": doc_scores,
            "extracted_fields": extracted_fields,
            # Extraction transparency fields
            "extraction_mode": extraction_result.extraction_mode,
            "extractor_fallback": extraction_result.extractor_fallback,
            "model_used": extraction_result.model,
            "extraction_json_valid": extraction_result.json_valid,
            "extraction_latency_ms": extraction_result.latency_ms,
            "extraction_error": extraction_result.error,
            # Face
            "face_match": face_match,
            "face_score": face_score,
            "error_message": face_result.get("error"),
        }


pipeline = VerificationPipeline()

