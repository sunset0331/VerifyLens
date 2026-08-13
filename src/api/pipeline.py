"""
api/pipeline.py
---------------
End-to-end Verification Pipeline that orchestrates all ML models.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Any

from PIL import Image

from src.document.classifier import DocumentClassifier
from src.document.ocr import OCRExtractor
from src.document.field_extractor import VLMFieldExtractor
from src.face.detector import FaceDetector
from src.face.embedder import FaceEmbedder
from src.face.matcher import FaceMatcher
from src.utils.config import config


class VerificationPipeline:
    """
    Singleton orchestrator for all ML models.
    Lazy-loads models on first request to speed up server boot time.
    """

    def __init__(self):
        self._doc_classifier = None
        self._ocr = None
        self._vlm = None
        self._face_detector = None
        self._face_embedder = None
        self._face_matcher = FaceMatcher()
        self.config = config

    def _init_document_models(self):
        if not self._doc_classifier:
            self._doc_classifier = DocumentClassifier()
        if not self._ocr:
            self._ocr = OCRExtractor()
        if not self._vlm:
            self._vlm = VLMFieldExtractor()

    def _init_face_models(self):
        if not self._face_detector:
            self._face_detector = FaceDetector()
        if not self._face_embedder:
            self._face_embedder = FaceEmbedder()

    async def run(self, id_image: Image.Image, selfie_image: Image.Image) -> Dict[str, Any]:
        """
        Run the full KYC pipeline on the given document and selfie.
        """
        # Run synchronous ML tasks in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        
        # 1. Initialize models (blocking, but only happens once)
        self._init_document_models()
        self._init_face_models()

        # 2. Document Classification
        doc_type, doc_scores = await loop.run_in_executor(
            None, self._doc_classifier.classify, id_image
        )

        # 3. Start Face Verification concurrently (independent of document text)
        face_task = loop.run_in_executor(
            None, 
            self._face_matcher.verify_pipeline, 
            self._face_detector, 
            self._face_embedder, 
            id_image, 
            selfie_image
        )

        # 4. OCR (fast)
        ocr_fields = await loop.run_in_executor(
            None, self._ocr.extract, id_image, doc_type
        )
        
        raw_text = ocr_fields.get("raw_text", "")

        # 5. VLM Extraction (using OCR text, sequential after OCR)
        vlm_fields = await loop.run_in_executor(
            None, self._vlm.extract_all, raw_text
        )

        # 6. Await Face Verification
        face_result = await face_task

        # 7. Fusion
        # Prefer VLM fields, fallback to OCR fields
        final_fields = {}
        for key in ["name", "dob", "doc_number", "gender", "address"]:
            val = vlm_fields.get(key)
            if not val and key in ocr_fields:
                val = ocr_fields[key]
            final_fields[key] = val
        
        # Verdict logic
        face_match = face_result.get("match", False)
        face_score = face_result.get("score", 0.0)
        
        # Simple rule engine:
        # If face matches and we extracted a document number and name, it's APPROVED
        # If face matches but missing fields, MANUAL_REVIEW
        # If face fails matching, REJECTED
        if not face_match:
            verdict = "REJECTED"
        elif final_fields.get("doc_number") and final_fields.get("name"):
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
            "extracted_fields": final_fields,
            "face_match": face_match,
            "face_score": face_score,
            "error_message": face_result.get("error")
        }


pipeline = VerificationPipeline()
