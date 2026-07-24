"""
face/matcher.py
----------------
Cosine similarity matching logic for face embeddings.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from src.utils.config import config


class FaceMatcher:
    """
    Compares two face embeddings using cosine similarity.
    """

    def __init__(self, threshold: float = None):
        self.threshold = threshold if threshold is not None else config.face.similarity_threshold

    def match(self, emb1: torch.Tensor, emb2: torch.Tensor) -> Tuple[bool, float]:
        """
        Compare two normalized embeddings.

        Returns
        -------
        is_match : bool
        score : float (0 to 1, higher is more similar)
        """
        # Ensure embeddings are on the same device and 1D
        emb1 = emb1.view(-1).to(emb2.device)
        emb2 = emb2.view(-1)

        # Since embeddings are already L2-normalized, cosine similarity is just the dot product
        score = torch.dot(emb1, emb2).item()
        
        # Clamp between 0 and 1 just in case
        score = max(0.0, min(1.0, score))

        is_match = score >= self.threshold
        return is_match, round(score, 4)

    def verify_pipeline(
        self,
        detector,
        embedder,
        img1,
        img2
    ) -> Dict:
        """
        End-to-end face verification: detect -> crop -> embed -> match.
        """
        face1 = detector.crop_face(img1)
        if face1 is None:
            return {"match": False, "score": 0.0, "error": "No face detected in image 1"}

        face2 = detector.crop_face(img2)
        if face2 is None:
            return {"match": False, "score": 0.0, "error": "No face detected in image 2"}

        emb1 = embedder.get_embedding(face1)
        emb2 = embedder.get_embedding(face2)

        is_match, score = self.match(emb1, emb2)
        return {
            "match": is_match,
            "score": score,
            "error": None
        }
