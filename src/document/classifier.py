"""
document/classifier.py
-----------------------
CLIP-based zero-shot document type classifier.

Uses a ViT-L/14 CLIP model to score an input document image against
natural-language class descriptions and returns the most likely doc type.

Supported classes: aadhaar, pan, passport, driving_license
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import open_clip
from PIL import Image

from src.utils.image_utils import load_image, ImageInput
from src.utils.config import config


# ── Class prompts ────────────────────────────────────────────────────────────
# Multiple prompts per class → ensemble scoring for robustness

DOC_PROMPTS: Dict[str, List[str]] = {
    "aadhaar": [
        "an aadhaar card issued by the government of india",
        "aadhaar identity card with 12 digit number",
        "UIDAI aadhaar card document",
    ],
    "pan": [
        "a PAN card issued by the income tax department of india",
        "permanent account number card",
        "indian pan card with 10 character alphanumeric number",
    ],
    "passport": [
        "an indian passport document",
        "international travel passport booklet cover",
        "passport with photo and machine readable zone",
    ],
    "driving_license": [
        "an indian driving license",
        "motor vehicle driving licence issued by RTO",
        "driving licence with vehicle class and validity date",
    ],
}


class DocumentClassifier:
    """
    Zero-shot document type classifier using CLIP.

    Example
    -------
    >>> clf = DocumentClassifier()
    >>> label, scores = clf.classify("path/to/doc.jpg")
    >>> print(label)   # "aadhaar"
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: Optional[str] = None,
    ):
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Pre-compute text embeddings for all class prompts
        self._text_embeddings, self._labels = self._encode_text_prompts()

    @torch.no_grad()
    def _encode_text_prompts(self) -> Tuple[torch.Tensor, List[str]]:
        """Encode all class prompts and average embeddings per class."""
        labels = []
        class_embeddings = []

        for label, prompts in DOC_PROMPTS.items():
            tokens = self.tokenizer(prompts).to(self.device)
            emb = self.model.encode_text(tokens)          # (num_prompts, D)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            class_emb = emb.mean(dim=0)                   # (D,)
            class_emb = class_emb / class_emb.norm()
            class_embeddings.append(class_emb)
            labels.append(label)

        return torch.stack(class_embeddings), labels      # (C, D), [C]

    @torch.no_grad()
    def classify(
        self, image: ImageInput
    ) -> Tuple[str, Dict[str, float]]:
        """
        Classify a document image.

        Parameters
        ----------
        image : ImageInput
            File path, bytes, PIL Image, or numpy array.

        Returns
        -------
        predicted_label : str
            Most likely document class.
        scores : dict[str, float]
            Softmax confidence for each class.
        """
        img = load_image(image)
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        img_emb = self.model.encode_image(tensor)         # (1, D)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        # Cosine similarities → softmax
        sims = (img_emb @ self._text_embeddings.T).squeeze(0)  # (C,)
        probs = sims.softmax(dim=-1).cpu().tolist()

        scores = {label: round(p, 4) for label, p in zip(self._labels, probs)}
        predicted = max(scores, key=scores.get)
        return predicted, scores
