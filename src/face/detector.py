"""
face/detector.py
-----------------
Face detection using MTCNN from facenet-pytorch.
Given an image, detects the most prominent face and returns the cropped face image.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from PIL import Image

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

from src.utils.image_utils import load_image, ImageInput


class FaceDetector:
    """
    MTCNN-based face detector.

    Example
    -------
    >>> detector = FaceDetector()
    >>> face_img = detector.crop_face("path/to/selfie.jpg")
    """

    def __init__(self, device: Optional[str] = None):
        if MTCNN is None:
            raise ImportError("facenet-pytorch is required. Run: pip install facenet-pytorch")

        self.device = device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.mtcnn = MTCNN(
            keep_all=False,       # Only return the single most probable face
            device=self.device,
            margin=20,            # Add a margin around the face
            min_face_size=40,
        )

    def crop_face(self, image: ImageInput) -> Optional[Image.Image]:
        """
        Detect face and return cropped PIL Image.
        Returns None if no face is found.
        """
        img = load_image(image)
        # MTCNN expects a PIL image and returns a cropped PIL image (or tensor if configured)
        # By default, if return_prob=False, it returns a tensor. But we can get bounding boxes.
        boxes, probs = self.mtcnn.detect(img)

        if boxes is None or len(boxes) == 0:
            return None

        # Take highest probability face
        box = boxes[0]
        x1, y1, x2, y2 = [int(b) for b in box]

        # Add margin
        margin = 20
        w, h = img.size
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        return img.crop((x1, y1, x2, y2))

    def has_face(self, image: ImageInput) -> bool:
        """Check if a face exists in the image."""
        img = load_image(image)
        boxes, _ = self.mtcnn.detect(img)
        return boxes is not None and len(boxes) > 0
