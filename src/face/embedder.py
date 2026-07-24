"""
face/embedder.py
-----------------
Face embedding using InceptionResnetV1 from facenet-pytorch.
Given a cropped face image, returns a 512D embedding vector.
"""

from __future__ import annotations

from typing import Optional

import torch
from PIL import Image

try:
    from facenet_pytorch import InceptionResnetV1
    from torchvision import transforms
except ImportError:
    InceptionResnetV1 = None
    transforms = None

from src.utils.image_utils import load_image, ImageInput


class FaceEmbedder:
    """
    InceptionResnetV1-based face embedder (pretrained on vggface2).

    Example
    -------
    >>> embedder = FaceEmbedder()
    >>> emb = embedder.get_embedding(face_img)
    """

    def __init__(self, device: Optional[str] = None):
        if InceptionResnetV1 is None:
            raise ImportError("facenet-pytorch and torchvision are required.")

        self.device = device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # Standard preprocessing for facenet-pytorch
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    @torch.no_grad()
    def get_embedding(self, face_image: ImageInput) -> torch.Tensor:
        """
        Get a normalized 512D face embedding.

        Parameters
        ----------
        face_image : ImageInput
            Already cropped face image.

        Returns
        -------
        torch.Tensor of shape (512,)
        """
        img = load_image(face_image)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        emb = self.model(tensor).squeeze(0)  # (512,)
        # Normalize the embedding
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb
