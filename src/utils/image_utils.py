"""
utils/image_utils.py — Shared image preprocessing helpers used across all modules.
"""

import io
import base64
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image, ImageOps, ExifTags


ImageInput = Union[str, Path, bytes, Image.Image, np.ndarray]


def load_image(source: ImageInput) -> Image.Image:
    """
    Load an image from a file path, URL, bytes, base64 string, PIL Image,
    or numpy array. Returns a PIL Image in RGB mode.
    """
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(source)
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    elif isinstance(source, str) and source.startswith("data:image"):
        # base64 data URI
        _, encoded = source.split(",", 1)
        img = Image.open(io.BytesIO(base64.b64decode(encoded)))
    else:
        img = Image.open(source)

    # Auto-rotate based on EXIF orientation
    img = _fix_exif_rotation(img)
    return img.convert("RGB")


def _fix_exif_rotation(img: Image.Image) -> Image.Image:
    """Correct image orientation based on EXIF metadata."""
    try:
        exif = img._getexif()
        if exif is None:
            return img
        orientation_key = next(
            k for k, v in ExifTags.TAGS.items() if v == "Orientation"
        )
        orientation = exif.get(orientation_key)
        rotations = {3: 180, 6: 270, 8: 90}
        if orientation in rotations:
            img = img.rotate(rotations[orientation], expand=True)
    except (AttributeError, StopIteration, TypeError):
        pass
    return img


def resize_with_pad(
    img: Image.Image,
    target_size: Tuple[int, int],
    fill_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Resize image to target_size preserving aspect ratio, padding with fill_color."""
    img.thumbnail(target_size, Image.LANCZOS)
    padded = Image.new("RGB", target_size, fill_color)
    offset = ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2)
    padded.paste(img, offset)
    return padded


def image_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL Image to a base64 data URI string."""
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def crop_region(
    img: Image.Image, box: Tuple[int, int, int, int], pad: int = 0
) -> Image.Image:
    """Crop a region from image with optional padding. box = (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    w, h = img.size
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return img.crop((x1, y1, x2, y2))
