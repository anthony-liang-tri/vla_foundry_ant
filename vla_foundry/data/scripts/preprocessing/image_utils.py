import io
from typing import Tuple

import numpy as np
from PIL import Image

# Global JPEG encoder pool to avoid repeated PIL overhead
_jpeg_quality = 95


def init_jpeg_encoder(quality: int = 95):
    """Initialize global JPEG encoder settings."""
    global _jpeg_quality
    _jpeg_quality = quality


def image_to_bytes(
    image: np.ndarray, quality: int = None, target_size: tuple = (224, 224)
) -> Tuple[bytes, Tuple[int, int]]:
    """Optimized image to JPEG conversion with resize and minimal allocations."""
    if quality is None:
        quality = _jpeg_quality

    # Ensure uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    # Convert to PIL and resize to target size (224x224 for VLM models)
    pil_image = Image.fromarray(image)
    original_image_size = pil_image.size
    if target_size and pil_image.size != target_size:
        pil_image = pil_image.resize(target_size, Image.LANCZOS)  # High-quality resize

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), original_image_size
