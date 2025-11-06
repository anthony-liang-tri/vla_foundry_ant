import io
from typing import Tuple, Union

import numpy as np
from PIL import Image

# Global JPEG encoder pool to avoid repeated PIL overhead
_jpeg_quality = 95


def init_jpeg_encoder(quality: int = 95):
    """Initialize global JPEG encoder settings."""
    global _jpeg_quality
    _jpeg_quality = quality


def resize_image(image: Union[np.ndarray, Image.Image], target_size: tuple = (224, 224)) -> np.ndarray:
    """Resize image to target size while maintaining aspect ratio (center crop)."""
    # Calculate aspect-ratio-preserving dimensions
    is_pil = isinstance(image, Image.Image)

    pil_image = image if is_pil else Image.fromarray(image)
    target_width, target_height = target_size
    orig_width, orig_height = pil_image.size

    # Calculate scale to cover target dimensions (no black bars)
    scale = max(target_width / orig_width, target_height / orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    # Resize to cover the target area
    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

    # Center crop to exact target size
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    pil_image = pil_image.crop((left, top, right, bottom))

    return np.array(pil_image) if is_pil else pil_image


def image_to_bytes(
    image: np.ndarray, quality: int = None, target_size: tuple = (224, 224)
) -> Tuple[bytes, Tuple[int, int]]:
    """Optimized image to JPEG conversion with resize and minimal allocations."""
    if quality is None:
        quality = _jpeg_quality

    # Ensure uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    # Convert to PIL and resize to target size while maintaining aspect ratio (center crop)
    pil_image = Image.fromarray(image)
    original_image_size = pil_image.size
    if target_size and pil_image.size != target_size:
        pil_image = resize_image(pil_image, target_size)

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), original_image_size
