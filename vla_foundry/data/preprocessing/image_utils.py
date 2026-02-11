import io
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

# Global JPEG encoder pool to avoid repeated PIL overhead
_jpeg_quality = 95


def init_jpeg_encoder(quality: int = 95):
    """Initialize global JPEG encoder settings."""
    global _jpeg_quality
    _jpeg_quality = quality


class ImageResizingMethod(Enum):
    CENTER_CROP = "center_crop"
    RESIZE_NO_CROP = "resize_no_crop"
    RESIZE_FIT = "resize_fit"


def resize_image(
    image: Union[np.ndarray, Image.Image],
    target_size: tuple,
    resize_method=ImageResizingMethod.CENTER_CROP,
    fill_color: tuple = (0, 0, 0),
) -> np.ndarray:
    """Resize image to target size according to the given `resize_method`."""
    # Calculate aspect-ratio-preserving dimensions
    is_pil = isinstance(image, Image.Image)

    pil_image = image if is_pil else Image.fromarray(image)
    target_width, target_height = target_size
    orig_width, orig_height = pil_image.size

    if resize_method == ImageResizingMethod.CENTER_CROP:
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
    elif resize_method == ImageResizingMethod.RESIZE_NO_CROP:
        # Directly resize to target dimensions (may distort)
        pil_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    elif resize_method == ImageResizingMethod.RESIZE_FIT:
        # Calculate scale to fit within target dimensions
        scale = min(target_width / orig_width, target_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Resize to fit within target
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        # Create canvas and paste centered
        result = Image.new(pil_image.mode, target_size, fill_color)
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        result.paste(pil_image, (paste_x, paste_y))
        pil_image = result
    else:
        raise ValueError(f"Unrecognized image resizing method: {resize_method}")

    return pil_image if is_pil else np.array(pil_image)


def depth_image_to_bytes(
    image: np.ndarray, target_size: Optional[Tuple[int, int]] = None
) -> Tuple[bytes, Tuple[int, int]]:
    """Convert depth image to PNG with uint16 format (millimeters)."""
    # Ensure uint16 format for depth (mm units)
    assert image.dtype == np.uint16, "depth images must use np.uint16"

    # Convert to PIL
    pil_image = Image.fromarray(image, mode="I;16")
    original_image_size = pil_image.size

    # Resize if needed
    if target_size is not None and pil_image.size != target_size:
        pil_image = resize_image(pil_image, target_size=target_size)

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue(), original_image_size


def image_to_bytes(
    image: np.ndarray,
    quality: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
    resize_method=ImageResizingMethod.CENTER_CROP,
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
    if target_size is not None and pil_image.size != target_size:
        pil_image = resize_image(pil_image, resize_method=resize_method, target_size=target_size)

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), original_image_size
