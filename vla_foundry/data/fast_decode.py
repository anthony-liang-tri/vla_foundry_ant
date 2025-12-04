"""
Fast image decoding for WebDataset using OpenCV.

OpenCV's JPEG decoder is typically 2-3x faster than PIL.
This module provides a custom decoder for WebDataset that uses OpenCV.
"""

import io

import cv2
import numpy as np
from PIL import Image

# Disable OpenCV's internal threading to avoid contention with DataLoader workers
cv2.setNumThreads(0)


def fast_image_decoder(key: str, data: bytes):
    """
    WebDataset-compatible decoder function using OpenCV.

    Usage:
        wds.decode(fast_image_decoder, handler=log_and_continue)
    """
    # Check if this is an image file
    if not key.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return None

    # Decode with OpenCV
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        # Fallback to PIL if OpenCV fails
        return Image.open(io.BytesIO(data)).convert("RGB")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image for compatibility with torchvision transforms
    return Image.fromarray(img)
