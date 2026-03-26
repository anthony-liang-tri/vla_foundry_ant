"""
Fast image decoding for WebDataset using OpenCV.

OpenCV's JPEG decoder is typically 2-3x faster than PIL.
This module provides a custom decoder for WebDataset that uses OpenCV.
"""

import io
import logging

import cv2
import numpy as np
import tifffile
from PIL import Image

logger = logging.getLogger(__name__)

# Disable OpenCV's internal threading to avoid contention with DataLoader workers
cv2.setNumThreads(0)


def fast_image_decoder(key: str, data: bytes):
    """
    WebDataset-compatible decoder function using OpenCV.

    Usage:
        wds.decode(fast_image_decoder, handler=log_and_continue)
    """
    # Check if this is an image file
    if key.endswith((".jpg", ".jpeg", ".png", ".webp")):
        try:
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
        except Exception:
            # If all decoding fails, return None and let webdataset handle it
            return None

    elif key.endswith((".tiff", ".tif")):
        try:
            # Decode TIFF files (typically point maps in uint16 format)
            # Use tifffile for robust TIFF decoding, especially for multi-channel uint16 TIFFs
            # Return numpy array directly - PIL can't handle multi-channel uint16
            img = tifffile.imread(io.BytesIO(data))

            # Validate the image data
            if not isinstance(img, np.ndarray):
                raise ValueError(f"tifffile returned non-array type: {type(img)}")

            # For multi-channel images, ensure correct shape
            if img.ndim == 3 and img.shape[2] > 4:
                # Unusual number of channels, might be packed differently
                logger.warning(f"TIFF has unusual shape {img.shape} for key {key}")

            # Return numpy array directly (PIL can't handle multi-channel uint16)
            # The robotics pipeline converts to numpy anyway, so this is fine
            return img
        except Exception as e:
            # If TIFF decoding fails, try PIL as fallback (will lose uint16 precision)
            logger.debug(f"tifffile decoding failed for {key}: {e}, trying PIL fallback")
            try:
                pil_img = Image.open(io.BytesIO(data))
                # Convert to numpy to match expected format
                return np.array(pil_img)
            except Exception as e2:
                # If all decoding fails, log and return None
                logger.warning(f"All TIFF decoding methods failed for {key}: tifffile={e}, PIL={e2}")
                return None

    else:
        return None
