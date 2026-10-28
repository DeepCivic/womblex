"""Page-image encoding shared by the OCR readers that send an image to a service."""

from __future__ import annotations

import io

import numpy as np


def img_to_png_bytes(img: np.ndarray) -> bytes:
    """Encode an RGB/grayscale numpy image as PNG bytes."""
    from PIL import Image

    if img.ndim == 2:
        pil = Image.fromarray(img, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(img[:, :, :3], mode="RGB")

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()
