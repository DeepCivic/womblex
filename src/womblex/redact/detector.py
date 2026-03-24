"""Redaction detection and masking.

Detects black rectangular regions (redactions) in document page images
using OpenCV contour detection. Masks redacted regions with white before
OCR to prevent the OCR engine from producing garbage text.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RedactionInfo:
    """A detected redaction region."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    page: int
    area_px: int


class RedactionDetector:
    """CV2 heuristic detection of redacted (blacked-out) regions.

    Isolated behind this class so the detection method can be
    swapped for a model-based approach later.
    """

    def __init__(
        self,
        threshold: int = 50,
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.9,
        min_aspect_ratio: float = 0.1,
        max_aspect_ratio: float = 10.0,
    ) -> None:
        self.threshold = threshold
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def detect(self, image: np.ndarray, page: int = 0) -> list[RedactionInfo]:
        """Detect redacted regions in a page image.

        Args:
            image: RGB or grayscale image as numpy array.
            page: Page number (for metadata).

        Returns:
            List of detected redaction regions.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_area = image.shape[0] * image.shape[1]
        redactions: list[RedactionInfo] = []

        for contour in contours:
            if self._is_redaction_candidate(contour, image_area):
                x, y, w, h = cv2.boundingRect(contour)
                redactions.append(
                    RedactionInfo(
                        bbox=(x, y, x + w, y + h),
                        page=page,
                        area_px=w * h,
                    )
                )

        return redactions

    def _is_redaction_candidate(self, contour: np.ndarray, image_area: int) -> bool:
        """Filter contours that look like redaction boxes."""
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # Area constraints relative to page size
        area_ratio = area / image_area
        if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
            return False

        # Aspect ratio: redaction boxes are typically wider than tall
        # but can be square-ish too — just reject extreme shapes
        if h == 0:
            return False
        aspect = w / h
        if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
            return False

        return True

    def mask(self, image: np.ndarray, redactions: list[RedactionInfo]) -> np.ndarray:
        """White-out redacted regions before OCR.

        Args:
            image: Original image (RGB or grayscale).
            redactions: Detected redaction regions.

        Returns:
            Copy of image with redacted regions replaced by white.
        """
        masked = image.copy()
        for r in redactions:
            x1, y1, x2, y2 = r.bbox
            masked[y1:y2, x1:x2] = 255
        return masked
