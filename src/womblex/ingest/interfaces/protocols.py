"""Backend protocols for pluggable ingest components.

Each protocol defines the minimal interface a backend must satisfy.
Concrete implementations (PaddleOCRReader, YOLOLayoutAnalyzer,
preprocess_for_ocr) already conform — these protocols formalise the
contracts so that alternative backends (document-trained layout models,
dedicated HTR recognisers) can be injected without changing strategy code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# OCR reader
# ---------------------------------------------------------------------------


@dataclass
class OCRRegionResult:
    """A single text region from OCR, with four-corner bbox."""

    bbox: list[list[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    text: str
    confidence: float  # 0-1


@runtime_checkable
class OCRReader(Protocol):
    """Protocol for OCR backends.

    Any class with a ``readtext`` method returning EasyOCR-compatible
    tuples satisfies this protocol.  PaddleOCRReader is the default.
    """

    def readtext(
        self, img: np.ndarray
    ) -> list[tuple[list[list[int]], str, float]]:
        """Detect and recognise text in *img*.

        Returns list of ``(bbox, text, confidence)`` where bbox is
        four corner points and confidence is 0-1.
        """
        ...


# ---------------------------------------------------------------------------
# Layout analyser
# ---------------------------------------------------------------------------


@dataclass
class LayoutRegionResult:
    """A layout region with bounding box, label, and block type."""

    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) pixel coords
    label: str  # raw model class name
    block_type: str  # mapped womblex block_type
    confidence: float


@runtime_checkable
class LayoutAnalyzer(Protocol):
    """Protocol for layout analysis backends.

    Any class with an ``analyze`` method returning LayoutRegionResult-compatible
    objects satisfies this protocol.  YOLOLayoutAnalyzer is the default.
    """

    def analyze(
        self, img: np.ndarray, conf_threshold: float = 0.3
    ) -> list[LayoutRegionResult]:
        """Detect layout regions in *img*.

        Returns regions sorted top-to-bottom by y-coordinate.
        """
        ...


# ---------------------------------------------------------------------------
# Image preprocessor
# ---------------------------------------------------------------------------


@dataclass
class PreprocessResult:
    """Output of image preprocessing: the processed image and applied steps."""

    image: np.ndarray  # grayscale processed image
    steps: list[str]  # e.g. ["deskew", "otsu_binarise"]


@runtime_checkable
class Preprocessor(Protocol):
    """Protocol for image preprocessing backends.

    Any callable with signature ``(img) -> PreprocessResult`` or the
    module-level ``preprocess_for_ocr`` function (which returns a tuple)
    satisfies this protocol.
    """

    def __call__(self, img: np.ndarray) -> PreprocessResult:
        """Preprocess *img* for OCR.

        Returns a PreprocessResult with the processed grayscale image
        and a list of steps applied (for metadata/observability).
        """
        ...
