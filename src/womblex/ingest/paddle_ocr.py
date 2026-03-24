"""PaddleOCR wrapper backed by rapidocr-onnxruntime.

Uses the ``rapidocr-onnxruntime`` package which bundles pre-exported
PaddleOCR v4 ONNX models (det + rec + cls).  No separate model download
required — models ship with the pip package (~15 MB wheel).

Layout analysis uses YOLOv8 via ``ultralytics`` with the bundled
``models/yolov8n.pt`` weight file.  COCO class names are mapped to
document block types via ``_YOLO_COCO_LABEL_MAP``.

Table recognition uses ``rapid-table`` (SLANet, model downloaded on first use).
Table recognition degrades gracefully if the SLANet model download fails
(e.g. air-gapped environments).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger(__name__)

# Tesseract-style lang code → RapidOCR language mapping.
_LANG_MAP: dict[str, str] = {
    "eng": "en",
    "fra": "french",
    "deu": "german",
    "spa": "es",
    "ita": "it",
    "chi_sim": "ch",
    "jpn": "japan",
    "kor": "korean",
}


@dataclass
class OCRRegion:
    """A single text region detected by PaddleOCR."""

    bbox: list[list[int]]  # four corner points [[x1,y1], ...]
    text: str
    confidence: float  # 0-1 scale


@dataclass
class LayoutRegion:
    """A layout region detected by YOLO."""

    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    label: str  # raw YOLO class name
    block_type: str  # mapped womblex block_type
    confidence: float


@dataclass
class TableStructure:
    """Table structure extracted by SLANet."""

    html: str
    headers: list[str]
    rows: list[list[str]]
    bbox: tuple[float, float, float, float]
    confidence: float


class PaddleOCRReader:
    """OCR reader backed by rapidocr-onnxruntime.

    Lazily initialises the RapidOCR engine on first call.  The engine
    bundles PaddleOCR v4 det + rec + cls ONNX models so no external
    model files are needed.
    """

    def __init__(self, lang: str = "en", use_int8: bool = True) -> None:
        self.lang = lang
        self.use_int8 = use_int8
        self._engine: RapidOCR | None = None

    def _ensure_loaded(self) -> None:
        """Initialise RapidOCR engine if not already loaded."""
        if self._engine is not None:
            return

        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()
        logger.info("RapidOCR (PaddleOCR ONNX) loaded for lang=%s", self.lang)

    def readtext(self, img: np.ndarray) -> list[tuple[list[list[int]], str, float]]:
        """Detect and recognise text, returning EasyOCR-compatible tuples.

        Returns list of ``(bbox, text, confidence)`` where bbox is
        ``[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]`` and confidence is 0-1.
        """
        if img is None or img.size == 0:
            return []

        self._ensure_loaded()
        assert self._engine is not None

        result, _elapse = self._engine(img)
        if not result:
            return []

        output: list[tuple[list[list[int]], str, float]] = []
        for bbox_points, text, confidence in result:
            # RapidOCR returns bbox as list of [x, y] float pairs — cast to int
            bbox = [[int(p[0]), int(p[1])] for p in bbox_points]
            output.append((bbox, text, float(confidence)))

        return output


class TableRecognizer:
    """SLANet table structure recognition via rapid-table.

    The rapid-table package downloads its ONNX model on first use.
    If the download fails (e.g. air-gapped environment), ``recognize()``
    returns None and callers fall back to heuristic table extraction.
    """

    def __init__(self) -> None:
        self._engine: object | None = None
        self._available: bool | None = None

    def _ensure_loaded(self) -> None:
        if self._available is not None:
            return

        try:
            from rapid_table import RapidTable  # type: ignore[import-untyped]

            self._engine = RapidTable()
            self._available = True
            logger.info("rapid-table loaded for SLANet table recognition")
        except Exception:
            self._available = False
            logger.warning(
                "rapid-table model unavailable (download may have failed) "
                "— table recognition disabled, falling back to heuristics"
            )

    def recognize(
        self, img: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> TableStructure | None:
        """Recognise table structure within a bounding box region."""
        self._ensure_loaded()
        if not self._available or self._engine is None:
            return None

        x0, y0, x1, y1 = [int(v) for v in bbox]
        h, w = img.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)

        if x1 - x0 < 10 or y1 - y0 < 10:
            return None

        crop = img[y0:y1, x0:x1]
        html, _elapse = self._engine(crop)  # type: ignore[union-attr]
        if not html:
            return None

        # Parse headers/rows from HTML if available
        headers, rows = _parse_table_html(html)
        return TableStructure(
            html=html, headers=headers, rows=rows, bbox=(float(x0), float(y0), float(x1), float(y1)), confidence=0.8,
        )


def _parse_table_html(html: str) -> tuple[list[str], list[list[str]]]:
    """Extract headers and rows from a simple HTML table string."""
    import re

    headers: list[str] = []
    rows: list[list[str]] = []

    th_matches = re.findall(r"<th[^>]*>(.*?)</th>", html, re.DOTALL)
    if th_matches:
        headers = [re.sub(r"<[^>]+>", "", h).strip() for h in th_matches]

    for tr_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr_match.group(1), re.DOTALL)
        if cells:
            rows.append([re.sub(r"<[^>]+>", "", c).strip() for c in cells])

    return headers, rows


# YOLO COCO class name → womblex block_type mapping.
# The base YOLOv8n model uses 80 COCO classes, not document-specific labels.
# This maps COCO detections to the closest document layout equivalent.
# Classes not listed here default to "figure" (visual/non-text region).
_YOLO_COCO_LABEL_MAP: dict[str, str] = {
    # Objects that look like text/paragraph regions in document scans
    "person": "paragraph",
    "book": "paragraph",
    # Tabular structures
    "dining table": "table",
    # Screen/display objects → figure (likely embedded images or charts)
    "tv": "figure",
    "laptop": "figure",
    "cell phone": "figure",
    "monitor": "figure",
    # Office/print objects → figure
    "keyboard": "figure",
    "mouse": "figure",
    "scissors": "figure",
    "clock": "figure",
}


class YOLOLayoutAnalyzer:
    """Layout region detection via a local YOLOv8 model.

    Loads the pre-downloaded ``models/yolov8n.pt`` weight file to avoid
    runtime downloads.  Detected bounding boxes are mapped to womblex
    ``LayoutRegion`` objects using ``_YOLO_COCO_LABEL_MAP`` which translates
    COCO class names to document block types (e.g. ``dining table`` → ``table``,
    ``person``/``book`` → ``paragraph``, screen objects → ``figure``).

    Requires ``ultralytics`` to be installed (optional dependency).
    """

    def __init__(self, model_path: str | None = None) -> None:
        from pathlib import Path as _Path

        if model_path is None:
            from womblex.utils.models import resolve_local_model_path
            resolved = resolve_local_model_path("yolov8n.pt")
            self._model_path = str(resolved)
        else:
            self._model_path = str(_Path(model_path))

        self._engine: object | None = None

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]

            self._engine = YOLO(self._model_path)
            logger.info("YOLOv8 layout model loaded from %s", self._model_path)
        except ImportError as exc:
            raise ImportError(
                "YOLOLayoutAnalyzer requires 'ultralytics'. "
                "Install with: pip install ultralytics"
            ) from exc

    def analyze(self, img: np.ndarray, conf_threshold: float = 0.3) -> list[LayoutRegion]:
        """Detect layout regions using YOLOv8 inference.

        Args:
            img: RGB image as a numpy array.
            conf_threshold: Minimum detection confidence to include.

        Returns:
            List of detected layout regions sorted top-to-bottom.
        """
        self._ensure_loaded()
        assert self._engine is not None

        results = self._engine(img, conf=conf_threshold, verbose=False)  # type: ignore[operator]
        regions: list[LayoutRegion] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                cls_id = int(box.cls[0])
                x0, y0, x1, y1 = (float(v) for v in box.xyxy[0])
                label = result.names.get(cls_id, str(cls_id)) if result.names else str(cls_id)
                block_type = _YOLO_COCO_LABEL_MAP.get(label, "figure")
                regions.append(LayoutRegion(
                    bbox=(x0, y0, x1, y1),
                    label=label,
                    block_type=block_type,
                    confidence=conf,
                ))

        regions.sort(key=lambda r: r.bbox[1])
        return regions


# ------------------------------------------------------------------
# Module-level cache
# ------------------------------------------------------------------

_paddle_readers: dict[str, PaddleOCRReader] = {}
_layout_analyzer: YOLOLayoutAnalyzer | None = None
_table_recognizer: TableRecognizer | None = None


def get_paddle_reader(lang: str = "eng", use_int8: bool = True) -> PaddleOCRReader:
    """Return a cached PaddleOCR reader for the given Tesseract-style lang code."""
    mapped = _LANG_MAP.get(lang, lang)
    key = f"{mapped}_{use_int8}"
    if key not in _paddle_readers:
        _paddle_readers[key] = PaddleOCRReader(lang=mapped, use_int8=use_int8)
    return _paddle_readers[key]


def get_layout_analyzer() -> YOLOLayoutAnalyzer:
    """Return a cached YOLOv8 layout analyzer."""
    global _layout_analyzer
    if _layout_analyzer is None:
        _layout_analyzer = YOLOLayoutAnalyzer()
    return _layout_analyzer


def get_table_recognizer() -> TableRecognizer:
    """Return a cached table structure recognizer."""
    global _table_recognizer
    if _table_recognizer is None:
        _table_recognizer = TableRecognizer()
    return _table_recognizer


def preprocess_for_ocr(img: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Preprocess an image for OCR: grayscale, deskew, binarise.

    Pure image processing — no redaction (that's a separate pipeline stage).
    Returns the processed grayscale image and list of applied steps.
    """
    import cv2

    steps: list[str] = []

    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy() if img.ndim == 2 else img

    from womblex.ingest.heuristics_cv2 import detect_skew_angle
    skew = detect_skew_angle(gray)
    if abs(skew.angle) > 0.5 and skew.confidence > 0.3:
        h, w = gray.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), skew.angle, 1.0)
        gray = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        steps.append("deskew")

    # Skip binarisation for clean digital renders. A digital render has low
    # noise and moderate dynamic range (actual text present). Scanned images
    # and sparse formula/diagram images still benefit from binarisation.
    from womblex.ingest.heuristics_numpy import analyze_histogram, analyze_otsu_threshold
    hist = analyze_histogram(gray)
    if not hist.is_scanned and hist.dynamic_range > 0.1:
        steps.append("binarise_skipped")
    else:
        otsu = analyze_otsu_threshold(gray)
        if otsu.is_bimodal:
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            steps.append("otsu_binarise")
        else:
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
            steps.append("adaptive_binarise")

    return gray, steps
