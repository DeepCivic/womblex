"""PaddleOCR wrapper backed by rapidocr-onnxruntime.

Uses the ``rapidocr-onnxruntime`` package which bundles pre-exported
PaddleOCR v4 ONNX models (det + rec + cls).  No separate model download
required — models ship with the pip package (~15 MB wheel).

Layout analysis uses YOLOv8 via ``ultralytics`` with the bundled
``models/yolov8n.pt`` weight file.  COCO class names are mapped to
document block types via ``_YOLO_COCO_LABEL_MAP``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from womblex.ingest.interfaces.protocols import (
    LayoutRegionResult,
    OCRPageResult,
    OCRRegionResult,
)

if TYPE_CHECKING:
    from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference thread capping
# ---------------------------------------------------------------------------
#
# RapidOCR (onnxruntime) and the YOLO layout filter (torch) each default their
# thread pools to the full core count (onnxruntime config ships
# ``intra_op_num_threads = -1``; torch sizes to cpu_count). Loaded together in a
# per-page loop they spin up ~100 threads but contend for the cores, yielding
# little real parallelism (~1.2 cores of useful work) while thrashing on a
# low-core deployment target (the Chromebook profile this corpus targets).
#
# Capping makes CPU usage a deliberate, bounded choice. Default 4; override via
# the ``WOMBLEX_INFERENCE_THREADS`` env var or ``extraction.ocr.num_threads``
# (threaded in through :func:`set_inference_threads`). See docs/decisions.md.

_DEFAULT_INFERENCE_THREADS = 4
_inference_threads: int = int(
    os.environ.get("WOMBLEX_INFERENCE_THREADS", _DEFAULT_INFERENCE_THREADS)
)


def set_inference_threads(n: int | None) -> None:
    """Set the process-wide cap on OCR/layout inference threads.

    ``None`` (or a non-positive value) leaves the current value unchanged. The
    cap is applied lazily at model-construction time, so call this before the
    first OCR/layout op (extraction entry points do).
    """
    global _inference_threads
    if n is not None and n >= 1:
        _inference_threads = int(n)


def get_inference_threads() -> int:
    """Return the current inference thread cap."""
    return _inference_threads


def _apply_thread_env(n: int) -> None:
    """Cap BLAS / OpenMP thread pools (numpy, OpenCV, OMP-built onnxruntime).

    Sets the standard ``*_NUM_THREADS`` env vars unless the user already set
    them (an explicit user value wins). Must run *before* the heavy import so
    the pools size correctly at load.
    """
    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, str(n))

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

# Backward-compatible aliases — canonical definitions live in interfaces/protocols.py
OCRRegion = OCRRegionResult
LayoutRegion = LayoutRegionResult


class PaddleOCRReader:
    """OCR reader backed by rapidocr-onnxruntime.

    Prefers PaddleOCR v5 mobile models when found under
    ``<models_dir>/paddleocr-v5/`` (handwriting + better word segmentation
    than v4). Falls back to the v4 models bundled inside the
    ``rapidocr-onnxruntime`` wheel when v5 is not installed.
    """

    _V5_DIR = "paddleocr-v5"
    _V5_FILES: ClassVar[dict[str, str]] = {
        "det": "ppocrv5-mobile-det.onnx",
        "rec": "ppocrv5-mobile-rec.onnx",
        "cls": "ppocrv5-cls.onnx",
        "dict": "ppocrv5_dict.txt",
    }

    def __init__(self, lang: str = "en", use_int8: bool = True) -> None:
        self.lang = lang
        self.use_int8 = use_int8
        self._engine: RapidOCR | None = None

    def _resolve_v5_paths(self) -> dict[str, str] | None:
        from womblex.utils.models import resolve_local_model_path
        resolved = resolve_local_model_path(self._V5_DIR)
        if isinstance(resolved, str):
            return None
        paths: dict[str, str] = {}
        for key, fname in self._V5_FILES.items():
            p = resolved / fname
            if not p.is_file():
                logger.warning("PaddleOCR v5 file missing: %s — falling back to v4", p)
                return None
            paths[key] = str(p)
        return paths

    def _ensure_loaded(self) -> None:
        """Initialise RapidOCR engine if not already loaded."""
        if self._engine is not None:
            return

        n = get_inference_threads()
        _apply_thread_env(n)
        from rapidocr_onnxruntime import RapidOCR

        # Cap each onnxruntime session (det/cls/rec) — RapidOCR routes
        # `<model>_`-prefixed kwargs to that model's SessionOptions. Without
        # this each session defaults to intra_op_num_threads = all cores.
        thread_opts = {
            f"{m}_{opt}": v
            for m in ("det", "cls", "rec")
            for opt, v in (("intra_op_num_threads", n), ("inter_op_num_threads", 1))
        }

        v5 = self._resolve_v5_paths()
        if v5 is not None:
            self._engine = RapidOCR(
                det_model_path=v5["det"],
                rec_model_path=v5["rec"],
                cls_model_path=v5["cls"],
                rec_keys_path=v5["dict"],
                **thread_opts,
            )
            logger.info(
                "RapidOCR (PaddleOCR v5 mobile) loaded for lang=%s (threads=%d)",
                self.lang, n,
            )
        else:
            self._engine = RapidOCR(**thread_opts)
            logger.info(
                "RapidOCR (PaddleOCR v4 bundled) loaded for lang=%s (threads=%d)",
                self.lang, n,
            )

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

    def read_page(self, img: np.ndarray) -> OCRPageResult:
        """OCR an entire page, returning region-based results."""
        tuples = self.readtext(img)
        regions = [
            OCRRegionResult(bbox=bbox, text=text, confidence=conf)
            for bbox, text, conf in tuples
        ]
        avg_conf = sum(r.confidence for r in regions) / len(regions) if regions else 0.0
        return OCRPageResult(
            regions=regions,
            markdown=None,
            reading_order_native=False,
            confidence=avg_conf,
        )


# DocLayNet class name → womblex block_type mapping.
# Primary mapping. The 11-class DocLayNet taxonomy aligns directly with
# ElementKind: Picture → figure, Section-header / Title → heading, etc.
# Formula has no dedicated kind so it collapses to paragraph (text is
# preserved on the element; downstream consumers can read the original
# label via meta).
_YOLO_DOCLAYNET_LABEL_MAP: dict[str, str] = {
    "Caption": "caption",
    "Footnote": "footnote",
    "Formula": "paragraph",
    "List-item": "list_item",
    "Page-footer": "footer",
    "Page-header": "header",
    "Picture": "figure",
    "Section-header": "heading",
    "Table": "table",
    "Text": "paragraph",
    "Title": "heading",
}

# Legacy COCO class name → womblex block_type mapping.
# Retained for the fallback path only — when the DocLayNet checkpoint is
# unavailable, the COCO-trained yolov8n.pt produces detections whose
# class names have no document meaning. These mappings are best-effort
# guesses and produce mostly noise on real document pages (see
# docs/decisions.md "Element-kind classification"). Prefer the DocLayNet path.
_YOLO_COCO_LABEL_MAP: dict[str, str] = {
    "person": "paragraph",
    "book": "paragraph",
    "dining table": "table",
    "tv": "figure",
    "laptop": "figure",
    "cell phone": "figure",
    "monitor": "figure",
    "keyboard": "figure",
    "mouse": "figure",
    "scissors": "figure",
    "clock": "figure",
}


def _select_label_map(model_names: dict[int, str]) -> tuple[dict[str, str], str]:
    """Pick a label map based on the loaded model's class names.

    Returns ``(label_map, taxonomy_name)``. DocLayNet is detected by the
    presence of any DocLayNet-unique class (e.g. ``Section-header``);
    everything else falls back to COCO. ``taxonomy_name`` is used in logs
    and downstream telemetry.
    """
    classes = set(model_names.values())
    if "Section-header" in classes or "Page-footer" in classes:
        return _YOLO_DOCLAYNET_LABEL_MAP, "doclaynet"
    return _YOLO_COCO_LABEL_MAP, "coco"


# Recommended inference resolution per taxonomy. DocLayNet was trained at
# 1280×1280 — the model card recommends that resolution for small-class
# recall (Caption / Footnote). Empirically on government FOI documents
# (the ACT_EarlyChildhoodIncidents cohort) 832 matches or beats 1280 on
# the dominant text classes at ~3× the speed. The few real Caption /
# Footnote regions present are missed by the model at any resolution,
# so the 1280 cost isn't paying for itself on this corpus. Override to
# 1280 when running against documents with heavy small-class content.
_TAXONOMY_IMGSZ: dict[str, int] = {
    "doclaynet": 832,
    "coco": 640,
}


class YOLOLayoutAnalyzer:
    """Layout region detection via a local YOLO model.

    Resolves the DocLayNet-trained ``yolo11n_doc_layout.pt`` first, falling
    back to the COCO-trained ``yolov8n.pt`` only if the DocLayNet
    checkpoint is missing. The fallback exists to keep the layout path
    functional in partial installs — its output has no real document
    semantics. Class names from the loaded model select the matching
    label map at first use.

    Requires ``ultralytics`` to be installed (optional dependency).
    """

    def __init__(self, model_path: str | None = None) -> None:
        from pathlib import Path as _Path

        if model_path is None:
            from womblex.utils.models import resolve_local_model_path
            # DocLayNet-trained checkpoint is the primary path.
            resolved = resolve_local_model_path("yolo11n_doc_layout.pt")
            if isinstance(resolved, str):
                # Not present locally; fall back to COCO so the layout path
                # stays functional (predictions are mostly noise, but the
                # plumbing keeps working).
                resolved = resolve_local_model_path("yolov8n.pt")
            self._model_path = str(resolved)
        else:
            self._model_path = str(_Path(model_path))

        self._engine: object | None = None
        self._label_map: dict[str, str] = _YOLO_COCO_LABEL_MAP
        self._taxonomy: str = "coco"
        self._imgsz: int = _TAXONOMY_IMGSZ["coco"]

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        try:
            n = get_inference_threads()
            _apply_thread_env(n)
            # Cap torch's intra-op pool (defaults to cpu_count). interop must be
            # set before any parallel work — best-effort, ignore if already used.
            import torch
            from ultralytics import YOLO  # type: ignore[import-untyped]

            torch.set_num_threads(n)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

            self._engine = YOLO(self._model_path)
            names = getattr(self._engine, "names", {}) or {}
            self._label_map, self._taxonomy = _select_label_map(names)
            self._imgsz = _TAXONOMY_IMGSZ[self._taxonomy]
            logger.info(
                "YOLO layout model loaded from %s (taxonomy=%s, imgsz=%d)",
                self._model_path, self._taxonomy, self._imgsz,
            )
        except ImportError as exc:
            raise ImportError(
                "YOLOLayoutAnalyzer requires 'ultralytics'. "
                "Install with: pip install ultralytics"
            ) from exc

    def analyze(self, img: np.ndarray, conf_threshold: float = 0.3) -> list[LayoutRegion]:
        """Detect layout regions, returning sorted ``LayoutRegion`` objects.

        Inference resolution defaults to the per-taxonomy value in
        ``_TAXONOMY_IMGSZ`` (DocLayNet: 832, COCO: 640). Detected class
        names map through the loaded model's selected label map.
        """
        self._ensure_loaded()
        assert self._engine is not None

        results = self._engine(  # type: ignore[operator]
            img, conf=conf_threshold, verbose=False, imgsz=self._imgsz,
        )
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
                block_type = self._label_map.get(label, "paragraph")
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


def get_paddle_reader(lang: str = "eng", use_int8: bool = True) -> PaddleOCRReader:
    """Return a cached PaddleOCR reader for the given Tesseract-style lang code."""
    mapped = _LANG_MAP.get(lang, lang)
    key = f"{mapped}_{use_int8}"
    if key not in _paddle_readers:
        _paddle_readers[key] = PaddleOCRReader(lang=mapped, use_int8=use_int8)
    return _paddle_readers[key]


# Engine name aliases — canonical name on the left, accepted aliases on the right.
_ENGINE_ALIASES: dict[str, str] = {
    "paddleocr": "paddleocr",
    "paddle": "paddleocr",
    "rapidocr": "paddleocr",
    "mistral-ocr": "mistral-ocr",
    "mistral": "mistral-ocr",
    "mistralocr": "mistral-ocr",
    "pixtral": "mistral-ocr",
    "bedrock": "mistral-ocr",
    "ollama": "ollama",
    "ollama-ocr": "ollama",
}

# Canonical names of the LLM/VLM engines that return page-level markdown with
# reading order already resolved (skip preprocessing + layout sorting).
LLM_OCR_ENGINES: frozenset[str] = frozenset({"mistral-ocr", "ollama"})


def is_llm_engine(engine: str) -> bool:
    """True if *engine* (name or alias) is an LLM/VLM markdown engine."""
    return _ENGINE_ALIASES.get(engine.lower()) in LLM_OCR_ENGINES


def get_ocr_reader(
    engine: str = "paddleocr",
    lang: str = "eng",
    model: str | None = None,
    region: str | None = None,
    base_url: str | None = None,
    prompt: str | None = None,
):
    """Return a cached OCR reader for the requested engine.

    ``engine`` accepts canonical names (``paddleocr``, ``mistral-ocr``,
    ``ollama``) and common aliases. Engine-specific kwargs are forwarded
    only to engines that use them (passing others is a no-op):

    - ``mistral-ocr`` (Bedrock Pixtral): ``model``, ``region``.
    - ``ollama`` (local VLM): ``model``, ``base_url``, ``prompt``.
    """
    canonical = _ENGINE_ALIASES.get(engine.lower())
    if canonical is None:
        raise ValueError(
            f"unknown OCR engine: {engine!r} "
            f"(known: {sorted(set(_ENGINE_ALIASES.values()))})"
        )

    if canonical == "paddleocr":
        return get_paddle_reader(lang=lang)

    if canonical == "mistral-ocr":
        from womblex.ingest.llm_ocr import get_mistral_reader
        return get_mistral_reader(model=model, region=region)

    if canonical == "ollama":
        from womblex.ingest.llm_ocr import get_ollama_reader
        return get_ollama_reader(model=model, base_url=base_url, prompt=prompt)

    raise ValueError(f"unhandled engine after alias resolution: {canonical!r}")


def get_layout_analyzer() -> YOLOLayoutAnalyzer:
    """Return a cached YOLOv8 layout analyzer."""
    global _layout_analyzer
    if _layout_analyzer is None:
        _layout_analyzer = YOLOLayoutAnalyzer()
    return _layout_analyzer


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
