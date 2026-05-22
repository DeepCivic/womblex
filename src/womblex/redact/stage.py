"""Redaction operation.

Runs as a separate pass after extraction. Renders PDF pages as images,
detects black-box redaction regions, and applies the configured mode
to the affected page text.

Modes:
- ``flag``:     Set ``has_redaction=True`` on affected chunks (no text change).
- ``blackout``: Replace affected page text with ``<REDACTED>`` markers.
- ``delete``:   Clear affected page text entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from womblex.config import RedactionConfig
from womblex.redact.detector import RedactionDetector, RedactionInfo

if TYPE_CHECKING:
    from womblex.ingest.elements import Element
    from womblex.ingest.extract import ExtractionResult, PageResult
    from womblex.process.chunker import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class RedactionReport:
    """Summary of redactions detected across a document."""

    page_redactions: dict[int, list[RedactionInfo]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.page_redactions.values())

    @property
    def affected_pages(self) -> list[int]:
        return sorted(self.page_redactions.keys())


def build_detector(config: RedactionConfig) -> RedactionDetector:
    """Build a RedactionDetector from config."""
    return RedactionDetector(
        threshold=config.threshold,
        min_area_ratio=config.min_area_ratio,
        max_area_ratio=config.max_area_ratio,
    )


_VECTOR_MIN_WIDTH_PT = 3.0   # filters narrow vertical separator lines (manifest column rules)
_VECTOR_MIN_HEIGHT_PT = 8.0  # filters glyph-rendering small filled rects (body glyphs ≤ 7pt tall)

# YOLO COCO classes whose regions, when present, typically land on the
# form-field backgrounds and embedded chart / figure regions where raster-path
# false positives originate (02737-class scanned_mixed CRM forms). When the
# raster fallback runs, contour hits whose centre falls inside one of these
# regions are dropped. See ``_YOLO_COCO_LABEL_MAP`` in ``ingest/paddle_ocr.py``
# for the mapping rationale.
_LAYOUT_EXCLUSION_CLASSES = frozenset({
    "tv", "laptop", "monitor", "cell phone", "keyboard", "mouse",
    "book", "dining table",
})


def detect_redactions(
    path: Path,
    page_count: int,
    detector: RedactionDetector,
    dpi: int = 150,
    use_layout_filter: bool = True,
) -> RedactionReport:
    """Detect redacted regions per page; prefer vector ops, fall back to raster.

    For each page:

    - First check ``page.get_drawings()`` for filled near-black rectangles
      (matches native-PDF vector-drawn redactions; no area threshold).
    - If none found, rasterise the page at *dpi* and run the CV2 contour
      detector (handles raster overlays and scanned pages). When
      *use_layout_filter* is true, run YOLO layout analysis on the rasterised
      image and pass figure/chart/form-background regions as exclusion zones
      to the contour detector — suppresses raster false positives on dark
      form-field backgrounds and embedded chart regions (02737-class
      scanned_mixed CRM forms). The filter is best-effort: if ``ultralytics``
      isn't installed or layout analysis fails, detection falls back to the
      raw raster pass with no exclusion.

    Bboxes are returned in pixel coordinates at *dpi* regardless of which
    path produced them, so consumers see a single coord system.

    Args:
        path: Path to the PDF file.
        page_count: Number of pages to scan (from extraction metadata).
        detector: Configured RedactionDetector instance.
        dpi: Resolution for page rendering / coord scaling.
        use_layout_filter: Run YOLO layout analysis on raster-fallback pages
            and drop contour hits inside figure / chart / form-background
            regions. Best-effort; falls back to raw raster pass on error.

    Returns:
        RedactionReport with per-page detection results.
    """
    import fitz

    report = RedactionReport()
    try:
        doc = fitz.open(str(path))
        pages_to_scan = min(page_count, len(doc))
        scale = dpi / 72.0  # PDF coord (72 DPI) → pixel coord at *dpi*
        for page_num in range(pages_to_scan):
            page = doc[page_num]

            vector_redactions = _detect_vector_redactions(page, page_num, scale)
            if vector_redactions:
                report.page_redactions[page_num] = vector_redactions
                continue

            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            exclude_rects = _layout_exclude_rects(img) if use_layout_filter else None
            raster_redactions = detector.detect(
                img, page=page_num, exclude_rects=exclude_rects,
            )
            if raster_redactions:
                report.page_redactions[page_num] = raster_redactions
        doc.close()
    except Exception as e:
        logger.warning("Redaction detection failed for %s: %s", path, e)

    return report


def _layout_exclude_rects(
    img: np.ndarray,
) -> list[tuple[int, int, int, int]] | None:
    """Return figure/chart/form-background bboxes from YOLO layout analysis.

    Best-effort: returns ``None`` on any failure (missing ultralytics, model
    weights absent, inference error). Caller treats ``None`` and ``[]``
    interchangeably — both mean "no exclusion".
    """
    try:
        from womblex.ingest.paddle_ocr import get_layout_analyzer
        analyzer = get_layout_analyzer()
        regions = analyzer.analyze(img)
    except Exception as e:
        logger.debug("layout filter unavailable; falling back to raw raster: %s", e)
        return None

    rects: list[tuple[int, int, int, int]] = []
    for region in regions:
        if region.label not in _LAYOUT_EXCLUSION_CLASSES:
            continue
        x0, y0, x1, y1 = region.bbox
        rects.append((int(x0), int(y0), int(x1), int(y1)))
    return rects


def _detect_vector_redactions(page, page_num: int, scale: float) -> list[RedactionInfo]:
    """Enumerate filled near-black rectangles from ``page.get_drawings()``.

    Bboxes converted from PDF coords (72 DPI) to pixel coords using *scale*
    so all ``RedactionInfo.bbox`` values share one coord system regardless of
    which detection path produced them.
    """
    out: list[RedactionInfo] = []
    for d in page.get_drawings():
        if d.get("type") not in ("f", "fs", "sf"):
            continue
        if not _is_near_black_fill(d.get("fill")):
            continue
        rect = d.get("rect")
        if rect is None or rect.width < _VECTOR_MIN_WIDTH_PT or rect.height < _VECTOR_MIN_HEIGHT_PT:
            continue
        x1 = int(rect.x0 * scale)
        y1 = int(rect.y0 * scale)
        x2 = int(rect.x1 * scale)
        y2 = int(rect.y1 * scale)
        out.append(RedactionInfo(
            bbox=(x1, y1, x2, y2),
            page=page_num,
            area_px=(x2 - x1) * (y2 - y1),
        ))
    return out


def _is_near_black_fill(fill) -> bool:
    """Treat fill as near-black if max channel ≤ 0.1 (CMYK: K ≥ 0.9 + others ≤ 0.1)."""
    if fill is None:
        return False
    if isinstance(fill, (int, float)):
        return fill <= 0.1
    if len(fill) == 1:
        return fill[0] <= 0.1
    if len(fill) == 3:
        return max(fill) <= 0.1
    if len(fill) == 4:
        return fill[3] >= 0.9 and max(fill[:3]) <= 0.1
    return False


def apply_text_redaction(
    pages: list[PageResult],
    report: RedactionReport,
    mode: str,
) -> list[PageResult]:
    """Modify page text based on the redaction mode.

    ``flag`` makes no text changes — use ``annotate_chunks`` instead.
    ``blackout`` prepends ``<REDACTED>`` to affected page text.
    ``delete`` clears affected page text entirely.

    Args:
        pages: Per-page extraction results (mutated in-place).
        report: Detected redaction regions.
        mode: One of ``flag``, ``blackout``, ``delete``.

    Returns:
        The (mutated) pages list.
    """
    if mode == "flag" or not report.total:
        return pages

    affected = set(report.affected_pages)
    for page in pages:
        if page.page_number not in affected:
            continue
        if mode == "blackout":
            page.text = f"<REDACTED>\n{page.text}" if page.text else "<REDACTED>"
        elif mode == "delete":
            page.text = ""

    return pages


def annotate_chunks(
    chunks: list[TextChunk],
    report: RedactionReport,
) -> list[TextChunk]:
    """Mark chunks whose source pages contain redacted regions.

    Sets ``chunk.has_redaction = True`` for any chunk overlapping an
    affected page. Does not modify chunk text.
    """
    if not report.total:
        return chunks

    affected = set(report.affected_pages)
    for chunk in chunks:
        if hasattr(chunk, "source_pages") and chunk.source_pages:
            if any(p in affected for p in chunk.source_pages):
                chunk.has_redaction = True
        elif hasattr(chunk, "page_number") and chunk.page_number in affected:
            chunk.has_redaction = True

    return chunks


def annotate_elements(
    elements: list[Element],
    report: RedactionReport,
) -> list[Element]:
    """Set ``meta['has_redaction']='true'`` on elements whose page is in *report*.

    Page-level propagation: every element on an affected page is flagged.
    Avoids the pixel-coord (report bboxes at detection DPI) vs PDF-coord
    (element bboxes at 72 DPI) conversion that bbox-level overlap would
    require. Mutates elements in place; returns the same list.
    """
    if not report.total:
        return elements

    affected = report.page_redactions
    for element in elements:
        if element.page is not None and element.page in affected:
            element.meta["has_redaction"] = "true"
    return elements


def annotate_extraction(
    extraction: ExtractionResult,
    report: RedactionReport,
) -> ExtractionResult:
    """Annotate an ExtractionResult with redaction metadata.

    Adds per-page warning strings so downstream consumers know which
    pages had redacted content detected.
    """
    if not report.total:
        return extraction

    for page_num, redactions in report.page_redactions.items():
        extraction.warnings.append(
            f"page {page_num}: {len(redactions)} redacted region(s) detected"
        )

    return extraction
