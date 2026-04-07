"""Redaction operation.

Runs as a separate pass after extraction. Renders PDF pages as images,
detects black-box redaction regions, and applies the configured mode
to the affected page text.

Modes:
- ``flag``:     Set ``has_redaction=True`` on affected chunks (no text change).
- ``blackout``: Replace affected page text with ``[REDACTED]`` markers.
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


def detect_redactions(
    path: Path,
    page_count: int,
    detector: RedactionDetector,
    dpi: int = 150,
) -> RedactionReport:
    """Render each page of a PDF and detect redacted regions.

    Args:
        path: Path to the PDF file.
        page_count: Number of pages to scan (from extraction metadata).
        detector: Configured RedactionDetector instance.
        dpi: Resolution for page rendering.

    Returns:
        RedactionReport with per-page detection results.
    """
    import fitz

    report = RedactionReport()
    try:
        doc = fitz.open(str(path))
        pages_to_scan = min(page_count, len(doc))
        for page_num in range(pages_to_scan):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            redactions = detector.detect(img, page=page_num)
            if redactions:
                report.page_redactions[page_num] = redactions
        doc.close()
    except Exception as e:
        logger.warning("Redaction detection failed for %s: %s", path, e)

    return report


def apply_text_redaction(
    pages: list[PageResult],
    report: RedactionReport,
    mode: str,
) -> list[PageResult]:
    """Modify page text based on the redaction mode.

    ``flag`` makes no text changes — use ``annotate_chunks`` instead.
    ``blackout`` prepends ``[REDACTED]`` to affected page text.
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
            page.text = f"[REDACTED]\n{page.text}" if page.text else "[REDACTED]"
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
