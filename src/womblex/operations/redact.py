"""run_redaction — detect black-box regions and apply mode to page text."""

from __future__ import annotations

import logging

from womblex.config import WomblexConfig
from womblex.operations.models import DocumentResult
from womblex.redact.stage import (
    annotate_elements,
    annotate_extraction,
    apply_text_redaction,
    build_detector,
    detect_redactions,
)

logger = logging.getLogger(__name__)


def run_redaction(
    results: list[DocumentResult], config: WomblexConfig
) -> list[DocumentResult]:
    """Detect and handle redacted regions in extracted documents.

    Renders each PDF page as an image, runs the black-box detector, and
    applies the configured mode:

    - ``flag``:     Annotate chunks/records (no text change).
    - ``blackout``: Prepend ``<REDACTED>`` to affected page text.
    - ``delete``:   Clear affected page text entirely.

    Non-PDF documents (spreadsheets, DOCX) are skipped — redaction
    detection requires a rasterisable page source.

    Args:
        results: DocumentResults from ``run_extraction``.
        config: Pipeline configuration (uses ``config.redaction``).

    Returns:
        The same list with redaction applied.
    """
    if not config.redaction.enabled:
        return results

    detector = build_detector(config.redaction)
    mode = config.redaction.mode

    for dr in results:
        if dr.status != "completed" or not dr.extraction:
            continue
        if dr.path.suffix.lower() not in {".pdf"}:
            continue

        report = detect_redactions(
            dr.path,
            dr.extraction.page_count,
            detector,
            dpi=config.redaction.dpi,
            use_layout_filter=config.redaction.use_layout_filter,
        )

        if not report.total:
            continue

        dr.extraction.redaction_report = report
        annotate_extraction(dr.extraction, report)
        annotate_elements(dr.extraction.elements, report)

        if mode != "flag":
            apply_text_redaction(dr.extraction.pages, report, mode)

        logger.info(
            "Redaction [%s]: doc=%s pages_affected=%d regions=%d",
            mode, dr.doc_id, len(report.affected_pages), report.total,
        )

    return results
