"""Extraction strategies for native (text-layer) PDF documents.

Covers NATIVE_NARRATIVE, NATIVE_WITH_STRUCTURED, and STRUCTURED document
types where text is extracted from the PDF text layer without OCR.
"""

from __future__ import annotations

import logging

import fitz

from womblex.ingest.extract import (
    ExtractionMetadata,
    ExtractionResult,
    PageResult,
    TableData,
    TextBlock,
    _build_text_blocks,
    _extract_form_fields,
    _extract_images_from_page,
    _extract_tables_from_page,
    _text_coverage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. native_narrative
# ---------------------------------------------------------------------------


class NativeNarrativeExtractor:
    """Extract clean text preserving paragraph structure from native PDFs."""

    def extract(self, doc: fitz.Document) -> ExtractionResult:
        pages: list[PageResult] = []
        all_blocks: list[TextBlock] = []

        for page in doc:
            text = page.get_text("text", flags=fitz.TEXT_DEHYPHENATE)
            pages.append(PageResult(page_number=page.number, text=text, method="native"))
            all_blocks.extend(_build_text_blocks(page))

        coverage = _text_coverage(pages)
        return ExtractionResult(
            pages=pages,
            method="native_narrative",
            text_blocks=all_blocks,
            metadata=ExtractionMetadata(
                extraction_strategy="native_narrative",
                confidence=min(0.95, 0.8 + coverage * 0.15),
                processing_time=0.0,
                page_count=len(doc),
                text_coverage=coverage,
            ),
        )


# ---------------------------------------------------------------------------
# 2. native_with_structured
# ---------------------------------------------------------------------------


class NativeWithStructuredExtractor:
    """Extract text, tables, forms, and images from native structured PDFs."""

    def extract(self, doc: fitz.Document) -> ExtractionResult:
        pages: list[PageResult] = []
        all_tables = []
        all_forms = []
        all_images = []
        all_blocks: list[TextBlock] = []

        for page in doc:
            text = page.get_text("text", flags=fitz.TEXT_DEHYPHENATE)
            pages.append(PageResult(page_number=page.number, text=text, method="native"))

            all_tables.extend(_extract_tables_from_page(page))
            all_forms.extend(_extract_form_fields(page))
            all_images.extend(_extract_images_from_page(page))
            all_blocks.extend(_build_text_blocks(page))

        coverage = _text_coverage(pages)
        return ExtractionResult(
            pages=pages,
            method="native_with_structured",
            tables=all_tables,
            forms=all_forms,
            images=all_images,
            text_blocks=all_blocks,
            metadata=ExtractionMetadata(
                extraction_strategy="native_with_structured",
                confidence=0.85,
                processing_time=0.0,
                page_count=len(doc),
                text_coverage=coverage,
            ),
        )


# ---------------------------------------------------------------------------
# 3. structured
# ---------------------------------------------------------------------------


class StructuredExtractor:
    """Extract table structures with header relationships and data types."""

    def extract(self, doc: fitz.Document) -> ExtractionResult:
        pages: list[PageResult] = []
        all_tables = []
        all_blocks: list[TextBlock] = []

        for page in doc:
            text = page.get_text("text")
            pages.append(PageResult(page_number=page.number, text=text, method="native"))

            tables = _extract_tables_from_page(page)
            all_tables.extend(tables)
            all_blocks.extend(_build_text_blocks(page))

        coverage = _text_coverage(pages)
        return ExtractionResult(
            pages=pages,
            method="structured",
            tables=all_tables,
            text_blocks=all_blocks,
            metadata=ExtractionMetadata(
                extraction_strategy="structured",
                confidence=0.8 if all_tables else 0.6,
                processing_time=0.0,
                page_count=len(doc),
                text_coverage=coverage,
            ),
        )
