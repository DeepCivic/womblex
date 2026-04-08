"""Extraction strategies for non-PDF file formats.

Covers DOCX, TEXT, and non-textual fallback document types.
SpreadsheetExtractor lives in spreadsheet.py (already separate).
"""

from __future__ import annotations

import logging
from pathlib import Path

from womblex.ingest.extract import (
    ExtractionMetadata,
    ExtractionResult,
    PageResult,
    Position,
    TableData,
    TextBlock,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


class DocxExtractor:
    """Extract text and tables from Word documents."""

    def extract_path(self, path: Path) -> ExtractionResult:
        """Extract from a DOCX file path (not a fitz.Document)."""
        try:
            from docx import Document
        except ImportError:
            return ExtractionResult(
                pages=[],
                method="docx",
                error="python-docx not installed; cannot extract DOCX.",
                metadata=ExtractionMetadata(
                    extraction_strategy="docx",
                    confidence=0.0,
                    processing_time=0.0,
                    page_count=0,
                    text_coverage=0.0,
                ),
            )

        try:
            doc = Document(str(path))
        except Exception as e:
            return ExtractionResult(
                pages=[],
                method="docx",
                error=f"Failed to read DOCX: {e}",
                metadata=ExtractionMetadata(
                    extraction_strategy="docx",
                    confidence=0.0,
                    processing_time=0.0,
                    page_count=0,
                    text_coverage=0.0,
                ),
            )

        # Extract paragraph text
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n\n".join(paragraphs)

        # Extract tables
        all_tables = []
        pos = Position(x=0.0, y=0.0, width=1.0, height=1.0)
        for tbl in doc.tables:
            rows_data = []
            for row in tbl.rows:
                rows_data.append([cell.text for cell in row.cells])
            headers = rows_data[0] if rows_data else []
            data_rows = rows_data[1:] if len(rows_data) > 1 else []
            all_tables.append(TableData(headers=headers, rows=data_rows, position=pos, confidence=0.85))

        # Build text blocks
        blocks = []
        for para in doc.paragraphs:
            if not para.text.strip():
                continue
            block_type = "heading" if para.style and "heading" in para.style.name.lower() else "paragraph"
            blocks.append(TextBlock(text=para.text, position=pos, block_type=block_type, confidence=0.9))

        return ExtractionResult(
            pages=[PageResult(page_number=0, text=text, method="docx")],
            method="docx",
            tables=all_tables,
            text_blocks=blocks,
            metadata=ExtractionMetadata(
                extraction_strategy="docx",
                confidence=0.9,
                processing_time=0.0,
                page_count=1,
                text_coverage=1.0 if text else 0.0,
            ),
        )


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


class TextExtractor:
    """Plain text file passthrough -- reads the file as-is."""

    def extract_path(self, path: Path) -> ExtractionResult:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        return ExtractionResult(
            pages=[PageResult(page_number=0, text=text, method="text")],
            method="text",
            metadata=ExtractionMetadata(
                extraction_strategy="text",
                confidence=1.0,
                processing_time=0.0,
                page_count=1,
                text_coverage=1.0 if text.strip() else 0.0,
            ),
        )


# ---------------------------------------------------------------------------
# Non-textual fallback
# ---------------------------------------------------------------------------


class NonTextualExtractor:
    """Placeholder for documents that cannot be extracted -- flags for manual review."""

    def extract(self, doc: "fitz.Document") -> ExtractionResult:
        import fitz as _fitz  # noqa: F811 — lazy import for type

        return ExtractionResult(
            pages=[],
            method="non_textual",
            error="Document flagged as non-textual; requires manual review.",
            metadata=ExtractionMetadata(
                extraction_strategy="non_textual",
                confidence=0.0,
                processing_time=0.0,
                page_count=len(doc),
                text_coverage=0.0,
            ),
        )
