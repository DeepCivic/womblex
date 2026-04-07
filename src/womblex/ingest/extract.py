"""Text extraction strategies for different document types.

Each strategy implements the ExtractionStrategy protocol and returns
an ExtractionResult with per-page text, structured content, and metadata.
Output is designed to map directly to the Parquet output schema.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import fitz
import numpy as np

from womblex.ingest.detect import DocumentProfile, DocumentType

if TYPE_CHECKING:
    from womblex.redact.stage import RedactionReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured content models (Parquet-ready)
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """Normalised bounding box (0-1 document-relative coordinates)."""

    x: float
    y: float
    width: float
    height: float


@dataclass
class TableData:
    """Extracted table with structure preserved."""

    headers: list[str]
    rows: list[list[str]]
    position: Position
    confidence: float


@dataclass
class FormField:
    """Extracted form field (label-value pair)."""

    field_name: str
    value: str
    position: Position
    confidence: float


@dataclass
class ImageData:
    """Image metadata from a document page."""

    alt_text: str
    position: Position
    confidence: float


@dataclass
class TextBlock:
    """A segment of text with positional and type metadata."""

    text: str
    position: Position
    block_type: str  # paragraph, heading, list_item, caption, etc.
    confidence: float


# ---------------------------------------------------------------------------
# Extraction result models
# ---------------------------------------------------------------------------


@dataclass
class PageResult:
    """Extracted text for a single page."""

    page_number: int
    text: str
    method: str


@dataclass
class ExtractionMetadata:
    """Document-level extraction metadata."""

    extraction_strategy: str
    confidence: float
    processing_time: float
    page_count: int
    text_coverage: float
    preprocessing_steps: list[str] = field(default_factory=list)
    content_mix: dict[str, float] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of text extraction from a document."""

    pages: list[PageResult] = field(default_factory=list)
    method: str = ""
    error: str | None = None
    tables: list[TableData] = field(default_factory=list)
    forms: list[FormField] = field(default_factory=list)
    images: list[ImageData] = field(default_factory=list)
    text_blocks: list[TextBlock] = field(default_factory=list)
    metadata: ExtractionMetadata | None = None
    warnings: list[str] = field(default_factory=list)
    document_id: str | None = None  # set by extractors that produce multiple results per file
    redaction_report: RedactionReport | None = None

    @property
    def full_text(self) -> str:
        """Concatenate all page texts."""
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def page_count(self) -> int:
        return len(self.pages)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class ExtractionStrategy(Protocol):
    """Protocol for document extraction strategies."""

    def extract(self, doc: fitz.Document) -> ExtractionResult: ...


# ---------------------------------------------------------------------------
# Utility helpers shared by strategies
# ---------------------------------------------------------------------------


def _page_to_gray(page: fitz.Page, dpi: int = 150) -> np.ndarray:
    """Render a page to a grayscale numpy array."""
    import cv2
    import numpy as np

    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.copy()


def _normalise_rect(rect: fitz.Rect, page_width: float, page_height: float) -> Position:
    """Convert a PyMuPDF Rect to normalised 0-1 coordinates."""
    return Position(
        x=rect.x0 / page_width if page_width else 0.0,
        y=rect.y0 / page_height if page_height else 0.0,
        width=(rect.x1 - rect.x0) / page_width if page_width else 0.0,
        height=(rect.y1 - rect.y0) / page_height if page_height else 0.0,
    )


def _normalise_bbox(
    bbox: tuple[float, float, float, float], page_width: float, page_height: float
) -> Position:
    """Convert (x0, y0, x1, y1) to normalised 0-1 coordinates."""
    x0, y0, x1, y1 = bbox
    return Position(
        x=x0 / page_width if page_width else 0.0,
        y=y0 / page_height if page_height else 0.0,
        width=(x1 - x0) / page_width if page_width else 0.0,
        height=(y1 - y0) / page_height if page_height else 0.0,
    )


def _extract_tables_from_page(page: fitz.Page) -> list[TableData]:
    """Extract tables from a page using PyMuPDF's table finder."""
    import sys
    import io as _io

    tables: list[TableData] = []
    pw, ph = page.rect.width, page.rect.height

    try:
        old_stdout = sys.stdout
        sys.stdout = _io.StringIO()
        try:
            found = page.find_tables()
        finally:
            sys.stdout = old_stdout

        for tbl in found.tables:
            if tbl.row_count < 1 or tbl.col_count < 1:
                continue
            extracted = tbl.extract()
            headers = [str(c) if c else "" for c in extracted[0]] if extracted else []
            rows = [[str(c) if c else "" for c in row] for row in extracted[1:]] if len(extracted) > 1 else []
            pos = _normalise_rect(fitz.Rect(tbl.bbox), pw, ph)
            tables.append(TableData(headers=headers, rows=rows, position=pos, confidence=0.8))
    except Exception:
        pass

    return tables


def _extract_images_from_page(page: fitz.Page) -> list[ImageData]:
    """Extract image metadata from a page."""
    images_out: list[ImageData] = []
    pw, ph = page.rect.width, page.rect.height

    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            rects = page.get_image_rects(xref)
            for rect in rects:
                pos = _normalise_rect(rect, pw, ph)
                images_out.append(ImageData(alt_text="", position=pos, confidence=0.7))
        except Exception:
            continue

    return images_out


def _extract_form_fields(page: fitz.Page) -> list[FormField]:
    """Extract interactive form fields from a page."""
    fields: list[FormField] = []
    pw, ph = page.rect.width, page.rect.height

    for widget in page.widgets():
        name = widget.field_name or ""
        value = widget.field_value or ""
        pos = _normalise_rect(widget.rect, pw, ph)
        fields.append(FormField(field_name=name, value=value, position=pos, confidence=0.9))

    return fields


def _ocr_text_block(
    page: fitz.Page, text: str, conf: float, block_type: str = "paragraph"
) -> TextBlock | None:
    """Build a TextBlock from OCR output, or None if text is empty."""
    text = text.strip()
    if not text:
        return None
    pw, ph = page.rect.width, page.rect.height
    return TextBlock(
        text=text,
        position=_normalise_bbox((0, 0, pw, ph), pw, ph),
        block_type=block_type,
        confidence=conf / 100 if conf else 0.0,
    )


def _build_text_blocks(page: fitz.Page) -> list[TextBlock]:
    """Extract text blocks with positional data and type classification."""
    blocks: list[TextBlock] = []
    pw, ph = page.rect.width, page.rect.height

    raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # text blocks only
            continue
        bbox = block.get("bbox", (0, 0, 0, 0))
        pos = _normalise_bbox(bbox, pw, ph)

        # Collect all text from the block
        block_text = ""
        max_font_size = 0.0
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                block_text += span.get("text", "")
                fs = span.get("size", 0)
                if fs > max_font_size:
                    max_font_size = fs

        block_text = block_text.strip()
        if not block_text:
            continue

        # Classify block type by font size heuristic
        if max_font_size >= 16:
            block_type = "heading"
        elif len(block_text) < 40:
            block_type = "caption"
        else:
            block_type = "paragraph"

        blocks.append(TextBlock(text=block_text, position=pos, block_type=block_type, confidence=0.9))

    return blocks


# ---------------------------------------------------------------------------
# Post-extraction text normalisation
# ---------------------------------------------------------------------------

# Running footer pattern — spaced-out "X | P a g e" from OCR of page footers.
# OCR frequently corrupts the letters: P→F/F', a→&/0/2, g→&/8/3/<, e→&/€/?/F/P/digit.
# Anchored to start/end of line to avoid false positives in body text.
_FOOTER_RE = re.compile(
    r"(?m)^\s*\d+\s*[|&]\s*[PF]'?\s*[a&02]\s*[g&823<]\s*[e&€PF?\d]\s*$"
)

# Split footer pattern — footer broken across two lines by OCR.
# Line 1: bare page number with optional pipe/ampersand.
# Line 2: spaced-character "P a g e" fragment (requires spaces between positions
# to distinguish from body text starting with P/F).
_FOOTER_SPLIT_RE = re.compile(
    r"(?m)^\s*\d{1,2}\s*[|&]?\s*$\n\s*[PF]'?\s+[a&02]\s+[g&823<]\s*[e&€PF?\d]?\s*$"
)

# Corrupted '://' in URLs from broken font encoding.
# OCR renders '//' as combinations of l, L, I, spaces, colons, and underscores.
# Only fires when followed by 'www' (case-insensitive) to avoid false positives.
_URL_SCHEME_RE = re.compile(r"http(?!://)([\s:./lLI_]+)(?=[wW])")


def _normalise_text(text: str) -> str:
    """Apply targeted post-extraction corrections to a single page's text.

    Fixes known artefacts from government PDF document sets:
    - Broken ToUnicode font maps produce '$' or '€' where 's' follows an apostrophe
    - Running footers rendered as spaced characters by OCR (single-line and split)
    - Corrupted '://' in URLs from broken font encoding
    """
    # RES-001 extended: apostrophe + dollar/euro → apostrophe + s
    text = text.replace("\u2019 $", "\u2019s")
    text = text.replace("\u2019$", "\u2019s")
    text = text.replace("' $", "'s")
    text = text.replace("'$", "'s")
    text = text.replace("\u2019\u20ac", "\u2019s")
    text = text.replace("'\u20ac", "'s")
    # RES-002: URL scheme corruption
    text = _URL_SCHEME_RE.sub("http://", text)
    # RES-003 extended: running page footers (single-line then split across two lines)
    text = _FOOTER_RE.sub("", text)
    text = _FOOTER_SPLIT_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

# Imported from strategies module to keep this file under 750 lines.
# The strategies module uses the helpers above.
from womblex.ingest.strategies import (  # noqa: E402
    NativeNarrativeExtractor,
    NativeWithStructuredExtractor,
    StructuredExtractor,
    ScannedMachinewrittenExtractor,
    ScannedHandwrittenExtractor,
    ScannedMixedExtractor,
    HybridExtractor,
    ImageExtractor,
    DocxExtractor,
    TextExtractor,
)
from womblex.ingest.spreadsheet import SpreadsheetExtractor  # noqa: E402


def get_extractor(
    profile: DocumentProfile,
    dpi: int = 200,
    lang: str = "eng",
) -> ExtractionStrategy | SpreadsheetExtractor | DocxExtractor:
    """Select the appropriate extraction strategy for a document profile.

    Note: SPREADSHEET and DOCX types return path-based extractors that
    implement ``extract_path(path)`` instead of ``extract(doc)``.
    Use ``extract_text()`` which handles both protocols.
    """
    match profile.doc_type:
        case DocumentType.NATIVE_NARRATIVE:
            return NativeNarrativeExtractor()
        case DocumentType.NATIVE_WITH_STRUCTURED:
            return NativeWithStructuredExtractor()
        case DocumentType.STRUCTURED:
            return StructuredExtractor()
        case DocumentType.SCANNED_MACHINEWRITTEN:
            return ScannedMachinewrittenExtractor(dpi=dpi, lang=lang)
        case DocumentType.SCANNED_HANDWRITTEN:
            return ScannedHandwrittenExtractor(dpi=dpi, lang=lang)
        case DocumentType.SCANNED_MIXED:
            return ScannedMixedExtractor(dpi=dpi, lang=lang)
        case DocumentType.HYBRID:
            return HybridExtractor(dpi=dpi, lang=lang)
        case DocumentType.IMAGE:
            return ImageExtractor(dpi=dpi, lang=lang)
        case DocumentType.SPREADSHEET:
            return SpreadsheetExtractor(profile=profile)
        case DocumentType.DOCX:
            return DocxExtractor()
        case DocumentType.TEXT:
            return TextExtractor()
        case _:
            return NativeNarrativeExtractor()


def extract_text(
    path: Path,
    profile: DocumentProfile,
    dpi: int = 200,
    lang: str = "eng",
    max_pages: int | None = None,
) -> list[ExtractionResult]:
    """Extract text from a document using the strategy matching its profile.

    Returns one ExtractionResult per logical unit. PDFs and DOCX return a
    single-element list. Spreadsheets return one element per row or sheet.

    When *max_pages* is set, PDF extraction is limited to the first N pages.
    """
    extractor = get_extractor(profile, dpi=dpi, lang=lang)
    logger.info(
        "strategy selected: doc=%s type=%s confidence=%.2f strategy=%s",
        path.name, profile.doc_type.value, profile.confidence,
        type(extractor).__name__,
    )

    # Path-based extractors (spreadsheet, DOCX)
    if hasattr(extractor, "extract_path"):
        t0 = time.monotonic()
        raw = extractor.extract_path(path)
        elapsed = time.monotonic() - t0
        results: list[ExtractionResult] = raw if isinstance(raw, list) else [raw]
        per = elapsed / len(results) if results else elapsed
        for r in results:
            if r.metadata:
                r.metadata.processing_time = per
            _apply_normalisation_and_warnings(r, path)
        return results

    # PDF-based extractors
    doc = fitz.open(str(path))
    try:
        if max_pages is not None and doc.page_count > max_pages:
            doc.select(list(range(max_pages)))
        t0 = time.monotonic()
        result = extractor.extract(doc)
        elapsed = time.monotonic() - t0
        if result.metadata:
            result.metadata.processing_time = elapsed
        _apply_normalisation_and_warnings(result, path)
        return [result]
    finally:
        doc.close()


def _apply_normalisation_and_warnings(result: ExtractionResult, path: Path) -> None:
    """Normalise page text in-place and capture blank page warnings."""
    for page in result.pages:
        if not page.text.strip():
            warning = f"blank page {page.page_number} (method={result.method})"
            result.warnings.append(warning)
            logger.warning(
                "blank page extracted — possible silent failure: doc=%s page=%d method=%s",
                path.name,
                page.page_number,
                result.method,
            )
        else:
            page.text = _normalise_text(page.text)
