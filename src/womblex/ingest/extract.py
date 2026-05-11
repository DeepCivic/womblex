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
    """Extracted table with structure preserved.

    `context` carries per-table metadata captured immediately above the
    table on the source page (e.g. report-reference labels, scope tags
    from spreadsheet-printed-to-PDF docs). Empty for tables without a
    leading metadata block.
    """

    headers: list[str]
    rows: list[list[str]]
    position: Position
    confidence: float
    context: dict[str, str] = field(default_factory=dict)


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
    document_metadata: dict[str, str] = field(default_factory=dict)
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
    """Protocol for document extraction strategies (PDF-based)."""

    def extract(self, doc: fitz.Document) -> ExtractionResult: ...


class PathExtractionStrategy(Protocol):
    """Protocol for file-path-based extraction (DOCX, spreadsheet, text)."""

    def extract_path(self, path: Path) -> ExtractionResult | list[ExtractionResult]: ...


# ---------------------------------------------------------------------------
# Utility helpers shared by strategies
# ---------------------------------------------------------------------------


def _text_coverage(pages: list[PageResult]) -> float:
    """Fraction of pages with meaningful text (>50 chars)."""
    if not pages:
        return 0.0
    filled = sum(1 for p in pages if len(p.text.strip()) > 50)
    return filled / len(pages)


def _page_to_gray(page: fitz.Page, dpi: int = 150) -> np.ndarray:
    """Render a page to a grayscale numpy array."""
    import cv2

    pix = page.get_pixmap(dpi=dpi)
    img = _pixmap_to_array(pix, drop_alpha=False)
    if pix.n >= 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.copy()


def _pixmap_to_array(pix: fitz.Pixmap, *, drop_alpha: bool = False) -> np.ndarray:
    """Convert a PyMuPDF Pixmap to a numpy array, optionally dropping alpha."""
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if drop_alpha and pix.n == 4:
        return img[:, :, :3]
    return img


def _avg_ocr_confidence(results: list[tuple], *, scale: float = 1.0) -> float:
    """Mean confidence across non-empty OCR results.

    `results` is an EasyOCR-style ``(bbox, text, confidence)`` list.
    ``scale=1`` returns 0–1; ``scale=100`` returns 0–100.
    """
    confs = [float(r[2]) for r in results if r[1].strip()]
    if not confs:
        return 0.0
    return (sum(confs) / len(confs)) * scale


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


def _find_native_tables(
    page: fitz.Page,
) -> list[tuple[TableData, fitz.Rect, list[list]]]:
    """Detect tables and return ``(TableData, bbox_rect, cells)`` per hit.

    The bbox lets the caller exclude table regions from prose emission;
    the raw ``cells`` (PyMuPDF ``tbl.extract()`` output, including the
    header row) lets the caller emit column-major if the prose path
    would otherwise collapse cells row-major.

    Strategy precedence: ``lines`` (ruled cells) → ``text`` (whitespace
    alignment, ≥3 rows × ≥2 cols). Lines hits require non-zero row/col
    counts only — letter-format compliance notices ship a single 3-col
    ruled rules-of-the-Law table that should be captured here.
    """
    import sys
    import io as _io

    found: list[tuple[TableData, fitz.Rect, list[list]]] = []
    pw, ph = page.rect.width, page.rect.height

    old_stdout = sys.stdout
    sys.stdout = _io.StringIO()
    try:
        found_lines: list = []
        try:
            found_lines = list(page.find_tables(strategy="lines").tables)
        except Exception:
            pass

        found_text: list = []
        if not found_lines:
            try:
                found_text = list(page.find_tables(strategy="text").tables)
            except Exception:
                pass

        for tbl in found_lines:
            if tbl.row_count < 1 or tbl.col_count < 1:
                continue
            extracted = tbl.extract()
            headers = [str(c) if c else "" for c in extracted[0]] if extracted else []
            rows = [[str(c) if c else "" for c in row] for row in extracted[1:]] if len(extracted) > 1 else []
            rect = fitz.Rect(tbl.bbox)
            pos = _normalise_rect(rect, pw, ph)
            found.append((
                TableData(headers=headers, rows=rows, position=pos, confidence=0.8),
                rect,
                extracted,
            ))

        for tbl in found_text:
            if tbl.row_count < 3 or tbl.col_count < 2:
                continue
            extracted = tbl.extract()
            headers = [str(c) if c else "" for c in extracted[0]] if extracted else []
            rows = [[str(c) if c else "" for c in row] for row in extracted[1:]] if len(extracted) > 1 else []
            rect = fitz.Rect(tbl.bbox)
            pos = _normalise_rect(rect, pw, ph)
            found.append((
                TableData(headers=headers, rows=rows, position=pos, confidence=0.6),
                rect,
                extracted,
            ))
    finally:
        sys.stdout = old_stdout

    return found


def _extract_tables_from_page(page: fitz.Page) -> list[TableData]:
    """Extract tables from a page using PyMuPDF's table finder.

    Backward-compatible thin wrapper around `_find_native_tables` —
    callers that only need the structured table data (e.g. the OCR-side
    scanned-machinewritten grid fallback) keep their existing signature.
    """
    return [td for td, _rect, _cells in _find_native_tables(page)]


def _emit_table_column_major(cells: list[list]) -> str:
    """Render a 2-D cell grid column-major: each column is a paragraph.

    Mirrors the OCR-side `_table_aware_text` column emission so the
    native and OCR paths produce comparable shapes for downstream prose
    consumers. Empty/None cells are dropped within each column.
    """
    if not cells:
        return ""
    n_cols = max((len(row) for row in cells), default=0)
    if n_cols == 0:
        return ""
    parts: list[str] = []
    for c in range(n_cols):
        col_cells: list[str] = []
        for row in cells:
            if c >= len(row):
                continue
            cell = row[c]
            if cell is None:
                continue
            s = str(cell).strip()
            if s:
                col_cells.append(s)
        if col_cells:
            parts.append("\n".join(col_cells))
    return "\n\n".join(parts)


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


# Form-field extraction lives in womblex.ingest.forms — re-exported below
# for backward compat (the legacy strategy modules import from here).
from womblex.ingest.forms import (  # noqa: E402,F401
    _extract_form_fields,
    _extract_form_pairs_from_lines,
    _extract_form_pairs_from_text,
    _extract_forms,
    _looks_like_form_label,
)


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


_FOOTER_PAGE_RE = re.compile(r"^\s*\d+\s*\|?\s*[Pｐ]\s*[aａ]\s*[gｇ]\s*[eｅ]\s*$", re.IGNORECASE)
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,3}\s*$")
_SIGNATURE_RE = re.compile(r"^\s*Yours\s+(sincerely|faithfully|truly)\b", re.IGNORECASE)
_SENTENCE_TERMINATORS = (".", "?", "!", ":")


def _classify_native_block(
    text: str, max_font_size: float, is_bold: bool, y_norm: float
) -> str:
    """Classify a native PDF text block by position, typography, and content.

    Reserves `caption` for downstream image-adjacency tagging — emitting
    `caption` from a font/length heuristic produces overwhelming false
    positives on letter-style prose (page footers, signatures, dates,
    section headings, page numbers all looked like captions under the old
    rule).
    """
    if _FOOTER_PAGE_RE.match(text) or _PAGE_NUMBER_RE.match(text):
        return "footer"
    if _SIGNATURE_RE.match(text):
        return "signature"
    if y_norm > 0.92 and len(text) < 100:
        return "footer"
    if y_norm < 0.08 and len(text) < 100:
        return "header"
    # Heading: explicit large size, OR bold short non-sentence text.
    # 14pt threshold lowered from 16 — letter headings are typically 13–14pt.
    if max_font_size >= 14:
        return "heading"
    if is_bold and len(text) <= 80 and not text.rstrip().endswith(_SENTENCE_TERMINATORS):
        return "heading"
    return "paragraph"


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

        # Collect text + typography signals from spans
        block_text = ""
        max_font_size = 0.0
        any_bold = False
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                block_text += span.get("text", "")
                fs = span.get("size", 0)
                if fs > max_font_size:
                    max_font_size = fs
                # PyMuPDF span flags: bit 4 (16) = bold
                if span.get("flags", 0) & 16:
                    any_bold = True

        block_text = block_text.strip()
        if not block_text:
            continue

        block_type = _classify_native_block(
            block_text, max_font_size, any_bold, pos.y
        )
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

# Page-footer artifact "<digit>lPage" / "<digit>IPage" -> "<digit> | Page"
# (OCR reads the pipe character as lowercase L or capital I). Catches both
# ASCII "Page" and the fullwidth Unicode variant "ｐａge" / "Ｐａｇｅ" seen
# on stylised letterheads (e.g. 00729 p4).
_FOOTER_PIPE_RE = re.compile(
    r"\b(\d+)\s*[lI]\s*[PＰｐ][aａＡ][gｇＧ][eｅＥ]\b"
)

# Body-context pipe-as-I in ACT Gov boilerplate footers like
# "GPO Box 158 Canberra ACT 2601 | phone: 132281 | www.act.gov.au".
# OCR reads the separator pipe as a capital I when it sits between a
# space and a lowercase keyword. Restricted to a fixed keyword set to
# avoid false positives on legitimate sentence-initial "I" + verb.
_BODY_PIPE_RE = re.compile(r" I (?=(?:phone|email|fax|www|http)\b)", re.IGNORECASE)

# Stylised-letterhead OCR artefacts cataloged from quality_audit.md plus
# additional patterns surfaced from CER labelling diffs. Substring replacements
# applied verbatim - list deliberately specific to avoid false positives.
_LETTERHEAD_FIXES: list[tuple[str, str]] = [
    ("(AcT)", "(ACT)"),
    ("Govermment", "Government"),
    ("Couse", "Cause"),
    ("OsHC", "OSHC"),
    ("Asurance", "Assurance"),
    ("Complionce", "Compliance"),
    ("Incorperated", "Incorporated"),
    ("Oofficers", "Officers"),
    ("ıi.", "ii."),  # italic-i (U+0131) misread as roman numeral
    ("ıii.", "iii."),
    ("ıv.", "iv."),
]
_DEAR_RE = re.compile(r"\bDeal(?=\s*\n)")

# Word-spacing repair for native PDF text-layer artefacts where space glyphs
# are missing from the font encoding. Patterns are conservative.
_MONTH_YEAR_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)(\d{4})\b"
)
_ACT_POSTCODE_RE = re.compile(r"\b(ACT)(\d{4})\b")
_COMMA_NOSPACE_RE = re.compile(r"(?<=\w),(?=[A-Za-z])")


def _normalise_text(text: str) -> str:
    """Apply targeted post-extraction corrections to a single page's text.

    Fixes known artefacts from government PDF document sets:
    - Broken ToUnicode font maps produce '$' or 'E' where 's' follows an apostrophe
    - Running footers rendered as spaced characters by OCR (single-line and split)
    - Corrupted '://' in URLs from broken font encoding
    - Stylised-letterhead OCR errors (Couse -> Cause, etc.)
    - Word-spacing collapse from PDF text-layer space-glyph gaps
    """
    # RES-001 extended: apostrophe + dollar/euro -> apostrophe + s
    text = text.replace("’ $", "’s")
    text = text.replace("’$", "’s")
    text = text.replace("' $", "'s")
    text = text.replace("'$", "'s")
    text = text.replace("’€", "’s")
    text = text.replace("'€", "'s")
    # RES-002: URL scheme corruption
    text = _URL_SCHEME_RE.sub("http://", text)
    # RES-003 extended: running page footers (single-line then split across two lines)
    text = _FOOTER_RE.sub("", text)
    text = _FOOTER_SPLIT_RE.sub("", text)
    # RES-004: page-footer pipe-as-l (1lPage -> 1 | Page)
    text = _FOOTER_PIPE_RE.sub(r"\1 | Page", text)
    # RES-004b: body-context pipe-as-I in ACT Gov footer separators
    text = _BODY_PIPE_RE.sub(" | ", text)
    # RES-005: stylised letterhead OCR fixes
    for bad, good in _LETTERHEAD_FIXES:
        text = text.replace(bad, good)
    text = _DEAR_RE.sub("Dear", text)
    # RES-006: word-spacing repair (native PDF text-layer)
    text = _MONTH_YEAR_RE.sub(r"\1 \2", text)
    text = _ACT_POSTCODE_RE.sub(r"\1 \2", text)
    text = _COMMA_NOSPACE_RE.sub(", ", text)
    return text


# ---------------------------------------------------------------------------
# Strategy resolution
# ---------------------------------------------------------------------------


def get_extractor(
    profile: DocumentProfile,
    dpi: int = 200,
    lang: str = "eng",
    engine: str = "paddleocr",
    engine_options: dict | None = None,
) -> ExtractionStrategy | PathExtractionStrategy:
    """Select the legacy extractor for non-orchestrator document types.

    Native and scanned PDFs are handled by `extract_pdf_with_plan`
    (per-page profile + orchestrator) — those types do not appear here.
    Only IMAGE, SPREADSHEET, DOCX, and TEXT still use the legacy strategy
    classes.
    """
    from womblex.ingest.strategies_scanned import ImageExtractor
    from womblex.ingest.strategies_file import DocxExtractor, TextExtractor
    from womblex.ingest.spreadsheet import SpreadsheetExtractor

    opts = engine_options or {}
    match profile.doc_type:
        case DocumentType.IMAGE:
            return ImageExtractor(dpi=dpi, lang=lang, engine=engine, engine_options=opts)
        case DocumentType.SPREADSHEET:
            return SpreadsheetExtractor(profile=profile)
        case DocumentType.DOCX:
            return DocxExtractor()
        case DocumentType.TEXT:
            return TextExtractor()
        case _:
            raise ValueError(
                f"get_extractor() only handles non-PDF types; got {profile.doc_type}. "
                "PDFs route through extract_pdf_with_plan."
            )


def extract_text(
    path: Path,
    profile: DocumentProfile,
    dpi: int = 200,
    lang: str = "eng",
    max_pages: int | None = None,
    engine: str = "paddleocr",
    engine_options: dict | None = None,
    spreadsheet_print: dict | None = None,
) -> list[ExtractionResult]:
    """Extract text from a document using the strategy matching its profile.

    Returns one ExtractionResult per logical unit. PDFs and DOCX return a
    single-element list. Spreadsheets return one element per row or sheet.

    When *max_pages* is set, PDF extraction is limited to the first N pages.

    ``engine`` selects the OCR backend (``paddleocr`` default, or
    ``deepseek-ocr`` for the local LLM). ``engine_options`` forwards
    engine-specific kwargs (e.g. ``model``, ``base_url``, ``prompt``).
    """
    # Non-PDF path-based extractors keep the legacy strategy switch for now;
    # the Phase 2 orchestrator covers PDFs only.
    if profile.doc_type in (DocumentType.SPREADSHEET, DocumentType.DOCX, DocumentType.TEXT):
        extractor: PathExtractionStrategy = get_extractor(  # type: ignore[assignment]
            profile, dpi=dpi, lang=lang, engine=engine, engine_options=engine_options,
        )
        logger.info(
            "strategy selected: doc=%s type=%s confidence=%.2f strategy=%s",
            path.name, profile.doc_type.value, profile.confidence,
            type(extractor).__name__,
        )
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

    # PDF: profile per page, summarise doc type, dispatch via orchestrator.
    from womblex.ingest.orchestrator import extract_pdf_with_plan

    doc = fitz.open(str(path))
    try:
        if max_pages is not None and doc.page_count > max_pages:
            doc.select(list(range(max_pages)))
        t0 = time.monotonic()
        result = extract_pdf_with_plan(
            doc, profile,
            dpi=dpi, lang=lang, engine=engine, engine_options=engine_options,
            filename=path.name, spreadsheet_print=spreadsheet_print,
        )
        elapsed = time.monotonic() - t0
        if result.metadata:
            result.metadata.processing_time = elapsed
        logger.info(
            "plan-driven extract: doc=%s type=%s pages=%d (%.2fs)",
            path.name, result.metadata.extraction_strategy if result.metadata else "?",
            result.page_count, elapsed,
        )
        _apply_normalisation_and_warnings(
            result, path, doc=doc, dpi=dpi, lang=lang,
            engine=engine, engine_options=engine_options,
        )
        return [result]
    finally:
        doc.close()


def _apply_normalisation_and_warnings(
    result: ExtractionResult,
    path: Path,
    doc: fitz.Document | None = None,
    dpi: int = 200,
    lang: str = "eng",
    engine: str = "paddleocr",
    engine_options: dict | None = None,
) -> None:
    """Normalise page text in-place, capture blank page warnings, OCR-fallback.

    When a page extracted by a native strategy comes back empty (typically an
    image-only cover page in an otherwise digital PDF), re-render that page
    and run the configured OCR engine over it. If OCR returns text, splice it
    back into the PageResult and tag the page method as ``native_ocr_fallback``.
    """
    for page in result.pages:
        if page.text.strip():
            page.text = _normalise_text(page.text)
            continue

        recovered = False
        if doc is not None and page.method == "native" and 0 <= page.page_number < doc.page_count:
            try:
                from womblex.ingest.strategies_scanned import _ocr_page
                ocr_text, _conf, _steps, _native_order = _ocr_page(
                    doc[page.page_number], dpi=dpi, lang=lang,
                    engine=engine, engine_options=engine_options,
                )
                if ocr_text.strip():
                    page.text = _normalise_text(ocr_text)
                    page.method = "native_ocr_fallback"
                    recovered = True
                    result.warnings.append(
                        f"blank page {page.page_number} recovered via OCR "
                        f"(strategy={result.method})"
                    )
                    logger.info(
                        "blank page recovered via OCR: doc=%s page=%d strategy=%s chars=%d",
                        path.name, page.page_number, result.method, len(ocr_text),
                    )
            except Exception as e:
                logger.warning(
                    "OCR fallback failed: doc=%s page=%d error=%s",
                    path.name, page.page_number, e,
                )

        if not recovered:
            warning = f"blank page {page.page_number} (method={result.method})"
            result.warnings.append(warning)
            logger.warning(
                "blank page extracted — possible silent failure: doc=%s page=%d method=%s",
                path.name,
                page.page_number,
                result.method,
            )
