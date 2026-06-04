"""Text extraction strategies for different document types.

Each strategy implements the ExtractionStrategy protocol and returns
an ExtractionResult with per-page text, structured content, and metadata.
Output is designed to map directly to the Parquet output schema.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Protocol

import fitz
import numpy as np

from womblex.ingest.detect import DocumentProfile, DocumentType
from womblex.ingest.elements import Cell, Element, FieldEntry  # noqa: F401 (re-exported)
from womblex.ingest.views import (  # re-exported for back-compat
    ExtractionMetadata,  # noqa: F401
    ExtractionResult,
    FormField,  # noqa: F401
    ImageData,
    PageResult,
    Position,
    TableData,
    TextBlock,
)

logger = logging.getLogger(__name__)


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


def _count_blocks_in_bbox(page: fitz.Page, bbox: fitz.Rect) -> int:
    """Count `get_text("dict")` text blocks whose centre falls inside ``bbox``.

    Used as a cross-check against PyMuPDF's `find_tables` over-firing: a
    real table has at least one natural text block per row (each cell row
    is its own paragraph in PyMuPDF's decomposition), while a prose-as-
    table over-claims rows by carving sub-block whitespace into pseudo-
    rows. The block count here is therefore a structural ceiling on how
    many real rows the table region can contain.
    """
    count = 0
    raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        bx0, by0, bx1, by1 = block.get("bbox", (0, 0, 0, 0))
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        if bbox.x0 <= cx <= bbox.x1 and bbox.y0 <= cy <= bbox.y1:
            count += 1
    return count


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

    Cross-check gate: a candidate is rejected when fewer natural text
    blocks fall inside its bbox than the row count it claims. PyMuPDF
    over-fires on prose-with-indents (text strategy) and on redaction
    boxes / form rules (lines strategy); both failure modes inflate row
    count by carving sub-block whitespace into pseudo-rows. Real tables
    decompose into ≥1 block per row in `get_text("dict")`.
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
            rect = fitz.Rect(tbl.bbox)
            n_rows = len(extracted)
            if n_rows and _count_blocks_in_bbox(page, rect) < n_rows:
                continue
            headers = [str(c) if c else "" for c in extracted[0]] if extracted else []
            rows = [[str(c) if c else "" for c in row] for row in extracted[1:]] if len(extracted) > 1 else []
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
            rect = fitz.Rect(tbl.bbox)
            n_rows = len(extracted)
            if n_rows and _count_blocks_in_bbox(page, rect) < n_rows:
                continue
            headers = [str(c) if c else "" for c in extracted[0]] if extracted else []
            rows = [[str(c) if c else "" for c in row] for row in extracted[1:]] if len(extracted) > 1 else []
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
    _extract_form_pairs_from_regions,
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

# Bare "1. " prefix excluded — in this corpus it's almost always a numbered
# paragraph, not a list item.
_LIST_ITEM_RE = re.compile(r"^\s*(?:\([a-z]\)|\([ivx]+\)|\(\d+\)|[•·]|[-*]\s)")

_SENTENCE_TERMINATORS = (".", "?", "!", ":")


def _classify_native_block(
    text: str, max_font_size: float, is_bold: bool, y_norm: float
) -> str:
    """Classify a native PDF text block by position, typography, and content.

    Reserves `caption` and `signature` — see docs/decisions.md "Element-kind
    classification" for why a font/length heuristic for `caption` and a closing-
    phrase regex for `signature` were both removed (false-positive heavy on
    letter-style prose).
    """
    if _FOOTER_PAGE_RE.match(text) or _PAGE_NUMBER_RE.match(text):
        return "footer"
    if y_norm > 0.92 and len(text) < 100:
        return "footer"
    if y_norm < 0.08 and len(text) < 100:
        return "header"
    if _LIST_ITEM_RE.match(text):
        return "list_item"
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
    """Extract a document using the strategy matching its profile.

    Returns a list of ExtractionResults wrapped from each extractor's
    return value. PDFs, DOCX, and spreadsheets each return a single-element
    list (one result per source). The list shape is retained for call-site
    symmetry — callers iterate ``for extraction in extractions``.

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
    """OCR-fallback for blank native pages; emit blank-page warnings.

    Text is verbatim from the producing extractor — no post-processing.
    When a native-strategy page comes back empty (typically an image-only
    cover page in an otherwise digital PDF), re-render that page and run
    the configured OCR engine over it. If OCR returns text, splice it
    back into the PageResult and tag the page method as
    ``native_ocr_fallback``.
    """
    for page in result.pages:
        if page.text.strip():
            continue

        recovered = False
        if doc is not None and page.method == "native" and 0 <= page.page_number < doc.page_count:
            try:
                from womblex.ingest.strategies_scanned import _ocr_page
                ocr_text, _conf, _steps, _native_order, _regions, _pix = _ocr_page(
                    doc[page.page_number], dpi=dpi, lang=lang,
                    engine=engine, engine_options=engine_options,
                )
                if ocr_text.strip():
                    page.text = ocr_text
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
