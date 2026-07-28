"""Extraction strategies for scanned and hybrid PDF documents.

Covers SCANNED_MACHINEWRITTEN, SCANNED_HANDWRITTEN, SCANNED_MIXED,
HYBRID, and IMAGE document types — all requiring OCR (PaddleOCR by
default, or an LLM/VLM engine such as Mistral OCR via AWS Bedrock).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import fitz

from womblex.ingest.elements import TEXT_KINDS
from womblex.ingest.extract import (
    ExtractionMetadata,
    ExtractionResult,
    PageResult,
    TableData,
    TextBlock,
    _extract_images_from_page,
    _normalise_bbox,
    _normalise_rect,
    _ocr_text_block,
    _page_to_gray,
    _pixmap_to_array,
    _text_coverage,
)
from womblex.ingest.interfaces.protocols import OCRRegionResult
from womblex.ingest.ocr_tables import regions_in_rect, span_from_region
from womblex.ingest.paddle_ocr import (
    get_layout_analyzer,
    get_ocr_reader,
    is_llm_engine,
    preprocess_for_ocr,
)
from womblex.ingest.table_grid import Span, cluster_x_centroids, rows_from_spans

logger = logging.getLogger(__name__)

# A full-page OCR block whose layout kind is a non-text region (figure,
# caption, …) is reclassified to a text paragraph once its OCR yields at
# least this many words.  Full-page document scans produce a paragraph+ of
# coherent text and must reach chunking (the non-text kinds are excluded
# from TEXT_KINDS); genuine figures yield only incidental words.  See K9-fig.
_OCR_TEXT_KIND_MIN_WORDS = 5


# ---------------------------------------------------------------------------
# Shared OCR helpers
# ---------------------------------------------------------------------------


def _spatial_sort_regions(regions, line_tolerance: float = 0.5) -> str:
    """Order OCR regions row-major and join into newline-delimited text.

    Within a row (regions whose y-centroids cluster within ``line_tolerance``
    × average height) regions are joined left-to-right with spaces.  Rows are
    joined with newlines.  Fixes the column-mixing failure mode where paddle
    returns table cells in detection order.
    """
    spans = [span_from_region(r) for r in regions if r.text.strip()]
    rows, _avg_h = rows_from_spans(spans, line_tolerance=line_tolerance)
    return "\n".join(" ".join(s.text for s in row) for row in rows)


def _table_aware_text(
    regions,
    line_tolerance: float = 0.5,
    table_min_cols: int = 3,
    table_min_start_rows: int = 2,
) -> str:
    """Spatial-sort regions, then re-emit detected table blocks column-major.

    Detection uses a two-phase rule:

    - **Start**: ``table_min_start_rows`` consecutive rows each with
      ``table_min_cols`` or more items establish the table and its column
      structure (cluster x-centroids of all items in the start rows).
    - **Continue**: subsequent rows are absorbed into the table while every
      item falls near one of the established column centres. A row with a
      single item, or any item too far from any column centre, ends the run
      — the next iteration starts re-evaluating from there.

    Detected blocks are emitted column-major (each column joined into its
    own paragraph). Everything else uses the row-major reading order from
    ``_spatial_sort_regions``.
    """
    spans = [span_from_region(r) for r in regions if r.text.strip()]
    rows, avg_h = rows_from_spans(spans, line_tolerance=line_tolerance)
    if not rows:
        return ""

    out_blocks: list[str] = []
    i = 0
    while i < len(rows):
        end = _find_table_end(rows, i, table_min_cols, table_min_start_rows, avg_h)
        if end > i + 1:  # at least 2 rows constitute a meaningful table
            out_blocks.append(_emit_columns(rows[i:end], avg_h))
            i = end
        else:
            out_blocks.append(" ".join(s.text for s in rows[i]))
            i += 1
    return "\n".join(out_blocks)


def _find_table_end(
    rows: list[list[Span]], start: int, min_cols: int, min_start_rows: int, avg_h: float,
) -> int:
    """Return the row index just past a detected table run, or ``start`` if none."""
    if start + min_start_rows > len(rows):
        return start
    if not all(len(rows[start + k]) >= min_cols for k in range(min_start_rows)):
        return start

    start_items = [s for k in range(min_start_rows) for s in rows[start + k]]
    col_gap = max(avg_h * 3.0, 30.0)
    col_centers = cluster_x_centroids([s.cx for s in start_items], col_gap)
    if len(col_centers) < min_cols:
        return start  # the start rows didn't cluster into enough distinct columns

    fit_dist = max(avg_h * 5.0, 50.0)
    j = start + min_start_rows
    while j < len(rows):
        row = rows[j]
        # Continuation requires multiple items AND every item near a column.
        if len(row) < 2:
            break
        if not all(min(abs(s.cx - c) for c in col_centers) <= fit_dist for s in row):
            break
        j += 1
    return j


def _emit_columns(table_rows: list[list[Span]], avg_h: float) -> str:
    """Emit a detected table block column-major, with blank lines between columns."""
    all_items = [s for row in table_rows for s in row]
    if not all_items:
        return ""
    col_gap = max(avg_h * 3.0, 30.0)
    col_centers = cluster_x_centroids([s.cx for s in all_items], col_gap)
    if not col_centers:
        return " ".join(s.text for s in all_items)

    columns: list[list[Span]] = [[] for _ in col_centers]
    for s in all_items:
        idx = min(range(len(col_centers)), key=lambda k: abs(s.cx - col_centers[k]))
        columns[idx].append(s)

    parts: list[str] = []
    for col in columns:
        col.sort(key=lambda s: s.cy)
        parts.append(" ".join(s.text for s in col))
    return "\n\n".join(p for p in parts if p)


def _ocr_page(
    page: fitz.Page,
    dpi: int,
    lang: str,
    engine: str = "paddleocr",
    engine_options: dict | None = None,
) -> tuple[str, float, list[str], bool, list, tuple[int, int]]:
    """OCR a page and return text plus per-region detections.

    Returns ``(text, confidence_0_100, steps, reading_order_native,
    regions, (pix_width, pix_height))`` where ``regions`` is the
    per-detection list from the OCR engine (empty for LLM-OCR engines
    that resolve reading order natively and only emit markdown). Region
    bboxes are in image-pixel coords at *dpi*; callers normalise by the
    returned ``pix_width`` / ``pix_height`` to get 0-1 floats.

    Region-based engines (paddleocr) get the standard preprocess pipeline
    (deskew + binarise). LLM/VLM engines (mistral-ocr, ollama) skip
    preprocessing because they ingest the colour render directly and
    resolve reading order themselves.
    """
    import cv2

    from womblex.ingest.heuristics_cv2 import calculate_blur_score

    pix = page.get_pixmap(dpi=dpi)
    img = _pixmap_to_array(pix)
    pix_dims = (int(pix.width), int(pix.height))

    # Pre-OCR blur check (cheap, useful for any engine)
    pre_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img
    blur = calculate_blur_score(pre_gray)

    reader = get_ocr_reader(engine=engine, lang=lang, **(engine_options or {}))

    # LLM/VLM engines work better on the unmodified colour render; binarised
    # output throws off models trained on natural document images.
    if is_llm_engine(engine):
        ocr_input = img
        steps = ["llm_native_input"]
    else:
        ocr_input, steps = preprocess_for_ocr(img)

    if blur is not None and blur < 50:
        steps.append("low_blur_warning")
        logger.warning("blurry page: doc=%s page=%d blur_score=%.1f", page.parent.name, page.number, blur)

    page_result = reader.read_page(ocr_input)

    if page_result.reading_order_native and page_result.markdown is not None:
        text = page_result.markdown.strip()
        avg_conf = page_result.confidence * 100.0
    else:
        text = _table_aware_text(page_result.regions)
        avg_conf = page_result.confidence * 100.0

    if avg_conf < 40.0:
        logger.warning("low OCR confidence: doc=%s page=%d confidence=%.1f", page.parent.name, page.number, avg_conf)

    return (
        text, avg_conf, steps, page_result.reading_order_native,
        page_result.regions, pix_dims,
    )


def _word_inside(word: tuple, rect: fitz.Rect) -> bool:
    """Check if a PyMuPDF word tuple's midpoint falls inside a rect."""
    cx = (word[0] + word[2]) / 2
    cy = (word[1] + word[3]) / 2
    return bool(rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1)


def _ocr_region_block_type(
    text: str,
    layout_kind: str = "figure",
    min_words: int = _OCR_TEXT_KIND_MIN_WORDS,
) -> str:
    """Reclassify an OCR'd region by text volume when its layout kind is non-text.

    Full-page scans collapse a whole page's OCR into one block whose kind is the
    dominant layout region's — sometimes ``figure``. A figure block holds a
    paragraph+ of coherent text yet is excluded from chunking (figure ∉
    TEXT_KINDS), silently losing document content. When the OCR yields >=
    ``min_words`` words the block is promoted to ``paragraph``; sparser output
    (page numbers, bare logos) keeps its ``layout_kind``. Text kinds (incl.
    ``caption``) and ``table`` pass through unchanged.
    """
    if layout_kind in TEXT_KINDS or layout_kind == "table":
        return layout_kind
    return "paragraph" if len(text.split()) >= min_words else layout_kind


# max_overlapping_words=2 tolerates incidental native glyphs (e.g. a caption
# label) inside an image rect; anything more means the text layer already
# covers the region.  min_rect_size=50 px filters icons/bullets that won't
# OCR meaningfully at 72 dpi.
def _ocr_image_regions(
    page: fitz.Page,
    native_words: list,
    dpi: int,
    lang: str,
    engine: str = "paddleocr",
    engine_options: dict | None = None,
    max_overlapping_words: int = 2,
    min_rect_size: int = 50,
) -> tuple[list[TextBlock], list[str]]:
    """OCR image rects on a native page that have no overlapping native text.

    Sub-page conditional OCR — used by the orchestrator's `_apply_native_page`
    operation when a page has a native text layer but also embeds image content
    (e.g. a redacted form scan inserted into an otherwise-native report). Only
    image rects with
    at most ``max_overlapping_words`` native words inside are OCR'd, so
    pages where the native text layer already covers everything do not
    incur OCR cost.
    """
    blocks: list[TextBlock] = []
    steps: list[str] = []

    pw, ph = page.rect.width, page.rect.height
    if pw <= 0 or ph <= 0:
        return blocks, steps

    image_rects: list[fitz.Rect] = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            image_rects.extend(page.get_image_rects(xref))
        except Exception:
            continue
    if not image_rects:
        return blocks, steps

    candidate_rects: list[fitz.Rect] = []
    for rect in image_rects:
        if rect.width < min_rect_size or rect.height < min_rect_size:
            continue
        overlap = sum(1 for w in native_words if _word_inside(w, rect))
        if overlap <= max_overlapping_words:
            candidate_rects.append(rect)
    if not candidate_rects:
        return blocks, steps

    reader = get_ocr_reader(engine=engine, lang=lang, **(engine_options or {}))

    for rect in candidate_rects:
        try:
            pix = page.get_pixmap(dpi=dpi, clip=rect)
            img = _pixmap_to_array(pix, drop_alpha=True)
            page_result = reader.read_page(img)
        except Exception as exc:
            logger.warning(
                "subpage OCR failed: page=%d err=%s", page.number, exc,
            )
            continue

        if page_result.reading_order_native and page_result.markdown is not None:
            text = page_result.markdown.strip()
        else:
            text = "\n".join(r.text for r in page_result.regions if r.text.strip())
        text = text.strip()
        if not text:
            continue

        blocks.append(TextBlock(
            text=text,
            position=_normalise_rect(rect, pw, ph),
            block_type=_ocr_region_block_type(text),
            confidence=page_result.confidence,
        ))
        steps.append("subpage_ocr")

    return blocks, steps


def _markdown_page_block(
    page: fitz.Page, text: str, conf: float,
) -> list[TextBlock]:
    """Wrap LLM-derived markdown as a single page-spanning paragraph block.

    LLM engines resolve reading order natively, so per-region layout
    sorting is skipped. The full page text becomes one block at the
    page rect — downstream chunking/normalisation operates as usual.
    """
    if not text.strip():
        return []
    pw, ph = page.rect.width, page.rect.height
    return [TextBlock(
        text=text.strip(),
        position=_normalise_bbox((0, 0, pw, ph), pw, ph),
        block_type="paragraph",
        confidence=conf / 100.0 if conf > 1.0 else conf,
    )]


def _layout_blocks_and_tables(
    page: fitz.Page,
    dpi: int,
    text: str,
    conf: float,
    ocr_regions: Sequence[OCRRegionResult] | None = None,
    ocr_pix_dims: tuple[int, int] | None = None,
) -> tuple[list[TextBlock], list[TableData]]:
    """Run YOLO layout analysis on a page, returning typed TextBlocks and tables.

    Falls back to a single paragraph block if the layout model is unavailable.

    ``ocr_regions`` / ``ocr_pix_dims`` carry the per-detection OCR output
    that produced *text*, in image-pixel coords at *dpi* — the raw
    material for reconstructing cells inside a detected table rect. They
    are supplied together or not at all: regions without their render
    dimensions cannot be checked against this pass's own render, so they
    are dropped. Only region-based engines (paddleocr) supply them; LLM/VLM
    engines resolve reading order natively, return no regions, and are
    dispatched to ``_markdown_page_block`` instead, so they never reach
    this function. See docs/table-cell-reconstruction-plan.md (A0) —
    ``tables`` is still returned empty until the reconstructor lands.
    """
    blocks: list[TextBlock] = []
    tables: list[TableData] = []

    try:
        analyzer = get_layout_analyzer()
        pix = page.get_pixmap(dpi=dpi)
        img = _pixmap_to_array(pix)

        # The OCR render and this layout render are the same page at the
        # same dpi, so their pixel spaces coincide and region bboxes can be
        # intersected with layout rects directly. Verify rather than assume:
        # unless the OCR render's dimensions are supplied *and* match, the
        # coordinates are not known to be comparable and the regions are
        # dropped. That costs reconstruction inputs but never produces a
        # mis-binned grid.
        cell_source = list(ocr_regions or ())
        layout_dims = (int(pix.width), int(pix.height))
        ocr_dims = tuple(ocr_pix_dims) if ocr_pix_dims is not None else None
        if cell_source and ocr_dims != layout_dims:
            logger.warning(
                "OCR/layout renders not comparable, dropping cell regions: "
                "page=%d ocr_dims=%s layout_dims=%s",
                page.number, ocr_dims, layout_dims,
            )
            cell_source = []

        regions = analyzer.analyze(img)
        if not regions:
            raise RuntimeError("no layout regions detected")

        for region in regions:
            rx0, ry0, rx1, ry1 = region.bbox
            pos = _normalise_bbox((rx0, ry0, rx1, ry1), float(pix.width), float(pix.height))

            if region.block_type == "table":
                # Reconstruction inputs, logged so the size of the gap is
                # traceable per page before the reconstructor exists. Gated:
                # the intersection is real work, not a format string.
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "layout table region: page=%d confidence=%.2f ocr_regions=%d",
                        page.number, region.confidence,
                        len(regions_in_rect(cell_source, (rx0, ry0, rx1, ry1))),
                    )
                blocks.append(TextBlock(
                    text="[TABLE]",
                    position=pos,
                    block_type="table",
                    confidence=region.confidence,
                ))
            else:
                blocks.append(TextBlock(
                    text="",  # layout region text not yet segmented from OCR output
                    position=pos,
                    block_type=region.block_type,
                    confidence=region.confidence,
                ))

        # If layout produced blocks but none have text, fall back to single block
        if blocks and not any(b.text.strip() for b in blocks if b.block_type != "table"):
            block = _ocr_text_block(page, text, conf)
            if block:
                dominant = max(regions, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
                # The whole page's OCR text is being collapsed into this one
                # block. If the dominant region is a non-text kind (figure) but
                # the page yielded substantial prose, it is a full-page scan,
                # not a figure — tag it paragraph so it is not silently dropped
                # from chunking. K9-fig.
                block = TextBlock(
                    text=block.text,
                    position=block.position,
                    block_type=_ocr_region_block_type(block.text, dominant.block_type),
                    confidence=block.confidence,
                )
                return [block], tables

    except (FileNotFoundError, Exception):
        pass

    if not blocks:
        block = _ocr_text_block(page, text, conf)
        if block:
            blocks = [block]

    return blocks, tables


# ---------------------------------------------------------------------------
# 8. image
# ---------------------------------------------------------------------------


class ImageExtractor:
    """Extract text and metadata from standalone image files / image PDFs."""

    def __init__(
        self,
        dpi: int = 200,
        lang: str = "eng",
        engine: str = "paddleocr",
        engine_options: dict | None = None,
    ) -> None:
        self.dpi = dpi
        self.lang = lang
        self.engine = engine
        self.engine_options = engine_options or {}

    def extract(self, doc: fitz.Document) -> ExtractionResult:
        from womblex.ingest.elements import Element
        from womblex.ingest.heuristics_cv2 import calculate_blur_score

        pages: list[PageResult] = []
        elements: list[Element] = []
        order = 0
        confidences: list[float] = []
        steps: list[str] = []

        reader = get_ocr_reader(engine=self.engine, lang=self.lang, **self.engine_options)

        for page in doc:
            gray = _page_to_gray(page, dpi=self.dpi)
            blur = calculate_blur_score(gray)
            if blur is not None and blur < 50:
                steps.append("low_blur_warning")

            pix = page.get_pixmap(dpi=self.dpi)
            img = _pixmap_to_array(pix)
            page_result = reader.read_page(img)
            if page_result.reading_order_native and page_result.markdown is not None:
                text = page_result.markdown.strip()
            else:
                text = "\n".join(r.text for r in page_result.regions if r.text.strip())
            avg_conf = page_result.confidence * 100.0
            confidences.append(avg_conf)

            pages.append(PageResult(page_number=page.number, text=text, method="ocr"))

            pw, ph = page.rect.width, page.rect.height
            page_conf = avg_conf / 100 if avg_conf else 0.0
            if text.strip():
                elements.append(Element(
                    order=order, kind="paragraph", extractor="ocr_paddle",
                    page=page.number,
                    bbox=_normalise_bbox((0, 0, pw, ph), pw, ph),
                    text=text.strip(), confidence=page_conf,
                ))
                order += 1
            for im in _extract_images_from_page(page):
                elements.append(Element(
                    order=order, kind="image", extractor="figure_image",
                    page=page.number, bbox=im.position,
                    alt_text=im.alt_text, confidence=im.confidence,
                ))
                order += 1

        avg_conf_doc = sum(confidences) / len(confidences) if confidences else 0.0
        coverage = _text_coverage(pages)
        unique_steps = sorted(set(steps))

        return ExtractionResult(
            pages=pages,
            elements=elements,
            method="image",
            metadata=ExtractionMetadata(
                extraction_strategy="image",
                confidence=avg_conf_doc / 100 if avg_conf_doc else 0.0,
                processing_time=0.0,
                page_count=len(doc),
                text_coverage=coverage,
                preprocessing_steps=unique_steps,
            ),
        )
