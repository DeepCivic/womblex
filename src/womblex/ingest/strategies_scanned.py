"""Extraction strategies for scanned and hybrid PDF documents.

Covers SCANNED_MACHINEWRITTEN, SCANNED_HANDWRITTEN, SCANNED_MIXED,
HYBRID, and IMAGE document types — all requiring OCR via PaddleOCR.
"""

from __future__ import annotations

import logging

import fitz

from womblex.ingest.extract import (
    ExtractionMetadata,
    ExtractionResult,
    ImageData,
    PageResult,
    TableData,
    TextBlock,
    _build_text_blocks,
    _extract_images_from_page,
    _extract_tables_from_page,
    _normalise_bbox,
    _normalise_rect,
    _ocr_text_block,
    _page_to_gray,
    _pixmap_to_array,
    _text_coverage,
)
from womblex.ingest.paddle_ocr import (
    get_layout_analyzer,
    get_ocr_reader,
    preprocess_for_ocr,
)

logger = logging.getLogger(__name__)


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
    items: list[tuple[str, float, float]] = []
    heights: list[float] = []
    for r in regions:
        if not r.text.strip():
            continue
        xs = [p[0] for p in r.bbox]
        ys = [p[1] for p in r.bbox]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        items.append((r.text, cx, cy))
        heights.append(max(ys) - min(ys))
    if not items:
        return ""

    avg_h = sum(heights) / len(heights) if heights else 1.0
    threshold = avg_h * line_tolerance
    items.sort(key=lambda t: (t[2], t[1]))

    rows: list[list[tuple[str, float, float]]] = [[items[0]]]
    for item in items[1:]:
        if abs(item[2] - rows[-1][0][2]) <= threshold:
            rows[-1].append(item)
        else:
            rows.append([item])

    out_lines: list[str] = []
    for row in rows:
        row.sort(key=lambda t: t[1])
        out_lines.append(" ".join(t[0] for t in row))
    return "\n".join(out_lines)


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
    items: list[tuple[str, float, float]] = []
    heights: list[float] = []
    for r in regions:
        if not r.text.strip():
            continue
        xs = [p[0] for p in r.bbox]
        ys = [p[1] for p in r.bbox]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        items.append((r.text, cx, cy))
        heights.append(max(ys) - min(ys))
    if not items:
        return ""

    avg_h = sum(heights) / len(heights) if heights else 1.0
    row_threshold = avg_h * line_tolerance
    items.sort(key=lambda t: (t[2], t[1]))

    rows: list[list[tuple[str, float, float]]] = [[items[0]]]
    for item in items[1:]:
        if abs(item[2] - rows[-1][0][2]) <= row_threshold:
            rows[-1].append(item)
        else:
            rows.append([item])
    for row in rows:
        row.sort(key=lambda t: t[1])

    out_blocks: list[str] = []
    i = 0
    while i < len(rows):
        end = _find_table_end(rows, i, table_min_cols, table_min_start_rows, avg_h)
        if end > i + 1:  # at least 2 rows constitute a meaningful table
            out_blocks.append(_emit_columns(rows[i:end], avg_h))
            i = end
        else:
            out_blocks.append(" ".join(t[0] for t in rows[i]))
            i += 1
    return "\n".join(out_blocks)


def _find_table_end(rows, start: int, min_cols: int, min_start_rows: int, avg_h: float) -> int:
    """Return the row index just past a detected table run, or ``start`` if none."""
    if start + min_start_rows > len(rows):
        return start
    if not all(len(rows[start + k]) >= min_cols for k in range(min_start_rows)):
        return start

    start_items = [it for k in range(min_start_rows) for it in rows[start + k]]
    col_gap = max(avg_h * 3.0, 30.0)
    col_centers = _cluster_x_centroids([it[1] for it in start_items], col_gap)
    if len(col_centers) < min_cols:
        return start  # the start rows didn't cluster into enough distinct columns

    fit_dist = max(avg_h * 5.0, 50.0)
    j = start + min_start_rows
    while j < len(rows):
        row = rows[j]
        # Continuation requires multiple items AND every item near a column.
        if len(row) < 2:
            break
        if not all(min(abs(it[1] - c) for c in col_centers) <= fit_dist for it in row):
            break
        j += 1
    return j


def _cluster_x_centroids(xs: list[float], gap_threshold: float) -> list[float]:
    """1-D agglomerative clustering: items within ``gap_threshold`` group together."""
    if not xs:
        return []
    xs_sorted = sorted(xs)
    clusters: list[list[float]] = [[xs_sorted[0]]]
    for x in xs_sorted[1:]:
        if x - clusters[-1][-1] > gap_threshold:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    return [sum(c) / len(c) for c in clusters]


def _emit_columns(table_rows, avg_h: float) -> str:
    """Emit a detected table block column-major, with blank lines between columns."""
    all_items = [it for row in table_rows for it in row]
    if not all_items:
        return ""
    col_gap = max(avg_h * 3.0, 30.0)
    col_centers = _cluster_x_centroids([it[1] for it in all_items], col_gap)
    if not col_centers:
        return " ".join(t[0] for t in all_items)

    columns: list[list[tuple[str, float, float]]] = [[] for _ in col_centers]
    for it in all_items:
        idx = min(range(len(col_centers)), key=lambda k: abs(it[1] - col_centers[k]))
        columns[idx].append(it)

    parts: list[str] = []
    for col in columns:
        col.sort(key=lambda t: t[2])
        parts.append(" ".join(t[0] for t in col))
    return "\n\n".join(p for p in parts if p)


def _ocr_page(
    page: fitz.Page,
    dpi: int,
    lang: str,
    engine: str = "paddleocr",
    engine_options: dict | None = None,
) -> tuple[str, float, list[str], bool]:
    """OCR a page and return ``(text, confidence_0_100, steps, reading_order_native)``.

    Region-based engines (paddleocr) get the standard preprocess pipeline
    (deskew + binarise). LLM-based engines (deepseek-ocr) skip preprocessing
    because they ingest the colour render directly and resolve reading order
    themselves.
    """
    import cv2

    from womblex.ingest.heuristics_cv2 import calculate_blur_score

    pix = page.get_pixmap(dpi=dpi)
    img = _pixmap_to_array(pix)

    # Pre-OCR blur check (cheap, useful for any engine)
    pre_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img
    blur = calculate_blur_score(pre_gray)

    reader = get_ocr_reader(engine=engine, lang=lang, **(engine_options or {}))

    # LLM engines work better on the unmodified colour render; binarised
    # output throws off models trained on natural document images.
    if engine.lower() in {"paddleocr", "paddle", "rapidocr"}:
        ocr_input, steps = preprocess_for_ocr(img)
    else:
        ocr_input = img
        steps = ["llm_native_input"]

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

    return text, avg_conf, steps, page_result.reading_order_native


def _word_inside(word: tuple, rect: fitz.Rect) -> bool:
    """Check if a PyMuPDF word tuple's midpoint falls inside a rect."""
    cx = (word[0] + word[2]) / 2
    cy = (word[1] + word[3]) / 2
    return rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1


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

    Sub-page conditional OCR — used by HybridExtractor when a page has a
    native text layer but also embeds image content (e.g. a redacted form
    scan inserted into an otherwise-native report).  Only image rects with
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
        if not text.strip():
            continue

        blocks.append(TextBlock(
            text=text.strip(),
            position=_normalise_rect(rect, pw, ph),
            block_type="figure",
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
) -> tuple[list[TextBlock], list[TableData]]:
    """Run YOLO layout analysis on a page, returning typed TextBlocks and tables.

    Falls back to a single paragraph block if the layout model is unavailable.
    """
    blocks: list[TextBlock] = []
    tables: list[TableData] = []

    try:
        analyzer = get_layout_analyzer()
        pix = page.get_pixmap(dpi=dpi)
        img = _pixmap_to_array(pix)

        regions = analyzer.analyze(img)
        if not regions:
            raise RuntimeError("no layout regions detected")

        for region in regions:
            rx0, ry0, rx1, ry1 = region.bbox
            pos = _normalise_bbox((rx0, ry0, rx1, ry1), float(pix.width), float(pix.height))

            if region.block_type == "table":
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
                block = TextBlock(
                    text=block.text,
                    position=block.position,
                    block_type=dominant.block_type,
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
