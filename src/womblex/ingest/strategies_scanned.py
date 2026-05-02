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
    FormField,
    ImageData,
    PageResult,
    TableData,
    TextBlock,
    _avg_ocr_confidence,
    _build_text_blocks,
    _extract_form_fields,
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
    get_paddle_reader,
    preprocess_for_ocr,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared OCR helpers
# ---------------------------------------------------------------------------


def _ocr_page(
    page: fitz.Page, dpi: int, lang: str,
) -> tuple[str, float, list[str]]:
    """OCR a page: blur check -> deskew -> binarise -> PaddleOCR. Confidence 0-100."""
    import cv2

    from womblex.ingest.heuristics_cv2 import calculate_blur_score

    pix = page.get_pixmap(dpi=dpi)
    img = _pixmap_to_array(pix)

    # Pre-OCR blur check
    pre_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img
    blur = calculate_blur_score(pre_gray)

    gray, steps = preprocess_for_ocr(img)

    if blur is not None and blur < 50:
        steps.append("low_blur_warning")
        logger.warning("blurry page: doc=%s page=%d blur_score=%.1f", page.parent.name, page.number, blur)

    reader = get_paddle_reader(lang)
    results = reader.readtext(gray)
    text = "\n".join(r[1] for r in results if r[1].strip())
    avg_conf = _avg_ocr_confidence(results, scale=100)

    if avg_conf < 40.0:
        logger.warning("low OCR confidence: doc=%s page=%d confidence=%.1f", page.parent.name, page.number, avg_conf)

    return text, avg_conf, steps


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

    reader = get_paddle_reader(lang)

    for rect in candidate_rects:
        try:
            pix = page.get_pixmap(dpi=dpi, clip=rect)
            img = _pixmap_to_array(pix, drop_alpha=True)
            results = reader.readtext(img)
        except Exception as exc:
            logger.warning(
                "subpage OCR failed: page=%d err=%s", page.number, exc,
            )
            continue

        text = "\n".join(r[1] for r in results if r[1].strip())
        if not text.strip():
            continue
        avg_conf = _avg_ocr_confidence(results)

        blocks.append(TextBlock(
            text=text.strip(),
            position=_normalise_rect(rect, pw, ph),
            block_type="figure",
            confidence=avg_conf,
        ))
        steps.append("subpage_ocr")

    return blocks, steps


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
# 4. scanned_machinewritten
# ---------------------------------------------------------------------------


class ScannedMachinewrittenExtractor:
    """OCR extraction optimised for machine-typed scanned documents."""

    def __init__(self, dpi: int = 200, lang: str = "eng") -> None:
        self.dpi = dpi
        self.lang = lang

    def extract(self, doc: fitz.Document) -> ExtractionResult:
        from womblex.ingest.heuristics_cv2 import detect_table_grid

        pages: list[PageResult] = []
        all_blocks: list[TextBlock] = []
        all_tables: list[TableData] = []
        confidences: list[float] = []
        combined_steps: list[str] = []

        for page in doc:
            text, conf, steps = _ocr_page(page, self.dpi, self.lang)
            pages.append(PageResult(page_number=page.number, text=text, method="ocr"))
            confidences.append(conf)
            combined_steps.extend(steps)

            page_blocks, page_tables = _layout_blocks_and_tables(
                page, self.dpi, text, conf,
            )
            all_blocks.extend(page_blocks)
            all_tables.extend(page_tables)

            if not page_tables:
                gray = _page_to_gray(page, dpi=self.dpi)
                grid = detect_table_grid(gray)
                if grid.has_grid:
                    all_tables.extend(_extract_tables_from_page(page))

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        coverage = _text_coverage(pages)
        unique_steps = sorted(set(combined_steps))

        return ExtractionResult(
            pages=pages,
            method="scanned_machinewritten",
            tables=all_tables,
            text_blocks=all_blocks,
            metadata=ExtractionMetadata(
                extraction_strategy="scanned_machinewritten",
                confidence=avg_conf / 100 if avg_conf else 0.0,
                processing_time=0.0,
                page_count=len(doc),
                text_coverage=coverage,
                preprocessing_steps=unique_steps,
            ),
        )


# ---------------------------------------------------------------------------
# 5. scanned_handwritten
# ---------------------------------------------------------------------------


class ScannedHandwrittenExtractor:
    """OCR extraction for handwritten documents with confidence tracking."""

    def __init__(self, dpi: int = 200, lang: str = "eng") -> None:
        self.dpi = dpi
        self.lang = lang

    def extract(self, doc: fitz.Document) -> ExtractionResult:
        pages: list[PageResult] = []
        all_blocks: list[TextBlock] = []
        confidences: list[float] = []
        combined_steps: list[str] = []

        for page in doc:
            text, conf, steps = _ocr_page(page, self.dpi, self.lang)
            pages.append(PageResult(page_number=page.number, text=text, method="ocr"))
            confidences.append(conf)
            combined_steps.extend(steps)

            page_blocks, _ = _layout_blocks_and_tables(page, self.dpi, text, conf)
            all_blocks.extend(page_blocks)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        coverage = _text_coverage(pages)
        unique_steps = sorted(set(combined_steps))

        return ExtractionResult(
            pages=pages,
            method="scanned_handwritten",
            text_blocks=all_blocks,
            metadata=ExtractionMetadata(
                extraction_strategy="scanned_handwritten",
                confidence=avg_conf / 100 if avg_conf else 0.0,
                processing_time=0.0,
                page_count=len(doc),
                text_coverage=coverage,
                preprocessing_steps=unique_steps,
            ),
        )


# ---------------------------------------------------------------------------
# 6. scanned_mixed
# ---------------------------------------------------------------------------


class ScannedMixedExtractor:
    """Extract text from documents with both typed and handwritten content."""

    def __init__(self, dpi: int = 200, lang: str = "eng") -> None:
        self.dpi = dpi
        self.lang = lang

    def extract(self, doc: fitz.Document) -> ExtractionResult:
        from womblex.ingest.heuristics_cv2 import analyze_contour_complexity

        pages: list[PageResult] = []
        all_blocks: list[TextBlock] = []
        all_tables: list[TableData] = []
        confidences: list[float] = []
        combined_steps: list[str] = []
        typed_count = 0
        handwritten_count = 0

        for page in doc:
            gray = _page_to_gray(page, dpi=self.dpi)
            complexity = analyze_contour_complexity(gray)
            is_typed = complexity.regularity > 0.5

            if is_typed:
                typed_count += 1
            else:
                handwritten_count += 1

            text, conf, steps = _ocr_page(page, self.dpi, self.lang)
            pages.append(PageResult(page_number=page.number, text=text, method="ocr"))
            confidences.append(conf)
            combined_steps.extend(steps)

            page_blocks, page_tables = _layout_blocks_and_tables(
                page, self.dpi, text, conf,
            )
            all_tables.extend(page_tables)

            content_type = "typed" if is_typed else "handwritten"
            for block in page_blocks:
                block = TextBlock(
                    text=block.text,
                    position=block.position,
                    block_type=content_type,
                    confidence=block.confidence,
                )
                all_blocks.append(block)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        coverage = _text_coverage(pages)
        unique_steps = sorted(set(combined_steps))
        total = typed_count + handwritten_count
        content_mix = {}
        if total > 0:
            content_mix = {
                "typed": typed_count / total,
                "handwritten": handwritten_count / total,
            }

        return ExtractionResult(
            pages=pages,
            method="scanned_mixed",
            tables=all_tables,
            text_blocks=all_blocks,
            metadata=ExtractionMetadata(
                extraction_strategy="scanned_mixed",
                confidence=avg_conf / 100 if avg_conf else 0.0,
                processing_time=0.0,
                page_count=len(doc),
                text_coverage=coverage,
                preprocessing_steps=unique_steps,
                content_mix=content_mix,
            ),
        )


# ---------------------------------------------------------------------------
# 7. hybrid
# ---------------------------------------------------------------------------


class HybridExtractor:
    """Extract from documents mixing native text and scanned pages."""

    def __init__(self, dpi: int = 200, lang: str = "eng") -> None:
        self.dpi = dpi
        self.lang = lang

    def extract(self, doc: fitz.Document) -> ExtractionResult:
        pages: list[PageResult] = []
        all_tables: list[TableData] = []
        all_forms: list[FormField] = []
        all_images: list[ImageData] = []
        all_blocks: list[TextBlock] = []
        confidences: list[float] = []
        combined_steps: list[str] = []
        native_count = 0
        ocr_count = 0

        for page in doc:
            native_text = page.get_text("text", flags=fitz.TEXT_DEHYPHENATE).strip()
            is_native = len(native_text) > 100

            if is_native:
                native_count += 1
                page_blocks = _build_text_blocks(page)

                native_words = page.get_text("words")
                sub_blocks, sub_steps = _ocr_image_regions(
                    page, native_words, self.dpi, self.lang,
                )

                page_text = native_text
                method = "native"
                if sub_blocks:
                    page_text = native_text + "\n\n" + "\n\n".join(b.text for b in sub_blocks)
                    method = "native+ocr"
                    page_blocks.extend(sub_blocks)
                    combined_steps.extend(sub_steps)

                pages.append(PageResult(page_number=page.number, text=page_text, method=method))
                all_tables.extend(_extract_tables_from_page(page))
                all_forms.extend(_extract_form_fields(page))
                all_images.extend(_extract_images_from_page(page))
                all_blocks.extend(page_blocks)
                confidences.append(95.0)
            else:
                ocr_count += 1
                text, conf, steps = _ocr_page(page, self.dpi, self.lang)
                pages.append(PageResult(page_number=page.number, text=text, method="ocr"))
                confidences.append(conf)
                combined_steps.extend(steps)

                page_blocks, page_tables = _layout_blocks_and_tables(
                    page, self.dpi, text, conf,
                )
                all_blocks.extend(page_blocks)
                all_tables.extend(page_tables)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        coverage = _text_coverage(pages)
        unique_steps = sorted(set(combined_steps))
        total = native_count + ocr_count
        content_mix = {}
        if total > 0:
            content_mix = {"native": native_count / total, "scanned": ocr_count / total}

        return ExtractionResult(
            pages=pages,
            method="hybrid",
            tables=all_tables,
            forms=all_forms,
            images=all_images,
            text_blocks=all_blocks,
            metadata=ExtractionMetadata(
                extraction_strategy="hybrid",
                confidence=avg_conf / 100 if avg_conf else 0.0,
                processing_time=0.0,
                page_count=len(doc),
                text_coverage=coverage,
                preprocessing_steps=unique_steps,
                content_mix=content_mix,
            ),
        )


# ---------------------------------------------------------------------------
# 8. image
# ---------------------------------------------------------------------------


class ImageExtractor:
    """Extract text and metadata from standalone image files / image PDFs."""

    def __init__(self, dpi: int = 200, lang: str = "eng") -> None:
        self.dpi = dpi
        self.lang = lang

    def extract(self, doc: fitz.Document) -> ExtractionResult:
        from womblex.ingest.heuristics_cv2 import calculate_blur_score

        pages: list[PageResult] = []
        all_images: list[ImageData] = []
        all_blocks: list[TextBlock] = []
        confidences: list[float] = []
        steps: list[str] = []

        reader = get_paddle_reader(self.lang)

        for page in doc:
            gray = _page_to_gray(page, dpi=self.dpi)
            blur = calculate_blur_score(gray)
            if blur is not None and blur < 50:
                steps.append("low_blur_warning")

            pix = page.get_pixmap(dpi=self.dpi)
            img = _pixmap_to_array(pix)
            results = reader.readtext(img)
            text = "\n".join(r[1] for r in results if r[1].strip())
            avg_conf = _avg_ocr_confidence(results, scale=100)
            confidences.append(avg_conf)

            pages.append(PageResult(page_number=page.number, text=text, method="ocr"))
            all_images.extend(_extract_images_from_page(page))

            pw, ph = page.rect.width, page.rect.height
            if text.strip():
                all_blocks.append(
                    TextBlock(
                        text=text.strip(),
                        position=_normalise_bbox((0, 0, pw, ph), pw, ph),
                        block_type="paragraph",
                        confidence=avg_conf / 100 if avg_conf else 0.0,
                    )
                )

        avg_conf_doc = sum(confidences) / len(confidences) if confidences else 0.0
        coverage = _text_coverage(pages)
        unique_steps = sorted(set(steps))

        return ExtractionResult(
            pages=pages,
            method="image",
            images=all_images,
            text_blocks=all_blocks,
            metadata=ExtractionMetadata(
                extraction_strategy="image",
                confidence=avg_conf_doc / 100 if avg_conf_doc else 0.0,
                processing_time=0.0,
                page_count=len(doc),
                text_coverage=coverage,
                preprocessing_steps=unique_steps,
            ),
        )
