"""Plan-driven extraction orchestrator (Phase 2).

Walks per-page `PageProfile`s and dispatches page-level operations,
collapsing the previous document-level strategy switch
(NativeNarrative / NativeWithStructured / Structured / Hybrid /
ScannedMachinewritten / ScannedHandwritten / ScannedMixed) into one
loop with mode-based dispatch.

Per-page operations are the same primitives Phase 1 produced:
- native pages: extract_page_text + _build_text_blocks + _extract_forms
  + _extract_tables_from_page + _extract_images_from_page
- OCR pages: _ocr_page + _extract_form_pairs_from_lines + either
  _markdown_page_block (markdown engines) or _layout_blocks_and_tables
- mixed-typed docs: per-page typed/handwritten classification on OCR
  pages
- scanned-machinewritten: table-grid fallback when layout pass finds
  none

The orchestrator accepts a doc-level type hint (from `summarise_doc_type`
or the legacy detector) for two narrow uses: enabling the mixed-typed
classification and the scanned-machinewritten table-grid fallback. All
other dispatch is page-level.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import fitz

from womblex.ingest.detect import DocumentProfile, DocumentType
from womblex.ingest.extract import (
    ExtractionMetadata,
    ExtractionResult,
    FormField,
    ImageData,
    PageResult,
    TableData,
    TextBlock,
    _build_text_blocks,
    _emit_table_column_major,
    _extract_form_pairs_from_lines,
    _extract_forms,
    _extract_images_from_page,
    _extract_tables_from_page,
    _find_native_tables,
    _page_to_gray,
    _text_coverage,
)
from womblex.ingest.grid_projection import extract_page_text
from womblex.ingest.page_profile import PageProfile, qualify_for_spreadsheet_print

logger = logging.getLogger(__name__)


@dataclass
class _PageAccum:
    """Accumulator across a single page's operations."""

    page_number: int
    text: str = ""
    method: str = ""
    confidence: float = 0.0
    blocks: list[TextBlock] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    forms: list[FormField] = field(default_factory=list)
    images: list[ImageData] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)


def _apply_native_page(
    page: fitz.Page,
    profile: PageProfile,
    accum: _PageAccum,
    *,
    dpi: int = 200,
    lang: str = "eng",
    engine: str = "paddleocr",
    engine_options: dict | None = None,
) -> None:
    """Extract a native text-layer page; also OCR sizable embedded image
    regions that aren't already covered by the text layer.

    `_ocr_image_regions` self-protects: only fires on image rects ≥50 px
    in both dimensions with ≤2 overlapping native words, so all-native
    pages without embedded images pay only metadata-access overhead.
    """
    from womblex.ingest.strategies_scanned import _ocr_image_regions

    # Detect tables. Ruled tables (lines strategy, confidence ≥ 0.8) drive
    # prose-region exclusion + column-major re-emission, because they
    # represent reliable cell structure that the row-major prose path
    # would otherwise interleave (e.g. the 3-column rules-of-the-Law
    # table on Compliance Notice page 1).
    #
    # Text-strategy hits (whitespace-aligned columns, confidence ≈ 0.6)
    # stay in `tables` but do NOT drive prose partitioning — they
    # over-fire on ordinary multi-column prose where word x-positions
    # happen to cluster. The spreadsheet-print path handles the
    # whitespace-aligned manifest case separately.
    table_records = _find_native_tables(page)
    ruled_table_records = [r for r in table_records if r[0].confidence >= 0.8]
    ruled_rects = [rect for _td, rect, _cells in ruled_table_records]
    native_text = extract_page_text(page, exclude_rects=ruled_rects or None)
    if ruled_table_records:
        ruled_table_records.sort(key=lambda r: r[1].y0)
        column_blocks = [
            _emit_table_column_major(cells)
            for _td, _rect, cells in ruled_table_records
        ]
        column_blocks = [b for b in column_blocks if b]
        if column_blocks:
            joined_tables = "\n\n".join(column_blocks)
            native_text = (
                f"{native_text}\n\n{joined_tables}" if native_text else joined_tables
            )
    accum.text = native_text
    accum.method = "native"
    accum.confidence = 95.0
    accum.blocks.extend(_build_text_blocks(page))
    accum.tables.extend(td for td, _rect, _cells in table_records)
    accum.forms.extend(_extract_forms(page))
    accum.images.extend(_extract_images_from_page(page))

    # Only attempt sub-page OCR if the page actually has images. Skips the
    # paddle-reader cold-start for native docs without any embedded images.
    if profile.image_count == 0:
        return

    native_words = page.get_text("words")
    sub_blocks, sub_steps = _ocr_image_regions(
        page, native_words, dpi, lang,
        engine=engine, engine_options=engine_options or {},
    )
    if sub_blocks:
        accum.text = native_text + "\n\n" + "\n\n".join(b.text for b in sub_blocks)
        accum.method = "native+ocr"
        accum.blocks.extend(sub_blocks)
        accum.steps.extend(sub_steps)


def _apply_ocr_page(
    page: fitz.Page,
    profile: PageProfile,
    accum: _PageAccum,
    *,
    dpi: int,
    lang: str,
    engine: str,
    engine_options: dict,
    doc_type: DocumentType,
) -> None:
    # Imported lazily to keep startup paths free of OCR deps.
    from womblex.ingest.strategies_scanned import (
        _layout_blocks_and_tables,
        _markdown_page_block,
        _ocr_page,
    )

    text, conf, steps, native_order = _ocr_page(
        page, dpi, lang, engine, engine_options,
    )
    accum.text = text
    accum.method = "ocr"
    accum.confidence = conf
    accum.steps.extend(steps)
    accum.forms.extend(_extract_form_pairs_from_lines(text))

    if native_order:
        accum.blocks.extend(_markdown_page_block(page, text, conf))
        page_tables: list[TableData] = []
    else:
        page_blocks, page_tables = _layout_blocks_and_tables(page, dpi, text, conf)
        accum.blocks.extend(page_blocks)

    # Mixed-doc tagging: classify content_type per page on OCR pages.
    if doc_type == DocumentType.SCANNED_MIXED:
        from womblex.ingest.heuristics_cv2 import analyze_contour_complexity

        gray = _page_to_gray(page, dpi=dpi)
        complexity = analyze_contour_complexity(gray)
        is_typed = complexity.regularity > 0.5
        content_type = "typed" if is_typed else "handwritten"
        accum.blocks = [
            TextBlock(text=b.text, position=b.position, block_type=content_type, confidence=b.confidence)
            for b in accum.blocks
        ]
        accum.steps.append(f"mixed:{content_type}")

    accum.tables.extend(page_tables)

    # Scanned-machinewritten table-grid fallback (parity with the legacy
    # ScannedMachinewrittenExtractor): if layout pass produced no tables
    # and the page rendered as a grid, retry via PyMuPDF.
    if (
        not page_tables
        and doc_type == DocumentType.SCANNED_MACHINEWRITTEN
    ):
        from womblex.ingest.heuristics_cv2 import detect_table_grid

        gray = _page_to_gray(page, dpi=dpi)
        grid = detect_table_grid(gray)
        if grid.has_grid:
            accum.tables.extend(_extract_tables_from_page(page))


def extract_with_plan(
    doc: fitz.Document,
    profiles: list[PageProfile],
    doc_type: DocumentType,
    *,
    dpi: int = 200,
    lang: str = "eng",
    engine: str = "paddleocr",
    engine_options: dict | None = None,
    filename: str = "",
    spreadsheet_print: dict | None = None,
) -> ExtractionResult:
    """Execute a per-page extraction plan and merge results.

    Dispatch is page-level via ``profile.has_text_layer``. The doc-level
    ``doc_type`` is consumed for two narrow concerns: mixed-typed/handwritten
    tagging on OCR pages, and the scanned-machinewritten table-grid
    fallback. All other behaviour is page-driven.

    ``spreadsheet_print`` config dict (optional):
    - ``metadata_location``: ``"both"`` | ``"table"`` | ``"document"`` (default ``"both"``)
    - ``filename_hints``: tuple of substrings to fast-trigger the qualifier
    """
    opts = engine_options or {}
    pages: list[PageResult] = []
    all_blocks: list[TextBlock] = []
    all_tables: list[TableData] = []
    all_forms: list[FormField] = []
    all_images: list[ImageData] = []
    confidences: list[float] = []
    combined_steps: list[str] = []
    native_count = ocr_count = 0
    document_metadata: dict[str, str] = {}

    # Spreadsheet-print fast path: cheap qualifier → structural vet. If the
    # vet returns rows, those tables replace per-page table extraction (the
    # text/blocks dispatch still runs page-by-page below).
    sp_cfg = spreadsheet_print or {}
    sp_hints = tuple(sp_cfg.get("filename_hints", ())) or None
    sp_loc = sp_cfg.get("metadata_location", "both")
    sp_qualifier_kwargs: dict = {}
    if sp_hints:
        sp_qualifier_kwargs["filename_hints"] = sp_hints
    spreadsheet_tables: list[TableData] = []
    is_spreadsheet_print = False
    if doc_type in (DocumentType.STRUCTURED, DocumentType.NATIVE_WITH_STRUCTURED) and \
            qualify_for_spreadsheet_print(profiles, filename, **sp_qualifier_kwargs):
        from womblex.ingest.spreadsheet_print import extract_spreadsheet_print
        spreadsheet_tables, document_metadata = extract_spreadsheet_print(
            doc, metadata_location=sp_loc,
        )
        is_spreadsheet_print = bool(spreadsheet_tables)

    for page in doc:
        profile = profiles[page.number]
        accum = _PageAccum(page_number=page.number)

        if profile.has_text_layer:
            native_count += 1
            _apply_native_page(
                page, profile, accum,
                dpi=dpi, lang=lang, engine=engine, engine_options=opts,
            )
        else:
            ocr_count += 1
            _apply_ocr_page(
                page, profile, accum,
                dpi=dpi, lang=lang, engine=engine, engine_options=opts,
                doc_type=doc_type,
            )

        pages.append(PageResult(page_number=page.number, text=accum.text, method=accum.method))
        all_blocks.extend(accum.blocks)
        # Skip per-page tables when spreadsheet-print owns the table column.
        if not is_spreadsheet_print:
            all_tables.extend(accum.tables)
        all_forms.extend(accum.forms)
        all_images.extend(accum.images)
        confidences.append(accum.confidence)
        combined_steps.extend(accum.steps)

    if is_spreadsheet_print:
        all_tables = spreadsheet_tables
        combined_steps.append("spreadsheet_print")

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    coverage = _text_coverage(pages)
    unique_steps = sorted(set(combined_steps))

    content_mix: dict[str, float] = {}
    total = native_count + ocr_count
    if total > 0:
        content_mix = {"native": native_count / total, "scanned": ocr_count / total}
    if is_spreadsheet_print:
        content_mix["spreadsheet_print"] = 1.0

    # Confidence is reported in 0–1 to match existing strategy behaviour:
    # native pages contribute 95 (treated as percent); OCR pages contribute
    # paddle's 0–100 confidence. Normalise to 0–1 for the metadata field.
    confidence_01 = avg_conf / 100.0 if avg_conf > 1 else avg_conf

    return ExtractionResult(
        pages=pages,
        method=doc_type.value,
        tables=all_tables,
        forms=all_forms,
        images=all_images,
        text_blocks=all_blocks,
        document_metadata=document_metadata,
        metadata=ExtractionMetadata(
            extraction_strategy=doc_type.value,
            confidence=confidence_01,
            processing_time=0.0,
            page_count=len(doc),
            text_coverage=coverage,
            preprocessing_steps=unique_steps,
            content_mix=content_mix,
        ),
    )


def extract_pdf_with_plan(
    doc: fitz.Document,
    profile: DocumentProfile,
    *,
    dpi: int = 200,
    lang: str = "eng",
    engine: str = "paddleocr",
    engine_options: dict | None = None,
    filename: str = "",
    spreadsheet_print: dict | None = None,
) -> ExtractionResult:
    """Convenience: profile pages, summarise type, run the orchestrator."""
    from womblex.ingest.page_profile import profile_pages, summarise_doc_type

    profiles = profile_pages(doc)
    doc_type = summarise_doc_type(profiles, profile)
    logger.debug(
        "plan: pages=%d native=%d ocr=%d tables=%d forms=%d type=%s",
        len(profiles),
        sum(1 for p in profiles if p.has_text_layer),
        sum(1 for p in profiles if p.needs_ocr),
        sum(1 for p in profiles if p.has_table_signal),
        sum(1 for p in profiles if p.has_form_signal),
        doc_type.value,
    )
    return extract_with_plan(
        doc, profiles, doc_type,
        dpi=dpi, lang=lang, engine=engine, engine_options=engine_options,
        filename=filename, spreadsheet_print=spreadsheet_print,
    )
