"""Per-page profile for plan-driven extraction.

Phase 2 of the Womblex roadmap replaces document-level strategy routing
with per-page profiling. The detector in detect.py historically collected
per-page signals then aggregated them away into a single DocumentType;
this module retains them so the orchestrator can dispatch operations
page-by-page.

FOI bundles are heterogeneous within a single file (cover letter +
columnar table + form + signed declaration). Document-level routing
collapses this to one strategy. Page-level profiling matches the data.
"""
from __future__ import annotations

from dataclasses import dataclass

import fitz

from womblex.ingest.detect import (
    DocumentProfile,
    DocumentType,
    _MIN_TEXT_LENGTH,
    _MIN_VECTOR_DRAWINGS,
    _has_form_structure,
    _has_handwriting_signals,
    _has_structural_tables,
    _has_table_structure,
)


@dataclass
class PageProfile:
    """Per-page detection signals retained for plan-driven dispatch."""

    page_number: int
    width: float
    height: float
    char_count: int
    image_count: int
    vector_drawings: int
    has_text_layer: bool          # char_count > _MIN_TEXT_LENGTH
    needs_ocr: bool               # no text layer but content exists (images/vectors)
    has_table_signal: bool        # text-pattern OR PyMuPDF find_tables
    has_form_signal: bool         # ≥10 short text spans (label-shape) or AcroForm widgets
    has_handwriting_signal: bool  # only computed for non-text-layer pages

    @property
    def mode(self) -> str:
        """Page-level extraction mode (summary attribute)."""
        if self.has_text_layer:
            if self.has_table_signal and self.char_count < 200:
                return "table_only"
            return "native"
        if self.needs_ocr:
            return "ocr_handwriting" if self.has_handwriting_signal else "ocr"
        return "blank"


def profile_pages(doc: fitz.Document) -> list[PageProfile]:
    """Build a PageProfile for every page in the document.

    Unlike `detect.detect_file_type`, this does NOT sample — it walks every
    page and retains the per-page signals. The cost is acceptable on
    CPU-only pipelines (≈ a few hundred ms per doc); the win is page-level
    routing accuracy.
    """
    profiles: list[PageProfile] = []
    for page in doc:
        text = page.get_text().strip()
        char_count = len(text)
        has_text_layer = char_count > _MIN_TEXT_LENGTH

        images = page.get_images()
        image_count = len(images)
        has_image = image_count > 0

        vector_drawings = 0
        if not has_text_layer and not has_image:
            vector_drawings = len(page.get_drawings())

        needs_ocr = (not has_text_layer) and (
            has_image or vector_drawings >= _MIN_VECTOR_DRAWINGS
        )

        has_table_signal = False
        if has_text_layer:
            has_table_signal = _has_table_structure(text) or _has_structural_tables(page)

        has_form_signal = False
        if has_text_layer:
            try:
                has_form_signal = _has_form_structure(page)
            except Exception:
                has_form_signal = False

        has_handwriting_signal = False
        if needs_ocr:
            try:
                has_handwriting_signal = _has_handwriting_signals(page)
            except Exception:
                has_handwriting_signal = False

        profiles.append(
            PageProfile(
                page_number=page.number,
                width=page.rect.width,
                height=page.rect.height,
                char_count=char_count,
                image_count=image_count,
                vector_drawings=vector_drawings,
                has_text_layer=has_text_layer,
                needs_ocr=needs_ocr,
                has_table_signal=has_table_signal,
                has_form_signal=has_form_signal,
                has_handwriting_signal=has_handwriting_signal,
            )
        )
    return profiles


_DEFAULT_FILENAME_HINTS = (
    "schedule", "index", "manifest", "register",
    "list-of", "table-of", "appendix",
)


def qualify_for_spreadsheet_print(
    profiles: list[PageProfile],
    filename: str = "",
    *,
    filename_hints: tuple[str, ...] = _DEFAULT_FILENAME_HINTS,
    min_native_pages: int = 1,
) -> bool:
    """Cheap per-doc check: is this doc *worth* structurally vetting for
    spreadsheet-print shape?

    Re-uses fields already on PageProfile — no new computation. Trips
    when:
    - ≥ ``min_native_pages`` pages have a text layer (rules out scanned)
    - ≥ 1 page has a table signal (existing structural-table detector)
    - The filename matches any of ``filename_hints`` (cheap fast path), OR
      ≥ 50 % of pages have a table signal (catches manifests without
      hint-y filenames)

    The structural vet (column count, row coverage) runs only on
    qualifying docs — see ``spreadsheet_print.extract_spreadsheet_print``.
    """
    if not profiles:
        return False
    n_native = sum(1 for p in profiles if p.has_text_layer)
    if n_native < min_native_pages:
        return False
    n_table = sum(1 for p in profiles if p.has_table_signal)
    if n_table < 1:
        return False

    filename_lower = filename.lower()
    name_hit = any(h in filename_lower for h in filename_hints)
    if name_hit:
        return True

    # No hint match — require a stronger structural signal.
    return (n_table / len(profiles)) >= 0.5


def summarise_doc_type(profiles: list[PageProfile], legacy: DocumentProfile) -> DocumentType:
    """Derive a document-level type from per-page profiles.

    Used as a summary attribute on the result and as a hint for the
    orchestrator (e.g. mixed-typed/handwritten tagging). Falls back to
    the legacy profile's doc_type when profile signals are inconclusive
    (covers UNKNOWN, IMAGE, etc.).
    """
    if not profiles:
        return legacy.doc_type

    n = len(profiles)
    n_native = sum(1 for p in profiles if p.has_text_layer)
    n_ocr = sum(1 for p in profiles if p.needs_ocr)
    n_handwriting = sum(1 for p in profiles if p.has_handwriting_signal)
    n_tables = sum(1 for p in profiles if p.has_table_signal)

    # If detector flagged scanned types via morphology/OCR confidence,
    # trust it (those signals don't have a per-page equivalent here).
    if legacy.doc_type in (
        DocumentType.SCANNED_HANDWRITTEN,
        DocumentType.SCANNED_MIXED,
    ):
        return legacy.doc_type

    # Hybrid: mix of native and OCR pages
    if n_native and n_ocr:
        return DocumentType.HYBRID

    if n_native == n:
        if n_tables / n >= 0.8:
            return DocumentType.STRUCTURED
        if n_tables > 0 or any(p.image_count > 0 for p in profiles):
            return DocumentType.NATIVE_WITH_STRUCTURED
        return DocumentType.NATIVE_NARRATIVE

    if n_ocr == n:
        if n_handwriting / n >= 0.5:
            return DocumentType.SCANNED_HANDWRITTEN
        if n_handwriting > 0:
            return DocumentType.SCANNED_MIXED
        return DocumentType.SCANNED_MACHINEWRITTEN

    return legacy.doc_type
