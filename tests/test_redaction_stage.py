"""Tests for womblex.redact.stage.

RedactionReport, text redaction, annotation and vector-direct detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tests._pdf_builders import PdfBuilder
from womblex.ingest.elements import Element, ElementKind
from womblex.ingest.pdf.types import Rect
from womblex.redact import RedactionInfo
from womblex.redact.stage import (
    RedactionReport,
    annotate_chunks,
    annotate_elements,
    annotate_extraction,
    apply_text_redaction,
    build_detector,
    detect_redactions,
)

if TYPE_CHECKING:
    from womblex.process.chunker import TextChunk


# ---------------------------------------------------------------------------
# Stage logic: RedactionReport
# ---------------------------------------------------------------------------


class TestRedactionReport:
    def test_empty_report(self) -> None:
        report = RedactionReport()
        assert report.total == 0
        assert report.affected_pages == []

    def test_report_with_data(self) -> None:
        r1 = RedactionInfo(bbox=(0, 0, 100, 50), page=0, area_px=5000)
        r2 = RedactionInfo(bbox=(0, 0, 200, 30), page=2, area_px=6000)
        report = RedactionReport(page_redactions={0: [r1], 2: [r2]})
        assert report.total == 2
        assert report.affected_pages == [0, 2]


# ---------------------------------------------------------------------------
# Stage logic: apply_text_redaction
# ---------------------------------------------------------------------------


class TestApplyTextRedaction:
    def _make_pages(self, texts: list[str]) -> list:
        from womblex.ingest.extract import PageResult

        return [PageResult(page_number=i, text=t, method="test") for i, t in enumerate(texts)]

    def _make_report(self, *pages: int) -> RedactionReport:
        return RedactionReport(
            page_redactions={
                p: [RedactionInfo(bbox=(0, 0, 100, 50), page=p, area_px=5000)]
                for p in pages
            }
        )

    def test_flag_mode_no_change(self) -> None:
        pages = self._make_pages(["hello", "world"])
        report = self._make_report(0)
        result = apply_text_redaction(pages, report, mode="flag")
        assert result[0].text == "hello"
        assert result[1].text == "world"

    def test_blackout_mode_prepends_marker(self) -> None:
        pages = self._make_pages(["sensitive text", "clean text"])
        report = self._make_report(0)
        apply_text_redaction(pages, report, mode="blackout")
        assert pages[0].text.startswith("<REDACTED>")
        assert pages[1].text == "clean text"

    def test_blackout_mode_empty_page(self) -> None:
        pages = self._make_pages([""])
        report = self._make_report(0)
        apply_text_redaction(pages, report, mode="blackout")
        assert pages[0].text == "<REDACTED>"

    def test_delete_mode_clears_page(self) -> None:
        pages = self._make_pages(["sensitive text", "clean text"])
        report = self._make_report(0)
        apply_text_redaction(pages, report, mode="delete")
        assert pages[0].text == ""
        assert pages[1].text == "clean text"

    def test_empty_report_no_change(self) -> None:
        pages = self._make_pages(["hello"])
        report = RedactionReport()
        apply_text_redaction(pages, report, mode="blackout")
        assert pages[0].text == "hello"


# ---------------------------------------------------------------------------
# Stage logic: annotate_chunks
# ---------------------------------------------------------------------------


class TestAnnotateChunks:
    def _make_chunk(self, index: int, **kwargs: object) -> TextChunk:
        from womblex.process.chunker import TextChunk

        chunk = TextChunk(
            text=f"chunk {index}",
            start_char=0,
            end_char=7,
            chunk_index=index,
        )
        for k, v in kwargs.items():
            setattr(chunk, k, v)
        return chunk

    def _make_report(self, pages: dict[int, int]) -> RedactionReport:
        """Build a report with *n* redactions per page."""
        page_redactions: dict[int, list[RedactionInfo]] = {}
        for page, count in pages.items():
            page_redactions[page] = [
                RedactionInfo(bbox=(0, 0, 100, 50), page=page, area_px=5000)
                for _ in range(count)
            ]
        return RedactionReport(page_redactions=page_redactions)

    def test_empty_report_returns_unchanged(self) -> None:
        chunks = [self._make_chunk(0), self._make_chunk(1)]
        report = RedactionReport()
        result = annotate_chunks(chunks, report)
        assert result is chunks
        assert chunks[0].has_redaction is False
        assert chunks[1].has_redaction is False

    def test_flags_chunk_with_source_pages(self) -> None:
        chunk = self._make_chunk(0, source_pages=[0, 1])
        report = self._make_report({1: 1})
        annotate_chunks([chunk], report)
        assert chunk.has_redaction is True

    def test_skips_chunk_with_unaffected_source_pages(self) -> None:
        chunk = self._make_chunk(0, source_pages=[2, 3])
        report = self._make_report({0: 1})
        annotate_chunks([chunk], report)
        assert chunk.has_redaction is False

    def test_flags_chunk_with_page_number(self) -> None:
        chunk = self._make_chunk(0, page_number=5)
        report = self._make_report({5: 2})
        annotate_chunks([chunk], report)
        assert chunk.has_redaction is True

    def test_skips_chunk_with_unaffected_page_number(self) -> None:
        chunk = self._make_chunk(0, page_number=3)
        report = self._make_report({0: 1})
        annotate_chunks([chunk], report)
        assert chunk.has_redaction is False

    def test_source_pages_takes_precedence_over_page_number(self) -> None:
        """When both attributes exist, source_pages is checked first."""
        chunk = self._make_chunk(0, source_pages=[0], page_number=5)
        report = self._make_report({5: 1})
        annotate_chunks([chunk], report)
        # source_pages=[0] doesn't overlap affected page 5, so no flag
        assert chunk.has_redaction is False

    def test_chunk_without_page_attrs_is_skipped(self) -> None:
        chunk = self._make_chunk(0)
        report = self._make_report({0: 1})
        annotate_chunks([chunk], report)
        assert chunk.has_redaction is False

    def test_mixed_chunks(self) -> None:
        c0 = self._make_chunk(0, source_pages=[1])
        c1 = self._make_chunk(1, page_number=2)
        c2 = self._make_chunk(2, page_number=9)
        report = self._make_report({1: 1, 2: 1})
        annotate_chunks([c0, c1, c2], report)
        assert c0.has_redaction is True
        assert c1.has_redaction is True
        assert c2.has_redaction is False

    def test_empty_source_pages_falls_through(self) -> None:
        """source_pages=[] is falsy — should fall through to page_number."""
        chunk = self._make_chunk(0, source_pages=[], page_number=0)
        report = self._make_report({0: 1})
        annotate_chunks([chunk], report)
        assert chunk.has_redaction is True


# ---------------------------------------------------------------------------
# Stage logic: annotate_elements
# ---------------------------------------------------------------------------


class TestAnnotateElements:
    def _make_element(self, order: int, page: int | None, kind: ElementKind = "paragraph") -> Element:
        return Element(order=order, kind=kind, extractor="test", page=page)

    def _make_report(self, pages: dict[int, int]) -> RedactionReport:
        page_redactions: dict[int, list[RedactionInfo]] = {}
        for page, count in pages.items():
            page_redactions[page] = [
                RedactionInfo(bbox=(0, 0, 100, 50), page=page, area_px=5000)
                for _ in range(count)
            ]
        return RedactionReport(page_redactions=page_redactions)

    def test_empty_report_returns_unchanged(self) -> None:
        elements = [self._make_element(0, 0), self._make_element(1, 1)]
        result = annotate_elements(elements, RedactionReport())
        assert result is elements
        assert "has_redaction" not in elements[0].meta
        assert "has_redaction" not in elements[1].meta

    def test_flags_elements_on_affected_page(self) -> None:
        e = self._make_element(0, page=2)
        annotate_elements([e], self._make_report({2: 3}))
        assert e.meta["has_redaction"] == "true"

    def test_skips_elements_on_unaffected_page(self) -> None:
        e = self._make_element(0, page=1)
        annotate_elements([e], self._make_report({2: 1}))
        assert "has_redaction" not in e.meta

    def test_skips_elements_with_no_page(self) -> None:
        e = self._make_element(0, page=None)
        annotate_elements([e], self._make_report({0: 1}))
        assert "has_redaction" not in e.meta

    def test_mixed_elements(self) -> None:
        e0 = self._make_element(0, page=0)  # affected
        e1 = self._make_element(1, page=1)  # not affected
        e2 = self._make_element(2, page=2)  # affected (different page)
        report = self._make_report({0: 2, 2: 5})
        annotate_elements([e0, e1, e2], report)
        assert e0.meta["has_redaction"] == "true"
        assert "has_redaction" not in e1.meta
        assert e2.meta["has_redaction"] == "true"


# ---------------------------------------------------------------------------
# Stage logic: annotate_extraction
# ---------------------------------------------------------------------------


class TestAnnotateExtraction:
    def test_adds_warnings(self) -> None:
        from womblex.ingest.extract import ExtractionResult

        extraction = ExtractionResult(pages=[], method="test")
        r1 = RedactionInfo(bbox=(0, 0, 100, 50), page=0, area_px=5000)
        report = RedactionReport(page_redactions={0: [r1]})

        annotate_extraction(extraction, report)
        assert len(extraction.warnings) == 1
        assert "page 0" in extraction.warnings[0]
        assert "1 redacted region" in extraction.warnings[0]

    def test_no_warnings_for_empty_report(self) -> None:
        from womblex.ingest.extract import ExtractionResult

        extraction = ExtractionResult(pages=[], method="test")
        report = RedactionReport()

        annotate_extraction(extraction, report)
        assert len(extraction.warnings) == 0


# ---------------------------------------------------------------------------
# Vector-direct redaction detection (via page.drawings())
# ---------------------------------------------------------------------------


class TestVectorRedactionPath:
    """Vector path catches native-drawn black-fill rectangles without
    rasterising, and without the area threshold that filters small bars
    out of the raster contour detector."""

    def _build(
        self, tmp_path: Path, rect: Rect, colour: tuple[float, float, float] = (0, 0, 0),
    ) -> Path:
        return PdfBuilder(tmp_path / "vec.pdf").page().rect(rect, colour).save()  # A4

    def test_detects_native_vector_black_fill(self, tmp_path: Path) -> None:
        from womblex.config import RedactionConfig
        pdf_path = self._build(tmp_path, Rect(100, 200, 300, 220))
        report = detect_redactions(pdf_path, 1, build_detector(RedactionConfig()))
        assert 0 in report.page_redactions
        assert len(report.page_redactions[0]) >= 1

    def test_filters_thin_vector_lines(self, tmp_path: Path) -> None:
        """A 1pt-tall horizontal underline should not register (min-side filter)."""
        from womblex.config import RedactionConfig
        pdf_path = self._build(tmp_path, Rect(100, 200, 300, 201))
        report = detect_redactions(pdf_path, 1, build_detector(RedactionConfig()))
        assert 0 not in report.page_redactions

    def test_filters_narrow_vertical_separators(self, tmp_path: Path) -> None:
        """A 0.4pt-wide vertical line (table column separator) should not register."""
        from womblex.config import RedactionConfig
        pdf_path = self._build(tmp_path, Rect(150, 100, 150.4, 800))
        report = detect_redactions(pdf_path, 1, build_detector(RedactionConfig()))
        assert 0 not in report.page_redactions

    def test_filters_glyph_sized_fills(self, tmp_path: Path) -> None:
        """A 5pt × 6pt fill (body-glyph rendering) should not register."""
        from womblex.config import RedactionConfig
        pdf_path = self._build(tmp_path, Rect(100, 200, 105, 206))
        report = detect_redactions(pdf_path, 1, build_detector(RedactionConfig()))
        assert 0 not in report.page_redactions

    def test_vector_bbox_in_pixel_coords(self, tmp_path: Path) -> None:
        """Vector path scales PDF coords (72dpi) to detection dpi for consistency."""
        from womblex.config import RedactionConfig
        pdf_path = self._build(tmp_path, Rect(100, 200, 200, 230))
        config = RedactionConfig(dpi=150)
        report = detect_redactions(pdf_path, 1, build_detector(config), dpi=config.dpi)
        regions = report.page_redactions[0]
        assert len(regions) == 1
        x1, _y1, x2, _y2 = regions[0].bbox
        # 100pt * 150/72 ≈ 208; 200pt * 150/72 ≈ 416
        assert x1 == int(100 * 150 / 72)
        assert x2 == int(200 * 150 / 72)

    def test_ignores_non_black_fills(self, tmp_path: Path) -> None:
        """A light-grey filled rectangle should not register (header shading etc)."""
        from womblex.config import RedactionConfig
        pdf_path = self._build(tmp_path, Rect(100, 200, 300, 220), (0.8, 0.8, 0.8))
        report = detect_redactions(pdf_path, 1, build_detector(RedactionConfig()))
        assert 0 not in report.page_redactions
