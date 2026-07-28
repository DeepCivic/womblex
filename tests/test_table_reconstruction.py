"""Table-cell reconstruction on OCR'd pages (#17).

Round 1, A0: scope is the region-based (paddleocr) path. These tests pin
the scoping decision and the plumbing that carries OCR regions into the
layout pass — no cells are produced yet.
"""

from __future__ import annotations

import logging

import fitz
import pytest

from womblex.ingest.interfaces.protocols import LayoutRegionResult, OCRRegionResult
from womblex.ingest.page_profile import PageProfile
from womblex.ingest.strategies_scanned import _layout_blocks_and_tables, _regions_in_rect


def _region(x0: int, y0: int, x1: int, y1: int, text: str = "cell") -> OCRRegionResult:
    return OCRRegionResult(
        bbox=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        text=text,
        confidence=0.9,
    )


def _ocr_profile() -> PageProfile:
    """A scanned page profile — no text layer, OCR required."""
    return PageProfile(
        page_number=0, width=612.0, height=792.0,
        char_count=0, image_count=1, vector_drawings=0,
        has_text_layer=False, needs_ocr=True,
        has_table_signal=True, has_form_signal=False,
        has_handwriting_signal=False,
    )


class _StubAnalyzer:
    def __init__(self, regions: list[LayoutRegionResult]) -> None:
        self._regions = regions

    def analyze(self, img) -> list[LayoutRegionResult]:
        return self._regions


@pytest.fixture
def blank_page() -> fitz.Page:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Some page text")
    return page


class TestRegionsInRect:
    """Centroid containment — the OCR-quad → table-rect intersection."""

    def test_selects_by_centroid(self) -> None:
        inside = _region(100, 100, 200, 130)
        outside = _region(100, 500, 200, 530)
        got = _regions_in_rect([inside, outside], (50.0, 50.0, 300.0, 300.0))
        assert got == [inside]

    def test_straddling_region_follows_its_middle(self) -> None:
        # Spans the rect's bottom edge; centroid (y=295) is inside.
        straddles_in = _region(100, 260, 200, 330)
        # Mostly below; centroid (y=315) is outside.
        straddles_out = _region(100, 280, 200, 350)
        got = _regions_in_rect([straddles_in, straddles_out], (50.0, 50.0, 300.0, 300.0))
        assert got == [straddles_in]

    def test_blank_regions_dropped(self) -> None:
        assert _regions_in_rect([_region(100, 100, 200, 130, "   ")], (0.0, 0.0, 300.0, 300.0)) == []

    def test_no_regions(self) -> None:
        assert _regions_in_rect([], (0.0, 0.0, 300.0, 300.0)) == []


class TestLayoutPassPlumbing:
    """A0 — the layout pass accepts OCR regions and guards their coord space."""

    @pytest.fixture(autouse=True)
    def _stub_analyzer(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "womblex.ingest.strategies_scanned.get_layout_analyzer",
            lambda: _StubAnalyzer([
                LayoutRegionResult(bbox=(0, 0, 400, 400), label="Table",
                                   block_type="table", confidence=0.96),
                LayoutRegionResult(bbox=(0, 500, 400, 700), label="Text",
                                   block_type="paragraph", confidence=0.90),
            ]),
        )

    def test_regions_are_optional(self, blank_page: fitz.Page) -> None:
        """Callers without regions (legacy, tests) keep today's behaviour."""
        blocks, tables = _layout_blocks_and_tables(blank_page, 200, "page text", 90.0)
        # Pins the A3 starting point: layout-derived non-table blocks carry
        # no text, so the fallback collapses the page — including the table's
        # content — onto one block, and no table is produced.
        assert len(blocks) == 1
        assert blocks[0].text == "page text"
        assert tables == []

    def test_matching_render_dims_keep_regions(
        self, blank_page: fitz.Page, caplog,
    ) -> None:
        pix = blank_page.get_pixmap(dpi=200)
        with caplog.at_level(logging.WARNING, logger="womblex.ingest.strategies_scanned"):
            _blocks, tables = _layout_blocks_and_tables(
                blank_page, 200, "page text", 90.0,
                ocr_regions=[_region(10, 10, 100, 40)],
                ocr_pix_dims=(int(pix.width), int(pix.height)),
            )
        assert "render mismatch" not in caplog.text
        # A0 ships no reconstructor — tables stay empty on every path.
        assert tables == []

    def test_mismatched_render_dims_drop_regions(
        self, blank_page: fitz.Page, caplog,
    ) -> None:
        """Non-comparable coordinate spaces lose the inputs, never mis-bin."""
        with caplog.at_level(logging.WARNING, logger="womblex.ingest.strategies_scanned"):
            _blocks, tables = _layout_blocks_and_tables(
                blank_page, 200, "page text", 90.0,
                ocr_regions=[_region(10, 10, 100, 40)],
                ocr_pix_dims=(17, 23),
            )
        assert "render mismatch" in caplog.text
        assert tables == []


class TestOcrPageScoping:
    """A0 — only the region-based branch reaches the layout pass."""

    def _patch_ocr_page(self, monkeypatch, *, native_order: bool, regions: list) -> None:
        def _fake_ocr_page(page, dpi, lang, engine, engine_options):
            return ("page text", 90.0, [], native_order, regions, (1700, 2200))

        monkeypatch.setattr(
            "womblex.ingest.strategies_scanned._ocr_page", _fake_ocr_page,
        )

    def test_region_engine_forwards_regions_and_dims(
        self, blank_page: fitz.Page, monkeypatch,
    ) -> None:
        from womblex.ingest.detect import DocumentType
        from womblex.ingest.orchestrator import _apply_ocr_page, _PageAccum

        regions = [_region(10, 10, 100, 40)]
        self._patch_ocr_page(monkeypatch, native_order=False, regions=regions)

        seen: dict = {}

        def _fake_layout(page, dpi, text, conf, ocr_regions=None, ocr_pix_dims=None):
            seen["regions"] = ocr_regions
            seen["pix_dims"] = ocr_pix_dims
            return [], []

        monkeypatch.setattr(
            "womblex.ingest.strategies_scanned._layout_blocks_and_tables", _fake_layout,
        )

        accum = _PageAccum(page_number=0)
        _apply_ocr_page(
            blank_page, _ocr_profile(), accum,
            dpi=200, lang="eng", engine="paddleocr", engine_options={},
            doc_type=DocumentType.SCANNED_MACHINEWRITTEN,
        )
        assert seen["regions"] == regions
        assert seen["pix_dims"] == (1700, 2200)

    def test_llm_engine_bypasses_layout_pass(
        self, blank_page: fitz.Page, monkeypatch,
    ) -> None:
        """LLM/VLM engines emit markdown with no regions — nothing to reconstruct."""
        from womblex.ingest.detect import DocumentType
        from womblex.ingest.orchestrator import _apply_ocr_page, _PageAccum

        self._patch_ocr_page(monkeypatch, native_order=True, regions=[])

        def _must_not_run(*args, **kwargs):
            raise AssertionError("layout pass reached on a native-reading-order engine")

        monkeypatch.setattr(
            "womblex.ingest.strategies_scanned._layout_blocks_and_tables", _must_not_run,
        )

        accum = _PageAccum(page_number=0)
        _apply_ocr_page(
            blank_page, _ocr_profile(), accum,
            dpi=200, lang="eng", engine="mistral-ocr", engine_options={},
            doc_type=DocumentType.SCANNED_MACHINEWRITTEN,
        )
        assert accum.tables == []
