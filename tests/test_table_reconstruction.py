"""Table-cell reconstruction on OCR'd pages (#17).

Round 1. A0 pinned the scope (region-based engines) and the plumbing that
carries OCR regions into the layout pass. A1 added the shared grid module
(``table_grid``) and the OCR feeder (``ocr_tables.reconstruct_table``) —
cells can now be produced from a table rect, but nothing is wired into
the layout pass yet (that is A3), so ``tables`` stays empty on every
extraction path.
"""

from __future__ import annotations

import logging

import fitz
import pytest

from womblex.ingest.interfaces.protocols import LayoutRegionResult, OCRRegionResult
from womblex.ingest.ocr_tables import (
    reconstruct_table,
    regions_in_rect,
    span_from_region,
)
from womblex.ingest.page_profile import PageProfile
from womblex.ingest.strategies_scanned import (
    _layout_blocks_and_tables,
    _spatial_sort_regions,
    _table_aware_text,
)
from womblex.ingest.table_grid import Span, rows_from_spans


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
def blank_page():
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Some page text")
    yield page
    doc.close()


class TestRegionsInRect:
    """Centroid containment — the OCR-quad → table-rect intersection."""

    def test_selects_by_centroid(self) -> None:
        inside = _region(100, 100, 200, 130)
        outside = _region(100, 500, 200, 530)
        got = regions_in_rect([inside, outside], (50.0, 50.0, 300.0, 300.0))
        assert got == [inside]

    def test_straddling_region_follows_its_middle(self) -> None:
        # Spans the rect's bottom edge; centroid (y=295) is inside.
        straddles_in = _region(100, 260, 200, 330)
        # Mostly below; centroid (y=315) is outside.
        straddles_out = _region(100, 280, 200, 350)
        got = regions_in_rect([straddles_in, straddles_out], (50.0, 50.0, 300.0, 300.0))
        assert got == [straddles_in]

    def test_blank_regions_dropped(self) -> None:
        assert regions_in_rect([_region(100, 100, 200, 130, "   ")], (0.0, 0.0, 300.0, 300.0)) == []

    def test_no_regions(self) -> None:
        assert regions_in_rect([], (0.0, 0.0, 300.0, 300.0)) == []


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
        assert "dropping cell regions" not in caplog.text
        # A0 ships no reconstructor — tables stay empty on every path.
        assert tables == []

    def test_debug_log_counts_regions_inside_the_table_rect(
        self, blank_page: fitz.Page, caplog,
    ) -> None:
        """The gap's size is traceable per page before the reconstructor lands."""
        pix = blank_page.get_pixmap(dpi=200)
        with caplog.at_level(logging.DEBUG, logger="womblex.ingest.strategies_scanned"):
            _layout_blocks_and_tables(
                blank_page, 200, "page text", 90.0,
                ocr_regions=[
                    _region(10, 10, 100, 40),      # inside the table rect
                    _region(10, 600, 100, 640),    # below it
                ],
                ocr_pix_dims=(int(pix.width), int(pix.height)),
            )
        assert "layout table region: page=0 confidence=0.96 ocr_regions=1" in caplog.text

    def test_regions_without_dims_are_dropped(
        self, blank_page: fitz.Page, caplog,
    ) -> None:
        """Unverifiable is treated as non-comparable: regions need their dims."""
        with caplog.at_level(logging.WARNING, logger="womblex.ingest.strategies_scanned"):
            _blocks, tables = _layout_blocks_and_tables(
                blank_page, 200, "page text", 90.0,
                ocr_regions=[_region(10, 10, 100, 40)],
            )
        assert "dropping cell regions" in caplog.text
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
        assert "dropping cell regions" in caplog.text
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


class TestSharedGridHelpers:
    """A1 — one algorithm in ``table_grid``, consumed by both feeders."""

    def test_span_from_region_reduces_quad_to_bounds(self) -> None:
        r = OCRRegionResult(
            bbox=[[10, 20], [110, 25], [108, 55], [12, 50]],
            text="cell", confidence=0.9,
        )
        s = span_from_region(r)
        assert (s.x_left, s.y_top, s.x_right, s.y_bottom) == (10, 20, 110, 55)
        assert s.text == "cell"

    def test_rows_from_spans_clusters_by_centroid(self) -> None:
        spans = [
            Span(y_top=0, y_bottom=10, x_left=50, x_right=60, text="b"),
            Span(y_top=1, y_bottom=11, x_left=10, x_right=20, text="a"),
            Span(y_top=30, y_bottom=40, x_left=10, x_right=20, text="c"),
        ]
        rows, avg_h = rows_from_spans(spans)
        assert [[s.text for s in row] for row in rows] == [["a", "b"], ["c"]]
        assert avg_h == 10.0

    def test_rows_from_spans_empty(self) -> None:
        assert rows_from_spans([]) == ([], 0.0)

    def test_spatial_sort_reads_row_major(self) -> None:
        regions = [
            _region(400, 100, 500, 130, "B"),
            _region(100, 100, 200, 130, "A"),
            _region(100, 300, 200, 330, "C"),
        ]
        assert _spatial_sort_regions(regions) == "A B\nC"

    def test_table_aware_text_emits_table_runs_column_major(self) -> None:
        regions = [
            _region(100 + 300 * c, 100 + 100 * r, 220 + 300 * c, 130 + 100 * r,
                    f"{'abc'[c]}{r + 1}")
            for r in range(3) for c in range(3)
        ]
        assert _table_aware_text(regions) == "a1 a2 a3\n\nb1 b2 b3\n\nc1 c2 c3"


def _grid_regions(
    n_body_rows: int = 3,
    n_cols: int = 4,
    *,
    col_pitch: int = 300,
    row_pitch: int = 100,
    conf: float = 0.9,
) -> list[OCRRegionResult]:
    """A clean grid at 200 dpi spacing: header row H1..Hn plus body cells r{r}c{c}."""
    regions = []
    for r in range(n_body_rows + 1):
        for c in range(n_cols):
            x0, y0 = 100 + col_pitch * c, 100 + row_pitch * r
            regions.append(OCRRegionResult(
                bbox=[[x0, y0], [x0 + 120, y0], [x0 + 120, y0 + 30], [x0, y0 + 30]],
                text=f"H{c + 1}" if r == 0 else f"r{r}c{c + 1}",
                confidence=conf,
            ))
    return regions


class TestReconstructTable:
    """A1 — the OCR feeder produces a TableData, or refuses (never a partial)."""

    RECT = (50.0, 50.0, 1300.0, 500.0)
    DIMS = (1700, 2200)

    def test_clean_grid_reconstructs(self) -> None:
        table = reconstruct_table(_grid_regions(), self.RECT, 200, 0.96, pix_dims=self.DIMS)
        assert table is not None
        assert table.headers == ["H1", "H2", "H3", "H4"]
        assert table.rows == [
            ["r1c1", "r1c2", "r1c3", "r1c4"],
            ["r2c1", "r2c2", "r2c3", "r2c4"],
            ["r3c1", "r3c2", "r3c3", "r3c4"],
        ]

    def test_lineage_confidence_and_producer(self) -> None:
        """A5 — confidence from constituent regions, producer marker set."""
        table = reconstruct_table(
            _grid_regions(conf=0.8), self.RECT, 200, 0.96, pix_dims=self.DIMS,
        )
        assert table is not None
        assert table.confidence == pytest.approx(0.8)
        assert table.context["producer"] == "table_grid"

    def test_detector_confidence_caps_region_confidence(self) -> None:
        table = reconstruct_table(
            _grid_regions(conf=0.9), self.RECT, 200, 0.5, pix_dims=self.DIMS,
        )
        assert table is not None
        assert table.confidence == pytest.approx(0.5)

    def test_position_normalised_by_pix_dims(self) -> None:
        table = reconstruct_table(_grid_regions(), self.RECT, 200, 0.96, pix_dims=self.DIMS)
        assert table is not None
        assert table.position.x == pytest.approx(50 / 1700)
        assert table.position.y == pytest.approx(50 / 2200)
        assert table.position.width == pytest.approx(1250 / 1700)
        assert table.position.height == pytest.approx(450 / 2200)

    def test_refuses_too_few_columns(self) -> None:
        got = reconstruct_table(
            _grid_regions(n_cols=2), self.RECT, 200, 0.96, pix_dims=self.DIMS,
        )
        assert got is None

    def test_refuses_too_few_rows(self) -> None:
        got = reconstruct_table(
            _grid_regions(n_body_rows=1), self.RECT, 200, 0.96, pix_dims=self.DIMS,
        )
        assert got is None

    def test_refuses_poor_column_fit(self) -> None:
        """Spans the column model can't place trip the assignment gate."""
        # Strays sit inside the rect but left of column 1 beyond tolerance.
        regions = _grid_regions() + [
            _region(55, 200, 95, 230, "stray1"),
            _region(55, 300, 95, 330, "stray2"),
        ]
        assert reconstruct_table(regions, self.RECT, 200, 0.96, pix_dims=self.DIMS) is None

    def test_refuses_empty_rect(self) -> None:
        regions = _grid_regions()
        # A rect nowhere near the regions holds no centroids.
        assert reconstruct_table(regions, (1400.0, 50.0, 1600.0, 500.0), 200, 0.96,
                                 pix_dims=self.DIMS) is None
