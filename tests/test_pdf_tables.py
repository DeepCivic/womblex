"""The pdfium table finder: pdfplumber's `TableFinder` fed from pdfium.

Pages are drawn at known coordinates so the expected grid is the one drawn; the
fitz backend is checked on the same pages where it is installed.
"""

from __future__ import annotations

import pytest
from reportlab.pdfgen.canvas import Canvas

from womblex.ingest.pdf import _tables, open_document
from womblex.ingest.pdf.types import Rect

_HEIGHT = 842


def _grid(path, *, one_path: bool) -> None:
    """A 3-row, 2-column ruled grid with a label in each cell. Drawn either as
    separate lines or as one path, which a bounding box would collapse."""
    canvas = Canvas(str(path), pagesize=(595, _HEIGHT))
    rows = [_HEIGHT - y for y in (200, 220, 240, 260)]
    cols = (72, 172, 272)
    if one_path:
        grid = canvas.beginPath()
        for y in rows:
            grid.moveTo(cols[0], y)
            grid.lineTo(cols[-1], y)
        for x in cols:
            grid.moveTo(x, rows[0])
            grid.lineTo(x, rows[-1])
        canvas.drawPath(grid, stroke=1, fill=0)
    else:
        for y in rows:
            canvas.line(cols[0], y, cols[-1], y)
        for x in cols:
            canvas.line(x, rows[0], x, rows[-1])
    canvas.setFont("Helvetica", 9)
    for row in range(3):
        for col in range(2):
            canvas.drawString(80 + col * 100, rows[row] - 14, f"r{row}c{col}")
    canvas.save()


def _aligned(path) -> None:
    """Unruled columns: four rows of three left-aligned fields."""
    canvas = Canvas(str(path), pagesize=(595, _HEIGHT))
    canvas.setFont("Helvetica", 10)
    for row in range(4):
        for col in range(3):
            canvas.drawString(72 + col * 150, _HEIGHT - (200 + row * 18), f"item{row}{col}")
    canvas.save()


def _prose(path) -> None:
    canvas = Canvas(str(path), pagesize=(595, _HEIGHT))
    canvas.setFont("Helvetica", 11)
    for i in range(6):
        canvas.drawString(72, _HEIGHT - (100 + i * 14), "The quick brown wombat reads the compliance notice.")
    canvas.save()


def _find(path, strategy):
    with open_document(path, backend="pdfium") as doc:
        return doc[0].find_tables(strategy=strategy)


@pytest.mark.parametrize("one_path", [False, True], ids=["lines", "single-path"])
def test_ruled_grid(tmp_path, one_path) -> None:
    path = tmp_path / "grid.pdf"
    _grid(path, one_path=one_path)
    (table,) = _find(path, "lines")
    assert (table.row_count, table.col_count) == (3, 2)
    assert table.rows == (("r0c0", "r0c1"), ("r1c0", "r1c1"), ("r2c0", "r2c1"))
    assert [round(v) for v in table.bbox.as_tuple()] == [72, 200, 272, 260]


def test_text_strategy_finds_unruled_columns(tmp_path) -> None:
    path = tmp_path / "aligned.pdf"
    _aligned(path)
    assert _find(path, "lines") == []
    (table,) = _find(path, "text")
    assert table.col_count == 3 and table.row_count >= 3
    assert table.rows[0] == ("item00", "item01", "item02")


def test_prose_is_not_a_ruled_table(tmp_path) -> None:
    path = tmp_path / "prose.pdf"
    _prose(path)
    assert _find(path, "lines") == []


def test_matches_fitz_on_the_ruled_grid(tmp_path) -> None:
    pytest.importorskip("fitz")
    path = tmp_path / "grid.pdf"
    _grid(path, one_path=False)
    with open_document(path, backend="fitz") as doc:
        (expected,) = doc[0].find_tables(strategy="lines")
    (table,) = _find(path, "lines")
    assert (table.row_count, table.col_count) == (expected.row_count, expected.col_count)
    assert table.rows == expected.rows


class TestEdges:
    def test_axis_steps_become_edges_and_diagonals_do_not(self) -> None:
        edges = _tables.edges_from_polylines([[(0.0, 5.0), (10.0, 5.0), (10.0, 20.0), (0.0, 0.0)]])
        assert [(e["orientation"], e["x0"], e["x1"], e["top"], e["bottom"]) for e in edges] == [
            ("h", 0.0, 10.0, 5.0, 5.0),
            ("v", 10.0, 10.0, 5.0, 20.0),
        ]

    def test_float_noise_is_still_axis_aligned(self) -> None:
        (edge,) = _tables.edges_from_polylines([[(0.0, 5.0), (10.0, 5.0004)]])
        assert edge["orientation"] == "h" and edge["height"] == 0.0

    def test_edges_are_only_built_when_the_strategy_reads_them(self) -> None:
        def boom():
            raise AssertionError("edges read by the text strategy")

        assert _tables.find_tables("text", Rect(0, 0, 100, 100), list, boom) == []
