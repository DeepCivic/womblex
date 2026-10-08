"""Block grouping under the pdfium backend matches MuPDF's on the synthetic PDFs.

MuPDF starts a block where the baseline pitch exceeds about 1.5 font sizes, keeps a
row's fragments (table cells) together, and does not break on a size or weight
change alone. The spacing fixture varies leading and paragraph gap so the rule is
measured at more than one pitch.
"""

from __future__ import annotations

import collections
from pathlib import Path

import pytest

from tests._synthetic import AUDIT_PDF, NOTICE_PDF, SCHEDULE_PDF, SPACING_PDF
from womblex.ingest import pdf
from womblex.ingest.detect import detect_file_type
from womblex.ingest.extract import extract_text
from womblex.ingest.pdf import open_document

PARITY = [AUDIT_PDF, SPACING_PDF, NOTICE_PDF, SCHEDULE_PDF]


def _lines_per_block(path: Path, backend: str) -> list[list[int]]:
    return [[len(b.lines) for b in page.text_dict()] for page in open_document(path, backend=backend)]


def _kinds(path: Path, backend: str, monkeypatch: pytest.MonkeyPatch) -> collections.Counter[str]:
    monkeypatch.setattr(pdf, "DEFAULT_BACKEND", backend)
    return collections.Counter(
        e.kind for r in extract_text(path, detect_file_type(path)) for e in r.elements
    )


@pytest.mark.parametrize("path", PARITY, ids=lambda p: p.name)
def test_blocks_hold_the_same_lines_as_mupdf(path: Path) -> None:
    assert _lines_per_block(path, "pdfium") == _lines_per_block(path, "fitz")


@pytest.mark.parametrize("path", [AUDIT_PDF, SPACING_PDF], ids=lambda p: p.name)
def test_element_kinds_and_counts_match_mupdf(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _kinds(path, "pdfium", monkeypatch) == _kinds(path, "fitz", monkeypatch)


def _draw(path: Path, texts: list[tuple[float, float, str]], *, angle: int = 0, rotate_page: int = 0) -> Path:
    from reportlab.pdfgen.canvas import Canvas

    canvas = Canvas(str(path), pagesize=(842, 595), invariant=1)
    if rotate_page:
        canvas.setPageRotation(rotate_page)
    canvas.translate(400, 300)
    canvas.rotate(angle)
    canvas.setFont("Helvetica", 7)
    for x, y, text in texts:
        canvas.drawString(x, y, text)
    canvas.save()
    return path


#: (x, y, text) in the text's own frame, y up: header cells, a two-line header,
#: an indented second line, and lines either side of the baseline.
_SHAPES = {
    "two-line header under the last cell": [(0, 0, "AAAA"), (50, 0, "BBBB"), (100, 0, "Sub"), (100, -8, "code"), (150, 0, "Next")],
    "two-line header under the first cell": [(0, 0, "AAAA"), (0, -8, "a2"), (50, 0, "BBBB")],
    "line below, same start": [(0, 0, "AAAAAAAA"), (0, -9, "BBBB")],
    "line below, indented": [(0, 0, "AAAAAAAAAAAAAAAAAAAA"), (6, -8.4, "BBBBBBBB")],
    "cell after, 3pt up": [(0, 0, "AAAA"), (50, 3, "BBBB")],
    "cell after, 6pt down": [(0, 0, "AAAA"), (50, -6, "BBBB")],
}

#: Axis-aligned only: how MuPDF joins an overprint and a line above is measured there.
_AXIS_SHAPES = {
    "overprinted line": [(0, 0, "AAAAAAAA"), (0, 3, "BBBB")],
    "overprinted line, indented": [(0, 0, "AAAAAAAA"), (10, 3, "BBBB")],
    "line above": [(0, 0, "AAAAAAAA"), (0, 8, "BBBB")],
}


@pytest.mark.parametrize("angle", [0, 5, 90, 180, 270])
@pytest.mark.parametrize("shape", _SHAPES)
def test_header_shapes_group_as_mupdf_does(
    shape: str, angle: int, tmp_path: Path,
) -> None:
    path = _draw(tmp_path / "shape.pdf", _SHAPES[shape], angle=angle)
    assert _lines_per_block(path, "pdfium") == _lines_per_block(path, "fitz")


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
@pytest.mark.parametrize("shape", _AXIS_SHAPES)
def test_overprints_and_lines_above_group_as_mupdf_does(shape: str, angle: int, tmp_path: Path) -> None:
    path = _draw(tmp_path / "shape.pdf", _AXIS_SHAPES[shape], angle=angle)
    assert _lines_per_block(path, "pdfium") == _lines_per_block(path, "fitz")


def test_a_rotated_page_with_upright_text_is_grouped_in_the_unrotated_frame(tmp_path: Path) -> None:
    path = _draw(tmp_path / "rotated.pdf", _SHAPES["two-line header under the last cell"], angle=90, rotate_page=90)
    assert _lines_per_block(path, "pdfium") == _lines_per_block(path, "fitz")


@pytest.mark.parametrize("angle", [90, 270])
def test_vertical_text_keeps_content_order(angle: int, tmp_path: Path) -> None:
    # Drawn out of position order: pdfium sorts vertical text by position, MuPDF does not.
    path = _draw(tmp_path / "order.pdf", [(0, -9, "L0"), (0, -18, "L1"), (0, 0, "L2")], angle=angle)

    def lines(backend: str) -> list[str]:
        return ["".join(s.text for s in line.spans) for page in open_document(path, backend=backend) for b in page.text_dict() for line in b.lines]

    assert lines("pdfium") == lines("fitz") == ["L0", "L1", "L2"]
