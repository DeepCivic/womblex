"""The pdfium text engine: segmentation on synthetic characters, and the page
methods against pages the builder draws, checked against the fitz backend."""

from __future__ import annotations

import pytest

from tests._pdf_builders import PdfBuilder
from womblex.ingest.pdf import _text, open_document
from womblex.ingest.pdf.types import Rect

LINE = 14.0


def _chars(text: str, x: float, y: float, *, size: float = 10, font: str = "Helv", bold: bool = False):
    out = []
    for i, ch in enumerate(text):
        left = x + i * size * 0.5
        out.append(_text.Char(ch, Rect(left, y, left + size * 0.5, y + size), size, font, bold))
    return out


class TestSegment:
    def test_consecutive_lines_form_one_block(self) -> None:
        chars = _chars("first line", 72, 100) + _chars("second line", 72, 100 + LINE)
        assert [len(b) for b in _text.segment(chars)] == [2]

    def test_a_vertical_gap_starts_a_block(self) -> None:
        chars = _chars("first", 72, 100) + _chars("second", 72, 160)
        assert [len(b) for b in _text.segment(chars)] == [1, 1]

    def test_a_fragment_out_of_stream_order_rejoins_its_line(self) -> None:
        chars = _chars("alpha", 72, 100) + _chars("later", 72, 300) + _chars("beta", 112, 100)
        lines = [line.text for block in _text.segment(chars) for line in block]
        assert lines == ["alpha beta", "later"]

    def test_a_column_gutter_is_not_bridged(self) -> None:
        chars = _chars("left", 72, 100) + _chars("right", 340, 100)
        assert [line.text for block in _text.segment(chars) for line in block] == ["left", "right"]

    def test_spans_split_on_font_and_bold(self) -> None:
        chars = _chars("plain ", 72, 100) + _chars("heavy", 102, 100, font="Helv-Bold", bold=True)
        (block,) = _text.text_dict(chars)
        assert [(s.text, s.bold) for s in block.lines[0].spans] == [("plain ", False), ("heavy", True)]


class TestDehyphenate:
    def _wrapped(self, tail: str, head: str) -> list[_text.Char]:
        return _chars(tail, 72, 100) + _chars(head, 72, 100 + LINE)

    def test_joins_a_hyphen_before_a_letter(self) -> None:
        text = _text.plain_text(self._wrapped("a hyphen-", "ated word"), dehyphenate=True)
        assert text == "a hyphenated word\n"

    def test_off_keeps_the_hyphen_and_the_break(self) -> None:
        text = _text.plain_text(self._wrapped("a hyphen-", "ated word"), dehyphenate=False)
        assert text == "a hyphen-\nated word\n"

    @pytest.mark.parametrize("tail, head", [("range 1 -", "2 apples"), ("a -", "dash")])
    def test_leaves_a_hyphen_not_after_a_letter_or_before_a_letter(self, tail, head) -> None:
        assert "-\n" in _text.plain_text(self._wrapped(tail, head), dehyphenate=True)

    def test_words_follow_the_join(self) -> None:
        got = [w.text for w in _text.words(self._wrapped("a hyphen-", "ated word"), dehyphenate=True)]
        assert got == ["a", "hyphenated", "word"]


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory: pytest.TempPathFactory):
    builder = PdfBuilder(tmp_path_factory.mktemp("text") / "text.pdf").page(400, 600)
    builder.text(72, 100, "Heading", size=16, bold=True)
    builder.text(72, 130, "A body line of ordinary text").text(72, 144, "that carries on below it.")
    builder.text(72, 400, "Far below")
    return builder.save()


class TestPage:
    def test_plain_text_matches_fitz(self, pdf_path) -> None:
        with open_document(pdf_path, backend="fitz") as fitz_doc, open_document(pdf_path, backend="pdfium") as doc:
            assert doc[0].plain_text() == fitz_doc[0].plain_text()

    def test_words_match_fitz_text_and_order(self, pdf_path) -> None:
        with open_document(pdf_path, backend="fitz") as fitz_doc, open_document(pdf_path, backend="pdfium") as doc:
            got, want = doc[0].words(), fitz_doc[0].words()
            assert [w.text for w in got] == [w.text for w in want]
            assert [(w.block_no, w.line_no, w.word_no) for w in got] == [
                (w.block_no, w.line_no, w.word_no) for w in want
            ]
            for a, b in zip(got, want, strict=True):
                assert a.x0 == pytest.approx(b.x0, abs=1) and a.y1 == pytest.approx(b.y1, abs=3)

    def test_blocks_group_like_fitz(self, pdf_path) -> None:
        with open_document(pdf_path, backend="fitz") as fitz_doc, open_document(pdf_path, backend="pdfium") as doc:
            assert [b.text for b in doc[0].text_blocks()] == [b.text for b in fitz_doc[0].text_blocks()]

    def test_dict_reports_size_font_and_bold(self, pdf_path) -> None:
        with open_document(pdf_path, backend="pdfium") as doc:
            spans = [s for b in doc[0].text_dict() for line in b.lines for s in line.spans]
        heading = spans[0]
        assert (heading.text, heading.size, heading.bold) == ("Heading", 16.0, True)
        assert heading.font == "Helvetica-Bold"
        assert not spans[1].bold and spans[1].size == 11.0

    def test_empty_page(self, tmp_path) -> None:
        with PdfBuilder(tmp_path / "e.pdf").page().open() as doc:
            assert doc[0].plain_text() == "" and doc[0].words() == [] and doc[0].text_dict() == []
