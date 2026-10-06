"""The fitz backend behind the seam: MuPDF's shapes in, `types.py`'s out."""

from __future__ import annotations

import numpy as np
import pytest

fitz = pytest.importorskip("fitz")

from womblex.ingest.pdf import open_document
from womblex.ingest.pdf.types import Rect


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory: pytest.TempPathFactory):
    """A two-page PDF: text and a ruled table, then one image drawn twice."""
    path = tmp_path_factory.mktemp("pdf") / "seam.pdf"
    doc = fitz.open()

    page = doc.new_page()
    page.insert_text((72, 100), "Compliance notice issued today", fontsize=11)
    page.insert_text((72, 130), "Second paragraph of the notice", fontsize=11)
    for i in range(3):  # a 2x2 grid of ruled cells
        y = 200 + i * 20
        page.draw_line(fitz.Point(72, y), fitz.Point(272, y))
    for x in (72, 172, 272):
        page.draw_line(fitz.Point(x, 200), fitz.Point(x, 240))
    for row in range(2):
        for col in range(2):
            page.insert_text((80 + col * 100, 214 + row * 20), f"r{row}c{col}", fontsize=9)
    page.draw_rect(fitz.Rect(300, 300, 400, 340), color=(0, 0, 0), fill=(0, 0, 0))

    second = doc.new_page()
    png = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20))
    png.set_rect(png.irect, (255, 0, 0))
    stream = png.tobytes("png")
    second.insert_image(fitz.Rect(50, 50, 100, 100), stream=stream)
    second.insert_image(fitz.Rect(150, 150, 200, 200), stream=stream)

    doc.save(str(path))
    doc.close()
    return path


class TestDocument:
    def test_opens_and_reports_its_pages(self, pdf_path) -> None:
        with open_document(pdf_path) as doc:
            assert doc.page_count == 2
            assert len(doc) == 2
            assert doc.name.endswith("seam.pdf")
            assert [page.number for page in doc] == [0, 1]
            assert doc[1].number == 1

    def test_select_truncates_in_memory(self, pdf_path) -> None:
        with open_document(pdf_path) as doc:
            doc.select([0])
            assert doc.page_count == 1
        with open_document(pdf_path) as untouched:
            assert untouched.page_count == 2

    def test_unknown_backend_lists_the_known_ones(self, pdf_path) -> None:
        with pytest.raises(ValueError, match="unknown PDF backend 'nope'; known backends: fitz"):
            open_document(pdf_path, backend="nope")

    def test_page_geometry(self, pdf_path) -> None:
        with open_document(pdf_path) as doc:
            page = doc[0]
            assert isinstance(page.rect, Rect)
            assert page.rect.width > 0 and page.rect.height > 0
            assert page.rotation == 0
            assert page.rotation_matrix == (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
            assert page.doc_name.endswith("seam.pdf")


class TestText:
    def test_plain_text(self, pdf_path) -> None:
        with open_document(pdf_path) as doc:
            assert "Compliance notice" in doc[0].plain_text()

    def test_words_keep_the_eight_tuple_shape(self, pdf_path) -> None:
        with open_document(pdf_path) as doc:
            words = doc[0].words()
        assert words and all(len(w) == 8 for w in words)
        first = words[0]
        assert first[4] == first.text
        assert all(isinstance(v, float) for v in first[:4])

    def test_text_dict_yields_blocks_with_spans(self, pdf_path) -> None:
        with open_document(pdf_path) as doc:
            blocks = doc[0].text_dict()
        text_blocks = [b for b in blocks if b.kind == "text"]
        assert text_blocks and all(b.text == "" for b in text_blocks)
        spans = [s for b in text_blocks for line in b.lines for s in line.spans]
        assert any("Compliance" in s.text for s in spans)
        assert all(s.size > 0 and isinstance(s.bold, bool) for s in spans)

    def test_text_blocks_carry_text_not_spans(self, pdf_path) -> None:
        with open_document(pdf_path) as doc:
            blocks = doc[0].text_blocks()
        assert blocks and all(b.lines == () for b in blocks)
        assert any("notice" in b.text for b in blocks)


class TestTables:
    def test_finds_the_ruled_grid(self, pdf_path) -> None:
        with open_document(pdf_path) as doc:
            tables = doc[0].find_tables(strategy="lines")
        assert tables, "the ruled 2x2 grid should be found"
        table = tables[0]
        assert table.row_count >= 2 and table.col_count >= 2
        assert isinstance(table.bbox, Rect)
        flat = [cell for row in table.rows for cell in row]
        assert any(cell and "r0c0" in cell for cell in flat)


class TestRender:
    def test_renders_rgb_and_honours_the_clip(self, pdf_path) -> None:
        with open_document(pdf_path) as doc:
            page = doc[0]
            full = page.render(dpi=72)
            clipped = page.render(dpi=72, clip=Rect(0.0, 0.0, 100.0, 100.0))
        assert full.dtype == np.uint8
        assert full.ndim == 3 and full.shape[2] == 3
        assert clipped.shape[0] < full.shape[0]


class TestImagesAndDrawings:
    def test_one_page_image_per_draw(self, pdf_path) -> None:
        """MuPDF lists one `get_images` entry per draw, all with the same xref.

        Iterating that against `get_image_rects` without collapsing the xrefs
        yields N**2 instances for N placements; this is the guard on that.
        """
        with open_document(pdf_path) as doc:
            images = doc[1].images()
        assert len(images) == 2, f"two draws should be two instances, got {len(images)}"
        assert len({img.xref for img in images}) == 1, "both draws share one resource"
        assert {round(img.rect.x0) for img in images} == {50, 150}

    def test_filled_black_rect_is_a_filled_drawing(self, pdf_path) -> None:
        with open_document(pdf_path) as doc:
            drawings = doc[0].drawings()
        filled = [d for d in drawings if d.filled and d.fill == (0.0, 0.0, 0.0)]
        assert filled, "the black rect should come back as a filled drawing"
        assert round(filled[0].rect.width) == 100


class TestWidgets:
    def test_acroform_fields(self, tmp_path) -> None:
        path = tmp_path / "form.pdf"
        doc = fitz.open()
        page = doc.new_page()
        widget = fitz.Widget()
        widget.field_name = "provider_name"
        widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
        widget.field_value = "Papilio Barton"
        widget.rect = fitz.Rect(72, 100, 272, 120)
        page.add_widget(widget)
        doc.save(str(path))
        doc.close()

        with open_document(path) as opened:
            fields = opened[0].widgets()
        assert [(f.field_name, f.field_value) for f in fields] == [("provider_name", "Papilio Barton")]
