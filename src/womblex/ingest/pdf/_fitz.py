"""The fitz (PyMuPDF) backend behind the `ingest.pdf` seam.

The only module in `src/` that may import fitz once the port is complete. It
translates between MuPDF's shapes and `types.py`'s: dicts and tuples in,
dataclasses out; pixmaps in, arrays out.
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import fitz
import numpy as np

from womblex.ingest.pdf.types import (
    Block,
    Drawing,
    DrawingKind,
    FoundTable,
    Line,
    Matrix,
    PageImage,
    Rect,
    Span,
    TableStrategy,
    Widget,
    Word,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# Suppress the pymupdf_layout suggestion find_tables() emits.
warnings.filterwarnings("ignore", message=".*pymupdf_layout.*")

#: MuPDF's span flag bit for a bold font. Its font names are the other signal,
#: since synthetic bold does not always set the flag.
_BOLD_FLAG = 1 << 4

#: MuPDF's drawing type codes, in its own spelling.
_DRAWING_KINDS: dict[str, DrawingKind] = {
    "f": "fill", "s": "stroke", "fs": "fill_stroke", "sf": "fill_stroke",
}


def _rect(r: Any) -> Rect:
    return Rect.of((float(r[0]), float(r[1]), float(r[2]), float(r[3])))


def _text_kwargs(dehyphenate: bool) -> dict[str, int]:
    """``flags`` only when dehyphenating: passing 0 would drop MuPDF's defaults."""
    return {"flags": fitz.TEXT_DEHYPHENATE} if dehyphenate else {}


def _is_bold(span: dict) -> bool:
    if int(span.get("flags", 0)) & _BOLD_FLAG:
        return True
    return "bold" in str(span.get("font", "")).lower()


class FitzPage:
    """One fitz page, presented as `types.Page`."""

    def __init__(self, page: fitz.Page) -> None:
        self._page = page

    @property
    def number(self) -> int:
        return int(self._page.number)

    @property
    def rect(self) -> Rect:
        return _rect(self._page.rect)

    @property
    def rotation(self) -> int:
        return int(self._page.rotation)

    @property
    def rotation_matrix(self) -> Matrix:
        m = self._page.rotation_matrix
        return (m.a, m.b, m.c, m.d, m.e, m.f)

    @property
    def doc_name(self) -> str:
        parent = self._page.parent
        return str(getattr(parent, "name", "") or "")

    def plain_text(self, *, dehyphenate: bool = True) -> str:
        return str(self._page.get_text("text", **_text_kwargs(dehyphenate)))

    def text_dict(self) -> list[Block]:
        raw = self._page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        out: list[Block] = []
        for number, block in enumerate(raw.get("blocks", [])):
            bbox = _rect(block["bbox"])
            if block.get("type") != 0:
                out.append(Block(bbox=bbox, number=number, kind="image"))
                continue
            lines = tuple(
                Line(
                    bbox=_rect(line["bbox"]),
                    spans=tuple(
                        Span(
                            text=str(span.get("text", "")),
                            bbox=_rect(span["bbox"]),
                            size=float(span.get("size", 0.0)),
                            font=str(span.get("font", "")),
                            bold=_is_bold(span),
                        )
                        for span in line.get("spans", [])
                    ),
                )
                for line in block.get("lines", [])
            )
            out.append(Block(bbox=bbox, number=number, lines=lines))
        return out

    def words(self, *, dehyphenate: bool = True) -> list[Word]:
        raw = self._page.get_text("words", **_text_kwargs(dehyphenate))
        return [
            Word(float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4]),
                 int(w[5]), int(w[6]), int(w[7]))
            for w in raw
        ]

    def text_blocks(self, *, dehyphenate: bool = True) -> list[Block]:
        raw = self._page.get_text("blocks", **_text_kwargs(dehyphenate))
        return [
            Block(
                bbox=Rect.of((float(b[0]), float(b[1]), float(b[2]), float(b[3]))),
                number=int(b[5]),
                kind="text" if int(b[6]) == 0 else "image",
                text=str(b[4]),
            )
            for b in raw
        ]

    def find_tables(self, *, strategy: TableStrategy = "lines") -> list[FoundTable]:
        # find_tables prints a layout hint to stdout on some builds.
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            found = self._page.find_tables(strategy=strategy)
            out: list[FoundTable] = []
            for table in found.tables:
                rows = tuple(
                    tuple(None if cell is None else str(cell) for cell in row)
                    for row in table.extract()
                )
                out.append(FoundTable(
                    bbox=_rect(table.bbox),
                    row_count=int(table.row_count),
                    col_count=int(table.col_count),
                    rows=rows,
                ))
        finally:
            sys.stdout = old_stdout
        return out

    def render(self, *, dpi: int, clip: Rect | None = None) -> np.ndarray:
        rect = None if clip is None else fitz.Rect(*clip.as_tuple())
        pix = self._page.get_pixmap(dpi=dpi, clip=rect)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            return np.ascontiguousarray(img[:, :, :3])
        if pix.n == 1:
            return np.repeat(img, 3, axis=2)
        return img

    def images(self) -> list[PageImage]:
        """Every drawn instance, de-duplicated by xref first.

        `get_images` lists one entry per draw, all carrying the same xref, and
        `get_image_rects(xref)` already returns every rect for that resource —
        so iterating both without collapsing the xrefs yields each draw once
        per draw, N**2 for N placements. Grouping first is what makes one draw
        one `PageImage`.
        """
        out: list[PageImage] = []
        for xref in dict.fromkeys(int(info[0]) for info in self._page.get_images(full=True)):
            try:
                rects = self._page.get_image_rects(xref)
            except Exception:
                continue
            out += [PageImage(rect=_rect(r), xref=xref) for r in rects]
        return out

    def drawings(self) -> list[Drawing]:
        out: list[Drawing] = []
        for drawing in self._page.get_drawings():
            kind = _DRAWING_KINDS.get(str(drawing.get("type", "")))
            rect = drawing.get("rect")
            if kind is None or rect is None:
                continue
            fill = drawing.get("fill")
            out.append(Drawing(
                kind=kind,
                rect=_rect(rect),
                fill=None if fill is None else (float(fill[0]), float(fill[1]), float(fill[2])),
            ))
        return out

    def widgets(self) -> list[Widget]:
        return [
            Widget(
                field_name=str(w.field_name or ""),
                field_value=str(w.field_value or ""),
                rect=_rect(w.rect),
            )
            for w in self._page.widgets()
        ]


class FitzDocument:
    """One fitz document, presented as `types.Document`."""

    def __init__(self, path: Path | None = None, *, native: fitz.Document | None = None) -> None:
        self._doc = native if native is not None else fitz.open(str(path))

    @property
    def page_count(self) -> int:
        return int(self._doc.page_count)

    @property
    def name(self) -> str:
        return str(self._doc.name or "")

    def __len__(self) -> int:
        return self.page_count

    def __getitem__(self, index: int) -> FitzPage:
        return FitzPage(self._doc[index])

    def __iter__(self) -> Iterator[FitzPage]:
        return (FitzPage(page) for page in self._doc)

    def select(self, pages: list[int]) -> None:
        self._doc.select(pages)

    def close(self) -> None:
        self._doc.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_document(path: Path) -> FitzDocument:
    return FitzDocument(path)
