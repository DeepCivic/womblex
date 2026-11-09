"""HTML extraction: block-level structure via the standard-library parser.

Headings, paragraphs, list items, preformatted blocks and tables become
elements in document order, as ``MarkdownExtractor`` does for Markdown. The
standard library is enough because only block boundaries are needed, so the
format adds no dependency. Inline markup is dropped and its text kept, with
runs of whitespace collapsed as a browser would render them. ``script``,
``style`` and ``head`` content is not document text and is skipped.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import cast

from womblex.ingest.elements import Cell, Element, ElementKind
from womblex.ingest.extract import ExtractionMetadata, ExtractionResult, PageResult

_SKIP = {"script", "style", "head", "template", "noscript"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Containers whose own text, outside any nested block, is a paragraph.
_BLOCKS = {"p", "div", "section", "article", "blockquote", "caption", "dd", "dt", "figcaption"}
_WS = re.compile(r"\s+")
_META_CHARSET = re.compile(rb"<meta[^>]+charset=[\"']?\s*([\w-]+)", re.IGNORECASE)


def read_html(path: Path) -> str:
    """File text: UTF-8 (BOM removed), else a declared ``<meta charset>``, else Latin-1."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    declared = _META_CHARSET.search(raw[:4096])
    for encoding in ([declared.group(1).decode("ascii")] if declared else []) + ["latin-1"]:
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("latin-1")


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[Element] = []
        self._skip = 0
        self._kind: ElementKind | None = None
        self._buf: list[str] = []
        self._pre = False
        self._list_depth = 0
        self._table: list[list[str]] | None = None
        self._header_rows: list[int] = []
        self._cell: list[str] | None = None
        self._row_is_header = False
        # Tables inside a cell (layout tables) are flattened into that cell's
        # text: restarting the table would drop the rows already read.
        self._nested = 0

    # -- helpers -----------------------------------------------------------
    def _flush(self) -> None:
        text = "".join(self._buf)
        text = text.strip() if self._pre else _WS.sub(" ", text).strip()
        self._buf = []
        if text and self._kind:
            self.elements.append(Element(
                order=len(self.elements), kind=self._kind, extractor="html",
                text=text, confidence=0.9,
            ))
        self._kind = None

    def _open(self, kind: ElementKind) -> None:
        self._flush()
        self._kind = kind

    # -- parser callbacks --------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skip += 1
        if self._skip:
            return
        if tag == "table" and self._table is not None:
            self._nested += 1
            self._spaced()
        elif self._nested:
            if tag in ("tr", "td", "th", "br"):
                self._spaced()
        elif tag == "table":
            self._flush()
            self._table, self._header_rows = [], []
        elif tag == "tr" and self._table is not None:
            self._end_cell()  # </td> and </tr> may be omitted
            self._table.append([])
            self._row_is_header = False
        elif tag in ("td", "th") and self._table is not None:
            self._end_cell()
            if not self._table:
                self._table.append([])  # a cell before any <tr>
            self._cell = []
            self._row_is_header = self._row_is_header or tag == "th"
        elif self._table is not None:
            return
        elif tag in _HEADINGS:
            self._open("heading")
        elif tag == "li":
            self._open("list_item")
        elif tag in ("ul", "ol"):
            self._flush()
            self._list_depth += 1
        elif tag == "pre":
            self._open("paragraph")
            self._pre = True
        elif tag in _BLOCKS:
            self._open("list_item" if self._list_depth and tag == "p" else "paragraph")
        elif tag == "br":
            self._buf.append("\n" if self._pre else " ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if self._nested:
            if tag == "table":
                self._nested -= 1
            self._spaced()
        elif tag in ("td", "th"):
            self._end_cell()
        elif tag == "table" and self._table is not None:
            self._finish_table()
        elif self._table is not None:
            return
        elif tag in _HEADINGS or tag in _BLOCKS or tag in ("li", "pre"):
            self._flush()
            self._pre = False
        elif tag in ("ul", "ol"):
            self._flush()
            self._list_depth = max(0, self._list_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._cell is not None:
            self._cell.append(data)
        elif self._table is None:
            if self._kind is None and data.strip():
                self._kind = "paragraph"  # text outside any block tag
            self._buf.append(data)

    def _spaced(self) -> None:
        if self._cell is not None:
            self._cell.append(" ")

    def _end_cell(self) -> None:
        if self._cell is None or not self._table:
            return
        self._table[-1].append(_WS.sub(" ", "".join(self._cell)).strip())
        self._cell = None
        if self._row_is_header and len(self._table) - 1 not in self._header_rows:
            self._header_rows.append(len(self._table) - 1)

    def _finish_table(self) -> None:
        self._end_cell()
        rows = self._table or []
        cells = [
            Cell(row=r, col=c, value=value)
            for r, row in enumerate(rows) for c, value in enumerate(row)
        ]
        if any(cell.value for cell in cells):
            self.elements.append(Element(
                order=len(self.elements), kind="table", extractor="html",
                cells=cells, header_rows=self._header_rows, confidence=0.85,
            ))
        self._table, self._header_rows, self._cell, self._nested = None, [], None, 0

    def close(self) -> None:
        super().close()
        if self._table is not None:
            self._finish_table()
        self._flush()


def html_elements(text: str) -> list[Element]:
    """Ordered elements for HTML *text*."""
    parser = _Collector()
    parser.feed(text)
    parser.close()
    return parser.elements


class HtmlExtractor:
    """Extract elements from an HTML file in document order."""

    def extract_path(self, path: Path) -> ExtractionResult:
        elements = html_elements(read_html(path))
        page_text = "\n\n".join(cast("str", e.text) for e in elements if e.text)
        return ExtractionResult(
            pages=[PageResult(page_number=0, text=page_text, method="html")],
            elements=elements,
            method="html",
            metadata=ExtractionMetadata(
                extraction_strategy="html",
                confidence=0.9,
                processing_time=0.0,
                page_count=1,
                text_coverage=1.0 if page_text else 0.0,
            ),
        )
