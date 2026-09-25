"""Markdown extraction: block-level structure via markdown-it-py + GFM tables.

Thin adapter over ``markdown-it-py``: the parser is used only to find block
boundaries (headings, paragraphs, list items, tables). ``Token.content`` is
the parser's raw pre-inline-parse text for a block — markdown syntax inside
it (``**bold**`` etc) is left untouched, matching the verbatim text policy at
the extraction boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from markdown_it import MarkdownIt
from markdown_it.token import Token

from womblex.ingest.elements import Cell, Element, ElementKind
from womblex.ingest.extract import ExtractionMetadata, ExtractionResult, PageResult

_MD = MarkdownIt("commonmark").enable("table")


class MarkdownExtractor:
    """Extract elements from a Markdown file in document order.

    Headings, paragraphs, list items and GFM pipe tables become
    ``heading`` / ``paragraph`` / ``list_item`` / ``table`` elements,
    interleaved as they appear in the source — the same document-order
    handling ``DocxExtractor`` gives prose alongside tables.
    """

    def extract_path(self, path: Path) -> ExtractionResult:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        elements = _elements_from_tokens(_MD.parse(text))

        page_text = "\n\n".join(e.text for e in elements if e.text)
        return ExtractionResult(
            pages=[PageResult(page_number=0, text=page_text, method="markdown")],
            elements=elements,
            method="markdown",
            metadata=ExtractionMetadata(
                extraction_strategy="markdown",
                confidence=0.9,
                processing_time=0.0,
                page_count=1,
                text_coverage=1.0 if page_text else 0.0,
            ),
        )


def _elements_from_tokens(tokens: list[Token]) -> list[Element]:
    elements: list[Element] = []
    order = 0
    list_depth = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type in ("bullet_list_open", "ordered_list_open"):
            list_depth += 1
            i += 1
        elif token.type in ("bullet_list_close", "ordered_list_close"):
            list_depth -= 1
            i += 1
        elif token.type == "heading_open":
            text = tokens[i + 1].content.strip()
            if text:
                elements.append(Element(
                    order=order, kind="heading", extractor="markdown",
                    text=text, confidence=0.9,
                ))
                order += 1
            i += 3
        elif token.type == "paragraph_open":
            text = tokens[i + 1].content.strip()
            if text:
                kind = cast("ElementKind", "list_item" if list_depth > 0 else "paragraph")
                elements.append(Element(
                    order=order, kind=kind, extractor="markdown",
                    text=text, confidence=0.9,
                ))
                order += 1
            i += 3
        elif token.type in ("fence", "code_block", "html_block"):
            text = token.content.strip()
            if text:
                elements.append(Element(
                    order=order, kind="paragraph", extractor="markdown",
                    text=text, confidence=0.85,
                ))
                order += 1
            i += 1
        elif token.type == "table_open":
            end = _matching_close(tokens, i, "table_close")
            cells, header_rows = _table_cells(tokens[i:end + 1])
            if cells:
                elements.append(Element(
                    order=order, kind="table", extractor="markdown",
                    cells=cells, header_rows=header_rows, confidence=0.85,
                ))
                order += 1
            i = end + 1
        else:
            i += 1
    return elements


def _matching_close(tokens: list[Token], start: int, close_type: str) -> int:
    for j in range(start + 1, len(tokens)):
        if tokens[j].type == close_type:
            return j
    return len(tokens) - 1


def _table_cells(tokens: list[Token]) -> tuple[list[Cell], list[int]]:
    """Cells + header row indices for one ``table_open``..``table_close`` span.

    GFM tables have exactly one header row (the thead); a stray extra thead
    row would still be captured correctly since each ``tr_open`` inside it
    is recorded.
    """
    cells: list[Cell] = []
    header_rows: list[int] = []
    in_header = False
    row = -1
    col = 0
    for token in tokens:
        if token.type == "thead_open":
            in_header = True
        elif token.type == "thead_close":
            in_header = False
        elif token.type == "tr_open":
            row += 1
            col = 0
            if in_header:
                header_rows.append(row)
        elif token.type == "inline":
            cells.append(Cell(row=row, col=col, value=token.content.strip()))
            col += 1
    return cells, header_rows
