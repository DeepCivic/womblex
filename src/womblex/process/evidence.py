"""Build and check evidence references (:mod:`womblex.store.evidence`).

:class:`EvidenceIndex` holds one document's elements under one element-text
layer and answers two questions about a span: *where is it* (the ``*_ref``
methods return an evidence reference at the level the location supports) and
*does the receipt hold* (:meth:`EvidenceIndex.holds`, at that level's own
precision). The narrative, the element offset map and the table markdowns all
come from ``chunker`` — the one place those coordinate spaces are defined — so
a reference cannot drift from the chunks and mentions it joins to.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from womblex.ingest.elements import BBox, Element
from womblex.process.chunk_stage import _load_elements
from womblex.process.chunker import TableText, element_spans, reassemble_narrative, table_texts
from womblex.process.text_overlay import apply_overlay, load_overlay
from womblex.store.evidence import (
    CELL_LAYER,
    DOCUMENT,
    ELEMENT,
    SPAN,
    TABLE_LAYER,
    evidence,
)

logger = logging.getLogger(__name__)


def _bbox(b: BBox | None) -> dict[str, float] | None:
    return None if b is None else {"x": b.x, "y": b.y, "width": b.width, "height": b.height}


class EvidenceIndex:
    """One document's anchors under one element-text layer.

    ``elements`` must already carry the layer's text (``apply_overlay``).
    """

    def __init__(self, elements: list[Element], text_layer: str = "elements") -> None:
        self.text_layer = text_layer
        self.narrative, _ = reassemble_narrative(elements)
        self._spans = element_spans(elements)
        self._by_order = {e.order: e for e in elements}
        self._tables: list[TableText] = table_texts(elements)

    def _located(
        self, level: str, elem: Element | None, *, sheet: str | None = None,
        cell: tuple[int, int] | None = None, bbox: BBox | None = None, **fields: Any,
    ) -> dict[str, Any]:
        return evidence(
            level,
            elem_order=elem.order if elem is not None else None,
            page=elem.page if elem is not None else None,
            bbox=_bbox(bbox or (elem.bbox if elem is not None else None)),
            sheet=sheet,
            cell_row=cell[0] if cell else None, cell_col=cell[1] if cell else None,
            **fields,
        )

    def _cell_box(self, elem: Element, cell: tuple[int, int] | None) -> BBox | None:
        if elem.kind != "table" or cell is None:
            return None
        return next((c.bbox for c in elem.cells or ()
                     if (c.row, c.col) == cell and c.bbox is not None), None)

    def narrative_ref(self, start: int, end: int) -> dict[str, Any] | None:
        """Span evidence for ``narrative[start:end]``; ``None`` outside the narrative."""
        if not 0 <= start <= end <= len(self.narrative):
            return None
        idx = bisect.bisect_right(self._spans, start, key=lambda s: s[1]) - 1
        order = self._spans[idx][0] if idx >= 0 and self._spans else None
        elem = self._by_order.get(order) if order is not None else None
        return self._located(
            SPAN, elem, char_start=start, char_end=end, text_layer=self.text_layer)

    def table_ref(self, table: TableText, start: int, end: int) -> dict[str, Any] | None:
        """Span evidence for ``table.markdown[start:end]``."""
        if not 0 <= start <= end <= len(table.markdown):
            return None
        elem = self._by_order.get(table.elem_order) if table.elem_order is not None else None
        return self._located(
            SPAN, elem, sheet=table.sheet, char_start=start, char_end=end,
            text_layer=TABLE_LAYER)

    def cell_ref(
        self, elem_order: int, start: int, end: int, *,
        cell: tuple[int, int] | None = None, sheet: str | None = None,
    ) -> dict[str, Any] | None:
        """Span evidence for ``[start:end]`` of one cell's value."""
        elem = self._by_order.get(elem_order)
        if elem is None:
            return None
        return self._located(
            SPAN, elem, sheet=sheet, cell=cell, bbox=self._cell_box(elem, cell),
            char_start=start, char_end=end, text_layer=CELL_LAYER)

    def element_ref(
        self, elem_order: int, *, cell: tuple[int, int] | None = None, sheet: str | None = None,
    ) -> dict[str, Any] | None:
        """Element evidence: the span lies somewhere in this element (or one of its cells)."""
        elem = self._by_order.get(elem_order)
        if elem is None:
            return None
        return self._located(
            ELEMENT, elem, sheet=sheet, cell=cell, bbox=self._cell_box(elem, cell))

    def document_ref(self) -> dict[str, Any]:
        """Document evidence: the span lies somewhere in this document."""
        return evidence(DOCUMENT, text_layer=self.text_layer)

    def resolve_table(
        self, *, elem_order: int | None, sheet: str | None = None,
        chunk_text: str | None = None, start: int = 0,
    ) -> TableText | None:
        """The table a chunk or mention belongs to.

        A table element is named by ``elem_order``. A sheet has no element
        position, so it is named by ``sheet`` when known, else found as the sheet
        whose markdown holds ``chunk_text`` at ``start``.
        """
        for t in self._tables:
            if elem_order is not None:
                if t.elem_order == elem_order:
                    return t
            elif t.elem_order is None and (
                (sheet is not None and t.sheet == sheet)
                or (sheet is None and chunk_text is not None
                    and t.markdown[start:start + len(chunk_text)] == chunk_text)
            ):
                return t
        return None

    def source_text(self, ref: Mapping[str, Any]) -> str | None:
        """The whole text a ``span`` reference's offsets index, or ``None`` if not found."""
        layer = ref.get("text_layer")
        if layer == TABLE_LAYER:
            t = self.resolve_table(elem_order=ref.get("elem_order"), sheet=ref.get("sheet"))
            return t.markdown if t is not None else None
        if layer == CELL_LAYER:
            return self._cell_text(ref)
        return self.narrative

    def _cell_text(self, ref: Mapping[str, Any]) -> str | None:
        order = ref.get("elem_order")
        elem = self._by_order.get(order) if order is not None else None
        if elem is None:
            return None
        if elem.kind == "sheet_cell":
            return elem.value or ""
        cell = (ref.get("cell_row"), ref.get("cell_col"))
        return next((c.value or "" for c in elem.cells or () if (c.row, c.col) == cell), None)

    def _element_text(self, ref: Mapping[str, Any]) -> str | None:
        """The text an ``element`` reference confines a span to."""
        order = ref.get("elem_order")
        elem = self._by_order.get(order) if order is not None else None
        if elem is None:
            return None
        if ref.get("cell_row") is not None:
            return self._cell_text({**ref, "text_layer": CELL_LAYER})
        if elem.kind == "table":
            t = self.resolve_table(elem_order=elem.order)
            return t.markdown if t is not None else None
        if elem.kind == "sheet_cell":
            return elem.value or ""
        return elem.text or ""

    def holds(self, ref: Mapping[str, Any], text: str) -> bool | None:
        """Whether the receipt holds at its own precision; ``None`` if it cannot be checked here.

        ``span``: the source slice equals ``text``. ``element``: ``text`` lies in
        that element or cell. ``document``: ``text`` lies in the narrative or a
        table markdown. A ``chunk`` receipt needs the chunk's text, which this
        index does not hold, so it is ``None`` here.
        """
        level = ref.get("anchor_level")
        if level == SPAN:
            source = self.source_text(ref)
            start, end = ref.get("char_start"), ref.get("char_end")
            if source is None or start is None or end is None:
                return False
            return source[start:end] == text
        if level == ELEMENT:
            source = self._element_text(ref)
            return source is not None and text in source
        if level == DOCUMENT:
            return text in self.narrative or any(text in t.markdown for t in self._tables)
        return None


class EvidenceIndexes:
    """A batch's per-document indexes, built on demand.

    Elements load once; each ``(source_hash, layer)`` index is built when first
    asked for, applying that layer's overlay. ``cell`` and ``table_markdown``
    references read the verbatim layer — overlays only rewrite text kinds.
    """

    def __init__(self, base_path: Path, text_source: str = "elements") -> None:
        self.base_path = base_path
        self.text_source = text_source
        try:
            self._elements = _load_elements(base_path)
        except FileNotFoundError:
            logger.warning(
                "evidence: %s has no elements or table-cells sidecar; its spans cannot be "
                "located or checked beyond the chunk", base_path.stem)
            self._elements = {}
        self._overlays: dict[str, dict[tuple[str, int], str] | None] = {}
        self._cache: dict[tuple[str, str], EvidenceIndex] = {}

    def get(self, source_hash: str, layer: str | None = None) -> EvidenceIndex | None:
        layer = layer or self.text_source
        if layer in (CELL_LAYER, TABLE_LAYER):
            layer = "elements"
        key = (source_hash, layer)
        if key not in self._cache:
            elements = self._elements.get(source_hash)
            if elements is None:
                return None
            if layer not in self._overlays:
                self._overlays[layer] = load_overlay(self.base_path, layer, required=True)
            elems = [replace(e) for e in elements]
            apply_overlay(source_hash, elems, self._overlays[layer])
            self._cache[key] = EvidenceIndex(elems, layer)
        return self._cache[key]
