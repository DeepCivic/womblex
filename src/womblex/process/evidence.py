"""Build and check evidence references (:mod:`womblex.store.evidence`).

:class:`EvidenceIndex` holds one document's elements under one element-text
layer and answers two questions about a span: *where is it* (``narrative_ref`` /
``table_ref`` / ``cell_ref`` return the evidence columns) and *what text sits
there* (``source_text``). The stages build rows with the first; the span check
asserts the second reproduces the row's text, at write time
(:func:`assert_evidence`) and over finished runs (:func:`verify_shards`).

The narrative, the element offset map and the table markdowns all come from
``chunker`` — the one place those coordinate spaces are defined — so a
reference cannot drift from the chunks and mentions it joins to.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from womblex.ingest.elements import BBox, Element
from womblex.process.chunk_stage import _batch_bases, _load_elements
from womblex.process.chunker import TableText, element_spans, reassemble_narrative, table_texts
from womblex.process.text_overlay import apply_overlay, load_overlay
from womblex.store.evidence import CELL_LAYER, TABLE_LAYER, EvidenceError, no_evidence

logger = logging.getLogger(__name__)

#: Rows reported per refusal; the count is always complete.
_REPORT_LIMIT = 5


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

    def _ref(
        self, elem: Element | None, start: int, end: int, layer: str, *,
        sheet: str | None = None, cell: tuple[int, int] | None = None,
        bbox: BBox | None = None,
    ) -> dict[str, Any]:
        ref = no_evidence()
        ref.update({
            "elem_order": elem.order if elem is not None else None,
            "page": elem.page if elem is not None else None,
            "bbox": _bbox(bbox or (elem.bbox if elem is not None else None)),
            "sheet": sheet,
            "cell_row": cell[0] if cell else None,
            "cell_col": cell[1] if cell else None,
            "char_start": start, "char_end": end, "text_layer": layer,
        })
        return ref

    def narrative_ref(self, start: int, end: int) -> dict[str, Any] | None:
        """Evidence for ``narrative[start:end]``; ``None`` outside the narrative."""
        if not 0 <= start <= end <= len(self.narrative):
            return None
        idx = bisect.bisect_right(self._spans, start, key=lambda s: s[1]) - 1
        order = self._spans[idx][0] if idx >= 0 and self._spans else None
        elem = self._by_order.get(order) if order is not None else None
        return self._ref(elem, start, end, self.text_layer)

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

    def table_ref(self, table: TableText, start: int, end: int) -> dict[str, Any] | None:
        """Evidence for ``table.markdown[start:end]``."""
        if not 0 <= start <= end <= len(table.markdown):
            return None
        elem = self._by_order.get(table.elem_order) if table.elem_order is not None else None
        return self._ref(elem, start, end, TABLE_LAYER, sheet=table.sheet)

    def cell_ref(
        self, elem_order: int, start: int, end: int, *,
        cell: tuple[int, int] | None = None, sheet: str | None = None,
    ) -> dict[str, Any] | None:
        """Evidence for ``[start:end]`` of one cell's value."""
        elem = self._by_order.get(elem_order)
        if elem is None:
            return None
        bbox = None
        if elem.kind == "table" and cell is not None:
            bbox = next((c.bbox for c in elem.cells or ()
                         if (c.row, c.col) == cell and c.bbox is not None), None)
        return self._ref(elem, start, end, CELL_LAYER, sheet=sheet, cell=cell, bbox=bbox)

    def source_text(self, ref: Mapping[str, Any]) -> str | None:
        """The whole text ``ref``'s offsets index, or ``None`` if it cannot be found."""
        layer = ref.get("text_layer")
        if layer == TABLE_LAYER:
            order = ref.get("elem_order")
            t = self.resolve_table(elem_order=order, sheet=ref.get("sheet"))
            return t.markdown if t is not None else None
        if layer == CELL_LAYER:
            order = ref.get("elem_order")
            elem = self._by_order.get(order) if order is not None else None
            if elem is None:
                return None
            if elem.kind == "sheet_cell":
                return elem.value or ""
            row, col = ref.get("cell_row"), ref.get("cell_col")
            return next((c.value or "" for c in elem.cells or ()
                         if (c.row, c.col) == (row, col)), None)
        return self.narrative

    def reproduces(self, ref: Mapping[str, Any], text: str) -> bool:
        """True when the reference's offsets select exactly ``text``."""
        source = self.source_text(ref)
        start, end = ref.get("char_start"), ref.get("char_end")
        if source is None or start is None or end is None:
            return False
        return source[start:end] == text


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
                "evidence: %s has no elements sidecar; spans cannot be anchored", base_path.stem)
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
            elems = [_copy(e) for e in elements]
            apply_overlay(source_hash, elems, self._overlays[layer])
            self._cache[key] = EvidenceIndex(elems, layer)
        return self._cache[key]


def _copy(e: Element) -> Element:
    return replace(e)


@dataclass
class EvidenceReport:
    """Outcome of checking sidecar rows against their sources."""

    checked: int = 0
    unanchored: int = 0
    mismatches: list[str] = field(default_factory=list)
    n_mismatched: int = 0

    @property
    def ok(self) -> bool:
        return self.n_mismatched == 0

    def merge(self, other: EvidenceReport) -> None:
        self.checked += other.checked
        self.unanchored += other.unanchored
        self.n_mismatched += other.n_mismatched
        self.mismatches.extend(other.mismatches)


#: ``(source_hash, text_layer)`` -> that document's index, or ``None`` if unknown.
IndexLookup = Callable[[str, str | None], EvidenceIndex | None]


def check_rows(
    rows: Iterable[Mapping[str, Any]], lookup: IndexLookup, *, text_key: str, label: str,
) -> EvidenceReport:
    """Check each anchored row's evidence reproduces ``row[text_key]``.

    A row with no ``char_start`` is unanchored — counted, not refused: a span the
    stage could not place is reported, never guessed at.
    """
    report = EvidenceReport()
    for row in rows:
        if row.get("char_start") is None:
            report.unanchored += 1
            continue
        report.checked += 1
        index = lookup(row["source_hash"], row.get("text_layer"))
        if index is None or not index.reproduces(row, row[text_key] or ""):
            report.n_mismatched += 1
            if len(report.mismatches) < _REPORT_LIMIT:
                report.mismatches.append(
                    f"{label} {row['source_hash'][:12]} "
                    f"[{row.get('text_layer')}:{row.get('char_start')}-{row.get('char_end')}] "
                    f"{(row[text_key] or '')[:40]!r}"
                )
    return report


def assert_evidence(
    rows: list[dict[str, Any]], lookup: IndexLookup, *, text_key: str, label: str,
    base: Path,
) -> EvidenceReport:
    """Refuse a batch whose evidence does not reproduce its text.

    Called before the sidecar is written, so a mismatch publishes nothing.
    """
    report = check_rows(rows, lookup, text_key=text_key, label=label)
    if report.unanchored:
        logger.warning(
            "%s: %s has %d span(s) it could not anchor to the source (evidence left null)",
            label, base.stem, report.unanchored,
        )
    if not report.ok:
        raise EvidenceError(
            f"{label}: {base.stem} has {report.n_mismatched} of {report.checked} span(s) whose "
            f"evidence does not reproduce their text; nothing written. First: "
            + "; ".join(report.mismatches)
        )
    return report


def verify_shards(shard_dir: Path) -> EvidenceReport:
    """Check every money, entity-link and PII sidecar in ``shard_dir`` against its source."""
    from womblex.store.entity_links_output import read_entity_links
    from womblex.store.money_output import read_money_spans
    from womblex.store.pii_output import read_pii_spans

    readers: tuple[tuple[str, str, Callable[[Path], Any]], ...] = (
        ("money_spans", "text", read_money_spans),
        ("entity_links", "mention_text", read_entity_links),
        ("pii_spans", "text", read_pii_spans),
    )
    total = EvidenceReport()
    for base in _batch_bases(shard_dir):
        indexes: EvidenceIndexes | None = None
        for label, text_key, read in readers:
            try:
                rows = read(base).to_pylist()
            except FileNotFoundError:
                continue
            indexes = indexes or EvidenceIndexes(base)
            total.merge(check_rows(
                rows, indexes.get, text_key=text_key, label=f"{label}:{base.stem}"))
    return total
