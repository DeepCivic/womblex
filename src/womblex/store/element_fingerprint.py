"""Element fingerprints: whether an element's content changed between two versions.

The ``elements.fingerprint`` column is the first 8 bytes of SHA-256 over the
element's ``content_digest`` payload without ``order``, so inserting an element
leaves every other element's fingerprint as it was; reading order stays in
``elem_order``. Every element carries one except ``sheet_cell``: a table's
covers its cells and a form's its fields (they are in the payload), and a
sheet's sits on its ``sheet_meta`` row, taken over that row and the
``sheet_cell`` elements that follow it, in (row, col) order.

A fingerprint marks content, not identity: identical elements (running headers,
page breaks) share one. The occurrence ordinal tells them apart and is derived,
never stored. A plain hash, not keyed: the column sits on a ``raw`` file.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from womblex.ingest.elements import Cell, Element, FieldEntry
from womblex.store.content_digest import content_fields

FINGERPRINT_BYTES = 8


def _digest(payload: object) -> bytes:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).digest()[:FINGERPRINT_BYTES]


def _payload(e: Element) -> list[object]:
    return [e.kind, *content_fields(e)]


def fingerprints(elements: Sequence[Element]) -> dict[int, bytes | None]:
    """Each element's fingerprint by ``order``; ``None`` for ``sheet_cell``."""
    ordered = sorted(elements, key=lambda e: e.order)
    out: dict[int, bytes | None] = {}
    for i, e in enumerate(ordered):
        if e.kind == "sheet_cell":
            out[e.order] = None
            continue
        if e.kind != "sheet_meta":
            out[e.order] = _digest(_payload(e))
            continue
        cells: list[Element] = []
        for nxt in ordered[i + 1:]:
            if nxt.kind != "sheet_cell":
                break
            cells.append(nxt)
        cells.sort(key=lambda c: (c.row if c.row is not None else -1, c.col if c.col is not None else -1))
        out[e.order] = _digest([_payload(e), [_payload(c) for c in cells]])
    return out


def occurrences(elements: Sequence[Element], prints: dict[int, bytes | None]) -> dict[int, int]:
    """Each fingerprinted element's index among the document's elements of its kind and fingerprint."""
    seen: dict[tuple[str, bytes], int] = defaultdict(int)
    out: dict[int, int] = {}
    for e in sorted(elements, key=lambda e: e.order):
        fp = prints.get(e.order)
        if fp is None:
            continue
        out[e.order] = seen[(e.kind, fp)]
        seen[(e.kind, fp)] += 1
    return out


def read_shard_elements(base: Path) -> dict[str, list[Element]]:
    """A shard's elements per ``source_hash``, with table cells, form fields and meta.

    Enough to recompute a fingerprint the shard does not carry: geometry and
    confidence, which the payload leaves out, are not restored.
    """
    from womblex.store.output import read_elements, read_form_fields, read_table_cells

    cells: dict[tuple[str, int], list[Cell]] = defaultdict(list)
    for r in read_table_cells(base).to_pylist():
        cells[(r["source_hash"], r["parent_elem_order"])].append(Cell(
            row=r["row"], col=r["col"], value=r["value"] or "", rowspan=r["rowspan"] or 1,
            colspan=r["colspan"] or 1, value_type=r["value_type"] or "text",
        ))
    fields: dict[tuple[str, int], list[tuple[int, FieldEntry]]] = defaultdict(list)
    for r in read_form_fields(base).to_pylist():
        fields[(r["source_hash"], r["parent_elem_order"])].append((r["field_index"], FieldEntry(
            name=r["name"] or "", value=r["value"] or "", field_type=r["field_type"] or "text",
        )))
    out: dict[str, list[Element]] = defaultdict(list)
    for r in read_elements(base).to_pylist():
        key = (r["source_hash"], r["elem_order"])
        out[r["source_hash"]].append(Element(
            order=r["elem_order"], kind=r["kind"], extractor=r["extractor"] or "", page=r["page"],
            text=r["text"], cells=cells.get(key), header_rows=r["header_rows"],
            fields=[f for _, f in sorted(fields[key], key=lambda p: p[0])] if key in fields else None,
            alt_text=r["alt_text"], sheet=r["sheet"], row=r["row"], col=r["col"], value=r["value"],
            value_type=r["value_type"], formula=r["formula"], number_format=r["number_format"],
            merge_range=r["merge_range"], meta=dict(r["meta"]) if r["meta"] else {},
        ))
    for elems in out.values():
        elems.sort(key=lambda e: e.order)
    return dict(out)


__all__ = ["FINGERPRINT_BYTES", "fingerprints", "occurrences", "read_shard_elements"]
