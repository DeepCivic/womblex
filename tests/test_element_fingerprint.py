"""Element fingerprints: order-free, one per figure and per sheet, recomputable from a shard."""

from __future__ import annotations

from dataclasses import replace

import pyarrow.parquet as pq
import pytest

from tests._synthetic import SYNTHETIC_DIR
from womblex.ingest.detect import detect_file_type
from womblex.ingest.elements import Cell, Element, FieldEntry
from womblex.ingest.extract import extract_text
from womblex.store.element_fingerprint import (
    FINGERPRINT_BYTES,
    fingerprints,
    occurrences,
    read_shard_elements,
)
from womblex.store.output import ELEMENT_SCHEMA, _shard_paths, read_elements, write_results


def _para(order: int, text: str) -> Element:
    return Element(order=order, kind="paragraph", extractor="t", text=text)


def _sheet(order: int, name: str, values: list[str]) -> list[Element]:
    meta = Element(order=order, kind="sheet_meta", extractor="t", sheet=name, meta={"rows": "1"})
    cells = [
        Element(order=order + 1 + i, kind="sheet_cell", extractor="t", sheet=name, row=0, col=i, value=v)
        for i, v in enumerate(values)
    ]
    return [meta, *cells]


def _renumber(elements: list[Element]) -> list[Element]:
    return [replace(e, order=i) for i, e in enumerate(elements)]


def test_width_and_determinism():
    elems = [_para(0, "Wombats dig burrows.")]
    fp = fingerprints(elems)[0]
    assert fp is not None and len(fp) == FINGERPRINT_BYTES
    assert fingerprints([_para(5, "Wombats dig burrows.")])[5] == fp


def test_insert_leaves_other_fingerprints_unchanged():
    before = [_para(0, "a"), _para(1, "b"), _para(2, "c")]
    after = _renumber([before[0], _para(0, "inserted"), before[1], before[2]])
    old, new = fingerprints(before), fingerprints(after)
    assert [old[0], old[1], old[2]] == [new[0], new[2], new[3]]


def test_table_and_form_cover_their_cells_and_fields():
    table = Element(order=0, kind="table", extractor="t", cells=[Cell(row=0, col=0, value="12")])
    form = Element(order=1, kind="form", extractor="t", fields=[FieldEntry(name="Name", value="Kim")])
    base = fingerprints([table, form])
    edited = fingerprints([
        replace(table, cells=[Cell(row=0, col=0, value="13")]),
        replace(form, fields=[FieldEntry(name="Name", value="Lee")]),
    ])
    assert base[0] != edited[0] and base[1] != edited[1]


def test_sheet_fingerprint_on_meta_and_cells_carry_none():
    elems = _renumber([*_sheet(0, "A", ["1", "2"]), *_sheet(0, "B", ["3"])])
    prints = fingerprints(elems)
    assert all(prints[e.order] is None for e in elems if e.kind == "sheet_cell")
    metas = [e.order for e in elems if e.kind == "sheet_meta"]
    assert all(prints[o] is not None for o in metas)

    changed = _renumber([*_sheet(0, "A", ["1", "9"]), *_sheet(0, "B", ["3"])])
    moved = fingerprints(changed)
    assert moved[metas[0]] != prints[metas[0]]
    assert moved[metas[1]] == prints[metas[1]]


def test_sheet_cells_hash_in_row_col_order():
    meta, c0, c1 = _sheet(0, "A", ["1", "2"])
    swapped = [meta, replace(c1, order=1), replace(c0, order=2)]
    assert fingerprints([meta, c0, c1])[0] == fingerprints(swapped)[0]


def test_repeats_share_a_fingerprint_and_count_up():
    elems = [_para(0, "Page 1"), _para(1, "body"), _para(2, "Page 1"), _para(3, "Page 1")]
    prints = fingerprints(elems)
    assert prints[0] == prints[2] == prints[3]
    assert occurrences(elems, prints) == {0: 0, 1: 0, 2: 1, 3: 2}


def test_ordinal_is_per_kind():
    elems = [_para(0, "x"), Element(order=1, kind="heading", extractor="t", text="x")]
    assert occurrences(elems, fingerprints(elems)) == {0: 0, 1: 0}


_FIXTURES = [
    "documents/koala-habitat-audit_transcript.txt",
    "documents/wombat-portfolio-budget-statements.docx",
    "spreadsheets/platypus-sightings-register.csv",
    "spreadsheets/echidna-population-statistics.xlsx",
    pytest.param("documents/quokka-care-decision-notice_redacted.pdf", marks=pytest.mark.slow),
    pytest.param("documents/koala-habitat-audit.pdf", marks=pytest.mark.slow),
    pytest.param("documents/bilby-schedule-of-documents.pdf", marks=pytest.mark.slow),
]


@pytest.mark.parametrize("rel", _FIXTURES)
def test_written_fingerprint_equals_recomputed(rel, tmp_path):
    path = SYNTHETIC_DIR / rel
    res = extract_text(path, detect_file_type(path))[0]
    shard = tmp_path / "batch-0001.parquet"
    write_results([("doc", str(path), res)], shard)
    rows = read_elements(shard).to_pylist()
    written = {r["elem_order"]: r["fingerprint"] for r in rows}
    assert written == fingerprints(res.elements)
    (elems,) = read_shard_elements(shard).values()
    assert fingerprints(elems) == written


def test_shard_without_column_reads_null(tmp_path):
    shard = tmp_path / "batch-0001.parquet"
    write_results([("doc", "t.txt", extract_text(
        SYNTHETIC_DIR / "documents/koala-habitat-audit_transcript.txt",
        detect_file_type(SYNTHETIC_DIR / "documents/koala-habitat-audit_transcript.txt"),
    )[0])], shard)
    path = _shard_paths(shard)["elements"]
    pq.write_table(pq.read_table(path).drop_columns(["fingerprint"]), path)
    table = read_elements(shard)
    assert table.schema.equals(ELEMENT_SCHEMA)
    assert table.num_rows > 0 and table.column("fingerprint").null_count == table.num_rows
