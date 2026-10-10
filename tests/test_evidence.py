"""The evidence reference: building it, checking it, and reading older files.

Money, entity-link and PII sidecars locate each span with one shape
(``store/evidence.py``); a span whose evidence does not reproduce its text
refuses the batch, and ``verify-evidence`` re-checks finished runs.
"""

from __future__ import annotations

import argparse
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests._shard import DOC, contact_shard, table_markdown
from womblex.cli.verify import cmd_verify_evidence
from womblex.config import MoneyConfig
from womblex.process.evidence import EvidenceIndexes, assert_evidence, verify_shards
from womblex.process.money_stage import money_shards
from womblex.store.entity_links_output import read_entity_links
from womblex.store.evidence import EVIDENCE_COLUMNS, EvidenceError, no_evidence
from womblex.store.money_output import (
    MONEY_SPANS_SCHEMA,
    money_spans_path_for,
    read_money_spans,
)
from womblex.store.pii_output import read_pii_spans

NARRATIVE = "Contact Jane Doe about the $5,000 grant.\n\nSigned by the delegate."


def test_narrative_ref_names_the_element_the_span_starts_in(tmp_path):
    base = contact_shard(tmp_path)
    index = EvidenceIndexes(base).get(DOC)
    assert index is not None and index.narrative == NARRATIVE

    start = NARRATIVE.index("delegate")
    ref = index.narrative_ref(start, start + len("delegate"))
    assert ref is not None
    assert (ref["elem_order"], ref["page"], ref["text_layer"]) == (2, 1, "elements")
    assert index.reproduces(ref, "delegate")
    assert not index.reproduces(ref, "Delegate")


def test_a_span_outside_the_narrative_has_no_reference(tmp_path):
    index = EvidenceIndexes(contact_shard(tmp_path)).get(DOC)
    assert index is not None
    assert index.narrative_ref(0, len(NARRATIVE) + 1) is None


def test_table_and_cell_refs_resolve_to_their_text(tmp_path):
    base = contact_shard(tmp_path)
    index = EvidenceIndexes(base).get(DOC)
    assert index is not None
    markdown = table_markdown(base)

    table = index.resolve_table(elem_order=1)
    assert table is not None
    start = markdown.index("John Citizen")
    ref = index.table_ref(table, start, start + len("John Citizen"))
    assert ref is not None and ref["elem_order"] == 1 and ref["text_layer"] == "table_markdown"
    assert index.reproduces(ref, "John Citizen")

    cell_ref = index.cell_ref(1, 0, 5, cell=(1, 1))
    assert cell_ref is not None
    assert (cell_ref["cell_row"], cell_ref["cell_col"], cell_ref["text_layer"]) == (1, 1, "cell")
    assert index.reproduces(cell_ref, "1,200"[:5])


def test_money_evidence_reproduces_every_locus(tmp_path):
    base = contact_shard(tmp_path)
    money_shards(tmp_path, MoneyConfig())

    rows = read_money_spans(base).to_pylist()
    assert {r["locus"] for r in rows} == {"narrative", "table_cell"}
    indexes = EvidenceIndexes(base)
    for r in rows:
        index = indexes.get(r["source_hash"], r["text_layer"])
        assert index is not None and index.reproduces(r, r["text"]), r
    narrative = next(r for r in rows if r["locus"] == "narrative")
    assert narrative["value"] == Decimal("5000.0000") and narrative["elem_order"] == 0


def test_a_span_that_does_not_reproduce_its_text_refuses_the_batch(tmp_path):
    base = contact_shard(tmp_path)
    indexes = EvidenceIndexes(base)
    index = indexes.get(DOC)
    assert index is not None
    good = index.narrative_ref(NARRATIVE.index("$5,000"), NARRATIVE.index("$5,000") + 6)
    row = {"source_hash": DOC, "text": "$6,000", **(good or no_evidence())}

    with pytest.raises(EvidenceError, match="does not reproduce"):
        assert_evidence([row], indexes.get, text_key="text", label="money_spans", base=base)


def test_an_unanchored_span_is_counted_not_refused(tmp_path):
    base = contact_shard(tmp_path)
    row = {"source_hash": DOC, "text": "anything", **no_evidence()}
    report = assert_evidence(
        [row], EvidenceIndexes(base).get, text_key="text", label="pii_spans", base=base)
    assert (report.checked, report.unanchored, report.ok) == (0, 1, True)


def test_verify_shards_passes_a_clean_run_and_flags_a_shifted_span(tmp_path, capsys):
    base = contact_shard(tmp_path)
    money_shards(tmp_path, MoneyConfig())
    assert verify_shards(tmp_path).ok

    path = money_spans_path_for(base)
    table = pq.read_table(str(path))
    shifted = pc_add(table, "char_start", 1)
    pq.write_table(shifted, str(path))

    report = verify_shards(tmp_path)
    assert not report.ok and report.n_mismatched >= 1

    assert cmd_verify_evidence(argparse.Namespace(run_dir=tmp_path)) == 2
    assert "mismatch" in capsys.readouterr().out


def pc_add(table: pa.Table, column: str, delta: int) -> pa.Table:
    import pyarrow.compute as pc

    idx = table.schema.get_field_index(column)
    return table.set_column(idx, column, pc.add(table[column], pa.scalar(delta, pa.int32())))


# ---------------------------------------------------------------------------
# Files written before the evidence reference
# ---------------------------------------------------------------------------


def test_money_spans_written_before_evidence_keep_what_maps(tmp_path):
    kept = [f for f in MONEY_SPANS_SCHEMA if f.name not in EVIDENCE_COLUMNS]
    legacy = {f.name: pa.nulls(2, f.type) for f in kept}
    legacy.update({
        "source_hash": pa.array([DOC, DOC]), "locus": pa.array(["narrative", "table_cell"]),
        "text": pa.array(["$5,000", "1,200"]),
        "text_source": pa.array(["elements", None], pa.string()),
        "start_char": pa.array([5, None], pa.int32()), "end_char": pa.array([11, None], pa.int32()),
        "page": pa.array([1, None], pa.int32()), "elem_order": pa.nulls(2, pa.int32()),
        "parent_elem_order": pa.array([None, 1], pa.int32()), "sheet": pa.nulls(2, pa.string()),
        "row": pa.array([None, 2], pa.int32()), "col": pa.array([None, 1], pa.int32()),
    })
    pq.write_table(pa.table(legacy), str(tmp_path / "batch-0001.money_spans.parquet"))

    table = read_money_spans(tmp_path / "batch-0001.parquet")
    narrative, cell_row = table.to_pylist()
    assert (narrative["char_start"], narrative["char_end"], narrative["text_layer"]) == (
        5, 11, "elements")
    assert (cell_row["elem_order"], cell_row["cell_row"], cell_row["cell_col"]) == (1, 2, 1)
    assert cell_row["text_layer"] == "cell" and cell_row["char_start"] is None
    assert set(EVIDENCE_COLUMNS) <= set(table.schema.names)
    assert "start_char" not in table.schema.names


def test_pii_and_link_files_written_before_evidence_read_with_null_evidence(tmp_path):
    pii = pa.table({
        "source_hash": [DOC], "chunk_index": pa.array([0], pa.int32()),
        "content_type": ["narrative"], "start": pa.array([8], pa.int32()),
        "end": pa.array([16], pa.int32()), "text": ["Jane Doe"], "entity_type": ["PERSON"],
        "entity_id": ["e1"], "detector": ["enrichment"], "score": pa.array([1.0], pa.float32()),
        "replacement": ["<PERSON_1>"],
    })
    pq.write_table(pii, str(tmp_path / "batch-0001.pii_spans.parquet"))
    links = pa.table({
        "source_hash": [DOC], "candidate_text": ["Acme"], "candidate_kind": ["corporate"],
        "mention_start": pa.array([0], pa.int32()), "mention_end": pa.array([4], pa.int32()),
        "entity_id": ["PR-1"], "entity_type": ["provider"], "canonical_name": ["Acme"],
        "parent_entity_id": [""], "confidence": pa.array([1.0], pa.float32()),
        "match_method": ["alias"], "matched": [True],
    })
    pq.write_table(links, str(tmp_path / "batch-0001.entity_links.parquet"))

    span = read_pii_spans(tmp_path).to_pylist()[0]
    link = read_entity_links(tmp_path).to_pylist()[0]
    for row in (span, link):
        assert all(row[c] is None for c in EVIDENCE_COLUMNS)
        assert "start" not in row and "mention_start" not in row
    assert span["text"] == "Jane Doe" and link["mention_text"] is None
