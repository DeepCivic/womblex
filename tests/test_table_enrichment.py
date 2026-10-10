"""Table chunks get PII candidates of their own, and links/PII carry evidence.

A fake Isaacus client (offline) records what the enrich stage sends. Each
table's markdown is a request text of its own, so narrative offsets do not
move; the PII stage maps a table's mentions onto that table's chunks and
locates the masked span with an evidence reference into the table markdown.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests._shard import DOC, contact_shard, table_markdown
from tests.test_enrich_packing import _FakeClient, _FakeCounter
from womblex.analyse.enrich_stage import enrich_shards
from womblex.config import EnrichmentConfig, LinkingConfig, PIIConfig, ReferenceConfig
from womblex.link.stage import link_shards
from womblex.pii.pii_stage import pii_shards
from womblex.process.evidence import verify_shards
from womblex.store.enrichment_output import (
    ENTITY_SCHEMA,
    enrichment_entities_path_for,
    enrichment_meta_path_for,
    read_enrichment_entities,
)
from womblex.store.entity_links_output import read_entity_links
from womblex.store.output import write_chunks
from womblex.store.pii_output import read_clean_text, read_pii_spans

NARRATIVE = "Contact Jane Doe about the $5,000 grant.\n\nSigned by the delegate."


def _chunks(base, markdown: str) -> None:
    common = {"source_hash": DOC, "has_redaction": False, "page_start": 1, "page_end": 1}
    write_chunks([
        {**common, "chunk_index": 0, "text": NARRATIVE, "start_char": 0,
         "end_char": len(NARRATIVE), "content_type": "narrative", "elem_order": None},
        {**common, "chunk_index": 1, "text": markdown, "start_char": 0,
         "end_char": len(markdown), "content_type": "table", "elem_order": 1},
    ], base)


def _enrich(d, **config) -> _FakeClient:
    client = _FakeClient()
    enrich_shards(d, EnrichmentConfig(**config), client=client, token_counter=_FakeCounter())
    return client


def test_tables_are_sent_separately_from_the_narrative(tmp_path):
    base = contact_shard(tmp_path)
    markdown = table_markdown(base)

    client = _FakeClient()
    result = enrich_shards(
        tmp_path, EnrichmentConfig(), client=client, token_counter=_FakeCounter())

    sent = [t for call in client.calls for t in call]
    assert sorted(sent) == sorted([NARRATIVE, markdown])
    assert result.narrative_tokens == len(NARRATIVE.split())
    assert result.table_tokens == len(markdown.split())

    rows = read_enrichment_entities(base).to_pylist()
    narrative = [r for r in rows if r["text_layer"] == "elements"]
    tables = [r for r in rows if r["text_layer"] == "table_markdown"]
    assert narrative and tables
    assert all(r["entity_id"] == "p1" for r in narrative)
    assert [(r["entity_id"], r["elem_order"], r["chunk_index"]) for r in tables] == [("t0:p1", 1, -1)]

    meta = pq.read_table(str(enrichment_meta_path_for(base))).to_pylist()
    assert meta[0]["table_count"] == 1


def test_include_tables_off_sends_only_the_narrative(tmp_path):
    base = contact_shard(tmp_path)
    client = _enrich(tmp_path, include_tables=False)

    assert [t for call in client.calls for t in call] == [NARRATIVE]
    assert {r["text_layer"] for r in read_enrichment_entities(base).to_pylist()} == {"elements"}
    assert pq.read_table(str(enrichment_meta_path_for(base))).to_pylist()[0]["table_count"] is None


def test_a_table_chunk_is_masked_from_its_tables_mentions(tmp_path):
    base = contact_shard(tmp_path)
    markdown = table_markdown(base)
    _enrich(tmp_path)
    _chunks(base, markdown)

    pii_shards(tmp_path, PIIConfig(use_regex_backstop=False))

    spans = {s["content_type"]: s for s in read_pii_spans(tmp_path).to_pylist()}
    table_span = spans["table"]
    assert table_span["text_layer"] == "table_markdown" and table_span["elem_order"] == 1
    assert markdown[table_span["char_start"]:table_span["char_end"]] == table_span["text"]
    assert spans["narrative"]["text_layer"] == "elements"

    clean = {r["content_type"]: r for r in read_clean_text(tmp_path).to_pylist()}
    assert clean["table"]["mask_status"] == "masked"
    assert clean["table"]["text"].startswith("<PERSON_")
    assert verify_shards(tmp_path).ok


def test_a_table_chunk_is_uncovered_when_its_tables_were_not_enriched(tmp_path):
    base = contact_shard(tmp_path)
    _enrich(tmp_path, include_tables=False)
    _chunks(base, table_markdown(base))

    pii_shards(tmp_path, PIIConfig(use_regex_backstop=False))

    status = {r["content_type"]: r["mask_status"] for r in read_clean_text(tmp_path).to_pylist()}
    assert status == {"narrative": "masked", "table": "not_masked"}


def test_an_enriched_table_with_no_pii_reads_no_entity(tmp_path):
    base = contact_shard(tmp_path)
    _enrich(tmp_path)
    _chunks(base, table_markdown(base))
    # Drop the table's mentions but keep the meta row that says it was enriched.
    path = enrichment_entities_path_for(base)
    kept = [r for r in read_enrichment_entities(base).to_pylist() if r["text_layer"] != "table_markdown"]
    pq.write_table(pa.Table.from_pylist(kept, schema=ENTITY_SCHEMA), str(path))

    pii_shards(tmp_path, PIIConfig(use_regex_backstop=False))

    status = {r["content_type"]: r["mask_status"] for r in read_clean_text(tmp_path).to_pylist()}
    assert status["table"] == "no_entity"


def test_graph_refresh_leaves_table_mentions_alone(tmp_path):
    from womblex.analyse.graph_refresh import refresh_graph_edges

    base = contact_shard(tmp_path)
    _enrich(tmp_path)
    _chunks(base, table_markdown(base))

    refresh_graph_edges(tmp_path)

    rows = read_enrichment_entities(base).to_pylist()
    assert [r["chunk_index"] for r in rows if r["text_layer"] == "table_markdown"] == [-1]
    assert [r["chunk_index"] for r in rows if r["text_layer"] == "elements"] == [0]


@pytest.fixture
def reference(tmp_path) -> ReferenceConfig:
    csv = tmp_path / "register.csv"
    csv.write_text("Id,Name\nPR-1,Jane Doe Holdings\n", encoding="utf-8")
    return ReferenceConfig(
        path=csv, id_col="Id", name_col="Name", entity_type="provider",
        match_fuzzy_cols=["Name"],
    )


def test_links_carry_evidence_for_narrative_and_table_mentions(tmp_path, reference):
    base = contact_shard(tmp_path)
    markdown = table_markdown(base)
    base_row = {
        "source_hash": DOC, "entity_label": "person", "entity_type": "corporate",
        "role": "other", "chunk_index": -1,
    }
    pq.write_table(pa.Table.from_pylist([
        {**base_row, "entity_id": "p1", "name": "Jane Doe", "mention_start": 8,
         "mention_end": 16, "text_layer": "elements", "elem_order": None, "sheet": None},
        {**base_row, "entity_id": "t0:p1", "name": "John Citizen",
         "mention_start": markdown.index("John Citizen"),
         "mention_end": markdown.index("John Citizen") + 12,
         "text_layer": "table_markdown", "elem_order": 1, "sheet": None},
    ], schema=ENTITY_SCHEMA), str(enrichment_entities_path_for(base)))

    link_shards(tmp_path, LinkingConfig(reference=reference))

    by_text = {r["candidate_text"]: r for r in read_entity_links(tmp_path).to_pylist()}
    assert by_text["Jane Doe"]["mention_text"] == "Jane Doe"
    assert by_text["Jane Doe"]["elem_order"] == 0 and by_text["Jane Doe"]["text_layer"] == "elements"
    assert by_text["John Citizen"]["mention_text"] == "John Citizen"
    assert by_text["John Citizen"]["text_layer"] == "table_markdown"
    assert verify_shards(tmp_path).ok


def test_a_batch_enriched_before_tables_were_sent_is_re_enriched_on_resume(tmp_path):
    from womblex.store.checkpoint import CheckpointManager

    base = contact_shard(tmp_path)
    ckpt = CheckpointManager(tmp_path / ".enrich-ckpt", "t_enrich")
    ckpt.load()
    cfg = EnrichmentConfig()
    enrich_shards(tmp_path, cfg, client=_FakeClient(), token_counter=_FakeCounter(),
                  checkpoint_mgr=ckpt)

    # Resume: nothing to do.
    again = _FakeClient()
    enrich_shards(tmp_path, cfg, client=again, token_counter=_FakeCounter(), checkpoint_mgr=ckpt)
    assert again.calls == []

    # The same sidecar as an older release wrote it: no table_count column.
    meta = enrichment_meta_path_for(base)
    pq.write_table(pq.read_table(str(meta)).drop(["table_count"]), str(meta))
    redo = _FakeClient()
    enrich_shards(tmp_path, cfg, client=redo, token_counter=_FakeCounter(), checkpoint_mgr=ckpt)
    assert redo.calls, "tables were never sent for this batch, so it must run again"
