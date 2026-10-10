"""Each table's markdown is enriched as its own request text (offline fake client).

A table mention indexes that table's markdown, so narrative consumers skip it.
"""

from __future__ import annotations

import pyarrow.parquet as pq

from tests._shard import DOC, contact_shard, table_markdown
from tests.test_enrich_packing import _FakeClient, _FakeCounter
from womblex.analyse.enrich_stage import enrich_shards
from womblex.config import EnrichmentConfig, PIIConfig
from womblex.pii.pii_stage import _known_spans_by_doc
from womblex.store.enrichment_output import enrichment_meta_path_for, read_enrichment_entities
from womblex.store.output import write_chunks

NARRATIVE = "Contact Jane Doe about the $5,000 grant.\n\nSigned by the delegate."


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

    assert sorted(t for call in client.calls for t in call) == sorted([NARRATIVE, markdown])
    assert result.narrative_tokens == len(NARRATIVE.split())
    assert result.table_tokens == len(markdown.split())

    rows = read_enrichment_entities(base).to_pylist()
    narrative = [r for r in rows if r["text_layer"] == "elements"]
    tables = [r for r in rows if r["text_layer"] == "table_markdown"]
    assert narrative and all(r["entity_id"] == "p1" for r in narrative)
    assert [(r["entity_id"], r["elem_order"], r["chunk_index"]) for r in tables] == [("t0:p1", 1, -1)]
    assert pq.read_table(str(enrichment_meta_path_for(base))).to_pylist()[0]["table_count"] == 1


def test_include_tables_off_sends_only_the_narrative(tmp_path):
    base = contact_shard(tmp_path)
    client = _enrich(tmp_path, include_tables=False)

    assert [t for call in client.calls for t in call] == [NARRATIVE]
    assert {r["text_layer"] for r in read_enrichment_entities(base).to_pylist()} == {"elements"}
    assert pq.read_table(str(enrichment_meta_path_for(base))).to_pylist()[0]["table_count"] is None


def test_narrative_consumers_ignore_table_mentions(tmp_path):
    base = contact_shard(tmp_path)
    _enrich(tmp_path)

    known = _known_spans_by_doc(base, {"natural"}, set(PIIConfig().entities))

    assert {e for _s, _e, _t, e in known[DOC]} == {"p1"}


def test_graph_refresh_leaves_table_mentions_alone(tmp_path):
    from womblex.analyse.graph_refresh import refresh_graph_edges

    base = contact_shard(tmp_path)
    markdown = table_markdown(base)
    _enrich(tmp_path)
    common = {"source_hash": DOC, "has_redaction": False, "page_start": 1, "page_end": 1}
    write_chunks([
        {**common, "chunk_index": 0, "text": NARRATIVE, "start_char": 0,
         "end_char": len(NARRATIVE), "content_type": "narrative", "elem_order": None},
        {**common, "chunk_index": 1, "text": markdown, "start_char": 0,
         "end_char": len(markdown), "content_type": "table", "elem_order": 1},
    ], base)

    refresh_graph_edges(tmp_path)

    rows = read_enrichment_entities(base).to_pylist()
    assert [r["chunk_index"] for r in rows if r["text_layer"] == "table_markdown"] == [-1]
    assert [r["chunk_index"] for r in rows if r["text_layer"] == "elements"] == [0]


def test_a_batch_enriched_without_tables_is_re_enriched_on_resume(tmp_path):
    from womblex.store.checkpoint import CheckpointManager

    contact_shard(tmp_path)
    ckpt = CheckpointManager(tmp_path / ".enrich-ckpt", "t_enrich")
    ckpt.load()
    enrich_shards(tmp_path, EnrichmentConfig(include_tables=False), client=_FakeClient(),
                  token_counter=_FakeCounter(), checkpoint_mgr=ckpt)

    # Same as a file from an older release: no footer says tables were sent.
    redo = _FakeClient()
    cfg = EnrichmentConfig()
    enrich_shards(tmp_path, cfg, client=redo, token_counter=_FakeCounter(), checkpoint_mgr=ckpt)
    assert redo.calls, "tables were never sent for this batch, so it must run again"

    again = _FakeClient()
    enrich_shards(tmp_path, cfg, client=again, token_counter=_FakeCounter(), checkpoint_mgr=ckpt)
    assert again.calls == []
