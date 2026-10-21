"""Graph refresh's own sidecar: written beside the chunks it linked against, read only while they match."""

from __future__ import annotations

import pyarrow.parquet as pq

from tests.test_graph_refresh import _chunk_row, _setup_shard
from womblex.analyse.graph_refresh import refresh_graph_edges
from womblex.analyse.query import load_entity_mentions, load_graph_edges
from womblex.store.enrichment_output import GRAPH_EDGE_SCHEMA, enrichment_entities_path_for, graph_edges_path_for
from womblex.store.graph_links_output import (
    CHUNKS_DIGEST_KEY,
    chunks_digest,
    graph_links_path_for,
    read_matching_links,
)
from womblex.store.output import _write_rows, write_chunks


def test_refresh_writes_links_stamped_with_the_chunks_digest(tmp_path):
    d, base, _src = _setup_shard(tmp_path)
    refresh_graph_edges(d)
    path = graph_links_path_for(base)
    rows = {r["entity_id"]: r for r in pq.read_table(str(path)).to_pylist()}
    assert (rows["loc:0"]["chunk_index"], rows["loc:0"]["chunk_indices"]) == (0, [0])
    assert (rows["loc:1"]["chunk_index"], rows["loc:1"]["chunk_indices"]) == (1, [1])
    assert pq.read_metadata(str(path)).metadata[CHUNKS_DIGEST_KEY].decode() == chunks_digest(base)
    assert read_matching_links(base) is not None


def test_a_reader_takes_the_links_while_the_chunks_match(tmp_path):
    d, base, src = _setup_shard(tmp_path)
    refresh_graph_edges(d)
    mentions = load_entity_mentions(enrichment_entities_path_for(base))  # the file itself says -1
    assert {m.entity_id: m.chunk_index for m in mentions} == {"loc:0": 0, "loc:1": 1}
    _write_rows([], graph_edges_path_for(base), GRAPH_EDGE_SCHEMA)  # what enrich left: no mention edges
    edges = [e for e in load_graph_edges(graph_edges_path_for(base)) if e.relation == "mentioned_in"]
    assert {e.target_id for e in edges} == {f"{src}:chunk:0", f"{src}:chunk:1"}  # derived from the links


def test_links_stop_matching_after_a_rechunk_and_the_reader_falls_back(tmp_path):
    d, base, src = _setup_shard(tmp_path)
    refresh_graph_edges(d)
    write_chunks([_chunk_row(src, 0, 0, 50), _chunk_row(src, 1, 50, 200)], base)  # a different chunking
    assert read_matching_links(base) is None
    mentions = {m.entity_id: m.chunk_index for m in load_entity_mentions(enrichment_entities_path_for(base))}
    assert mentions == {"loc:0": -1, "loc:1": -1}  # the earlier file's own value, not the stale link's


def test_a_run_with_no_links_sidecar_reads_as_before(tmp_path):
    _d, base, _ = _setup_shard(tmp_path)
    assert read_matching_links(base) is None
    assert [m.chunk_index for m in load_entity_mentions(enrichment_entities_path_for(base))] == [-1, -1]


def test_a_checkpointed_batch_is_linked_again_after_a_rechunk(tmp_path):
    from womblex.store.checkpoint import CheckpointManager

    d, base, src = _setup_shard(tmp_path)
    ckpt = CheckpointManager(tmp_path / ".graph-refresh-checkpoint", "t_graph_refresh")
    ckpt.load()
    assert refresh_graph_edges(d, checkpoint_mgr=ckpt).batches_written == 1
    assert refresh_graph_edges(d, checkpoint_mgr=ckpt).batches_written == 0  # resumed: links still match

    write_chunks([_chunk_row(src, 0, 0, 50), _chunk_row(src, 1, 50, 200)], base)  # a rechunk
    assert refresh_graph_edges(d, checkpoint_mgr=ckpt).batches_written == 1
    assert read_matching_links(base) is not None
    mentions = {m.entity_id: m.chunk_index for m in load_entity_mentions(enrichment_entities_path_for(base))}
    assert mentions == {"loc:0": 0, "loc:1": 1}
