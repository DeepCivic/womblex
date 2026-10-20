"""Offline tests for the graph-edge refresh stage.

No Isaacus: a shard is built with the real records ingest (for the manifest
the stage discovers batches by), then entity + chunk sidecars are written
directly with known offsets so the refresh links mentions to chunks
deterministically.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from womblex.analyse.graph_refresh import refresh_graph_edges
from womblex.ingest.records import RecordFieldMapping, ingest_records, records_source_hash
from womblex.store.enrichment_output import ENTITY_SCHEMA, enrichment_entities_path_for, graph_edges_path_for
from womblex.store.graph_links_output import graph_links_path_for
from womblex.store.output import _write_rows, write_chunks

_MAPPING = RecordFieldMapping(id_field="version_id", text_field="text")
_DOC_ID = "doc-1"
_TEXT = "word " * 60  # arbitrary; the refresh works on offsets, not content


def _src() -> str:
    return records_source_hash(_DOC_ID, _TEXT)


def _entity_row(src: str, entity_id: str, start: int, end: int) -> dict:
    return {
        "source_hash": src, "entity_id": entity_id, "entity_label": "location",
        "name": "somewhere", "entity_type": "address", "role": "",
        "mention_start": start, "mention_end": end, "chunk_index": -1,
    }


def _chunk_row(src: str, chunk_index: int, start: int, end: int) -> dict:
    return {
        "source_hash": src, "chunk_index": chunk_index, "text": "x",
        "start_char": start, "end_char": end, "content_type": "narrative",
        "has_redaction": False, "page_start": None, "page_end": None,
    }


def write_entity_rows(rows: list[dict], base: Path) -> None:
    """Write mention rows as enrichment would have, with `chunk_index` unset."""
    _write_rows(rows, enrichment_entities_path_for(base), ENTITY_SCHEMA)


def _setup_shard(tmp_path: Path) -> tuple[Path, Path, str]:
    d = tmp_path / "documents"
    ingest_records([{"version_id": _DOC_ID, "text": _TEXT}], d, _MAPPING)
    src = _src()
    base = d / "batch-0001.parquet"
    # two chunks: [0,100) and [100,200); a mention at 120-130 lands in chunk 1
    write_chunks([_chunk_row(src, 0, 0, 100), _chunk_row(src, 1, 100, 200)], base)
    write_entity_rows(
        [_entity_row(src, "loc:0", 10, 20), _entity_row(src, "loc:1", 120, 130)], base,
    )
    return d, base, src


def _links(base: Path) -> dict[str, dict]:
    return {r["entity_id"]: r for r in pq.read_table(str(graph_links_path_for(base))).to_pylist()}


def test_links_each_mention_to_the_chunks_it_overlaps(tmp_path):
    d, base, _ = _setup_shard(tmp_path)
    result = refresh_graph_edges(d)
    assert (result.batches_written, result.docs_refreshed, result.links_written) == (1, 1, 2)
    links = _links(base)
    assert (links["loc:0"]["chunk_index"], links["loc:0"]["chunk_indices"]) == (0, [0])  # 10-20
    assert (links["loc:1"]["chunk_index"], links["loc:1"]["chunk_indices"]) == (1, [1])  # 120-130


def test_a_mention_across_a_boundary_links_to_both_chunks_and_a_table_mention_to_none(tmp_path):
    d, base, src = _setup_shard(tmp_path)
    table = {**_entity_row(src, "t0:loc", 5, 9), "text_layer": "table_markdown"}
    write_entity_rows([_entity_row(src, "loc:0", 95, 105), table], base)
    refresh_graph_edges(d)
    links = _links(base)
    assert links["loc:0"]["chunk_indices"] == [0, 1] and links["loc:0"]["chunk_index"] == 0
    assert "t0:loc" not in links


def test_earlier_sidecars_are_left_as_they_were(tmp_path):
    d, base, _ = _setup_shard(tmp_path)
    before = enrichment_entities_path_for(base).read_bytes()
    refresh_graph_edges(d)
    assert enrichment_entities_path_for(base).read_bytes() == before
    assert not graph_edges_path_for(base).exists()


def test_idempotent(tmp_path):
    d, base, _ = _setup_shard(tmp_path)
    refresh_graph_edges(d)
    first = _links(base)
    refresh_graph_edges(d)
    assert _links(base) == first


def test_skips_batch_without_chunks(tmp_path):
    d = tmp_path / "documents"
    ingest_records([{"version_id": _DOC_ID, "text": _TEXT}], d, _MAPPING)
    base = d / "batch-0001.parquet"
    write_entity_rows([_entity_row(_src(), "loc:0", 10, 20)], base)
    # no chunks sidecar → nothing to link
    assert refresh_graph_edges(d).batches_written == 0
