"""A document the enricher was not sent is a row, not a gap (SMS uplift D2b)."""

from __future__ import annotations

from pathlib import Path

from tests._shard import contact_shard
from womblex.analyse.enrich_stage import enrich_shards
from womblex.config import EnrichmentConfig
from womblex.store.enrichment_output import (
    ENRICHMENT_META_SCHEMA,
    STATUS_SKIPPED_SHORT,
    enrichment_meta_path_for,
    read_enrichment_meta,
    write_enrichment_meta_shard,
)


def test_a_short_document_is_a_skipped_short_row_and_never_sent(tmp_path: Path):
    base = contact_shard(tmp_path)
    sent: list[object] = []

    class _Client:  # a request would be a paid call: there must be none
        def __getattr__(self, name: str) -> object:
            sent.append(name)
            raise AssertionError(f"the enricher was called ({name}) for a document under the minimum")

    result = enrich_shards(tmp_path, EnrichmentConfig(skip_short_documents=10_000, include_tables=False), client=_Client())

    assert result.docs_enriched == 0 and not sent
    rows = read_enrichment_meta(base).to_pylist()
    assert [(r["source_hash"], r["status"]) for r in rows] == [("doc1", STATUS_SKIPPED_SHORT)]
    assert rows[0]["segment_count"] == 0 and rows[0]["doc_type_enriched"] == ""
    assert enrichment_meta_path_for(base).exists()


def test_a_skipped_narrative_keeps_its_table_count(tmp_path: Path):
    base = tmp_path / "batch-0001.parquet"

    write_enrichment_meta_shard([], base, table_counts={"doc1": 2, "doc2": 1}, skipped=["doc1"])

    rows = {r["source_hash"]: r for r in read_enrichment_meta(base).to_pylist()}
    assert (rows["doc1"]["status"], rows["doc1"]["table_count"]) == (STATUS_SKIPPED_SHORT, 2)
    assert (rows["doc2"]["status"], rows["doc2"]["table_count"]) == ("enriched", 1)


def test_a_file_written_before_the_column_reads_status_as_null(tmp_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    base = tmp_path / "batch-0001.parquet"
    old = ENRICHMENT_META_SCHEMA.remove(ENRICHMENT_META_SCHEMA.get_field_index("status"))
    row = {f.name: 0 for f in old} | {"source_hash": "doc1", "doc_type_enriched": "", "jurisdiction": "",
                                      "title": "", "table_count": None}
    pq.write_table(pa.Table.from_pylist([row], schema=old), str(enrichment_meta_path_for(base)))

    assert read_enrichment_meta(base).to_pylist()[0]["status"] is None
