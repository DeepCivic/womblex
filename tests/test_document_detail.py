"""One document's chunks and their annotations (`store/document_detail.py`, `/v1/runs/{id}/documents/{hash}/detail`)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from womblex.cloud.deployment import DeploymentSettings
from womblex.store import run_reads
from womblex.store.demo_corpus import demo_run_dir
from womblex.store.document_detail import _chunk_relative_pii, document_detail
from womblex.store.enrichment_output import ENRICHMENT_ENTITIES_SUFFIX, ENTITY_SCHEMA, LEGACY_HASH_COLUMN
from womblex.store.output import CHUNKS_SCHEMA, CHUNKS_SUFFIX

pytest.importorskip("fsspec")

DEMO = "run-synthetic-demo"


def _demo(tmp_path: Path, remote: bool) -> tuple[DeploymentSettings, Path]:
    """The demo run under either layout, and its shard directory."""
    run = tmp_path / "runs" / DEMO if remote else tmp_path / DEMO
    shutil.copytree(demo_run_dir(), run)
    settings = (DeploymentSettings(output_root=None, store_uri=str(tmp_path)) if remote
                else DeploymentSettings(output_root=tmp_path, store_uri=None))
    return settings, run / "documents"


def _a_chunked_document(settings: DeploymentSettings) -> str:
    return (run_reads.stage_presence(settings, DEMO, "money") or [])[0]


@pytest.mark.parametrize("remote", [False, True])
class TestDetail:
    def test_chunks_in_order_with_their_annotations(self, tmp_path: Path, remote: bool) -> None:
        settings, _ = _demo(tmp_path, remote)
        doc = _a_chunked_document(settings)
        detail = document_detail(settings, DEMO, doc)
        assert detail is not None
        indices = [c["chunk_index"] for c in detail["chunks"]]
        assert indices and indices == sorted(indices)
        assert {frozenset(c) for c in detail["chunks"]} == {frozenset(CHUNKS_SCHEMA.names)}
        assert all(r["source_hash"] == doc for key in ("chunks", "entities", "money_spans") for r in detail[key])
        assert detail["money_spans"] and all(m["locus"] == "narrative" for m in detail["money_spans"])
        assert all(isinstance(m["value"], str) for m in detail["money_spans"] if m["value"] is not None)

    def test_an_absent_run_is_none_and_an_absent_document_empty(self, tmp_path: Path, remote: bool) -> None:
        settings, _ = _demo(tmp_path, remote)
        assert document_detail(settings, "run-absent", "h") is None
        assert document_detail(settings, DEMO, "no-such-hash") == {
            "chunks": [], "entities": [], "pii_spans": [], "money_spans": [], "quality": []}

    def test_a_corrupt_sidecar_narrows_the_answer(self, tmp_path: Path, remote: bool) -> None:
        settings, docs = _demo(tmp_path, remote)
        doc = _a_chunked_document(settings)
        before = document_detail(settings, DEMO, doc)
        (docs / f"batch-0099{CHUNKS_SUFFIX}").write_bytes(b"not parquet")
        assert document_detail(settings, DEMO, doc) == before


class TestOlderShards:
    def _local(self, tmp_path: Path) -> tuple[DeploymentSettings, Path]:
        docs = tmp_path / "run-a" / "documents"
        docs.mkdir(parents=True)
        return DeploymentSettings(output_root=tmp_path, store_uri=None), docs

    def test_a_shard_older_than_a_column_reads_it_as_null(self, tmp_path: Path) -> None:
        settings, docs = self._local(tmp_path)
        row = {"source_hash": "h", "chunk_index": 0, "text": "t", "start_char": 0, "end_char": 1,
               "content_type": "narrative", "token_count": 1, "elem_order": 3}
        old = pa.schema([f for f in CHUNKS_SCHEMA if f.name != "elem_order"])
        pq.write_table(pa.Table.from_pylist([row], schema=old), docs / f"batch-0001{CHUNKS_SUFFIX}")
        pq.write_table(pa.Table.from_pylist([{**row, "chunk_index": 1}], schema=CHUNKS_SCHEMA),
                       docs / f"batch-0002{CHUNKS_SUFFIX}")
        chunks = (document_detail(settings, "run-a", "h") or {})["chunks"]
        assert [c["elem_order"] for c in chunks] == [None, 3]

    def test_a_pre_rename_enrichment_sidecar_is_read_by_its_legacy_column(self, tmp_path: Path) -> None:
        settings, docs = self._local(tmp_path)
        legacy = pa.schema([pa.field(LEGACY_HASH_COLUMN, f.type) if f.name == "source_hash" else f
                            for f in ENTITY_SCHEMA])
        pq.write_table(pa.Table.from_pylist([{LEGACY_HASH_COLUMN: "h", "entity_id": "PR-1", "name": "Jane"}],
                                            schema=legacy), docs / f"batch-0001{ENRICHMENT_ENTITIES_SUFFIX}")
        (entity,) = (document_detail(settings, "run-a", "h") or {})["entities"]
        assert entity["source_hash"] == "h" and LEGACY_HASH_COLUMN not in entity
        assert run_reads.stage_presence(settings, "run-a", "enrich") == ["h"]


def test_a_pii_span_gets_its_chunk_offset_only_when_located_to_the_span() -> None:
    detail = {
        "chunks": [{"chunk_index": 3, "start_char": 100}],
        "pii_spans": [
            {"chunk_index": 3, "anchor_level": "span", "char_start": 104, "char_end": 112},
            {"chunk_index": 3, "anchor_level": "document", "char_start": None, "char_end": None},
        ],
    }
    _chunk_relative_pii(detail)
    assert [(s["start"], s["end"]) for s in detail["pii_spans"]] == [(4, 12), (-1, -1)]


def test_the_route_needs_read_raw_and_ownership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from tests.test_api_app import FakeQueue
    from womblex.api import app as api_app
    from womblex.api.auth import hash_token, parse_registry

    monkeypatch.setattr(api_app, "RunBoard", FakeQueue)
    shutil.copytree(demo_run_dir(), tmp_path / "runs" / "run-a")
    registry = parse_registry({"clients": [
        {"client_id": "alice", "token_sha256": hash_token("raw"), "scopes": ["read", "read_raw"]},
        {"client_id": "alice2", "token_sha256": hash_token("read"), "scopes": ["read"]},
        {"client_id": "bob", "token_sha256": hash_token("bob"), "scopes": ["read", "read_raw"]},
    ]})
    api = TestClient(api_app.create_api_app(store_uri=str(tmp_path), db_dsn="postgresql://x/y", registry=registry))
    doc = (run_reads.stage_presence(DeploymentSettings(output_root=None, store_uri=str(tmp_path)), "run-a", "money")
           or [])[0]
    path = f"/v1/runs/run-a/documents/{doc}/detail"
    body = api.get(path, headers={"Authorization": "Bearer raw"}).json()
    assert body["run_id"] == "run-a" and body["chunks"]
    assert api.get(path, headers={"Authorization": "Bearer read"}).status_code == 403
    assert api.get(path, headers={"Authorization": "Bearer bob"}).status_code == 404
