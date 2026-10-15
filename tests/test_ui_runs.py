"""Console run read routes: run list, manifest, stage presence, audit, logs and chunk detail (womblex.ui).

Runs are plain directories with real, schema-correct manifest shards (no
fixture PDFs needed). The ``api_client`` fixture parametrizes every API test
over local (``output_root``) and store-backed (``RemoteStore``'s local
fsspec backend, as test_cloud.py uses — no S3/MinIO needed) settings, so
each test exercises both read paths through one body.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from tests._ui_helpers import _ROWS, _write_consolidated_manifest, _write_manifest_shard
from womblex.store.enrichment_output import (
    ENRICHMENT_ENTITIES_SUFFIX,
    ENTITY_SCHEMA,
    LEGACY_HASH_COLUMN,
)
from womblex.store.money_output import MONEY_SPANS_SCHEMA, MONEY_SPANS_SUFFIX
from womblex.store.output import CHUNKS_SCHEMA, CHUNKS_SUFFIX
from womblex.store.pii_output import PII_SPANS_SCHEMA, PII_SPANS_SUFFIX
from womblex.store.quality_output import CHUNK_QUALITY_SCHEMA, CHUNK_QUALITY_SUFFIX
from womblex.ui import readers


class TestRunsApi:
    def test_health_and_empty_list(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/runs").json() == {"runs": []}

    def test_lists_runs_newest_first(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-20260101T000000Z" / "documents", _ROWS[:1])
        _write_manifest_shard(run_root / "run-20260201T000000Z" / "documents", _ROWS)
        runs = client.get("/api/runs").json()["runs"]
        assert [r["run_id"] for r in runs] == ["run-20260201T000000Z", "run-20260101T000000Z"]
        assert runs[0]["document_count"] == 2
        assert runs[0]["stages"] == ["extract"]

    def test_get_manifest_from_shards(self, api_client: tuple[TestClient, Path]) -> None:
        """Also covers a distributed run before `finalize` has published a consolidated manifest."""
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        body = client.get("/api/runs/run-a/manifest").json()
        assert body["run_id"] == "run-a"
        error_row = next(d for d in body["documents"] if d["doc_id"] == "doc-b")
        assert error_row["status"] == "error"
        assert error_row["error"] == "boom"

    def test_get_manifest_prefers_consolidated(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        run_dir = run_root / "run-a"
        _write_manifest_shard(run_dir / "documents", _ROWS[:1])
        _write_consolidated_manifest(run_dir, _ROWS)
        resp = client.get("/api/runs/run-a/manifest")
        assert len(resp.json()["documents"]) == 2

    def test_get_manifest_404(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/runs/nope/manifest").status_code == 404

    def test_stage_presence(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        shard_dir = run_root / "run-a" / "documents"
        _write_manifest_shard(shard_dir, _ROWS)
        chunks_path = shard_dir / "batch-0001.chunks.parquet"
        pq.write_table(pa.table({"source_hash": ["hash-a"]}), str(chunks_path))
        body = client.get("/api/runs/run-a/stage-presence/chunk").json()
        assert body == {"run_id": "run-a", "stage": "chunk", "source_hashes": ["hash-a"]}
        # A stage with no sidecar in this run reports present, not missing.
        empty = client.get("/api/runs/run-a/stage-presence/embed").json()
        assert empty["source_hashes"] == []

    def test_stage_presence_reads_the_enrich_sidecar_hash_column(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        """The enrich sidecar joins on `source_hash` like every other one."""
        client, run_root = api_client
        shard_dir = run_root / "run-a" / "documents"
        _write_manifest_shard(shard_dir, _ROWS)
        _write_shard(shard_dir, ENRICHMENT_ENTITIES_SUFFIX, ENTITY_SCHEMA, [_ENTITY_ROW])
        body = client.get("/api/runs/run-a/stage-presence/enrich").json()
        assert body == {"run_id": "run-a", "stage": "enrich", "source_hashes": ["hash-a"]}

    def test_stage_presence_reads_a_pre_rename_enrich_sidecar(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        """A sidecar written before 0.6.0 carries the hash in the legacy column.

        The console reads parquet directly so the predicate pushes down, which
        bypasses the store reader's shim — without the same fallback here,
        `enrich` reports empty on a completed pre-rename run.
        """
        client, run_root = api_client
        shard_dir = run_root / "run-a" / "documents"
        _write_manifest_shard(shard_dir, _ROWS)
        _write_shard(
            shard_dir, ENRICHMENT_ENTITIES_SUFFIX, _legacy_entity_schema(),
            [_legacy_entity_row(_ENTITY_ROW)],
        )
        body = client.get("/api/runs/run-a/stage-presence/enrich").json()
        assert body == {"run_id": "run-a", "stage": "enrich", "source_hashes": ["hash-a"]}

    def test_stage_presence_skips_an_unreadable_sidecar(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        """One corrupt batch must not blank the other batches' answer."""
        client, run_root = api_client
        shard_dir = run_root / "run-a" / "documents"
        _write_manifest_shard(shard_dir, _ROWS)
        pq.write_table(
            pa.table({"source_hash": ["hash-a"]}), str(shard_dir / "batch-0001.chunks.parquet")
        )
        (shard_dir / "batch-0002.chunks.parquet").write_bytes(b"not parquet")
        body = client.get("/api/runs/run-a/stage-presence/chunk").json()
        assert body["source_hashes"] == ["hash-a"]

    def test_stage_presence_unknown_stage(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS[:1])
        assert client.get("/api/runs/run-a/stage-presence/nope").status_code == 400

    def test_stage_presence_404(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/runs/nope/stage-presence/chunk").status_code == 404

    def test_audit(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        report = client.get("/api/runs/run-a/audit").json()
        assert report["manifest_row_count"] == 0  # the stub elements file makes the batch unreadable
        assert len(report["corrupted_batches"]) == 1

    def test_audit_404(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/runs/nope/audit").status_code == 404


def _write_log(run_root: Path, run_id: str, name: str, body: str) -> None:
    logs_dir = run_root / run_id / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / name).write_text(body, encoding="utf-8")


class TestRunLogsApi:
    """Batch logs readable and downloadable from the console."""

    def test_lists_logs_newest_first(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        _write_log(run_root, "run-a", "batch-0001.log", "first")
        _write_log(run_root, "run-a", "batch-0002.log", "second")
        body = client.get("/api/runs/run-a/logs").json()
        assert body["run_id"] == "run-a"
        assert [entry["name"] for entry in body["logs"]] == [
            "batch-0002.log", "batch-0001.log",
        ]

    def test_list_404_when_run_missing(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/runs/nope/logs").status_code == 404

    def test_run_with_no_logs_lists_empty_not_404(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        """A run created before this change exists but has no `logs/` prefix; it
        lists empty (an explained empty state) rather than 404-ing."""
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-old" / "documents", _ROWS)
        resp = client.get("/api/runs/run-old/logs")
        assert resp.status_code == 200
        assert resp.json()["logs"] == []

    def test_reads_one_log_as_plain_text(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        _write_log(run_root, "run-a", "batch-0001.log", "Extraction failed: doc=bad error=boom")
        resp = client.get("/api/runs/run-a/logs/batch-0001.log")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        assert "Extraction failed: doc=bad" in resp.text
        assert "content-disposition" not in resp.headers

    def test_download_sets_content_disposition(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        _write_log(run_root, "run-a", "batch-0001.log", "log body")
        resp = client.get("/api/runs/run-a/logs/batch-0001.log?download=1")
        assert resp.status_code == 200
        assert 'attachment; filename="batch-0001.log"' in resp.headers["content-disposition"]

    def test_absent_log_404s_with_the_available_list(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        _write_log(run_root, "run-a", "batch-0001.log", "present")
        resp = client.get("/api/runs/run-a/logs/batch-9999.log")
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert [entry["name"] for entry in detail["available"]] == ["batch-0001.log"]

    @pytest.mark.parametrize(
        "bad_name",
        ["batch-1.log", "batch-0001.txt", "passwd", "batch-0001.log.bak"],
    )
    def test_a_malformed_name_404s_the_same_way_never_touching_the_store(
        self, api_client: tuple[TestClient, Path], bad_name: str
    ) -> None:
        """A rejected name and an absent name return the identical 404 + available
        list, so the endpoint cannot be used to probe outside `logs/`."""
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        _write_log(run_root, "run-a", "batch-0001.log", "present")
        resp = client.get(f"/api/runs/run-a/logs/{bad_name}")
        assert resp.status_code == 404
        assert [e["name"] for e in resp.json()["detail"]["available"]] == ["batch-0001.log"]

    def test_a_traversal_never_reaches_the_filesystem(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        # Encoded so the router passes it through as one path segment rather
        # than resolving `..` itself.
        resp = client.get("/api/runs/run-a/logs/..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code == 404

    def test_read_404_when_run_missing(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/runs/nope/logs/batch-0001.log").status_code == 404

    def test_is_safe_log_name(self) -> None:
        assert readers.is_safe_log_name("batch-0001.log")
        assert not readers.is_safe_log_name("batch-1.log")
        assert not readers.is_safe_log_name("../batch-0001.log")
        assert not readers.is_safe_log_name("batch-0001.log.bak")


_CHUNK_ROW = {
    "source_hash": "hash-a", "chunk_index": 1, "text": "second chunk",
    "start_char": 20, "end_char": 32, "content_type": "narrative",
    "has_redaction": False, "page_start": 1, "page_end": 1, "elem_order": 3,
    "token_count": 2,
}
_CHUNK_ROW_0 = {**_CHUNK_ROW, "chunk_index": 0, "text": "first chunk", "start_char": 0, "end_char": 19}
_OTHER_DOC_CHUNK = {**_CHUNK_ROW, "source_hash": "hash-b"}

_ENTITY_ROW = {
    "source_hash": "hash-a", "entity_id": "PR-1", "entity_label": "person",
    "name": "Jane Citizen", "entity_type": "natural", "role": "other",
    "mention_start": 0, "mention_end": 12, "chunk_index": 0,
}

_PII_ROW = {
    "source_hash": "hash-a", "chunk_index": 0, "content_type": "narrative",
    "start": 0, "end": 12, "text": "Jane Citizen", "entity_type": "PERSON",
    "entity_id": "PR-1", "detector": "enrichment", "score": 0.9, "replacement": "<PERSON_1>",
}

_MONEY_ROW = {
    "source_hash": "hash-a", "locus": "narrative", "text_source": "elements",
    "start_char": 5, "end_char": 10, "page": 1, "elem_order": 3, "parent_elem_order": None,
    "sheet": None, "row": None, "col": None, "text": "$100", "value": Decimal("100.0000"),
    "currency": "AUD", "currency_source": "symbol", "evidence": "p1", "modifier": None,
    "multiplier": None, "negative": False, "confidence": 0.9, "range_group": None,
    "range_role": None, "column_id": None, "context": "",
}
_MONEY_ROW_TABLE_CELL = {**_MONEY_ROW, "locus": "table_cell", "start_char": None, "end_char": None}

_QUALITY_ROW = {
    "source_hash": "hash-a", "chunk_index": 0, "content_type": "narrative",
    "char_len": 19, "alpha_frac": 0.9, "is_short": False, "boilerplate_flag": False,
    "exact_dup_id": None, "near_dup_id": None,
}


def _legacy_entity_schema() -> pa.Schema:
    """ENTITY_SCHEMA as it was before 0.6.0 — hash under the legacy column."""
    return pa.schema([
        pa.field(LEGACY_HASH_COLUMN, f.type) if f.name == "source_hash" else f
        for f in ENTITY_SCHEMA
    ])


def _legacy_entity_row(row: dict) -> dict:
    return {LEGACY_HASH_COLUMN if k == "source_hash" else k: v for k, v in row.items()}


def _write_shard(
    shard_dir: Path, suffix: str, schema: pa.Schema, rows: list[dict], *, batch: str = "batch-0001",
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), str(shard_dir / f"{batch}{suffix}"))


class TestChunkDetailApi:
    def test_404_when_run_missing(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/runs/nope/chunks/hash-a").status_code == 404

    def test_empty_when_chunk_stage_not_run(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS[:1])
        body = client.get("/api/runs/run-a/chunks/hash-a").json()
        assert body == {
            "run_id": "run-a", "source_hash": "hash-a",
            "chunks": [], "entities": [], "pii_spans": [], "money_spans": [], "quality": [],
        }

    def test_returns_chunks_ordered_and_scoped_overlays(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        client, run_root = api_client
        shard_dir = run_root / "run-a" / "documents"
        _write_manifest_shard(shard_dir, _ROWS)
        _write_shard(shard_dir, CHUNKS_SUFFIX, CHUNKS_SCHEMA, [_CHUNK_ROW, _CHUNK_ROW_0, _OTHER_DOC_CHUNK])
        _write_shard(shard_dir, ENRICHMENT_ENTITIES_SUFFIX, ENTITY_SCHEMA, [_ENTITY_ROW])
        _write_shard(shard_dir, PII_SPANS_SUFFIX, PII_SPANS_SCHEMA, [_PII_ROW])
        _write_shard(
            shard_dir, MONEY_SPANS_SUFFIX, MONEY_SPANS_SCHEMA, [_MONEY_ROW, _MONEY_ROW_TABLE_CELL],
        )
        _write_shard(shard_dir, CHUNK_QUALITY_SUFFIX, CHUNK_QUALITY_SCHEMA, [_QUALITY_ROW])

        body = client.get("/api/runs/run-a/chunks/hash-a").json()
        assert [c["chunk_index"] for c in body["chunks"]] == [0, 1]
        assert all(c["source_hash"] == "hash-a" for c in body["chunks"])

        assert len(body["entities"]) == 1
        assert body["entities"][0]["source_hash"] == "hash-a"
        assert LEGACY_HASH_COLUMN not in body["entities"][0]

        assert [p["entity_type"] for p in body["pii_spans"]] == ["PERSON"]

        # Only the narrative-locus span is returned — table_cell spans anchor
        # to table_cells.parquet, not chunk text, so they don't overlay here.
        assert len(body["money_spans"]) == 1
        assert body["money_spans"][0]["locus"] == "narrative"
        assert body["money_spans"][0]["value"] == "100.0000"

        assert len(body["quality"]) == 1
        assert body["quality"][0]["char_len"] == 19

    def test_drifted_shard_rows_keep_the_canonical_key_set(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        """A run spanning a schema bump must not hand the frontend ragged rows.

        `elem_order` post-dates parser 2.0 (`store.output._CHUNKS_BACKFILL`),
        so a long-lived run can hold batches on either side of it.
        """
        client, run_root = api_client
        shard_dir = run_root / "run-a" / "documents"
        _write_manifest_shard(shard_dir, _ROWS[:1])
        old_schema = pa.schema([f for f in CHUNKS_SCHEMA if f.name != "elem_order"])
        pre_bump = {k: v for k, v in _CHUNK_ROW_0.items() if k != "elem_order"}
        pq.write_table(
            pa.Table.from_pylist([pre_bump], schema=old_schema),
            str(shard_dir / f"batch-0001{CHUNKS_SUFFIX}"),
        )
        _write_shard(shard_dir, CHUNKS_SUFFIX, CHUNKS_SCHEMA, [_CHUNK_ROW], batch="batch-0002")

        chunks = client.get("/api/runs/run-a/chunks/hash-a").json()["chunks"]
        assert len(chunks) == 2
        assert {frozenset(c) for c in chunks} == {frozenset(CHUNKS_SCHEMA.names)}
        assert chunks[0]["elem_order"] is None  # back-filled, not absent
        assert chunks[1]["elem_order"] == 3

    def test_a_corrupt_sidecar_narrows_the_answer_rather_than_blanking_it(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        client, run_root = api_client
        shard_dir = run_root / "run-a" / "documents"
        _write_manifest_shard(shard_dir, _ROWS[:1])
        _write_shard(shard_dir, CHUNKS_SUFFIX, CHUNKS_SCHEMA, [_CHUNK_ROW_0])
        (shard_dir / f"batch-0002{CHUNKS_SUFFIX}").write_bytes(b"not parquet")
        body = client.get("/api/runs/run-a/chunks/hash-a").json()
        assert [c["chunk_index"] for c in body["chunks"]] == [0]
