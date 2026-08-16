"""Tests for the console read API skeleton (womblex.ui).

Runs are plain directories with real, schema-correct manifest shards (no
fixture PDFs needed). The ``api_client`` fixture parametrizes every API test
over local (``output_root``) and store-backed (``RemoteStore``'s local
fsspec backend, as test_cloud.py uses — no S3/MinIO needed) settings, so
each test exercises both read paths through one body.
"""
from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from womblex.cli import ALL_COMMANDS
from womblex.store.enrichment_output import ENRICHMENT_ENTITIES_SUFFIX, ENTITY_SCHEMA
from womblex.store.feedback_output import write_feedback_record
from womblex.store.money_output import MONEY_SPANS_SCHEMA, MONEY_SPANS_SUFFIX
from womblex.store.output import CHUNKS_SCHEMA, CHUNKS_SUFFIX, ELEMENTS_SUFFIX, MANIFEST_SCHEMA
from womblex.store.pii_output import PII_SPANS_SCHEMA, PII_SPANS_SUFFIX
from womblex.store.quality_output import CHUNK_QUALITY_SCHEMA, CHUNK_QUALITY_SUFFIX
from womblex.ui import dashboard, readers
from womblex.ui.app import create_app
from womblex.ui.deps import UISettings, resolve_settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def _compose_service(name: str) -> dict:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    return compose["services"][name]

_OK_ROW = {
    "source_hash": "hash-a", "collection_id": "test", "doc_id": "doc-a",
    "filename": "doc-a.pdf", "ext": ".pdf", "extraction_method": "native",
    "elements_count": 3, "table_cells_count": 0, "form_fields_count": 0,
    "status": "ok", "error": "", "extracted_at_iso": "2026-08-16T00:00:00Z",
    "parser_version": "2.0",
}
# A failed document, which the Corpus Inspector's failure filter needs to see:
# the manifest carries it as a row, not as an omission.
_FAILED_ROW = {
    **_OK_ROW,
    "source_hash": "hash-b", "doc_id": "doc-b", "filename": "doc-b.pdf",
    "status": "error", "error": "boom",
}
_ROWS = [_OK_ROW, _FAILED_ROW]


def _write_manifest_shard(shard_dir: Path, rows: list[dict]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA),
        str(shard_dir / "batch-0001._manifest.parquet"),
    )
    (shard_dir / f"batch-0001{ELEMENTS_SUFFIX}").write_bytes(b"stub")


def _write_consolidated_manifest(run_dir: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA), str(run_dir / "manifest.parquet"))


class TestUISettings:
    def test_needs_exactly_one_source(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            UISettings(output_root=None, store_uri=None)
        with pytest.raises(ValueError, match="exactly one"):
            UISettings(output_root=tmp_path, store_uri="s3://bucket")

    def test_is_remote(self, tmp_path: Path) -> None:
        assert UISettings(output_root=tmp_path, store_uri=None).is_remote is False
        assert UISettings(output_root=None, store_uri="s3://bucket").is_remote is True


class TestResolveSettings:
    def test_explicit_args_win(self, tmp_path: Path) -> None:
        assert resolve_settings(tmp_path, None).output_root == tmp_path
        assert resolve_settings(None, "s3://bucket").store_uri == "s3://bucket"

    def test_env_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WOMBLEX_UI_OUTPUT_ROOT", str(tmp_path))
        assert resolve_settings(None, None).output_root == tmp_path

    def test_neither_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WOMBLEX_UI_OUTPUT_ROOT", raising=False)
        monkeypatch.delenv("WOMBLEX_STORE_URI", raising=False)
        with pytest.raises(ValueError, match="no run source"):
            resolve_settings(None, None)

    def test_both_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="only one"):
            resolve_settings(tmp_path, "s3://bucket")

    def test_feedback_dir_explicit_arg_wins(self, tmp_path: Path) -> None:
        fb = tmp_path / "elsewhere"
        settings = resolve_settings(tmp_path, None, feedback_dir=fb)
        assert settings.feedback_dir == fb

    def test_feedback_dir_env_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fb = tmp_path / "elsewhere"
        monkeypatch.setenv("WOMBLEX_UI_FEEDBACK_DIR", str(fb))
        assert resolve_settings(tmp_path, None).feedback_dir == fb

    def test_feedback_dir_defaults_to_none(self, tmp_path: Path) -> None:
        assert resolve_settings(tmp_path, None).feedback_dir is None


@pytest.fixture(params=["local", "remote"])
def api_client(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[TestClient, Path]:
    """(client, run_root) — write ``<run_root>/<run_id>/documents/`` to seed a run."""
    if request.param == "remote":
        pytest.importorskip("fsspec")
        store_root = tmp_path / "store"
        return TestClient(create_app(store_uri=str(store_root))), store_root / "runs"
    return TestClient(create_app(output_root=tmp_path)), tmp_path


@pytest.fixture(params=["local", "remote"])
def feedback_client(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[TestClient, Path, Path]:
    """(client, run_root, feedback_root) — feedback_root is where reports land.

    Separate from ``api_client`` because the two run sources put feedback in
    different places relative to ``run_root`` (docs/ui-plan.md §4): nested
    under ``output_root`` locally (a plain dir, not a ``run-*`` one, so
    retention never touches it) but a sibling *of* ``runs/`` remotely — the
    store root has no read-only mount to work around.
    """
    if request.param == "remote":
        pytest.importorskip("fsspec")
        store_root = tmp_path / "store"
        return (
            TestClient(create_app(store_uri=str(store_root))),
            store_root / "runs",
            store_root / "feedback",
        )
    return TestClient(create_app(output_root=tmp_path)), tmp_path, tmp_path / "feedback"


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


class TestFeedbackApi:
    """The report action (docs/ui-plan.md §4, merge 7): one file per report."""

    def test_writes_one_file_per_report(
        self, feedback_client: tuple[TestClient, Path, Path]
    ) -> None:
        client, run_root, feedback_root = feedback_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS[:1])
        resp = client.post(
            "/api/runs/run-a/feedback",
            json={
                "record_type": "chunk", "source_hash": "hash-a", "chunk_index": 3,
                "row": {"text": "…", "content_type": "narrative", "has_redaction": True},
                "note": "PERSON mask missed a signature block",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["run_id"] == "run-a"
        assert body["record_type"] == "chunk"
        assert body["source_hash"] == "hash-a"
        assert body["chunk_index"] == 3
        assert body["note"] == "PERSON mask missed a signature block"
        assert body["reported_at"].endswith("Z")

        files = list((feedback_root / "run-a").glob("*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text()) == body
        # Never nested inside the run directory it reports on.
        assert not (run_root / "run-a" / "feedback").exists()

    def test_two_reports_never_collide(
        self, feedback_client: tuple[TestClient, Path, Path]
    ) -> None:
        client, run_root, feedback_root = feedback_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS[:1])
        for _ in range(2):
            resp = client.post(
                "/api/runs/run-a/feedback",
                json={"record_type": "document", "source_hash": "hash-a", "row": {}, "note": "dup"},
            )
            assert resp.status_code == 201
        assert len(list((feedback_root / "run-a").glob("*.json"))) == 2

    def test_404_when_run_missing(self, feedback_client: tuple[TestClient, Path, Path]) -> None:
        client, _run_root, _feedback_root = feedback_client
        resp = client.post(
            "/api/runs/nope/feedback",
            json={"record_type": "document", "source_hash": "hash-a", "row": {}, "note": ""},
        )
        assert resp.status_code == 404

    def test_reported_by_from_header(
        self, feedback_client: tuple[TestClient, Path, Path]
    ) -> None:
        client, run_root, _feedback_root = feedback_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS[:1])
        resp = client.post(
            "/api/runs/run-a/feedback",
            headers={"X-Womblex-Reported-By": "reviewer@example.com"},
            json={"record_type": "document", "source_hash": "hash-a", "row": {}, "note": ""},
        )
        assert resp.json()["reported_by"] == "reviewer@example.com"

    def test_reported_by_defaults_to_none_without_a_header_or_env_var(
        self, feedback_client: tuple[TestClient, Path, Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("WOMBLEX_UI_REPORTED_BY", raising=False)
        client, run_root, _feedback_root = feedback_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS[:1])
        resp = client.post(
            "/api/runs/run-a/feedback",
            json={"record_type": "document", "source_hash": "hash-a", "row": {}, "note": ""},
        )
        assert resp.json()["reported_by"] is None

    def test_a_traversing_run_id_cannot_escape_the_feedback_root(self, tmp_path: Path) -> None:
        """``..`` in a run_id must not walk the write back up into ``runs/``.

        Routing already stops this over HTTP (a path param never matches
        ``/``), so this exercises ``readers.write_feedback`` directly — it is
        library API, and the sibling-of-runs invariant (docs/ui-plan.md §4)
        is the module's guarantee, not the router's.
        """
        output_root = tmp_path / "runs"
        _write_manifest_shard(output_root / "run-a" / "documents", _ROWS[:1])
        settings = UISettings(output_root=output_root, store_uri=None)
        # `output_root/../runs/run-a` resolves to a real directory, so an
        # is_dir() check alone admits it.
        assert readers.write_feedback(
            settings, "../runs/run-a", record_type="document", source_hash="hash-a",
            chunk_index=None, row={}, note="", reported_by=None,
        ) is None
        assert list(tmp_path.rglob("*.json")) == []

    def test_write_feedback_record_refuses_an_unsafe_run_id(self, tmp_path: Path) -> None:
        """The root/run_id join is where containment is enforced."""
        for run_id in ["../escape", "a/b", "..", ".", "", "/abs"]:
            with pytest.raises(ValueError, match="unsafe run_id"):
                write_feedback_record(tmp_path, run_id, {"run_id": run_id})
        assert list(tmp_path.rglob("*.json")) == []

    def test_feedback_dir_override_is_honoured_in_local_mode(self, tmp_path: Path) -> None:
        """A deployment that mounts output_root read-only needs this escape hatch."""
        output_root = tmp_path / "runs"
        feedback_dir = tmp_path / "elsewhere"
        _write_manifest_shard(output_root / "run-a" / "documents", _ROWS[:1])
        client = TestClient(create_app(output_root=output_root, feedback_dir=feedback_dir))
        resp = client.post(
            "/api/runs/run-a/feedback",
            json={"record_type": "document", "source_hash": "hash-a", "row": {}, "note": ""},
        )
        assert resp.status_code == 201
        assert len(list((feedback_dir / "run-a").glob("*.json"))) == 1
        assert not (output_root / "feedback").exists()


_CHUNK_ROW = {
    "source_hash": "hash-a", "chunk_index": 1, "text": "second chunk",
    "start_char": 20, "end_char": 32, "content_type": "narrative",
    "has_redaction": False, "page_start": 1, "page_end": 1, "elem_order": 3,
}
_CHUNK_ROW_0 = {**_CHUNK_ROW, "chunk_index": 0, "text": "first chunk", "start_char": 0, "end_char": 19}
_OTHER_DOC_CHUNK = {**_CHUNK_ROW, "source_hash": "hash-b"}

_ENTITY_ROW = {
    "document_id": "hash-a", "entity_id": "PR-1", "entity_label": "person",
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
        assert "document_id" not in body["entities"][0]

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


def _write_checkpoint(run_dir: Path, stage: str, **counters: int) -> None:
    """Write a stage checkpoint where the pipeline itself writes one.

    Goes through ``CheckpointManager`` rather than hand-rolling the JSON so
    the dashboard is tested against the real writer's filename and shape.
    """
    from womblex.store.checkpoint import CheckpointManager

    dirname = dashboard.CHECKPOINT_DIRNAMES[stage]
    mgr = CheckpointManager(run_dir / dirname, f"testset_{stage}")
    mgr.load()
    mgr.update(
        [f"doc-{i}" for i in range(counters.get("processed", 1))],
        succeeded=counters.get("succeeded", 1),
        failed=counters.get("failed", 0),
        batch_num=counters.get("batch_num", 1),
    )


class TestDashboardApi:
    """The Dashboard reads the queue when there is one and the run's own
    per-stage checkpoints always (docs/ui-plan.md merge 8). These exercise
    the queue-less shape — a plain local deployment — and the checkpoint
    half, which is the same in both.
    """

    def test_checkpoint_dirnames_come_from_the_contracts(self) -> None:
        """The map is derived, not re-typed: a renamed dir cannot drift."""
        from womblex.cloud.stage_contracts import STAGE_CONTRACTS

        assert dashboard.CHECKPOINT_DIRNAMES["chunk"] == ".chunk-checkpoint"
        assert "quality" not in dashboard.CHECKPOINT_DIRNAMES  # whole-run scope, no checkpoint
        for stage, dirname in dashboard.CHECKPOINT_DIRNAMES.items():
            assert STAGE_CONTRACTS[stage].checkpoint_dirname == dirname

    def test_no_queue_is_not_an_error(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        body = client.get("/api/dashboard").json()
        assert body["queue"] is None
        assert body["queue_error"] is None  # absent by configuration, not by failure
        assert body["stages"] == []

    def test_unreachable_queue_reports_rather_than_raises(self, tmp_path: Path) -> None:
        """A dead DSN must not take the checkpoint half of the screen down with it."""
        client = TestClient(create_app(output_root=tmp_path, db_dsn="postgresql://nope:1/nope"))
        _write_checkpoint(tmp_path / "run-a", "chunk", processed=2)
        body = client.get("/api/dashboard", params={"run_id": "run-a"}).json()
        assert body["queue"] is None
        assert body["queue_error"]
        assert [s["stage"] for s in body["stages"]] == ["chunk"]

    def test_stage_progress_from_checkpoints(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        client, run_root = api_client
        run_dir = run_root / "run-a"
        _write_manifest_shard(run_dir / "documents", _ROWS)
        _write_checkpoint(run_dir, "chunk", processed=4, succeeded=3, failed=1, batch_num=2)
        _write_checkpoint(run_dir, "pii", processed=1)
        body = client.get("/api/dashboard", params={"run_id": "run-a"}).json()
        by_stage = {s["stage"]: s for s in body["stages"]}
        assert set(by_stage) == {"chunk", "pii"}
        assert by_stage["chunk"]["processed"] == 4
        assert by_stage["chunk"]["succeeded"] == 3
        assert by_stage["chunk"]["failed"] == 1
        assert by_stage["chunk"]["last_batch"] == 2
        assert by_stage["chunk"]["name"] == "testset_chunk"

    def test_stages_empty_without_a_run_id(self, api_client: tuple[TestClient, Path]) -> None:
        """Checkpoints are per-run artefacts, so there is no cross-run answer."""
        client, run_root = api_client
        _write_checkpoint(run_root / "run-a", "chunk", processed=2)
        assert client.get("/api/dashboard").json()["stages"] == []

    def test_missing_run_is_empty_not_404(self, api_client: tuple[TestClient, Path]) -> None:
        """An enqueued run whose first batch hasn't landed has no directory yet."""
        client, _ = api_client
        resp = client.get("/api/dashboard", params={"run_id": "run-nope"})
        assert resp.status_code == 200
        assert resp.json()["stages"] == []

    def test_rejects_a_non_positive_window(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/dashboard", params={"stale_after": 0}).status_code == 422

    def test_unreadable_checkpoint_is_skipped(self, tmp_path: Path) -> None:
        """One corrupt checkpoint narrows the answer; it doesn't blank the screen."""
        from womblex.store.checkpoint import read_checkpoints

        ckpt_dir = tmp_path / ".chunk-checkpoint"
        ckpt_dir.mkdir()
        (ckpt_dir / "bad_checkpoint.json").write_text("{not json")
        (ckpt_dir / "good_checkpoint.json").write_text(
            json.dumps({"total_processed": 3, "started_at": "", "updated_at": ""})
        )
        progress = read_checkpoints(ckpt_dir)
        assert [p.name for p in progress] == ["good"]
        # Unusable timestamps mean no rate to report, rather than a fabricated one.
        assert progress[0].documents_per_minute is None

    def test_documents_per_minute_spans_the_checkpoint(self, tmp_path: Path) -> None:
        from womblex.store.checkpoint import read_checkpoints

        ckpt_dir = tmp_path / ".chunk-checkpoint"
        ckpt_dir.mkdir()
        (ckpt_dir / "run_checkpoint.json").write_text(
            json.dumps({
                "total_processed": 120,
                "started_at": "2026-08-16T00:00:00+00:00",
                "updated_at": "2026-08-16T00:02:00+00:00",
            })
        )
        assert read_checkpoints(ckpt_dir)[0].documents_per_minute == 60.0

    def test_run_id_cannot_escape_the_run_root(self, tmp_path: Path) -> None:
        """`run_id` is a query param, so nothing upstream rejects `../`.

        The remote branch is contained by its `list_dirs` check; the local
        join has to contain itself, or the two deployments disagree about
        what is reachable.
        """
        outside = tmp_path / "outside"
        _write_checkpoint(outside, "chunk", processed=9)
        run_root = tmp_path / "runs"
        run_root.mkdir()  # must exist, or the OS cannot resolve `..` through it
        client = TestClient(create_app(output_root=run_root))
        body = client.get("/api/dashboard", params={"run_id": "../outside"}).json()
        assert body["stages"] == []

    def test_malformed_checkpoints_are_skipped_not_raised(self, tmp_path: Path) -> None:
        """Well-formed JSON can still be unusable; the reader must survive it."""
        from womblex.store.checkpoint import read_checkpoints

        ckpt_dir = tmp_path / ".chunk-checkpoint"
        ckpt_dir.mkdir()
        # A null timestamp reaches `datetime.fromisoformat(None)`; a top-level
        # list reaches `.get` on a list. Both parse as JSON, neither is a
        # checkpoint, and neither may escape as a 500.
        (ckpt_dir / "null_checkpoint.json").write_text(
            json.dumps({"total_processed": 1, "started_at": None, "updated_at": None})
        )
        (ckpt_dir / "list_checkpoint.json").write_text("[1, 2, 3]")
        (ckpt_dir / "ok_checkpoint.json").write_text(
            json.dumps({"total_processed": 1, "started_at": "", "updated_at": ""})
        )
        progress = {p.name: p for p in read_checkpoints(ckpt_dir)}
        # The list is not a checkpoint at all, so it is dropped. The null
        # timestamps are: its counters are real progress and are kept — only
        # the rate they cannot support is withheld.
        assert set(progress) == {"null", "ok"}
        assert progress["null"].processed == 1
        assert progress["null"].documents_per_minute is None

    def test_no_rate_from_a_single_write(self, tmp_path: Path) -> None:
        """A just-started run's two timestamps are one write, not an interval."""
        from womblex.store.checkpoint import read_checkpoints

        ckpt_dir = tmp_path / ".chunk-checkpoint"
        ckpt_dir.mkdir()
        (ckpt_dir / "run_checkpoint.json").write_text(
            json.dumps({
                "total_processed": 3,
                "started_at": "2026-08-16T00:00:00.000100+00:00",
                "updated_at": "2026-08-16T00:00:00.000200+00:00",
            })
        )
        # Dividing by a 0.1ms span would report ~1.8M documents/minute.
        assert read_checkpoints(ckpt_dir)[0].documents_per_minute is None


_MINIMAL_CONFIG = {
    "dataset": {"name": "t"},
    "paths": {"input_root": ".", "output_root": ".", "checkpoint_dir": "."},
}
_NO_DATASET = {"paths": _MINIMAL_CONFIG["paths"]}


class TestComposerApi:
    """The Pipeline Composer reads no run — `config.py` and
    `cloud/stage_contracts.py` are static (docs/ui-plan.md merge 9), so
    `api_client`'s local/remote parametrisation is irrelevant here; a bare
    `TestClient` is enough.
    """

    @pytest.fixture
    def client(self, tmp_path: Path) -> TestClient:
        return TestClient(create_app(output_root=tmp_path))

    def test_graph_nodes_cover_extract_plus_every_stage(self, client: TestClient) -> None:
        from womblex.cloud.stage_contracts import STAGE_NAMES

        body = client.get("/api/composer/graph").json()
        assert [n["id"] for n in body["nodes"]] == ["extract", *STAGE_NAMES]

    def test_graph_edges_are_one_per_pair_and_resolve_their_producers(
        self, client: TestClient
    ) -> None:
        """chunk reading three extraction sidecars is one dependency, not three —
        otherwise every renderer dedupes to avoid parallel arrows."""
        body = client.get("/api/composer/graph").json()
        pairs = [(e["from"], e["to"]) for e in body["edges"]]
        assert len(pairs) == len(set(pairs))
        edges = {(e["from"], e["to"]): e["suffixes"] for e in body["edges"]}
        assert edges[("extract", "chunk")] == [
            ".elements.parquet", ".table_cells.parquet", "._manifest.parquet",
        ]
        # embed reads chunk's own output — a stage-to-stage edge, not extract.
        assert edges[("chunk", "embed")] == [".chunks.parquet"]

    def test_graph_is_acyclic_and_every_stage_descends_from_extract(
        self, client: TestClient
    ) -> None:
        from womblex.cloud.stage_contracts import STAGE_NAMES

        body = client.get("/api/composer/graph").json()
        adjacency: dict[str, set[str]] = {}
        for edge in body["edges"]:
            adjacency.setdefault(edge["from"], set()).add(edge["to"])
        reached: set[str] = set()

        def walk(node: str, seen: frozenset[str]) -> None:
            for nxt in adjacency.get(node, ()):
                assert nxt not in seen, f"cycle through {nxt}"
                reached.add(nxt)
                walk(nxt, seen | {nxt})

        walk("extract", frozenset({"extract"}))
        # A stage unreachable from extract would be one the composer cannot
        # order — the plan's guardrail is precisely this reachability.
        assert reached == set(STAGE_NAMES)

    def test_schema_is_the_womblex_config_schema(self, client: TestClient) -> None:
        body = client.get("/api/composer/schema").json()
        assert set(body["properties"]) >= {"dataset", "paths", "chunking", "pii"}
        assert "dataset" in body.get("required", [])

    def test_validate_accepts_a_minimal_config(self, client: TestClient) -> None:
        resp = client.post("/api/composer/validate", json=_MINIMAL_CONFIG)
        assert resp.json() == {"valid": True, "errors": [], "unknown_keys": []}

    def test_validate_names_typos_the_schema_would_silently_drop(
        self, client: TestClient
    ) -> None:
        """Pydantic ignores unrecognised keys, so a typo validates clean and then
        vanishes from the download. The composer must say so."""
        resp = client.post("/api/composer/validate", json={
            **_MINIMAL_CONFIG,
            "chunkng": {"chunk_size": 99},            # typo'd section
            "chunking": {"chnk_size": 99},            # typo'd field in a real section
        })
        body = resp.json()
        # Still valid: the CLI loads this file too, so failing it would make the
        # composer stricter than the thing it configures.
        assert body["valid"] is True
        assert sorted(body["unknown_keys"]) == ["chunking.chnk_size", "chunkng"]

    def test_validate_does_not_mistake_free_form_dict_values_for_typos(
        self, client: TestClient
    ) -> None:
        """`normalise.substitutions` is an operator's own letterhead map — its keys
        are data, not config fields, and must not be walked into."""
        resp = client.post("/api/composer/validate", json={
            **_MINIMAL_CONFIG,
            "normalise": {"substitutions": {"Depatment": "Department", "AB C": "ABC"}},
        })
        assert resp.json()["unknown_keys"] == []

    def test_validate_walks_through_an_optional_nested_model(
        self, client: TestClient
    ) -> None:
        """`linking.reference` is `ReferenceConfig | None` — a union, still walkable."""
        resp = client.post("/api/composer/validate", json={
            **_MINIMAL_CONFIG,
            "linking": {"reference": {"path": "r.csv", "nonsense_key": 1}},
        })
        assert resp.json()["unknown_keys"] == ["linking.reference.nonsense_key"]

    def test_validate_reports_unknown_keys_on_an_invalid_config_too(
        self, client: TestClient
    ) -> None:
        resp = client.post("/api/composer/validate", json={"chunkng": {}})
        body = resp.json()
        assert body["valid"] is False
        assert body["unknown_keys"] == ["chunkng"]

    def test_validate_reports_pydantic_errors_for_a_missing_field(self, client: TestClient) -> None:
        resp = client.post("/api/composer/validate", json=_NO_DATASET)
        body = resp.json()
        assert body["valid"] is False
        assert any(err["loc"] == ["dataset"] for err in body["errors"])

    def test_validate_touches_no_filesystem(self, client: TestClient) -> None:
        """Paths are validated as strings, never resolved — otherwise an unauthenticated
        endpoint (plan §6) would answer "does this file exist?" for any path."""
        resp = client.post("/api/composer/validate", json={
            "dataset": {"name": "t"},
            "paths": {"input_root": "/nonexistent/xyz", "output_root": "/root/.ssh",
                      "checkpoint_dir": "/etc/shadow"},
        })
        assert resp.json()["valid"] is True

    def test_yaml_download_roundtrips_a_valid_config(self, client: TestClient) -> None:
        resp = client.post("/api/composer/yaml", json=_MINIMAL_CONFIG)
        assert resp.status_code == 200
        assert resp.headers["content-disposition"] == "attachment; filename=womblex.yaml"
        parsed = yaml.safe_load(resp.text)
        assert parsed["dataset"]["name"] == "t"
        # Pydantic-applied defaults are in the download, not just what was posted.
        assert "chunking" in parsed
        # The download's whole purpose: `load_config` must accept it back.
        from womblex.config import WomblexConfig

        assert WomblexConfig(**parsed).dataset.name == "t"
        assert not resp.text.startswith("#")  # nothing dropped, so no warning header

    def test_yaml_download_records_the_keys_it_dropped(self, client: TestClient) -> None:
        """The warning rides on the artefact, not only the /validate response — the
        file is what gets committed and mailed around."""
        resp = client.post("/api/composer/yaml", json={
            **_MINIMAL_CONFIG, "chunkng": {"chunk_size": 99},
        })
        assert resp.status_code == 200
        assert "chunkng" in resp.text.partition("\n\n")[0]
        assert "WARNING" in resp.text
        # Comments are inert: the file still loads, and the typo is genuinely gone.
        parsed = yaml.safe_load(resp.text)
        assert "chunkng" not in parsed

    def test_yaml_download_422s_on_an_invalid_config(self, client: TestClient) -> None:
        resp = client.post("/api/composer/yaml", json=_NO_DATASET)
        assert resp.status_code == 422
        assert any(err["loc"] == ["dataset"] for err in resp.json()["detail"])


class TestSpaMount:
    """The console serves the built SPA (docs/ui-plan.md merge 4) when one is
    present alongside it, and falls back to the API-only shape when not —
    the same image serves cloud and audit-only deployments.
    """

    @pytest.fixture
    def spa_dir(self, tmp_path: Path) -> Path:
        build = tmp_path / "spa"
        (build / "_app").mkdir(parents=True)
        (build / "index.html").write_text("<html><body>console shell</body></html>")
        (build / "_app" / "app.js").write_text("console.log('spa')")
        return build

    def test_no_spa_dir_serves_api_only(self, tmp_path: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path, spa_dir=tmp_path / "nope"))
        assert client.get("/api/health").status_code == 200
        assert client.get("/").status_code == 404

    def test_serves_index_at_root(self, tmp_path: Path, spa_dir: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path, spa_dir=spa_dir))
        resp = client.get("/")
        assert resp.status_code == 200
        assert "console shell" in resp.text

    def test_serves_index_for_client_side_routes(self, tmp_path: Path, spa_dir: Path) -> None:
        """`/corpus` is a SvelteKit client route, not a file — the SPA router takes over."""
        client = TestClient(create_app(output_root=tmp_path, spa_dir=spa_dir))
        resp = client.get("/corpus")
        assert resp.status_code == 200
        assert "console shell" in resp.text

    def test_serves_real_asset_files(self, tmp_path: Path, spa_dir: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path, spa_dir=spa_dir))
        resp = client.get("/_app/app.js")
        assert resp.status_code == 200
        assert "spa" in resp.text

    def test_api_routes_win_over_the_spa_fallback(self, tmp_path: Path, spa_dir: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path, spa_dir=spa_dir))
        assert client.get("/api/runs").json() == {"runs": []}

    def test_unmatched_api_path_404s_instead_of_serving_the_shell(
        self, tmp_path: Path, spa_dir: Path
    ) -> None:
        """A wrong endpoint must not answer 200 + HTML — a JSON client would
        report that as a parse error rather than the 404 it is.
        """
        client = TestClient(create_app(output_root=tmp_path, spa_dir=spa_dir))
        resp = client.get("/api/bogus")
        assert resp.status_code == 404
        assert resp.headers["content-type"].startswith("application/json")

    def test_client_routes_still_serve_the_shell_when_api_404s(
        self, tmp_path: Path, spa_dir: Path
    ) -> None:
        """The /api guard must not catch routes that merely start with 'api'."""
        client = TestClient(create_app(output_root=tmp_path, spa_dir=spa_dir))
        assert "console shell" in client.get("/apiary").text

    def test_path_traversal_falls_back_to_index_instead_of_escaping(
        self, tmp_path: Path, spa_dir: Path
    ) -> None:
        """Unit-tested directly: an HTTP client may normalise `..` in the URL
        before it ever reaches the server, which would let this pass without
        exercising the containment check at all.
        """
        from womblex.ui.app import resolve_spa_path

        secret = tmp_path / "secret.txt"
        secret.write_text("outside the SPA root")
        resolved = resolve_spa_path(spa_dir, f"../{secret.name}")
        assert resolved == spa_dir / "index.html"


class TestSidecarImage:
    """The console container is only correct if it agrees with the CLI and the app.

    Nothing else in the suite reads the deployment files, so a renamed flag or
    a second run source would otherwise surface as a container that exits 1 in
    someone's compose stack.
    """

    def test_entrypoint_parses_against_the_real_cli(self) -> None:
        """`womblex ui`'s flags are the image's entrypoint — they must still exist."""
        match = re.search(
            r"^ENTRYPOINT (\[.*\])$", (REPO_ROOT / "Dockerfile.ui").read_text(), re.MULTILINE
        )
        assert match, "Dockerfile.ui has no JSON-form ENTRYPOINT"
        entrypoint = json.loads(match.group(1))
        assert entrypoint[0] == "womblex"

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        for cmd in ALL_COMMANDS:
            cmd.register(sub.add_parser(cmd.name))
        args = parser.parse_args(entrypoint[1:])  # SystemExit if a flag is gone
        assert args.command == "ui"
        # Loopback is the CLI default, and inside a container it would refuse
        # every request arriving via the published port.
        assert args.host == "0.0.0.0"

    def test_compose_ui_service_is_read_only_with_tmpfs(self) -> None:
        ui = _compose_service("ui")
        assert ui["build"]["dockerfile"] == "Dockerfile.ui"
        assert ui["read_only"] is True
        # A store-backed read stages the manifest through a temp dir.
        assert "/tmp" in ui["tmpfs"]

    def test_compose_ui_service_resolves_exactly_one_run_source(self) -> None:
        """Two sources make resolve_settings raise, so the container would exit 1."""
        env = _compose_service("ui")["environment"]
        assert "WOMBLEX_STORE_URI" in env
        assert "WOMBLEX_UI_OUTPUT_ROOT" not in env

    def test_frontend_builder_stage_runs_a_real_package_script(self) -> None:
        """The builder stage's `npm run build` must name a script `ui/package.json` declares."""
        dockerfile = (REPO_ROOT / "Dockerfile.ui").read_text()
        match = re.search(r"^RUN npm run (\S+)$", dockerfile, re.MULTILINE)
        assert match, "Dockerfile.ui has no `npm run <script>` build step"
        scripts = json.loads((REPO_ROOT / "ui" / "package.json").read_text())["scripts"]
        assert match.group(1) in scripts

    def test_final_stage_copies_the_builder_stages_declared_output(self) -> None:
        """`svelte.config.js`'s adapter output dir must match what the Dockerfile COPYs out."""
        svelte_config = (REPO_ROOT / "ui" / "svelte.config.js").read_text()
        assert "pages: 'build'" in svelte_config
        dockerfile = (REPO_ROOT / "Dockerfile.ui").read_text()
        assert "COPY --from=frontend-builder /app/ui/build /app/ui/build" in dockerfile


class TestFrontendCi:
    """`ui/`'s CI job (docs/ui-plan.md §6 "CI") must call scripts that still exist —
    otherwise a rename here silently stops linting or building the SPA.
    """

    def test_frontend_job_runs_declared_package_scripts(self) -> None:
        workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
        frontend = workflow["jobs"]["frontend"]
        scripts = json.loads((REPO_ROOT / "ui" / "package.json").read_text())["scripts"]
        run_steps = [
            step["run"] for step in frontend["steps"]
            if step.get("working-directory") == "ui" and "run" in step
        ]
        assert run_steps, "no step in the frontend job runs anything in ui/"
        for run in run_steps:
            match = re.match(r"npm (run )?(\S+)$", run.strip())
            assert match, f"unrecognised frontend CI step: {run!r}"
            script = match.group(2)
            if script != "ci":  # `npm ci` installs; it isn't a package.json script
                assert script in scripts, f"ui/package.json has no {script!r} script"


class TestCliUi:
    """Only the paths that return before reaching uvicorn.run()."""

    def test_output_root_and_store_are_mutually_exclusive(self, tmp_path: Path) -> None:
        from womblex.cli import main as cli_main

        with pytest.raises(SystemExit):
            cli_main(["ui", "--output-root", str(tmp_path), "--store", "s3://bucket"])

    def test_no_run_source_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from womblex.cli import main as cli_main

        monkeypatch.delenv("WOMBLEX_UI_OUTPUT_ROOT", raising=False)
        monkeypatch.delenv("WOMBLEX_STORE_URI", raising=False)
        assert cli_main(["ui"]) == 1
