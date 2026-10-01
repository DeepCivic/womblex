"""The `/v1` API: auth, ownership scoping, run reads, metrics."""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Self

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from womblex.api import app as api_app
from womblex.api.app import create_api_app, run_state
from womblex.api.auth import hash_token, parse_registry
from womblex.cloud import queue as queue_mod
from womblex.cloud.queue import RunSummary, Throughput, WorkerState
from womblex.store.contract import CONTRACT_VERSION_KEY, SENSITIVITY_KEY

RUNS = {
    "run-a": RunSummary("run-a", "alice", {"done": 2}, "2026-01-01", "2026-01-02"),
    "run-b": RunSummary("run-b", "bob", {"running": 1}, None, None),
    "run-cli": RunSummary("run-cli", None, {"pending": 1}, None, None),
}


class FakeQueue:
    written: ClassVar[list[tuple]] = []

    def __init__(self, dsn: str, **_kw: object) -> None:
        pass

    def ensure_schema(self) -> None:
        pass

    def enqueue(self, run_id, specs, *, owner=None):
        FakeQueue.written.append(("batches", run_id, len(specs), owner))
        return len(specs)

    def enqueue_stages(self, run_id, stages, shard_prefix, *, max_attempts=3, owner=None):
        FakeQueue.written.append(("stages", run_id, tuple(stages), owner))
        return len(stages)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def runs(self, owner=None, *, run_id=None, limit=100):
        return [
            r for r in RUNS.values()
            if (owner is None or r.owner == owner) and run_id in (None, r.run_id)
        ][:limit]

    def stats(self, run_id=None, *, owner=None):
        return {"done": 2}

    def workers(self, run_id=None, *, owner=None):
        return [WorkerState("w1", 1, None, None)]

    def throughput(self, run_id=None, *, window_seconds=3600.0, owner=None):
        return Throughput(window_seconds, 2, 0.5, None)


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setattr(api_app, "JobQueue", FakeQueue)
    monkeypatch.setattr(queue_mod, "JobQueue", FakeQueue)
    FakeQueue.written = []
    for owner in ("alice", "bob"):
        (tmp_path / "ingest" / owner / "up1").mkdir(parents=True)
        (tmp_path / "ingest" / owner / "up1" / "a.pdf").write_bytes(b"%PDF-1.4")
    registry = parse_registry({"clients": [
        {"client_id": "alice", "token_sha256": hash_token("ta"), "scopes": ["submit", "read"]},
        {"client_id": "bob", "token_sha256": hash_token("tb"), "scopes": ["read"]},
        {"client_id": "ops", "token_sha256": hash_token("to"), "scopes": ["admin"]},
    ]})
    app = create_api_app(
        store_uri=str(tmp_path / "store"), db_dsn="postgresql://x/y", registry=registry,
        ingest_uri=str(tmp_path / "ingest"),
    )
    return TestClient(app)


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_health_needs_no_token(client):
    assert client.get("/v1/health").json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, auth("wrong")])
def test_missing_or_wrong_token_is_401(client, headers):
    assert client.get("/v1/runs", headers=headers).status_code == 401


def test_listing_is_owner_scoped_and_admin_sees_all(client):
    mine = client.get("/v1/runs", headers=auth("ta")).json()["runs"]
    assert [r["run_id"] for r in mine] == ["run-a"]
    everything = client.get("/v1/runs", headers=auth("to")).json()["runs"]
    assert {r["run_id"] for r in everything} == set(RUNS)


def test_another_clients_run_is_404_everywhere(client):
    for path in ("", "/manifest", "/metrics"):
        assert client.get(f"/v1/runs/run-b{path}", headers=auth("ta")).status_code == 404
    assert client.get("/v1/runs/run-cli", headers=auth("ta")).status_code == 404
    assert client.get("/v1/runs/run-cli", headers=auth("to")).status_code == 200


def test_run_detail_reports_derived_state(client):
    body = client.get("/v1/runs/run-a", headers=auth("ta")).json()
    assert body["state"] == "done" and body["total"] == 2


@pytest.mark.parametrize("counts, state", [
    ({"failed": 1, "done": 3}, "failed"),
    ({"running": 1}, "running"),
    ({"pending": 1, "done": 1}, "running"),
    ({"pending": 2}, "pending"),
    ({"done": 2}, "done"),
])
def test_run_state(counts, state):
    assert run_state(counts) == state


def test_manifest_serves_the_run_documents_table(client, monkeypatch):
    monkeypatch.setattr(
        api_app.ui_readers, "get_manifest_rows", lambda settings, run_id: [{"source_hash": "h"}],
    )
    body = client.get("/v1/runs/run-a/manifest", headers=auth("ta")).json()
    assert body == {"run_id": "run-a", "documents": [{"source_hash": "h"}]}


def test_metrics_are_json(client):
    body = client.get("/v1/runs/run-a/metrics", headers=auth("ta")).json()
    assert body["stats"] == {"done": 2}
    assert body["workers"][0]["worker_id"] == "w1"
    assert body["throughput"]["completed"] == 2


def test_submit_mints_an_owned_run_from_the_callers_prefix(client):
    resp = client.post("/v1/runs", headers=auth("ta"), json={"input_prefix": "alice/up1"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["run_id"].startswith("run-") and body["document_count"] == 1
    assert body["stages"] == []
    assert FakeQueue.written == [("batches", body["run_id"], 1, "alice")]


def test_two_submissions_get_distinct_run_ids(client):
    ids = {
        client.post("/v1/runs", headers=auth("ta"), json={"input_prefix": "alice/up1"}).json()["run_id"]
        for _ in range(2)
    }
    assert len(ids) == 2


@pytest.mark.parametrize("prefix", ["bob/up1", "alicex", ""])
def test_submit_outside_the_callers_tree_is_403(client, prefix):
    resp = client.post("/v1/runs", headers=auth("ta"), json={"input_prefix": prefix})
    assert resp.status_code == 403 and FakeQueue.written == []


def test_submit_with_an_escaping_prefix_is_400(client):
    resp = client.post("/v1/runs", headers=auth("ta"), json={"input_prefix": "alice/../bob"})
    assert resp.status_code == 400


def test_submit_needs_the_submit_scope(client):
    assert client.post("/v1/runs", headers=auth("tb"), json={"input_prefix": "bob"}).status_code == 403


def test_admin_may_name_any_prefix_and_owns_nothing(client):
    resp = client.post("/v1/runs", headers=auth("to"), json={"input_prefix": "bob/up1"})
    assert resp.status_code == 201 and FakeQueue.written[0][3] is None


def test_config_stages_are_enqueued_with_the_owner(client):
    config = {"chunking": {"enabled": True}, "money": {"enabled": False}}
    resp = client.post(
        "/v1/runs", headers=auth("ta"), json={"input_prefix": "alice/up1", "config": config},
    )
    run_id = resp.json()["run_id"]
    assert resp.json()["stages"] == ["chunk"]
    assert FakeQueue.written[1] == ("stages", run_id, ("chunk",), "alice")


@pytest.mark.parametrize("extra", [
    {"config": {"chunking": {"chunk_size": "not-a-number"}}},
    {"preset": "no-such-preset"},
])
def test_a_bad_config_writes_no_rows(client, extra):
    resp = client.post("/v1/runs", headers=auth("ta"), json={"input_prefix": "alice/up1", **extra})
    assert resp.status_code == 400 and FakeQueue.written == []


def test_preset_and_config_together_are_refused(client):
    body = {"input_prefix": "alice/up1", "preset": "DEFAULT-Isaacus", "config": {}}
    assert client.post("/v1/runs", headers=auth("ta"), json=body).status_code == 422


def test_submit_without_an_ingest_location_is_503(tmp_path, monkeypatch):
    monkeypatch.setattr(queue_mod, "JobQueue", FakeQueue)
    app = create_api_app(store_uri=str(tmp_path), db_dsn="postgresql://x/y", registry=None)
    resp = TestClient(app).post("/v1/runs", json={"input_prefix": "x"})
    assert resp.status_code == 503


def test_submit_with_the_queue_down_is_503(client, monkeypatch):
    import psycopg

    def refuse(*_a, **_kw):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(queue_mod, "JobQueue", refuse)
    resp = client.post("/v1/runs", headers=auth("ta"), json={"input_prefix": "alice/up1"})
    assert resp.status_code == 503 and resp.json()["detail"] == "job queue unreachable"


def test_files_list_keys_with_their_contract_footer(client, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    docs = tmp_path / "store" / "runs" / "run-a" / "documents"
    docs.mkdir(parents=True)
    table = pa.table({"x": [1, 2]}).replace_schema_metadata(
        {CONTRACT_VERSION_KEY: "1.0", SENSITIVITY_KEY: "raw"},
    )
    pq.write_table(table, docs / "batch-0001.elements.parquet")
    (docs.parent / "egress_manifest.json").write_text("{}")
    body = client.get("/v1/runs/run-a/files", headers=auth("ta")).json()
    assert body["files"] == [
        {"key": "runs/run-a/documents/batch-0001.elements.parquet", "rows": 2,
         "contract_version": "1.0", "sensitivity": "raw"},
        {"key": "runs/run-a/egress_manifest.json", "rows": None,
         "contract_version": None, "sensitivity": None},
    ]
    assert client.get("/v1/runs/run-b/files", headers=auth("ta")).status_code == 404


def _surface(spec: dict) -> dict:
    """Each operation's success status + response model, and each model's fields."""
    ops = {}
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            code, resp = next(iter(op["responses"].items()))
            ref = resp.get("content", {}).get("application/json", {}).get("schema", {})
            ops[f"{method.upper()} {path}"] = f"{code} {ref.get('$ref', '').rsplit('/', 1)[-1]}"
    models = {
        name: sorted(schema.get("properties", {}))
        for name, schema in spec["components"]["schemas"].items()
        if name not in {"HTTPValidationError", "ValidationError"}
    }
    return {"operations": ops, "models": models}


def test_openapi_surface_is_pinned(client):
    """A change to the public surface fails here, so it is visible in review."""
    assert _surface(client.get("/openapi.json").json()) == {
        "operations": {
            "GET /v1/health": "200 Health",
            "GET /v1/ready": "200 Ready",
            "GET /v1/runs": "200 RunList",
            "POST /v1/runs": "201 RunSubmitted",
            "GET /v1/runs/{run_id}": "200 RunStatus",
            "GET /v1/runs/{run_id}/files": "200 RunFiles",
            "GET /v1/runs/{run_id}/manifest": "200 RunManifest",
            "GET /v1/runs/{run_id}/metrics": "200 RunMetrics",
        },
        "models": {
            "Health": ["status"],
            "Ready": ["queue", "ready", "store"],
            "RunFile": ["contract_version", "key", "rows", "sensitivity"],
            "RunFiles": ["files", "run_id"],
            "RunList": ["runs"],
            "RunManifest": ["documents", "run_id"],
            "RunMetrics": ["run_id", "stats", "throughput", "workers"],
            "RunRequest": ["batch_size", "config", "input_prefix", "preset"],
            "RunStatus": ["counts", "created_at", "run_id", "state", "total", "updated_at"],
            "RunSubmitted": ["batch_count", "document_count", "run_id", "stages"],
        },
    }
