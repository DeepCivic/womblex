"""The `/v1` API: auth, ownership scoping, run reads, metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Self

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from womblex.api import app as api_app
from womblex.api.app import create_api_app, run_state
from womblex.api.auth import hash_token, parse_registry
from womblex.cloud.queue import RunSummary, Throughput, WorkerState

RUNS = {
    "run-a": RunSummary("run-a", "alice", {"done": 2}, "2026-01-01", "2026-01-02"),
    "run-b": RunSummary("run-b", "bob", {"running": 1}, None, None),
    "run-cli": RunSummary("run-cli", None, {"pending": 1}, None, None),
}


class FakeQueue:
    def __init__(self, dsn: str, **_kw: object) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def runs(self, owner=None, *, limit=100):
        return [r for r in RUNS.values() if owner is None or r.owner == owner]

    def stats(self, run_id=None, *, owner=None):
        return {"done": 2}

    def workers(self, run_id=None, *, owner=None):
        return [WorkerState("w1", 1, None, None)]

    def throughput(self, run_id=None, *, window_seconds=3600.0, owner=None):
        return Throughput(window_seconds, 2, 0.5, None)


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setattr(api_app, "JobQueue", FakeQueue)
    registry = parse_registry({"clients": [
        {"client_id": "alice", "token_sha256": hash_token("ta"), "scopes": ["submit", "read"]},
        {"client_id": "bob", "token_sha256": hash_token("tb"), "scopes": ["read"]},
        {"client_id": "ops", "token_sha256": hash_token("to"), "scopes": ["admin"]},
    ]})
    app = create_api_app(store_uri=str(tmp_path), db_dsn="postgresql://x/y", registry=registry)
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
            "GET /v1/runs/{run_id}": "200 RunStatus",
            "GET /v1/runs/{run_id}/manifest": "200 RunManifest",
            "GET /v1/runs/{run_id}/metrics": "200 RunMetrics",
        },
        "models": {
            "Health": ["status"],
            "Ready": ["queue", "ready", "store"],
            "RunList": ["runs"],
            "RunManifest": ["documents", "run_id"],
            "RunMetrics": ["run_id", "stats", "throughput", "workers"],
            "RunStatus": ["counts", "created_at", "run_id", "state", "total", "updated_at"],
        },
    }
