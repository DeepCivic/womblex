"""The queue in one read and each stage's checkpoint progress (`/v1/queue`, `/v1/runs/{id}/progress`)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from womblex.cloud.deployment import DeploymentSettings
from womblex.cloud.queue_snapshot import empty_snapshot, queue_snapshot
from womblex.pipeline_order import PIPELINE_ORDER
from womblex.store.demo_corpus import demo_run_dir
from womblex.store.stage_progress import stage_progress

pytest.importorskip("dbos")


@pytest.fixture
def sqlite_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from womblex.cloud import dbos_app

    monkeypatch.setenv("WOMBLEX_DBOS_PATH", str(tmp_path / "dbos.sqlite"))
    dbos_app.ensure_schema(None)


def _enqueue(run_id: str, owner: str | None) -> None:
    from womblex.cloud.jobs import JobSpec, RunBoard

    with RunBoard(None) as board:
        board.enqueue(run_id, [JobSpec(batch_num=n, input_keys=[f"{n}.pdf"], shard_prefix=f"runs/{run_id}/documents",
                                       ingest_root="s3://b/in") for n in (1, 2)], owner=owner)


class TestQueueSnapshot:
    def test_every_view_scoped_by_run_and_owner(self, sqlite_queue: None) -> None:
        _enqueue("run-a", "alice")
        _enqueue("run-b", "bob")
        everything = queue_snapshot(None)
        assert everything["stats"] == {"pending": 4} and everything["total"] == 4 and everything["summary"] is None
        alices = queue_snapshot(None, owner="alice")
        assert {j["run_id"] for j in alices["jobs"]} == {"run-a"} and alices["total"] == 2
        one = queue_snapshot(None, run_id="run-a", job_limit=1)
        assert len(one["jobs"]) == 1 and one["summary"] and "run-a" in one["summary"]
        assert set(one["throughput"]) == set(empty_snapshot()["throughput"])

    def test_a_database_without_the_schema_is_an_empty_queue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from womblex.cloud.jobs import RunBoard

        def _missing(*_a: object, **_kw: object) -> dict:
            raise RuntimeError('relation "workflow_status" does not exist')

        monkeypatch.setattr(RunBoard, "__init__", lambda self, *_a, **_kw: None)
        monkeypatch.setattr(RunBoard, "__enter__", lambda self: self)
        monkeypatch.setattr(RunBoard, "__exit__", lambda self, *exc: None)
        monkeypatch.setattr(RunBoard, "stats", _missing)
        assert queue_snapshot("postgresql://x/y", window_seconds=60) == empty_snapshot(60)
        monkeypatch.setattr(RunBoard, "stats", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("denied")))
        with pytest.raises(RuntimeError, match="denied"):
            queue_snapshot("postgresql://x/y")


@pytest.mark.parametrize("remote", [False, True])
def test_stage_progress_reads_the_runs_checkpoints_in_pipeline_order(tmp_path: Path, remote: bool) -> None:
    pytest.importorskip("fsspec")
    run = tmp_path / "runs" / "run-a" if remote else tmp_path / "run-a"
    shutil.copytree(demo_run_dir(), run)
    settings = (DeploymentSettings(output_root=None, store_uri=str(tmp_path)) if remote
                else DeploymentSettings(output_root=tmp_path, store_uri=None))
    rows = stage_progress(settings, "run-a") or []
    stages = list(dict.fromkeys(r["stage"] for r in rows))
    assert {"chunk", "enrich", "embed", "money"} <= set(stages)
    assert stages == [s for s in PIPELINE_ORDER if s in stages]
    assert stage_progress(settings, "run-absent") is None
    assert stage_progress(settings, "../escape") is None


def test_the_routes(tmp_path: Path, sqlite_queue: None) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from womblex.api.app import create_api_app
    from womblex.api.auth import hash_token, parse_registry

    _enqueue("run-a", "alice")
    _enqueue("run-b", "bob")
    shutil.copytree(demo_run_dir(), tmp_path / "runs" / "run-a")
    registry = parse_registry({"clients": [
        {"client_id": "alice", "token_sha256": hash_token("ta"), "scopes": ["read"]},
        {"client_id": "bob", "token_sha256": hash_token("tb"), "scopes": ["read"]},
    ]})
    import os

    api = TestClient(create_api_app(store_uri=str(tmp_path), db_dsn=f"sqlite:///{os.environ['WOMBLEX_DBOS_PATH']}",
                                    registry=registry))
    alice = {"Authorization": "Bearer ta"}
    body = api.get("/v1/queue", headers=alice).json()
    assert body["total"] == 2 and {j["run_id"] for j in body["jobs"]} == {"run-a"}
    assert api.get("/v1/queue", params={"job_limit": 0}, headers=alice).status_code == 422
    progress = api.get("/v1/runs/run-a/progress", headers=alice).json()
    assert progress["run_id"] == "run-a" and progress["stages"]
    assert api.get("/v1/runs/run-a/progress", headers={"Authorization": "Bearer tb"}).status_code == 404
