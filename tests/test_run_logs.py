"""A run's batch logs (`store/run_logs.py`, `/v1/runs/{id}/logs`)."""
from __future__ import annotations

from pathlib import Path

import pytest

from womblex.cloud.deployment import DeploymentSettings
from womblex.store.run_logs import is_safe_log_name, list_run_logs, read_run_log

pytest.importorskip("fsspec")


@pytest.fixture(params=["local", "store"])
def runs(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[DeploymentSettings, Path]:
    """Settings, and the directory runs sit under for that layout."""
    if request.param == "store":
        return DeploymentSettings(output_root=None, store_uri=str(tmp_path)), tmp_path / "runs"
    return DeploymentSettings(output_root=tmp_path, store_uri=None), tmp_path


def _log(root: Path, run_id: str, name: str, text: str) -> None:
    (root / run_id / "logs").mkdir(parents=True, exist_ok=True)
    (root / run_id / "logs" / name).write_text(text)


class TestLibrary:
    def test_logs_are_listed_newest_first_and_read(self, runs: tuple[DeploymentSettings, Path]) -> None:
        settings, root = runs
        _log(root, "run-a", "batch-0001.log", "first")
        _log(root, "run-a", "batch-0002.log", "Extraction failed: doc=bad")
        _log(root, "run-a", "notes.txt", "not a log")
        listed = list_run_logs(settings, "run-a") or []
        assert [e["name"] for e in listed] == ["batch-0002.log", "batch-0001.log"]
        assert listed[0]["size"] == len("Extraction failed: doc=bad")
        assert read_run_log(settings, "run-a", "batch-0002.log") == "Extraction failed: doc=bad"

    def test_absent_runs_logs_and_unsafe_names(self, runs: tuple[DeploymentSettings, Path]) -> None:
        settings, root = runs
        (root / "run-old" / "documents").mkdir(parents=True)
        assert list_run_logs(settings, "run-old") == []
        assert list_run_logs(settings, "run-none") is None
        assert list_run_logs(settings, "..") is None
        assert read_run_log(settings, "..", "batch-0001.log") is None
        assert read_run_log(settings, "run-old", "batch-0001.log") is None
        for bad in ("batch-1.log", "batch-0001.log.bak", "../../etc/passwd"):
            assert read_run_log(settings, "run-old", bad) is None

    def test_is_safe_log_name(self) -> None:
        assert is_safe_log_name("batch-0001.log")
        assert not any(is_safe_log_name(n) for n in ("batch-1.log", "batch-0001.txt", "passwd"))


def test_the_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from tests.test_api_app import FakeQueue
    from womblex.api import app as api_app
    from womblex.api.auth import hash_token, parse_registry

    monkeypatch.setattr(api_app, "RunBoard", FakeQueue)
    _log(tmp_path / "runs", "run-a", "batch-0001.log", "log body")
    registry = parse_registry({"clients": [
        {"client_id": "alice", "token_sha256": hash_token("ta"), "scopes": ["read"]},
        {"client_id": "bob", "token_sha256": hash_token("tb"), "scopes": ["read"]},
    ]})
    api = TestClient(api_app.create_api_app(store_uri=str(tmp_path), db_dsn="postgresql://x/y", registry=registry))
    alice = {"Authorization": "Bearer ta"}
    assert [e["name"] for e in api.get("/v1/runs/run-a/logs", headers=alice).json()["logs"]] == ["batch-0001.log"]
    one = api.get("/v1/runs/run-a/logs/batch-0001.log", headers=alice)
    assert one.text == "log body" and one.headers["content-type"].startswith("text/plain")
    assert "content-disposition" not in one.headers
    dl = api.get("/v1/runs/run-a/logs/batch-0001.log?download=true", headers=alice)
    assert 'filename="batch-0001.log"' in dl.headers["content-disposition"]
    assert api.get("/v1/runs/run-a/logs/..%2F..%2Fetc%2Fpasswd", headers=alice).status_code == 404
    for name in ("batch-9999.log", "passwd", "batch-0001.log.bak"):
        missing = api.get(f"/v1/runs/run-a/logs/{name}", headers=alice)
        assert missing.status_code == 404
        assert [e["name"] for e in missing.json()["detail"]["available"]] == ["batch-0001.log"]
    assert api.get("/v1/runs/run-a/logs", headers={"Authorization": "Bearer tb"}).status_code == 404
