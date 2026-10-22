"""Runs as the store or output root holds them (`store/run_reads.py`, `/v1/store/runs`, stage presence, audit)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from womblex.cloud.deployment import DeploymentSettings
from womblex.store import run_reads
from womblex.store.demo_corpus import demo_run_dir

pytest.importorskip("fsspec")

DEMO = "run-synthetic-demo"


@pytest.fixture(params=["local", "store"])
def source(request: pytest.FixtureRequest, tmp_path: Path) -> DeploymentSettings:
    """The vendored demo run under each layout: `<root>/<run>/` locally, `runs/<run>/` in a store."""
    if request.param == "store":
        shutil.copytree(demo_run_dir(), tmp_path / "runs" / DEMO)
        return DeploymentSettings(output_root=None, store_uri=str(tmp_path))
    shutil.copytree(demo_run_dir(), tmp_path / DEMO)
    return DeploymentSettings(output_root=tmp_path, store_uri=None)


class TestReads:
    def test_runs_are_listed_with_their_counts_and_stages(self, source: DeploymentSettings) -> None:
        (run,) = run_reads.list_run_summaries(source)
        assert run.run_id == DEMO and run.document_count == 7
        assert {"extract", "chunk", "enrich", "embed", "money"} <= set(run.stages)

    def test_stage_presence_names_the_documents_with_a_row(self, source: DeploymentSettings) -> None:
        chunked = run_reads.stage_presence(source, DEMO, "chunk")
        assert chunked and chunked == sorted(chunked)
        assert run_reads.stage_presence(source, DEMO, "link") == []
        assert run_reads.stage_presence(source, "run-absent", "chunk") is None
        with pytest.raises(KeyError):
            run_reads.stage_presence(source, DEMO, "nope")

    def test_an_unreadable_sidecar_is_skipped(self, source: DeploymentSettings, tmp_path: Path) -> None:
        docs = (tmp_path / "runs" / DEMO if source.is_remote else tmp_path / DEMO) / "documents"
        (docs / "batch-0099.chunks.parquet").write_bytes(b"not parquet")
        assert run_reads.stage_presence(source, DEMO, "chunk")

    def test_the_audit_reads_the_extraction_shards(self, source: DeploymentSettings) -> None:
        audit = run_reads.shard_audit(source, DEMO)
        assert audit is not None and audit["manifest_row_count"] == 7 and audit["shard_count"] >= 1
        assert run_reads.shard_audit(source, "run-absent") is None


class TestRoutes:
    @pytest.fixture
    def api(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from tests.test_api_app import FakeQueue
        from womblex.api import app as api_app
        from womblex.api.auth import hash_token, parse_registry

        monkeypatch.setattr(api_app, "RunBoard", FakeQueue)
        for run_id in ("run-a", DEMO):  # run-a is alice's on the queue; the demo run only the store knows
            shutil.copytree(demo_run_dir(), tmp_path / "runs" / run_id)
        registry = parse_registry({"clients": [
            {"client_id": "alice", "token_sha256": hash_token("ta"), "scopes": ["read"]},
            {"client_id": "bob", "token_sha256": hash_token("tb"), "scopes": ["read"]},
            {"client_id": "ops", "token_sha256": hash_token("to"), "scopes": ["admin"]},
        ]})
        return TestClient(api_app.create_api_app(store_uri=str(tmp_path), db_dsn="postgresql://x/y",
                                                 registry=registry))

    def test_listing_the_store_is_admin_only(self, api) -> None:  # type: ignore[no-untyped-def]
        assert api.get("/v1/store/runs", headers={"Authorization": "Bearer ta"}).status_code == 403
        runs = api.get("/v1/store/runs", headers={"Authorization": "Bearer to"}).json()["runs"]
        assert [r["run_id"] for r in runs] == [DEMO, "run-a"] and runs[0]["document_count"] == 7

    def test_per_run_reads_follow_ownership(self, api) -> None:  # type: ignore[no-untyped-def]
        alice, bob, ops = ({"Authorization": f"Bearer {t}"} for t in ("ta", "tb", "to"))
        for path in ("/stages/chunk", "/audit"):
            assert api.get(f"/v1/runs/run-a{path}", headers=alice).status_code == 200
            assert api.get(f"/v1/runs/run-a{path}", headers=bob).status_code == 404
            assert api.get(f"/v1/runs/{DEMO}{path}", headers=alice).status_code == 404
            assert api.get(f"/v1/runs/{DEMO}{path}", headers=ops).status_code == 200
            assert api.get(f"/v1/runs/run-gone{path}", headers=ops).status_code == 404
        body = api.get("/v1/runs/run-a/stages/chunk", headers=alice).json()
        assert body["stage"] == "chunk" and body["source_hashes"]
        assert api.get("/v1/runs/run-a/stages/nope", headers=alice).status_code == 400
