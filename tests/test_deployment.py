"""Deployment settings, the store reads and the readiness probes the service API shares."""
from __future__ import annotations

from pathlib import Path

import pytest

from womblex.cloud.deployment import DeploymentSettings, deployment_paths
from womblex.cloud.reachability import queue_reachable, store_reachable
from womblex.store import remote as remote_mod
from womblex.store.remote_manifest import StoreUnreachable, open_store, remote_manifest_rows


class TestDeploymentSettings:
    def test_needs_exactly_one_source(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            DeploymentSettings(output_root=None, store_uri=None)
        with pytest.raises(ValueError, match="exactly one"):
            DeploymentSettings(output_root=tmp_path, store_uri="s3://bucket")

    def test_is_remote(self, tmp_path: Path) -> None:
        assert DeploymentSettings(output_root=tmp_path, store_uri=None).is_remote is False
        assert DeploymentSettings(output_root=None, store_uri="s3://bucket").is_remote is True

    def test_ingest_and_store_must_be_disjoint(self) -> None:
        DeploymentSettings(output_root=None, store_uri="s3://womblex", ingest_uri="s3://womblex/inbox")
        with pytest.raises(ValueError, match="not disjoint"):
            DeploymentSettings(output_root=None, store_uri="s3://womblex", ingest_uri="s3://womblex")


class TestDeploymentPaths:
    def test_a_local_root_is_written_verbatim(self, tmp_path: Path) -> None:
        paths, env_vars = deployment_paths(DeploymentSettings(output_root=tmp_path, store_uri=None))
        assert paths["output_root"] == str(tmp_path) and paths["input_root"] == str(tmp_path)
        assert env_vars == []

    def test_an_object_store_becomes_a_placeholder_and_names_its_env_var(self) -> None:
        settings = DeploymentSettings(
            output_root=None, store_uri="s3://womblex", ingest_uri="s3://womblex/inbox",
        )
        paths, env_vars = deployment_paths(settings)
        assert paths == {"input_root": ".", "output_root": ".", "checkpoint_dir": "."}
        assert env_vars == ["$WOMBLEX_INGEST_URI", "$WOMBLEX_STORE_URI"]


class TestRemoteManifest:
    def test_open_store_wraps_a_backend_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(_uri: str, **_kw: object) -> object:
            raise ImportError("Install s3fs to access S3")

        monkeypatch.setattr(remote_mod.RemoteStore, "from_uri", staticmethod(_raise))
        with pytest.raises(StoreUnreachable, match="s3fs"):
            open_store("s3://redline")

    def test_an_absent_run_is_none(self, tmp_path: Path) -> None:
        pytest.importorskip("fsspec")
        (tmp_path / "runs" / "run-a").mkdir(parents=True)
        assert remote_manifest_rows(str(tmp_path), "run-b") is None

    @pytest.mark.parametrize("consolidated", [True, False])
    def test_rows_come_from_the_run_manifest_or_its_shard_manifests(
        self, tmp_path: Path, consolidated: bool,
    ) -> None:
        """Before `womblex finalize` there is no `manifest.parquet`; the shard manifests answer."""
        pytest.importorskip("fsspec")
        import shutil

        from womblex.store.demo_corpus import demo_run_dir

        run = tmp_path / "runs" / "run-a"
        shutil.copytree(demo_run_dir(), run)
        expected = {r["source_hash"] for r in remote_manifest_rows(str(tmp_path), "run-a") or []}
        if not consolidated:
            (run / "manifest.parquet").unlink()
        rows = remote_manifest_rows(str(tmp_path), "run-a") or []
        assert expected and {r["source_hash"] for r in rows} == expected


class TestReachability:
    def test_an_empty_local_store_answers(self, tmp_path: Path) -> None:
        pytest.importorskip("fsspec")
        assert store_reachable(str(tmp_path)) == (True, None)

    def test_a_store_that_cannot_be_opened_reports_why(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(_uri: str, **_kw: object) -> object:
            raise ImportError("Install s3fs to access S3")

        monkeypatch.setattr(remote_mod.RemoteStore, "from_uri", staticmethod(_raise))
        ok, error = store_reachable("s3://redline")
        assert ok is False and error is not None and "s3fs" in error

    def test_no_dsn_is_not_a_queue(self) -> None:
        assert queue_reachable(None) == (False, "no queue configured")

    def test_a_fresh_database_answers(self, tmp_path: Path) -> None:
        assert queue_reachable(f"sqlite:///{tmp_path / 'fresh.sqlite'}") == (True, None)

    @pytest.mark.parametrize(("message", "reachable"), [
        ('relation "dbos.workflow_status" does not exist', True),  # Postgres, no schema yet
        ("no such table: workflow_status", True),  # SQLite, no schema yet
        ("permission denied for table workflow_status", False),
    ])
    def test_only_a_missing_schema_reads_as_an_empty_queue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, message: str, reachable: bool,
    ) -> None:
        from womblex.cloud.jobs import RunBoard

        def _raise(_self: object, *_a: object, **_kw: object) -> dict:
            raise RuntimeError(message)

        monkeypatch.setattr(RunBoard, "stats", _raise)
        ok, error = queue_reachable(f"sqlite:///{tmp_path / 'q.sqlite'}")
        assert ok is reachable and (error is None) is reachable

    def test_a_queue_that_cannot_be_reached_reports_why(self) -> None:
        ok, error = queue_reachable("postgresql://nobody@127.0.0.1:1/none")
        assert ok is False and error
