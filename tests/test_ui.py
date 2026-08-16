"""Tests for the console read API skeleton (womblex.ui).

Runs are plain directories with real, schema-correct manifest shards (no
fixture PDFs needed). The ``api_client`` fixture parametrizes every API test
over local (``output_root``) and store-backed (``RemoteStore``'s local
fsspec backend, as test_cloud.py uses — no S3/MinIO needed) settings, so
each test exercises both read paths through one body.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from womblex.store.output import ELEMENTS_SUFFIX, MANIFEST_SCHEMA
from womblex.ui.app import create_app
from womblex.ui.deps import UISettings, resolve_settings

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


@pytest.fixture(params=["local", "remote"])
def api_client(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[TestClient, Path]:
    """(client, run_root) — write ``<run_root>/<run_id>/documents/`` to seed a run."""
    if request.param == "remote":
        pytest.importorskip("fsspec")
        store_root = tmp_path / "store"
        return TestClient(create_app(store_uri=str(store_root))), store_root / "runs"
    return TestClient(create_app(output_root=tmp_path)), tmp_path


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
