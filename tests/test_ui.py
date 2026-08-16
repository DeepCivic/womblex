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
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from womblex.cli import ALL_COMMANDS
from womblex.store.output import ELEMENTS_SUFFIX, MANIFEST_SCHEMA
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
