"""Console settings, store-unreachable degradation, SPA mount and the `womblex ui` CLI (womblex.ui)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from womblex.ui import readers
from womblex.ui.app import create_app
from womblex.ui.deps import UISettings, resolve_settings


class TestUISettings:
    def test_needs_exactly_one_source(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            UISettings(output_root=None, store_uri=None)
        with pytest.raises(ValueError, match="exactly one"):
            UISettings(output_root=tmp_path, store_uri="s3://bucket")

    def test_is_remote(self, tmp_path: Path) -> None:
        assert UISettings(output_root=tmp_path, store_uri=None).is_remote is False
        assert UISettings(output_root=None, store_uri="s3://bucket").is_remote is True

    def test_ingest_and_store_must_be_disjoint(self) -> None:
        UISettings(output_root=None, store_uri="s3://womblex", ingest_uri="s3://womblex/inbox")
        with pytest.raises(ValueError, match="not disjoint"):
            UISettings(output_root=None, store_uri="s3://womblex", ingest_uri="s3://womblex")


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

    def test_presets_dir_explicit_arg_wins(self, tmp_path: Path) -> None:
        presets_dir = tmp_path / "presets"
        assert resolve_settings(tmp_path, None, presets_dir=presets_dir).presets_dir == presets_dir

    def test_presets_dir_env_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        presets_dir = tmp_path / "presets"
        monkeypatch.setenv("WOMBLEX_UI_PRESETS_DIR", str(presets_dir))
        assert resolve_settings(tmp_path, None).presets_dir == presets_dir

    def test_presets_dir_defaults_to_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WOMBLEX_UI_PRESETS_DIR", raising=False)
        assert resolve_settings(tmp_path, None).presets_dir is None

    def test_presets_writable_needs_a_dir_locally_but_not_remotely(self, tmp_path: Path) -> None:
        """Local mode needs a writable presets dir; remote always writes to the store."""
        assert resolve_settings(tmp_path, None).presets_writable is False
        assert resolve_settings(tmp_path, None, presets_dir=tmp_path / "p").presets_writable is True
        # Remote mode saves to the store's own presets/ prefix, so it needs no dir.
        assert resolve_settings(None, "s3://bucket").presets_writable is True

    def test_ingest_uri_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WOMBLEX_INGEST_URI", "s3://womblex/inbox")
        assert resolve_settings(None, "s3://womblex").ingest_uri == "s3://womblex/inbox"

class TestStoreUnreachable:
    """A store that cannot be opened degrades to a legible 503, not an opaque 500.

    The canonical cause in a cloud deployment is ``womblex[cloud]`` (and so
    ``s3fs``) not being installed: ``RemoteStore.from_uri('s3://…')`` raises
    ``ImportError: Install s3fs to access S3``. That is a deployment fault, not
    a bug in the request, so the read routes surface it as 503 carrying the
    underlying message — the same cause the Resources card's *Test connection*
    reports — rather than letting the raw exception become a 500 top-banner.
    """

    def _client(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
        def _boom(_uri: str, _credentials: tuple[str, str] | None = None) -> object:
            raise readers.StoreUnreachable("Install s3fs to access S3")

        monkeypatch.setattr(readers, "_open_store", _boom)
        return TestClient(create_app(store_uri=str(tmp_path / "store")))

    def test_list_runs_returns_503_with_the_cause(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        resp = client.get("/api/runs")
        assert resp.status_code == 503
        assert "s3fs" in resp.json()["detail"]

    def test_presets_returns_503_with_the_cause(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        resp = client.get("/api/composer/presets")
        assert resp.status_code == 503
        assert "s3fs" in resp.json()["detail"]

    def test_manifest_and_audit_also_degrade_to_503(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        assert client.get("/api/runs/run-a/manifest").status_code == 503
        assert client.get("/api/runs/run-a/audit").status_code == 503
        assert client.get("/api/runs/run-a/stage-presence/chunk").status_code == 503
        assert client.get("/api/runs/run-a/chunks/hash-a").status_code == 503

    def test_open_store_wraps_a_backend_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wrap happens in `_open_store`, so every reader shares one mapping."""
        import womblex.store.remote as remote_mod

        def _raise(_uri: str, **_kw: object) -> object:
            raise ImportError("Install s3fs to access S3")

        monkeypatch.setattr(remote_mod.RemoteStore, "from_uri", staticmethod(_raise))
        with pytest.raises(readers.StoreUnreachable, match="s3fs"):
            readers._open_store("s3://redline")


class TestSpaMount:
    """The console serves the built SPA (docs/functional_requirements.md, Requirement 15) when one is
    present alongside it, and falls back to the API-only shape when not —
    the same image serves cloud and local deployments.
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
