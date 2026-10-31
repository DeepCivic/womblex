"""What a run over an ingest prefix would pick up (`cloud/ingest_preview.py`, `GET /v1/ingest`)."""
from __future__ import annotations

from pathlib import Path

import pytest

from womblex.cloud.deployment import DeploymentSettings
from womblex.cloud.dispatch import PrefixNotOwned
from womblex.cloud.ingest_preview import ingest_preflight

pytest.importorskip("fsspec")


def _seed(root: Path, prefix: str, names: list[str]) -> None:
    (root / prefix).mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / prefix / name).write_bytes(b"%PDF-1.4 stub")


def _settings(tmp_path: Path) -> DeploymentSettings:
    return DeploymentSettings(output_root=None, store_uri=str(tmp_path / "store"), ingest_uri=str(tmp_path / "inbox"))


class TestPreflight:
    def test_no_ingest_configured(self, tmp_path: Path) -> None:
        assert ingest_preflight(DeploymentSettings(output_root=tmp_path, store_uri=None)) == {
            "uri": None, "input_prefix": "", "kind": None, "reachable": False, "document_count": 0,
            "sample": [], "subdirectories": [], "error": "no ingest location configured",
        }

    def test_a_flat_location_reports_its_count_and_sample(self, tmp_path: Path) -> None:
        _seed(tmp_path / "inbox", "", ["d0.pdf", "d1.pdf", "notes.xyz"])
        body = ingest_preflight(_settings(tmp_path))
        assert (body["kind"], body["reachable"], body["document_count"]) == ("local", True, 2)
        assert sorted(body["sample"]) == ["d0.pdf", "d1.pdf"]

    def test_a_nested_layout_reports_the_refusal_and_the_prefixes_that_would_run(self, tmp_path: Path) -> None:
        _seed(tmp_path / "inbox", "", ["top.pdf"])
        _seed(tmp_path / "inbox", "2026-08/health", ["n.pdf"])
        _seed(tmp_path / "inbox", "scratch", ["a.pdf", "b.pdf"])
        body = ingest_preflight(_settings(tmp_path))
        assert body["reachable"] is True and body["document_count"] == 0 and "2026-08/" in body["error"]
        assert body["subdirectories"] == [
            {"name": "2026-08", "input_prefix": "2026-08", "document_count": 1},
            {"name": "scratch", "input_prefix": "scratch", "document_count": 2},
        ]
        deeper = ingest_preflight(_settings(tmp_path), input_prefix="2026-08")
        assert deeper["subdirectories"] == [{"name": "health", "input_prefix": "2026-08/health", "document_count": 1}]

    def test_the_prefix_is_echoed_normalised(self, tmp_path: Path) -> None:
        _seed(tmp_path / "inbox", "2026-08", ["a.pdf"])
        body = ingest_preflight(_settings(tmp_path), input_prefix="./2026-08/")
        assert (body["input_prefix"], body["sample"]) == ("2026-08", ["2026-08/a.pdf"])

    def test_the_count_previewed_is_the_count_enqueued(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from tests.test_dispatch import _Board

        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _Board)
        monkeypatch.setattr("womblex.cloud.dispatch.ensure_schema", lambda _dsn: None)
        monkeypatch.setattr("womblex.cloud.worker_registry.require_served", lambda _dsn, _needed: None)
        from womblex.cloud.dispatch import enqueue_extraction

        _seed(tmp_path / "inbox", "2026-08", ["a.pdf", "b.pdf"])
        _seed(tmp_path / "inbox", "2026-09", ["c.pdf"])
        settings = DeploymentSettings(output_root=None, store_uri=str(tmp_path / "store"),
                                      ingest_uri=str(tmp_path / "inbox"), db_dsn="postgresql://x/y")
        preview = ingest_preflight(settings, input_prefix="2026-08")
        result = enqueue_extraction(settings, input_prefix=preview["input_prefix"], run_id="r1")
        assert result.document_count == preview["document_count"] == 2

    def test_an_escaping_or_unowned_prefix_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unsafe input_prefix"):
            ingest_preflight(_settings(tmp_path), input_prefix="../elsewhere")
        with pytest.raises(PrefixNotOwned):
            ingest_preflight(_settings(tmp_path), input_prefix="bob/x", owner="alice")

    def test_an_unreachable_location_is_reported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from womblex.store import remote

        def _raise(_uri: str, **_kw: object) -> object:
            raise OSError("endpoint down")

        monkeypatch.setattr(remote.RemoteStore, "from_uri", staticmethod(_raise))
        body = ingest_preflight(_settings(tmp_path))
        assert body["reachable"] is False and body["error"] == "endpoint down"


class TestRoute:
    @pytest.fixture
    def api(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from womblex.api.app import create_api_app
        from womblex.api.auth import hash_token, parse_registry

        _seed(tmp_path / "inbox", "alice/up1", ["a.pdf"])
        registry = parse_registry({"clients": [
            {"client_id": "alice", "token_sha256": hash_token("ta"), "scopes": ["submit"]},
            {"client_id": "reader", "token_sha256": hash_token("tr"), "scopes": ["read"]},
            {"client_id": "ops", "token_sha256": hash_token("to"), "scopes": ["admin"]},
        ]})
        return TestClient(create_api_app(store_uri=str(tmp_path / "store"), db_dsn="postgresql://x/y",
                                         registry=registry, ingest_uri=str(tmp_path / "inbox")))

    def test_a_submitter_previews_only_its_own_tree(self, api) -> None:  # type: ignore[no-untyped-def]
        alice = {"Authorization": "Bearer ta"}
        body = api.get("/v1/ingest", params={"input_prefix": "alice/up1"}, headers=alice).json()
        assert (body["document_count"], body["sample"]) == (1, ["alice/up1/a.pdf"])
        assert api.get("/v1/ingest", params={"input_prefix": "bob"}, headers=alice).status_code == 403
        assert api.get("/v1/ingest", headers=alice).status_code == 403
        assert api.get("/v1/ingest", params={"input_prefix": "../x"}, headers=alice).status_code == 400
        assert api.get("/v1/ingest", headers={"Authorization": "Bearer tr"}).status_code == 403
        root = api.get("/v1/ingest", headers={"Authorization": "Bearer to"}).json()
        assert [d["input_prefix"] for d in root["subdirectories"]] == ["alice"]
