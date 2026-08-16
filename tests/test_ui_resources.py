"""Tests for the Resources Console (docs/ui-plan.md merge 10).

Kept out of ``test_ui.py`` (already at the 750-line file cap) rather than
growing it further — ``TestResourcesApi`` needs none of that file's shared
fixtures, since every case builds its own ``TestClient`` directly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from womblex.ui import resources
from womblex.ui.app import create_app


class TestResourcesApi:
    """The Resources Console reuses existing library reads rather than adding
    detection logic of its own (docs/ui-plan.md merge 10): `GET /api/resources`
    is cheap and network-free; `POST /test/*` is the live check behind each
    card's action button.
    """

    def test_store_card_local(self, tmp_path: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path))
        card = client.get("/api/resources").json()["store"]
        assert card == {
            "kind": "local", "uri": str(tmp_path), "is_object_store": False, "options": {},
        }

    def test_store_card_remote_flags_object_store(self, tmp_path: Path) -> None:
        pytest.importorskip("fsspec")
        client = TestClient(create_app(store_uri=f"s3://{tmp_path}/bucket"))
        card = client.get("/api/resources").json()["store"]
        assert card["kind"] == "remote"
        assert card["is_object_store"] is True

    def test_store_card_remote_local_backend_is_not_flagged_object_store(
        self, tmp_path: Path
    ) -> None:
        """A plain dir used as ``store_uri`` (fsspec's local backend, as
        ``test_ui.py``'s ``api_client`` fixture does) is real for
        ``settings.is_remote``, but it is not an object store."""
        pytest.importorskip("fsspec")
        client = TestClient(create_app(store_uri=str(tmp_path / "store")))
        card = client.get("/api/resources").json()["store"]
        assert card["kind"] == "remote"
        assert card["is_object_store"] is False

    def test_store_card_never_leaks_aws_credentials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "super-secret-value")  # pragma: allowlist secret
        client = TestClient(create_app(store_uri=f"s3://{tmp_path}/bucket"))
        resp = client.get("/api/resources")
        assert "super-secret-value" not in resp.text
        assert resp.json()["store"]["options"]["credentials_configured"] is True

    def test_store_card_reports_s3_endpoint_and_region(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WOMBLEX_S3_ENDPOINT", "http://minio:9000")
        monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
        client = TestClient(create_app(store_uri=f"s3://{tmp_path}/bucket"))
        options = client.get("/api/resources").json()["store"]["options"]
        assert options["endpoint_url"] == "http://minio:9000"
        assert options["region"] == "ap-southeast-2"

    def test_queue_card_unconfigured(self, tmp_path: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path))
        assert client.get("/api/resources").json()["queue"] == {
            "configured": False, "dsn_masked": None,
        }

    def test_queue_card_masks_the_password(self, tmp_path: Path) -> None:
        dsn = "postgresql://operator:s3cret@db.internal:5432/womblex"  # pragma: allowlist secret
        client = TestClient(create_app(output_root=tmp_path, db_dsn=dsn))
        resp = client.get("/api/resources")
        assert "s3cret" not in resp.text
        card = resp.json()["queue"]
        assert card["configured"] is True
        assert card["dsn_masked"] == "postgresql://operator:***@db.internal:5432/womblex"

    def test_isaacus_card_hosted_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
        monkeypatch.setenv("ISAACUS_API_KEY", "sk-abcdef123456")
        client = TestClient(create_app(output_root=tmp_path))
        resp = client.get("/api/resources")
        assert "abcdef123456" not in resp.text  # only the masked tail may appear
        card = resp.json()["isaacus"]
        assert card["deployment"] == "hosted"
        assert card["endpoints"] == []
        assert card["api_key_configured"] is True
        assert card["api_key_masked"].endswith("3456")
        assert card["unserved_models"] == []

    def test_isaacus_card_no_key_configured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
        monkeypatch.delenv("ISAACUS_API_KEY", raising=False)
        client = TestClient(create_app(output_root=tmp_path))
        card = client.get("/api/resources").json()["isaacus"]
        assert card["api_key_configured"] is False
        assert card["api_key_masked"] is None

    def test_isaacus_card_reports_unserved_models_on_sagemaker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "embed-001=kanon-2-embedder")
        client = TestClient(create_app(output_root=tmp_path))
        card = client.get("/api/resources").json()["isaacus"]
        assert card["deployment"] == "sagemaker"
        assert card["endpoints"] == [
            {"name": "embed-001", "region": None, "models": ["kanon-2-embedder"]}
        ]
        assert card["unserved_models"] == ["kanon-2-enricher"]

    def test_isaacus_card_universal_endpoint_serves_everything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "kanon-2-universal-001")
        client = TestClient(create_app(output_root=tmp_path))
        assert client.get("/api/resources").json()["isaacus"]["unserved_models"] == []

    def test_test_store_local_reachable(self, tmp_path: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path))
        assert client.post("/api/resources/test/store").json() == {
            "reachable": True, "error": None,
        }

    def test_test_store_local_unreachable(self, tmp_path: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path / "does-not-exist"))
        body = client.post("/api/resources/test/store").json()
        assert body["reachable"] is False
        assert body["error"]

    def test_test_store_remote_reachable(self, tmp_path: Path) -> None:
        pytest.importorskip("fsspec")
        client = TestClient(create_app(store_uri=str(tmp_path / "store")))
        assert client.post("/api/resources/test/store").json() == {
            "reachable": True, "error": None,
        }

    def test_test_store_remote_unreachable_reports_rather_than_raises(
        self, tmp_path: Path
    ) -> None:
        """An unknown fsspec protocol fails synchronously, no network needed —
        the same "report, don't 500" contract as the queue's unreachable test."""
        pytest.importorskip("fsspec")
        client = TestClient(create_app(store_uri="not-a-real-protocol://nope"))
        body = client.post("/api/resources/test/store").json()
        assert body["reachable"] is False
        assert body["error"]

    def test_test_queue_not_configured(self, tmp_path: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path))
        body = client.post("/api/resources/test/queue").json()
        assert body == {"reachable": False, "error": "no queue configured", "queue": None}

    def test_test_queue_unreachable_reports_rather_than_raises(self, tmp_path: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path, db_dsn="postgresql://nope:1/nope"))
        body = client.post("/api/resources/test/queue").json()
        assert body["reachable"] is False
        assert body["error"]
        assert body["queue"] is None

    def test_mask_dsn_leaves_a_passwordless_dsn_untouched(self) -> None:
        assert resources._mask_dsn("postgresql://db.internal:5432/womblex") == (
            "postgresql://db.internal:5432/womblex"
        )

    def test_mask_dsn_none_is_none(self) -> None:
        assert resources._mask_dsn(None) is None
