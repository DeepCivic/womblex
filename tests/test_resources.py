"""What a deployment is wired to, credentials masked (`cloud/resources.py`, `/v1/resources`)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from womblex.cloud import resources
from womblex.cloud.deployment import DeploymentSettings


def _local(tmp_path: Path, **kw: object) -> DeploymentSettings:
    return DeploymentSettings(output_root=tmp_path, store_uri=None, **kw)  # type: ignore[arg-type]


def _s3(tmp_path: Path) -> DeploymentSettings:
    return DeploymentSettings(output_root=None, store_uri=f"s3://{tmp_path.name}/bucket")


class TestDescription:
    def test_a_local_store_and_no_ingest_or_queue(self, tmp_path: Path) -> None:
        got = resources.deployment_resources(_local(tmp_path))
        assert got["store"] == {"kind": "local", "uri": str(tmp_path), "is_object_store": False, "options": {}}
        assert got["ingest"] == {"configured": False, "uri": None, "is_object_store": False, "options": {}}
        assert got["queue"] == {"configured": False, "dsn_masked": None}

    def test_an_object_store_reports_endpoint_region_and_masked_credentials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("WOMBLEX_S3_ENDPOINT", "http://minio:9000")
        monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE1234")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "super-secret-value")  # pragma: allowlist secret
        got = resources.deployment_resources(_s3(tmp_path))
        assert "super-secret-value" not in json.dumps(got) and "AKIAEXAMPLE" not in json.dumps(got)
        options = got["store"]["options"]
        assert got["store"]["is_object_store"] is True
        assert (options["endpoint_url"], options["region"]) == ("http://minio:9000", "ap-southeast-2")
        assert options["credentials_configured"] is True and options["credentials_masked"].endswith("1234")

    def test_no_credentials_when_unset(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("AWS_ACCESS_KEY_ID", "WOMBLEX_S3_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
            monkeypatch.delenv(var, raising=False)
        options = resources.deployment_resources(_s3(tmp_path))["store"]["options"]
        assert (options["credentials_configured"], options["credentials_masked"]) == (False, None)

    def test_the_queue_password_is_masked(self, tmp_path: Path) -> None:
        dsn = "postgresql://operator:s3cret@db.internal:5432/womblex"  # pragma: allowlist secret
        queue = resources.deployment_resources(_local(tmp_path, db_dsn=dsn))["queue"]
        assert queue == {"configured": True, "dsn_masked": "postgresql://operator:***@db.internal:5432/womblex"}

    @pytest.mark.parametrize(("dsn", "secret"), [
        ("host=db.internal user=ops password=hunter2 dbname=w", "hunter2"),  # pragma: allowlist secret
        ("host=db password='two words' dbname=w", "words"),  # pragma: allowlist secret
        ("postgresql://ops@db:5432/w?password=hunter2", "hunter2"),  # pragma: allowlist secret
        ("postgresql://ops:p@ss@db:5432/w", "p@ss"),  # pragma: allowlist secret
    ])
    def test_every_dsn_form_is_masked(self, dsn: str, secret: str) -> None:
        masked = resources.mask_dsn(dsn)
        assert masked is not None and secret not in masked

    def test_masks_leave_what_is_not_secret(self) -> None:
        assert resources.mask_dsn("postgresql://db:5432/w") == "postgresql://db:5432/w"
        assert resources.mask_dsn(None) is None
        assert resources.mask_secret("abcd") == "•" * 8

    def test_isaacus_hosted_with_a_masked_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
        monkeypatch.setenv("ISAACUS_API_KEY", "sk-abcdef123456")  # pragma: allowlist secret
        got = resources.isaacus_description()
        assert "abcdef123456" not in json.dumps(got)  # pragma: allowlist secret
        assert (got["deployment"], got["endpoints"], got["api_key_configured"]) == ("hosted", [], True)
        assert got["api_key_masked"].endswith("3456") and got["unserved_models"] == []

    def test_isaacus_names_a_misspelled_endpoints_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
        monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINT", "kanon-2-bundle-001")
        assert resources.isaacus_description()["endpoints_typo"] == "ISAACUS_SAGEMAKER_ENDPOINT"
        monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "kanon-2-universal-001")
        got = resources.isaacus_description()
        assert (got["deployment"], got["endpoints_typo"], got["unserved_models"]) == ("sagemaker", None, [])

    def test_isaacus_names_the_models_no_endpoint_serves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "embed-001=kanon-2-embedder")
        got = resources.isaacus_description()
        assert got["endpoints"] == [{"name": "embed-001", "region": None, "models": ["kanon-2-embedder"]}]
        assert got["unserved_models"] == ["kanon-2-enricher"]

    def test_the_models_checked_are_the_config_defaults(self) -> None:
        from womblex.config import EmbeddingConfig, EnrichmentConfig

        assert set(resources.ISAACUS_MODELS) == {
            EmbeddingConfig.model_fields["model"].default, EnrichmentConfig.model_fields["model"].default}


class TestChecks:
    def test_a_local_store(self, tmp_path: Path) -> None:
        assert resources.check_store(_local(tmp_path)) == (True, None)
        ok, error = resources.check_store(_local(tmp_path / "missing"))
        assert ok is False and error

    def test_an_object_store(self, tmp_path: Path) -> None:
        pytest.importorskip("fsspec")
        assert resources.check_store(DeploymentSettings(output_root=None, store_uri=str(tmp_path))) == (True, None)
        ok, error = resources.check_store(DeploymentSettings(output_root=None, store_uri="not-a-protocol://x"))
        assert ok is False and error

    def test_the_ingest_location(self, tmp_path: Path) -> None:
        pytest.importorskip("fsspec")
        assert resources.check_ingest(_local(tmp_path)) == (False, "no ingest location configured")
        assert resources.check_ingest(_local(tmp_path, ingest_uri=str(tmp_path))) == (True, None)
        ok, error = resources.check_ingest(_local(tmp_path, ingest_uri="not-a-protocol://x"))
        assert ok is False and error


def test_the_routes_are_admin_only(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from womblex.api.app import create_api_app
    from womblex.api.auth import hash_token, parse_registry

    registry = parse_registry({"clients": [
        {"client_id": "a", "token_sha256": hash_token("ta"), "scopes": ["read", "submit", "read_raw"]},
        {"client_id": "ops", "token_sha256": hash_token("to"), "scopes": ["admin"]},
    ]})
    api = TestClient(create_api_app(store_uri=str(tmp_path), registry=registry,
                                    db_dsn="postgresql://ops:pw@127.0.0.1:1/x"))  # pragma: allowlist secret
    assert api.get("/v1/resources", headers={"Authorization": "Bearer ta"}).status_code == 403
    ops = {"Authorization": "Bearer to"}
    got = api.get("/v1/resources", headers=ops)
    assert got.status_code == 200 and ":pw@" not in got.text
    assert api.post("/v1/resources/test/store", headers=ops).json() == {"target": "store", "reachable": True, "error": None}
    assert api.post("/v1/resources/test/ingest", headers=ops).json()["reachable"] is False
    assert api.post("/v1/resources/test/queue", headers=ops).json()["reachable"] is False
    assert api.post("/v1/resources/test/nope", headers=ops).status_code == 422


class TestDeployments:
    """`deployments`: each registered deployment and the models it may serve, no value shown."""

    def _by_name(self, tmp_path: Path) -> dict[str, dict]:
        got = resources.deployment_resources(_local(tmp_path))["deployments"]
        return {d["name"]: d for d in got}

    def test_every_built_in_is_listed_with_the_models_that_name_it(self, tmp_path: Path) -> None:
        listed = self._by_name(tmp_path)
        assert {"isaacus-api", "isaacus-sagemaker", "bedrock", "ollama"} <= set(listed)
        embedder = {"slot": "embedding", "name": "kanon-2-embedder", "vendor_model": "kanon-2-embedder"}
        assert embedder.items() <= next(m for m in listed["isaacus-api"]["models"] if m["slot"] == "embedding").items()
        assert [m["name"] for m in listed["bedrock"]["models"]] == ["mistral-ocr"]
        assert [m["name"] for m in listed["ollama"]["models"]] == ["ollama"]
        assert listed["ollama"]["billed"] is False and listed["isaacus-api"]["billed"] is True

    def test_variables_are_reported_set_or_not_and_no_value_is_shown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
        monkeypatch.setenv("ISAACUS_API_KEY", "iuak_live_value_1234")  # pragma: allowlist secret -- a test placeholder
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.0.0.7:11434")
        listed = self._by_name(tmp_path)
        assert listed["isaacus-api"]["variables"] == [{"names": ["ISAACUS_API_KEY"], "configured": True}]
        assert listed["isaacus-sagemaker"]["variables"] == [{"names": ["ISAACUS_SAGEMAKER_ENDPOINTS"], "configured": False}]
        assert "OLLAMA_BASE_URL" in listed["ollama"]["optional_variables"]
        text = json.dumps(resources.deployment_resources(_local(tmp_path)))
        assert "iuak_live_value_1234" not in text and "10.0.0.7" not in text

    def test_sagemaker_serves_the_models_its_endpoints_declare(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "embed-001=kanon-2-embedder")
        served = {m["name"]: m["served"] for m in self._by_name(tmp_path)["isaacus-sagemaker"]["models"]}
        assert served == {"kanon-2-embedder": True, "kanon-2-enricher": False}

    def test_a_plugin_deployment_and_its_models_are_listed_with_their_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from womblex.utils import deployment_registry as dr
        from womblex.utils import model_registry as mr

        monkeypatch.setattr(dr, "_deployments", dict(dr._deployments))
        monkeypatch.setattr(mr, "_entries", {s: dict(m) for s, m in mr._entries.items()})
        monkeypatch.setattr(mr, "_aliases", {s: dict(m) for s, m in mr._aliases.items()})
        dr.register(dr.Deployment(name="acme", check=lambda m: None, needs_network=True, billed=True,
                                  source="acme-dist", version="3.0"))
        mr.register(mr.SLOT_EMBEDDING, "acme-embed", lambda **_: None, source="acme-dist",
                    deployments=("acme",), vendor_model="acme-embed-v3")
        acme = self._by_name(tmp_path)["acme"]
        assert (acme["distribution"], acme["version"]) == ("acme-dist", "3.0")
        assert acme["models"] == [{"slot": "embedding", "name": "acme-embed",
                                   "vendor_model": "acme-embed-v3", "served": True}]


class TestStorageOptions:
    """A location's own `storage_options` are described by name, never by value, and never resolved."""

    def test_an_env_reference_is_shown_as_the_variable_and_a_literal_secret_is_masked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("R2_SECRET", raising=False)  # unset: describing a location must not need it
        settings = DeploymentSettings(
            output_root=None, store_uri=f"s3://{tmp_path.name}/bucket",
            store_options={"key": "r2-id", "secret": {"env": "R2_SECRET"},
                           "client_kwargs": {"endpoint_url": "https://acct.r2.example"}},
        )
        shown = resources.deployment_resources(settings)["store"]["storage_options"]
        assert shown == {"key": "••••••••", "secret": {"env": "R2_SECRET"},
                         "client_kwargs": {"endpoint_url": "https://acct.r2.example"}}
        assert "r2-id" not in json.dumps(resources.deployment_resources(settings))

    def test_the_environment_declares_each_locations_options_and_settings_report_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store, ingest = f"s3://{tmp_path.name}-store/womblex", f"s3://{tmp_path.name}-in/in"
        monkeypatch.setenv("WOMBLEX_STORE_URI", store)
        monkeypatch.setenv("WOMBLEX_INGEST_URI", ingest)
        monkeypatch.setenv("WOMBLEX_STORE_STORAGE_OPTIONS", '{"anon": false}')
        monkeypatch.setenv("ACME_TOKEN", "t-value")
        monkeypatch.setenv("WOMBLEX_INGEST_STORAGE_OPTIONS", '{"profile": "acme", "token": {"env": "ACME_TOKEN"}}')
        got = resources.deployment_resources(DeploymentSettings(output_root=None, store_uri=store, ingest_uri=ingest))
        assert got["store"]["storage_options"] == {"anon": False}
        assert got["ingest"]["storage_options"] == {"profile": "acme", "token": {"env": "ACME_TOKEN"}}

    def test_a_location_with_none_shows_an_empty_object(self, tmp_path: Path) -> None:
        assert resources.deployment_resources(_s3(tmp_path))["store"]["storage_options"] == {}
