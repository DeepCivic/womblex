"""Every sidecar records the deployment each model ran on; the run record lists them."""

from __future__ import annotations

import json

import pytest

from tests.test_run_record import _extracted, extraction  # noqa: F401
from womblex.config import EmbeddingConfig, EnrichmentConfig
from womblex.store.deployment_record import DEPLOYMENTS_KEY, deployments_payload, read_footer_deployments
from womblex.store.run_manifest import read_run_record, write_run_manifest
from womblex.utils import deployment_registry as dr
from womblex.utils import model_registry as mr
from womblex.utils.deployment_registry import Deployment


@pytest.fixture(autouse=True)
def _forget_use():
    mr.reset_used_entries()
    yield
    mr.reset_used_entries()


def _embed(deployment: str | None = None):
    from womblex.analyse.embed import make_embedding_provider

    return make_embedding_provider(EmbeddingConfig(deployment=deployment), client=object())


def test_a_process_that_used_no_deployed_model_writes_no_key() -> None:
    assert deployments_payload() == []


def test_a_model_is_recorded_with_the_deployment_it_ran_on(monkeypatch) -> None:
    from womblex import __version__

    monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
    _embed()
    assert deployments_payload() == [{
        "slot": "embedding", "name": "kanon-2-embedder", "deployment": "isaacus-api",
        "distribution": "womblex", "version": __version__, "endpoint": "sdk-default",
    }]


def test_a_named_deployment_is_the_one_recorded_even_when_the_environment_defaults_elsewhere(monkeypatch) -> None:
    monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "embed-001@ap-southeast-2")
    _embed("isaacus-api")
    (entry,) = deployments_payload()
    assert entry["deployment"] == "isaacus-api"


def test_a_sagemaker_endpoint_spelled_as_an_arn_is_recorded_as_its_name_and_region(monkeypatch) -> None:
    arn = "arn:aws:sagemaker:ap-southeast-2:123456789012:endpoint/embed-001"
    monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", f"{arn}=kanon-2-embedder")
    _embed()
    (entry,) = deployments_payload()
    assert (entry["deployment"], entry["endpoint"], entry["region"]) == (
        "isaacus-sagemaker", "embed-001", "ap-southeast-2")
    assert "123456789012" not in json.dumps(deployments_payload())


def test_the_model_check_building_a_provider_to_probe_it_records_nothing() -> None:
    with mr.suppress_use_recording():
        _embed()
    assert deployments_payload() == []


def test_ai_chunking_records_its_deployment_under_the_chunking_slot(monkeypatch) -> None:
    from womblex.providers.isaacus.client import make_ai_chunking_client

    monkeypatch.setenv("ISAACUS_API_KEY", "k")  # pragma: allowlist secret -- a test placeholder
    monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
    make_ai_chunking_client("kanon-2-enricher", "isaacus-api")
    assert [(e["slot"], e["name"], e["deployment"]) for e in deployments_payload()] == [
        ("chunking", "kanon-2-enricher", "isaacus-api")]


def test_a_plugin_deployment_is_recorded_with_its_distribution_and_version(monkeypatch) -> None:
    monkeypatch.setattr(dr, "_deployments", dict(dr._deployments))
    dr.register(Deployment(name="acme", check=lambda m: None, needs_network=True, billed=True,
                           source="acme-dist", version="2.1.0", detail=lambda m: {"endpoint": "acme-eu"}))
    mr.record_deployment("embedding", "acme-embed", "acme")
    assert deployments_payload() == [{
        "slot": "embedding", "name": "acme-embed", "deployment": "acme",
        "distribution": "acme-dist", "version": "2.1.0", "endpoint": "acme-eu",
    }]


def test_a_deployment_recorded_then_unregistered_is_still_named(monkeypatch) -> None:
    monkeypatch.setattr(dr, "_deployments", {})
    monkeypatch.setattr(dr, "_loaded", True)
    mr.record_deployment("embedding", "m", "gone")
    assert deployments_payload() == [{"slot": "embedding", "name": "m", "deployment": "gone"}]


@pytest.mark.parametrize("raw", [b"not json", b'{"a": 1}', b'[{"slot": "x"}]', b"\xff"])
def test_a_malformed_key_reads_as_none_recorded(raw: bytes) -> None:
    assert read_footer_deployments({DEPLOYMENTS_KEY.encode(): raw}) == []
    assert read_footer_deployments(None) == []


def test_the_run_record_lists_the_deployments_with_the_stages_that_carried_them(tmp_path, extraction, monkeypatch) -> None:  # noqa: F811
    monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
    _embed()
    shards = _extracted(tmp_path, extraction)
    record = read_run_record(write_run_manifest(shards))
    assert record["record_version"] == 10
    (entry,) = record["deployments"]
    assert (entry["slot"], entry["deployment"], entry["stages"]) == ("embedding", "isaacus-api", ["extract"])


def test_a_run_that_used_none_records_an_empty_list(tmp_path, extraction) -> None:  # noqa: F811
    record = read_run_record(write_run_manifest(_extracted(tmp_path, extraction)))
    assert record["deployments"] == []


def test_enrichment_is_recorded_too(monkeypatch) -> None:
    from womblex.analyse.enrich import make_enrichment_provider

    monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
    make_enrichment_provider(EnrichmentConfig(), client=object())
    assert [(e["slot"], e["deployment"]) for e in deployments_payload()] == [("enrichment", "isaacus-api")]
