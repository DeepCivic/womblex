"""A slot's `deployment:` setting: honoured, refused when the deployment cannot serve the model, never rerouted."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_model_check import _config
from womblex.config import DatasetConfig, PathsConfig, WomblexConfig
from womblex.providers.isaacus import client as isaacus_client
from womblex.store.run_stamp import config_digest
from womblex.utils import deployment_registry as dr
from womblex.utils import model_check
from womblex.utils.deployment_choice import choice_problems, deployment_of, pick_deployment
from womblex.utils.model_check import SCOPE_CHUNK, SCOPE_EMBED, SCOPE_ENRICH, check_models, check_registered

#: The digest of a config that sets no `deployment`, from before the setting existed.
DEFAULT_DIGEST = "sha256:74e7b29907377b9156752567f8dfc3fb9054979bf46353a4dd86f93df82fa612"


def _default() -> WomblexConfig:
    return WomblexConfig(dataset=DatasetConfig(name="x"), paths=PathsConfig(
        input_root=Path("/in"), output_root=Path("/out"), checkpoint_dir=Path("/ckpt")))


def test_a_config_that_names_no_deployment_keeps_the_digest_it_had() -> None:
    assert config_digest(_default()) == DEFAULT_DIGEST


def test_naming_a_deployment_changes_the_digest() -> None:
    config = _default()
    config.embedding.deployment = "isaacus-sagemaker"
    assert config_digest(config) != DEFAULT_DIGEST


def test_the_default_is_the_environments_isaacus_deployment_and_a_named_one_wins(monkeypatch) -> None:
    both = ("isaacus-api", "isaacus-sagemaker")
    monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
    assert pick_deployment(both) == "isaacus-api"
    monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "e-001")
    assert pick_deployment(both) == "isaacus-sagemaker"
    assert pick_deployment(both, " Isaacus-API ") == "isaacus-api"
    assert pick_deployment(("bedrock",)) == "bedrock"


def test_a_local_model_has_no_deployment() -> None:
    assert deployment_of("ocr", "paddleocr") is None
    assert deployment_of("ocr", "mistral-ocr") == "bedrock"
    assert deployment_of("embedding", "kanon-2-embedder", "isaacus-sagemaker") == "isaacus-sagemaker"


@pytest.mark.parametrize(("section", "field", "value", "scope", "text"), [
    ("enrichment", "deployment", "bedrock", SCOPE_ENRICH,
     ("enrichment model 'kanon-2-enricher' is not served by deployment 'bedrock'; it is served by: "
      "isaacus-api, isaacus-sagemaker")),
    ("embedding", "deployment", "nope", SCOPE_EMBED, "unknown deployment: 'nope' (known:"),
    ("extraction", "ocr", "ollama", "extract",
     "ocr model 'paddleocr' runs locally and takes no deployment (got 'ollama')"),
])
def test_a_deployment_the_model_cannot_use_is_refused_before_launch(section, field, value, scope, text) -> None:
    config = _config(**({section: {field: value}} if section != "extraction" else
                        {"extraction": {"ocr": {"deployment": value}}}))
    ((got_scope, _slot, _model, message),) = choice_problems(config)
    assert got_scope == scope and text in message
    with pytest.raises(ValueError, match="not served|unknown deployment|runs locally"):
        check_registered(config)  # what `POST /v1/runs` and `dispatch-workflow` call


def test_chunking_deployment_needs_a_chunking_model() -> None:
    config = _config(chunking={"chunking_deployment": "isaacus-api"})
    ((scope, *_, message),) = choice_problems(config)
    assert scope == SCOPE_CHUNK and "needs chunking.chunking_model" in message


def test_a_valid_choice_is_not_a_problem() -> None:
    config = _config(embedding={"deployment": "isaacus-sagemaker"}, enrichment={"deployment": "isaacus-api"},
                     chunking={"chunking_model": "kanon-2-enricher", "chunking_deployment": "isaacus-api"})
    assert choice_problems(config) == []
    check_registered(config)


def test_a_refusal_is_a_failed_check_scoped_to_its_stage_and_nothing_else_is_rerouted(monkeypatch) -> None:
    monkeypatch.setenv("ISAACUS_API_KEY", "k")  # pragma: allowlist secret -- a test placeholder
    monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: SimpleNamespace(status_code=404))
    config = _config(enrichment={"enabled": True, "deployment": "bedrock"}, embedding={"enabled": True})
    result = check_models(config, "load", scopes=(SCOPE_ENRICH, SCOPE_EMBED))
    (bad,) = result.failures
    assert bad.scopes == (SCOPE_ENRICH,) and "not served by deployment 'bedrock'" in bad.reason
    assert result.failures_for((SCOPE_EMBED,)) == ()  # the embedder still passes on the hosted API


def test_sagemaker_chosen_without_its_endpoints_names_what_it_serves(monkeypatch) -> None:
    monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
    config = _config(embedding={"enabled": True, "deployment": "isaacus-sagemaker"})
    (bad,) = check_models(config, "load", scopes=(SCOPE_EMBED,)).failures
    assert bad.source == "isaacus-sagemaker"
    assert ("deployment 'isaacus-sagemaker' does not serve 'kanon-2-embedder'; "
            "it serves: nothing (ISAACUS_SAGEMAKER_ENDPOINTS is not set)") in bad.reason


def test_two_models_of_one_vendor_can_run_on_two_deployments(monkeypatch) -> None:
    import boto3
    import httpx

    monkeypatch.setenv("ISAACUS_API_KEY", "k")  # pragma: allowlist secret -- a test placeholder
    monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "embed-001@ap-southeast-2=kanon-2-embedder")
    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: SimpleNamespace(status_code=404))
    monkeypatch.setattr(boto3, "Session", lambda **_: SimpleNamespace(region_name=None, client=lambda *a, **k: SimpleNamespace(
        describe_endpoint=lambda EndpointName: {"EndpointStatus": "InService"})))
    config = _config(enrichment={"enabled": True, "deployment": "isaacus-api"},
                     embedding={"enabled": True, "deployment": "isaacus-sagemaker"})
    uses = {u.name: u.deployment for u in model_check.deployed_models(config)}
    assert uses == {"kanon-2-enricher": "isaacus-api", "kanon-2-embedder": "isaacus-sagemaker"}
    result = check_models(config, "load", scopes=(SCOPE_ENRICH, SCOPE_EMBED))
    assert result.ok and {c.source for c in result.checks} == {"isaacus-api", "isaacus-sagemaker"}


def test_each_deployment_gets_its_own_client(monkeypatch) -> None:
    made: list[str] = []
    monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "e-001")
    monkeypatch.setattr(isaacus_client, "make_api_client", lambda: made.append("api") or "api-client")
    monkeypatch.setattr(isaacus_client, "make_sagemaker_client",
                        lambda endpoints, models: made.append("sagemaker") or "sagemaker-client")
    assert isaacus_client.make_ai_chunking_client("kanon-2-enricher", "isaacus-api") == "api-client"
    assert isaacus_client.make_ai_chunking_client("kanon-2-enricher") == "sagemaker-client"  # the environment's default
    assert isaacus_client.make_ai_chunking_client("kanon-2-enricher", "isaacus-api") == "api-client"
    assert made == ["api", "sagemaker"]  # one client per deployment
    assert isaacus_client.make_ai_chunking_client(None, "isaacus-api") is None


def test_the_stage_runner_builds_the_client_of_the_chosen_deployment(tmp_path, monkeypatch) -> None:
    from tests.test_cloud import _minimal_config
    from womblex.cloud.stage_contracts import STAGE_CONTRACTS
    from womblex.cloud.stage_runner import prepare_stage_context

    monkeypatch.setenv("ISAACUS_API_KEY", "k")  # pragma: allowlist secret -- a test placeholder
    monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "e-001")
    monkeypatch.setattr(isaacus_client, "make_api_client", lambda: "api-client")
    monkeypatch.setattr(model_check, "check_models", lambda *a, **k: SimpleNamespace(failures=()))
    config = _minimal_config(tmp_path)
    config.embedding.enabled = True
    config.embedding.deployment = "isaacus-api"  # SageMaker is the environment's default; the setting wins
    assert prepare_stage_context(STAGE_CONTRACTS["embed"], config).client == "api-client"
    assert dr.resolve("isaacus-api").name == "isaacus-api"
