"""The contracts and the worker gate on the deployments a stage's models are served by."""

from __future__ import annotations

import hashlib
import json

import pytest

from tests.test_cloud import _minimal_config
from tests.test_enrich_packing import _FakeCounter
from womblex.analyse.enrich import ProviderLimits
from womblex.cloud import dbos_app
from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.cloud.worker import WORKER_SCOPES, served_queues
from womblex.utils import deployment_registry as dr
from womblex.utils import model_check
from womblex.utils import model_registry as mr
from womblex.utils.deployment_registry import Deployment
from womblex.utils.model_registry import SLOT_ENRICHMENT, register

REMOTE, LEGACY, LOCAL = "test-remote-ner", "test-legacy-isaacus-ner", "test-local-gates-ner"


class _Provider:
    name = "x"
    limits = ProviderLimits(max_texts_per_request=1, token_ceiling=None, token_counter=_FakeCounter(),  # type: ignore[arg-type]
                            needs_network=True, returns_document=False)

    def enrich(self, texts):
        return []



@pytest.fixture(autouse=True)
def _providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the test providers for one test only, so no other test sees a deployment it lacks."""
    model_check._register_builtins()  # import the built-ins first: they register once per process, not per copy
    monkeypatch.setattr(mr, "_entries", {s: dict(m) for s, m in mr._entries.items()})
    monkeypatch.setattr(mr, "_aliases", {s: dict(m) for s, m in mr._aliases.items()})
    register(SLOT_ENRICHMENT, REMOTE, lambda **_: _Provider(), source="tests", deployments=("acme",),
             vendor_model="acme/ner-v2")
    register(SLOT_ENRICHMENT, LOCAL, lambda **_: _Provider(), source="tests")
    register(SLOT_ENRICHMENT, LEGACY, lambda **_: _Provider(), source="tests", traits={"isaacus": True})


def _config(tmp_path, model: str):
    config = _minimal_config(tmp_path)
    config.enrichment.enabled = True
    config.enrichment.model = model
    return config


def test_a_model_on_a_non_isaacus_deployment_is_gated_and_checked_through_it(tmp_path) -> None:
    config = _config(tmp_path, REMOTE)
    assert STAGE_CONTRACTS["enrich"].requires_deployment(config)
    assert STAGE_CONTRACTS["enrich"].models(config) == (REMOTE,)
    (use,) = model_check.deployed_models(config, (model_check.SCOPE_ENRICH,))
    assert (use.slot, use.name, use.deployment) == (SLOT_ENRICHMENT, REMOTE, "acme")  # not recorded as Isaacus
    assert use.vendor_model == "acme/ner-v2"  # what the deployment is handed
    # a deployed model is not also built and loaded as if it ran locally
    assert not [u for u in model_check.configured_models(config, (model_check.SCOPE_ENRICH,)) if u.name == REMOTE]


def test_a_plugin_with_only_the_isaacus_trait_is_still_served_by_the_isaacus_deployments(tmp_path) -> None:
    config = _config(tmp_path, LEGACY)
    assert STAGE_CONTRACTS["enrich"].requires_deployment(config)
    (use,) = model_check.deployed_models(config, (model_check.SCOPE_ENRICH,))
    assert (use.slot, use.deployment) == ("isaacus", "isaacus-api")


def test_a_local_model_needs_no_deployment(tmp_path) -> None:
    config = _config(tmp_path, LOCAL)
    assert not STAGE_CONTRACTS["enrich"].requires_deployment(config)
    assert not STAGE_CONTRACTS["chunk"].requires_deployment(config)  # plain chunking is offline
    assert model_check.deployed_models(config) == []


def test_the_queue_key_for_an_unchanged_config_is_the_one_rows_already_queued_carry(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
    config = _config(tmp_path, "kanon-2-enricher")
    config.embedding.enabled = True
    # the recipe the key has always had: sorted (slot, name, options) of the models the scope uses
    ident = [("isaacus", "kanon-2-enricher", "{}")]
    assert model_check.models_key(config, "enrich") == hashlib.sha256(json.dumps(ident).encode()).hexdigest()[:10]
    assert model_check.models_key(config, "enrich") == "8bec191542"


def test_a_worker_whose_deployment_fails_still_serves_extraction_but_not_the_stage(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dr, "_deployments", dict(dr._deployments))
    dr.register(Deployment(name="acme", check=lambda m: (_ for _ in ()).throw(ConnectionError("down")),
                           needs_network=True, billed=True))
    config = _config(tmp_path, REMOTE)
    config.embedding.enabled = False
    check = model_check.check_models(config, "load", scopes=WORKER_SCOPES)
    queues = served_queues(check, "s3://inbox", config)
    assert dbos_app.extract_queue("s3://inbox", "") in queues
    assert not any(q.startswith(dbos_app.stage_queue("enrich")) for q in queues)
    (bad,) = (c for c in check.failures if c.source == "acme")  # the embedder has no key here, which is its own failure
    assert (bad.slot, bad.name) == (SLOT_ENRICHMENT, REMOTE) and "ConnectionError: down" in bad.reason


def test_an_unknown_deployment_name_on_a_model_fails_its_check_naming_the_known_ones(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path, REMOTE)
    monkeypatch.setattr(dr, "_deployments", {})
    monkeypatch.setattr(dr, "_loaded", True)
    result = model_check.check_models(config, "load", scopes=(model_check.SCOPE_ENRICH,))
    (bad,) = result.failures
    assert "unknown deployment: 'acme'" in bad.reason
    with pytest.raises(model_check.ModelCheckError):
        result.raise_for_failures((model_check.SCOPE_ENRICH,))


def test_a_stage_on_a_non_isaacus_deployment_is_gated_on_that_deployment_not_on_isaacus(tmp_path, monkeypatch) -> None:
    from womblex.cloud.stage_runner import StagePreconditionError, prepare_stage_context
    from womblex.utils import availability

    monkeypatch.setattr(dr, "_deployments", dict(dr._deployments))
    dr.register(Deployment(name="acme", check=lambda m: None, needs_network=True, billed=True,
                           env=(("ACME_KEY",),)))
    monkeypatch.setattr(availability, "isaacus_available", lambda: False)
    monkeypatch.delenv("ACME_KEY", raising=False)
    config = _config(tmp_path, REMOTE)
    with pytest.raises(StagePreconditionError, match="needs deployment 'acme' \\(ACME_KEY unset\\)"):
        prepare_stage_context(STAGE_CONTRACTS["enrich"], config)
    monkeypatch.setenv("ACME_KEY", "set")
    with pytest.raises(StagePreconditionError) as exc:  # past the gate: B3 still builds the Isaacus client
        prepare_stage_context(STAGE_CONTRACTS["enrich"], config)
    assert "needs" not in str(exc.value)
