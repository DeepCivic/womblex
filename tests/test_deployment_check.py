"""Deployments are checked once however many models use them, and a failure is named for its model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_model_check import _config
from womblex.utils import deployment_registry as dr
from womblex.utils.deployment_check import check_deployed, check_pinging_model
from womblex.utils.deployment_registry import Deployment
from womblex.utils.model_check import SCOPE_EMBED, SCOPE_ENRICH, CheckLevel, ModelUse, check_models


@pytest.fixture
def acme(monkeypatch: pytest.MonkeyPatch):
    """A registered deployment that records each free check and each smoke call."""
    monkeypatch.setattr(dr, "_deployments", {})
    monkeypatch.setattr(dr, "_loaded", True)
    calls: list[tuple[str, object]] = []

    def check(models):
        calls.append(("check", list(models)))
        if "bad" in models:
            raise PermissionError("model 'bad' is not enabled")

    dr.register(Deployment(
        name="acme", check=check, needs_network=True, billed=True,
        smoke=lambda slot, model: calls.append(("smoke", (slot, model))),
    ))
    return calls


def _use(name: str, scope: str = SCOPE_ENRICH, deployment: str = "acme") -> ModelUse:
    return ModelUse("isaacus", name, {}, (scope,), deployment)


def test_two_models_on_one_deployment_are_checked_by_one_call(acme) -> None:
    checks = check_deployed([_use("m1"), _use("m2", SCOPE_EMBED)], CheckLevel.LOAD)
    assert acme == [("check", ["m1", "m2"])]
    assert [(c.name, c.ok, c.source) for c in checks] == [("m1", True, "acme"), ("m2", True, "acme")]


def test_a_failure_is_attributed_to_the_model_that_caused_it(acme) -> None:
    checks = check_deployed([_use("good"), _use("bad", SCOPE_EMBED)], CheckLevel.LOAD)
    assert [(c.name, c.ok) for c in checks] == [("good", True), ("bad", False)]
    assert checks[1].reason == "PermissionError: model 'bad' is not enabled"
    assert acme == [("check", ["good", "bad"]), ("check", ["good"]), ("check", ["bad"])]


def test_one_model_on_two_scopes_is_checked_once(acme) -> None:
    check_deployed([_use("m1"), _use("m1", "chunk")], CheckLevel.LOAD)
    assert acme == [("check", ["m1"])]


def test_smoke_sends_each_model_its_own_billed_call_and_not_the_free_check(acme, caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        checks = check_deployed([_use("m1"), _use("m2", SCOPE_EMBED)], CheckLevel.SMOKE)
    assert acme == [("smoke", ("enrichment", "m1")), ("smoke", ("embedding", "m2"))]
    assert all(c.ok for c in checks)
    assert sum("billed" in r.getMessage() for r in caplog.records) == 2


def test_a_failing_smoke_call_fails_only_its_model(acme, monkeypatch) -> None:
    def smoke(slot, model):
        if model == "m2":
            raise RuntimeError("401 invalid key")

    dr.register(Deployment(name="acme2", check=lambda m: None, needs_network=True, billed=True, smoke=smoke))
    checks = check_deployed([_use("m1", deployment="acme2"), _use("m2", deployment="acme2")], CheckLevel.SMOKE)
    assert [(c.name, c.ok) for c in checks] == [("m1", True), ("m2", False)]
    assert "401 invalid key" in checks[1].reason


def test_a_deployment_with_no_smoke_call_is_checked_free_at_smoke(acme) -> None:
    dr.register(Deployment(name="free", check=lambda m: acme.append(("free", list(m))), needs_network=False,
                           billed=False))
    (check,) = check_deployed([_use("m1", deployment="free")], CheckLevel.SMOKE)
    assert check.ok and acme == [("free", ["m1"])]


def test_an_unregistered_deployment_fails_its_models_naming_the_known_ones(acme) -> None:
    (check,) = check_deployed([_use("m1", deployment="nope")], CheckLevel.LOAD)
    assert not check.ok and "unknown deployment: 'nope' (known: ['acme'])" in check.reason


def test_sagemaker_names_the_unserved_model_and_still_passes_the_served_one(monkeypatch) -> None:
    import boto3

    monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "enrich-001@ap-southeast-2=kanon-2-enricher")

    class Session:
        region_name = None

        def __init__(self, **_: object) -> None:
            pass

        def client(self, service: str, region_name: str) -> object:
            return SimpleNamespace(describe_endpoint=lambda EndpointName: {"EndpointStatus": "InService"})

    monkeypatch.setattr(boto3, "Session", Session)
    cfg = _config(enrichment={"enabled": True}, embedding={"enabled": True})
    result = check_models(cfg, "load", scopes=(SCOPE_ENRICH, SCOPE_EMBED))
    assert {c.name: c.ok for c in result.checks} == {"kanon-2-enricher": True, "kanon-2-embedder": False}
    (bad,) = result.failures
    assert "does not serve 'kanon-2-embedder'" in bad.reason and bad.source == "isaacus-sagemaker"


def test_a_billed_ping_is_warned_and_an_unbilled_one_is_not(caplog) -> None:
    import logging

    class Billed:
        ping_is_billed = True

        def ping(self) -> None:
            pass

    class Free:
        def ping(self) -> None:
            pass

    with caplog.at_level(logging.WARNING):
        check_pinging_model("ocr", "free", Free())
        assert not caplog.records
        check_pinging_model("ocr", "billed", Billed())
    assert [r.getMessage() for r in caplog.records] == [
        "model check: sending one billed request to ocr 'billed' to verify it"]


def test_a_blank_api_key_is_no_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    from womblex.utils.availability import isaacus_available

    monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
    monkeypatch.setenv("ISAACUS_API_KEY", "   ")
    assert not isaacus_available()
    monkeypatch.setenv("ISAACUS_API_KEY", "k")  # pragma: allowlist secret -- a test placeholder
    assert isaacus_available()
