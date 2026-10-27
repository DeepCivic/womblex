"""The service-backed model check: free connectivity at `load`, a warned billed request at `smoke`,
and a worker's Isaacus check limited to the stages its config enables."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from tests.test_model_check import _PAID, _clean, _config, fake_isaacus  # noqa: F401
from womblex.utils.model_check import SCOPE_EMBED, SCOPE_ENRICH, check_models


@pytest.fixture
def hosted(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A hosted-API deployment with a key; records each connectivity request's URL."""
    import httpx

    monkeypatch.setenv("ISAACUS_API_KEY", "k")  # pragma: allowlist secret -- a test placeholder
    monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
    monkeypatch.delenv("ISAACUS_BASE_URL", raising=False)
    urls: list[str] = []
    monkeypatch.setattr(httpx, "get", lambda url, **_: urls.append(url) or SimpleNamespace(status_code=404))
    return urls


def test_load_checks_connectivity_with_no_billed_request(fake_isaacus, hosted) -> None:  # noqa: F811
    made = fake_isaacus()
    result = check_models(_config(**_PAID), "load", scopes=(SCOPE_ENRICH, SCOPE_EMBED))
    assert result.ok and made == []  # no client built, so nothing billed
    assert hosted == ["https://api.isaacus.com/v1"]  # one reachability request for the deployment, not per model


@pytest.mark.parametrize("why", ["no key", "unreachable"])
def test_load_fails_without_a_key_or_an_answer(fake_isaacus, hosted, monkeypatch, why) -> None:  # noqa: F811
    import httpx

    if why == "no key":
        monkeypatch.delenv("ISAACUS_API_KEY")
    else:
        monkeypatch.setattr(httpx, "get", lambda *_a, **_k: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    (bad,) = check_models(_config(**_PAID), "load", scopes=(SCOPE_ENRICH,)).failures
    assert ("ISAACUS_API_KEY is not set" if why == "no key" else "refused") in bad.reason


@pytest.mark.parametrize(("status", "ok"), [("InService", True), ("Creating", False)])
def test_load_on_sagemaker_asks_whether_the_endpoint_is_in_service(monkeypatch, status, ok) -> None:
    import boto3

    monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "enrich-001@ap-southeast-2=kanon-2-enricher")
    asked: list[tuple[str, str]] = []

    class Session:
        region_name = None

        def __init__(self, **_: object) -> None:
            pass

        def client(self, service: str, region_name: str) -> object:
            def describe_endpoint(EndpointName: str) -> dict:
                asked.append((service, EndpointName))
                return {"EndpointStatus": status}
            return SimpleNamespace(describe_endpoint=describe_endpoint)

    monkeypatch.setattr(boto3, "Session", Session)
    result = check_models(_config(enrichment={"enabled": True}), "load", scopes=(SCOPE_ENRICH,))
    assert result.ok is ok and asked == [("sagemaker", "enrich-001")]


def test_smoke_warns_before_each_billed_request(fake_isaacus, caplog, monkeypatch) -> None:  # noqa: F811
    from womblex.providers.isaacus import client

    made = fake_isaacus()
    warned_first: list[bool] = []
    real = client.make_api_client  # the fixture's recording factory

    def make():  # the warning is already logged when the billed request goes
        warned_first.append(sum("billed" in r.getMessage() for r in caplog.records) == len(made) + 1)
        return real()

    monkeypatch.setattr(client, "make_api_client", make)
    with caplog.at_level(logging.WARNING):
        assert check_models(_config(**_PAID), "smoke", scopes=(SCOPE_ENRICH, SCOPE_EMBED)).ok
    assert warned_first == [True, True] and len(made) == 2


def test_a_billed_ping_is_warned_before_it_is_sent(caplog) -> None:
    from womblex.providers.bedrock.ocr import MistralOCRReader
    from womblex.utils import model_registry as reg
    from womblex.utils.model_check import SCOPE_EXTRACT

    class Remote:
        ping_is_billed = True

        def ping(self) -> None:
            assert any("billed" in r.getMessage() for r in caplog.records)

    reg.register(reg.SLOT_OCR, "remote-ocr", lambda **_: Remote(), source="d")
    cfg = _config(extraction={"ocr": {"engine": "remote-ocr"}}, redaction={"enabled": False})
    with caplog.at_level(logging.WARNING):
        assert check_models(cfg, "load", scopes=(SCOPE_EXTRACT,)).ok
    assert MistralOCRReader.ping_is_billed  # Bedrock's one-token Converse request is billed


def test_a_worker_checks_no_isaacus_model_for_a_disabled_stage_and_does_not_serve_it(fake_isaacus, hosted) -> None:  # noqa: F811
    from womblex.cloud import dbos_app
    from womblex.cloud.worker import WORKER_SCOPES, served_queues
    from womblex.utils.model_check import enabled_scopes, models_key

    made = fake_isaacus()
    config = _config(enrichment={"enabled": True})  # embedding stays disabled
    check = check_models(config, "smoke", scopes=WORKER_SCOPES, isaacus_scopes=enabled_scopes(config))
    assert [c.name for c in check.checks if c.slot == "isaacus"] == ["kanon-2-enricher"] and len(made) == 1
    served = served_queues(check, "s3://inbox", config)
    assert dbos_app.stage_queue("enrich", models_key(config, "enrich")) in served
    assert not any(q.startswith(dbos_app.stage_queue("embed")) for q in served)  # never checked: not served
