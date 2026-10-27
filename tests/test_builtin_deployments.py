"""The four built-in deployments, and the models registered against them."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from womblex.providers.isaacus import client as isaacus_client
from womblex.providers.isaacus.deployment import smoke_call
from womblex.utils import deployment_registry as dr
from womblex.utils import model_registry as mr


def test_the_built_ins_are_registered() -> None:
    assert {"isaacus-api", "isaacus-sagemaker", "bedrock", "ollama"} <= set(dr.known_deployments())


def test_the_default_models_name_the_deployments_that_serve_them() -> None:
    import womblex.analyse.embed
    import womblex.analyse.enrich
    import womblex.ingest.paddle_ocr  # noqa: F401

    for slot, name in ((mr.SLOT_ENRICHMENT, "kanon-2-enricher"), (mr.SLOT_EMBEDDING, "kanon-2-embedder")):
        entry = mr.resolve(slot, name)
        assert entry.deployments == ("isaacus-api", "isaacus-sagemaker") and entry.vendor_model == name
    assert mr.resolve(mr.SLOT_OCR, "mistral-ocr").deployments == ("bedrock",)
    assert mr.resolve(mr.SLOT_OCR, "ollama").deployments == ("ollama",)
    assert mr.resolve(mr.SLOT_OCR, "paddleocr").deployments == ()  # local: no deployment


def test_the_hosted_api_needs_its_key_and_serves_every_model(monkeypatch: pytest.MonkeyPatch) -> None:
    d = dr.resolve("isaacus-api")
    monkeypatch.delenv("ISAACUS_API_KEY", raising=False)
    assert d.unset() == [("ISAACUS_API_KEY",)]
    assert d.serves("any-model")
    with pytest.raises(ValueError, match="ISAACUS_API_KEY is not set"):
        d.check(["kanon-2-embedder"])


def test_sagemaker_serves_what_the_environment_declares(monkeypatch: pytest.MonkeyPatch) -> None:
    d = dr.resolve("isaacus-sagemaker")
    monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
    assert d.unset() == [("ISAACUS_SAGEMAKER_ENDPOINTS",)]
    assert not d.serves("kanon-2-embedder")  # no endpoint, nothing served
    monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "enrich-001=kanon-2-enricher")
    assert d.unset() == []
    assert d.serves("kanon-2-enricher") and not d.serves("kanon-2-embedder")


def test_the_sagemaker_check_refuses_a_model_no_endpoint_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    d = dr.resolve("isaacus-sagemaker")
    monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS", raising=False)
    with pytest.raises(ValueError, match="declares no SageMaker endpoint"):
        d.check(["kanon-2-embedder"])
    monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "enrich-001=kanon-2-enricher")
    with pytest.raises(ValueError, match="serves kanon-2-embedder"):
        d.check(["kanon-2-embedder"])


def test_check_reachable_still_picks_the_deployment_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(isaacus_client, "check_sagemaker_reachable", lambda *m: seen.append(f"sagemaker:{m[0]}"))
    monkeypatch.setattr(isaacus_client, "check_api_reachable", lambda timeout=10.0: seen.append("api"))
    monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "e-001")
    isaacus_client.check_reachable("kanon-2-embedder")
    monkeypatch.delenv("ISAACUS_SAGEMAKER_ENDPOINTS")
    isaacus_client.check_reachable("kanon-2-embedder")
    assert seen == ["sagemaker:kanon-2-embedder", "api"]


def _client(embedding: list[float] | None = None, results: int = 1) -> MagicMock:
    client = MagicMock()
    client.embeddings.create.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(embedding=embedding)] if embedding is not None else [])
    client.enrichments.create.return_value = SimpleNamespace(results=[object()] * results)
    return client


def test_the_smoke_call_sends_the_slots_request_and_leaves_the_shared_client_open() -> None:
    c = _client(embedding=[0.1])
    smoke_call(c, mr.SLOT_EMBEDDING, "kanon-2-embedder")
    assert c.embeddings.create.call_args.kwargs["model"] == "kanon-2-embedder"
    c = _client()
    smoke_call(c, mr.SLOT_ENRICHMENT, "kanon-2-enricher")
    c.enrichments.create.assert_called_once()
    c.close.assert_not_called()


@pytest.mark.parametrize("slot,client", [
    (mr.SLOT_EMBEDDING, _client(embedding=[])),
    (mr.SLOT_ENRICHMENT, _client(results=0)),
])
def test_a_smoke_call_that_returns_nothing_fails(slot: str, client: MagicMock) -> None:
    with pytest.raises(RuntimeError, match="returned no"):
        smoke_call(client, slot, "m")


@pytest.mark.parametrize("deployment,module,cls", [
    ("bedrock", "womblex.providers.bedrock.ocr", "MistralOCRReader"),
    ("ollama", "womblex.providers.ollama.ocr", "OllamaOCRReader"),
])
def test_the_ocr_deployments_check_by_pinging_their_reader_for_the_model_asked(
    monkeypatch: pytest.MonkeyPatch, deployment: str, module: str, cls: str,
) -> None:
    import importlib

    reader = getattr(importlib.import_module(module), cls)
    pinged: list[str] = []
    monkeypatch.setattr(reader, "ping", lambda self: pinged.append(getattr(self, "model_id", None) or self.model))
    dr.resolve(deployment).check(["some-model"])
    assert pinged == ["some-model"]


def test_a_failing_ocr_ping_fails_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    from womblex.providers.ollama.ocr import OllamaOCRReader

    def refuse(self: OllamaOCRReader) -> None:
        raise ConnectionError("refused")

    monkeypatch.setattr(OllamaOCRReader, "ping", refuse)
    with pytest.raises(ConnectionError):
        dr.resolve("ollama").check(["llama3.2-vision"])
