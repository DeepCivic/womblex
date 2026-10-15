"""The `enrichment` slot: a provider other than Isaacus's runs offline, with no Isaacus gate."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.test_cloud import _minimal_config
from tests.test_enrich_packing import _fake_document, _FakeClient, _FakeCounter, _make_shard
from womblex.analyse.enrich import ProviderLimits, _convert_document, calls_isaacus, effective_chunking
from womblex.analyse.enrich_stage import enrich_shards
from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.config import EnrichmentConfig
from womblex.operations import DocumentResult, run_enrichment
from womblex.pipeline_order import enabled_downstream_stages
from womblex.store.enrichment_doc import enrichment_doc_path_for
from womblex.store.enrichment_output import read_enrichment_entities
from womblex.utils import model_check
from womblex.utils.model_registry import DEFAULT_MODELS, SLOT_ENRICHMENT, is_default_group, register

LOCAL = "test-local-ner"


class _LocalProvider:
    """No network, no Document, two texts per request."""

    name = LOCAL

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.limits = ProviderLimits(max_texts_per_request=2, token_ceiling=10_000,
                                     token_counter=_FakeCounter(),  # type: ignore[arg-type]
                                     needs_network=False, returns_document=False)

    def enrich(self, texts: list[str]) -> list[tuple[Any, object | None]]:
        self.calls.append(list(texts))
        return [(_convert_document(_fake_document(t)), None) for t in texts]


_built: list[_LocalProvider] = []


def _factory(**_: object) -> _LocalProvider:
    _built.append(_LocalProvider())
    return _built[-1]


register(SLOT_ENRICHMENT, LOCAL, _factory, source="tests")


def _config(tmp_path, model: str):
    config = _minimal_config(tmp_path)
    config.enrichment.enabled = True
    config.enrichment.model = model
    config.chunking.chunking_model = "kanon-2-enricher"
    return config


def test_the_default_provider_is_isaacus_kanon_2() -> None:
    assert DEFAULT_MODELS[SLOT_ENRICHMENT] == EnrichmentConfig().model == "kanon-2-enricher"
    assert is_default_group({SLOT_ENRICHMENT: "kanon-2-enricher"})
    assert not is_default_group({SLOT_ENRICHMENT: LOCAL})
    assert calls_isaacus("kanon-2-enricher") and not calls_isaacus(LOCAL)


def test_a_local_provider_enriches_without_a_client_and_persists_no_document(tmp_path) -> None:
    shard = _make_shard(tmp_path, [{"version_id": f"d{i}", "text": "a b c"} for i in range(3)])
    result = enrich_shards(shard.parent, EnrichmentConfig(model=LOCAL), client=None,
                           persist_document=True)
    assert result.docs_enriched == 3
    assert [len(c) for c in _built[-1].calls] == [2, 1]  # packed by the provider's own limit
    assert read_enrichment_entities(shard).num_rows == 3
    assert not enrichment_doc_path_for(shard).exists()


def test_the_isaacus_provider_still_persists_its_document(tmp_path) -> None:
    shard = _make_shard(tmp_path, [{"version_id": "d0", "text": "a b c"}])
    client = _FakeClient()
    create = client.enrichments.create

    def with_json(**kw: Any) -> Any:  # the SDK Document serialises itself
        resp = create(**kw)
        for r in resp.results:
            r.document.model_dump_json = lambda: "{}"
        return resp

    client.enrichments.create = with_json  # type: ignore[method-assign]
    enrich_shards(shard.parent, EnrichmentConfig(), client=client,
                  token_counter=_FakeCounter(), persist_document=True)
    assert enrichment_doc_path_for(shard).exists()


@pytest.mark.parametrize(("model", "isaacus"), [("kanon-2-enricher", True), (LOCAL, False)])
def test_the_contracts_key_off_the_provider(tmp_path, model: str, isaacus: bool) -> None:
    config = _config(tmp_path, model)
    enrich, chunk = STAGE_CONTRACTS["enrich"], STAGE_CONTRACTS["chunk"]
    assert enrich.requires_isaacus_api(config) is isaacus
    assert enrich.models(config) == ((model,) if isaacus else ())
    assert any(s.endswith("enrichment_doc.parquet") for s in enrich.outputs(config)) is isaacus
    assert any(c.suffix.endswith("enrichment_doc.parquet")
               for c in chunk.conditional_inputs(config)) is isaacus


@pytest.mark.parametrize(("model", "isaacus"), [("kanon-2-enricher", True), (LOCAL, False)])
def test_the_model_check_pings_isaacus_and_loads_anything_else(tmp_path, model: str, isaacus: bool) -> None:
    config = _config(tmp_path, model)
    scopes = (model_check.SCOPE_ENRICH,)
    assert [u.name for u in model_check.isaacus_models(config, scopes)] == ([model] if isaacus else [])
    slots = [u.slot for u in model_check.configured_models(config, scopes)]
    assert (SLOT_ENRICHMENT in slots) is not isaacus
    if not isaacus:
        result = model_check.check_models(config, "smoke", scopes=scopes)
        assert result.ok and [c.slot for c in result.checks] == [SLOT_ENRICHMENT]


def test_an_unknown_provider_is_refused_before_launch(tmp_path) -> None:
    config = _config(tmp_path, "no-such-enricher")
    with pytest.raises(ValueError, match="unknown enrichment model"):
        model_check.check_registered(config)
    assert not model_check.check_models(config, "load", scopes=(model_check.SCOPE_ENRICH,)).ok


@pytest.mark.parametrize(("model", "ai"), [("kanon-2-enricher", True), (LOCAL, False)])
def test_without_a_document_to_reuse_chunking_is_local_semchunk(tmp_path, caplog, model: str, ai: bool) -> None:
    """AI chunking reuses the enricher's Document; without one it is off, never a paid call of its own."""
    config = _config(tmp_path, model)
    config.chunking.enabled = True
    assert STAGE_CONTRACTS["chunk"].requires_isaacus_api(config) is ai
    assert [u.name for u in model_check.isaacus_models(config, (model_check.SCOPE_CHUNK,))] == (
        ["kanon-2-enricher"] if ai else [])
    assert ("graph-refresh" in enabled_downstream_stages(config)) is ai
    assert effective_chunking(config).chunking_model == ("kanon-2-enricher" if ai else None)
    assert ("set aside" in caplog.text) is not ai


def test_run_enrichment_goes_through_the_slot(tmp_path) -> None:
    """The in-memory path calls the configured provider, and refuses an unknown one before any document."""
    doc = DocumentResult(path=Path("d.pdf"), doc_id="d", status="completed",
                         extraction=SimpleNamespace(full_text="a b c"))  # type: ignore[arg-type]
    run_enrichment([doc], _config(tmp_path, LOCAL), client=None)
    assert doc.enrichment is not None and _built[-1].calls == [["a b c"]]
    with pytest.raises(ValueError, match="unknown enrichment model"):
        run_enrichment([doc], _config(tmp_path, "no-such-enricher"), client=None)
