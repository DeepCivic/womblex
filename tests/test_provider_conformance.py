"""Every registered model meets its slot's interface; a package's models are tested by registering them.

The cases are read from the registries, so a new built-in is covered by registering it and an
installed plugin is covered the same way (`tests/_fixture_vendor.py` stands in for one).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests import _fixture_vendor
from womblex.config import EmbeddingConfig, EnrichmentConfig
from womblex.utils import deployment_registry as dr
from womblex.utils import model_registry as mr
from womblex.utils.model_check import _register_builtins

_register_builtins()

#: Options a factory cannot be built without. A model needing others fails its case, naming this table.
REQUIRED_OPTIONS: dict[tuple[str, str], dict[str, Any]] = {
    (mr.SLOT_TOKENIZER, "huggingface"): {"name": "isaacus/kanon-2-tokenizer"},
    (mr.SLOT_SPELLFIX_DICTIONARY, "hunspell"): {"name": "en_AU"},
}


def build(slot: str, name: str) -> Any:
    """The model registered as *name* in *slot*, built as a run builds it (nothing is called on it)."""
    options = REQUIRED_OPTIONS.get((slot, name), {})
    if slot == mr.SLOT_OCR:
        from womblex.ingest.paddle_ocr import get_ocr_reader

        return get_ocr_reader(name, **options)
    if slot == mr.SLOT_LAYOUT:
        from womblex.ingest.paddle_ocr import get_layout_analyzer

        return get_layout_analyzer(name, **options)
    if slot == mr.SLOT_ENRICHMENT:
        from womblex.analyse.enrich import make_enrichment_provider

        return make_enrichment_provider(EnrichmentConfig(model=name))
    if slot == mr.SLOT_EMBEDDING:
        from womblex.analyse.embed import make_embedding_provider

        return make_embedding_provider(EmbeddingConfig(model=name))
    return mr.resolve(slot, name).factory(**options)


def check_conforms(slot: str, name: str) -> None:
    """Assert the built model meets *slot*'s interface (`docs/model-plugins.md`, "Slots")."""
    try:
        model = build(slot, name)
    except TypeError as e:
        raise AssertionError(f"{slot} {name!r} needs options to be built; add them to REQUIRED_OPTIONS ({e})") from e
    if slot == mr.SLOT_OCR:
        assert callable(model.read_page)
    elif slot == mr.SLOT_LAYOUT:
        assert callable(model.analyze)
    elif slot == mr.SLOT_PII_CONTEXT:
        assert callable(model.encode)
    elif slot == mr.SLOT_TOKENIZER:
        assert (isinstance(model, str) and model) or callable(model)
    elif slot == mr.SLOT_SPELLFIX_DICTIONARY:
        assert callable(model.lookup)
    elif slot == mr.SLOT_ENRICHMENT:
        assert isinstance(model.name, str) and callable(model.enrich)
        limits = model.limits
        assert limits.max_texts_per_request >= 1
        assert limits.token_ceiling is None or limits.token_ceiling >= 1
        assert callable(limits.token_counter.count_batch)
        assert isinstance(limits.needs_network, bool) and isinstance(limits.returns_document, bool)
    elif slot == mr.SLOT_EMBEDDING:
        assert isinstance(model.name, str) and callable(model.embed)
        assert model.limits.max_texts_per_request >= 1 and isinstance(model.limits.needs_network, bool)
        counter = model.limits.token_counter
        assert counter is None or callable(counter.count_batch)
    else:  # pragma: no cover - a new slot needs its interface written here
        raise AssertionError(f"no conformance check for slot {slot!r}")


def check_declarations(slot: str, name: str) -> None:
    """Assert what the registry holds about the model: traits a dict, deployments registered."""
    entry = mr.resolve(slot, name)
    assert isinstance(entry.traits, dict)
    assert set(entry.deployments) <= set(dr.known_deployments()), f"{slot} {name}: unregistered deployment"


BUILT_INS = [(slot, name) for slot in mr.SLOTS for name in mr.known_names(slot)]


@pytest.mark.parametrize(("slot", "name"), BUILT_INS, ids=[f"{s}:{n}" for s, n in BUILT_INS])
def test_every_registered_model_meets_its_slots_interface(slot: str, name: str) -> None:
    check_declarations(slot, name)
    check_conforms(slot, name)


def test_every_slot_has_a_registered_model_to_test() -> None:
    assert {slot for slot, _ in BUILT_INS} == set(mr.SLOTS)


def test_a_model_that_breaks_its_slots_interface_fails_its_case(monkeypatch) -> None:
    monkeypatch.setattr(mr, "_entries", {s: dict(m) for s, m in mr._entries.items()})
    monkeypatch.setattr(mr, "_aliases", {s: dict(m) for s, m in mr._aliases.items()})
    mr.register(mr.SLOT_OCR, "broken-reader", lambda **_: object(), source="tests")
    with pytest.raises(AttributeError):
        check_conforms(mr.SLOT_OCR, "broken-reader")
    mr.register(mr.SLOT_EMBEDDING, "names-no-deployment", lambda **_: object(), source="tests",
                deployments=("not-registered",))
    with pytest.raises(AssertionError, match="unregistered deployment"):
        check_declarations(mr.SLOT_EMBEDDING, "names-no-deployment")


def test_a_package_registering_one_deployment_and_models_in_two_slots_is_picked_up(monkeypatch) -> None:
    _fixture_vendor.install(monkeypatch)
    assert _fixture_vendor.NAME in dr.known_deployments()
    for slot, name in _fixture_vendor.MODELS:
        assert name in mr.known_names(slot)
        check_declarations(slot, name)
        check_conforms(slot, name)
    assert dr.distribution_version(dr.resolve(_fixture_vendor.NAME)) == ("fixture-vendor-dist", "1.2.3")
    from tests.test_deployment_registry import assert_conforms

    assert_conforms(dr.resolve(_fixture_vendor.NAME))
    entry = mr.resolve(mr.SLOT_EMBEDDING, _fixture_vendor.EMBED)
    assert (entry.deployments, entry.vendor_model, entry.source) == (
        (_fixture_vendor.NAME,), _fixture_vendor.VENDOR_EMBED, "fixture-vendor-dist")
