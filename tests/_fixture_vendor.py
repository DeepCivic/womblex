"""A fixture plugin: one deployment and models in two slots, as an installed package supplies them.

It touches nothing but this module: `install(monkeypatch)` presents it through the same two
entry-point groups a real package declares (`womblex.deployments`, `womblex.models.<slot>`), so a
test sees what an installed plugin would give it.
"""

from __future__ import annotations

from typing import Any

import pytest

from womblex.analyse.embed import EmbeddingLimits
from womblex.analyse.enrich import ProviderLimits
from womblex.utils import deployment_registry as dr
from womblex.utils import model_registry as mr
from womblex.utils.deployment_registry import Deployment

NAME = "fixture-vendor"
EMBED = "fixture-vendor-embed"
NER = "fixture-vendor-ner"
VENDOR_EMBED = "fixture-embed-v1"

DEPLOYMENT = Deployment(
    name=NAME, check=lambda models: None, needs_network=True, billed=True,
    env=(("FIXTURE_VENDOR_KEY",),), source="fixture-vendor-dist", version="1.2.3",
)


class _Counter:
    def count_batch(self, texts: list[str]) -> list[int]:
        return [len(t.split()) for t in texts]


class _Embedder:
    name = NAME
    limits = EmbeddingLimits(max_texts_per_request=16, needs_network=True)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]


class _Ner:
    name = NAME
    limits = ProviderLimits(max_texts_per_request=4, token_ceiling=None, token_counter=_Counter(),  # type: ignore[arg-type]
                            needs_network=True, returns_document=False)

    def enrich(self, texts: list[str]) -> list[tuple[Any, object | None]]:
        return []


def make_embedder(*, config: Any, client: object = None, **_: object) -> _Embedder:
    return _Embedder()


def make_ner(*, config: Any, client: object = None, token_counter: object = None, **_: object) -> _Ner:
    return _Ner()


make_embedder.womblex_deployments = (NAME,)  # type: ignore[attr-defined]
make_embedder.womblex_vendor_model = VENDOR_EMBED  # type: ignore[attr-defined]
make_ner.womblex_deployments = (NAME,)  # type: ignore[attr-defined]

#: The models the plugin supplies: (slot, name).
MODELS = ((mr.SLOT_EMBEDDING, EMBED), (mr.SLOT_ENRICHMENT, NER))


def install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the plugin installed for one test: its deployment and both models selectable."""
    from womblex.utils.model_check import _register_builtins

    _register_builtins()  # the built-ins register once per process, so import them before copying
    monkeypatch.setattr(mr, "_entries", {s: dict(m) for s, m in mr._entries.items()})
    monkeypatch.setattr(mr, "_aliases", {s: dict(m) for s, m in mr._aliases.items()})
    monkeypatch.setattr(mr, "_loaded", set())
    dr.known_deployments()  # the built-ins register once per process too
    monkeypatch.setattr(dr, "_deployments", dict(dr._deployments))
    monkeypatch.setattr(dr, "_loaded", False)
    by_group = {
        dr.ENTRY_POINT_GROUP: [(NAME, DEPLOYMENT, "fixture-vendor-dist", "1.2.3")],
        mr.ENTRY_POINT_GROUP_PREFIX + mr.SLOT_EMBEDDING: [(EMBED, make_embedder, "fixture-vendor-dist", "1.2.3")],
        mr.ENTRY_POINT_GROUP_PREFIX + mr.SLOT_ENRICHMENT: [(NER, make_ner, "fixture-vendor-dist", "1.2.3")],
    }
    loader = lambda group: iter(by_group.get(group, []))
    monkeypatch.setattr(mr, "plugin_objects", loader)
    monkeypatch.setattr(dr, "plugin_objects", loader)
