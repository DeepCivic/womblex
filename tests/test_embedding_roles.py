"""Embedding roles (`document` / `query`) and the `dimensions` capability, per provider."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_model_check import _config
from womblex.analyse.embed import (
    EmbeddingLimits,
    capability_problems,
    embed_with_role,
    role_of,
)
from womblex.config import EmbeddingConfig
from womblex.providers.isaacus.embedding import IsaacusEmbedding
from womblex.utils import model_registry as mr
from womblex.utils.model_check import SCOPE_EMBED, check_models, check_registered


@pytest.mark.parametrize(("task", "role"), [
    ("retrieval/document", "document"), ("retrieval/query", "query"),
    ("document", "document"), ("query", "query"), (None, None), ("classification", None),
])
def test_a_task_is_a_role_or_none(task, role) -> None:
    assert role_of(task) == role


def test_the_config_accepts_the_short_names_as_aliases_of_the_isaacus_tasks() -> None:
    assert EmbeddingConfig(task="query").task == "retrieval/query"
    assert EmbeddingConfig(task="document").task == "retrieval/document"
    assert EmbeddingConfig(task="retrieval/query").task == "retrieval/query"
    assert EmbeddingConfig(task=None).task is None
    assert EmbeddingConfig().task == "retrieval/document"  # the default is unchanged


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(embeddings=[SimpleNamespace(index=i, embedding=[0.1]) for i, _ in enumerate(kw["texts"])])


def test_the_isaacus_adapter_maps_a_role_to_its_task_and_otherwise_uses_the_configured_one() -> None:
    client = _Recorder()
    provider = IsaacusEmbedding(client, EmbeddingConfig(task="retrieval/document"))
    provider.embed(["a"], role="query")
    provider.embed(["b"])
    provider.embed(["c"], role="document")
    assert [c["task"] for c in client.calls] == ["retrieval/query", "retrieval/document", "retrieval/document"]


def test_a_provider_whose_embed_takes_no_role_is_called_without_one() -> None:
    class Old:
        name = "old"
        limits = EmbeddingLimits(8, needs_network=False)

        def embed(self, texts):
            return [[1.0] for _ in texts]

    class New(Old):
        def embed(self, texts, role=None):
            return [[1.0 if role == "query" else 0.0] for _ in texts]

    assert embed_with_role(Old(), ["x"], "document") == [[1.0]]  # type: ignore[arg-type]
    assert embed_with_role(New(), ["x"], "query") == [[1.0]]  # type: ignore[arg-type]
    assert embed_with_role(New(), ["x"], None) == [[0.0]]  # type: ignore[arg-type]


def test_a_dimensions_setting_the_provider_cannot_honour_is_refused_before_launch(monkeypatch) -> None:
    monkeypatch.setattr(mr, "_entries", {s: dict(m) for s, m in mr._entries.items()})
    monkeypatch.setattr(mr, "_aliases", {s: dict(m) for s, m in mr._aliases.items()})
    mr.register(mr.SLOT_EMBEDDING, "fixed-width", lambda **_: None, source="tests")
    mr.register(mr.SLOT_EMBEDDING, "any-width", lambda **_: None, source="tests", traits={"dimensions": True})
    config = _config(embedding={"enabled": True, "model": "fixed-width", "dimensions": 256})
    ((scope, _slot, model, message),) = capability_problems(config)
    assert (scope, model) == ("embed", "fixed-width") and "cannot return vectors of a requested width" in message
    with pytest.raises(ValueError, match="remove embedding.dimensions"):
        check_registered(config)  # what `POST /v1/runs` and `dispatch-workflow` call
    result = check_models(config, "load", scopes=(SCOPE_EMBED,))
    assert any("requested width" in c.reason for c in result.failures)
    for model in ("any-width", "kanon-2-embedder"):
        ok = _config(embedding={"enabled": True, "model": model, "dimensions": 256})
        assert capability_problems(ok) == []


def test_no_dimensions_setting_is_never_a_problem() -> None:
    assert capability_problems(_config(embedding={"model": "kanon-2-embedder"})) == []
    assert capability_problems(_config(embedding={"model": "no-such-model", "dimensions": 8})) == []  # reported by its own check


from tests.test_embed_stage import shard_with_chunks  # noqa: F401


@pytest.mark.parametrize(("task", "expected"), [
    ("retrieval/query", "query"), ("document", "document"), (None, None)])
def test_the_stage_passes_the_role_the_task_names(shard_with_chunks, monkeypatch, task, expected) -> None:  # noqa: F811
    from womblex.analyse import embed_stage

    seen: list = []

    class P:
        name = "p"
        limits = EmbeddingLimits(8, needs_network=False)

        def embed(self, texts, role=None):
            seen.append(role)
            return [[0.5] for _ in texts]

    monkeypatch.setattr(embed_stage, "make_embedding_provider", lambda *a, **k: P())
    result = embed_stage.embed_shards(shard_with_chunks, EmbeddingConfig(task=task), client=None)
    assert result.chunks_embedded >= 1 and set(seen) == {expected}
