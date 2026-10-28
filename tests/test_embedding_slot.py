"""The `embedding` slot: requests batched by the provider's limit; only Isaacus is gated on Isaacus."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_cloud import _minimal_config
from tests.test_embed_stage import shard_with_chunks  # noqa: F401
from womblex.analyse.embed import EmbeddingLimits, calls_isaacus
from womblex.analyse.embed_stage import embed_shards, plan_embedding
from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.config import EmbeddingConfig
from womblex.store.embed_output import read_embeddings
from womblex.utils import model_check
from womblex.utils.model_registry import DEFAULT_MODELS, SLOT_EMBEDDING, is_default_group, register

LOCAL = "test-local-embedder"


class _LocalProvider:
    name = LOCAL
    limits = EmbeddingLimits(max_texts_per_request=1, needs_network=False)

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]


_built: list[_LocalProvider] = []


def _factory(**_: object) -> _LocalProvider:
    _built.append(_LocalProvider())
    return _built[-1]


register(SLOT_EMBEDDING, LOCAL, _factory, source="tests")


class _FakeIsaacus:
    """`client.embeddings.create` returning one 3-d vector per text, by index."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, **kw: object) -> SimpleNamespace:
        self.calls.append(kw)
        texts = kw["texts"]
        return SimpleNamespace(embeddings=[
            SimpleNamespace(index=i, embedding=[0.1, 0.2, 0.3]) for i in range(len(texts))  # type: ignore[arg-type]
        ])


def test_the_default_provider_is_isaacus_kanon_2() -> None:
    assert DEFAULT_MODELS[SLOT_EMBEDDING] == EmbeddingConfig().model == "kanon-2-embedder"
    assert is_default_group({SLOT_EMBEDDING: "kanon-2-embedder"})
    assert calls_isaacus("kanon-2-embedder") and not calls_isaacus(LOCAL)


def test_the_isaacus_provider_sends_the_configured_task(shard_with_chunks) -> None:  # noqa: F811
    client = _FakeIsaacus()
    result = embed_shards(shard_with_chunks, EmbeddingConfig(), client=client)
    assert result.chunks_embedded == 2
    assert [c["task"] for c in client.calls] == ["retrieval/document"]
    rows = read_embeddings(shard_with_chunks).to_pylist()
    assert all(r["model"] == "kanon-2-embedder" and r["dim"] == 3 for r in rows)


def test_a_local_provider_embeds_without_a_client_batched_by_its_limit(shard_with_chunks) -> None:  # noqa: F811
    result = embed_shards(shard_with_chunks, EmbeddingConfig(model=LOCAL), client=None)
    assert result.chunks_embedded == 2
    assert [len(c) for c in _built[-1].calls] == [1, 1]
    assert {r["model"] for r in read_embeddings(shard_with_chunks).to_pylist()} == {LOCAL}
    assert plan_embedding(shard_with_chunks, EmbeddingConfig(model=LOCAL),
                          token_counter=SimpleNamespace(count_batch=lambda ts: [1] * len(ts))).requests == 2


@pytest.mark.parametrize(("model", "isaacus"), [("kanon-2-embedder", True), (LOCAL, False)])
def test_the_contract_and_model_check_key_off_the_provider(tmp_path, model: str, isaacus: bool) -> None:
    config = _minimal_config(tmp_path)
    config.embedding.enabled, config.embedding.model = True, model
    embed = STAGE_CONTRACTS["embed"]
    assert embed.requires_deployment(config) is isaacus
    assert embed.models(config) == ((model,) if isaacus else ())
    scopes = (model_check.SCOPE_EMBED,)
    assert [u.name for u in model_check.deployed_models(config, scopes)] == ([model] if isaacus else [])
    assert (SLOT_EMBEDDING in [u.slot for u in model_check.configured_models(config, scopes)]) is not isaacus
    if not isaacus:
        assert model_check.check_models(config, "smoke", scopes=scopes).ok


def test_an_unknown_embedder_is_refused_before_launch(tmp_path) -> None:
    config = _minimal_config(tmp_path)
    config.embedding.enabled, config.embedding.model = True, "no-such-embedder"
    with pytest.raises(ValueError, match="unknown embedding model"):
        model_check.check_registered(config)


def test_the_isaacus_embedding_adapter_makes_one_call_and_passes_max_retries_to_the_sdk():
    from unittest.mock import MagicMock

    from womblex.analyse.embed import MAX_TEXTS_PER_REQUEST, embed_texts, make_embedding_provider

    client = MagicMock()
    client.embeddings.create.side_effect = Exception("429 rate limit exceeded")
    with pytest.raises(RuntimeError, match="Embedding failed"):
        embed_texts(["a"], client)
    assert client.embeddings.create.call_count == 1  # retries are the SDK's, not a loop here

    provider = make_embedding_provider(EmbeddingConfig(max_retries=4), client)
    client.with_options.assert_called_once_with(max_retries=4)
    assert provider.limits.max_texts_per_request == MAX_TEXTS_PER_REQUEST
