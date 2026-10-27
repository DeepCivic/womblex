"""Isaacus embeddings API wrapper (kanon-2-embedder).

Thin adapter over ``client.embeddings.create``. Batches into the API's
128-text-per-request limit and returns vectors in input order; retries are the
SDK's own (``max_retries`` on the client). The embedding
*task* matters: documents being indexed use ``retrieval/document``;
search queries use ``retrieval/query`` (see CLAUDE.md "Isaacus task
types matter").

An `EmbeddingProvider` is what the embed stage calls: texts in, one vector each,
in order. It declares its `EmbeddingLimits`, and the stage sends at most
`max_texts_per_request` texts per call. Providers are the `embedding` registry
slot (`embedding.model`; plugins through `womblex.models.embedding`), built by
`make_embedding_provider`; `kanon-2-embedder` (`IsaacusEmbedding`) is the default,
and its `isaacus` trait answers the pre-launch gates before anything is built.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from womblex.providers.isaacus.embedding import (  # noqa: F401 -- re-exported for one release
    IsaacusEmbedding,
    make_isaacus_embedding,
)
from womblex.utils.model_registry import SLOT_EMBEDDING, record_use, register, resolve
from womblex.utils.token_packer import TokenCounter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.config import EmbeddingConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "kanon-2-embedder"
DEFAULT_TASK = "retrieval/document"
MAX_TEXTS_PER_REQUEST = 128


def embed_texts(
    texts: list[str],
    client: object,
    *,
    model: str = DEFAULT_MODEL,
    task: str | None = DEFAULT_TASK,
    dimensions: int | None = None,
) -> list[list[float]]:
    """Embed ``texts`` via the Isaacus embedding API, preserving order.

    Splits into ``MAX_TEXTS_PER_REQUEST`` chunks. Each text must contain a
    non-whitespace character (the caller filters empties). Raises
    ``RuntimeError`` if a request fails.
    """
    out: list[list[float] | None] = [None] * len(texts)
    for start in range(0, len(texts), MAX_TEXTS_PER_REQUEST):
        batch = texts[start : start + MAX_TEXTS_PER_REQUEST]
        resp = _embed_batch(batch, client, model=model, task=task, dimensions=dimensions)
        for emb in resp.embeddings:  # type: ignore[attr-defined]
            out[start + emb.index] = emb.embedding
    if any(v is None for v in out):
        raise RuntimeError("embedding API returned fewer vectors than inputs")
    return out  # type: ignore[return-value]


def _embed_batch(
    texts: list[str], client: object, *, model: str, task: str | None, dimensions: int | None,
) -> object:
    kwargs: dict = {"model": model, "texts": texts}
    if task is not None:
        kwargs["task"] = task
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    try:
        return client.embeddings.create(**kwargs)  # type: ignore[attr-defined]
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        raise RuntimeError(f"Embedding failed: {e}") from e


# ---------------------------------------------------------------------------
# Provider seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingLimits:
    """What a provider declares, so the stage batches requests by it and `plan` counts its tokens.

    `token_counter` is the provider's own (`count_batch(texts) -> list[int]`); a provider that
    declares none has its texts and requests counted, not its tokens.
    """

    max_texts_per_request: int
    needs_network: bool
    token_counter: TokenCounter | None = None


class EmbeddingProvider(Protocol):
    name: str
    limits: EmbeddingLimits

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in input order."""
        ...


register(SLOT_EMBEDDING, DEFAULT_MODEL, make_isaacus_embedding, traits={"isaacus": True},
         deployments=("isaacus-api", "isaacus-sagemaker"), vendor_model=DEFAULT_MODEL)


def calls_isaacus(model: str) -> bool:
    """Whether the embedding provider registered as *model* is served by Isaacus; `False` if unknown."""
    try:
        return bool(resolve(SLOT_EMBEDDING, model).traits.get("isaacus"))
    except ValueError:
        return False  # the model check reports the unknown name


def make_embedding_provider(config: EmbeddingConfig, client: object = None) -> EmbeddingProvider:
    """Build the provider `config.model` names; *client* is used only by an Isaacus provider."""
    entry = resolve(SLOT_EMBEDDING, config.model)
    provider: EmbeddingProvider = entry.factory(config=config, client=client)
    record_use(entry)
    return provider
