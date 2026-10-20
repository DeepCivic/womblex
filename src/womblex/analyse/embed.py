"""Isaacus embeddings API wrapper (kanon-2-embedder).

Thin adapter over ``client.embeddings.create``. Batches into the API's
128-text-per-request limit, handles 429 rate-limit retries with
exponential backoff, and returns vectors in input order. The embedding
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
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from womblex.utils.model_registry import SLOT_EMBEDDING, record_use, register, resolve
from womblex.utils.token_packer import TokenCounter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.config import EmbeddingConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "kanon-2-embedder"
DEFAULT_TASK = "retrieval/document"
MAX_TEXTS_PER_REQUEST = 128
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 2.0


def embed_texts(
    texts: list[str],
    client: object,
    *,
    model: str = DEFAULT_MODEL,
    task: str | None = DEFAULT_TASK,
    dimensions: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> list[list[float]]:
    """Embed ``texts`` via the Isaacus embedding API, preserving order.

    Splits into ``MAX_TEXTS_PER_REQUEST`` chunks. Each text must contain a
    non-whitespace character (the caller filters empties). Raises
    ``RuntimeError`` if a request fails after all retries.
    """
    out: list[list[float] | None] = [None] * len(texts)
    for start in range(0, len(texts), MAX_TEXTS_PER_REQUEST):
        batch = texts[start : start + MAX_TEXTS_PER_REQUEST]
        resp = _embed_batch(
            batch, client, model=model, task=task, dimensions=dimensions,
            max_retries=max_retries, retry_base_delay=retry_base_delay,
        )
        for emb in resp.embeddings:  # type: ignore[attr-defined]
            out[start + emb.index] = emb.embedding
    if any(v is None for v in out):
        raise RuntimeError("embedding API returned fewer vectors than inputs")
    return out  # type: ignore[return-value]


def _embed_batch(
    texts: list[str], client: object, *, model: str, task: str | None,
    dimensions: int | None, max_retries: int, retry_base_delay: float,
) -> object:
    kwargs: dict = {"model": model, "texts": texts}
    if task is not None:
        kwargs["task"] = task
    if dimensions is not None:
        kwargs["dimensions"] = dimensions

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return client.embeddings.create(**kwargs)  # type: ignore[attr-defined]
        except Exception as e:
            last_error = e
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                delay = retry_base_delay * (2 ** attempt)
                logger.warning(
                    "Rate limited on embeddings (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, max_retries + 1, delay,
                )
                time.sleep(delay)
                continue
            logger.error("Embedding failed: %s", e)
            raise RuntimeError(f"Embedding failed: {e}") from e
    raise RuntimeError(
        f"Embedding failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error


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


@dataclass
class IsaacusEmbedding:
    """Kanon-2 embeddings through the Isaacus SDK, under `EmbeddingConfig`'s task and dimensions."""

    client: object
    config: EmbeddingConfig
    limits: EmbeddingLimits = EmbeddingLimits(MAX_TEXTS_PER_REQUEST, needs_network=True,
                                              token_counter=TokenCounter())  # Kanon-2's own, offline
    name: str = "isaacus"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(
            texts, self.client,
            model=self.config.model,
            task=self.config.task,
            dimensions=self.config.dimensions,
            max_retries=self.config.max_retries,
            retry_base_delay=self.config.retry_base_delay,
        )


def _make_isaacus(*, config: EmbeddingConfig, client: object = None, **_: object) -> IsaacusEmbedding:
    return IsaacusEmbedding(client, config)


register(SLOT_EMBEDDING, DEFAULT_MODEL, _make_isaacus, traits={"isaacus": True})


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
