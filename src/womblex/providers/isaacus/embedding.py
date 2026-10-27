"""Kanon-2 embeddings through the Isaacus SDK, the `kanon-2-embedder` adapter.

The provider protocol, `EmbeddingLimits` and the request batching stay in
`analyse/embed.py`, which registers this adapter; they are imported inside the
functions that use them so the two modules never import each other at load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from womblex.providers.isaacus.client import with_retries

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.analyse.embed import EmbeddingLimits
    from womblex.config import EmbeddingConfig


def _kanon_limits() -> EmbeddingLimits:
    from womblex.analyse.embed import MAX_TEXTS_PER_REQUEST, EmbeddingLimits
    from womblex.utils.token_packer import TokenCounter

    return EmbeddingLimits(MAX_TEXTS_PER_REQUEST, needs_network=True,
                           token_counter=TokenCounter())  # Kanon-2's own, offline


@dataclass
class IsaacusEmbedding:
    """Kanon-2 embeddings through the Isaacus SDK, under `EmbeddingConfig`'s task and dimensions."""

    client: object
    config: EmbeddingConfig
    limits: EmbeddingLimits = field(default_factory=_kanon_limits)
    name: str = "isaacus"

    def embed(self, texts: list[str]) -> list[list[float]]:
        from womblex.analyse.embed import embed_texts

        return embed_texts(
            texts, self.client,
            model=self.config.model,
            task=self.config.task,
            dimensions=self.config.dimensions,
        )


def make_isaacus_embedding(*, config: EmbeddingConfig, client: object = None, **_: object) -> IsaacusEmbedding:
    """The `embedding` slot factory for `kanon-2-embedder`."""
    return IsaacusEmbedding(with_retries(client, config.max_retries), config)
