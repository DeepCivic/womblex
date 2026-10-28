"""An embedding provider over `/v1/embeddings`, for any OpenAI-compatible server.

A thin adapter over `httpx`: no retry loop of Womblex's own (retries and limits are the server's), no
dependency added. The server's model id is `embedding.provider_options.model`. Roles have no equivalent
here, so `document` and `query` are both ignored; a `dimensions` setting is sent and the width of what
comes back is checked, so a server that ignores it is a batch error, not a silently different width.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.analyse.embed import EmbeddingLimits
    from womblex.config import EmbeddingConfig

DEFAULT_MAX_TEXTS = 64


def _limits(config: EmbeddingConfig) -> EmbeddingLimits:
    from womblex.analyse.embed import EmbeddingLimits
    from womblex.providers.openai_compat.deployment import billed_by_setting

    return EmbeddingLimits(
        max_texts_per_request=int(config.provider_options.get("max_texts_per_request", DEFAULT_MAX_TEXTS)),
        needs_network=billed_by_setting(),
    )


@dataclass
class OpenAICompatEmbedding:
    """Embeddings from `POST {base}/embeddings` on the shared client."""

    client: Any
    config: EmbeddingConfig
    limits: EmbeddingLimits = field(init=False)
    name: str = "openai-compatible"

    def __post_init__(self) -> None:
        self.limits = _limits(self.config)

    @property
    def vendor_model(self) -> str:
        return str(self.config.provider_options["model"])

    def embed(self, texts: list[str], role: str | None = None) -> list[list[float]]:
        body: dict[str, Any] = {"model": self.vendor_model, "input": texts}
        if self.config.dimensions is not None:
            body["dimensions"] = self.config.dimensions
        response = self.client.post("/embeddings", json=body)
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda r: r.get("index", 0))
        vectors = [list(r["embedding"]) for r in rows]
        if len(vectors) != len(texts):
            raise RuntimeError(f"the server returned {len(vectors)} vectors for {len(texts)} texts")
        want = self.config.dimensions
        if want is not None and any(len(v) != want for v in vectors):
            raise RuntimeError(f"the server returned vectors of width {len(vectors[0])}, not the {want} asked for")
        return vectors


def make_openai_compat_embedding(*, config: EmbeddingConfig, client: object = None, **_: object) -> OpenAICompatEmbedding:
    """The `embedding` slot factory for `openai-compatible`; *client* is the deployment's shared one."""
    return OpenAICompatEmbedding(client, config)
