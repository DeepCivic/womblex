"""Kanon-2 enrichment through the Isaacus SDK, the `kanon-2-enricher` adapter.

The provider protocol, `ProviderLimits` and the SDK response conversion stay in
`analyse/enrich.py`, which registers this adapter; they are imported inside the
functions that use them so the two modules never import each other at load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from womblex.providers.isaacus.client import with_retries

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.analyse.enrich import ProviderLimits
    from womblex.analyse.models import EnrichmentResult
    from womblex.config import EnrichmentConfig
    from womblex.utils.token_packer import TokenCounter


@dataclass
class IsaacusEnrichment:
    """Kanon-2 enrichment through the Isaacus SDK.

    The API accepts 8 texts a request and long input whole (`overflow_strategy`), so the
    limits are `max_texts_per_request` and no split ceiling unless `split_ceiling` sets one.
    """

    client: object
    config: EnrichmentConfig
    limits: ProviderLimits
    name: str = "isaacus"

    def enrich(self, texts: list[str]) -> list[tuple[EnrichmentResult, object | None]]:
        from womblex.analyse.enrich import enrich_documents_raw

        return list(enrich_documents_raw(
            texts, self.client,
            model=self.config.model,
            overflow_strategy=self.config.overflow_strategy,
        ))


def isaacus_provider(
    config: EnrichmentConfig, client: object, token_counter: TokenCounter | None = None,
) -> IsaacusEnrichment:
    """The Isaacus provider; *client* may be `None` where only the limits are read (`plan`)."""
    from womblex.analyse.enrich import ProviderLimits
    from womblex.utils.token_packer import TokenCounter

    return IsaacusEnrichment(with_retries(client, config.max_retries), config, ProviderLimits(
        max_texts_per_request=config.max_texts_per_request,
        token_ceiling=config.split_ceiling,
        token_counter=token_counter or TokenCounter(config.tokenizer),
        needs_network=True,
        returns_document=True,
    ))


def make_isaacus_enrichment(
    *, config: EnrichmentConfig, client: object = None, token_counter: TokenCounter | None = None,
    **_: object,
) -> IsaacusEnrichment:
    """The `enrichment` slot factory for `kanon-2-enricher`."""
    return isaacus_provider(config, client, token_counter)
