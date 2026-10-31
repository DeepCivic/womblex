"""run_enrichment — call the enrichment provider on completed documents (requires chunks)."""

from __future__ import annotations

import logging

from womblex.config import WomblexConfig
from womblex.operations.models import BatchResult, DocumentResult

logger = logging.getLogger(__name__)


def run_enrichment(
    results: list[DocumentResult],
    config: WomblexConfig,
    client: object,
) -> None:
    """Enrich completed documents through the provider ``config.enrichment.model`` names.

    Failures are logged per-document but do not halt the batch. Results are
    stored on ``DocumentResult.enrichment`` and ``.graph``.

    Args:
        results: DocumentResults with chunks populated.
        config: Pipeline configuration (uses ``config.enrichment``).
        client: An ``isaacus.Isaacus`` client, used only by an Isaacus provider.
    """
    from womblex.analyse.enrich import make_enrichment_provider
    from womblex.analyse.graph import build_document_graph

    enrich_cfg = config.enrichment
    if not enrich_cfg.enabled:
        return
    # Before any document, so an unregistered `enrichment.model` is refused up front.
    provider = make_enrichment_provider(enrich_cfg, client)

    for dr in results:
        if dr.status != "completed" or not dr.extraction:
            continue

        full_text = dr.extraction.full_text
        if enrich_cfg.skip_short_documents > 0 and len(full_text) < enrich_cfg.skip_short_documents:
            logger.debug(
                "Skipping enrichment for %s (too short: %d chars)", dr.doc_id, len(full_text)
            )
            continue

        try:
            enrichment = provider.enrich([full_text])[0][0]
            dr.enrichment = enrichment
            dr.graph = build_document_graph(
                document_id=dr.doc_id,
                enrichment=enrichment,
                chunks=dr.chunks or None,
            )
            logger.info(
                "Enriched %s: %d segments, %d persons, %d locations, %d nodes, %d edges",
                dr.doc_id,
                len(enrichment.segments),
                len(enrichment.persons),
                len(enrichment.locations),
                len(dr.graph.nodes),
                len(dr.graph.edges),
            )
        except Exception as e:
            logger.error("Enrichment failed for %s: %s", dr.doc_id, e)


def enrich_batch(
    batch: BatchResult,
    config: WomblexConfig,
    client: object,
) -> None:
    """Enrich a batch via the Isaacus API.

    Thin wrapper around ``run_enrichment`` for backward compatibility.
    """
    run_enrichment(batch.results, config, client)
