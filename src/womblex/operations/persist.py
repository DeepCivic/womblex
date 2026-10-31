"""Parquet output helpers for batch results (E2E composition path)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from womblex.ingest.extract import ExtractionResult
from womblex.operations.models import BatchResult, DocumentResult
from womblex.store.run_stamp import RunStamp
from womblex.store.source_provenance import IngestProvenance

logger = logging.getLogger(__name__)


def _failed_extraction(r: DocumentResult) -> ExtractionResult:
    """What a document that failed before any extraction exists writes: no elements, the error, and the
    document type detection had reached, if any."""
    method = r.profile.doc_type.value if r.profile is not None else ""
    return ExtractionResult(method=method, error=r.error or "extraction failed", document_id=r.doc_id)


def result_rows(batch: BatchResult) -> list[tuple[str, str, ExtractionResult]]:
    """One ``(doc_id, path, extraction)`` per result, failed documents included, in batch order.

    A failure is written as a ``status: error`` manifest row, never dropped, so every document the
    batch was given is accounted for in its shard.
    """
    return [
        (r.doc_id, str(r.path), r.extraction if r.extraction is not None else _failed_extraction(r))
        for r in batch.results
        if r.status in ("completed", "error")
    ]


def write_batch_parquet(
    batch: BatchResult,
    output_path: Path,
    *,
    provenance: IngestProvenance | None = None,
    stamp: RunStamp | None = None,
    batch_num: int = 0,
    redaction_regions: bool = False,
) -> Path | None:
    """Write the batch's results to Parquet, failed documents as ``status: error`` rows.

    ``provenance`` names the declared ingest root the documents were read
    from; it reaches the manifest columns and every shard's footer. ``stamp``
    names the run that produced them, in the same footer. ``batch_num`` keys the
    placeholder hash of a source that could not be read (``store.manifest_rows``).
    ``redaction_regions`` is true when redaction detection ran over the batch: the regions it
    found are written beside the shard (``*.redaction_regions.parquet``, empty if there were none).
    """
    from womblex.store.output import write_results

    rows = result_rows(batch)
    if not rows:
        logger.warning("No results to write to Parquet")
        return None

    written = write_results(rows, output_path, provenance=provenance, stamp=stamp, batch_num=batch_num)
    if redaction_regions:
        _write_regions(rows, written)
    return written


def _write_regions(rows: list[tuple[str, str, ExtractionResult]], shard_path: Path) -> None:
    """Write the regions the batch's redaction reports hold, keyed by the shard's own `source_hash`es.

    The manifest has one row per result, in order, so the hashes pair with *rows* by position.
    """
    from womblex.redact.stage import region_rows
    from womblex.store.output import read_manifest
    from womblex.store.redaction_regions_output import write_redaction_regions
    from womblex.store.run_stamp import sidecar_footer

    hashes = read_manifest(shard_path).column("source_hash").to_pylist()
    regions = [
        region
        for source_hash, (_, _, extraction) in zip(hashes, rows, strict=True)
        if extraction.redaction_report is not None
        for region in region_rows(source_hash, extraction.redaction_report)
    ]
    write_redaction_regions(regions, shard_path, metadata=sidecar_footer(shard_path, "redact"))


def write_batch_enrichment(batch: BatchResult, output_dir: Path) -> dict[str, Path | None]:
    """Write enrichment results and graph data to Parquet."""
    from womblex.store.enrichment_output import (
        write_enrichment_metadata,
        write_entity_mentions,
        write_graph_edges,
    )

    entity_rows: list[tuple[str, Any, list[object] | None]] = []
    graph_rows: list[tuple[str, Any]] = []
    meta_rows: list[tuple[str, Any]] = []

    for r in batch.results:
        if r.enrichment is None:
            continue
        entity_rows.append((r.doc_id, r.enrichment, r.chunks or None))  # type: ignore[arg-type]
        meta_rows.append((r.doc_id, r.enrichment))
        if r.graph is not None:
            graph_rows.append((r.doc_id, r.graph))

    paths: dict[str, Path | None] = {
        "entities": None,
        "graph_edges": None,
        "enrichment_meta": None,
    }

    if entity_rows:
        paths["entities"] = write_entity_mentions(entity_rows, output_dir / "entities.parquet")
    if graph_rows:
        paths["graph_edges"] = write_graph_edges(graph_rows, output_dir / "graph_edges.parquet")
    if meta_rows:
        paths["enrichment_meta"] = write_enrichment_metadata(
            meta_rows, output_dir / "enrichment_meta.parquet"
        )

    return paths
