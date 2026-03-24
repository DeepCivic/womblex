"""Parquet output for enrichment results and graph data.

Provides two output schemas:
- **entities.parquet**: Flat entity mentions for fast filtering queries.
- **graph_edges.parquet**: Relationship edges for graph reconstruction.

Document-level enrichment metadata is appended to the existing
documents Parquet via ``enrichment_metadata_columns()``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.analyse.graph import DocumentGraph
from womblex.analyse.models import EnrichmentResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity mention schema (flat, for Parquet-based filtering)
# ---------------------------------------------------------------------------

ENTITY_SCHEMA = pa.schema([
    ("document_id", pa.string()),
    ("entity_id", pa.string()),
    ("entity_label", pa.string()),       # person | location | term | external_document
    ("name", pa.string()),
    ("entity_type", pa.string()),        # natural, corporate, politic, country, state, etc.
    ("role", pa.string()),               # seller, buyer, other, etc. (persons only)
    ("mention_start", pa.int32()),
    ("mention_end", pa.int32()),
    ("chunk_index", pa.int32()),         # -1 if not mapped to a chunk
])

# ---------------------------------------------------------------------------
# Graph edge schema (for relationship reconstruction)
# ---------------------------------------------------------------------------

GRAPH_EDGE_SCHEMA = pa.schema([
    ("document_id", pa.string()),
    ("source_id", pa.string()),
    ("target_id", pa.string()),
    ("relation", pa.string()),
    ("prop_key", pa.string()),
    ("prop_value", pa.string()),
])

# ---------------------------------------------------------------------------
# Document enrichment metadata schema
# ---------------------------------------------------------------------------

ENRICHMENT_META_SCHEMA = pa.schema([
    ("document_id", pa.string()),
    ("doc_type_enriched", pa.string()),  # statute | regulation | decision | contract | other
    ("jurisdiction", pa.string()),
    ("title", pa.string()),
    ("segment_count", pa.int32()),
    ("person_count", pa.int32()),
    ("location_count", pa.int32()),
    ("term_count", pa.int32()),
    ("external_doc_count", pa.int32()),
    ("date_count", pa.int32()),
    ("heading_count", pa.int32()),
    ("junk_span_count", pa.int32()),
])


# ---------------------------------------------------------------------------
# Serialisation: entity mentions
# ---------------------------------------------------------------------------


def _entity_mentions_from_enrichment(
    document_id: str,
    enrichment: EnrichmentResult,
    chunks: list[object] | None = None,
) -> list[dict[str, Any]]:
    """Extract flat entity mention rows from an enrichment result."""
    from womblex.analyse.graph import _find_chunks_for_span

    rows: list[dict[str, Any]] = []
    chunk_list = chunks or []

    # Persons
    for per in enrichment.persons:
        name = per.name.decode(enrichment.text)
        for mention in per.mentions:
            chunk_indices = _find_chunks_for_span(mention, chunk_list) if chunk_list else []  # type: ignore[arg-type]
            rows.append({
                "document_id": document_id,
                "entity_id": per.id,
                "entity_label": "person",
                "name": name,
                "entity_type": per.type,
                "role": per.role,
                "mention_start": mention.start,
                "mention_end": mention.end,
                "chunk_index": chunk_indices[0] if chunk_indices else -1,
            })

    # Locations
    for loc in enrichment.locations:
        name = loc.name.decode(enrichment.text)
        for mention in loc.mentions:
            chunk_indices = _find_chunks_for_span(mention, chunk_list) if chunk_list else []  # type: ignore[arg-type]
            rows.append({
                "document_id": document_id,
                "entity_id": loc.id,
                "entity_label": "location",
                "name": name,
                "entity_type": loc.type,
                "role": "",
                "mention_start": mention.start,
                "mention_end": mention.end,
                "chunk_index": chunk_indices[0] if chunk_indices else -1,
            })

    # Terms
    for term in enrichment.terms:
        name = term.name.decode(enrichment.text)
        for mention in term.mentions:
            chunk_indices = _find_chunks_for_span(mention, chunk_list) if chunk_list else []  # type: ignore[arg-type]
            rows.append({
                "document_id": document_id,
                "entity_id": term.id,
                "entity_label": "term",
                "name": name,
                "entity_type": "",
                "role": "",
                "mention_start": mention.start,
                "mention_end": mention.end,
                "chunk_index": chunk_indices[0] if chunk_indices else -1,
            })

    # External documents
    for exd in enrichment.external_documents:
        name = exd.name.decode(enrichment.text)
        for mention in exd.mentions:
            chunk_indices = _find_chunks_for_span(mention, chunk_list) if chunk_list else []  # type: ignore[arg-type]
            rows.append({
                "document_id": document_id,
                "entity_id": exd.id,
                "entity_label": "external_document",
                "name": name,
                "entity_type": exd.type,
                "role": "",
                "mention_start": mention.start,
                "mention_end": mention.end,
                "chunk_index": chunk_indices[0] if chunk_indices else -1,
            })

    return rows


# ---------------------------------------------------------------------------
# Serialisation: graph edges
# ---------------------------------------------------------------------------


def _graph_edges_to_rows(
    document_id: str,
    graph: DocumentGraph,
) -> list[dict[str, Any]]:
    """Flatten graph edges to rows, one per edge-property pair."""
    rows: list[dict[str, Any]] = []
    for edge in graph.edges:
        if edge.properties:
            for key, value in edge.properties.items():
                rows.append({
                    "document_id": document_id,
                    "source_id": edge.source,
                    "target_id": edge.target,
                    "relation": edge.relation,
                    "prop_key": key,
                    "prop_value": str(value) if value is not None else "",
                })
        else:
            rows.append({
                "document_id": document_id,
                "source_id": edge.source,
                "target_id": edge.target,
                "relation": edge.relation,
                "prop_key": "",
                "prop_value": "",
            })
    return rows


# ---------------------------------------------------------------------------
# Serialisation: enrichment metadata
# ---------------------------------------------------------------------------


def _enrichment_meta_row(
    document_id: str,
    enrichment: EnrichmentResult,
) -> dict[str, Any]:
    """Build a single enrichment metadata row."""
    title = enrichment.title.decode(enrichment.text) if enrichment.title else ""
    return {
        "document_id": document_id,
        "doc_type_enriched": enrichment.type,
        "jurisdiction": enrichment.jurisdiction or "",
        "title": title,
        "segment_count": len(enrichment.segments),
        "person_count": len(enrichment.persons),
        "location_count": len(enrichment.locations),
        "term_count": len(enrichment.terms),
        "external_doc_count": len(enrichment.external_documents),
        "date_count": len(enrichment.dates),
        "heading_count": len(enrichment.headings),
        "junk_span_count": len(enrichment.junk),
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_entity_mentions(
    results: list[tuple[str, EnrichmentResult, list[object] | None]],
    output_path: Path,
) -> Path:
    """Write entity mentions to a Parquet file.

    Args:
        results: List of (document_id, EnrichmentResult, chunks) tuples.
        output_path: Destination Parquet file path.

    Returns:
        The output path written.
    """
    all_rows: list[dict[str, Any]] = []
    for doc_id, enrichment, chunks in results:
        all_rows.extend(_entity_mentions_from_enrichment(doc_id, enrichment, chunks))

    if not all_rows:
        table = pa.table(
            {f.name: pa.array([], type=f.type) for f in ENTITY_SCHEMA},
            schema=ENTITY_SCHEMA,
        )
    else:
        table = pa.Table.from_pylist(all_rows, schema=ENTITY_SCHEMA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))
    logger.info("Wrote %d entity mentions to %s", len(all_rows), output_path)
    return output_path


def write_graph_edges(
    graphs: list[tuple[str, DocumentGraph]],
    output_path: Path,
) -> Path:
    """Write graph edges to a Parquet file.

    Args:
        graphs: List of (document_id, DocumentGraph) tuples.
        output_path: Destination Parquet file path.

    Returns:
        The output path written.
    """
    all_rows: list[dict[str, Any]] = []
    for doc_id, graph in graphs:
        all_rows.extend(_graph_edges_to_rows(doc_id, graph))

    if not all_rows:
        table = pa.table(
            {f.name: pa.array([], type=f.type) for f in GRAPH_EDGE_SCHEMA},
            schema=GRAPH_EDGE_SCHEMA,
        )
    else:
        table = pa.Table.from_pylist(all_rows, schema=GRAPH_EDGE_SCHEMA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))
    logger.info("Wrote %d graph edges to %s", len(all_rows), output_path)
    return output_path


def write_enrichment_metadata(
    results: list[tuple[str, EnrichmentResult]],
    output_path: Path,
) -> Path:
    """Write enrichment metadata to a Parquet file.

    Args:
        results: List of (document_id, EnrichmentResult) tuples.
        output_path: Destination Parquet file path.

    Returns:
        The output path written.
    """
    rows = [_enrichment_meta_row(doc_id, enrichment) for doc_id, enrichment in results]

    if not rows:
        table = pa.table(
            {f.name: pa.array([], type=f.type) for f in ENRICHMENT_META_SCHEMA},
            schema=ENRICHMENT_META_SCHEMA,
        )
    else:
        table = pa.Table.from_pylist(rows, schema=ENRICHMENT_META_SCHEMA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))
    logger.info("Wrote %d enrichment metadata rows to %s", len(rows), output_path)
    return output_path
