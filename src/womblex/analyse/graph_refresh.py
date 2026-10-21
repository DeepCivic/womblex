"""Offline graph refresh: link mentions to chunks after chunking.

The corpus asset chunks *after* enriching (AI chunking reuses the persisted
ILGS Document), so the ``*.graph_edges.parquet`` written at enrich time cannot
carry mention→chunk edges — no chunks existed yet — and the
``*.enrichment_entities.parquet`` mentions all have ``chunk_index = -1``. This
stage closes that gap once chunks exist, without touching either file.

Both inputs carry character offsets into the same reassembled narrative
(entity mention spans; chunk ``start_char``/``end_char``), so the refresh is
**deterministic and API-free**: for every narrative mention it finds the
overlapping chunk(s) and writes them to ``*.graph_links.parquet``, stamped with
the digest of the chunks it linked against (`store/graph_links_output.py`).
Readers (`analyse/query.py`) use the links while that digest matches.

Idempotent: recomputing from the same offsets yields the same result, so a
re-run (or a resume) is safe, and it never skips on file existence alone — a
rechunk leaves a links file behind that no longer matches. Table mentions are
not linked: their offsets index a table's own markdown.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from womblex.process.chunk_stage import _batch_bases
from womblex.store.checkpoint import CheckpointManager
from womblex.store.enrichment_output import read_enrichment_entities
from womblex.store.evidence import TABLE_LAYER
from womblex.store.graph_links_output import read_matching_links, write_graph_links
from womblex.store.output import chunks_path_for, read_chunks, read_manifest

logger = logging.getLogger(__name__)

@dataclass
class GraphRefreshResult:
    batches_written: int
    docs_refreshed: int
    links_written: int


def refresh_graph_edges(
    shard_dir: Path,
    *,
    checkpoint_mgr: CheckpointManager | None = None,
) -> GraphRefreshResult:
    """Write ``*.graph_links.parquet`` for every batch in ``shard_dir``.

    Needs both ``*.enrichment_entities.parquet`` and ``*.chunks.parquet``
    siblings. Batches missing either are skipped (logged). When
    ``checkpoint_mgr`` is provided, a batch whose docs are all checkpointed is
    skipped on resume only while its links still match its chunks: after a
    rechunk it is linked again.
    """
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")

    bases = _batch_bases(shard_dir)
    if not bases:
        logger.warning("refresh_graph_edges: no batches found in %s", shard_dir)
        return GraphRefreshResult(0, 0, 0)

    batches_written = 0
    docs_refreshed = 0
    links_written = 0

    for base in bases:
        if (checkpoint_mgr is not None and _all_docs_checkpointed(base, checkpoint_mgr)
                and read_matching_links(base) is not None):
            logger.info("refresh_graph_edges: skipping %s (all docs checkpointed, links match)", base.stem)
            continue

        entities_table = read_enrichment_entities(base)
        if entities_table.num_rows == 0 or not chunks_path_for(base).exists():
            logger.info(
                "refresh_graph_edges: %s has no entities or no chunks sidecar — skipping",
                base.stem,
            )
            continue

        links = _link_rows(entities_table.to_pylist(), _narrative_chunks(base))
        write_graph_links(links, base)
        docs = len({r["source_hash"] for r in links if r["chunk_indices"]})

        batches_written += 1
        docs_refreshed += docs
        links_written += len(links)

        if checkpoint_mgr is not None:
            doc_ids = _doc_ids(base)
            if doc_ids:
                checkpoint_mgr.update(
                    doc_ids=doc_ids, succeeded=docs, failed=0,
                    batch_num=int(base.stem.replace("batch-", "") or 0),
                )

        logger.info("refresh_graph_edges: %s refreshed %d docs, %d mention links", base.stem, docs, len(links))

    return GraphRefreshResult(
        batches_written=batches_written,
        docs_refreshed=docs_refreshed,
        links_written=links_written,
    )


def _link_rows(
    entity_rows: list[dict], chunks_by_hash: dict[str, list[tuple[int, int, int]]],
) -> list[dict]:
    """The `*.graph_links.parquet` rows: one per narrative mention, with every chunk it overlaps.

    ``chunk_index`` is the first overlapping chunk (-1 when none, or no chunks for the document).
    """
    rows: list[dict] = []
    for r in entity_rows:
        if r.get("text_layer") == TABLE_LAYER:
            continue  # offsets index a table's own markdown, not the narrative chunks are cut from
        overlaps = _overlapping_chunks(r["mention_start"], r["mention_end"], chunks_by_hash.get(r["source_hash"], []))
        rows.append({
            "source_hash": r["source_hash"], "entity_id": r["entity_id"],
            "mention_start": r["mention_start"], "mention_end": r["mention_end"],
            "chunk_index": overlaps[0] if overlaps else -1, "chunk_indices": overlaps,
        })
    return rows


def _overlapping_chunks(
    start: int, end: int, chunks: list[tuple[int, int, int]],
) -> list[int]:
    """Chunk indices whose [start_char, end_char) overlaps [start, end).

    ``chunks`` is ``(chunk_index, start_char, end_char)`` pre-sorted by
    chunk_index, so the returned list is ordered and ``[0]`` is the first
    overlapping chunk.
    """
    return [ci for ci, s, e in chunks if s < end and start < e]


# ---------------------------------------------------------------------------
# Sidecar readers
# ---------------------------------------------------------------------------


def _narrative_chunks(base: Path) -> dict[str, list[tuple[int, int, int]]]:
    """Per-source narrative chunks as ``(chunk_index, start_char, end_char)``.

    Only narrative chunks share the mention coordinate space (table chunks
    carry offsets in their own markdown), so only those are linked. Sorted by
    chunk_index for stable "first overlapping chunk" selection.
    """
    by_hash: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for row in read_chunks(base).to_pylist():
        if row["content_type"] != "narrative":
            continue
        by_hash[row["source_hash"]].append(
            (row["chunk_index"], row["start_char"], row["end_char"]),
        )
    for mentions in by_hash.values():
        mentions.sort(key=lambda t: t[0])
    return dict(by_hash)


def _doc_ids(base: Path) -> list[str]:
    try:
        return list(read_manifest(base).column("doc_id").to_pylist())
    except Exception:
        return []


def _all_docs_checkpointed(base: Path, mgr: CheckpointManager) -> bool:
    doc_ids = _doc_ids(base)
    return bool(doc_ids) and all(d in mgr.state.processed_ids for d in doc_ids)


__all__ = ["GraphRefreshResult", "refresh_graph_edges"]
