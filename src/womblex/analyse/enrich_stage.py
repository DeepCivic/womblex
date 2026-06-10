"""Per-stage Kanon-2 enrichment over an existing shard directory.

Consumes ``*.elements.parquet`` + ``*._manifest.parquet`` already written
by the extraction stage and writes ``*.enrichment_entities.parquet`` +
``*.enrichment_meta.parquet`` + ``*.graph_edges.parquet`` siblings per
batch, joinable to the other sidecars on ``source_hash``. The graph-edges
sidecar is the shippable enrichment graph (relations, hierarchy, citations,
and — when a ``*.chunks.parquet`` sibling exists — mention→chunk edges).

Reassembles the document narrative as the ``\\n\\n``-joined text of
TEXT_KINDS elements in ``elem_order`` per source_hash via the shared
:func:`womblex.process.chunker.reassemble_narrative` — the same text the
chunker sees, by construction — and feeds it to the enrichment API.

Enrichment runs one document at a time so a single API failure isolates to
that document (the batch continues), mirroring the corpus-wide "individual
document failures shouldn't stop the batch" policy. The downstream
:mod:`womblex.link.stage` consumes the entities sidecar.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from womblex.analyse.enrich import enrich_document_raw
from womblex.analyse.graph import build_document_graph
from womblex.analyse.models import EnrichmentResult
from womblex.config import EnrichmentConfig
from womblex.ingest.elements import Element
from womblex.process.chunk_stage import _batch_bases
from womblex.process.chunker import TextChunk, reassemble_narrative
from womblex.process.text_overlay import apply_overlay, load_overlay
from womblex.store.checkpoint import CheckpointManager
from womblex.store.enrichment_doc import (
    enrichment_doc_path_for,
    write_enrichment_doc_shard,
)
from womblex.store.enrichment_output import (
    enrichment_entities_path_for,
    graph_edges_path_for,
    write_enrichment_entities_shard,
    write_enrichment_meta_shard,
    write_graph_edges_shard,
)
from womblex.store.output import chunks_path_for, read_chunks, read_elements, read_manifest

logger = logging.getLogger(__name__)


@dataclass
class EnrichStageResult:
    batches_written: int
    docs_enriched: int
    total_entities: int


def enrich_shards(
    shard_dir: Path,
    enrichment_config: EnrichmentConfig,
    *,
    client: object,
    text_source: str = "elements",
    persist_document: bool = False,
    checkpoint_mgr: CheckpointManager | None = None,
) -> EnrichStageResult:
    """Enrich every batch in ``shard_dir`` and write enrichment siblings.

    ``client`` is an ``isaacus.Isaacus`` instance (injected so tests can
    pass a fake). Skips batches whose entities sidecar already exists when
    ``checkpoint_mgr`` is provided and every contained doc is checkpointed.

    When ``persist_document`` is true, also writes a
    ``*.enrichment_doc.parquet`` sibling holding the raw ILGS Document per
    doc (stamped with ``text_source``) so the chunk stage can reuse it for
    semchunk-4 AI chunking without re-enriching (``docs/decisions.md``).
    """
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")

    bases = _batch_bases(shard_dir)
    if not bases:
        logger.warning("enrich_shards: no batches found in %s", shard_dir)
        return EnrichStageResult(0, 0, 0)

    batches_written = 0
    docs_enriched = 0
    total_entities = 0

    for base in bases:
        if checkpoint_mgr is not None and _all_docs_checkpointed(
            base, checkpoint_mgr, persist_document=persist_document,
        ):
            logger.info("enrich_shards: skipping %s (all docs checkpointed)", base.stem)
            continue

        narratives, doc_ids_by_hash = _load_narratives(base, text_source)
        results: list[tuple[str, EnrichmentResult]] = []
        doc_rows: list[tuple[str, str, str]] = []  # (source_hash, text_source, json)
        errored: set[str] = set()

        for source_hash, text in narratives.items():
            if len(text.strip()) < max(1, enrichment_config.skip_short_documents):
                continue  # nothing to enrich — terminal, counts as resolved
            try:
                enr, raw_doc = enrich_document_raw(
                    text, client,
                    model=enrichment_config.model,
                    overflow_strategy=enrichment_config.overflow_strategy,
                    max_retries=enrichment_config.max_retries,
                    retry_base_delay=enrichment_config.retry_base_delay,
                )
            except Exception as e:  # transient (e.g. network) — leave for retry
                logger.error(
                    "enrich_shards: enrichment failed for %s: %s",
                    doc_ids_by_hash.get(source_hash, source_hash), e,
                )
                errored.add(source_hash)
                continue
            results.append((source_hash, enr))
            if persist_document:
                # raw_doc is the SDK ILGS Document (typed object to avoid a hard
                # isaacus import); model_dump_json round-trips it losslessly.
                doc_rows.append((source_hash, text_source, raw_doc.model_dump_json()))  # type: ignore[attr-defined]
            total_entities += (
                len(enr.persons) + len(enr.locations)
                + len(enr.terms) + len(enr.external_documents)
            )

        write_enrichment_entities_shard(results, base)
        write_enrichment_meta_shard(results, base)
        chunks_by_hash = _load_chunks(base)
        write_graph_edges_shard(
            [
                (src, build_document_graph(src, enr, chunks_by_hash.get(src)))
                for src, enr in results
            ],
            base,
        )
        if persist_document:
            write_enrichment_doc_shard(doc_rows, base)
        batches_written += 1
        docs_enriched += len(results)

        if checkpoint_mgr is not None and doc_ids_by_hash:
            # Checkpoint every doc EXCEPT those that errored — an errored doc
            # (transient API/network failure) stays unprocessed so a resume
            # retries it rather than skipping it forever. The batch's sidecar
            # is rewritten on that resume.
            resolved_doc_ids = [
                doc_id for h, doc_id in doc_ids_by_hash.items() if h not in errored
            ]
            if resolved_doc_ids:
                checkpoint_mgr.update(
                    doc_ids=resolved_doc_ids,
                    succeeded=len(results),
                    failed=len(errored),
                    batch_num=int(base.stem.replace("batch-", "") or 0),
                )

        logger.info(
            "enrich_shards: %s enriched %d docs", base.stem, len(results),
        )

    return EnrichStageResult(
        batches_written=batches_written,
        docs_enriched=docs_enriched,
        total_entities=total_entities,
    )


# ---------------------------------------------------------------------------
# Narrative materialisation (text-only — no cells/fields needed for enrichment)
# ---------------------------------------------------------------------------


def _load_narratives(
    base_path: Path, text_source: str = "elements",
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``({source_hash: narrative}, {source_hash: doc_id})`` for a batch.

    When ``text_source`` is not ``'elements'`` the matching element-text overlay
    (normalise / spellfix) is applied before reassembly, so Kanon-2 enriches the
    same repaired text the chunker chunks — keeping mention/chunk offsets aligned.
    """
    try:
        manifest = read_manifest(base_path)
    except Exception:
        return {}, {}
    src_to_doc = dict(zip(
        manifest.column("source_hash").to_pylist(),
        manifest.column("doc_id").to_pylist(),
    ))

    elem_table = read_elements(base_path)
    if elem_table.num_rows == 0:
        return {}, src_to_doc

    by_hash: dict[str, list[Element]] = defaultdict(list)
    for row in elem_table.to_pylist():
        by_hash[row["source_hash"]].append(Element(
            order=row["elem_order"],
            kind=row["kind"],
            extractor=row["extractor"] or "",
            page=row["page"],
            text=row["text"],
        ))

    overrides = load_overlay(base_path, text_source)
    narratives: dict[str, str] = {}
    for src, elems in by_hash.items():
        elems.sort(key=lambda e: e.order)
        apply_overlay(src, elems, overrides)
        text, _ = reassemble_narrative(elems)
        narratives[src] = text
    return narratives, src_to_doc


def _load_chunks(base_path: Path) -> dict[str, list[TextChunk]]:
    """Narrative chunks per source_hash from the batch's chunks sidecar, if present.

    Entity mention spans are offsets into the reassembled narrative, so only
    narrative chunks (which share that coordinate space) feed the graph's
    chunk-mention edges; table chunks carry offsets in their own markdown space.
    """
    if not chunks_path_for(base_path).exists():
        return {}
    try:
        chunk_rows = read_chunks(base_path).to_pylist()
    except Exception as e:  # unreadable sidecar — graph still ships, minus chunk edges
        logger.warning("enrich_shards: unreadable chunks sidecar for %s: %s", base_path.stem, e)
        return {}
    by_hash: dict[str, list[TextChunk]] = defaultdict(list)
    for row in chunk_rows:
        if row["content_type"] != "narrative":
            continue
        by_hash[row["source_hash"]].append(TextChunk(
            text=row["text"],
            start_char=row["start_char"],
            end_char=row["end_char"],
            chunk_index=row["chunk_index"],
            content_type=row["content_type"],
        ))
    return by_hash


def _all_docs_checkpointed(
    base_path: Path, mgr: CheckpointManager, *, persist_document: bool = False,
) -> bool:
    """True if the entities sidecar exists and every manifest doc is checkpointed.

    The graph-edges sidecar must also exist — a batch enriched before graph
    shipping landed is re-enriched on resume so it gains its
    ``*.graph_edges.parquet`` (mirroring ``persist_document``: the graph is
    only buildable from the live EnrichmentResult).

    When ``persist_document`` is requested, the doc sidecar must also exist —
    otherwise a batch enriched before the flag was enabled would be skipped on
    resume and never gain its ``*.enrichment_doc.parquet``.
    """
    if not enrichment_entities_path_for(base_path).exists():
        return False
    if not graph_edges_path_for(base_path).exists():
        return False
    if persist_document and not enrichment_doc_path_for(base_path).exists():
        return False
    try:
        m = read_manifest(base_path)
    except Exception:
        return False
    doc_ids = m.column("doc_id").to_pylist()
    return bool(doc_ids) and all(d in mgr.state.processed_ids for d in doc_ids)


__all__ = ["EnrichStageResult", "enrich_shards"]
