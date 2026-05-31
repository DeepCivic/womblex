"""Per-stage entity linking over an existing shard directory.

Consumes ``*.enrichment_entities.parquet`` (written by the enrich stage),
selects candidate mentions by configured kind, resolves them against a
reference register, and writes ``*.entity_links.parquet`` siblings at
mention/span grain. Mirrors :mod:`womblex.process.chunk_stage`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from womblex.config import LinkingConfig
from womblex.link.matcher import Candidate, Link, resolve
from womblex.link.reference import ReferenceTable, load_reference
from womblex.process.chunk_stage import _batch_bases
from womblex.store.checkpoint import CheckpointManager
from womblex.store.enrichment_output import (
    enrichment_entities_path_for,
    read_enrichment_entities,
)
from womblex.store.output import entity_links_path_for, read_manifest, write_entity_links

logger = logging.getLogger(__name__)


@dataclass
class LinkStageResult:
    batches_written: int
    docs_linked: int
    total_links: int
    matched_links: int


def link_shards(
    shard_dir: Path,
    linking_config: LinkingConfig,
    *,
    checkpoint_mgr: CheckpointManager | None = None,
) -> LinkStageResult:
    """Link every batch's enrichment candidates and write entity_links siblings."""
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")
    if linking_config.reference is None:
        raise ValueError("linking.reference must be configured to run the link stage")

    bases = _batch_bases(shard_dir)
    if not bases:
        logger.warning("link_shards: no batches found in %s", shard_dir)
        return LinkStageResult(0, 0, 0, 0)

    reference = load_reference(linking_config.reference)
    logger.info("link_shards: reference loaded — %d entities, %d aliases",
                len(reference.entities), len(reference.aliases))

    batches_written = 0
    docs_linked = 0
    total_links = 0
    matched_links = 0

    for base in bases:
        if checkpoint_mgr is not None and _all_docs_checkpointed(base, checkpoint_mgr):
            logger.info("link_shards: skipping %s (all docs checkpointed)", base.stem)
            continue

        candidates_by_hash = _candidates_for_batch(base, linking_config.candidate_kinds)
        rows: list[dict] = []
        linked_now = 0
        for source_hash, cands in candidates_by_hash.items():
            links = resolve(
                cands, reference,
                name_threshold=linking_config.name_threshold,
                address_kinds=tuple(
                    k for k in linking_config.candidate_kinds if k == "address"
                ) or ("address",),
            )
            rows.extend(_link_to_row(source_hash, lk) for lk in links)
            if any(lk.matched for lk in links):
                linked_now += 1
            matched_links += sum(1 for lk in links if lk.matched)
            total_links += len(links)

        write_entity_links(rows, base)
        batches_written += 1
        docs_linked += linked_now

        if checkpoint_mgr is not None:
            doc_ids = _doc_ids(base)
            if doc_ids:
                checkpoint_mgr.update(
                    doc_ids=doc_ids,
                    succeeded=linked_now,
                    failed=len(doc_ids) - linked_now,
                    batch_num=int(base.stem.replace("batch-", "") or 0),
                )

        logger.info("link_shards: %s wrote %d link rows (%d docs linked)",
                    base.stem, len(rows), linked_now)

    return LinkStageResult(
        batches_written=batches_written,
        docs_linked=docs_linked,
        total_links=total_links,
        matched_links=matched_links,
    )


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def _candidates_for_batch(
    base_path: Path, candidate_kinds: list[str],
) -> dict[str, list[Candidate]]:
    """Read the entities sidecar and group candidate mentions by source_hash.

    The sharded entities sidecar carries source_hash in the ``document_id``
    column (see store.enrichment_output). Candidates are rows whose
    ``entity_type`` is in ``candidate_kinds`` (corporate persons + address
    locations by default).
    """
    table = read_enrichment_entities(base_path)
    if table.num_rows == 0:
        return {}
    kinds = set(candidate_kinds)
    out: dict[str, list[Candidate]] = defaultdict(list)
    for r in table.to_pylist():
        if r["entity_type"] not in kinds:
            continue
        out[r["document_id"]].append(Candidate(
            text=r["name"] or "",
            kind=r["entity_type"],
            source_hash=r["document_id"],
            mention_start=r["mention_start"] if r["mention_start"] is not None else -1,
            mention_end=r["mention_end"] if r["mention_end"] is not None else -1,
        ))
    return dict(out)


def _link_to_row(source_hash: str, link: Link) -> dict:
    e = link.entity
    return {
        "source_hash": source_hash,
        "candidate_text": link.candidate.text,
        "candidate_kind": link.candidate.kind,
        "mention_start": link.candidate.mention_start,
        "mention_end": link.candidate.mention_end,
        "entity_id": e.entity_id if e else "",
        "entity_type": e.entity_type if e else "",
        "canonical_name": e.name if e else "",
        "parent_entity_id": e.parent_id if e else "",
        "confidence": float(link.confidence),
        "match_method": link.method,
        "matched": link.matched,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _doc_ids(base_path: Path) -> list[str]:
    try:
        return read_manifest(base_path).column("doc_id").to_pylist()
    except Exception:
        return []


def _all_docs_checkpointed(base_path: Path, mgr: CheckpointManager) -> bool:
    if not entity_links_path_for(base_path).exists():
        return False
    doc_ids = _doc_ids(base_path)
    return bool(doc_ids) and all(d in mgr.state.processed_ids for d in doc_ids)


__all__ = ["LinkStageResult", "link_shards"]
