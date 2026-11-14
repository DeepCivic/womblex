"""Per-stage entity linking over an existing shard directory.

Resolves mentions against one or more reference registers and writes
``*.entity_links.parquet`` siblings at mention/span grain. Where the mentions
come from is ``linking.candidate_source``:

- ``enrichment`` — the mentions the enrich stage wrote
  (``*.enrichment_entities.parquet``), selected by kind and matched exactly or
  fuzzily (:mod:`womblex.link.matcher`);
- ``text`` — the registers' own names found in the narrative
  (:mod:`womblex.link.text_candidates`). It reads the elements only, so it runs
  on a corpus that was never enriched.

Either way a mention links to the first register, in the order declared, that
matches it. Mirrors :mod:`womblex.process.chunk_stage`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from womblex.config import LinkingConfig, ReferenceConfig
from womblex.link.matcher import Candidate, Link, Matcher
from womblex.link.reference import ReferenceTable, load_reference
from womblex.link.text_candidates import PhraseBook
from womblex.process.chunk_stage import _batch_bases
from womblex.process.evidence import EvidenceIndexes, stamp_evidence
from womblex.process.text_overlay import require_overlays
from womblex.store.checkpoint import CheckpointManager
from womblex.store.enrichment_output import (
    read_enrichment_entities,
)
from womblex.store.entity_links_output import entity_links_path_for, write_entity_links
from womblex.store.evidence import DOCUMENT, SPAN, TABLE_LAYER, evidence
from womblex.store.output import read_manifest

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
    text_source: str = "elements",
    checkpoint_mgr: CheckpointManager | None = None,
) -> LinkStageResult:
    """Link every batch's enrichment candidates and write entity_links siblings.

    ``text_source`` is the element-text layer the enrichment ran over; a declared
    layer missing from any batch refuses the stage.
    """
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")
    registers = linking_config.registers()
    if not registers:
        raise ValueError("linking.reference must be configured to run the link stage")

    bases = _batch_bases(shard_dir)
    if not bases:
        logger.warning("link_shards: no batches found in %s", shard_dir)
        return LinkStageResult(0, 0, 0, 0)

    require_overlays(bases, text_source)
    loaded = [(r, load_reference(r)) for r in registers]
    for register, table in loaded:
        logger.info("link_shards: reference %s loaded — %d entities, %d names, %d aliases",
                    register.path, len(table.entities), table.name_count, len(table.aliases))
    from_text = linking_config.candidate_source == "text"
    linker = _TextLinker(loaded) if from_text else _MentionLinker(loaded, linking_config)

    batches_written = 0
    docs_linked = 0
    total_links = 0
    matched_links = 0

    for base in bases:
        if checkpoint_mgr is not None and _all_docs_checkpointed(base, checkpoint_mgr):
            logger.info("link_shards: skipping %s (all docs checkpointed)", base.stem)
            continue

        indexes = EvidenceIndexes(base, text_source)
        rows: list[dict] = []
        linked_now = 0
        for source_hash, links in linker.links(base, indexes).items():
            rows.extend(_link_to_row(source_hash, lk, indexes) for lk in links)
            if any(lk.matched for lk in links):
                linked_now += 1
            matched_links += sum(1 for lk in links if lk.matched)
            total_links += len(links)

        stamp_evidence(
            rows, indexes.get, text_key="mention_text", label="entity_links", base=base,
        )
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


class _MentionLinker:
    """Enrichment mentions, each tried against the registers that take its kind."""

    def __init__(
        self, loaded: list[tuple[ReferenceConfig, ReferenceTable]], config: LinkingConfig,
    ) -> None:
        self._matchers: list[tuple[frozenset[str], Matcher]] = []
        for register, table in loaded:
            kinds = frozenset(register.candidate_kinds or config.candidate_kinds)
            self._matchers.append((kinds, Matcher(
                table, name_threshold=config.name_threshold,
                address_kinds=tuple(k for k in kinds if k == "address") or ("address",),
                narrow_above=config.narrow_above, narrow_to=config.narrow_to,
            )))
        self._kinds = sorted(set().union(*(k for k, _ in self._matchers)))

    def links(self, base: Path, _indexes: EvidenceIndexes) -> dict[str, list[Link]]:
        out: dict[str, list[Link]] = {}
        for source_hash, candidates in _candidates_for_batch(base, self._kinds).items():
            try:
                out[source_hash] = [self._match(c) for c in candidates]
            except Exception as e:  # one document must not stop the batch
                logger.warning("link_shards: %s failed (%s)", source_hash, e)
        return out

    def _match(self, candidate: Candidate) -> Link:
        link = Link(candidate, None, 0.0, "unmatched")
        for kinds, matcher in self._matchers:
            if candidate.kind in kinds:
                link = matcher.match(candidate)
                if link.matched:
                    break
        return link


class _TextLinker:
    """The registers' names, found in each document's narrative."""

    def __init__(self, loaded: list[tuple[ReferenceConfig, ReferenceTable]]) -> None:
        self._book = PhraseBook(loaded)
        logger.info("link_shards: matching %d distinct names in the text", len(self._book))

    def links(self, _base: Path, indexes: EvidenceIndexes) -> dict[str, list[Link]]:
        out: dict[str, list[Link]] = {}
        for source_hash in indexes.source_hashes:
            index = indexes.get(source_hash)
            if index is None or not index.narrative:
                continue
            try:
                out[source_hash] = self._book.find(
                    index.narrative, source_hash, text_layer=index.text_layer,
                    heading_spans=index.kind_spans("heading"),
                )
            except Exception as e:  # one document must not stop the batch
                logger.warning("link_shards: %s failed (%s)", source_hash, e)
        return out


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def _candidates_for_batch(
    base_path: Path, candidate_kinds: list[str],
) -> dict[str, list[Candidate]]:
    """Read the entities sidecar and group candidate mentions by source_hash.

    Candidates are rows whose ``entity_type`` is in ``candidate_kinds``
    (corporate persons + address locations by default).
    """
    table = read_enrichment_entities(base_path)
    if table.num_rows == 0:
        return {}
    kinds = set(candidate_kinds)
    out: dict[str, list[Candidate]] = defaultdict(list)
    for r in table.to_pylist():
        if r["entity_type"] not in kinds:
            continue
        out[r["source_hash"]].append(Candidate(
            text=r["name"] or "",
            kind=r["entity_type"],
            source_hash=r["source_hash"],
            mention_start=r["mention_start"] if r["mention_start"] is not None else -1,
            mention_end=r["mention_end"] if r["mention_end"] is not None else -1,
            text_layer=r["text_layer"], elem_order=r["elem_order"], sheet=r["sheet"],
            mention_text=r["mention_text"],
        ))
    return dict(out)


def _mention_evidence(indexes: EvidenceIndexes, source_hash: str, cand: Candidate) -> dict:
    """The mention's evidence reference: its span where it can be placed, else the document.

    A mention the provider gave no offsets for is located to the document; so is
    one whose offsets fall outside the source or whose slice is not the
    enricher's text for it. Without the elements the span is written as the
    enrichment gave it, unchecked.
    """
    start, end = cand.mention_start, cand.mention_end
    index = indexes.get(source_hash, cand.text_layer)
    if end <= start or start < 0:
        return index.document_ref() if index else evidence(DOCUMENT, text_layer=cand.text_layer)
    if index is None:
        return evidence(
            SPAN, elem_order=cand.elem_order, sheet=cand.sheet,
            char_start=start, char_end=end, text_layer=cand.text_layer or indexes.text_source)
    if cand.text_layer == TABLE_LAYER:
        table = index.resolve_table(elem_order=cand.elem_order, sheet=cand.sheet)
        ref = index.table_ref(table, start, end) if table is not None else None
    else:
        ref = index.narrative_ref(start, end)
    # Offsets whose slice is not the enricher's text for the mention are out of
    # step with the source; the mention is located to the document instead.
    if ref is not None and cand.mention_text and index.holds(ref, cand.mention_text) is False:
        ref = None
    return ref or index.document_ref()


def _link_to_row(source_hash: str, link: Link, indexes: EvidenceIndexes) -> dict:
    e = link.entity
    return {
        "source_hash": source_hash,
        "candidate_text": link.candidate.text,
        "candidate_kind": link.candidate.kind,
        **_mention_evidence(indexes, source_hash, link.candidate),
        "mention_text": link.candidate.mention_text,
        "entity_id": e.entity_id if e else "",
        "entity_type": e.entity_type if e else "",
        "canonical_name": e.name if e else "",
        "parent_entity_id": e.parent_id if e else "",
        "confidence": float(link.confidence),
        "match_method": link.method,
        "matched": link.matched,
        "match_detail": link.detail,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _doc_ids(base_path: Path) -> list[str]:
    try:
        return list(read_manifest(base_path).column("doc_id").to_pylist())
    except Exception:
        return []


def _all_docs_checkpointed(base_path: Path, mgr: CheckpointManager) -> bool:
    if not entity_links_path_for(base_path).exists():
        return False
    doc_ids = _doc_ids(base_path)
    return bool(doc_ids) and all(d in mgr.state.processed_ids for d in doc_ids)


__all__ = ["LinkStageResult", "link_shards"]
