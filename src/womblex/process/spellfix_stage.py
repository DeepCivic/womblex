"""Per-stage OCR character-confusion repair over a shard directory.

Consumes ``*.chunks.parquet`` (written by the chunk stage) and writes two
siblings per batch: ``*.chunks_repaired.parquet`` (the repaired chunk layer,
verbatim passthrough where nothing fired) and ``*.spellfix_corrections.parquet``
(the audit trail). The raw ``*.chunks.parquet`` is never modified.

Chunk-level post-fix: the symptom is OCR glyph confusions (``chi1d``) surviving
into chunks, so repair lands on the chunk text directly. Mirrors
:mod:`womblex.process.normalise_stage`: per-stage ``CheckpointManager``,
skip-existing on resume, batch-level isolation. Offline (no API).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from womblex.config import SpellfixConfig
from womblex.process.chunk_stage import _batch_bases
from womblex.store.checkpoint import CheckpointManager
from womblex.store.output import chunks_path_for, read_chunks, read_manifest
from womblex.store.spellfix_output import (
    chunks_repaired_path_for,
    write_repaired_chunks,
    write_spellfix_corrections,
)

logger = logging.getLogger(__name__)


@dataclass
class SpellfixStageResult:
    batches_written: int
    chunks_repaired: int
    corrections_applied: int


def spellfix_shards(
    shard_dir: Path,
    config: SpellfixConfig,
    *,
    checkpoint_mgr: CheckpointManager | None = None,
) -> SpellfixStageResult:
    """Repair every batch's chunks; write repaired + corrections siblings."""
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")

    bases = _batch_bases(shard_dir)
    if not bases:
        logger.warning("spellfix_shards: no batches found in %s", shard_dir)
        return SpellfixStageResult(0, 0, 0)

    batches_written = 0
    chunks_repaired = 0
    corrections_applied = 0

    for base in bases:
        if not chunks_path_for(base).exists():
            continue
        if checkpoint_mgr is not None and _all_docs_checkpointed(base, checkpoint_mgr):
            logger.info("spellfix_shards: skipping %s (all docs checkpointed)", base.stem)
            continue

        repaired_rows, audit_rows, n_changed = _repair_batch(base, config)

        write_repaired_chunks(repaired_rows, base)
        write_spellfix_corrections(audit_rows, base)
        batches_written += 1
        chunks_repaired += n_changed
        corrections_applied += len(audit_rows)

        if checkpoint_mgr is not None:
            doc_ids = _doc_ids(base)
            if doc_ids:
                checkpoint_mgr.update(
                    doc_ids=doc_ids,
                    succeeded=len(doc_ids),
                    failed=0,
                    batch_num=int(base.stem.replace("batch-", "") or 0),
                )

        logger.info(
            "spellfix_shards: %s repaired %d chunks (%d corrections)",
            base.stem, n_changed, len(audit_rows),
        )

    return SpellfixStageResult(
        batches_written=batches_written,
        chunks_repaired=chunks_repaired,
        corrections_applied=corrections_applied,
    )


def _repair_batch(
    base_path: Path, config: SpellfixConfig,
) -> tuple[list[dict], list[dict], int]:
    """Repair a batch's chunks; return ``(repaired_rows, audit_rows, n_changed)``."""
    from womblex.process.spellfix import repair_text

    table = read_chunks(base_path)
    repaired_rows: list[dict] = []
    audit_rows: list[dict] = []
    n_changed = 0
    for row in table.to_pylist():
        text = row.get("text") or ""
        fixed, corrections = repair_text(
            text,
            general_edits=config.general_edits,
            dict_name=config.dict_name,
        )
        row["text"] = fixed
        repaired_rows.append(row)
        if corrections:
            n_changed += 1
        for c in corrections:
            audit_rows.append({
                "source_hash": row["source_hash"],
                "chunk_index": row["chunk_index"],
                "content_type": row["content_type"],
                "offset": c.offset,
                "original": c.original,
                "corrected": c.corrected,
                "method": c.method,
            })
    return repaired_rows, audit_rows, n_changed


def _doc_ids(base_path: Path) -> list[str]:
    try:
        return list(read_manifest(base_path).column("doc_id").to_pylist())
    except Exception:
        return []


def _all_docs_checkpointed(base_path: Path, mgr: CheckpointManager) -> bool:
    if not chunks_repaired_path_for(base_path).exists():
        return False
    doc_ids = _doc_ids(base_path)
    return bool(doc_ids) and all(d in mgr.state.processed_ids for d in doc_ids)


__all__ = ["SpellfixStageResult", "spellfix_shards"]
