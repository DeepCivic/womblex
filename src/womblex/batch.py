"""Shared per-batch pipeline body.

``cmd_run`` (local, single-process) and the cloud worker (distributed) must
process a batch of documents *identically* — same stages, same sequencing,
same shard layout — or the two execution modes would silently diverge. This
module is the single home for that sequencing: extraction → optional redaction
→ optional chunking → optional PII → write one ``batch-NNNN.parquet`` shard
(and its sidecars).

Deliberately stateless: it does no checkpointing and no cumulative-size
bookkeeping. Those are caller concerns — the local runner uses a
``CheckpointManager`` + cross-batch size check, the distributed worker uses the
Postgres job queue as its checkpoint. Keeping them out keeps this body reusable
by both without a race.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from womblex.config import WomblexConfig
from womblex.operations import (
    BatchResult,
    run_chunking,
    run_extraction,
    run_pii_cleaning,
    run_redaction,
    write_batch_parquet,
)

logger = logging.getLogger(__name__)


@dataclass
class BatchOutcome:
    """Result of processing one batch: the results plus what was persisted."""

    batch: BatchResult
    shard_path: Path
    rows_written: int


def process_batch(
    batch_files: list[Path],
    config: WomblexConfig,
    *,
    batch_num: int,
    shard_dir: Path,
) -> BatchOutcome:
    """Run the configured stages over *batch_files* and write one shard.

    Mirrors the inner loop of ``cmd_run``. Stage gating follows the config
    flags (``redaction``/``chunking``/``pii`` ``.enabled``) exactly as the
    local runner does, so a worker fed the same config produces byte-identical
    shards. Returns a :class:`BatchOutcome`; the caller decides how to verify,
    checkpoint, or publish.
    """
    results = run_extraction(batch_files, config)
    if config.redaction.enabled:
        results = run_redaction(results, config)
    if config.chunking.enabled:
        results = run_chunking(results, config)
    if config.pii.enabled:
        results = run_pii_cleaning(results, config)

    batch = BatchResult(results=results)
    shard_path = shard_dir / f"batch-{batch_num:04d}.parquet"
    rows_written = sum(
        1 for r in batch.results if r.status == "completed" and r.extraction is not None
    )
    write_batch_parquet(batch, shard_path)
    return BatchOutcome(batch=batch, shard_path=shard_path, rows_written=rows_written)


__all__ = ["BatchOutcome", "process_batch"]
