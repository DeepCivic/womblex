"""Checkpoint management for resumable batch runs.

Saves progress after each batch so long jobs can resume after interruption.
Checkpoints are JSON files containing processed document IDs and batch metadata.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """State of a pipeline run for resumability."""

    processed_ids: set[str] = field(default_factory=set)
    total_processed: int = 0
    total_succeeded: int = 0
    total_failed: int = 0
    last_batch: int = 0
    started_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "processed_ids": list(self.processed_ids),
            "total_processed": self.total_processed,
            "total_succeeded": self.total_succeeded,
            "total_failed": self.total_failed,
            "last_batch": self.last_batch,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CheckpointState:
        return cls(
            processed_ids=set(data.get("processed_ids", [])),
            total_processed=data.get("total_processed", 0),
            total_succeeded=data.get("total_succeeded", 0),
            total_failed=data.get("total_failed", 0),
            last_batch=data.get("last_batch", 0),
            started_at=data.get("started_at", ""),
            updated_at=data.get("updated_at", ""),
        )


class CheckpointManager:
    """Manages checkpoint state for a pipeline run."""

    def __init__(self, checkpoint_dir: Path, dataset_name: str):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_dir / f"{dataset_name}_checkpoint.json"
        self.state = CheckpointState()

    def load(self) -> CheckpointState:
        """Load existing checkpoint if available."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    data = json.load(f)
                self.state = CheckpointState.from_dict(data)
                logger.info(
                    "Loaded checkpoint: %d documents already processed (batch %d)",
                    self.state.total_processed,
                    self.state.last_batch,
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load checkpoint, starting fresh: %s", e)
                self.state = CheckpointState()
        else:
            self.state = CheckpointState()
            self.state.started_at = datetime.now(UTC).isoformat()
        return self.state

    def save(self) -> None:
        """Save current checkpoint state."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state.updated_at = datetime.now(UTC).isoformat()
        with open(self.checkpoint_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        logger.debug("Saved checkpoint: %d processed", self.state.total_processed)

    def update(self, doc_ids: list[str], succeeded: int, failed: int, batch_num: int) -> None:
        """Update checkpoint after a batch completes."""
        self.state.processed_ids.update(doc_ids)
        self.state.total_processed += len(doc_ids)
        self.state.total_succeeded += succeeded
        self.state.total_failed += failed
        self.state.last_batch = batch_num
        self.save()

    def drop(self, doc_ids: list[str] | set[str]) -> None:
        """Remove ``doc_ids`` from the processed set and save.

        ``last_batch`` is intentionally left alone — batch numbers are
        identifiers, not array slots. Re-extracted docs land in new
        batches past the high-water mark.

        ``total_succeeded`` is decremented optimistically; we can't tell
        from the checkpoint whether a dropped doc originally succeeded
        or failed (the manifest knows but it's the thing we're dropping
        because of). Counters are observability, not load-bearing.
        """
        to_drop = set(doc_ids) & self.state.processed_ids
        if not to_drop:
            return
        self.state.processed_ids.difference_update(to_drop)
        self.state.total_processed = max(0, self.state.total_processed - len(to_drop))
        self.state.total_succeeded = max(0, self.state.total_succeeded - len(to_drop))
        self.save()
        logger.info("Checkpoint: dropped %d doc_id(s) for re-extraction", len(to_drop))

    def filter_unprocessed(self, paths: list[Path]) -> list[Path]:
        """Filter out already-processed documents."""
        return [p for p in paths if p.stem not in self.state.processed_ids]

    def clear(self) -> None:
        """Clear checkpoint state (for fresh runs)."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        self.state = CheckpointState()
        self.state.started_at = datetime.now(UTC).isoformat()
        logger.info("Cleared checkpoint")
