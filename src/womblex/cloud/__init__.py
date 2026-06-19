"""Distributed execution: a Postgres-backed job queue + worker.

Womblex already shards, checkpoints per batch, and isolates per-doc failures —
the only thing missing for horizontal scale-out is shared state and a safe
claim mechanism. This package adds both: a ``FOR UPDATE SKIP LOCKED`` queue
(``cloud.queue``) over one Postgres table, and a worker (``cloud.worker``) that
claims a batch, stages inputs/outputs via ``store.remote``, and runs the shared
``womblex.batch.process_batch`` body. No Redis, no Celery — one table.

In distributed mode the queue *is* the checkpoint (job ``status``), which is why
the worker does not use the local JSON ``CheckpointManager``: concurrent workers
writing one checkpoint file would race; distinct rows under ``SKIP LOCKED`` do
not.
"""

from __future__ import annotations

from womblex.cloud.queue import Job, JobQueue, JobSpec
from womblex.cloud.worker import run_worker

__all__ = ["Job", "JobQueue", "JobSpec", "run_worker"]
