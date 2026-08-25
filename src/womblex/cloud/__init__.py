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

Downstream of extraction, ``cloud.stage_contracts`` + ``cloud.stage_runner``
carry the same idea to the per-batch sidecar stages: declare what each
``*_shards()`` stage reads and writes (both partly a function of config), then
stage one batch in, run the unchanged stage, and publish all of its declared
outputs or none. That is ``womblex finalize``'s shape, generalised.
"""

from __future__ import annotations

from womblex.cloud.queue import Job, JobQueue, JobSpec
from womblex.cloud.stage_contracts import STAGE_CONTRACTS, MutationMode, StageContract, StageScope
from womblex.cloud.stage_runner import StageRunSummary, run_stage_local, run_stage_remote
from womblex.cloud.worker import run_worker

__all__ = [
    "STAGE_CONTRACTS",
    "Job",
    "JobQueue",
    "JobSpec",
    "MutationMode",
    "StageContract",
    "StageRunSummary",
    "StageScope",
    "run_stage_local",
    "run_stage_remote",
    "run_worker",
]
