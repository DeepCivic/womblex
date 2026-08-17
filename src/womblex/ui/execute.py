"""The Execution Controls' read/write model (docs/ui-plan.md merge 11).

The one console surface that *does* something to a run rather than reading
one — and the plan pins exactly how far that goes:

- **Dispatch is always the queue** (§4 "Running the pipeline from the
  screen"). The console never shells out and never runs a batch in-process:
  it enqueues, and the workers a platform brings up do the work. So the two
  write actions here — enqueue an extraction run, dispatch a downstream
  stage — are thin wrappers over :mod:`womblex.cli.cloud`'s own building
  blocks (``JobQueue.enqueue`` + the same key-listing / batching
  ``cmd_enqueue`` does), reached through the store the sidecar already
  reads. No web request can become an arbitrary command because there is no
  command, only a queue row.

- **Queue-only, so a store *and* a DSN are required.** A queue-less local
  console would need its own background runner and progress reporting, which
  the plan defers (§4). Execution therefore needs both a remote store (to
  enqueue keys from and publish shards to) and a job queue (to dispatch
  through); a local ``output_root``-only deployment can configure and audit
  but not run, and :func:`execution_status` says so rather than half-working.

- **`--allow-execute` is the switch** (§4, §6). Off gives a pure auditing
  console: every write action here refuses with :class:`ExecutionDisabled`
  before it touches the store or the queue, so an audit-only deployment
  cannot be talked into dispatching work.

"Log streaming" is the queue's own job-status transitions
(:meth:`JobQueue.list_jobs`) plus the per-stage checkpoints
:mod:`womblex.ui.dashboard` already reads — a batch-granular feed, labelled
as such (§4 "Live local-run progress"), not a fabricated line-by-line log
the pipeline does not emit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from womblex.cli._shared import SUPPORTED_EXTENSIONS
from womblex.cloud.stage_contracts import STAGE_NAMES
from womblex.store.feedback_output import is_safe_run_id
from womblex.store.retention import generate_run_id
from womblex.ui.deps import UISettings

logger = logging.getLogger(__name__)


class ExecutionDisabled(Exception):
    """A write action was attempted on a console that cannot execute.

    Carries a machine-readable ``reason`` the route maps to a 403 (execution
    off) or 409 (store/queue not configured), so the frontend can tell "this
    deployment is audit-only" from "wire up a queue first" without parsing a
    message.
    """

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ExecutionCapability:
    """Whether this console can dispatch work, and if not, precisely why.

    All three must hold: ``--allow-execute`` on, a remote store to enqueue
    from and publish to, and a job queue to dispatch through (see the module
    docstring). ``can_execute`` is their conjunction; the individual flags
    are surfaced so the screen can name the missing piece rather than a bare
    "disabled".
    """

    allow_execute: bool
    has_store: bool
    has_queue: bool
    stages: tuple[str, ...]

    @property
    def can_execute(self) -> bool:
        return self.allow_execute and self.has_store and self.has_queue

    def as_dict(self) -> dict:
        return {
            "can_execute": self.can_execute,
            "allow_execute": self.allow_execute,
            "has_store": self.has_store,
            "has_queue": self.has_queue,
            "stages": list(self.stages),
        }


def execution_status(settings: UISettings) -> ExecutionCapability:
    """What the console is allowed and able to dispatch — a cheap, network-free read.

    The screen loads this to decide whether to show the run/dispatch controls
    at all, and which explanation to show when it cannot. No store or queue
    connection is made here; reachability is the Resources Console's separate
    ``test`` actions.
    """
    return ExecutionCapability(
        allow_execute=settings.allow_execute,
        has_store=settings.is_remote,
        has_queue=bool(settings.db_dsn),
        stages=STAGE_NAMES,
    )


def _guard(settings: UISettings) -> ExecutionCapability:
    """Refuse any write action the console is not configured to perform.

    Ordered so the operator sees the most actionable failure first: an
    audit-only deployment is a deliberate choice (403), a missing store/queue
    is a wiring gap (409). Every write path calls this before touching either.
    """
    cap = execution_status(settings)
    if not cap.allow_execute:
        raise ExecutionDisabled(
            "execute_disabled",
            "This console is audit-only. Start it with --allow-execute to dispatch work.",
        )
    if not cap.has_store:
        raise ExecutionDisabled(
            "no_store",
            "Execution dispatches through a shared object store; this console reads a "
            "local output_root. Point it at a --store to enqueue work.",
        )
    if not cap.has_queue:
        raise ExecutionDisabled(
            "no_queue",
            "Execution dispatches through the job queue; no DSN is configured. Set one "
            "(--dsn / $WOMBLEX_DB_DSN) to enqueue work.",
        )
    return cap


@dataclass(frozen=True)
class EnqueueResult:
    """The outcome of an enqueue, for the screen to report and then poll.

    ``newly_enqueued`` distinguishes a fresh run from a resume: enqueue is
    idempotent on ``(run_id, batch_num)`` (:meth:`JobQueue.enqueue`), so
    re-dispatching a run that partly ran inserts only its missing batches and
    this counts them. ``run_id`` is what the Dashboard and Corpus Inspector
    are then pointed at to watch it drain.
    """

    run_id: str
    document_count: int
    batch_count: int
    newly_enqueued: int
    shard_prefix: str

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "document_count": self.document_count,
            "batch_count": self.batch_count,
            "newly_enqueued": self.newly_enqueued,
            "shard_prefix": self.shard_prefix,
        }


def enqueue_extraction(
    settings: UISettings,
    *,
    input_prefix: str,
    run_id: str | None = None,
    batch_size: int = 50,
    max_attempts: int = 3,
) -> EnqueueResult:
    """Plan an extraction run into the queue — the "configure-and-run" action.

    The same three steps ``womblex enqueue`` does (``cli/cloud.cmd_enqueue``),
    reached through the sidecar's own store: list supported documents under
    *input_prefix*, split them into ``batch_size`` batches, and write one
    idempotent queue row each. Workers (brought up by the platform, not by
    this call — the console runs no scheduler, plan §4) then claim and process
    them; the Dashboard watches the queue drain.

    Raises :class:`ExecutionDisabled` when the console cannot dispatch, and
    ``ValueError`` on bad input (no run source of documents, a batch size below
    one, an unsafe run_id) — the route maps the former to 403/409 and the
    latter to 400.
    """
    _guard(settings)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    resolved_run_id = run_id or generate_run_id()
    if not is_safe_run_id(resolved_run_id):
        raise ValueError(f"unsafe run_id: {resolved_run_id!r}")

    from womblex.cloud.queue import JobQueue, JobSpec
    from womblex.store.remote import RemoteStore

    store = RemoteStore.from_uri(cast(str, settings.store_uri))
    all_keys = store.list_files(input_prefix, "*")
    keys = sorted(k for k in all_keys if Path(k).suffix.lower() in SUPPORTED_EXTENSIONS)
    if not keys:
        raise ValueError(f"no supported documents under {input_prefix}")

    output_prefix = f"runs/{resolved_run_id}"
    shard_prefix = f"{output_prefix}/documents"
    specs = [
        JobSpec(
            batch_num=batch_idx,
            input_keys=keys[i : i + batch_size],
            shard_prefix=shard_prefix,
            max_attempts=max_attempts,
        )
        for batch_idx, i in enumerate(range(0, len(keys), batch_size), start=1)
    ]

    with JobQueue(cast(str, settings.db_dsn)) as queue:
        queue.ensure_schema()
        newly = queue.enqueue(resolved_run_id, specs)

    logger.info(
        "console enqueue: run_id=%s, %d doc(s) -> %d batch(es), %d newly enqueued",
        resolved_run_id, len(keys), len(specs), newly,
    )
    return EnqueueResult(
        run_id=resolved_run_id,
        document_count=len(keys),
        batch_count=len(specs),
        newly_enqueued=newly,
        shard_prefix=shard_prefix,
    )
