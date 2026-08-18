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

- **`--audit-only` is the switch** (§4, §6). By default the console can
  dispatch; pass ``--audit-only`` for a pure auditing console where every
  write action here refuses with :class:`ExecutionDisabled` before it
  touches the store or the queue, so an audit-only deployment cannot be
  talked into dispatching work.

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

    All four must hold: not ``--audit-only``, an output location a
    ``RemoteStore`` can publish shards to, an ingest location to enqueue
    documents from, and a job queue to dispatch through (see the module
    docstring). ``can_execute`` is their conjunction; the individual flags
    are surfaced so the screen can name the missing piece rather than a bare
    "disabled".
    """

    audit_only: bool
    has_store: bool
    has_ingest: bool
    has_queue: bool
    stages: tuple[str, ...]
    ingest_uri: str | None
    output_uri: str | None

    @property
    def can_execute(self) -> bool:
        return (not self.audit_only) and self.has_store and self.has_ingest and self.has_queue

    def as_dict(self) -> dict:
        return {
            "can_execute": self.can_execute,
            "audit_only": self.audit_only,
            "has_store": self.has_store,
            "has_ingest": self.has_ingest,
            "has_queue": self.has_queue,
            "stages": list(self.stages),
            "ingest_uri": self.ingest_uri,
            "output_uri": self.output_uri,
        }


def execution_status(settings: UISettings) -> ExecutionCapability:
    """What the console is allowed and able to dispatch — a cheap, network-free read.

    The screen loads this to decide whether to show the run/dispatch controls
    at all, and which explanation to show when it cannot. No store or queue
    connection is made here; reachability is the Resources Console's separate
    ``test`` actions.
    """
    return ExecutionCapability(
        audit_only=settings.audit_only,
        has_store=settings.is_remote,
        has_ingest=bool(settings.ingest_uri),
        has_queue=bool(settings.db_dsn),
        stages=STAGE_NAMES,
        ingest_uri=settings.ingest_uri,
        output_uri=settings.store_uri or (str(settings.output_root) if settings.output_root else None),
    )


def _guard(settings: UISettings) -> ExecutionCapability:
    """Refuse any write action the console is not configured to perform.

    Ordered so the operator sees the most actionable failure first: an
    audit-only deployment is a deliberate choice (403); a missing store,
    ingest location or queue is a wiring gap (409), checked in that order.
    Every write path calls this before touching either.
    """
    cap = execution_status(settings)
    if cap.audit_only:
        raise ExecutionDisabled(
            "execute_disabled",
            "This console is audit-only. Restart it without --audit-only to dispatch work.",
        )
    if not cap.has_store:
        raise ExecutionDisabled(
            "no_store",
            "Execution dispatches through a shared object store; this console reads a "
            "local output_root. Point it at a --store to enqueue work.",
        )
    if not cap.has_ingest:
        raise ExecutionDisabled(
            "no_ingest",
            "Execution enqueues documents from a configured ingest location; none is "
            "set. Set one (--ingest / $WOMBLEX_INGEST_URI) to enqueue work.",
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
    input_prefix: str | None = None,
    run_id: str | None = None,
    batch_size: int = 50,
    max_attempts: int = 3,
) -> EnqueueResult:
    """Plan an extraction run into the queue — the "configure-and-run" action.

    The same three steps ``womblex enqueue`` does, against the configured
    ingest location: list supported documents under *input_prefix* (the whole
    ingest root when omitted), split them into ``batch_size`` batches, and
    write one idempotent queue row each, stamped with the ingest root.

    Raises :class:`ExecutionDisabled` when the console cannot dispatch (the
    route maps it to 403/409) and ``ValueError`` on bad input (→ 400).
    """
    _guard(settings)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    resolved_run_id = run_id or generate_run_id()
    if not is_safe_run_id(resolved_run_id):
        raise ValueError(f"unsafe run_id: {resolved_run_id!r}")

    from womblex.cloud.queue import JobQueue, JobSpec
    from womblex.store.remote import RemoteStore

    ingest_uri = cast(str, settings.ingest_uri)
    ingest_store = RemoteStore.from_uri(ingest_uri)
    prefix = input_prefix or ""
    all_keys = ingest_store.list_files(prefix, "*", recursive=True)
    keys = sorted(k for k in all_keys if Path(k).suffix.lower() in SUPPORTED_EXTENSIONS)
    if not keys:
        raise ValueError(f"no supported documents under {ingest_uri}/{prefix}".rstrip("/"))

    output_prefix = f"runs/{resolved_run_id}"
    shard_prefix = f"{output_prefix}/documents"
    specs = [
        JobSpec(
            batch_num=batch_idx,
            input_keys=keys[i : i + batch_size],
            shard_prefix=shard_prefix,
            max_attempts=max_attempts,
            ingest_root=ingest_uri,
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


def ingest_preflight(settings: UISettings) -> dict:
    """Reachability + document count of the configured ingest location.

    Feeds the composer's "N documents ready" line, using the same recursive
    listing and ``SUPPORTED_EXTENSIONS`` filter the enqueue does — so the
    count shown is the count that would be enqueued.
    """
    if not settings.ingest_uri:
        return {
            "uri": None, "kind": None, "reachable": False,
            "document_count": 0, "sample": [],
            "error": "no ingest location configured",
        }
    from womblex.store.remote import RemoteStore, is_remote_uri

    uri = settings.ingest_uri
    kind = "remote" if is_remote_uri(uri) else "local"
    try:
        all_keys = RemoteStore.from_uri(uri).list_files("", "*", recursive=True)
    except Exception as e:
        logger.warning("execute: ingest unreachable: %s", e)
        return {
            "uri": uri, "kind": kind, "reachable": False,
            "document_count": 0, "sample": [], "error": str(e),
        }
    keys = sorted(k for k in all_keys if Path(k).suffix.lower() in SUPPORTED_EXTENSIONS)
    return {
        "uri": uri, "kind": kind, "reachable": True,
        "document_count": len(keys), "sample": keys[:5], "error": None,
    }
