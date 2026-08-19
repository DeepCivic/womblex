"""Postgres-backed job queue using ``FOR UPDATE SKIP LOCKED``.

One table, ``womblex_jobs``, holds one row per unit of work — an extraction
batch (``kind='batch'``) or a downstream stage over a whole run
(``kind='stage'``). Workers claim a pending row atomically — ``SKIP LOCKED`` lets N workers cooperate without double-firing
and without an external broker. The row's ``status`` is the distributed
checkpoint: a re-enqueue is idempotent on ``(run_id, batch_num)``, so resuming a
crashed run is just "enqueue again, start workers".

Stage rows reuse ``batch_num`` as their queue position, offset past every real
batch by :data:`STAGE_SEQ_BASE` (see :meth:`JobQueue.enqueue_stages`). That is
what makes ``ORDER BY batch_num`` claim all extraction before any stage, and
the stages among themselves in pipeline order, with no second ordering column
and no change to the existing unique constraint.

psycopg3 (synchronous) matches Womblex's synchronous batch model; the queue
holds one long-lived connection and wraps each claim in a transaction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self

logger = logging.getLogger(__name__)

# The one table Womblex owns in its database (plus its claim index). Pinned
# byte-for-byte to ``sql/womblex_jobs.sql`` by ``tests/test_cloud.py`` — that
# file is the DBA-reviewable artefact for provisioning the table in a shared or
# externally-managed DB, and is what ``psql -f`` applies; edit both together.
# Every statement is IF NOT EXISTS and scoped to this table (no DROP/TRUNCATE,
# no CREATE DATABASE/SCHEMA, no search_path), so Womblex coexists with another
# system's tables in the same database.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS womblex_jobs (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id        TEXT        NOT NULL,
    batch_num     INTEGER     NOT NULL,
    kind          TEXT        NOT NULL DEFAULT 'batch',
    stage         TEXT,
    status        TEXT        NOT NULL DEFAULT 'pending',
    input_keys    JSONB       NOT NULL,
    shard_prefix  TEXT        NOT NULL,
    ingest_root   TEXT,
    attempts      INTEGER     NOT NULL DEFAULT 0,
    max_attempts  INTEGER     NOT NULL DEFAULT 3,
    locked_by     TEXT,
    locked_at     TIMESTAMPTZ,
    error         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, batch_num)
);
ALTER TABLE womblex_jobs ADD COLUMN IF NOT EXISTS ingest_root TEXT;
ALTER TABLE womblex_jobs ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'batch';
ALTER TABLE womblex_jobs ADD COLUMN IF NOT EXISTS stage TEXT;
CREATE INDEX IF NOT EXISTS womblex_jobs_claim_idx
    ON womblex_jobs (status, batch_num);
"""

# Literal queries (no f-string interpolation) — the only varying part is the
# optional run filter, so each variant is spelled out in full.
_CLAIM_COLS = (
    "id, run_id, batch_num, input_keys, shard_prefix, attempts, ingest_root, kind, stage"
)
# A stage row runs over what everything ahead of it published, so it is only
# claimable once nothing earlier in its run is still pending or running: all of
# extraction (`batch_num` 1..N) and every stage below it in `PIPELINE_ORDER`.
# Batch rows carry no such gate — they are independent of one another and must
# stay claimable in parallel, which is the whole point of the fleet.
#
# A *failed* predecessor does not hold the queue. Its row already records the
# failure, and a dependent stage surfaces the gap as `not-ready` bases rather
# than the operator finding a run wedged on rows that will never settle.
_UNSETTLED_EARLIER = (
    "NOT EXISTS (SELECT 1 FROM womblex_jobs AS earlier "
    "WHERE earlier.run_id = womblex_jobs.run_id "
    "AND earlier.batch_num < womblex_jobs.batch_num "
    "AND earlier.status IN ('pending', 'running'))"
)
_CLAIMABLE = (
    f"status = 'pending' AND attempts < max_attempts "
    f"AND (kind <> 'stage' OR {_UNSETTLED_EARLIER})"
)
_CLAIM_TAIL = "ORDER BY batch_num FOR UPDATE SKIP LOCKED LIMIT 1"
_CLAIM_ANY = f"SELECT {_CLAIM_COLS} FROM womblex_jobs WHERE {_CLAIMABLE} {_CLAIM_TAIL}"
_CLAIM_RUN = (
    f"SELECT {_CLAIM_COLS} FROM womblex_jobs "
    f"WHERE {_CLAIMABLE} AND run_id = %s {_CLAIM_TAIL}"
)
_STATS_ALL = "SELECT status, count(*) FROM womblex_jobs GROUP BY status"
_STATS_RUN = "SELECT status, count(*) FROM womblex_jobs WHERE run_id = %s GROUP BY status"

# The read-only queries behind the console dashboard (docs/ui-plan.md §3).
# Their optional run filter is expressed as `(%s::text IS NULL OR ...)` rather
# than a second spelled-out variant: the parameter carries the value, so each
# query stays one literal string no matter how the caller scopes it. The cast
# is what lets Postgres type an untyped NULL parameter.
_RUN_FILTER = "(%s::text IS NULL OR run_id = %s::text)"
_JOB_COLS = (
    "id, run_id, batch_num, status, attempts, max_attempts, "
    "locked_by, locked_at, error, created_at, updated_at, kind, stage"
)
_LIST_JOBS = (
    f"SELECT {_JOB_COLS} FROM womblex_jobs WHERE {_RUN_FILTER} "
    f"AND (%s::text IS NULL OR status = %s::text) "
    f"ORDER BY updated_at DESC, batch_num DESC LIMIT %s"
)
# Same predicate `requeue_stale` acts on, without the UPDATE — the dashboard
# reports what a worker's `--stale-timeout` would recover, it never recovers it.
_STALE_JOBS = (
    f"SELECT {_JOB_COLS} FROM womblex_jobs "
    f"WHERE status = 'running' AND locked_at < now() - make_interval(secs => %s) "
    f"AND {_RUN_FILTER} ORDER BY locked_at"
)
_WORKERS = (
    "SELECT locked_by, count(*), min(locked_at), max(locked_at) FROM womblex_jobs "
    f"WHERE status = 'running' AND locked_by IS NOT NULL AND {_RUN_FILTER} "
    "GROUP BY locked_by ORDER BY locked_by"
)
_THROUGHPUT = (
    "SELECT count(*), max(updated_at) FROM womblex_jobs "
    "WHERE status = 'done' AND updated_at >= now() - make_interval(secs => %s) "
    f"AND {_RUN_FILTER}"
)


#: Queue position of the first stage row, past any plausible batch count.
#: Stage rows sit at ``STAGE_SEQ_BASE + stage_rank(stage)``, so the existing
#: ``UNIQUE (run_id, batch_num)`` gives one row per ``(run_id, stage)`` for
#: free, ``ORDER BY batch_num`` drains extraction before dispatching any stage,
#: and the stages claim in pipeline order. ``batch_num`` is ``INTEGER``, so a
#: billion leaves ample headroom above and below.
STAGE_SEQ_BASE = 1_000_000_000


@dataclass
class JobSpec:
    """A batch to enqueue: which input keys, and where its shards go."""

    batch_num: int
    input_keys: list[str]
    shard_prefix: str
    max_attempts: int = 3
    ingest_root: str | None = None


@dataclass
class Job:
    """A claimed unit of work — an extraction batch, or one downstream stage.

    ``kind`` discriminates: ``'batch'`` rows carry ``input_keys`` (the source
    documents to extract) and ``'stage'`` rows carry ``stage`` (the contract to
    run over ``shard_prefix``, whose inputs are already in the store). The two
    never mix — a stage row's ``input_keys`` is empty.
    """

    id: int
    run_id: str
    batch_num: int
    input_keys: list[str]
    shard_prefix: str
    attempts: int
    ingest_root: str | None = None
    kind: str = "batch"
    stage: str | None = None

    @property
    def label(self) -> str:
        """How this job names itself in a log line."""
        return f"stage {self.stage}" if self.kind == "stage" else f"batch {self.batch_num}"


@dataclass(frozen=True)
class JobRow:
    """One ``womblex_jobs`` row as read (not claimed) — the job list's grain.

    Timestamps are ISO-8601 strings rather than ``datetime``: every consumer
    so far either prints them or serialises them to JSON, and converting once
    here keeps the queue's tz-aware values from being re-derived per caller.
    """

    id: int
    run_id: str
    batch_num: int
    status: str
    attempts: int
    max_attempts: int
    locked_by: str | None
    locked_at: str | None
    error: str | None
    created_at: str | None
    updated_at: str | None
    kind: str = "batch"
    stage: str | None = None


@dataclass(frozen=True)
class WorkerState:
    """A worker's live hold on the queue, derived from ``locked_by``.

    This is *not* liveness: an exited worker leaves its locks behind, so a
    row here past the stale threshold means orphaned work, not a busy worker
    (docs/ui-plan.md §4).
    """

    worker_id: str
    running: int
    oldest_locked_at: str | None
    newest_locked_at: str | None


@dataclass(frozen=True)
class Throughput:
    """Completions inside a trailing window — the dashboard's rate tile."""

    window_seconds: float
    completed: int
    per_minute: float
    last_completed_at: str | None


def _iso(value: object) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _require_psycopg():  # type: ignore[no-untyped-def]
    try:
        import psycopg
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The job queue requires the 'cloud' extra. "
            "Install with: pip install womblex[cloud]"
        ) from e
    import psycopg

    return psycopg


class JobQueue:
    """Synchronous Postgres job queue over a single ``womblex_jobs`` table."""

    def __init__(self, dsn: str, *, connect_timeout: float | None = None):
        """Open the queue connection.

        ``connect_timeout`` (seconds) bounds the connect attempt. Workers
        leave it unset — a worker blocking until the OS gives up is fine,
        it has nothing else to do. A request-serving caller should set it:
        a routable-but-dead host otherwise holds the request for the TCP
        timeout, and a polling console would pin every thread it has.
        """
        psycopg = _require_psycopg()
        self._psycopg = psycopg
        # Pin UTF-8 so text columns decode to ``str`` regardless of server
        # encoding (an SQL_ASCII cluster otherwise hands back ``bytes``).
        extra = {} if connect_timeout is None else {"connect_timeout": int(connect_timeout)}
        self.conn = psycopg.connect(
            dsn, autocommit=False, client_encoding="UTF8", **extra,
        )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ensure_schema(self) -> None:
        with self.conn.transaction():
            self.conn.execute(_SCHEMA)
        logger.info("womblex_jobs schema ready")

    def enqueue(self, run_id: str, jobs: list[JobSpec]) -> int:
        """Insert *jobs* for *run_id*; idempotent on ``(run_id, batch_num)``.

        Returns the number of rows actually inserted (existing batches are
        skipped, so re-running ``enqueue`` to resume is safe).
        """
        from psycopg.types.json import Json

        inserted = 0
        with self.conn.transaction():
            for spec in jobs:
                cur = self.conn.execute(
                    """
                    INSERT INTO womblex_jobs
                        (run_id, batch_num, input_keys, shard_prefix, ingest_root, max_attempts)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, batch_num) DO NOTHING
                    """,
                    (run_id, spec.batch_num, Json(spec.input_keys),
                     spec.shard_prefix, spec.ingest_root, spec.max_attempts),
                )
                inserted += cur.rowcount
        logger.info("Enqueued %d new job(s) for run %s (of %d submitted)",
                    inserted, run_id, len(jobs))
        return inserted

    def enqueue_stages(
        self, run_id: str, stages: list[str], shard_prefix: str, *, max_attempts: int = 3,
    ) -> int:
        """Insert one stage row per name in *stages*; idempotent per stage.

        *stages* is whatever the caller wants run — dispatchers pass
        ``pipeline_order.enabled_downstream_stages(config)``, which is already
        ordered and config-gated. Order is not taken from the list: each row's
        queue position comes from `stage_rank`, so an out-of-order list still
        runs in pipeline order and a duplicated name collapses to one row.

        Returns the number of rows actually inserted. A stage already enqueued
        for this run is skipped, so pressing "run downstream stages" twice does
        not re-run what finished — the same resume-by-re-enqueue property batch
        rows have, and the stage runner is itself idempotent underneath it.
        """
        from psycopg.types.json import Json

        from womblex.pipeline_order import stage_rank

        inserted = 0
        with self.conn.transaction():
            for stage in stages:
                cur = self.conn.execute(
                    """
                    INSERT INTO womblex_jobs
                        (run_id, batch_num, kind, stage, input_keys, shard_prefix, max_attempts)
                    VALUES (%s, %s, 'stage', %s, %s, %s, %s)
                    ON CONFLICT (run_id, batch_num) DO NOTHING
                    """,
                    (run_id, STAGE_SEQ_BASE + stage_rank(stage), stage, Json([]),
                     shard_prefix, max_attempts),
                )
                inserted += cur.rowcount
        logger.info("Enqueued %d new stage job(s) for run %s (of %d submitted)",
                    inserted, run_id, len(stages))
        return inserted

    def claim(self, worker_id: str, run_id: str | None = None) -> Job | None:
        """Atomically claim the next claimable job, or ``None`` if none free.

        Batch rows are claimable as soon as they are pending. A stage row also
        waits on everything earlier in its run settling, so ``None`` here can
        mean "work exists but is not yet due" — the caller polls, it does not
        treat it as drained.
        """
        sql = _CLAIM_RUN if run_id else _CLAIM_ANY
        params: tuple = (run_id,) if run_id else ()
        with self.conn.transaction():
            row = self.conn.execute(sql, params).fetchone()
            if row is None:
                return None
            job_id = row[0]
            self.conn.execute(
                """
                UPDATE womblex_jobs
                SET status = 'running', locked_by = %s, locked_at = now(),
                    attempts = attempts + 1, updated_at = now()
                WHERE id = %s
                """,
                (worker_id, job_id),
            )
        return Job(
            id=row[0], run_id=row[1], batch_num=row[2],
            input_keys=list(row[3]), shard_prefix=row[4], attempts=row[5] + 1,
            ingest_root=row[6], kind=row[7], stage=row[8],
        )

    def complete(self, job_id: int) -> None:
        with self.conn.transaction():
            self.conn.execute(
                "UPDATE womblex_jobs SET status='done', error=NULL, updated_at=now() WHERE id=%s",
                (job_id,),
            )

    def fail(self, job_id: int, error: str) -> None:
        """Mark a job failed, or return it to ``pending`` if retries remain."""
        with self.conn.transaction():
            self.conn.execute(
                """
                UPDATE womblex_jobs
                SET status = CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'pending' END,
                    locked_by = NULL, locked_at = NULL,
                    error = %s, updated_at = now()
                WHERE id = %s
                """,
                (error[:2000], job_id),
            )

    def release(self, job_id: int, error: str) -> None:
        """Return a claimed job to ``pending`` without consuming an attempt.

        For a refusal that says nothing about the job — a worker wired to the
        wrong ingest root — where :meth:`fail` would burn the retry budget on
        a batch a correctly-wired worker could still run. The reason is still
        recorded, so the row reads as pending-with-a-reason on the dashboard
        rather than silently stalled.
        """
        with self.conn.transaction():
            self.conn.execute(
                """
                UPDATE womblex_jobs
                SET status='pending', attempts = GREATEST(attempts - 1, 0),
                    locked_by=NULL, locked_at=NULL, error=%s, updated_at=now()
                WHERE id=%s
                """,
                (error[:2000], job_id),
            )

    def requeue_stale(self, older_than_seconds: float) -> int:
        """Return ``running`` jobs locked longer than the threshold to ``pending``.

        Recovers work orphaned by a crashed worker (which never called
        ``fail``). Returns the number requeued.
        """
        with self.conn.transaction():
            cur = self.conn.execute(
                """
                UPDATE womblex_jobs
                SET status='pending', locked_by=NULL, locked_at=NULL, updated_at=now()
                WHERE status='running'
                  AND locked_at < now() - make_interval(secs => %s)
                """,
                (older_than_seconds,),
            )
            n: int = cur.rowcount
        if n:
            logger.warning("Requeued %d stale job(s) (locked > %.0fs)", n, older_than_seconds)
        return n

    def stats(self, run_id: str | None = None) -> dict[str, int]:
        """Count jobs by status (optionally for one run)."""
        sql = _STATS_RUN if run_id else _STATS_ALL
        params: tuple = (run_id,) if run_id else ()
        rows = self.conn.execute(sql, params).fetchall()
        return {status: count for status, count in rows}

    # --- read-only views (the console dashboard; docs/ui-plan.md §3) ---------

    def list_jobs(
        self, run_id: str | None = None, *, status: str | None = None, limit: int = 200,
    ) -> list[JobRow]:
        """Recent jobs, newest activity first, optionally scoped by run and status."""
        rows = self.conn.execute(
            _LIST_JOBS, (run_id, run_id, status, status, limit)
        ).fetchall()
        return [_job_row(r) for r in rows]

    def stale_jobs(
        self, older_than_seconds: float, run_id: str | None = None,
    ) -> list[JobRow]:
        """``running`` jobs locked longer than the threshold, oldest lock first.

        The read-only twin of :meth:`requeue_stale`: same predicate, no
        recovery. A worker requeues these; the console only names them.
        """
        rows = self.conn.execute(
            _STALE_JOBS, (older_than_seconds, run_id, run_id)
        ).fetchall()
        return [_job_row(r) for r in rows]

    def workers(self, run_id: str | None = None) -> list[WorkerState]:
        """Which workers hold which batches right now — the fleet view."""
        rows = self.conn.execute(_WORKERS, (run_id, run_id)).fetchall()
        return [
            WorkerState(
                worker_id=worker_id, running=running,
                oldest_locked_at=_iso(oldest), newest_locked_at=_iso(newest),
            )
            for worker_id, running, oldest, newest in rows
        ]

    def throughput(
        self, run_id: str | None = None, *, window_seconds: float = 3600.0,
    ) -> Throughput:
        """Batches completed in the trailing window, as a rate.

        Derived from ``updated_at`` on ``done`` rows — no new schema
        (docs/ui-plan.md §4). A retried batch that later succeeds counts once,
        because only its final transition leaves the row ``done``.
        """
        row = self.conn.execute(
            _THROUGHPUT, (window_seconds, run_id, run_id)
        ).fetchone()
        completed = int(row[0]) if row else 0
        return Throughput(
            window_seconds=window_seconds,
            completed=completed,
            per_minute=completed / (window_seconds / 60.0) if window_seconds > 0 else 0.0,
            last_completed_at=_iso(row[1]) if row else None,
        )


def _job_row(row: tuple) -> JobRow:
    return JobRow(
        id=row[0], run_id=row[1], batch_num=row[2], status=row[3],
        attempts=row[4], max_attempts=row[5], locked_by=row[6],
        locked_at=_iso(row[7]), error=row[8],
        created_at=_iso(row[9]), updated_at=_iso(row[10]),
        kind=row[11], stage=row[12],
    )


def utcnow() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "STAGE_SEQ_BASE",
    "Job",
    "JobQueue",
    "JobRow",
    "JobSpec",
    "Throughput",
    "WorkerState",
    "utcnow",
]
