"""Postgres-backed batch job queue using ``FOR UPDATE SKIP LOCKED``.

One table, ``womblex_jobs``, holds one row per batch. Workers claim a pending
row atomically — ``SKIP LOCKED`` lets N workers cooperate without double-firing
and without an external broker. The row's ``status`` is the distributed
checkpoint: a re-enqueue is idempotent on ``(run_id, batch_num)``, so resuming a
crashed run is just "enqueue again, start workers".

psycopg3 (synchronous) matches Womblex's synchronous batch model; the queue
holds one long-lived connection and wraps each claim in a transaction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS womblex_jobs (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id        TEXT        NOT NULL,
    batch_num     INTEGER     NOT NULL,
    status        TEXT        NOT NULL DEFAULT 'pending',
    input_keys    JSONB       NOT NULL,
    shard_prefix  TEXT        NOT NULL,
    attempts      INTEGER     NOT NULL DEFAULT 0,
    max_attempts  INTEGER     NOT NULL DEFAULT 3,
    locked_by     TEXT,
    locked_at     TIMESTAMPTZ,
    error         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, batch_num)
);
CREATE INDEX IF NOT EXISTS womblex_jobs_claim_idx
    ON womblex_jobs (status, batch_num);
"""

# Literal queries (no f-string interpolation) — the only varying part is the
# optional run filter, so each variant is spelled out in full.
_CLAIM_COLS = "id, run_id, batch_num, input_keys, shard_prefix, attempts"
_CLAIM_TAIL = "ORDER BY batch_num FOR UPDATE SKIP LOCKED LIMIT 1"
_CLAIM_ANY = (
    f"SELECT {_CLAIM_COLS} FROM womblex_jobs "
    f"WHERE status = 'pending' AND attempts < max_attempts {_CLAIM_TAIL}"
)
_CLAIM_RUN = (
    f"SELECT {_CLAIM_COLS} FROM womblex_jobs "
    f"WHERE status = 'pending' AND attempts < max_attempts AND run_id = %s {_CLAIM_TAIL}"
)
_STATS_ALL = "SELECT status, count(*) FROM womblex_jobs GROUP BY status"
_STATS_RUN = "SELECT status, count(*) FROM womblex_jobs WHERE run_id = %s GROUP BY status"


@dataclass
class JobSpec:
    """A batch to enqueue: which input keys, and where its shards go."""

    batch_num: int
    input_keys: list[str]
    shard_prefix: str
    max_attempts: int = 3


@dataclass
class Job:
    """A claimed batch."""

    id: int
    run_id: str
    batch_num: int
    input_keys: list[str]
    shard_prefix: str
    attempts: int


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

    def __init__(self, dsn: str):
        psycopg = _require_psycopg()
        self._psycopg = psycopg
        # Pin UTF-8 so text columns decode to ``str`` regardless of server
        # encoding (an SQL_ASCII cluster otherwise hands back ``bytes``).
        self.conn = psycopg.connect(dsn, autocommit=False, client_encoding="UTF8")

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
                        (run_id, batch_num, input_keys, shard_prefix, max_attempts)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, batch_num) DO NOTHING
                    """,
                    (run_id, spec.batch_num, Json(spec.input_keys),
                     spec.shard_prefix, spec.max_attempts),
                )
                inserted += cur.rowcount
        logger.info("Enqueued %d new job(s) for run %s (of %d submitted)",
                    inserted, run_id, len(jobs))
        return inserted

    def claim(self, worker_id: str, run_id: str | None = None) -> Job | None:
        """Atomically claim the next pending batch, or ``None`` if none free."""
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


def utcnow() -> datetime:
    return datetime.now(UTC)


__all__ = ["Job", "JobQueue", "JobSpec", "utcnow"]
