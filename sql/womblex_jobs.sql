-- Womblex job queue schema — the one table Womblex owns in its database.
--
-- Womblex is a well-behaved tenant: it creates and only ever touches
-- `womblex_jobs` (and its one index). There are no DROP/TRUNCATE, no
-- CREATE DATABASE/SCHEMA, and no search_path changes anywhere in the code —
-- every statement is scoped to this table. So Womblex can share a database
-- with another system (its tables coexisting alongside, e.g., `redline_*`
-- tables) provided the name `womblex_jobs` / `womblex_jobs_claim_idx` does
-- not collide with a table the other system owns.
--
-- Applying this file is the DBA-reviewable way to provision the table in a
-- shared or externally-managed database:
--
--     psql "$WOMBLEX_DB_DSN" -f sql/womblex_jobs.sql
--
-- It is equivalent to what `womblex jobs --create-schema` (or the first
-- `womblex enqueue --create-schema`) runs, and to what the console's Execution
-- enqueue path calls on first use. All statements are IF NOT EXISTS, so it is
-- idempotent and safe to re-apply. The console/dashboard never runs it;
-- creation is owned by init/enqueue or by applying this file.
--
-- This file is pinned byte-for-byte to `_SCHEMA` in
-- `src/womblex/cloud/queue.py` by `tests/test_cloud.py` — edit both together.

CREATE TABLE IF NOT EXISTS womblex_jobs (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id        TEXT        NOT NULL,
    batch_num     INTEGER     NOT NULL,
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
CREATE INDEX IF NOT EXISTS womblex_jobs_claim_idx
    ON womblex_jobs (status, batch_num);
