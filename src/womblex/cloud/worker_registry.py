"""Which work each worker serves (Phase R, D8).

One row per worker in the system database (`womblex_workers`), naming the queues it joined
after its model check and when it last said so. A worker records itself at launch, refreshes
the row while it runs and removes it when it stops; a row older than the staleness threshold
is a worker that stopped without saying so. Read by `live_queues`, so a dispatcher can tell
whether any live worker serves the work it is about to queue.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any, Self

from womblex.cloud import dbos_app

#: Seconds without a refresh after which a worker no longer counts as live (the board's
#: staleness threshold, `stale_after_seconds`).
LIVE_SECONDS = 900.0

_TABLE = "womblex_workers"


def _ddl() -> Any:
    import sqlalchemy as sa

    metadata = sa.MetaData()
    return sa.Table(
        _TABLE, metadata,
        sa.Column("worker_id", sa.Text, primary_key=True),
        sa.Column("queues_json", sa.Text, nullable=False),
        sa.Column("updated_at", sa.Text, nullable=False),
    ), metadata


class WorkerRegistry:
    """Record and read the work live workers serve, through one SQLAlchemy engine."""

    def __init__(self, dsn: str | None):
        import sqlalchemy as sa

        self._table, self._metadata = _ddl()
        self._engine = sa.create_engine(dbos_app.system_database_url(dsn))

    def close(self) -> None:
        self._engine.dispose()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def record(self, worker_id: str, queues: list[str]) -> None:
        """Say *worker_id* serves *queues*, now. Creates the table if absent."""
        import sqlalchemy as sa

        self._metadata.create_all(self._engine, tables=[self._table])
        row = {"queues_json": json.dumps(sorted(queues)), "updated_at": datetime.now(UTC).isoformat()}
        with self._engine.begin() as conn:
            done = conn.execute(sa.update(self._table).where(self._table.c.worker_id == worker_id).values(**row))
            if not done.rowcount:
                conn.execute(sa.insert(self._table).values(worker_id=worker_id, **row))

    def remove(self, worker_id: str) -> None:
        import sqlalchemy as sa

        self._metadata.create_all(self._engine, tables=[self._table])
        with self._engine.begin() as conn:
            conn.execute(sa.delete(self._table).where(self._table.c.worker_id == worker_id))

    def live_queues(self, within_seconds: float = LIVE_SECONDS) -> dict[str, list[str]]:
        """Each queue a worker refreshed within *within_seconds* serves -> those workers."""
        import sqlalchemy as sa

        if not sa.inspect(self._engine).has_table(_TABLE):
            return {}  # no worker has ever recorded itself
        cutoff = (datetime.now(UTC) - timedelta(seconds=within_seconds)).isoformat()
        with self._engine.connect() as conn:
            rows = conn.execute(sa.select(self._table).where(self._table.c.updated_at >= cutoff)).fetchall()
        served: dict[str, list[str]] = {}
        for row in rows:
            for queue in json.loads(row.queues_json):
                served.setdefault(queue, []).append(row.worker_id)
        return served


__all__ = ["LIVE_SECONDS", "WorkerRegistry"]
