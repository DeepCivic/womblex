"""Which work each worker serves (Phase R, D8).

One row per worker in the system database (`womblex_workers`), naming the queues it joined
after its model check and when it last said so. A worker records itself at launch, refreshes
the row while it runs and removes it when it stops; a row older than the staleness threshold
is a worker that stopped without saying so. Read by `live_queues`, so a dispatcher can tell
whether any live worker serves the work it is about to queue: `require_served` refuses, before
any row, work no live worker serves.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Self

from womblex.cloud import dbos_app

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

    from womblex.config import WomblexConfig

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
        self._table, self._metadata = _ddl()
        self._dsn = dsn
        self._made: Any = None
        # Reading a local database no writer has created reads nothing, and creates no file.
        self._unwritten = not dsn and not dbos_app.local_database_path().exists()

    @property
    def _engine(self) -> Any:
        import sqlalchemy as sa

        if self._made is None:
            self._made = sa.create_engine(dbos_app.system_database_url(self._dsn))
        return self._made

    def close(self) -> None:
        if self._made is not None:
            self._made.dispose()

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

        if self._unwritten or not sa.inspect(self._engine).has_table(_TABLE):
            return {}  # no worker has ever recorded itself
        cutoff = (datetime.now(UTC) - timedelta(seconds=within_seconds)).isoformat()
        with self._engine.connect() as conn:
            rows = conn.execute(sa.select(self._table).where(self._table.c.updated_at >= cutoff)).fetchall()
        served: dict[str, list[str]] = {}
        for row in rows:
            for queue in json.loads(row.queues_json):
                served.setdefault(queue, []).append(row.worker_id)
        return served


class NoLiveWorker(ValueError):
    """No live worker serves work about to be queued; raised before any row is written."""


def needed_queues(
    config: WomblexConfig, stages: Iterable[str] = (), ingest_root: str | None = None,
) -> dict[str, str]:
    """What a run under *config* needs served: `extraction` (when *ingest_root* is given), each
    of *stages* and the coordinators, each label -> its queue."""
    from womblex.utils.model_check import SCOPE_EXTRACT, models_key

    needed = {"coordinator": dbos_app.DOWNSTREAM_QUEUE} if stages else {}
    if ingest_root is not None:
        needed["extraction"] = dbos_app.extract_queue(ingest_root, models_key(config, SCOPE_EXTRACT))
    needed.update({s: dbos_app.stage_queue(s, models_key(config, s)) for s in stages})
    return needed


def require_served(dsn: str | None, needed: dict[str, str]) -> None:
    """Raise `NoLiveWorker` naming each of *needed* no live worker serves."""
    if not needed:
        return
    with WorkerRegistry(dsn) as registry:
        live = registry.live_queues()
    if missing := [label for label, queue in needed.items() if queue not in live]:
        raise NoLiveWorker(
            f"no live worker serves {', '.join(missing)} under this config: start a worker whose "
            "config names the same models and whose model check passes for them")


__all__ = ["LIVE_SECONDS", "NoLiveWorker", "WorkerRegistry", "needed_queues", "require_served"]
