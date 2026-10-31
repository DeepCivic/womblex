"""The inbound events a deployment has seen: `womblex_inbound_events` in the system database.

An inbound webhook (Requirement 35) carries an `event_id` its sender keeps when it resends. The
route inserts the id first: a primary-key conflict means the event was seen before, and the stored
`outcome_json` (an acceptance or a refusal) is what a resend gets back. Same DDL pattern as
`womblex_workflows` and `womblex_workers`: created if absent by a writer, no versioned migration.
Storage only; the route (`api/routes_inbound.py`) decides what an event does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Self

from womblex.cloud import dbos_app

_TABLE = "womblex_inbound_events"


def _ddl() -> Any:
    import sqlalchemy as sa

    metadata = sa.MetaData()
    return sa.Table(
        _TABLE, metadata,
        sa.Column("event_id", sa.Text, primary_key=True),
        sa.Column("workflow_id", sa.Text, nullable=False),
        sa.Column("hook", sa.Text, nullable=False),
        sa.Column("event", sa.Text, nullable=False),
        sa.Column("received_at", sa.Text, nullable=False),
        sa.Column("outcome_json", sa.Text, nullable=True),
        sa.Index("womblex_inbound_events_by_workflow", "workflow_id", "received_at"),
    ), metadata


def ensure_inbound_schema(dsn: str | None) -> None:
    """Create the table if it is not already there. Idempotent; writers call it, readers do not."""
    import sqlalchemy as sa

    table, metadata = _ddl()
    engine = sa.create_engine(dbos_app.system_database_url(dsn))
    try:
        metadata.create_all(engine, tables=[table])
    finally:
        engine.dispose()


@dataclass(frozen=True)
class InboundEvent:
    """One event as stored; `outcome` is `None` while its checks have not finished."""

    event_id: str
    workflow_id: str
    hook: str
    event: str
    received_at: str
    outcome: dict[str, Any] | None


def _event(row: Any) -> InboundEvent:
    return InboundEvent(row.event_id, row.workflow_id, row.hook, row.event, row.received_at,
                        json.loads(row.outcome_json) if row.outcome_json else None)


class InboundEvents:
    """Insert, settle and read inbound events through one SQLAlchemy engine."""

    def __init__(self, dsn: str | None):
        import sqlalchemy as sa

        self._table, _ = _ddl()
        self._engine = sa.create_engine(dbos_app.system_database_url(dsn))

    def close(self) -> None:
        self._engine.dispose()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def begin(self, event_id: str, workflow_id: str, hook: str, event: str) -> InboundEvent | None:
        """Record *event_id* as received. `None` if it is new; the stored event if it was seen before."""
        import sqlalchemy as sa
        import sqlalchemy.exc

        try:
            with self._engine.begin() as conn:
                conn.execute(sa.insert(self._table).values(
                    event_id=event_id, workflow_id=workflow_id, hook=hook, event=event,
                    received_at=datetime.now(UTC).isoformat(), outcome_json=None))
        except sqlalchemy.exc.IntegrityError:
            seen = self.get(event_id)
            if seen is None:  # deleted between the conflict and the read
                return self.begin(event_id, workflow_id, hook, event)
            return seen
        return None

    def settle(self, event_id: str, outcome: dict[str, Any]) -> None:
        """Store *outcome* (accepted or refused, with its code) for *event_id*."""
        import sqlalchemy as sa

        with self._engine.begin() as conn:
            conn.execute(sa.update(self._table).where(self._table.c.event_id == event_id)
                         .values(outcome_json=json.dumps(outcome, sort_keys=True)))

    def forget(self, event_id: str) -> None:
        """Drop *event_id* before any outcome was stored, so a failed handling can be resent under its id."""
        import sqlalchemy as sa

        with self._engine.begin() as conn:
            conn.execute(sa.delete(self._table).where(sa.and_(
                self._table.c.event_id == event_id, self._table.c.outcome_json.is_(None))))

    def get(self, event_id: str) -> InboundEvent | None:
        import sqlalchemy as sa

        with self._engine.connect() as conn:
            row = conn.execute(sa.select(self._table).where(self._table.c.event_id == event_id)).first()
        return None if row is None else _event(row)

    def list(self, workflow_id: str, *, limit: int = 100) -> list[InboundEvent]:
        """A workflow's events, newest first."""
        import sqlalchemy as sa

        stmt = (sa.select(self._table).where(self._table.c.workflow_id == workflow_id)
                .order_by(self._table.c.received_at.desc()).limit(limit))
        with self._engine.connect() as conn:
            return [_event(r) for r in conn.execute(stmt).fetchall()]


__all__ = ["InboundEvent", "InboundEvents", "ensure_inbound_schema"]
