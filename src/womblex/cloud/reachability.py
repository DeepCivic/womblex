"""Whether a deployment's store and queue answer — the service API's readiness probes.

Each reports ``(reachable, error)`` rather than raising: a caller that is
asked "is it up?" wants the answer and the reason, not an exception.
"""
from __future__ import annotations

import logging

from womblex.cloud.dispatch import QUEUE_CONNECT_TIMEOUT

logger = logging.getLogger(__name__)


def store_reachable(store_uri: str, explicit: dict | None = None) -> tuple[bool, str | None]:
    """Whether the object store at *store_uri* answers a listing of its ``runs/`` prefix.

    A valid but still-empty bucket has no ``runs/`` prefix (an object store has
    no real directories), so what is checked is that the listing *completed*; a
    store that cannot be reached fails because the connection error propagates
    out of the listing rather than being flattened into an empty result.
    """
    try:
        from womblex.store.remote import RemoteStore

        RemoteStore.from_uri(store_uri, explicit=explicit).list_dirs("runs")
    except Exception as e:
        logger.warning("store unreachable: %s", e)
        return False, str(e)
    return True, None


def is_missing_schema(exc: Exception) -> bool:
    """Whether *exc* says the DBOS system tables do not exist yet (Postgres or SQLite wording)."""
    text = str(exc).lower()
    return "workflow_status" in text and ("exist" in text or "no such table" in text)


def queue_reachable(db_dsn: str | None) -> tuple[bool, str | None]:
    """Whether the job queue at *db_dsn* answers.

    A reachable database whose DBOS system tables do not exist yet (a fresh
    Postgres no worker or dispatcher has written to) is an empty queue, not a
    fault. Anything else, a failed connection or a permissions error, is not.
    """
    if not db_dsn:
        return False, "no queue configured"
    try:
        from womblex.cloud.jobs import RunBoard

        with RunBoard(db_dsn, connect_timeout=QUEUE_CONNECT_TIMEOUT) as board:
            try:
                board.stats()
            except Exception as e:
                if is_missing_schema(e):
                    return True, None
                raise
    except Exception as e:
        logger.warning("queue unavailable: %s", e)
        return False, str(e)
    return True, None
