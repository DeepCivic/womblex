"""A second worker on a live executor id is refused; a dead predecessor's id is reusable."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import sqlalchemy as sa

from womblex.cloud import worker
from womblex.cloud.worker_registry import WorkerRegistry


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WOMBLEX_DBOS_PATH", str(tmp_path / "dbos.sqlite"))
    with WorkerRegistry(None) as reg:
        yield reg


def _age(reg: WorkerRegistry, worker_id: str, seconds: float) -> None:
    stamp = (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat()
    with reg._engine.begin() as conn:
        conn.execute(
            sa.update(reg._table).where(reg._table.c.worker_id == worker_id).values(updated_at=stamp)
        )


def test_no_row_means_the_id_is_free(registry) -> None:
    assert registry.refreshed_ago("host") is None
    worker._refuse_held_id(registry, "host")


def test_a_freshly_refreshed_row_refuses_the_id(registry) -> None:
    registry.record("host", ["q"])
    with WorkerRegistry(None) as later, pytest.raises(worker.WorkerIdInUse, match="host"):
        worker._refuse_held_id(later, "host")


def test_a_stale_row_from_a_dead_predecessor_is_reusable(registry) -> None:
    registry.record("host", ["q"])
    _age(registry, "host", 3 * worker.HEARTBEAT_SECONDS)
    with WorkerRegistry(None) as later:
        worker._refuse_held_id(later, "host")


def test_a_stopped_worker_leaves_no_row(registry) -> None:
    registry.record("host", ["q"])
    registry.remove("host")
    worker._refuse_held_id(registry, "host")
