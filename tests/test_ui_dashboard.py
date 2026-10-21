"""Console dashboard: queue reads and per-stage checkpoint progress (womblex.ui)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Self

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from tests._ui_helpers import _ROWS, _write_manifest_shard
from womblex.ui import dashboard
from womblex.ui.app import create_app


def _write_checkpoint(run_dir: Path, stage: str, **counters: int) -> None:
    """Write a stage checkpoint where the pipeline itself writes one.

    Goes through ``CheckpointManager`` rather than hand-rolling the JSON so
    the dashboard is tested against the real writer's filename and shape.
    """
    from womblex.store.checkpoint import CheckpointManager

    dirname = dashboard.CHECKPOINT_DIRNAMES[stage]
    mgr = CheckpointManager(run_dir / dirname, f"testset_{stage}")
    mgr.load()
    mgr.update(
        [f"doc-{i}" for i in range(counters.get("processed", 1))],
        succeeded=counters.get("succeeded", 1),
        failed=counters.get("failed", 0),
        batch_num=counters.get("batch_num", 1),
    )


def _missing_jobs_table_error() -> Exception:
    """The error a database raises when the DBOS system tables do not exist.

    Postgres names the ``workflow_status`` relation; the message is what
    ``dashboard._is_missing_jobs_table`` recognises, since the driver error
    arrives wrapped by SQLAlchemy.
    """
    return Exception('relation "dbos.workflow_status" does not exist')


class _MissingTableQueue:
    """A reachable ``RunBoard`` whose DBOS system tables have not been created.

    Connecting succeeds (a fresh Postgres before ``init``/enqueue); the first
    read raises ``UndefinedTable``. The dashboard must read this as an empty
    queue, not a fault, and must never create the table.
    """

    schema_ensured_count = 0

    def __init__(self, dsn: str, **_kw: object) -> None:
        self.dsn = dsn

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def stats(self, _run_id: str | None = None) -> dict:
        raise _missing_jobs_table_error()

    def list_jobs(self, *a: object, **k: object) -> list:
        raise _missing_jobs_table_error()

    def workers(self, *a: object, **k: object) -> list:
        raise _missing_jobs_table_error()

    def stale_jobs(self, *a: object, **k: object) -> list:
        raise _missing_jobs_table_error()

    def throughput(self, *a: object, **k: object) -> object:
        raise _missing_jobs_table_error()


class _BrokenQueue(_MissingTableQueue):
    """A reachable queue whose reads fail for a reason that is *not* a missing table.

    Proves the missing-table swallow is narrow: a real failure (here a
    permissions error) still surfaces as ``queue_error``, not a silent empty.
    """

    def stats(self, _run_id: str | None = None) -> dict:
        # A deliberately generic exception: the point is an *arbitrary* failure
        # that is not a missing table, so a custom type would defeat the test.
        raise Exception("permission denied for table womblex_jobs")  # noqa: TRY002


class TestDashboardApi:
    """The Dashboard reads the queue when there is one and the run's own
    per-stage checkpoints always (docs/functional_requirements.md, Requirement 16). These exercise
    the queue-less shape — a plain local deployment — and the checkpoint
    half, which is the same in both.
    """

    def test_checkpoint_dirnames_come_from_the_contracts(self) -> None:
        """The map is derived, not re-typed: a renamed dir cannot drift."""
        from womblex.cloud.stage_contracts import STAGE_CONTRACTS

        assert dashboard.CHECKPOINT_DIRNAMES["chunk"] == ".chunk-checkpoint"
        assert "quality" not in dashboard.CHECKPOINT_DIRNAMES  # whole-run scope, no checkpoint
        for stage, dirname in dashboard.CHECKPOINT_DIRNAMES.items():
            assert STAGE_CONTRACTS[stage].checkpoint_dirname == dirname

    def test_no_queue_is_not_an_error(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        body = client.get("/api/dashboard").json()
        assert body["queue"] is None
        assert body["queue_error"] is None  # absent by configuration, not by failure
        assert body["stages"] == []

    def test_unreachable_queue_reports_rather_than_raises(self, tmp_path: Path) -> None:
        """A dead DSN must not take the checkpoint half of the screen down with it."""
        client = TestClient(create_app(output_root=tmp_path, db_dsn="postgresql://nope:1/nope"))
        _write_checkpoint(tmp_path / "run-a", "chunk", processed=2)
        body = client.get("/api/dashboard", params={"run_id": "run-a"}).json()
        assert body["queue"] is None
        assert body["queue_error"]
        assert [s["stage"] for s in body["stages"]] == ["chunk"]

    def test_a_reachable_queue_with_no_jobs_table_reads_as_empty_not_a_fault(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fresh Postgres before `init`/first-enqueue has no `womblex_jobs`.

        Reading that as "queue unreachable" was a reported symptom: the console
        gates its whole dashboard/execution surface on the queue, so a fresh
        cluster read as a fault (and the enqueue form vanished) until the table
        happened to appear. A reachable-but-schemaless queue is an *empty* queue
        — `queue_error` stays None and the tiles render as zero. The console must
        not create the table (it is read-only; `init`/enqueue own creation).
        """
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _MissingTableQueue)
        client = TestClient(create_app(output_root=tmp_path, db_dsn="postgresql://x/y"))
        body = client.get("/api/dashboard").json()
        assert body["queue_error"] is None
        assert body["queue"] == {
            "stats": {}, "total": 0, "summary": None, "jobs": [], "workers": [], "stale": [],
            "throughput": {
                "window_seconds": dashboard.DEFAULT_THROUGHPUT_WINDOW,
                "completed": 0, "per_minute": 0.0, "last_completed_at": None,
            },
        }

    def test_a_missing_table_never_creates_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The read path must not call ensure_schema — the console never mutates."""
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _MissingTableQueue)
        monkeypatch.setattr(
            "womblex.cloud.dbos_app.ensure_schema",
            lambda _dsn: setattr(
                _MissingTableQueue, "schema_ensured_count",
                _MissingTableQueue.schema_ensured_count + 1,
            ),
        )
        client = TestClient(create_app(output_root=tmp_path, db_dsn="postgresql://x/y"))
        client.get("/api/dashboard")
        assert _MissingTableQueue.schema_ensured_count == 0

    def test_a_real_query_error_still_reports_as_unreachable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only a missing table is swallowed; a genuine failure is not hidden."""
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _BrokenQueue)
        client = TestClient(create_app(output_root=tmp_path, db_dsn="postgresql://x/y"))
        body = client.get("/api/dashboard").json()
        assert body["queue"] is None
        assert "permission denied" in body["queue_error"]

    def test_stage_progress_from_checkpoints(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        client, run_root = api_client
        run_dir = run_root / "run-a"
        _write_manifest_shard(run_dir / "documents", _ROWS)
        _write_checkpoint(run_dir, "chunk", processed=4, succeeded=3, failed=1, batch_num=2)
        _write_checkpoint(run_dir, "pii", processed=1)
        body = client.get("/api/dashboard", params={"run_id": "run-a"}).json()
        by_stage = {s["stage"]: s for s in body["stages"]}
        assert set(by_stage) == {"chunk", "pii"}
        assert by_stage["chunk"]["processed"] == 4
        assert by_stage["chunk"]["succeeded"] == 3
        assert by_stage["chunk"]["failed"] == 1
        assert by_stage["chunk"]["last_batch"] == 2
        assert by_stage["chunk"]["name"] == "testset_chunk"

    def test_stages_empty_without_a_run_id(self, api_client: tuple[TestClient, Path]) -> None:
        """Checkpoints are per-run artefacts, so there is no cross-run answer."""
        client, run_root = api_client
        _write_checkpoint(run_root / "run-a", "chunk", processed=2)
        assert client.get("/api/dashboard").json()["stages"] == []

    def test_missing_run_is_empty_not_404(self, api_client: tuple[TestClient, Path]) -> None:
        """An enqueued run whose first batch hasn't landed has no directory yet."""
        client, _ = api_client
        resp = client.get("/api/dashboard", params={"run_id": "run-nope"})
        assert resp.status_code == 200
        assert resp.json()["stages"] == []

    def test_rejects_a_non_positive_window(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/dashboard", params={"stale_after": 0}).status_code == 422

    def test_unreadable_checkpoint_is_skipped(self, tmp_path: Path) -> None:
        """One corrupt checkpoint narrows the answer; it doesn't blank the screen."""
        from womblex.store.checkpoint import read_checkpoints

        ckpt_dir = tmp_path / ".chunk-checkpoint"
        ckpt_dir.mkdir()
        (ckpt_dir / "bad_checkpoint.json").write_text("{not json")
        (ckpt_dir / "good_checkpoint.json").write_text(
            json.dumps({"total_processed": 3, "started_at": "", "updated_at": ""})
        )
        progress = read_checkpoints(ckpt_dir)
        assert [p.name for p in progress] == ["good"]
        # Unusable timestamps mean no rate to report, rather than a fabricated one.
        assert progress[0].documents_per_minute is None

    def test_documents_per_minute_spans_the_checkpoint(self, tmp_path: Path) -> None:
        from womblex.store.checkpoint import read_checkpoints

        ckpt_dir = tmp_path / ".chunk-checkpoint"
        ckpt_dir.mkdir()
        (ckpt_dir / "run_checkpoint.json").write_text(
            json.dumps({
                "total_processed": 120,
                "started_at": "2026-08-16T00:00:00+00:00",
                "updated_at": "2026-08-16T00:02:00+00:00",
            })
        )
        assert read_checkpoints(ckpt_dir)[0].documents_per_minute == 60.0

    def test_run_id_cannot_escape_the_run_root(self, tmp_path: Path) -> None:
        """`run_id` is a query param, so nothing upstream rejects `../`.

        The remote branch is contained by its `list_dirs` check; the local
        join has to contain itself, or the two deployments disagree about
        what is reachable.
        """
        outside = tmp_path / "outside"
        _write_checkpoint(outside, "chunk", processed=9)
        run_root = tmp_path / "runs"
        run_root.mkdir()  # must exist, or the OS cannot resolve `..` through it
        client = TestClient(create_app(output_root=run_root))
        body = client.get("/api/dashboard", params={"run_id": "../outside"}).json()
        assert body["stages"] == []

    def test_malformed_checkpoints_are_skipped_not_raised(self, tmp_path: Path) -> None:
        """Well-formed JSON can still be unusable; the reader must survive it."""
        from womblex.store.checkpoint import read_checkpoints

        ckpt_dir = tmp_path / ".chunk-checkpoint"
        ckpt_dir.mkdir()
        # A null timestamp reaches `datetime.fromisoformat(None)`; a top-level
        # list reaches `.get` on a list. Both parse as JSON, neither is a
        # checkpoint, and neither may escape as a 500.
        (ckpt_dir / "null_checkpoint.json").write_text(
            json.dumps({"total_processed": 1, "started_at": None, "updated_at": None})
        )
        (ckpt_dir / "list_checkpoint.json").write_text("[1, 2, 3]")
        (ckpt_dir / "ok_checkpoint.json").write_text(
            json.dumps({"total_processed": 1, "started_at": "", "updated_at": ""})
        )
        progress = {p.name: p for p in read_checkpoints(ckpt_dir)}
        # The list is not a checkpoint at all, so it is dropped. The null
        # timestamps are: its counters are real progress and are kept — only
        # the rate they cannot support is withheld.
        assert set(progress) == {"null", "ok"}
        assert progress["null"].processed == 1
        assert progress["null"].documents_per_minute is None

    def test_no_rate_from_a_single_write(self, tmp_path: Path) -> None:
        """A just-started run's two timestamps are one write, not an interval."""
        from womblex.store.checkpoint import read_checkpoints

        ckpt_dir = tmp_path / ".chunk-checkpoint"
        ckpt_dir.mkdir()
        (ckpt_dir / "run_checkpoint.json").write_text(
            json.dumps({
                "total_processed": 3,
                "started_at": "2026-08-16T00:00:00.000100+00:00",
                "updated_at": "2026-08-16T00:00:00.000200+00:00",
            })
        )
        # Dividing by a 0.1ms span would report ~1.8M documents/minute.
        assert read_checkpoints(ckpt_dir)[0].documents_per_minute is None
