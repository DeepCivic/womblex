"""Queue dispatch's refusals, before any row (`cloud/dispatch.py`).

Every write action refuses a deployment that cannot run it, and bad input,
before the board is touched. The board is faked: a refusal must not need one.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Self

import pytest

from womblex.cloud import dispatch
from womblex.cloud.deployment import DeploymentSettings
from womblex.cloud.workflow_order import StageOrderError

pytest.importorskip("fsspec")

_DSN = "postgresql://x/y"
#: Chunking and money are on by default; this turns every downstream stage off.
_NO_STAGES = {"chunking": {"enabled": False}, "money": {"enabled": False}}


class _Board:
    """A `RunBoard` stand-in that records what reached it."""

    calls: ClassVar[list[tuple]] = []

    def __init__(self, dsn: str, **_kw: object) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def enqueue(self, run_id: str, specs: list, **kw: object) -> int:
        _Board.calls.append(("enqueue", run_id, len(specs)))
        return len(specs)

    def enqueue_stages(self, run_id: str, stages: list, shard_prefix: str, **kw: object) -> int:
        _Board.calls.append(("stages", run_id, tuple(stages)))
        return 1


@pytest.fixture(autouse=True)
def _fake_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    _Board.calls = []
    monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _Board)
    monkeypatch.setattr("womblex.cloud.dispatch.ensure_schema", lambda _dsn: None)
    monkeypatch.setattr("womblex.cloud.worker_registry.require_served", lambda _dsn, _needed: None)


def _settings(tmp_path: Path, *, ingest: bool = True, dsn: str | None = _DSN) -> DeploymentSettings:
    ingest_root = tmp_path / "inbox"
    ingest_root.mkdir(exist_ok=True)
    return DeploymentSettings(
        output_root=None, store_uri=str(tmp_path / "store"), db_dsn=dsn,
        ingest_uri=str(ingest_root) if ingest else None,
    )


def _seed(settings: DeploymentSettings, *names: str) -> None:
    for name in names:
        (Path(str(settings.ingest_uri)) / name).write_bytes(b"%PDF-1.4 stub")


class TestStatus:
    def test_all_three_pieces_can_execute(self, tmp_path: Path) -> None:
        cap = dispatch.execution_status(_settings(tmp_path))
        assert cap.can_execute is True
        assert cap.as_dict()["ingest_uri"] == str(tmp_path / "inbox")

    def test_a_local_root_cannot(self, tmp_path: Path) -> None:
        cap = dispatch.execution_status(DeploymentSettings(output_root=tmp_path, store_uri=None))
        assert cap.as_dict() == {
            "can_execute": False, "has_store": False, "has_ingest": False, "has_queue": False,
            "ingest_uri": None, "output_uri": str(tmp_path),
        }


class TestGuard:
    """Store, ingest, queue — the most actionable gap first."""

    def test_no_store(self, tmp_path: Path) -> None:
        settings = DeploymentSettings(output_root=tmp_path, store_uri=None, db_dsn=_DSN,
                                      ingest_uri=str(tmp_path / "inbox"))
        with pytest.raises(dispatch.ExecutionDisabled) as e:
            dispatch.enqueue_extraction(settings)
        assert e.value.reason == "no_store"

    def test_no_ingest(self, tmp_path: Path) -> None:
        with pytest.raises(dispatch.ExecutionDisabled) as e:
            dispatch.enqueue_extraction(_settings(tmp_path, ingest=False))
        assert e.value.reason == "no_ingest"

    def test_stages_need_no_ingest(self, tmp_path: Path) -> None:
        result = dispatch.enqueue_downstream_stages(
            _settings(tmp_path, ingest=False), run_id="r1", config={"chunking": {"enabled": False}},
            order_checked=True,
        )
        assert result.as_dict()["stages"] == ["money"]

    def test_no_queue(self, tmp_path: Path) -> None:
        for call in (
            lambda s: dispatch.enqueue_extraction(s),
            lambda s: dispatch.enqueue_downstream_stages(s, run_id="r1", config={}),
        ):
            with pytest.raises(dispatch.ExecutionDisabled) as e:
                call(_settings(tmp_path, dsn=None))
            assert e.value.reason == "no_queue"
        assert _Board.calls == []


class TestEnqueueExtraction:
    def test_batches_the_supported_documents(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        _seed(settings, "a.pdf", "b.pdf", "c.pdf")
        result = dispatch.enqueue_extraction(settings, run_id="r1", batch_size=2)
        assert result.as_dict() == {
            "run_id": "r1", "document_count": 3, "batch_count": 2, "newly_enqueued": 2,
            "shard_prefix": "runs/r1/documents",
        }

    @pytest.mark.parametrize(("kwargs", "match"), [
        ({"batch_size": 0}, "batch_size"),
        ({"run_id": "../escape"}, "unsafe run_id"),
    ])
    def test_bad_input_is_refused(self, tmp_path: Path, kwargs: dict, match: str) -> None:
        settings = _settings(tmp_path)
        _seed(settings, "a.pdf")
        with pytest.raises(ValueError, match=match):
            dispatch.enqueue_extraction(settings, **kwargs)
        assert _Board.calls == []

    def test_no_supported_documents_is_refused(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        (Path(str(settings.ingest_uri)) / "notes.xyz").write_text("x")
        with pytest.raises(ValueError, match="no supported documents"):
            dispatch.enqueue_extraction(settings)
        assert _Board.calls == []


class TestEnqueueStages:
    def test_an_unsafe_run_id_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unsafe run_id"):
            dispatch.enqueue_downstream_stages(_settings(tmp_path), run_id="../x", config={})

    def test_a_config_enabling_no_stage_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no downstream stages"):
            dispatch.enqueue_downstream_stages(_settings(tmp_path), run_id="r1", config=_NO_STAGES)
        assert _Board.calls == []

    def test_a_stale_input_is_refused_before_any_row(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        docs = tmp_path / "store/runs/r1/documents"
        docs.mkdir(parents=True)
        for name in ("b1.elements.parquet", "b1.table_cells.parquet", "b1._manifest.parquet",
                     "b1.normalised_text.parquet"):
            (docs / name).write_bytes(b"")
        config = {"processing": {"text_source": "normalised"}, "money": {"enabled": False}}
        with pytest.raises(StageOrderError):
            dispatch.check_stage_order(settings, config, ("chunk", "normalise"), "r1")
        result = dispatch.enqueue_downstream_stages(settings, run_id="r1", config=config)
        assert result.stages == ("normalise", "chunk")
        assert _Board.calls == [("stages", "r1", ("normalise", "chunk"))]
