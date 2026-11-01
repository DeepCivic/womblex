"""`womblex run` carries on past a batch that fails its post-write integrity check."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

from tests._synthetic import REGISTER_CSV
from womblex.cli.pipeline import cmd_run
from womblex.store import output
from womblex.store.run_manifest import read_run_record


def _config(tmp_path: Path, input_root: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"dataset:\n  name: integ\n"
        f"paths:\n  input_root: {input_root}\n  output_root: {tmp_path / 'out'}\n"
        f"  checkpoint_dir: {tmp_path / 'ckpt'}\n"
        "redaction:\n  enabled: false\n"
    )
    return cfg


def _args(cfg: Path, *, resume: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        config=cfg, resume=resume, no_verify_resume=False, limit=None, skip=0,
        batch_size=1, run_id="r",
    )


@pytest.fixture
def failing_first_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Three one-document batches, the first of which fails verification."""
    input_root = tmp_path / "in"
    input_root.mkdir()
    for i in range(3):
        shutil.copy(REGISTER_CSV, input_root / f"doc-{i:02d}.csv")
    real = output.verify_shard_persistence

    def verify(path: Path, expected: int, prev: int) -> int:
        if path.name == "batch-0001.parquet":
            raise output.ShardVerificationError(f"forced failure: {path}")
        return real(path, expected, prev)

    monkeypatch.setattr(output, "verify_shard_persistence", verify)
    return _config(tmp_path, input_root)


def test_failed_check_runs_later_batches_and_lands(failing_first_batch: Path, tmp_path: Path) -> None:
    assert cmd_run(_args(failing_first_batch)) == 1

    shard_dir = tmp_path / "out" / "r" / "documents"
    for n in (1, 2, 3):
        assert (shard_dir / f"batch-{n:04d}._manifest.parquet").exists()
    record = read_run_record(tmp_path / "out" / "r" / "manifest.parquet")
    assert record is not None
    assert any("shard integrity" in p and "batch 1 " in p for p in record["partial"])

    state = json.loads((tmp_path / "ckpt" / "r" / "integ_checkpoint.json").read_text())
    assert state["last_batch"] == 3
    assert [f["batch"] for f in state["failed_batches"]] == [1]
    assert "doc-00" not in state["processed_ids"]
    assert {"doc-01", "doc-02"} <= set(state["processed_ids"])


def test_resume_re_extracts_a_failed_batch(
    failing_first_batch: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert cmd_run(_args(failing_first_batch)) == 1
    monkeypatch.undo()  # the check passes from now on

    assert cmd_run(_args(failing_first_batch, resume=True)) == 0
    shard_dir = tmp_path / "out" / "r" / "documents"
    assert not (shard_dir / "batch-0001._manifest.parquet").exists()
    assert (shard_dir / "batch-0001._manifest.parquet.corrupt").exists()
    assert (shard_dir / "batch-0004._manifest.parquet").exists()
    state = json.loads((tmp_path / "ckpt" / "r" / "integ_checkpoint.json").read_text())
    assert state["failed_batches"] == []
    assert "doc-00" in state["processed_ids"]
