"""Run-time behaviour from the ism-au feedback: re-run guidance and log noise."""

import argparse
import logging
import shutil
from pathlib import Path

import pytest

from tests.test_pipeline import _CSV_FILE, _write_minimal_config
from womblex.cli.pipeline import cmd_run
from womblex.config import ChunkingConfig, DatasetConfig, EnrichmentConfig, WomblexConfig
from womblex.utils.token_packer import quiet_length_warning


def _run(cfg: Path, **kw: object) -> int:
    base = {"resume": False, "no_verify_resume": False, "limit": None, "skip": 0,
            "batch_size": 1, "run_id": "r"}
    return cmd_run(argparse.Namespace(config=cfg, **{**base, **kw}))


class TestRerunGuidance:
    def test_fresh_run_into_a_populated_run_says_to_use_resume(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        input_root = tmp_path / "in"
        input_root.mkdir()
        shutil.copy(_CSV_FILE, input_root / "a.csv")
        cfg = _write_minimal_config(tmp_path, input_root)
        assert _run(cfg) == 0
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="womblex"):
            assert _run(cfg) == 0
        assert any("--resume" in r.getMessage() and "extracted" in r.getMessage()
                   for r in caplog.records)

    def test_first_run_and_resume_do_not_warn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        input_root = tmp_path / "in"
        input_root.mkdir()
        shutil.copy(_CSV_FILE, input_root / "a.csv")
        cfg = _write_minimal_config(tmp_path, input_root)
        with caplog.at_level(logging.WARNING, logger="womblex"):
            assert _run(cfg) == 0
            shutil.copy(_CSV_FILE, input_root / "b.csv")
            assert _run(cfg, resume=True) == 0
        assert not [r for r in caplog.records if "--resume" in r.getMessage()]


class TestLogNoise:
    def test_token_length_warning_is_dropped_and_others_kept(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        quiet_length_warning()
        quiet_length_warning()  # idempotent
        lg = logging.getLogger("transformers.tokenization_utils_base")
        assert len([f for f in lg.filters if type(f).__name__ == "_LengthWarningFilter"]) == 1
        with caplog.at_level(logging.WARNING):
            lg.warning("Token indices sequence length is longer than x (2 > 1)")
            lg.warning("something else")
        assert [r.getMessage() for r in caplog.records] == ["something else"]

    def test_ai_chunking_reuse_warning_is_logged_once_per_process(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        kwargs = {
            "dataset": DatasetConfig(name="t"),
            "paths": {"input_root": "/tmp/in", "output_root": "/tmp/out",
                      "checkpoint_dir": "/tmp/c"},
            "chunking": ChunkingConfig(chunking_model="kanon-2-enricher"),
            "enrichment": EnrichmentConfig(enabled=True),
        }
        with caplog.at_level(logging.WARNING):
            first = WomblexConfig(**kwargs)
            second = WomblexConfig(**kwargs)
        assert first.enrichment.persist_document and second.enrichment.persist_document
        assert len([r for r in caplog.records if "persist_document" in r.getMessage()]) == 1
