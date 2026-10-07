"""``run-stage layout``: fingerprint-aware reruns over a local shard directory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

import womblex.utils.model_registry as reg
from tests._pdf_builders import PdfBuilder
from womblex.batch import process_batch
from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.cloud.stage_runner import StagePreconditionError, run_stage_local, run_stage_remote
from womblex.config import DatasetConfig, PathsConfig, WomblexConfig
from womblex.ingest.interfaces.protocols import LayoutRegionResult
from womblex.ingest.pdf.types import Rect
from womblex.process.layout_stage import layout_shards
from womblex.store.layout_output import (
    MATCH,
    MISMATCH,
    UNKNOWN,
    layout_fingerprint_status,
    layout_fingerprint_statuses,
    read_footer_layout_fingerprint,
    read_layout_regions,
)


class _Analyzer:
    def __init__(self, label: str) -> None:
        self.label = label

    def analyze(self, img: np.ndarray, conf_threshold: float = 0.3) -> list[LayoutRegionResult]:
        h, w = img.shape[:2]
        return [LayoutRegionResult((0.0, 0.0, w / 2, h / 2), self.label, "paragraph", 0.9)]


@pytest.fixture(autouse=True)
def _fake_models():
    for name in ("stage-a", "stage-b"):
        reg.register(reg.SLOT_LAYOUT, name, lambda _n=name, **_: _Analyzer(_n), source="test-dist")


def _config(tmp_path: Path, model: str) -> WomblexConfig:
    return WomblexConfig(
        dataset=DatasetConfig(name="t"),
        paths=PathsConfig(input_root=tmp_path, output_root=tmp_path, checkpoint_dir=tmp_path),
        layout={"model": model},
        redaction={"enabled": False},
    )


@pytest.fixture
def run(tmp_path: Path):
    """A one-PDF, one-CSV batch extracted under ``stage-a``: ``(corpus, shards)``."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    img = Image.fromarray(np.full((200, 200, 3), 128, dtype=np.uint8))
    pdf = PdfBuilder(corpus / "scan.pdf").page().image(Rect(50, 50, 450, 450), img).save()
    (corpus / "t.csv").write_text("a,b\n1,2\n")
    shards = tmp_path / "shards"
    process_batch([pdf, corpus / "t.csv"], _config(tmp_path, "stage-a"), batch_num=1, shard_dir=shards)
    return corpus, shards


def _label(shards: Path) -> set[str]:
    return {r["label"] for r in read_layout_regions(shards).to_pylist()}


def _sidecar(shards: Path) -> Path:
    return shards / "batch-0001.layout_regions.parquet"


def test_extraction_leaves_the_sidecar_matching_the_elements(run) -> None:
    _, shards = run
    assert layout_fingerprint_statuses(shards) == {"batch-0001": MATCH}


def test_the_same_fingerprint_is_skipped(tmp_path: Path, run) -> None:
    corpus, shards = run
    before = _sidecar(shards).stat().st_mtime_ns
    result = layout_shards(shards, _config(tmp_path, "stage-a"), source_root=corpus)
    assert (result.batches_skipped, result.batches_written) == (1, 0)
    assert _sidecar(shards).stat().st_mtime_ns == before


def test_a_new_model_reruns_and_the_two_footers_then_disagree(tmp_path: Path, run) -> None:
    corpus, shards = run
    assert _label(shards) == {"stage-a"}
    result = layout_shards(shards, _config(tmp_path, "stage-b"), source_root=corpus)
    assert (result.batches_written, result.documents) == (1, 1)  # the CSV has no layout
    assert _label(shards) == {"stage-b"}
    assert layout_fingerprint_status(shards / "batch-0001.parquet") == MISMATCH
    fp = read_footer_layout_fingerprint(pq.read_metadata(str(_sidecar(shards))).metadata)
    assert fp is not None and fp.model == "stage-b"


def test_force_reruns_an_unchanged_fingerprint(tmp_path: Path, run) -> None:
    corpus, shards = run
    result = layout_shards(shards, _config(tmp_path, "stage-a"), source_root=corpus, force=True)
    assert result.batches_written == 1


def test_an_unresolvable_source_leaves_the_batch_as_it_was(tmp_path: Path, run) -> None:
    _, shards = run
    empty = tmp_path / "elsewhere"
    empty.mkdir()
    result = layout_shards(shards, _config(tmp_path, "stage-b"), source_root=empty)
    assert (result.batches_failed, result.batches_written) == (1, 0)
    assert _label(shards) == {"stage-a"}


def test_a_run_from_before_the_layout_stage_reads_as_unknown(run) -> None:
    _, shards = run
    _sidecar(shards).unlink()
    assert layout_fingerprint_status(shards / "batch-0001.parquet") == UNKNOWN


def test_the_contract_runs_locally_and_refuses_a_store(tmp_path: Path, run) -> None:
    from womblex.cloud.stage_contracts import RunContext

    corpus, shards = run
    contract = STAGE_CONTRACTS["layout"]
    run_stage_local(
        contract, shards, _config(tmp_path, "stage-b"),
        ctx=RunContext(source_root=str(corpus)),
    )
    assert _label(shards) == {"stage-b"}
    with pytest.raises(StagePreconditionError, match="--shards"):
        run_stage_remote(contract, object(), "p", _config(tmp_path, "stage-b"))  # type: ignore[arg-type]
