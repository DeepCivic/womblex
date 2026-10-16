"""Tests for womblex.redact.batch and the redact CLI.

Shard annotation, label validation, ``womblex redact --shards`` and the
layout filter on raster pages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests._pdf_builders import PdfBuilder
from womblex.cli.redact import cmd_annotate_redactions, cmd_redact
from womblex.config import RedactionConfig
from womblex.ingest.pdf.types import Rect
from womblex.redact.batch import (
    REDACTIONS_SCHEMA,
    annotate_redactions_for_shards,
    validate_redactions_against_labels,
)
from womblex.redact.stage import (
    build_detector,
    detect_redactions,
)

# ---------------------------------------------------------------------------
# Batch operations: annotate_redactions_for_shards + validate_redactions_against_labels
# ---------------------------------------------------------------------------


class TestRedactionBatch:
    def test_redactions_schema_columns(self) -> None:
        names = REDACTIONS_SCHEMA.names
        assert names == ["source_hash", "elem_order", "has_redaction"]

    def test_annotate_returns_empty_summary_for_empty_shard_dir(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        pdf_dir = tmp_path / "pdfs"
        shard_dir.mkdir()
        pdf_dir.mkdir()

        summary = annotate_redactions_for_shards(shard_dir, pdf_dir)
        assert summary == {}

    def test_validate_returns_empty_for_empty_labels_dir(self, tmp_path: Path) -> None:
        labels_dir = tmp_path / "labels"
        pdf_dir = tmp_path / "pdfs"
        labels_dir.mkdir()
        pdf_dir.mkdir()

        summaries = validate_redactions_against_labels(labels_dir, pdf_dir)
        assert summaries == []

    def test_annotate_writes_empty_sidecar_when_no_redactions(self, tmp_path: Path) -> None:
        """A batch whose source PDF has no detectable redactions should still produce
        a redactions.parquet (empty) so downstream consumers can rely on the file."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        shard_dir = tmp_path / "shards"
        pdf_dir = tmp_path / "pdfs"
        shard_dir.mkdir()
        pdf_dir.mkdir()

        # Programmatic PDF with no redactions (a blank page)
        PdfBuilder(pdf_dir / "blank.pdf").page().save()

        # Minimal manifest + elements
        manifest_tbl = pa.table({"source_hash": ["h1"], "filename": ["blank.pdf"]})
        pq.write_table(manifest_tbl, shard_dir / "batch-0001._manifest.parquet")
        elements_tbl = pa.table({
            "source_hash": ["h1"],
            "elem_order": pa.array([0], type=pa.int32()),
            "page": pa.array([0], type=pa.int32()),
        })
        pq.write_table(elements_tbl, shard_dir / "batch-0001.elements.parquet")

        summary = annotate_redactions_for_shards(shard_dir, pdf_dir)

        out_path = shard_dir / "batch-0001.redactions.parquet"
        assert out_path.exists()
        sidecar = pq.read_table(out_path)
        assert sidecar.num_rows == 0
        assert sidecar.schema.names == ["source_hash", "elem_order", "has_redaction"]
        assert summary == {"h1": 0}

    def test_annotate_writes_rows_for_affected_elements(self, tmp_path: Path) -> None:
        """When a PDF has a detectable redaction and the manifest has elements on that
        page, the sidecar carries one row per affected element."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        shard_dir = tmp_path / "shards"
        pdf_dir = tmp_path / "pdfs"
        shard_dir.mkdir()
        pdf_dir.mkdir()

        # PDF with a fat black rectangle on page 0
        PdfBuilder(pdf_dir / "redacted.pdf").page(600, 400).rect(Rect(100, 100, 300, 140)).save()

        manifest_tbl = pa.table({"source_hash": ["h2"], "filename": ["redacted.pdf"]})
        pq.write_table(manifest_tbl, shard_dir / "batch-0002._manifest.parquet")
        elements_tbl = pa.table({
            "source_hash": ["h2", "h2", "h2"],
            "elem_order": pa.array([0, 1, 2], type=pa.int32()),
            "page": pa.array([0, 0, 1], type=pa.int32()),  # 2 on affected page, 1 not
        })
        pq.write_table(elements_tbl, shard_dir / "batch-0002.elements.parquet")

        summary = annotate_redactions_for_shards(shard_dir, pdf_dir)

        sidecar = pq.read_table(shard_dir / "batch-0002.redactions.parquet")
        assert sidecar.num_rows == 2  # elem_orders 0 and 1 are on page 0
        assert set(sidecar.column("elem_order").to_pylist()) == {0, 1}
        assert all(sidecar.column("has_redaction").to_pylist())
        assert summary["h2"] >= 1

    def test_annotate_resumes_from_checkpoint(self, tmp_path: Path) -> None:
        """A batch listed in the checkpoint should be skipped on the second run;
        new batches should still be processed."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        shard_dir = tmp_path / "shards"
        pdf_dir = tmp_path / "pdfs"
        shard_dir.mkdir()
        pdf_dir.mkdir()
        checkpoint_path = tmp_path / "redactions_checkpoint.json"

        # Build a single batch + matching blank PDF
        PdfBuilder(pdf_dir / "blank.pdf").page().save()
        pq.write_table(
            pa.table({"source_hash": ["h1"], "filename": ["blank.pdf"]}),
            shard_dir / "batch-0001._manifest.parquet",
        )
        pq.write_table(
            pa.table({
                "source_hash": ["h1"],
                "elem_order": pa.array([0], type=pa.int32()),
                "page": pa.array([0], type=pa.int32()),
            }),
            shard_dir / "batch-0001.elements.parquet",
        )

        # First run: processes the batch, writes checkpoint
        annotate_redactions_for_shards(shard_dir, pdf_dir, checkpoint_path=checkpoint_path)
        assert checkpoint_path.exists()
        state = json.loads(checkpoint_path.read_text())
        assert "batch-0001" in state["processed_batches"]

        # Delete the sidecar to detect re-processing; second run should skip
        (shard_dir / "batch-0001.redactions.parquet").unlink()
        annotate_redactions_for_shards(shard_dir, pdf_dir, checkpoint_path=checkpoint_path)
        assert not (shard_dir / "batch-0001.redactions.parquet").exists()

    def test_annotate_skips_missing_pdf(self, tmp_path: Path) -> None:
        """A manifest pointing to a non-existent PDF should be skipped gracefully."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        shard_dir = tmp_path / "shards"
        pdf_dir = tmp_path / "pdfs"
        shard_dir.mkdir()
        pdf_dir.mkdir()

        manifest_tbl = pa.table({"source_hash": ["h3"], "filename": ["nonexistent.pdf"]})
        pq.write_table(manifest_tbl, shard_dir / "batch-0003._manifest.parquet")
        elements_tbl = pa.table({
            "source_hash": ["h3"],
            "elem_order": pa.array([0], type=pa.int32()),
            "page": pa.array([0], type=pa.int32()),
        })
        pq.write_table(elements_tbl, shard_dir / "batch-0003.elements.parquet")

        summary = annotate_redactions_for_shards(shard_dir, pdf_dir)
        # Missing PDF → no summary entry, no crash, empty sidecar still written
        assert "h3" not in summary
        assert (shard_dir / "batch-0003.redactions.parquet").exists()


# ---------------------------------------------------------------------------
# CLI surface: `womblex redact --shards` (I3) + `annotate-redactions` alias
# ---------------------------------------------------------------------------


def _seed_redaction_shards(tmp_path: Path) -> tuple[Path, Path]:
    """Build a one-batch shard dir + a PDF with a detectable redaction on page 0.

    Returns ``(shard_dir, pdf_dir)``. Mirrors the engine-level fixtures above.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard_dir = tmp_path / "shards"
    pdf_dir = tmp_path / "pdfs"
    shard_dir.mkdir()
    pdf_dir.mkdir()

    PdfBuilder(pdf_dir / "redacted.pdf").page(600, 400).rect(Rect(100, 100, 300, 140)).save()

    pq.write_table(
        pa.table({"source_hash": ["h1"], "filename": ["redacted.pdf"]}),
        shard_dir / "batch-0001._manifest.parquet",
    )
    pq.write_table(
        pa.table({
            "source_hash": ["h1", "h1"],
            "elem_order": pa.array([0, 1], type=pa.int32()),
            "page": pa.array([0, 0], type=pa.int32()),
        }),
        shard_dir / "batch-0001.elements.parquet",
    )
    return shard_dir, pdf_dir


def _redact_shards_args(shard_dir: Path, pdf_dir: Path | None, **overrides) -> argparse.Namespace:
    base = {
        "shards": shard_dir, "config": None, "pdfs": pdf_dir, "output": None,
        "checkpoint": None, "dpi": 150, "max_area_ratio": 0.05, "limit": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestCmdRedactShards:
    def test_writes_redactions_sidecar(self, tmp_path: Path) -> None:
        import pyarrow.parquet as pq

        shard_dir, pdf_dir = _seed_redaction_shards(tmp_path)
        assert cmd_redact(_redact_shards_args(shard_dir, pdf_dir)) == 0

        sidecar = shard_dir / "batch-0001.redactions.parquet"
        assert sidecar.exists()
        tbl = pq.read_table(sidecar)
        assert tbl.schema.names == ["source_hash", "elem_order", "has_redaction"]
        assert set(tbl.column("elem_order").to_pylist()) == {0, 1}

    def test_requires_pdfs_with_shards(self, tmp_path: Path) -> None:
        shard_dir, _ = _seed_redaction_shards(tmp_path)
        assert cmd_redact(_redact_shards_args(shard_dir, None)) == 1

    def test_rejects_shard_dir_without_manifests(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        pdfs = tmp_path / "pdfs"
        empty.mkdir()
        pdfs.mkdir()
        assert cmd_redact(_redact_shards_args(empty, pdfs)) == 1

    def test_rejects_nonexistent_shard_dir(self, tmp_path: Path) -> None:
        pdfs = tmp_path / "pdfs"
        pdfs.mkdir()
        assert cmd_redact(_redact_shards_args(tmp_path / "nope", pdfs)) == 1

    def test_honours_output_dir(self, tmp_path: Path) -> None:
        shard_dir, pdf_dir = _seed_redaction_shards(tmp_path)
        out = tmp_path / "sidecars"
        assert cmd_redact(_redact_shards_args(shard_dir, pdf_dir, output=out)) == 0
        assert (out / "batch-0001.redactions.parquet").exists()
        assert not (shard_dir / "batch-0001.redactions.parquet").exists()

    def test_dispatches_to_config_branch_when_shards_none(self, tmp_path: Path) -> None:
        # When --shards is absent, cmd_redact routes to the E2E --config branch
        # (load_config → FileNotFoundError for a missing path). The argparse-level
        # "exactly one of --shards/--config required" enforcement is verified
        # separately via the CLI smoke test, not reachable from a hand-built Namespace.
        args = _redact_shards_args(tmp_path / "x", tmp_path / "y")
        args.shards = None
        args.config = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError):
            cmd_redact(args)


class TestAnnotateRedactionsAlias:
    def test_alias_produces_same_sidecar_as_redact_shards(self, tmp_path: Path) -> None:
        import pyarrow.parquet as pq

        shard_dir, pdf_dir = _seed_redaction_shards(tmp_path)
        alias_args = argparse.Namespace(
            shards=shard_dir, pdfs=pdf_dir, output=None,
            checkpoint=None, dpi=150, max_area_ratio=0.05,
        )
        assert cmd_annotate_redactions(alias_args) == 0

        sidecar = shard_dir / "batch-0001.redactions.parquet"
        assert sidecar.exists()
        tbl = pq.read_table(sidecar)
        assert set(tbl.column("elem_order").to_pylist()) == {0, 1}

    def test_alias_rejects_missing_dirs(self, tmp_path: Path) -> None:
        args = argparse.Namespace(
            shards=tmp_path / "nope", pdfs=tmp_path / "nope2", output=None,
            checkpoint=None, dpi=150, max_area_ratio=0.05,
        )
        assert cmd_annotate_redactions(args) == 1


class TestLayoutFilterOnRasterPages:
    """The filter reads the layout step's regions; it never runs an analyser."""

    @staticmethod
    def _scan(tmp_path: Path) -> Path:
        px = np.full((400, 600, 3), 240, dtype=np.uint8)
        px[80:120, 100:500] = 0  # one dark bar, no vector redaction anywhere
        return (
            PdfBuilder(tmp_path / "scan.pdf").page(600, 400)
            .image(Rect(0, 0, 600, 400), Image.fromarray(px)).save()
        )

    @staticmethod
    def _layout(status: str, block_type: str = "figure"):
        from womblex.ingest.elements import BBox
        from womblex.ingest.layout_step import LayoutRegion, PageLayout

        regions = [LayoutRegion(BBox(0.1, 0.1, 0.8, 0.3), "x", block_type, 0.9)] if status == "ok" else []
        return {0: PageLayout(0, status, regions, error="boom" if status == "error" else "")}

    def _detect(self, tmp_path: Path, layout):
        return detect_redactions(
            self._scan(tmp_path), 1, build_detector(RedactionConfig()), dpi=72, layout=layout,
        )

    def test_a_figure_region_suppresses_the_hit(self, tmp_path: Path) -> None:
        assert self._detect(tmp_path, self._layout("ok")).total == 0

    def test_a_paragraph_region_does_not(self, tmp_path: Path) -> None:
        assert self._detect(tmp_path, self._layout("ok", "paragraph")).total == 1

    @pytest.mark.parametrize("case", ["none", "no-row", "error"])
    def test_no_layout_runs_unfiltered_and_says_so(self, tmp_path: Path, case, caplog) -> None:
        layout = {"none": None, "no-row": {}, "error": self._layout("error")}[case]
        with caplog.at_level("WARNING", logger="womblex.redact.stage"):
            report = self._detect(tmp_path, layout)
        assert (report.total, report.unfiltered_pages) == (1, [0])
        assert caplog.text.count("doc=scan.pdf pages=[0]") == 1  # one warning per document

    def test_a_usable_page_is_not_unfiltered(self, tmp_path: Path) -> None:
        assert self._detect(tmp_path, self._layout("empty")).unfiltered_pages == []

    def test_shards_read_the_batch_layout_sidecar(self, tmp_path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        from womblex.ingest.elements import BBox
        from womblex.ingest.layout_step import LayoutRegion, PageLayout
        from womblex.store.layout_output import (
            LayoutFingerprint,
            layout_rows,
            unfiltered_redaction_pages,
            write_layout_regions,
        )

        shards, pdfs = tmp_path / "shards", tmp_path / "pdfs"
        shards.mkdir()
        pdfs.mkdir()
        self._scan(tmp_path).rename(pdfs / "scan.pdf")
        pq.write_table(pa.table({"source_hash": ["h1"], "filename": ["scan.pdf"]}),
                       shards / "batch-0001._manifest.parquet")
        pq.write_table(pa.table({
            "source_hash": ["h1"], "elem_order": pa.array([0], type=pa.int32()),
            "page": pa.array([0], type=pa.int32()),
        }), shards / "batch-0001.elements.parquet")
        config = RedactionConfig(dpi=72)

        assert annotate_redactions_for_shards(shards, pdfs, config) == {"h1": 1}
        assert unfiltered_redaction_pages(shards) == [("h1", 0)]  # no sidecar, no layout

        fp = LayoutFingerprint("m", "d", "o", 72, "consumers")
        page = PageLayout(0, "ok", [LayoutRegion(BBox(0.1, 0.1, 0.8, 0.3), "x", "figure", 0.9)])
        write_layout_regions(
            layout_rows([("h1", [page])]), shards / "batch-0001.parquet", fp, redaction_consumed=True,
        )
        (shards / "batch-0001.redactions.parquet").unlink()
        assert annotate_redactions_for_shards(shards, pdfs, config) == {"h1": 0}
        assert unfiltered_redaction_pages(shards) == []
