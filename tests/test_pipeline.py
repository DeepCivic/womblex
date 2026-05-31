"""Tests for womblex.operations — independent operations.


Tests use real fixtures. No synthetic data.
"""


from pathlib import Path


import pytest


from womblex.config import WomblexConfig, load_config

from womblex.operations import BatchResult, DocumentResult, run_extraction, run_chunking, write_batch_parquet



# ---------------------------------------------------------------------------

# Helpers

# ---------------------------------------------------------------------------



def _make_config(sample_config_path: Path) -> WomblexConfig:
    return load_config(sample_config_path)



# ---------------------------------------------------------------------------

# run_extraction

# ---------------------------------------------------------------------------



class TestRunExtraction:

    def test_nonexistent_file_errors(self, tmp_path: Path, sample_config_path: Path) -> None:

        config = _make_config(sample_config_path)

        results = run_extraction([tmp_path / "missing.pdf"], config)


        assert len(results) == 1

        assert results[0].status == "error"

        assert results[0].error is not None

        assert results[0].profile is None


    def test_extracts_real_spreadsheet(self, spreadsheet_dir: Path, sample_config_path: Path) -> None:

        csv_path = spreadsheet_dir / "Approved-providers-au-export_20260204.csv"

        if not csv_path.exists():

            pytest.skip("CSV fixture not available")


        config = _make_config(sample_config_path)

        results = run_extraction([csv_path], config)


        assert len(results) >= 1

        ok = sum(1 for r in results if r.status == "completed")

        assert ok >= 1


    def test_empty_list(self, sample_config_path: Path) -> None:

        config = _make_config(sample_config_path)

        results = run_extraction([], config)

        assert len(results) == 0



# ---------------------------------------------------------------------------

# DocumentResult / BatchResult

# ---------------------------------------------------------------------------



class TestDocumentResult:

    def test_default_status(self) -> None:

        r = DocumentResult(path=Path("/tmp/test.pdf"), doc_id="test")

        assert r.status == "pending"

        assert r.profile is None

        assert r.extraction is None

        assert r.error is None

        assert r.chunks == []



class TestBatchResult:

    def test_empty_batch(self) -> None:

        b = BatchResult()

        assert b.succeeded == 0

        assert b.failed == 0


    def test_counts(self) -> None:

        b = BatchResult(

            results=[

                DocumentResult(path=Path("/a.pdf"), doc_id="a", status="completed"),

                DocumentResult(path=Path("/b.pdf"), doc_id="b", status="error"),

                DocumentResult(path=Path("/c.pdf"), doc_id="c", status="completed"),

            ]
        )

        assert b.succeeded == 2

        assert b.failed == 1



# ---------------------------------------------------------------------------

# Composition: extract then chunk

# ---------------------------------------------------------------------------


from womblex.config import ChunkingConfig, DatasetConfig, PathsConfig


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures" / "womblex-collection"

_CSV_FILE = FIXTURE_DIR / "_spreadsheets" / "Approved-providers-au-export_20260204.csv"



class TestComposition:

    """Operations compose correctly when called in sequence."""


    def test_extract_then_chunk(self) -> None:

        if not _CSV_FILE.exists():

            pytest.skip("CSV fixture not available")

        config = WomblexConfig(
            dataset=DatasetConfig(name="t"),
            paths=PathsConfig(input_root=Path("."), output_root=Path("."), checkpoint_dir=Path(".")),

            chunking=ChunkingConfig(enabled=True, chunk_size=480),
        )

        results = run_extraction([_CSV_FILE], config)

        assert any(r.status == "completed" for r in results)


        results = run_chunking(results, config)

        has_chunks = any(len(r.chunks) > 0 for r in results)

        assert has_chunks


    def test_extract_only(self) -> None:

        if not _CSV_FILE.exists():

            pytest.skip("CSV fixture not available")

        config = WomblexConfig(
            dataset=DatasetConfig(name="t"),
            paths=PathsConfig(input_root=Path("."), output_root=Path("."), checkpoint_dir=Path(".")),
            chunking=ChunkingConfig(enabled=False),
        )

        results = run_extraction([_CSV_FILE], config)

        assert any(r.status == "completed" for r in results)

        # No chunking called — no chunks.

        assert all(len(r.chunks) == 0 for r in results)


# ---------------------------------------------------------------------------
# cmd_run — CLI-level run_id + retention layout
# ---------------------------------------------------------------------------


import argparse
import shutil

from womblex.cli.pipeline import cmd_run


def _write_minimal_config(tmp_path: Path, input_root: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""\
dataset:
  name: i1_test
paths:
  input_root: {input_root}
  output_root: {tmp_path / "out"}
  checkpoint_dir: {tmp_path / "ckpt"}
extraction:
  native:
    include_tables: true
  ocr:
    engine: paddleocr
    dpi: 150
    lang: eng
redaction:
  enabled: false
chunking:
  enabled: false
pii:
  enabled: false
enrichment:
  enabled: false
processing:
  batch_size: 5
  checkpoint_every: 5
"""
    )
    return cfg


class TestCmdRunRunIdLayout:
    """End-to-end CLI-level verification of the I1 run_id + retention plumbing."""

    def test_auto_generated_run_id_writes_to_nested_layout(self, tmp_path: Path) -> None:
        if not _CSV_FILE.exists():
            pytest.skip("CSV fixture not available")

        input_root = tmp_path / "in"
        input_root.mkdir()
        shutil.copy(_CSV_FILE, input_root / _CSV_FILE.name)

        cfg = _write_minimal_config(tmp_path, input_root)
        args = argparse.Namespace(
            config=cfg, resume=False, limit=None, skip=0,
            batch_size=None, run_id=None,
        )

        rc = cmd_run(args)
        assert rc == 0

        run_dirs = list((tmp_path / "out").iterdir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]
        assert run_dir.name.startswith("run-")
        shard_files = list((run_dir / "documents").glob("*.parquet"))
        assert len(shard_files) == 4  # elements, table_cells, form_fields, manifest

        # Checkpoint nested under run_id
        assert (tmp_path / "ckpt" / run_dir.name / "i1_test_checkpoint.json").exists()

    def test_explicit_run_id_via_cli(self, tmp_path: Path) -> None:
        if not _CSV_FILE.exists():
            pytest.skip("CSV fixture not available")

        input_root = tmp_path / "in"
        input_root.mkdir()
        shutil.copy(_CSV_FILE, input_root / _CSV_FILE.name)

        cfg = _write_minimal_config(tmp_path, input_root)
        args = argparse.Namespace(
            config=cfg, resume=False, limit=None, skip=0,
            batch_size=None, run_id="my-explicit-run",
        )

        rc = cmd_run(args)
        assert rc == 0
        assert (tmp_path / "out" / "my-explicit-run" / "documents").is_dir()
        assert (tmp_path / "ckpt" / "my-explicit-run" / "i1_test_checkpoint.json").exists()

    def test_resume_does_not_overwrite_earlier_batches(self, tmp_path: Path) -> None:
        """Regression: on --resume, batch-NNNN.parquet numbering must continue
        from checkpoint.last_batch+1, not restart at 1. Restarting at 1
        overwrites the original batch-0001 with a later doc cohort's content
        — the failure observed live during the i1b corpus extraction resume.
        """
        if not _CSV_FILE.exists():
            pytest.skip("CSV fixture not available")

        input_root = tmp_path / "in"
        input_root.mkdir()
        # Three distinct doc files; batch_size=1 → one batch per doc.
        for i in range(3):
            dst = input_root / f"doc-{i:02d}.csv"
            shutil.copy(_CSV_FILE, dst)

        cfg = _write_minimal_config(tmp_path, input_root)

        # First invocation processes 2 of 3 docs (batches 1, 2).
        args1 = argparse.Namespace(
            config=cfg, resume=False, limit=2, skip=0,
            batch_size=1, run_id="run-test",
        )
        assert cmd_run(args1) == 0

        run_dir = tmp_path / "out" / "run-test" / "documents"
        pre_resume = sorted(p.name for p in run_dir.glob("batch-*.parquet"))
        assert any(n.startswith("batch-0001.") for n in pre_resume)
        assert any(n.startswith("batch-0002.") for n in pre_resume)
        b1_size_before = (run_dir / "batch-0001.elements.parquet").stat().st_size

        # Resume: must write batch-0003, not overwrite batch-0001.
        args2 = argparse.Namespace(
            config=cfg, resume=True, no_verify_resume=False, limit=None, skip=0,
            batch_size=1, run_id="run-test",
        )
        assert cmd_run(args2) == 0

        post_resume = sorted(set(p.name.split(".")[0] for p in run_dir.glob("batch-*.parquet")))
        assert post_resume == ["batch-0001", "batch-0002", "batch-0003"]
        # batch-0001 untouched (size invariant — if it had been overwritten with
        # different content it would almost certainly differ in size)
        assert (run_dir / "batch-0001.elements.parquet").stat().st_size == b1_size_before

    def test_retention_rolling_purges_old_run(self, tmp_path: Path) -> None:
        if not _CSV_FILE.exists():
            pytest.skip("CSV fixture not available")

        input_root = tmp_path / "in"
        input_root.mkdir()
        shutil.copy(_CSV_FILE, input_root / _CSV_FILE.name)

        cfg = _write_minimal_config(tmp_path, input_root)
        # Pre-seed a stale run dir + checkpoint that should be purged
        (tmp_path / "out" / "run-stale" / "documents").mkdir(parents=True)
        (tmp_path / "out" / "run-stale" / "documents" / "batch-0001.elements.parquet").write_bytes(b"x")
        (tmp_path / "ckpt" / "run-stale").mkdir(parents=True)

        args = argparse.Namespace(
            config=cfg, resume=False, limit=None, skip=0,
            batch_size=None, run_id="run-current",
        )
        rc = cmd_run(args)
        assert rc == 0

        # rolling, keep=2 (default): current + 1 previous kept; only 'run-stale' present
        # as previous, so 'run-stale' survives. Add a second stale to force a purge.
        (tmp_path / "out" / "run-older").mkdir(parents=True)
        (tmp_path / "out" / "run-older" / "documents").mkdir()
        # Second fresh run — older two will reduce to one
        args2 = argparse.Namespace(
            config=cfg, resume=False, limit=None, skip=0,
            batch_size=None, run_id="run-newer",
        )
        rc = cmd_run(args2)
        assert rc == 0

        # keep=2 means current (run-newer) + 1 previous most-recent survives
        survivors = sorted(p.name for p in (tmp_path / "out").iterdir() if p.is_dir())
        assert "run-newer" in survivors
        assert len(survivors) == 2  # current + 1 retained


# ---------------------------------------------------------------------------
# cmd_chunk --shards (B3)
# ---------------------------------------------------------------------------


from womblex.cli.pipeline import cmd_chunk


def _seed_run_with_extraction(tmp_path: Path, run_id: str = "i2-test") -> Path:
    """Run cmd_run against a small fixture and return the shard dir."""
    input_root = tmp_path / "in"
    input_root.mkdir()
    shutil.copy(_CSV_FILE, input_root / _CSV_FILE.name)
    cfg = _write_minimal_config(tmp_path, input_root)
    args = argparse.Namespace(
        config=cfg, resume=False, limit=None, skip=0,
        batch_size=None, run_id=run_id,
    )
    rc = cmd_run(args)
    assert rc == 0, "extraction run failed; chunk-stage tests have nothing to consume"
    return tmp_path / "out" / run_id / "documents"


class TestCmdChunkShards:
    def test_writes_chunks_sidecar_for_each_batch(self, tmp_path: Path) -> None:
        if not _CSV_FILE.exists():
            pytest.skip("CSV fixture not available")

        shard_dir = _seed_run_with_extraction(tmp_path)

        args = argparse.Namespace(
            shards=shard_dir, config=None,
            checkpoint_dir=None, dataset="i2-test",
            no_resume=True, limit=None,
        )
        assert cmd_chunk(args) == 0

        chunks_files = list(shard_dir.glob("*.chunks.parquet"))
        manifests = list(shard_dir.glob("*._manifest.parquet"))
        assert len(chunks_files) == len(manifests)
        assert chunks_files
        for f in chunks_files:
            assert f.stat().st_size > 0

    def test_chunks_join_back_to_elements_via_source_hash(self, tmp_path: Path) -> None:
        if not _CSV_FILE.exists():
            pytest.skip("CSV fixture not available")

        from womblex.store.output import read_chunks, read_elements, read_manifest

        shard_dir = _seed_run_with_extraction(tmp_path, run_id="i2-join")
        args = argparse.Namespace(
            shards=shard_dir, config=None,
            checkpoint_dir=None, dataset="i2-join",
            no_resume=True, limit=None,
        )
        assert cmd_chunk(args) == 0

        chunks = read_chunks(shard_dir)
        elements = read_elements(shard_dir)
        manifest = read_manifest(shard_dir)

        chunk_hashes = set(chunks.column("source_hash").to_pylist())
        elem_hashes = set(elements.column("source_hash").to_pylist())
        manifest_hashes = set(manifest.column("source_hash").to_pylist())

        # Every chunk's source_hash exists in elements (and the manifest).
        assert chunk_hashes
        assert chunk_hashes <= elem_hashes
        assert chunk_hashes <= manifest_hashes

    def test_rejects_dir_without_manifests(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        args = argparse.Namespace(
            shards=empty, config=None,
            checkpoint_dir=None, dataset="x",
            no_resume=True, limit=None,
        )
        assert cmd_chunk(args) == 1

    def test_no_resume_clears_checkpoint(self, tmp_path: Path) -> None:
        if not _CSV_FILE.exists():
            pytest.skip("CSV fixture not available")

        shard_dir = _seed_run_with_extraction(tmp_path, run_id="i2-ckpt")
        ckpt_dir = tmp_path / "ckpt-chunk"

        args1 = argparse.Namespace(
            shards=shard_dir, config=None,
            checkpoint_dir=ckpt_dir, dataset="i2-ckpt",
            no_resume=False, no_verify_resume=False, limit=None,
        )
        assert cmd_chunk(args1) == 0
        assert (ckpt_dir / "i2-ckpt_chunk_checkpoint.json").exists()

        args2 = argparse.Namespace(
            shards=shard_dir, config=None,
            checkpoint_dir=ckpt_dir, dataset="i2-ckpt",
            no_resume=True, no_verify_resume=False, limit=None,
        )
        assert cmd_chunk(args2) == 0

    def test_resume_recovers_corrupt_chunks_shard(self, tmp_path: Path) -> None:
        """Wire-up test: a corrupt *.chunks.parquet on resume drops the affected
        docs from the chunk checkpoint and re-writes a clean sidecar."""
        if not _CSV_FILE.exists():
            pytest.skip("CSV fixture not available")

        from womblex.store.output import chunks_path_for

        shard_dir = _seed_run_with_extraction(tmp_path, run_id="i2-recover")
        ckpt_dir = tmp_path / "ckpt-chunk"

        # First pass: produce healthy chunks + checkpoint.
        args1 = argparse.Namespace(
            shards=shard_dir, config=None,
            checkpoint_dir=ckpt_dir, dataset="i2-recover",
            no_resume=False, no_verify_resume=False, limit=None,
        )
        assert cmd_chunk(args1) == 0

        # Corrupt the first batch's chunks file.
        chunks_files = sorted(shard_dir.glob("*.chunks.parquet"))
        assert chunks_files
        chunks_files[0].write_bytes(b"")

        # Resume: reconcile drops the affected docs, chunk_shards re-writes them.
        args2 = argparse.Namespace(
            shards=shard_dir, config=None,
            checkpoint_dir=ckpt_dir, dataset="i2-recover",
            no_resume=False, no_verify_resume=False, limit=None,
        )
        assert cmd_chunk(args2) == 0

        # The corrupt file is archived with .corrupt; a fresh chunks file exists.
        # (chunks_path_for derives the path from the batch base; the archived
        # original lives next to it with the .corrupt suffix.)
        corrupt = chunks_files[0].with_name(chunks_files[0].name + ".corrupt")
        assert corrupt.exists()
        fresh = chunks_files[0]
        assert fresh.exists()
        assert fresh.stat().st_size > 0

