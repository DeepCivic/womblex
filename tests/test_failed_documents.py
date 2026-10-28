"""A document that fails is a manifest row, not a gap (SMS uplift D1).

An unreadable source gets `extraction_method: unreadable` and a placeholder `source_hash` over its
relpath and batch number; a document that failed after its bytes were read keeps its file-bytes hash.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from tests._synthetic import REGISTER_CSV
from womblex.batch import process_batch
from womblex.config import ChunkingConfig, DatasetConfig, PathsConfig, WomblexConfig
from womblex.process.chunk_stage import chunk_shards
from womblex.store.manifest_rows import UNREADABLE_METHOD, unreadable_hash
from womblex.store.output import read_manifest, verify_shard_persistence
from womblex.store.run_manifest import write_run_manifest
from womblex.store.source_provenance import IngestProvenance
from womblex.store.source_resolver import (
    HASH_BASIS_FILE_BYTES,
    HASH_BASIS_UNREADABLE,
    UNSUPPORTED_BASIS,
    SourceResolver,
    hash_basis_for,
)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """An ingest root holding a good register, a corrupt PDF and a path that cannot be read."""
    root = tmp_path / "inbox"
    root.mkdir()
    shutil.copy(REGISTER_CSV, root / "register.csv")
    (root / "corrupt.pdf").write_bytes(b"this is not a pdf")
    return root


def _config(tmp_path: Path) -> WomblexConfig:
    return WomblexConfig(
        dataset=DatasetConfig(name="t"),
        paths=PathsConfig(input_root=tmp_path, output_root=tmp_path, checkpoint_dir=tmp_path),
    )


def _batch(corpus: Path, shards: Path, batch_num: int, tmp_path: Path):
    files = [corpus / "register.csv", corpus / "corrupt.pdf", corpus / "vanished.pdf"]
    return process_batch(
        files, _config(tmp_path), batch_num=batch_num, shard_dir=shards,
        provenance=IngestProvenance.declare(corpus, "t"),
    )


def _rows(shards: Path, batch_num: int = 1) -> dict[str, dict]:
    path = shards / f"batch-{batch_num:04d}.parquet"
    return {r["filename"]: r for r in read_manifest(path).to_pylist()}


def test_every_document_given_to_a_batch_has_a_manifest_row(corpus: Path, tmp_path: Path):
    shards = tmp_path / "documents"
    outcome = _batch(corpus, shards, 1, tmp_path)

    rows = _rows(shards)
    assert sorted(rows) == ["corrupt.pdf", "register.csv", "vanished.pdf"]
    assert outcome.rows_written == 3
    verify_shard_persistence(outcome.shard_path, 3, 0)  # the shard's manifest count matches what was given

    vanished = rows["vanished.pdf"]
    assert (vanished["status"], vanished["extraction_method"]) == ("error", UNREADABLE_METHOD)
    assert vanished["source_hash"] == hashlib.sha256(b"unreadable:vanished.pdf:1").hexdigest()
    assert vanished["error"]


def test_a_failure_after_the_bytes_were_read_keeps_its_file_hash_and_is_not_marked_unreadable(
    corpus: Path, tmp_path: Path,
):
    shards = tmp_path / "documents"
    _batch(corpus, shards, 1, tmp_path)

    corrupt = _rows(shards)["corrupt.pdf"]
    assert corrupt["status"] == "error"
    assert corrupt["extraction_method"] != UNREADABLE_METHOD
    assert corrupt["source_hash"] == hashlib.sha256(b"this is not a pdf").hexdigest()


def test_an_unreadable_file_in_two_firings_has_two_hashes_in_one_chain(corpus: Path, tmp_path: Path):
    shards = tmp_path / "documents"
    _batch(corpus, shards, 1, tmp_path)
    _batch(corpus, shards, 2, tmp_path)

    manifest = pq.read_table(str(write_run_manifest(shards))).to_pylist()
    chain = [r for r in manifest if r["source_relpath"] == "vanished.pdf"]
    assert [r["source_hash"] for r in chain] == [unreadable_hash("vanished.pdf", 1), unreadable_hash("vanished.pdf", 2)]
    assert len({r["source_hash"] for r in chain}) == 2
    assert [r["is_current"] for r in chain] == [False, True]
    assert chain[1]["replaces"] == chain[0]["source_hash"]


def test_the_resolver_declines_an_unreadable_row_and_names_its_relpath(corpus: Path, tmp_path: Path):
    shards = tmp_path / "documents"
    _batch(corpus, shards, 1, tmp_path)
    manifest = read_manifest(shards / "batch-0001.parquet")

    row = _rows(shards)["vanished.pdf"]
    resolution = SourceResolver(manifest, corpus).resolve(row["source_hash"])

    assert resolution.status == UNSUPPORTED_BASIS and resolution.path is None
    assert resolution.hash_basis == HASH_BASIS_UNREADABLE
    assert "vanished.pdf" in resolution.detail
    assert hash_basis_for(UNREADABLE_METHOD) == HASH_BASIS_UNREADABLE != HASH_BASIS_FILE_BYTES
    # The corrupt PDF's bytes were read, so it resolves as any file-bytes row does.
    ok = SourceResolver(manifest, corpus).resolve(_rows(shards)["corrupt.pdf"]["source_hash"])
    assert ok.ok and ok.hash_basis == HASH_BASIS_FILE_BYTES


def test_a_stage_over_a_shard_with_failed_rows_still_chunks_the_good_document(corpus: Path, tmp_path: Path):
    shards = tmp_path / "documents"
    _batch(corpus, shards, 1, tmp_path)

    result = chunk_shards(shards, ChunkingConfig(chunk_size=64), text_source="elements")

    assert result.docs_chunked == 1
    status = read_manifest(shards / "batch-0001.parquet").column("status")
    assert pc.sum(pc.equal(status, "error")).as_py() == 2


def test_egress_reports_an_unreadable_row_unsupported_with_nothing_under_sources(corpus: Path, tmp_path: Path):
    from womblex.store.egress_sources import StoreSources

    shards = tmp_path / "documents"
    _batch(corpus, shards, 1, tmp_path)
    manifest = read_manifest(shards / "batch-0001.parquet")

    outcome = StoreSources(manifest, None).copy(_rows(shards)["vanished.pdf"]["source_hash"], object(), "prefix")  # type: ignore[arg-type]

    assert (outcome.status, outcome.raw_key) == (UNSUPPORTED_BASIS, None)


@pytest.mark.parametrize("stage", ["normalise", "spellfix", "money", "enrich", "pii"])
def test_no_downstream_stage_trips_over_a_shard_holding_failed_rows(corpus: Path, tmp_path: Path, stage: str):
    from womblex.analyse.enrich_stage import enrich_shards
    from womblex.config import EnrichmentConfig, MoneyConfig, NormaliseConfig, PIIConfig, SpellfixConfig
    from womblex.pii.pii_stage import pii_shards
    from womblex.process.money_stage import money_shards
    from womblex.process.normalise_stage import normalise_shards
    from womblex.process.spellfix_stage import spellfix_shards

    shards = tmp_path / "documents"
    _batch(corpus, shards, 1, tmp_path)
    runs = {
        "normalise": lambda: normalise_shards(shards, NormaliseConfig()),
        "spellfix": lambda: spellfix_shards(shards, SpellfixConfig()),
        "money": lambda: money_shards(shards, MoneyConfig()),
        # Every narrative is under the minimum, so no request is made: the stage must still land.
        "enrich": lambda: enrich_shards(
            shards, EnrichmentConfig(skip_short_documents=10**9, include_tables=False), client=object()),
        "pii": lambda: (chunk_shards(shards, ChunkingConfig(chunk_size=64), text_source="elements"),
                        pii_shards(shards, PIIConfig(use_regex_backstop=False)))[1],  # pii reads chunks
    }

    assert runs[stage]().batches_written == 1
