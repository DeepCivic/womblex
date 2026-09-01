"""The run stamp: which run, which Womblex, which configuration, which stage.

One case per acceptance criterion the mechanism can decide on its own — the
four keys on every extraction Parquet, a reader that ignores them, attribution
surviving the file being moved, the digest's basis and its formatting
invariance, local/distributed parity, and the credential non-goal.

Round-trips go through the real writer on the budget-statement DOCX fixture,
as ``test_output`` and ``test_source_provenance`` do; the digest cases build
configurations directly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from womblex import __version__
from womblex.config import (
    ChunkingConfig,
    DatasetConfig,
    PathsConfig,
    WomblexConfig,
)
from womblex.ingest.strategies_file import DocxExtractor
from womblex.store.output import _shard_paths, read_manifest, write_results
from womblex.store.run_stamp import (
    CONFIG_DIGEST_KEY,
    RUN_ID_KEY,
    STAGE_KEY,
    VERSION_KEY,
    RunStamp,
    config_digest,
    read_footer_stamp,
)

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures"
_BUDGET_DOCX = (
    _FIXTURES / "womblex-collection" / "_documents"
    / "foreign-affairs-and-trade-2025-26-portfolio-budget-statements.docx"
)

# The names a run's credentials arrive under. The stamp must contain none of
# their values — it carries a digest, a run id, a version and a stage.
CREDENTIAL_ENV = (
    "ISAACUS_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "WOMBLEX_S3_ACCESS_KEY_ID",
    "WOMBLEX_S3_SECRET_ACCESS_KEY",
    "WOMBLEX_DB_DSN",
)


def _config(tmp_path: Path, **over) -> WomblexConfig:
    """A minimal valid configuration rooted at *tmp_path*."""
    return WomblexConfig(
        dataset=DatasetConfig(name="test-corpus", **over.pop("dataset", {})),
        paths=PathsConfig(
            input_root=over.pop("input_root", tmp_path / "in"),
            output_root=tmp_path / "out",
            checkpoint_dir=tmp_path / "ckpt",
        ),
        **over,
    )


@pytest.fixture(scope="module")
def extraction():
    if not _BUDGET_DOCX.exists():
        pytest.skip(f"fixture not present: {_BUDGET_DOCX}")
    return DocxExtractor().extract_path(_BUDGET_DOCX)


@pytest.fixture
def stamped(tmp_path, extraction):
    """A shard written by run ``run-A`` at the ``extract`` stage."""
    config = _config(tmp_path)
    shard = tmp_path / "run-A" / "documents" / "batch-0001.parquet"
    write_results(
        [("budget", str(_BUDGET_DOCX), extraction)],
        shard,
        collection_id="test-corpus",
        stamp=RunStamp.declare("run-A", config, stage="extract"),
    )
    return shard


class TestDeclaration:
    def test_a_stamp_names_the_run_the_build_and_the_writer(self, tmp_path):
        stamp = RunStamp.declare("run-A", _config(tmp_path), stage="extract")
        assert (stamp.run_id, stamp.version, stamp.stage) == ("run-A", __version__, "extract")
        assert stamp.config_digest.startswith("sha256:")

    def test_an_empty_run_id_raises_rather_than_stamping_a_blank(self, tmp_path):
        with pytest.raises(ValueError, match="run_id is empty"):
            RunStamp.declare("  ", _config(tmp_path), stage="extract")

    def test_an_unnamed_stage_raises_rather_than_stamping_a_blank(self, tmp_path):
        with pytest.raises(ValueError, match="stage is empty"):
            RunStamp.declare("run-A", _config(tmp_path), stage="")

    def test_re_pointing_at_a_stage_keeps_the_rest_of_the_run(self, tmp_path):
        stamp = RunStamp.declare("run-A", _config(tmp_path), stage="extract")
        chunked = stamp.for_stage("chunk")
        assert chunked.stage == "chunk"
        assert (chunked.run_id, chunked.version, chunked.config_digest) == (
            stamp.run_id, stamp.version, stamp.config_digest,
        )


class TestConfigDigest:
    def test_the_same_configuration_digests_the_same(self, tmp_path):
        assert config_digest(_config(tmp_path)) == config_digest(_config(tmp_path))

    def test_a_behaviour_change_changes_the_digest(self, tmp_path):
        assert config_digest(_config(tmp_path)) != config_digest(
            _config(tmp_path, chunking=ChunkingConfig(chunk_size=99)),
        )

    def test_defaults_the_yaml_omitted_are_digested_as_supplied(self, tmp_path):
        """The basis is the validated configuration, so an explicitly-supplied
        default digests identically to an omitted one — which is what makes two
        YAML files differing only in formatting produce one digest."""
        assert config_digest(_config(tmp_path, chunking=ChunkingConfig())) == config_digest(
            _config(tmp_path),
        )

    def test_where_the_deployment_keeps_its_files_is_not_a_behaviour_change(self, tmp_path):
        """A worker extracts from a scratch dir; the local run reads the corpus
        in place. Same pipeline, same corpus, so the same digest."""
        assert config_digest(_config(tmp_path)) == config_digest(
            _config(tmp_path, input_root=tmp_path / "scratch" / "inputs"),
        )

    def test_the_run_id_is_stamped_in_its_own_right_not_folded_into_the_digest(self, tmp_path):
        assert config_digest(_config(tmp_path)) == config_digest(
            _config(tmp_path, dataset={"run_id": "run-B"}),
        )


class TestFooterMetadata:
    def test_every_extraction_parquet_carries_the_four_keys(self, stamped):
        for role, path in _shard_paths(stamped).items():
            keys = {k.decode() for k in pq.read_metadata(str(path)).metadata}
            assert {RUN_ID_KEY, VERSION_KEY, CONFIG_DIGEST_KEY, STAGE_KEY} <= keys, role

    def test_the_stamp_reads_back_as_the_run_that_wrote_it(self, stamped):
        meta = pq.read_metadata(str(_shard_paths(stamped)["elements"])).metadata
        stamp = read_footer_stamp(meta)
        assert stamp == {
            "run_id": "run-A",
            "version": __version__,
            "config_digest": stamp["config_digest"],
            "stage": "extract",
        }

    def test_a_reader_ignoring_the_footer_reads_the_file_unchanged(self, stamped):
        assert read_manifest(stamped).num_rows == 1
        assert pq.read_table(str(_shard_paths(stamped)["elements"])).num_rows > 0

    def test_a_file_moved_out_of_its_run_still_names_its_run(self, stamped, tmp_path):
        loose = tmp_path / "elsewhere" / "somebody-elses-name.parquet"
        loose.parent.mkdir(parents=True)
        shutil.copy(_shard_paths(stamped)["elements"], loose)
        assert read_footer_stamp(pq.read_metadata(str(loose)).metadata)["run_id"] == "run-A"

    def test_without_a_stamp_no_run_keys_are_written(self, tmp_path, extraction):
        shard = tmp_path / "out" / "batch-0001.parquet"
        write_results([("budget", str(_BUDGET_DOCX), extraction)], shard)
        meta = pq.read_metadata(str(_shard_paths(shard)["elements"])).metadata
        assert read_footer_stamp(meta) == {}

    def test_a_file_written_before_the_stamp_existed_reads_back_empty(self):
        assert read_footer_stamp(None) == {}

    def test_the_provenance_pair_survives_alongside_the_run_keys(self, tmp_path, extraction):
        """Two namespaced blocks share one footer; neither overwrites the other."""
        from womblex.store.source_provenance import IngestProvenance, read_footer_provenance

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        target = corpus / _BUDGET_DOCX.name
        shutil.copy(_BUDGET_DOCX, target)
        shard = tmp_path / "run-A" / "documents" / "batch-0001.parquet"
        write_results(
            [("budget", str(target), extraction)],
            shard,
            provenance=IngestProvenance.declare(corpus, "test-corpus"),
            stamp=RunStamp.declare("run-A", _config(tmp_path), stage="extract"),
        )
        meta = pq.read_metadata(str(_shard_paths(shard)["elements"])).metadata
        assert read_footer_stamp(meta)["run_id"] == "run-A"
        assert read_footer_provenance(meta)["ingest_root"] == f"file://{corpus}"


class TestLocalDistributedParity:
    def test_the_same_run_and_config_stamp_identically_whatever_ran_it(self, tmp_path):
        """The worker declares from the job row's run id and the same config;
        only the staging location differs, and that is not in the digest."""
        local = RunStamp.declare("run-A", _config(tmp_path), stage="extract")
        worker = RunStamp.declare(
            "run-A", _config(tmp_path, input_root=tmp_path / "scratch" / "inputs"),
            stage="extract",
        )
        assert local == worker


class TestCredentials:
    def test_no_credential_value_reaches_the_footer(self, tmp_path, monkeypatch):
        for name in CREDENTIAL_ENV:
            monkeypatch.setenv(name, f"SECRET-{name}")
        stamp = RunStamp.declare("run-A", _config(tmp_path), stage="extract")
        blob = b"".join(stamp.footer_metadata().values())
        for name in CREDENTIAL_ENV:
            assert f"SECRET-{name}".encode() not in blob
