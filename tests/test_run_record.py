"""The run record: what produced a run, observed from its own artefacts.

One case per acceptance criterion the mechanism can decide on its own — where
the record lives, that it regenerates from the shard directory alone and
reproduces apart from its timestamp, that stages are observed rather than
declared, that a model digest recomputes from the model files, that no
credential reaches the footer, and that a run predating the stamps is marked
partial rather than failing.

Shards are written through the real writer on the budget-statement DOCX
fixture, as `test_run_stamp` does.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from womblex.config import DatasetConfig, PathsConfig, WomblexConfig
from womblex.ingest.strategies_file import DocxExtractor
from womblex.store.embed_output import EMBEDDINGS_SCHEMA
from womblex.store.output import write_results
from womblex.store.run_manifest import (
    RUN_RECORD_KEY,
    RUN_RECORD_VERSION,
    _footers,
    read_run_record,
    run_manifest_path_for,
    write_run_manifest,
)
from womblex.store.run_stamp import RunStamp
from womblex.utils.models import (
    digest_model_path,
    reset_loaded_models,
    resolve_local_model_path,
)

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures"
_BUDGET_DOCX = (
    _FIXTURES / "womblex-collection" / "_documents"
    / "foreign-affairs-and-trade-2025-26-portfolio-budget-statements.docx"
)

# The names a run's credentials arrive under; none of their values may reach
# the record, which carries endpoints, regions, digests and model ids.
CREDENTIAL_ENV = (
    "ISAACUS_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "WOMBLEX_S3_ACCESS_KEY_ID",
    "WOMBLEX_S3_SECRET_ACCESS_KEY",
    "WOMBLEX_DB_DSN",
)


@pytest.fixture(autouse=True)
def _clean_record():
    reset_loaded_models()
    yield
    reset_loaded_models()


@pytest.fixture(scope="module")
def extraction():
    if not _BUDGET_DOCX.exists():
        pytest.skip(f"fixture not present: {_BUDGET_DOCX}")
    return DocxExtractor().extract_path(_BUDGET_DOCX)


def _config(tmp_path: Path) -> WomblexConfig:
    return WomblexConfig(
        dataset=DatasetConfig(name="test-corpus"),
        paths=PathsConfig(
            input_root=tmp_path / "in",
            output_root=tmp_path / "out",
            checkpoint_dir=tmp_path / "ckpt",
        ),
    )


def _extracted(tmp_path: Path, extraction, *, stamp: RunStamp | None = None) -> Path:
    """A one-batch shard directory, stamped for ``run-A`` unless told otherwise."""
    shards = tmp_path / "run-A" / "documents"
    write_results(
        [("budget", str(_BUDGET_DOCX), extraction)],
        shards / "batch-0001.parquet",
        collection_id="test-corpus",
        stamp=stamp if stamp is not None else RunStamp.declare(
            "run-A", _config(tmp_path), stage="extract",
        ),
    )
    return shards


def _sidecar(shards: Path, suffix: str, table: pa.Table, stage: str) -> Path:
    """A downstream sidecar of batch 1, stamped as written by *stage*."""
    target = shards / f"batch-0001{suffix}"
    stamp = RunStamp.inherit("run-A", "sha256:x", stage=stage)
    pq.write_table(
        table.replace_schema_metadata(stamp.footer_metadata()), str(target),
        compression="zstd",
    )
    return target


class TestWhereItLives:
    def test_the_record_is_footer_metadata_on_the_run_manifest(self, tmp_path, extraction):
        """No new output file, no new column: the manifest's own footer."""
        shards = _extracted(tmp_path, extraction)
        manifest = write_run_manifest(shards)

        assert manifest == run_manifest_path_for(shards)
        assert sorted(p.name for p in manifest.parent.iterdir()) == [
            "documents", "manifest.parquet",
        ]
        table = pq.read_table(str(manifest))
        assert RUN_RECORD_KEY not in table.schema.names
        assert read_run_record(manifest)["record_version"] == RUN_RECORD_VERSION

    def test_a_reader_ignoring_the_footer_reads_the_manifest_unchanged(
        self, tmp_path, extraction,
    ):
        shards = _extracted(tmp_path, extraction)
        manifest = write_run_manifest(shards)
        assert pq.read_table(str(manifest)).num_rows == 1


class TestRegeneration:
    def test_the_shard_directory_alone_regenerates_the_record(self, tmp_path, extraction):
        """No configuration is consulted — `womblex manifest` gets a directory."""
        shards = _extracted(tmp_path, extraction)
        first = read_run_record(write_run_manifest(shards))
        assert first["run"]["run_id"] == "run-A"
        assert first["inputs"]["documents"] == 1
        assert first["inputs"]["collection_id"] == "test-corpus"

    def test_re_running_over_an_unchanged_run_reproduces_the_record(
        self, tmp_path, extraction,
    ):
        shards = _extracted(tmp_path, extraction)
        first = read_run_record(write_run_manifest(shards))
        second = read_run_record(write_run_manifest(shards))
        assert first.pop("generated_at") is not None
        second.pop("generated_at")
        assert first == second


class TestStagesAreObserved:
    def test_a_stage_is_named_by_the_files_it_wrote(self, tmp_path, extraction):
        shards = _extracted(tmp_path, extraction)
        record = read_run_record(write_run_manifest(shards))
        stages = {s["stage"]: s for s in record["stages"]}
        assert set(stages) == {"extract"}
        assert stages["extract"]["files"] == 4
        assert stages["extract"]["rows"] > 0

    def test_a_stage_that_ran_and_produced_nothing_differs_from_one_that_never_ran(
        self, tmp_path, extraction,
    ):
        """The distinction a list taken from configuration flags cannot make:
        present with no rows, versus absent."""
        shards = _extracted(tmp_path, extraction)
        empty = pa.table(
            {f.name: pa.array([], type=f.type) for f in EMBEDDINGS_SCHEMA},
            schema=EMBEDDINGS_SCHEMA,
        )
        _sidecar(shards, ".embeddings.parquet", empty, "embed")

        stages = {s["stage"]: s for s in read_run_record(write_run_manifest(shards))["stages"]}
        assert stages["embed"] == {"stage": "embed", "files": 1, "rows": 0}
        assert "chunk" not in stages

    def test_stages_come_back_in_pipeline_order(self, tmp_path, extraction):
        shards = _extracted(tmp_path, extraction)
        empty = pa.table(
            {f.name: pa.array([], type=f.type) for f in EMBEDDINGS_SCHEMA},
            schema=EMBEDDINGS_SCHEMA,
        )
        _sidecar(shards, ".embeddings.parquet", empty, "embed")
        names = [s["stage"] for s in read_run_record(write_run_manifest(shards))["stages"]]
        assert names == ["extract", "embed"]


class TestLocalModels:
    def test_a_loaded_model_is_named_with_a_digest_that_recomputes(
        self, tmp_path, extraction,
    ):
        """The criterion: the digest checks out against the model files alone."""
        resolve_local_model_path("en_AU")
        shards = _extracted(tmp_path, extraction)
        record = read_run_record(write_run_manifest(shards))

        entry = next(m for m in record["local_models"] if m["name"] == "en_AU")
        assert entry["digest"] == digest_model_path(
            Path(str(resolve_local_model_path("en_AU"))),
        )
        assert entry["stages"] == ["extract"]

    def test_a_run_that_loaded_none_says_so_rather_than_listing_the_build(
        self, tmp_path, extraction,
    ):
        shards = _extracted(tmp_path, extraction)
        record = read_run_record(write_run_manifest(shards))
        assert record["local_models"] == []
        assert any("local models" in p for p in record["partial"])


class TestCredentials:
    def test_no_credential_value_reaches_the_record(self, tmp_path, extraction, monkeypatch):
        for name in CREDENTIAL_ENV:
            monkeypatch.setenv(name, f"SECRET-{name}")
        monkeypatch.setenv(
            "ISAACUS_SAGEMAKER_ENDPOINTS", "embed-001@ap-southeast-2=kanon-2-embedder",
        )
        shards = _extracted(tmp_path, extraction)
        manifest = write_run_manifest(shards)

        blob = pq.read_metadata(str(manifest)).metadata[RUN_RECORD_KEY.encode()]
        for name in CREDENTIAL_ENV:
            assert f"SECRET-{name}".encode() not in blob

    def test_a_deployment_is_named_by_endpoint_region_and_models(
        self, tmp_path, extraction, monkeypatch,
    ):
        monkeypatch.setenv(
            "ISAACUS_SAGEMAKER_ENDPOINTS", "embed-001@ap-southeast-2=kanon-2-embedder",
        )
        shards = _extracted(tmp_path, extraction)
        assert read_run_record(write_run_manifest(shards))["services"] == [{
            "kind": "isaacus-sagemaker",
            "endpoint": "embed-001",
            "region": "ap-southeast-2",
            "models": ["kanon-2-embedder"],
        }]


class TestPartialRecord:
    def test_a_run_predating_the_stamps_is_marked_partial_not_failed(
        self, tmp_path, extraction,
    ):
        """An unstamped run still yields documents, methods and statuses; the
        one fact it cannot supply is named rather than invented."""
        shards = _extracted(tmp_path, extraction, stamp=None)
        for path in shards.glob("*.parquet"):
            table = pq.read_table(str(path))
            pq.write_table(table.replace_schema_metadata({}), str(path))

        record = read_run_record(write_run_manifest(shards))
        assert record["run"] is None
        assert record["stages"] == []
        assert record["inputs"]["documents"] == 1
        assert any("run identity" in p for p in record["partial"])
        assert any("stages" in p for p in record["partial"])

    def test_an_empty_shard_directory_records_the_absence_rather_than_raising(
        self, tmp_path,
    ):
        shards = tmp_path / "run-A" / "documents"
        shards.mkdir(parents=True)
        record = read_run_record(write_run_manifest(shards))
        assert record["inputs"]["documents"] == 0
        assert any("no readable parquet" in p for p in record["partial"])

class TestStagedSubset:
    """The distributed shape: `womblex finalize` stages in only the manifests.

    A record built from that directory can see extraction and nothing else, and
    a missing stage is indistinguishable from one that never ran — so it either
    says so, or is handed the footers read where the shards actually live.
    """

    @pytest.fixture
    def with_embed(self, tmp_path, extraction) -> Path:
        shards = _extracted(tmp_path, extraction)
        empty = pa.table(
            {f.name: pa.array([], type=f.type) for f in EMBEDDINGS_SCHEMA},
            schema=EMBEDDINGS_SCHEMA,
        )
        _sidecar(shards, ".embeddings.parquet", empty, "embed")
        return shards

    @staticmethod
    def _manifests_only(shards: Path, tmp_path: Path) -> Path:
        staged = tmp_path / "staged" / "documents"
        staged.mkdir(parents=True)
        for path in shards.glob("*._manifest.parquet"):
            shutil.copy(path, staged / path.name)
        return staged

    def test_a_manifest_only_directory_declares_what_it_cannot_see(
        self, with_embed, tmp_path,
    ):
        staged = self._manifests_only(with_embed, tmp_path)
        record = read_run_record(write_run_manifest(staged))
        assert [s["stage"] for s in record["stages"]] == ["extract"]
        assert any("manifest shards alone" in p for p in record["partial"])

    def test_supplied_footers_restore_the_stages_the_staging_left_behind(
        self, with_embed, tmp_path,
    ):
        """What `womblex finalize` does: read the footers where the shards live
        and hand them in, rather than staging the whole run to observe it."""
        staged = self._manifests_only(with_embed, tmp_path)
        record = read_run_record(
            write_run_manifest(staged, footers=_footers(with_embed)),
        )
        assert [s["stage"] for s in record["stages"]] == ["extract", "embed"]
        assert not any("manifest shards alone" in p for p in record["partial"])

    def test_a_full_shard_directory_makes_no_such_claim(self, with_embed):
        record = read_run_record(write_run_manifest(with_embed))
        assert not any("manifest shards alone" in p for p in record["partial"])
        assert not any("does not hold the whole run" in p for p in record["partial"])

    def test_supplied_footers_still_declare_the_served_model_limit(
        self, with_embed, tmp_path,
    ):
        """Footers carry the stages and models, but served models are read from
        the embedding sidecars' rows — which supplied footers cannot stand in
        for, so the limit outlives the fix for the other two."""
        staged = self._manifests_only(with_embed, tmp_path)
        record = read_run_record(
            write_run_manifest(staged, footers=_footers(with_embed)),
        )
        assert any("does not hold the whole run" in p for p in record["partial"])


class TestEndpointIdentity:
    """No account identifier reaches the record, whatever shape the ARN takes."""

    ARNS = (
        "arn:aws:sagemaker:ap-southeast-2:123456789012:endpoint/embed-001",
        "arn:aws:sagemaker:ap-southeast-2:123456789012",
        "arn:aws:sagemaker:ap-southeast-2:123456789012:",
        "arn:aws:sagemaker:ap-southeast-2:123456789012:endpoint/",
        "arn:aws:sagemaker:ap-southeast-2:123456789012:endpoint/a/embed-001",
    )

    @pytest.mark.parametrize("arn", ARNS)
    def test_no_arn_shape_puts_the_account_id_in_the_record(
        self, arn, tmp_path, extraction, monkeypatch,
    ):
        monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", f"{arn}=kanon-2-embedder")
        shards = _extracted(tmp_path, extraction)
        blob = pq.read_metadata(
            str(write_run_manifest(shards)),
        ).metadata[RUN_RECORD_KEY.encode()]
        assert b"123456789012" not in blob

    def test_an_arn_is_reduced_to_its_endpoint_name_and_region(
        self, tmp_path, extraction, monkeypatch,
    ):
        monkeypatch.setenv(
            "ISAACUS_SAGEMAKER_ENDPOINTS",
            "arn:aws:sagemaker:ap-southeast-2:123456789012:endpoint/embed-001"
            "=kanon-2-embedder",
        )
        shards = _extracted(tmp_path, extraction)
        assert read_run_record(write_run_manifest(shards))["services"] == [{
            "kind": "isaacus-sagemaker",
            "endpoint": "embed-001",
            "region": "ap-southeast-2",
            "models": ["kanon-2-embedder"],
        }]

    def test_a_plain_endpoint_name_is_untouched(self, tmp_path, extraction, monkeypatch):
        monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "embed-001@ap-southeast-2")
        services = read_run_record(write_run_manifest(_extracted(tmp_path, extraction)))
        assert services["services"][0]["endpoint"] == "embed-001"
