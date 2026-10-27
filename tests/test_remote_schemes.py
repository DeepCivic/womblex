"""A store is whatever fsspec can open: usable schemes, and a run over `memory://` with no Womblex edit."""

from __future__ import annotations

import json
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
import pytest

from tests.test_run_record import _extracted, extraction  # noqa: F401
from tests.test_stage_runner import _config as _stage_config
from womblex.cloud.deployment import DeploymentSettings
from womblex.cloud.finalize import finalize_run
from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.cloud.stage_runner import run_stage_remote
from womblex.store.egress import build_bundle
from womblex.store.remote import (
    RemoteStore,
    assert_disjoint_locations,
    is_remote_uri,
    scheme_problem,
    usable_schemes,
    validate_location_uri,
)


def test_a_scheme_fsspec_knows_with_its_backend_installed_is_usable() -> None:
    for scheme in ("s3", "memory", "file"):
        assert scheme_problem(scheme) is None
    validate_location_uri("memory://inbox")
    validate_location_uri("s3://bucket/inbox")
    validate_location_uri("/data/inbox")


def test_a_scheme_whose_backend_is_missing_is_refused_naming_the_package(monkeypatch) -> None:
    real = fsspec.get_filesystem_class

    def get(scheme: str):
        if scheme == "gs":
            raise ImportError("Install gcsfs to access Google Storage")
        return real(scheme)

    monkeypatch.setattr(fsspec, "get_filesystem_class", get)
    with pytest.raises(ValueError, match="needs the package 'gcsfs', which is not installed"):
        validate_location_uri("gs://bucket/inbox")
    validate_location_uri("gs://bucket/inbox", require_backend=False)  # a spec is checked where the workers' backend need not be


def test_an_unknown_scheme_is_refused_naming_the_usable_ones() -> None:
    with pytest.raises(ValueError, match="not one fsspec knows; usable here: .*memory.*s3"):
        validate_location_uri("nosuchscheme://host/inbox")
    with pytest.raises(ValueError, match="not one fsspec knows"):
        validate_location_uri("nosuchscheme://host/inbox", require_backend=False)  # unknown to fsspec is never usable
    assert {"s3", "memory", "file"} <= set(usable_schemes())


def test_the_typo_checks_stay() -> None:
    with pytest.raises(ValueError, match="needs '://'"):
        validate_location_uri("s3:/bucket/inbox")
    with pytest.raises(ValueError, match="lowercase"):
        validate_location_uri("S3://bucket/inbox")
    with pytest.raises(ValueError, match="no bucket"):
        validate_location_uri("s3://")
    validate_location_uri("memory://")  # the root of memory is a place, as file:// is


@pytest.mark.parametrize(("uri", "remote"), [
    ("s3://b/x", True), ("gs://b/x", True), ("az://c/x", True), ("memory://x", True),
    ("file:///data/x", False), ("/data/x", False), ("relative/x", False),
])
def test_a_uri_is_remote_when_it_names_a_scheme_other_than_the_local_filesystem(uri: str, remote: bool) -> None:
    assert is_remote_uri(uri) is remote


def test_the_disjointness_check_covers_every_scheme() -> None:
    assert_disjoint_locations("memory://inbox", "memory://out")
    with pytest.raises(ValueError, match="not disjoint"):
        assert_disjoint_locations("memory://out/runs/in", "memory://out")
    with pytest.raises(ValueError, match="not disjoint"):
        DeploymentSettings(output_root=None, store_uri="memory://out", ingest_uri="memory://out/runs/in")


def test_a_destination_on_a_scheme_nothing_here_opens_is_refused_before_any_step() -> None:
    from womblex.cloud.delivery import check_destinations
    from womblex.cloud.workflow_store import ObjectStoreDestination

    dest = ObjectStoreDestination.model_construct(name="archive", kind="object_store", uri="nosuchscheme://x/y",
                                                  credentials=None)
    ((name, problem),) = check_destinations((dest,))  # type: ignore[arg-type]
    assert name == "archive" and "cannot write to nosuchscheme:// here: scheme 'nosuchscheme' is not one fsspec knows" in problem


def test_extract_chunk_finalize_and_egress_over_memory_with_no_womblex_edit(tmp_path: Path, extraction) -> None:  # noqa: F811
    store = RemoteStore.from_uri("memory://womblex-c1")
    local = _extracted(tmp_path, extraction)
    for path in sorted(local.glob("batch-0001*.parquet")):
        store.upload_file(path, f"run-A/documents/{path.name}")

    summary = run_stage_remote(STAGE_CONTRACTS["chunk"], store, "run-A/documents", _stage_config())
    assert (summary.failed, summary.processed) == (0, 1) and summary.error is None
    assert store.exists("run-A/documents/batch-0001.chunks.parquet")

    finalized = finalize_run(store, "run-A")
    assert finalized is not None and store.exists("run-A/manifest.parquet")

    destination = RemoteStore.from_uri("memory://womblex-c1-bundle")
    result = build_bundle("memory://womblex-c1/run-A", destination, run_id="run-A", include_sources=False)
    assert destination.exists("run-A/egress_manifest.json")
    descriptor = json.loads(destination.read_text("run-A/egress_manifest.json"))
    assert descriptor["run_id"] == "run-A" and (result.documents, result.sources_copied) == (1, 0)
    assert destination.exists("run-A/corpus/batch-0001.chunks.parquet")
    with destination.fs.open(destination._full("run-A/corpus/manifest.parquet"), "rb") as handle:
        assert pq.read_table(handle).num_rows >= 1
