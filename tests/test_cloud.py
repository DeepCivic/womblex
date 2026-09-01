"""Tests for the cloud distributed-execution pieces.

RemoteStore is exercised against the local filesystem (fsspec's local backend),
so it needs no S3/MinIO. The JobQueue tests need a real Postgres and skip
cleanly when ``WOMBLEX_DB_DSN`` / ``DATABASE_URL`` is unset.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from womblex.store.remote import (
    RemoteStore,
    assert_disjoint_locations,
    is_remote_uri,
    same_location,
    storage_options_from_env,
    store_root,
    validate_location_uri,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# RemoteStore reaches fsspec lazily; skip the whole module without the cloud extra.
pytest.importorskip("fsspec")


# --- schema artefact ---------------------------------------------------------


def test_sql_schema_file_matches_the_queue_ddl():
    """``sql/womblex_jobs.sql`` is the DBA-reviewable way to provision the one
    table Womblex owns (e.g. into a shared/externally-managed DB via ``psql
    -f``). It must be byte-for-byte the DDL ``ensure_schema`` runs, or an
    operator who applies the file gets a table that differs from what
    ``--create-schema`` would make. Compared on the executable statements only
    — the file's leading ``--`` comment header is documentation.
    """
    from womblex.cloud.queue import _SCHEMA

    def _ddl_only(text: str) -> str:
        # Drop full-line SQL comments and blank lines; the file carries a
        # documentation header the embedded ``_SCHEMA`` string does not.
        lines = [
            ln for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("--")
        ]
        return "\n".join(lines).strip()

    sql_file = (REPO_ROOT / "sql" / "womblex_jobs.sql").read_text()
    assert _ddl_only(sql_file) == _ddl_only(_SCHEMA)


# --- RemoteStore (local backend) ---------------------------------------------


def test_is_remote_uri():
    assert is_remote_uri("s3://bucket/x")
    assert is_remote_uri("gs://bucket/x")
    assert not is_remote_uri("/tmp/x")
    assert not is_remote_uri("file:///tmp/x")


def test_storage_options_from_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    monkeypatch.setenv("WOMBLEX_S3_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    opts = storage_options_from_env("s3://bucket/x")
    assert opts["key"] == "k"
    assert opts["secret"] == "s"
    assert opts["client_kwargs"]["endpoint_url"] == "http://minio:9000"
    assert opts["client_kwargs"]["region_name"] == "us-east-1"
    # The options above are s3fs-shaped — other backends must not receive
    # them even with AWS env vars set.
    assert storage_options_from_env("gs://bucket/x") == {}
    assert storage_options_from_env("/tmp/x") == {}


def test_storage_options_prefers_store_specific_credentials(monkeypatch):
    """The store creds come from WOMBLEX_S3_* first, so a cloud deployment can
    give s3fs a MinIO/S3 key WITHOUT setting the process-global AWS_ACCESS_KEY_ID
    — which would otherwise clobber boto3's instance-role resolution for the
    isaacus-sagemaker SigV4 signer and 403 (the cloud credential-conflict bug).
    """
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setenv("WOMBLEX_S3_ACCESS_KEY_ID", "store-key")
    monkeypatch.setenv("WOMBLEX_S3_SECRET_ACCESS_KEY", "store-secret")
    opts = storage_options_from_env("s3://bucket/x")
    assert opts["key"] == "store-key"
    assert opts["secret"] == "store-secret"  # pragma: allowlist secret -- test literal

    # Store-specific wins over the AWS fallback when both are set.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    opts = storage_options_from_env("s3://bucket/x")
    assert opts["key"] == "store-key"
    assert opts["secret"] == "store-secret"  # pragma: allowlist secret -- test literal


def test_storage_options_omits_credentials_when_none_are_set(monkeypatch):
    """No store-specific and no AWS keys => s3fs gets no explicit credentials
    and falls back to its own chain (the EC2 instance role for real AWS S3).
    An endpoint override alone must not synthesise a half-set credential.
    """
    for var in (
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
        "WOMBLEX_S3_ACCESS_KEY_ID", "WOMBLEX_S3_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    opts = storage_options_from_env("s3://bucket/x")
    assert "key" not in opts
    assert "secret" not in opts


def test_storage_options_explicit_credentials_win_over_env(monkeypatch):
    """An operator-saved (Resources Console) credential override beats the env
    keys the Dockerfile baked in — so a rotated key is used moving forward
    without a container rebuild (issue 3).
    """
    monkeypatch.setenv("WOMBLEX_S3_ACCESS_KEY_ID", "baked-key")
    monkeypatch.setenv("WOMBLEX_S3_SECRET_ACCESS_KEY", "baked-secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    opts = storage_options_from_env("s3://bucket/x", credentials=("saved-key", "saved-secret"))
    assert opts["key"] == "saved-key"
    assert opts["secret"] == "saved-secret"  # pragma: allowlist secret -- test literal


def test_from_uri_threads_the_credential_override(tmp_path, monkeypatch):
    """`RemoteStore.from_uri(..., credentials=...)` passes the override through
    to the storage options for an s3 URI (a local URI ignores them, as it has
    no s3fs options at all).
    """
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("WOMBLEX_S3_ACCESS_KEY_ID", raising=False)
    from womblex.store import remote as remote_mod

    captured: dict = {}

    def _fake_url_to_fs(uri, **opts):
        captured.update(opts)
        # Return something url_to_fs-shaped without touching a network.
        import fsspec
        return fsspec.filesystem("memory"), "bucket/x"

    monkeypatch.setattr(remote_mod._require_fsspec().core, "url_to_fs", _fake_url_to_fs)
    remote_mod.RemoteStore.from_uri("s3://bucket/x", credentials=("ov-key", "ov-secret"))
    assert captured.get("key") == "ov-key"
    assert captured.get("secret") == "ov-secret"


def test_store_root_splits_bucket_from_prefix():
    assert store_root("s3://womblex/inbox") == ("womblex", "inbox")
    assert store_root("s3://womblex") == ("womblex", "")
    assert store_root("s3://womblex/runs/x") == ("womblex", "runs/x")
    # Local paths have no bucket concept.
    assert store_root("/data/inbox") == ("", "data/inbox")


def test_assert_disjoint_locations():
    """Same bucket, different
    folders is fine; either location containing the other is a hard fail.
    """
    # Disjoint: no error.
    assert_disjoint_locations("s3://womblex/inbox", "s3://womblex")
    assert_disjoint_locations("/data/inbox", "/data/out")

    # Ingest contains the output.
    with pytest.raises(ValueError, match="s3://womblex.*s3://womblex/runs"):
        assert_disjoint_locations("s3://womblex", "s3://womblex")

    # Ingest nested inside the output.
    with pytest.raises(ValueError):
        assert_disjoint_locations("s3://womblex/runs/x", "s3://womblex")

    # Different buckets never overlap, no matter the prefix.
    assert_disjoint_locations("s3://other-bucket/runs", "s3://womblex")


def test_assert_disjoint_locations_honours_a_custom_output_prefix():
    """`--output-prefix` moves where shards land, so the guard has to check
    that prefix, not a hardcoded `runs/`."""
    # Default prefix: inbox and runs/ are disjoint.
    assert_disjoint_locations("s3://womblex/inbox", "s3://womblex")
    # ...but shards directed *into* the inbox are exactly what the rule forbids.
    with pytest.raises(ValueError):
        assert_disjoint_locations(
            "s3://womblex/inbox", "s3://womblex", runs_prefix="inbox/out",
        )


def test_validate_location_uri_rejects_the_typos_operators_actually_make():
    """A hand-typed location that fsspec would silently read as a *relative
    local path* is the failure the Resources Console has to catch on save."""
    validate_location_uri("s3://womblex/inbox")
    validate_location_uri("/data/inbox")
    validate_location_uri("gs://bucket/inbox")

    with pytest.raises(ValueError, match="s3://"):
        validate_location_uri("s3:/womblex/inbox")  # one slash
    with pytest.raises(ValueError, match="lowercase"):
        validate_location_uri("S3://womblex/inbox")
    with pytest.raises(ValueError, match="not one of"):
        validate_location_uri("ftp://host/inbox")
    with pytest.raises(ValueError, match="no bucket"):
        validate_location_uri("s3://")
    with pytest.raises(ValueError, match="empty"):
        validate_location_uri("   ")


def test_validate_location_uri_runs_before_any_network_lookup():
    """An unsupported scheme is refused on the string, so validating an
    operator-supplied URI never resolves a hostname."""
    with pytest.raises(ValueError):
        store_root("ftp://never-resolved.invalid/inbox")


def test_same_location_ignores_spelling_that_opens_the_same_place():
    assert same_location("s3://womblex/inbox", "s3://womblex/inbox/")
    assert same_location("s3://womblex/inbox", "s3://womblex//inbox")
    assert not same_location("s3://womblex/inbox", "s3://womblex/outbox")
    assert not same_location("s3://womblex/inbox", "s3://other/inbox")


def test_worker_ingest_root_comparison_is_normalised():
    """An enqueue and a worker configured from different places (a flag here,
    a compose env var there) differ by a trailing slash routinely — that must
    not refuse every job."""
    from womblex.cloud.worker import _same_ingest

    assert _same_ingest("s3://womblex/inbox", "s3://womblex/inbox/")
    assert not _same_ingest("s3://womblex/inbox", "s3://wrong/inbox")
    # An unparseable root falls back to an exact match; the refusal path
    # itself must never raise.
    assert not _same_ingest("s3:/broken", "s3://womblex/inbox")
    assert _same_ingest("s3:/broken", "s3:/broken")


def test_list_files_recursive_reaches_nested_prefixes(tmp_path):
    """Object stores have a flat keyspace: `inbox/2026-08/foo.pdf` is one key,
    not a folder. A non-recursive listing of the ingest root reports zero
    documents for a perfectly normal upload layout."""
    store = RemoteStore.from_uri(str(tmp_path / "inbox"))
    store.upload_file(_touch(tmp_path / "top.pdf"), "top.pdf")
    store.upload_file(_touch(tmp_path / "nested.pdf"), "2026-08/agency/nested.pdf")

    assert store.list_files("", "*") == ["2026-08", "top.pdf"]
    assert sorted(store.list_files("", "*", recursive=True)) == [
        "2026-08", "2026-08/agency", "2026-08/agency/nested.pdf", "top.pdf",
    ]


def _touch(path: Path) -> Path:
    path.write_text("x")
    return path


def test_remote_store_file_roundtrip(tmp_path):
    store_root = tmp_path / "store"
    store_root.mkdir()
    store = RemoteStore.from_uri(str(store_root))

    src = tmp_path / "a.pdf"
    src.write_bytes(b"hello")
    store.upload_file(src, "inputs/a.pdf")

    assert store.exists("inputs/a.pdf")
    assert store.list_files("inputs", "*.pdf") == ["inputs/a.pdf"]

    out = store.download_file("inputs/a.pdf", tmp_path / "dl" / "a.pdf")
    assert out.read_bytes() == b"hello"


def test_remote_store_move_replaces_the_destination(tmp_path):
    """`move` is the commit step of a temp-key-then-move in-place rewrite: it
    renames a staged object onto a live key, replacing whatever was there, and
    removes the source. On the local backend this is a rename; on S3 it is a
    server-side copy-then-delete.
    """
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    src = tmp_path / "new.parquet"
    src.write_bytes(b"new-bytes")
    store.upload_file(src, "live/x.parquet")
    store.upload_file(src, ".staging/x.parquet")
    # Give the live key different bytes so we can prove the overwrite.
    old = tmp_path / "old.parquet"
    old.write_bytes(b"old-bytes")
    store.upload_file(old, "live/x.parquet")

    store.move(".staging/x.parquet", "live/x.parquet")

    assert not store.exists(".staging/x.parquet")  # source consumed
    dl = store.download_file("live/x.parquet", tmp_path / "dl" / "x.parquet")
    assert dl.read_bytes() == b"new-bytes"  # destination replaced


def test_remote_store_read_text_and_delete(tmp_path):
    """The small in-place read/delete the console's saved presets use (ui-plan merge 9)."""
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    src = tmp_path / "one.preset.json"
    src.write_text('{"name": "one"}', encoding="utf-8")
    store.upload_file(src, "presets/one.preset.json")

    assert store.read_text("presets/one.preset.json") == '{"name": "one"}'
    store.delete("presets/one.preset.json")
    assert not store.exists("presets/one.preset.json")


def test_remote_store_download_to_dir_and_upload_glob(tmp_path):
    store = RemoteStore.from_uri(str(tmp_path / "store"))

    for name in ("inputs/x.pdf", "inputs/y.docx"):
        f = tmp_path / Path(name).name
        f.write_bytes(name.encode())
        store.upload_file(f, name)

    local = tmp_path / "scratch"
    fetched = store.download_to_dir(["inputs/x.pdf", "inputs/y.docx"], local)
    assert sorted(p.name for p in fetched) == ["x.pdf", "y.docx"]

    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "batch-0001.elements.parquet").write_bytes(b"e")
    (shards / "batch-0001._manifest.parquet").write_bytes(b"m")
    (shards / "other.txt").write_bytes(b"skip")

    uploaded = store.upload_glob(shards, "batch-0001.*", "runs/r1/documents")
    assert set(uploaded) == {
        "runs/r1/documents/batch-0001.elements.parquet",
        "runs/r1/documents/batch-0001._manifest.parquet",
    }
    assert store.exists("runs/r1/documents/batch-0001.elements.parquet")
    assert not store.exists("runs/r1/documents/other.txt")


def test_download_to_dir_nested_keeps_same_named_keys_apart(tmp_path):
    """Two documents under different prefixes sharing a basename stay two files."""
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    for prefix in ("2026-07", "2026-08"):
        f = tmp_path / f"{prefix}.pdf"
        f.write_bytes(prefix.encode())
        store.upload_file(f, f"{prefix}/report.pdf")

    local = tmp_path / "scratch"
    fetched = store.download_to_dir(
        ["2026-07/report.pdf", "2026-08/report.pdf"], local, nested=True
    )

    assert [p.relative_to(local).as_posix() for p in fetched] == [
        "2026-07/report.pdf",
        "2026-08/report.pdf",
    ]
    assert [p.read_bytes() for p in fetched] == [b"2026-07", b"2026-08"]


def test_download_to_dir_refuses_a_key_that_would_escape_the_staging_dir(tmp_path):
    """A key is whatever a queue row or listing named — it does not get to pick
    where it lands. Refused before anything is written."""
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    local = tmp_path / "scratch"

    with pytest.raises(ValueError, match="does not stage under"):
        store.download_to_dir(["../escaped.pdf"], local, nested=True)

    assert not (tmp_path / "escaped.pdf").exists()


def test_remote_store_list_dirs(tmp_path):
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    (tmp_path / "store" / "runs" / "run-a" / "documents").mkdir(parents=True)
    (tmp_path / "store" / "runs" / "run-b" / "documents").mkdir(parents=True)
    (tmp_path / "store" / "runs" / "stray.txt").write_text("x")

    assert store.list_dirs("runs") == ["run-a", "run-b"]
    # Files under the prefix are not directories.
    assert "stray.txt" not in store.list_dirs("runs")
    # A prefix that doesn't exist yet returns empty, not an error.
    assert store.list_dirs("missing") == []


# --- worker: ingest as a distinct store (local, no Postgres) ----------------


def _minimal_config(tmp_path: Path):
    from womblex.config import (
        ChunkingConfig,
        DatasetConfig,
        ExtractionConfig,
        PathsConfig,
        RedactionConfig,
        WomblexConfig,
    )

    return WomblexConfig(
        dataset=DatasetConfig(name="w"),
        paths=PathsConfig(
            input_root=tmp_path, output_root=tmp_path / "out", checkpoint_dir=tmp_path / ".ckpt"
        ),
        extraction=ExtractionConfig(),
        chunking=ChunkingConfig(enabled=False),
        redaction=RedactionConfig(enabled=False),
    )


def test_process_job_downloads_from_a_second_ingest_store(tmp_path):
    """The gap merge 1 closes: inputs and outputs can be different stores."""
    from womblex.cloud.queue import Job
    from womblex.cloud.worker import _process_job

    ingest_store = RemoteStore.from_uri(str(tmp_path / "ingest"))
    csv = tmp_path / "people.csv"
    csv.write_text("name,role\nAlice,Director\n")
    ingest_store.upload_file(csv, "people.csv")

    output_store = RemoteStore.from_uri(str(tmp_path / "store"))
    job = Job(
        id=1, run_id="r1", batch_num=1, input_keys=["people.csv"],
        shard_prefix="runs/r1/documents", attempts=1,
    )

    _process_job(job, _minimal_config(tmp_path), output_store, ingest_store)

    assert output_store.list_files("runs/r1/documents", "*._manifest.parquet")
    assert not output_store.exists("people.csv")  # never lands in the output tree
    assert ingest_store.exists("people.csv")       # the source document is untouched


def test_process_job_extracts_both_same_named_documents(tmp_path):
    """A job's keys come from a recursive listing, so two prefixes routinely hold
    a `people.csv` apiece. Flat staging landed them on one local file — one
    document extracted twice, one not at all, and the recorded source path the
    survivor's. Both are extracted, each under its own relpath."""
    import pyarrow.parquet as pq

    from womblex.cloud.queue import Job
    from womblex.cloud.worker import _process_job

    ingest_store = RemoteStore.from_uri(str(tmp_path / "ingest"))
    for prefix, body in (
        ("2026-07", "name,role\nAlice,Director\n"),
        ("2026-08", "name,role\nBob,Assistant\n"),
    ):
        local = tmp_path / f"{prefix}.csv"
        local.write_text(body)
        ingest_store.upload_file(local, f"{prefix}/people.csv")

    output_store = RemoteStore.from_uri(str(tmp_path / "store"))
    job = Job(
        id=1, run_id="r1", batch_num=1,
        input_keys=["2026-07/people.csv", "2026-08/people.csv"],
        shard_prefix="runs/r1/documents", attempts=1,
        ingest_root="s3://bucket/inbox",
    )

    _process_job(job, _minimal_config(tmp_path), output_store, ingest_store)

    manifest = pq.read_table(
        str(tmp_path / "store" / "runs" / "r1" / "documents" / "batch-0001._manifest.parquet")
    ).to_pylist()
    assert len(manifest) == 2
    assert len({row["source_hash"] for row in manifest}) == 2  # distinct documents
    assert sorted(row["source_relpath"] for row in manifest) == [
        "2026-07/people.csv",
        "2026-08/people.csv",
    ]


# --- run logs ---------------------------------------------------------------


def test_capture_batch_log_attaches_and_detaches_cleanly(tmp_path):
    """The handler is added for the block and removed after — no leak that would
    tee the next batch's records into this file."""
    import logging

    from womblex.utils.run_log import capture_batch_log

    womblex_logger = logging.getLogger("womblex")
    before = list(womblex_logger.handlers)
    log_path = tmp_path / "batch.log"
    with capture_batch_log(log_path):
        assert len(womblex_logger.handlers) == len(before) + 1
        logging.getLogger("womblex.some.module").error("a captured line")
    assert womblex_logger.handlers == before  # detached
    assert "a captured line" in log_path.read_text()


def test_capture_batch_log_detaches_even_when_the_block_raises(tmp_path):
    import logging

    from womblex.utils.run_log import capture_batch_log

    womblex_logger = logging.getLogger("womblex")
    before = list(womblex_logger.handlers)
    log_path = tmp_path / "batch.log"
    with pytest.raises(RuntimeError), capture_batch_log(log_path):
        logging.getLogger("womblex").error("before the raise")
        raise RuntimeError("boom")
    assert womblex_logger.handlers == before
    # The file is complete and readable even though the block raised — the
    # failing case is the one the operator needs.
    assert "before the raise" in log_path.read_text()


def test_process_job_publishes_the_batch_log_beside_the_shards(tmp_path):
    """A successful batch leaves `runs/<run_id>/logs/batch-NNNN.log` in the store."""
    from womblex.cloud.queue import Job
    from womblex.cloud.worker import _process_job

    ingest_store = RemoteStore.from_uri(str(tmp_path / "ingest"))
    csv = tmp_path / "people.csv"
    csv.write_text("name,role\nAlice,Director\n")
    ingest_store.upload_file(csv, "people.csv")

    output_store = RemoteStore.from_uri(str(tmp_path / "store"))
    job = Job(
        id=1, run_id="r1", batch_num=3, input_keys=["people.csv"],
        shard_prefix="runs/r1/documents", attempts=1,
    )

    _process_job(job, _minimal_config(tmp_path), output_store, ingest_store)

    assert output_store.exists("runs/r1/logs/batch-0003.log")


def test_process_job_publishes_the_log_even_when_the_batch_fails(tmp_path, monkeypatch):
    """The failing case is the one that matters: the log is uploaded outside the
    try, so a job that raises still leaves its `batch-NNNN.log` in the store, and
    the original error is what propagates."""
    from womblex.cloud import worker as worker_mod
    from womblex.cloud.queue import Job

    ingest_store = RemoteStore.from_uri(str(tmp_path / "ingest"))
    csv = tmp_path / "people.csv"
    csv.write_text("name,role\nAlice,Director\n")
    ingest_store.upload_file(csv, "people.csv")
    output_store = RemoteStore.from_uri(str(tmp_path / "store"))

    def _boom(*_a, **_kw):
        raise RuntimeError("processing exploded")

    monkeypatch.setattr(worker_mod, "process_batch", _boom)
    job = Job(
        id=1, run_id="r1", batch_num=1, input_keys=["people.csv"],
        shard_prefix="runs/r1/documents", attempts=1,
    )

    with pytest.raises(RuntimeError, match="processing exploded"):
        worker_mod._process_job(job, _minimal_config(tmp_path), output_store, ingest_store)

    assert output_store.exists("runs/r1/logs/batch-0001.log")


# --- finalize (local store, no Postgres) -------------------------------------


def test_finalize_consolidates_manifest(tmp_path):
    """End-to-end finalize: real shards -> store -> consolidated manifest."""
    import argparse

    import pyarrow.parquet as pq

    from womblex.batch import process_batch
    from womblex.cli.cloud import cmd_finalize
    from womblex.config import (
        ChunkingConfig,
        DatasetConfig,
        ExtractionConfig,
        PathsConfig,
        RedactionConfig,
        WomblexConfig,
    )

    csv = tmp_path / "people.csv"
    csv.write_text("name,role\nAlice,Director\nBob,Analyst\n")
    cfg = WomblexConfig(
        dataset=DatasetConfig(name="fin"),
        paths=PathsConfig(
            input_root=tmp_path, output_root=tmp_path / "out", checkpoint_dir=tmp_path / ".ckpt"
        ),
        extraction=ExtractionConfig(),
        chunking=ChunkingConfig(enabled=False),
        redaction=RedactionConfig(enabled=False),
    )

    local_shards = tmp_path / "local_shards"
    local_shards.mkdir()
    process_batch([csv], cfg, batch_num=1, shard_dir=local_shards)

    store_root = tmp_path / "store"
    store = RemoteStore.from_uri(str(store_root))
    run_id = "rfin"
    for p in local_shards.glob("batch-0001._manifest.parquet"):
        store.upload_file(p, f"runs/{run_id}/documents/{p.name}")

    rc = cmd_finalize(argparse.Namespace(
        store=str(store_root), run_id=run_id, output_prefix=None, dsn=None,
    ))
    assert rc == 0
    assert store.exists(f"runs/{run_id}/manifest.parquet")

    dl = store.download_file(f"runs/{run_id}/manifest.parquet", tmp_path / "manifest.parquet")
    assert pq.read_table(dl).num_rows == 1  # one source document


# --- JobQueue (needs Postgres) -----------------------------------------------


def _dsn() -> str | None:
    return os.environ.get("WOMBLEX_DB_DSN") or os.environ.get("DATABASE_URL")


@pytest.fixture()
def queue():
    dsn = _dsn()
    if not dsn:
        pytest.skip("no Postgres DSN (set WOMBLEX_DB_DSN / DATABASE_URL)")
    pytest.importorskip("psycopg")
    from womblex.cloud.queue import JobQueue

    q = JobQueue(dsn)
    q.ensure_schema()
    run_id = f"test-{uuid.uuid4().hex[:8]}"
    yield q, run_id
    # Clean up this run's rows.
    with q.conn.transaction():
        q.conn.execute("DELETE FROM womblex_jobs WHERE run_id = %s", (run_id,))
    q.close()


def test_enqueue_idempotent(queue):
    from womblex.cloud.queue import JobSpec

    q, run_id = queue
    specs = [JobSpec(batch_num=1, input_keys=["a.pdf"], shard_prefix="runs/x/documents")]
    assert q.enqueue(run_id, specs) == 1
    assert q.enqueue(run_id, specs) == 0  # ON CONFLICT DO NOTHING
    assert q.stats(run_id) == {"pending": 1}


def test_claim_complete_and_fail(queue):
    from womblex.cloud.queue import JobSpec

    q, run_id = queue
    q.enqueue(run_id, [
        JobSpec(batch_num=1, input_keys=["a.pdf"], shard_prefix="p", max_attempts=1),
        JobSpec(batch_num=2, input_keys=["b.pdf"], shard_prefix="p", max_attempts=2),
    ])

    job1 = q.claim("w1", run_id)
    assert job1 is not None and job1.batch_num == 1
    q.complete(job1.id)

    job2 = q.claim("w1", run_id)
    assert job2 is not None and job2.batch_num == 2
    q.fail(job2.id, "boom")  # max_attempts=2, attempts now 1 -> back to pending

    stats = q.stats(run_id)
    assert stats.get("done") == 1
    assert stats.get("pending") == 1

    job2b = q.claim("w1", run_id)
    assert job2b is not None and job2b.batch_num == 2
    q.fail(job2b.id, "boom again")  # attempts now 2 == max -> failed
    assert q.stats(run_id).get("failed") == 1


def test_worker_refuses_a_job_whose_ingest_root_mismatches(queue, tmp_path):
    """A job enqueued against one ingest root and claimed by a worker reading
    from another is refused immediately, not failed per file.
    """
    from womblex.cloud.queue import JobSpec
    from womblex.cloud.worker import run_worker

    q, run_id = queue
    q.enqueue(run_id, [
        JobSpec(
            batch_num=1, input_keys=["a.pdf"], shard_prefix="runs/x/documents",
            ingest_root="s3://right-bucket/inbox",
        ),
    ])

    store = tmp_path / "store"
    store.mkdir()
    completed = run_worker(
        _dsn(), str(store), _minimal_config(tmp_path),
        ingest_uri="s3://wrong-bucket/inbox", run_id=run_id, once=True,
    )
    assert completed == 0

    # Released, not failed: the batch is fine, this worker is the wrong one
    # for it, so it waits for a correctly-wired one with its retries intact.
    assert q.stats(run_id) == {"pending": 1}
    rows = q.list_jobs(run_id)
    assert rows[0].attempts == 0
    assert "s3://right-bucket/inbox" in rows[0].error
    assert "s3://wrong-bucket/inbox" in rows[0].error


def test_release_returns_a_job_without_consuming_an_attempt(queue):
    """`fail` burns the retry budget; `release` is for a refusal that says
    nothing about the job itself."""
    from womblex.cloud.queue import JobSpec

    q, run_id = queue
    q.enqueue(run_id, [
        JobSpec(batch_num=1, input_keys=["a.pdf"], shard_prefix=f"runs/{run_id}/documents"),
    ])
    job = q.claim("w1", run_id)
    assert job is not None and job.attempts == 1

    q.release(job.id, "wrong worker")
    assert q.stats(run_id) == {"pending": 1}
    assert q.list_jobs(run_id)[0].attempts == 0

    # Still claimable, with the full budget.
    again = q.claim("w2", run_id)
    assert again is not None and again.attempts == 1


def test_worker_ingest_root_none_falls_back_to_the_store(queue, tmp_path):
    """A legacy job (ingest_root NULL) is accepted by any worker — no
    mismatch, since NULL means 'use the worker's own root'. Also today's
    single-store behaviour: no --ingest means inputs and outputs share the
    one --store.
    """
    from womblex.cloud.queue import JobSpec
    from womblex.cloud.worker import run_worker

    q, run_id = queue
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    csv = tmp_path / "people.csv"
    csv.write_text("name,role\nAlice,Director\n")
    store.upload_file(csv, "people.csv")

    q.enqueue(run_id, [
        JobSpec(batch_num=1, input_keys=["people.csv"], shard_prefix=f"runs/{run_id}/documents"),
    ])

    completed = run_worker(
        _dsn(), str(tmp_path / "store"), _minimal_config(tmp_path),
        run_id=run_id, once=True,
    )
    assert completed == 1
    assert q.stats(run_id) == {"done": 1}


# --- the dashboard's read-only views (docs/ui-plan.md merge 8) ----------------


def test_list_jobs_and_fleet(queue):
    """The job list and the fleet view are the queue table and `locked_by`."""
    from womblex.cloud.queue import JobSpec

    q, run_id = queue
    q.enqueue(run_id, [
        JobSpec(batch_num=1, input_keys=["a.pdf"], shard_prefix="p"),
        JobSpec(batch_num=2, input_keys=["b.pdf"], shard_prefix="p"),
    ])
    claimed = q.claim("worker-1", run_id)
    assert claimed is not None

    rows = q.list_jobs(run_id)
    assert {r.batch_num for r in rows} == {1, 2}
    running = next(r for r in rows if r.status == "running")
    assert running.locked_by == "worker-1"
    assert running.locked_at is not None  # ISO string, not a datetime
    assert running.attempts == 1

    # Status filter and run scoping both narrow rather than re-query.
    assert [r.batch_num for r in q.list_jobs(run_id, status="pending")] == [2]
    assert q.list_jobs("no-such-run") == []

    fleet = q.workers(run_id)
    assert [(w.worker_id, w.running) for w in fleet] == [("worker-1", 1)]
    assert fleet[0].oldest_locked_at is not None


def test_stale_jobs_matches_requeue_stale(queue):
    """The read-only twin must name exactly the rows `requeue_stale` recovers."""
    from womblex.cloud.queue import JobSpec

    q, run_id = queue
    q.enqueue(run_id, [JobSpec(batch_num=1, input_keys=["a.pdf"], shard_prefix="p")])
    claimed = q.claim("worker-1", run_id)
    assert claimed is not None

    # A just-claimed job is not stale; the same job against a zero threshold is.
    assert q.stale_jobs(3600, run_id) == []
    assert [j.id for j in q.stale_jobs(0, run_id)] == [claimed.id]
    # Unscoped, the read names exactly the rows the recovery acts on.
    assert len(q.stale_jobs(0)) == q.requeue_stale(0)
    assert q.stale_jobs(0, run_id) == []  # no longer running, so no longer stale


def test_throughput_counts_completions_in_the_window(queue):
    from womblex.cloud.queue import JobSpec

    q, run_id = queue
    q.enqueue(run_id, [JobSpec(batch_num=1, input_keys=["a.pdf"], shard_prefix="p")])
    claimed = q.claim("worker-1", run_id)
    assert claimed is not None
    q.complete(claimed.id)

    recent = q.throughput(run_id, window_seconds=3600)
    assert recent.completed == 1
    assert recent.per_minute == pytest.approx(1 / 60)
    assert recent.last_completed_at is not None
    # Scoped: another run's completions are not this run's throughput.
    assert q.throughput("no-such-run").completed == 0


# --- stage jobs (issue 5 part 2) --------------------------------------------


def test_stage_jobs_wait_for_extraction_then_claim_in_pipeline_order(queue):
    """The gate this merge adds: a stage row is claimable only once nothing
    earlier in its run is still pending or running — all of extraction, then
    each stage below it in `PIPELINE_ORDER`."""
    from womblex.cloud.queue import JobSpec

    q, run_id = queue
    q.enqueue(run_id, [JobSpec(batch_num=1, input_keys=["a.pdf"], shard_prefix="runs/x/documents")])
    assert q.enqueue_stages(run_id, ["embed", "enrich", "chunk"], "runs/x/documents") == 3
    assert q.enqueue_stages(run_id, ["enrich"], "runs/x/documents") == 0  # idempotent per stage

    batch = q.claim("w1", run_id)
    assert batch is not None and batch.kind == "batch" and batch.batch_num == 1
    # Extraction is running: no stage is due, even with three of them pending.
    assert q.claim("w2", run_id) is None
    q.complete(batch.id)

    first = q.claim("w1", run_id)
    assert first is not None and first.kind == "stage" and first.stage == "enrich"
    assert first.input_keys == []
    # enrich is running, so chunk waits on it — one stage of a run at a time.
    assert q.claim("w2", run_id) is None
    q.complete(first.id)

    rest = []
    while (job := q.claim("w1", run_id)) is not None:
        rest.append(job.stage)
        q.complete(job.id)
    assert rest == ["chunk", "embed"]


def test_a_failed_stage_does_not_wedge_the_ones_behind_it(queue):
    """A settled-as-failed row records the failure; the stages behind it still
    run and surface the gap as not-ready bases, rather than the operator finding
    a run stuck on rows that will never resolve."""
    q, run_id = queue
    q.enqueue_stages(run_id, ["enrich", "chunk"], "runs/x/documents", max_attempts=1)

    enrich = q.claim("w1", run_id)
    assert enrich is not None and enrich.stage == "enrich"
    q.fail(enrich.id, "boom")
    assert q.stats(run_id).get("failed") == 1

    chunk = q.claim("w1", run_id)
    assert chunk is not None and chunk.stage == "chunk"


def test_stage_rows_are_ordered_past_every_batch(queue):
    """`batch_num` doubles as the queue position, which is what makes
    `ORDER BY batch_num` drain extraction first with no second sort column."""
    from womblex.cloud.queue import STAGE_SEQ_BASE

    q, run_id = queue
    q.enqueue_stages(run_id, ["money"], "runs/x/documents")
    row = q.list_jobs(run_id)[0]
    assert row.kind == "stage"
    assert row.stage == "money"
    assert row.batch_num > STAGE_SEQ_BASE


def test_stage_jobs_of_other_runs_are_not_gated_by_this_one(queue):
    """The gate is per-run: another run's undrained extraction must not hold
    back this run's stages."""
    from womblex.cloud.queue import JobSpec

    q, run_id = queue
    other = f"{run_id}-other"
    try:
        q.enqueue(other, [JobSpec(batch_num=1, input_keys=["a.pdf"], shard_prefix="p")])
        q.enqueue_stages(run_id, ["money"], "runs/x/documents")

        stage = q.claim("w1", run_id)
        assert stage is not None and stage.kind == "stage" and stage.stage == "money"
    finally:
        with q.conn.transaction():
            q.conn.execute("DELETE FROM womblex_jobs WHERE run_id = %s", (other,))


def test_the_stage_columns_are_added_to_an_existing_table_in_place(queue):
    """A live deployment's `womblex_jobs` predates `kind`/`stage`. The additive
    ALTERs upgrade it with its rows intact, and those rows read as batches —
    there is no migration step for the operator to run."""
    q, run_id = queue
    with q.conn.transaction():
        q.conn.execute("ALTER TABLE womblex_jobs DROP COLUMN IF EXISTS kind")
        q.conn.execute("ALTER TABLE womblex_jobs DROP COLUMN IF EXISTS stage")
        q.conn.execute(
            "INSERT INTO womblex_jobs (run_id, batch_num, input_keys, shard_prefix) "
            "VALUES (%s, 1, %s, 'p')",
            (run_id, '["a.pdf"]'),
        )

    q.ensure_schema()  # idempotent, and the ALTERs land here

    row = q.list_jobs(run_id)[0]
    assert (row.kind, row.stage) == ("batch", None)
    assert q.enqueue_stages(run_id, ["money"], "runs/x/documents") == 1


def test_process_job_runs_the_stage_contract_and_publishes_its_log(tmp_path, monkeypatch):
    """A stage row hands the run's shard prefix to the same `run_stage_remote`
    `womblex run-stage` calls — the queue is a second dispatcher for it, not a
    second implementation."""
    from womblex.cloud import stage_runner as sr
    from womblex.cloud.queue import STAGE_SEQ_BASE, Job
    from womblex.cloud.worker import _process_job

    seen = {}

    def _fake_run(contract, store, shard_prefix, config, **kwargs):
        seen.update(stage=contract.name, prefix=shard_prefix,
                    checkpoint_prefix=kwargs.get("checkpoint_prefix"))
        return sr.StageRunSummary(stage=contract.name, processed=1, bases=1)

    monkeypatch.setattr(sr, "prepare_stage_context", lambda contract, config: sr.RunContext())
    monkeypatch.setattr(sr, "run_stage_remote", _fake_run)

    store = RemoteStore.from_uri(str(tmp_path / "store"))
    job = Job(
        id=1, run_id="r1", batch_num=STAGE_SEQ_BASE + 7, input_keys=[],
        shard_prefix="runs/r1/documents", attempts=1, kind="stage", stage="money",
    )
    _process_job(job, _minimal_config(tmp_path), store, store)

    assert seen == {
        "stage": "money",
        "prefix": "runs/r1/documents",
        # Staged, so a crashed stage resumes: the claim gate means no
        # concurrent runner of the same run can clobber it.
        "checkpoint_prefix": "runs/r1/.money-checkpoint",
    }
    assert store.exists("runs/r1/logs/stage-money.log")


def _stage_job(**over):
    from womblex.cloud.queue import Job

    return Job(
        id=1, run_id="r1", batch_num=1, input_keys=[], shard_prefix="runs/r1/documents",
        attempts=1, kind="stage", stage=over.pop("stage", "chunk"), **over,
    )


def _patch_stage_run(monkeypatch, summary_kwargs):
    from womblex.cloud import stage_runner as sr

    monkeypatch.setattr(sr, "prepare_stage_context", lambda contract, config: sr.RunContext())
    monkeypatch.setattr(
        sr, "run_stage_remote",
        lambda contract, store, prefix, config, **kw: sr.StageRunSummary(
            stage=contract.name, **summary_kwargs,
        ),
    )


def test_a_stage_that_publishes_nothing_fails_its_row(tmp_path, monkeypatch):
    """A non-zero summary must not read as done — the row records it and retries."""
    from womblex.cloud.worker import StageJobFailed, _process_job

    _patch_stage_run(monkeypatch, {"bases": 2, "failed": 2})

    store = RemoteStore.from_uri(str(tmp_path / "store"))
    with pytest.raises(StageJobFailed):
        _process_job(_stage_job(), _minimal_config(tmp_path), store, store)
    # The log is still published — that is the case the operator most needs it.
    assert store.exists("runs/r1/logs/stage-chunk.log")


def test_a_stage_blocked_on_an_absent_upstream_is_not_ready_not_failed(tmp_path, monkeypatch):
    """Every base awaiting a sidecar upstream has not written yet is "early",
    not "broken" — a distinct exception so the loop releases instead of
    spending an attempt (which would land the stage terminally failed on a
    slow-draining run)."""
    from womblex.cloud.worker import StageJobFailed, StageNotReady, _process_job

    _patch_stage_run(monkeypatch, {
        "bases": 2, "not_ready": 2, "not_ready_missing": {".enrichment_doc.parquet"},
    })

    store = RemoteStore.from_uri(str(tmp_path / "store"))
    with pytest.raises(StageNotReady, match=r"\.enrichment_doc\.parquet"):
        _process_job(_stage_job(), _minimal_config(tmp_path), store, store)
    assert not issubclass(StageNotReady, StageJobFailed)
    assert store.exists("runs/r1/logs/stage-chunk.log")


def test_a_partially_blocked_stage_still_succeeds(tmp_path, monkeypatch):
    """Not-ready under the base count is a still-draining fleet: exit 0, done."""
    from womblex.cloud.worker import _process_job

    _patch_stage_run(monkeypatch, {"bases": 3, "processed": 2, "not_ready": 1, "published": 2})

    store = RemoteStore.from_uri(str(tmp_path / "store"))
    _process_job(_stage_job(), _minimal_config(tmp_path), store, store)


def test_prepare_stage_context_refuses_a_stage_isaacus_cannot_serve(monkeypatch, tmp_path):
    """Without it `chunk_shards` warns, writes nothing and returns cleanly —
    a remote no-op a queue would record as a completed job.

    ``enrich`` always needs the API. ``chunk`` needs it only under AI chunking
    (``chunking_model``); plain token chunking runs offline on the vendored
    tokeniser, so its gate is config-aware."""
    from womblex.cloud.stage_contracts import STAGE_CONTRACTS
    from womblex.cloud.stage_runner import StagePreconditionError, prepare_stage_context
    from womblex.utils import availability

    monkeypatch.setattr(availability, "isaacus_available", lambda: False)

    # enrich unconditionally needs the API.
    with pytest.raises(StagePreconditionError, match="needs Isaacus"):
        prepare_stage_context(STAGE_CONTRACTS["enrich"], _minimal_config(tmp_path))

    # chunk WITH AI chunking needs the API.
    ai_cfg = _minimal_config(tmp_path)
    ai_cfg.chunking.chunking_model = "kanon-2-enricher"
    with pytest.raises(StagePreconditionError, match="needs Isaacus"):
        prepare_stage_context(STAGE_CONTRACTS["chunk"], ai_cfg)

    # chunk WITHOUT a chunking_model is offline token chunking — no API needed,
    # so it must NOT refuse even with Isaacus unavailable (the keyless local
    # chunking path).
    assert prepare_stage_context(STAGE_CONTRACTS["chunk"], _minimal_config(tmp_path)) is not None

    # A stage with no Isaacus need is unaffected.
    assert prepare_stage_context(STAGE_CONTRACTS["money"], _minimal_config(tmp_path)) is not None
