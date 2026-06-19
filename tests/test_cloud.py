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

from womblex.store.remote import RemoteStore, is_remote_uri, storage_options_from_env

# RemoteStore reaches fsspec lazily; skip the whole module without the cloud extra.
pytest.importorskip("fsspec")


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
    opts = storage_options_from_env()
    assert opts["key"] == "k"
    assert opts["secret"] == "s"
    assert opts["client_kwargs"]["endpoint_url"] == "http://minio:9000"
    assert opts["client_kwargs"]["region_name"] == "us-east-1"


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
