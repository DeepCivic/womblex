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
    opts = storage_options_from_env("s3://bucket/x")
    assert opts["key"] == "k"
    assert opts["secret"] == "s"
    assert opts["client_kwargs"]["endpoint_url"] == "http://minio:9000"
    assert opts["client_kwargs"]["region_name"] == "us-east-1"
    # The options above are s3fs-shaped — other backends must not receive
    # them even with AWS env vars set.
    assert storage_options_from_env("gs://bucket/x") == {}
    assert storage_options_from_env("/tmp/x") == {}


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
