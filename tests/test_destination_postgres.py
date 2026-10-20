"""The Postgres destination: refused before launch when it cannot hold, and one row per embedded chunk."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.test_workflow_runner import _stored
from womblex.cloud import delivery
from womblex.cloud.destination_postgres import PostgresDelivery, chunk_rows
from womblex.cloud.workflow_store import validate_workflow_spec
from womblex.store.remote import RemoteStore

_PG = {"kind": "postgres", "name": "warehouse", "dsn_env": "WOMBLEX_TEST_WAREHOUSE_DSN"}


def _spec(steps: list[str], **destination):
    return validate_workflow_spec({"preset": "p", "steps": [{"stage": s} for s in steps],
                                   "destinations": [{**_PG, **destination}]})


@pytest.mark.parametrize("steps, destination, message", [
    (["chunk", "pii"], {}, "needs the `embed` step"),
    (["chunk", "embed"], {}, "masked text needs the `pii` step"),
    (["chunk", "embed", "pii"], {"table": "chunks; drop table x"}, "table must be"),
])
def test_what_a_postgres_destination_cannot_hold_is_refused(steps, destination, message):
    with pytest.raises(ValueError, match=message):
        _spec(steps, **destination)


def test_raw_text_needs_no_pii_step():
    (d,) = _spec(["chunk", "embed"], text_layer="raw").destinations
    assert (d.table, d.text_layer) == ("womblex_chunks", "raw")


def test_its_secret_and_its_driver_are_checked_before_launch(monkeypatch):
    (d,) = _spec(["chunk", "embed", "pii"]).destinations
    monkeypatch.delenv(_PG["dsn_env"], raising=False)
    monkeypatch.setitem(sys.modules, "pgvector.psycopg", None)  # the extra is not installed
    problems = [p for _, p in delivery.check_destinations((d,))]
    assert problems == ["environment variable WOMBLEX_TEST_WAREHOUSE_DSN not set",
                        "pgvector is not installed (install womblex[pgvector])"]


def _write(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), str(path))


@pytest.fixture
def run(tmp_path: Path) -> RemoteStore:
    docs = tmp_path / "run" / "documents"
    docs.mkdir(parents=True)
    key = {"source_hash": "h1", "content_type": "narrative"}
    _write(docs / "batch-0001.embeddings.parquet", [
        {**key, "chunk_index": i, "model": "m", "task": "t", "dim": 2, "vector": [0.1, 0.2]} for i in (0, 1)])
    _write(docs / "batch-0001.chunks.parquet", [{**key, "chunk_index": i, "text": f"Alice {i}"} for i in (0, 1)])
    _write(docs / "batch-0001.clean_text.parquet", [
        {**key, "chunk_index": 0, "text": "<PERSON_1> 0", "n_masked": 1, "mask_status": "masked"}])
    _write(docs / "batch-0002.embeddings.parquet", [
        {**key, "source_hash": "h2", "chunk_index": 0, "model": "m", "task": "t", "dim": 2, "vector": [0.3, 0.4]}])
    _write(docs / "batch-0002.chunks.parquet", [{**key, "source_hash": "h2", "chunk_index": 0, "text": "Bob"}])
    return RemoteStore.from_uri(str(tmp_path / "run"))


def test_masked_rows_carry_the_clean_text_and_never_fall_back_to_raw(run: RemoteStore):
    batches = list(chunk_rows(run, "masked"))
    assert [([(r["source_hash"], r["chunk_index"], r["text"], r["text_layer"], r["mask_status"]) for r in rows],
             skipped) for rows, skipped in batches] == [
        ([("h1", 0, "<PERSON_1> 0", "clean_text", "masked")], 1),  # chunk 1 has no masked text
        ([], 1),  # batch 2 was never masked: none of its raw text is sent
    ]


def test_raw_rows_name_their_layer(run: RemoteStore):
    rows = [r for batch, _ in chunk_rows(run, "raw") for r in batch]
    assert [(r["text"], r["text_layer"], r["embedding"]) for r in rows] == [
        ("Alice 0", "chunks", [0.1, 0.2]), ("Alice 1", "chunks", [0.1, 0.2]), ("Bob", "chunks", [0.3, 0.4])]


def test_a_postgres_delivery_is_recorded_with_its_rows(run: RemoteStore, monkeypatch):
    import womblex.cloud.destination_postgres as pg

    monkeypatch.setattr(pg, "deliver_postgres", lambda d, r, run_id: PostgresDelivery(rows=3, skipped=1))
    stored = _stored(["chunk", "embed", "pii"], destinations=[_PG])
    (outcome,) = delivery.deliver(stored, run, "r1")
    assert (outcome.status, outcome.uri, outcome.rows, outcome.rows_skipped) == ("delivered", "womblex_chunks", 3, 1)


@pytest.mark.skipif(not os.environ.get("WOMBLEX_TEST_PGVECTOR_DSN"),
                    reason="needs a Postgres server with pgvector (WOMBLEX_TEST_PGVECTOR_DSN)")
def test_a_redelivery_upserts_and_drops_the_runs_stale_rows(run: RemoteStore, monkeypatch, tmp_path: Path):
    pytest.importorskip("pgvector.psycopg")
    import psycopg

    from womblex.cloud.destination_postgres import deliver_postgres

    monkeypatch.setenv(_PG["dsn_env"], os.environ["WOMBLEX_TEST_PGVECTOR_DSN"])
    (d,) = _spec(["chunk", "embed"], text_layer="raw", table="womblex_test_chunks").destinations
    with psycopg.connect(os.environ[_PG["dsn_env"]]) as conn:
        conn.execute("DROP TABLE IF EXISTS womblex_test_chunks")
    assert deliver_postgres(d, run, "r1").rows == 3
    (tmp_path / "run" / "documents" / "batch-0002.embeddings.parquet").unlink()
    assert deliver_postgres(d, run, "r1").rows == 2
    with psycopg.connect(os.environ[_PG["dsn_env"]]) as conn:
        assert conn.execute("SELECT count(*) FROM womblex_test_chunks WHERE run_id = 'r1'").fetchone() == (2,)
