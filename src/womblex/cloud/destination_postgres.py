"""A workflow's Postgres destination: one row per embedded chunk, with its pgvector embedding.

Each row names its text layer: the masked ``clean_text`` by default, the raw ``chunks`` text
only where the destination asks (D-3.2), and the chunk's ``mask_status`` when masked. A chunk
whose text the layer lacks is left out and counted, never sent from another layer. Redelivery
replaces: rows are upserted on (run, document, content type, chunk) in one transaction, and the
run's rows this delivery did not write are removed. The DSN is read from the environment
variable the destination names; pgvector's adapters come from the ``[pgvector]`` extra.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.cloud.workflow_store import PostgresDestination
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)

#: The text layer a row carries, named as the service API names it (`api/readers.LAYERS`).
LAYER_OF = {"masked": "clean_text", "raw": "chunks"}

_KEY = ("source_hash", "content_type", "chunk_index")


@dataclass(frozen=True)
class PostgresDelivery:
    rows: int
    skipped: int  # embedded chunks the layer held no text for


def _table(run: RemoteStore, key: str) -> list[dict[str, Any]]:
    with run.open(key) as handle:
        return pq.read_table(handle).to_pylist()  # type: ignore[no-any-return]


def chunk_rows(run: RemoteStore, text_layer: str) -> Iterator[tuple[list[dict[str, Any]], int]]:
    """Per batch, the rows to deliver and how many embedded chunks had no text in *text_layer*."""
    source = "clean_text" if text_layer == "masked" else "chunks"
    for key in sorted(run.list_files("documents", "*.embeddings.parquet")):
        texts_key = key.removesuffix(".embeddings.parquet") + f".{source}.parquet"
        texts = {tuple(r[k] for k in _KEY): r for r in _table(run, texts_key)} if run.exists(texts_key) else {}
        rows, skipped = [], 0
        for e in _table(run, key):
            t = texts.get(tuple(e[k] for k in _KEY))
            if t is None or t.get("text") is None:
                skipped += 1
                continue
            rows.append({**{k: e[k] for k in _KEY}, "text": t["text"], "text_layer": LAYER_OF[text_layer],
                         "mask_status": t.get("mask_status"), "model": e["model"], "embedding": e["vector"]})
        if skipped:
            logger.warning("%s: %d embedded chunk(s) have no %s text and are not delivered", key, skipped, source)
        yield rows, skipped


def _statements(table: str) -> tuple[Any, Any, Any]:
    from psycopg import sql

    ident = sql.Identifier(*table.split("."))
    create = sql.SQL(
        "CREATE TABLE IF NOT EXISTS {} (run_id text NOT NULL, source_hash text NOT NULL, "
        "content_type text NOT NULL, chunk_index integer NOT NULL, text_layer text NOT NULL, text text NOT NULL, "
        "mask_status text, model text, embedding vector NOT NULL, delivered_at timestamptz NOT NULL, "
        "PRIMARY KEY (run_id, source_hash, content_type, chunk_index))").format(ident)
    upsert = sql.SQL(
        "INSERT INTO {} (run_id, source_hash, content_type, chunk_index, text_layer, text, mask_status, model, "
        "embedding, delivered_at) VALUES (%(run_id)s, %(source_hash)s, %(content_type)s, %(chunk_index)s, "
        "%(text_layer)s, %(text)s, %(mask_status)s, %(model)s, %(embedding)s, %(delivered_at)s) "
        "ON CONFLICT (run_id, source_hash, content_type, chunk_index) DO UPDATE SET text_layer = "
        "EXCLUDED.text_layer, text = EXCLUDED.text, mask_status = EXCLUDED.mask_status, model = EXCLUDED.model, "
        "embedding = EXCLUDED.embedding, delivered_at = EXCLUDED.delivered_at").format(ident)
    stale = sql.SQL("DELETE FROM {} WHERE run_id = %s AND delivered_at < %s").format(ident)
    return create, upsert, stale


def deliver_postgres(d: PostgresDestination, run: RemoteStore, run_id: str) -> PostgresDelivery:
    """Upsert the run's embedded chunks into *d*'s table and drop the run's rows it did not write."""
    import numpy as np
    import psycopg
    from pgvector.psycopg import register_vector

    create, upsert, stale = _statements(d.table)
    total = skipped_total = 0
    with psycopg.connect(os.environ[d.dsn_env]) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute(create)
        now = conn.execute("SELECT now()").fetchone()[0]  # type: ignore[index]
        with conn.cursor() as cur:
            for rows, skipped in chunk_rows(run, d.text_layer):
                skipped_total += skipped
                cur.executemany(upsert, [{**r, "run_id": run_id, "delivered_at": now,
                                          "embedding": np.asarray(r["embedding"], dtype=np.float32)} for r in rows])
                total += len(rows)
            cur.execute(stale, (run_id, now))
    logger.info("delivered %d chunk row(s) of run %s to %s (%d without %s text)",
                total, run_id, d.table, skipped_total, LAYER_OF[d.text_layer])
    return PostgresDelivery(total, skipped_total)


__all__ = ["LAYER_OF", "PostgresDelivery", "chunk_rows", "deliver_postgres"]
