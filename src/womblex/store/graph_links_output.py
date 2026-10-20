"""``*.graph_links.parquet`` schema + IO — graph refresh's own sidecar.

One row per narrative mention: the chunk(s) it falls in, found from character
offsets. The footer records the digest of the chunks the links were computed
against (`chunks_digest`), so a reader takes them only while the chunks beside
them are unchanged; after a rechunk they stop matching and the reader falls
back. The ``mentioned_in`` edges are not stored: each is (mention, chunk) plus
the mention's offsets, so a reader derives them from these rows.

Self-contained, like ``store/embed_output.py``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import _write_rows, chunks_path_for, read_chunks
from womblex.store.run_stamp import sidecar_footer

logger = logging.getLogger(__name__)

GRAPH_LINKS_SUFFIX = ".graph_links.parquet"

#: Footer key holding the digest of the chunks the links were computed against.
CHUNKS_DIGEST_KEY = b"womblex.chunks_digest"

# Joinable to `*.enrichment_entities.parquet` on
# (source_hash, entity_id, mention_start, mention_end). `chunk_index` is the first
# overlapping narrative chunk (-1 for none); `chunk_indices` is all of them.
GRAPH_LINKS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("entity_id", pa.string()),
    ("mention_start", pa.int32()),
    ("mention_end", pa.int32()),
    ("chunk_index", pa.int32()),
    ("chunk_indices", pa.list_(pa.int32())),
])

_CHUNK_COLUMNS = ("source_hash", "content_type", "chunk_index", "start_char", "end_char", "text")


def graph_links_path_for(base_path: Path) -> Path:
    """Return ``<base>.graph_links.parquet`` sibling for a shard base path."""
    return base_path.parent / f"{base_path.stem}{GRAPH_LINKS_SUFFIX}"


def chunks_digest(base_path: Path) -> str:
    """SHA-256 over the batch's chunk rows (identity, offsets and text); changes when a rechunk does."""
    h = hashlib.sha256()
    table = read_chunks(base_path).select(list(_CHUNK_COLUMNS))
    for row in sorted(table.to_pylist(), key=lambda r: (r["source_hash"], r["content_type"], r["chunk_index"])):
        h.update(repr(tuple(row[c] for c in _CHUNK_COLUMNS)).encode())
    return h.hexdigest()


def write_graph_links(rows: list[dict], base_path: Path) -> Path:
    """Write a batch's links, stamped with the digest of the chunks beside them.

    With no chunks sidecar there is nothing to link against: the file carries no digest and
    `read_matching_links` never returns it.
    """
    target = graph_links_path_for(base_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(sidecar_footer(base_path, "graph-refresh") or {})
    if chunks_path_for(base_path).exists():
        metadata[CHUNKS_DIGEST_KEY] = chunks_digest(base_path).encode()
    _write_rows(rows, target, GRAPH_LINKS_SCHEMA, metadata=metadata)
    logger.info("Wrote graph links %s: rows=%d", target.name, len(rows))
    return target


def read_matching_links(base_path: Path) -> pa.Table | None:
    """The batch's links when they were computed against the chunks now beside them, else ``None``.

    ``None`` covers a run written before this sidecar existed, a missing chunks
    sidecar, and links left behind by a rechunk.
    """
    path = graph_links_path_for(base_path)
    try:
        table = pq.read_table(str(path))
        stamped = (pq.read_metadata(str(path)).metadata or {}).get(CHUNKS_DIGEST_KEY)
        if stamped is None or stamped.decode() != chunks_digest(base_path):
            return None
    except (FileNotFoundError, OSError):
        return None
    return table.cast(GRAPH_LINKS_SCHEMA)


__all__ = [
    "CHUNKS_DIGEST_KEY",
    "GRAPH_LINKS_SCHEMA",
    "GRAPH_LINKS_SUFFIX",
    "chunks_digest",
    "graph_links_path_for",
    "read_matching_links",
    "write_graph_links",
]
