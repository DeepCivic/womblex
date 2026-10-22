"""Reading a stored run's document table from an object store, in place of a worker's disk.

The remote manifest is staged into a temp dir and handed to the same local
reader a finished run is read with, so there is one code path for the parquet
logic. The service API reads a run's documents through here.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import read_manifest
from womblex.store.run_manifest import RUN_MANIFEST_FILENAME

if TYPE_CHECKING:
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)


class StoreUnreachable(Exception):
    """The configured object store could not be opened or reached.

    Raised by :func:`open_store` when ``RemoteStore.from_uri`` fails: a missing
    backend package (``ImportError: Install s3fs to access S3``), bad
    credentials, or an unreachable endpoint. It carries the underlying message,
    so a caller can report the actionable cause rather than an opaque fault.
    """


def open_store(store_uri: str) -> RemoteStore:
    """Open the store at *store_uri*, or raise :class:`StoreUnreachable`."""
    from womblex.store.remote import RemoteStore

    try:
        return RemoteStore.from_uri(store_uri)
    except Exception as e:
        logger.warning("store unreachable (%s): %s", store_uri, e)
        raise StoreUnreachable(str(e)) from e


def remote_manifest_rows(store_uri: str, run_id: str) -> list[dict] | None:
    """Manifest rows (one per document) for *run_id*, or ``None`` if the run does not exist."""
    store = open_store(store_uri)
    if run_id not in store.list_dirs("runs"):
        return None
    return list(remote_manifest_table(store, f"runs/{run_id}").to_pylist())


def remote_manifest_table(store: RemoteStore, prefix: str) -> pa.Table:
    """Stage *prefix*'s manifest into a temp dir and read it with the local reader.

    Prefers the consolidated ``manifest.parquet`` and falls back to the
    per-batch shard manifests, so a distributed run reads correctly both before
    and after ``womblex finalize`` has published the consolidated file.

    Assumes *prefix* names an existing run; callers check that first against
    ``list_dirs("runs")``.
    """
    manifest_key = f"{prefix}/{RUN_MANIFEST_FILENAME}"
    with tempfile.TemporaryDirectory(prefix="womblex-") as tmp:
        tmp_dir = Path(tmp)
        if store.exists(manifest_key):
            local = store.download_file(manifest_key, tmp_dir / RUN_MANIFEST_FILENAME)
            return pq.read_table(str(local))
        documents_dir = tmp_dir / "documents"
        documents_dir.mkdir()
        manifest_keys = store.list_files(f"{prefix}/documents", "*._manifest.parquet")
        if manifest_keys:
            store.download_to_dir(manifest_keys, documents_dir)
        return read_manifest(documents_dir)
