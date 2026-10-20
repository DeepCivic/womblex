"""Copy a run's raw sources into an egress bundle, one outcome per ``source_hash``.

Two places a source can come from, with one status vocabulary
(:mod:`womblex.store.source_resolver`'s four, plus ``upload_failed``):

- **A local corpus** (:class:`LocalSources`): `SourceResolver` finds and checks
  the file, which is then uploaded.
- **The object store a run was ingested from** (:class:`StoreSources`): the
  manifest's ``source_relpath`` is the key under the ingest root. The object is
  streamed to a temporary key in the bundle while it is hashed, and moved into
  place only when the hash equals ``source_hash`` — nothing is staged locally,
  and a mismatched copy never lands under its final name.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from urllib.parse import unquote

import pyarrow as pa

from womblex.store.egress_output import raw_key_for
from womblex.store.remote import RemoteStore
from womblex.store.source_resolver import (
    HASH_BASIS_FILE_BYTES,
    HASH_MISMATCH,
    NOT_FOUND,
    RESOLVED,
    UNSUPPORTED_BASIS,
    SourceResolver,
    hash_basis_for,
)

logger = logging.getLogger(__name__)

#: A source that resolved but whose copy to the destination raised. Not part
#: of :mod:`womblex.store.source_resolver`'s vocabulary — it happens at
#: upload time, after resolution has already succeeded.
UPLOAD_FAILED = "upload_failed"


@dataclass(frozen=True)
class SourceOutcome:
    """One ``source_hash``'s copy: its status, the bundle key it landed at (``None`` where no file
    was produced), and the key this export attempted, which stale-source cleanup leaves alone."""

    status: str
    raw_key: str | None = None
    attempted: str | None = None


class LocalSources:
    """Sources in a local corpus, found and checked by :class:`SourceResolver`."""

    def __init__(self, resolver: SourceResolver) -> None:
        self.resolver = resolver
        self._resolutions = {r.source_hash: r for r in resolver.resolve_all()}

    def copy(self, source_hash: str, store: RemoteStore, prefix: str) -> SourceOutcome:
        resolution = self._resolutions.get(source_hash)
        if resolution is None:
            return SourceOutcome(NOT_FOUND)
        if not resolution.ok or resolution.path is None:
            return SourceOutcome(resolution.status)
        key = raw_key_for(source_hash, resolution.path.suffix.lower())
        try:
            store.upload_file(resolution.path, f"{prefix}/{key}")
        except Exception:
            logger.warning("failed to upload source for source_hash=%s (doc_id=%s)",
                           source_hash, resolution.doc_id, exc_info=True)
            return SourceOutcome(UPLOAD_FAILED, attempted=key)
        return SourceOutcome(RESOLVED, key, key)


class _Hashing:
    """A read-through wrapper that hashes what it hands on."""

    def __init__(self, inner: BinaryIO) -> None:
        self.inner, self.digest = inner, hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self.inner.read(size)
        self.digest.update(data)
        return data


class StoreSources:
    """Sources in the object store a run was ingested from, keyed by ``source_relpath``.

    *ingest* is ``None`` for a run whose every row is records-hashed: there is nothing to fetch.
    """

    def __init__(self, manifest: pa.Table, ingest: RemoteStore | None) -> None:
        self.ingest = ingest
        self._rows: dict[str, dict] = {}
        names = [c for c in ("source_hash", "doc_id", "source_relpath", "extraction_method")
                 if c in manifest.schema.names]
        for row in manifest.select(names).to_pylist():
            self._rows.setdefault(row["source_hash"], row)

    def copy(self, source_hash: str, store: RemoteStore, prefix: str) -> SourceOutcome:
        row = self._rows.get(source_hash) or {}
        if hash_basis_for(str(row.get("extraction_method") or "")) != HASH_BASIS_FILE_BYTES:
            return SourceOutcome(UNSUPPORTED_BASIS)
        rel = str(row.get("source_relpath") or "").strip("/")
        try:
            present = self.ingest is not None and bool(rel) and self.ingest.exists(rel)
        except Exception:
            logger.warning("could not look up source %s for source_hash=%s", rel, source_hash, exc_info=True)
            present = False
        if self.ingest is None or not present:
            return SourceOutcome(NOT_FOUND)
        key = raw_key_for(source_hash, PurePosixPath(rel).suffix.lower())
        partial = f"{prefix}/{key}.partial"
        try:
            with self.ingest.open(rel) as src:
                hashing = _Hashing(src)
                store.write_stream(hashing, partial)  # type: ignore[arg-type]
            if hashing.digest.hexdigest() != source_hash:
                store.delete(partial)
                return SourceOutcome(HASH_MISMATCH)
            store.move(partial, f"{prefix}/{key}")
        except Exception:
            logger.warning("failed to copy source %s for source_hash=%s (doc_id=%s)",
                           rel, source_hash, row.get("doc_id"), exc_info=True)
            return SourceOutcome(UPLOAD_FAILED, attempted=key)
        return SourceOutcome(RESOLVED, key, key)


def declared_root(manifest: pa.Table) -> str | Path:
    """The one ingest root the manifest declares: a local path for ``file://``, else the store URI.

    Raises ``ValueError`` when the manifest declares none or several, which names nothing to
    resolve against.
    """
    declared = (
        {str(v or "") for v in manifest.column("ingest_root").to_pylist()} - {""}
        if "ingest_root" in manifest.schema.names else set()
    )
    if len(declared) != 1:
        raise ValueError(
            f"the run declares {len(declared)} ingest roots; pass a source root explicitly"
        )
    root = declared.pop()
    # `qualify_root` records a local root via `Path.as_uri()`, which percent-encodes.
    return Path(unquote(root.removeprefix("file://"))) if root.startswith("file://") else root


__all__ = ["UPLOAD_FAILED", "LocalSources", "SourceOutcome", "StoreSources", "declared_root"]
