"""Bundle builder: export one finalised local run into an egress bundle.

The producer half of ``docs/egress.md``: Womblex writes the bundle and
stops — no retention, serving, versioning, or auth.

``build_bundle`` takes a finished local run (``<run_root>/documents/`` — every
file it contains, recursively, not only its top-level ``*.parquet`` shards —
plus its consolidated ``<run_root>/manifest.parquet``, as
:func:`womblex.store.run_manifest.write_run_manifest` produces) and a
destination :class:`~womblex.store.remote.RemoteStore`, and writes::

    <dest>/<run_id>/
      corpus/                     # documents/ mirrored recursively + manifest.parquet
      sources/<source_hash><ext>  # raw files, deduplicated by hash
      source_index.parquet        # source_hash -> raw key, ext, doc_id, filename, status
      egress_manifest.json        # bundle descriptor + source counts by status

Resolution reuses :class:`~womblex.store.source_resolver.SourceResolver`, so
``source_index.parquet`` carries its four-status vocabulary plus one more of
this module's own: ``upload_failed``, for a source that resolved but whose
copy to the destination raised — isolated per source so one bad upload does
not abort the export (``source_index.parquet`` and ``egress_manifest.json``
still get written). The run is read through a ``RemoteStore`` (a local
directory through fsspec's local backend), so a local run and one held in an
object store are copied by one code path, store to store, with no local
staging. A run ingested from an object store has its sources fetched from that
store by ``source_relpath`` and checked against ``source_hash``
(``store/egress_sources.py``), with the same statuses.

Re-exporting into a bundle folder that already has a ``sources/`` directory
(a re-run over an updated corpus) removes any file under it that the new
export did not just write or attempt to write — so a document dropped from
the manifest since the last export does not linger as an orphaned raw file
``source_index.parquet`` no longer lists. A source this export attempted but
failed to upload is left untouched either way, since its destination content
is of unknown provenance.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex import __version__
from womblex.store.contract import CONTRACT_VERSION
from womblex.store.egress_output import (
    EGRESS_STAGE,
    SOURCE_INDEX_FILENAME,
    SOURCES_DIRNAME,
    write_source_index,
)
from womblex.store.egress_sources import UPLOAD_FAILED, LocalSources, StoreSources, declared_root
from womblex.store.output import conform_shard
from womblex.store.remote import RemoteStore, is_remote_uri
from womblex.store.retention import is_safe_run_id
from womblex.store.run_manifest import RUN_MANIFEST_FILENAME
from womblex.store.run_stamp import stamp_from_metadata
from womblex.store.source_resolver import HASH_BASIS_FILE_BYTES, SourceResolver, hash_basis_for

logger = logging.getLogger(__name__)

CORPUS_DIRNAME = "corpus"
EGRESS_MANIFEST_FILENAME = "egress_manifest.json"

@dataclass(frozen=True)
class EgressResult:
    """What one :func:`build_bundle` call produced."""

    run_id: str
    bundle_prefix: str
    documents: int
    corpus_files: int
    sources_copied: int
    sources_by_status: dict[str, int]


def build_bundle(
    run_root: Path | str,
    store: RemoteStore,
    *,
    run_id: str,
    bundle_prefix: str | None = None,
    include_sources: bool = True,
    source_root: str | Path | None = None,
) -> EgressResult:
    """Export the run at *run_root* into a bundle under *store*.

    *run_root* is a finished run's output directory (``documents/`` shard files
    plus ``manifest.parquet`` at its root): a local path, or an object-store URI
    whose files are copied store to store with no local staging. Raises
    ``FileNotFoundError`` if either is missing — an unfinalised run has no
    manifest to build a bundle from.

    *bundle_prefix* defaults to *run_id*. *source_root* overrides where raw
    sources resolve from (a moved corpus), a local path or a store URI.
    ``include_sources=False`` resolves nothing and writes neither ``sources/``
    nor ``source_index.parquet`` — the "corpus-only" export. Raises
    ``ValueError``, before any write, if *run_id* disagrees with the run's stamp
    or sources are wanted with no single root to resolve against.
    """
    run = _open_run(run_root)
    if not run.exists(RUN_MANIFEST_FILENAME):
        raise FileNotFoundError(
            f"{run.root}/{RUN_MANIFEST_FILENAME} is missing — finalise the run "
            "(`womblex manifest --shards`, `womblex finalize`, or the end of the run) before egress"
        )
    # Recursive and not filtered to `*.parquet`: any file `documents/` holds
    # (a nested subfolder, a non-parquet sidecar) is part of the corpus, not
    # just its top-level shards.
    if not run.exists("documents"):
        raise FileNotFoundError(f"{run.root}/documents is missing — nothing to bundle")
    shard_keys = sorted(run.find("documents"))

    prefix = (bundle_prefix or run_id).strip("/")
    if not prefix or not all(is_safe_run_id(segment) for segment in prefix.split("/")):
        raise ValueError(
            f"unsafe bundle destination {prefix!r} — each path segment must be a "
            "plain name (no '..', '/', or null byte)"
        )
    corpus_prefix = f"{prefix}/{CORPUS_DIRNAME}"

    # Everything that can refuse the export is checked before the first write,
    # so a refusal never leaves a half-built bundle at the destination.
    corpus_keys = [*shard_keys, RUN_MANIFEST_FILENAME]
    footers = (meta for key in corpus_keys if key.endswith(".parquet")
               if (meta := run.read_parquet_footer(key)) is not None)
    stamp = stamp_from_metadata((m for m, _ in footers), EGRESS_STAGE)
    if stamp is not None and stamp.run_id != run_id:
        raise ValueError(f"run_id {run_id!r} does not match the run's own stamp {stamp.run_id!r}")
    with run.open(RUN_MANIFEST_FILENAME) as handle:
        manifest = conform_shard(pq.read_table(handle), "manifest", f"{run.root}/{RUN_MANIFEST_FILENAME}")
    rows = manifest.select(["source_hash", "doc_id", "filename", "ext"]).to_pylist()
    sources = _sources(run_root, manifest, source_root) if include_sources else None

    for key in corpus_keys:
        with run.open(key) as handle:
            store.write_stream(handle, f"{corpus_prefix}/{key.removeprefix('documents/')}")
    logger.info("Copied %d corpus file(s) -> %s/%s", len(corpus_keys), store.root, corpus_prefix)

    sources_copied = 0
    by_status: dict[str, int] = {}
    if sources is not None:
        # One outcome per hash: rows sharing one hash can disagree on `ext`, so
        # every such row's raw_key comes from the one file copied.
        outcomes = {h: sources.copy(h, store, prefix) for h in dict.fromkeys(r["source_hash"] for r in rows)}
        index_rows: list[dict] = []
        for row in rows:
            outcome = outcomes[row["source_hash"]]
            by_status[outcome.status] = by_status.get(outcome.status, 0) + 1
            index_rows.append({**row, "raw_key": outcome.raw_key, "status": outcome.status})
        sources_copied = sum(1 for o in outcomes.values() if o.raw_key is not None)

        # A failed upload's key is kept too: its destination content (if any
        # landed before the failure) is of unknown provenance, so it is neither
        # claimed by the new index nor deleted out from under it.
        keep = {o.attempted for o in outcomes.values() if o.attempted}
        _clean_stale_sources(store, prefix, keep=keep)

        with tempfile.TemporaryDirectory(prefix="womblex-egress-") as tmp:
            local_index = write_source_index(index_rows, Path(tmp), stamp=stamp)
            store.upload_file(local_index, f"{prefix}/{SOURCE_INDEX_FILENAME}")
        logger.info(
            "Resolved %d source(s) for %s: %s", len(rows), run_id,
            ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())),
        )

    _write_egress_manifest(store, prefix, run_id=run_id, by_status=by_status,
                            documents=len(rows), sources_copied=sources_copied)

    return EgressResult(
        run_id=run_id, bundle_prefix=prefix, documents=len(rows),
        corpus_files=len(corpus_keys), sources_copied=sources_copied,
        sources_by_status=by_status,
    )


def _open_run(run_root: Path | str) -> RemoteStore:
    """The run's files as a store: an object store's in place, a local directory through fsspec's
    local backend, so both are copied by one code path."""
    if isinstance(run_root, str) and is_remote_uri(run_root):
        return RemoteStore.from_uri(run_root)
    return RemoteStore.from_uri(str(Path(run_root).expanduser().resolve()))


def _clean_stale_sources(store: RemoteStore, prefix: str, *, keep: set[str]) -> None:
    """Remove anything under this bundle's ``sources/`` that *keep* does not name.

    Runs after this export's uploads, so a re-export over an updated corpus —
    a document dropped, a hash changed — does not leave a raw file behind that
    the fresh ``source_index.parquet`` no longer lists. *keep* already
    includes the keys of failed uploads (see the caller), so a transient
    upload failure never causes a previously-copied file to be deleted.

    Isolated per file, like the upload loop this runs after: one delete that
    raises (a transient backend error, an object already gone out-of-band)
    is logged and skipped rather than aborting the export before
    ``source_index.parquet`` / ``egress_manifest.json`` get written.
    """
    sources_prefix = f"{prefix}/{SOURCES_DIRNAME}"
    existing = store.list_files(sources_prefix, recursive=True)
    keep_full = {f"{prefix}/{k}" for k in keep}
    stale = [rel for rel in existing if rel not in keep_full]
    removed = 0
    for rel in stale:
        try:
            store.delete(rel)
        except Exception:
            logger.warning("failed to remove stale source file %s", rel, exc_info=True)
        else:
            removed += 1
    if removed:
        logger.info("Removed %d stale source file(s) from a previous export", removed)


def _sources(
    run_root: Path | str, manifest: pa.Table, source_root: str | Path | None,
) -> LocalSources | StoreSources:
    """Where the run's raw sources are copied from: a local corpus (`SourceResolver`), or the
    object store it was ingested from, by `source_relpath`.

    A run with no file-hashed row never reads a root (every row is ``unsupported_basis``), so a
    records ingest with no ``ingest_root`` exports.
    """
    if source_root is None:
        methods = manifest.column("extraction_method").to_pylist()
        if all(hash_basis_for(str(m or "")) != HASH_BASIS_FILE_BYTES for m in methods):
            return StoreSources(manifest, None)
        source_root = declared_root(manifest)
    if isinstance(source_root, str) and is_remote_uri(source_root):
        ingest = RemoteStore.from_uri(source_root)
        try:  # an unreadable store is refused here, before the first write, not found row by row
            ingest.exists("")
        except Exception as e:
            raise ValueError(f"cannot read sources from {source_root}: {e}") from e
        return StoreSources(manifest, ingest)
    return LocalSources(SourceResolver(manifest, source_root))


def _write_egress_manifest(
    store: RemoteStore, prefix: str, *, run_id: str, by_status: dict[str, int],
    documents: int, sources_copied: int,
) -> None:
    descriptor = {
        "run_id": run_id,
        "contract_version": CONTRACT_VERSION,
        "womblex_version": __version__,
        "created_at_iso": datetime.now(UTC).isoformat(),
        "documents": documents,
        "sources_copied": sources_copied,
        "sources_by_status": by_status,
    }
    with tempfile.TemporaryDirectory(prefix="womblex-egress-") as tmp:
        local_path = Path(tmp) / EGRESS_MANIFEST_FILENAME
        local_path.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n")
        store.upload_file(local_path, f"{prefix}/{EGRESS_MANIFEST_FILENAME}")


__all__ = [
    "CORPUS_DIRNAME",
    "EGRESS_MANIFEST_FILENAME",
    "UPLOAD_FAILED",
    "EgressResult",
    "build_bundle",
]
