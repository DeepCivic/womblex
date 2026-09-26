"""Bundle builder: export one finalised local run into an egress bundle.

The producer half of ``docs/egress-bundle-contract.review.md``: Womblex writes
the bundle and stops — no retention, serving, versioning, or auth.

``build_bundle`` takes a finished local run (``<run_root>/documents/`` shard
files plus its consolidated ``<run_root>/manifest.parquet``, as
:func:`womblex.store.run_manifest.write_run_manifest` produces) and a
destination :class:`~womblex.store.remote.RemoteStore`, and writes::

    <dest>/<run_id>/
      corpus/                     # the shard files + manifest.parquet, unchanged
      sources/<source_hash><ext>  # raw files, deduplicated by hash
      source_index.parquet        # source_hash -> raw key, ext, doc_id, filename, status
      egress_manifest.json        # bundle descriptor + per-document resolution report

Resolution reuses :class:`~womblex.store.source_resolver.SourceResolver`, so
``source_index.parquet`` carries its four-status vocabulary, non-fatal per
document. Every write is a ``RemoteStore.upload_file`` off a local path, so a
local directory and an object-store URI are one code path. Scoped to a local
run, as ``SourceResolver`` is: a distributed run is staged locally first.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from womblex import __version__
from womblex.store.egress_output import (
    EGRESS_STAGE,
    SOURCE_INDEX_FILENAME,
    raw_key_for,
    write_source_index,
)
from womblex.store.feedback_output import is_safe_run_id
from womblex.store.remote import RemoteStore
from womblex.store.run_manifest import RUN_MANIFEST_FILENAME
from womblex.store.run_stamp import stamp_from_footers
from womblex.store.source_resolver import (
    HASH_BASIS_FILE_BYTES,
    Resolution,
    SourceResolver,
    hash_basis_for,
    load_manifest,
)

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
    run_root: Path,
    store: RemoteStore,
    *,
    run_id: str,
    bundle_prefix: str | None = None,
    include_sources: bool = True,
    source_root: str | Path | None = None,
) -> EgressResult:
    """Export the local run at *run_root* into a bundle under *store*.

    *run_root* is a finished ``womblex run`` output directory (``documents/``
    shard files plus ``manifest.parquet`` at its root); raises
    ``FileNotFoundError`` if either is missing — an unfinalised run has no
    manifest to build a bundle from.

    *bundle_prefix* defaults to *run_id*. *source_root* overrides where raw
    sources resolve from, as in :meth:`SourceResolver.for_run` (a moved corpus).
    ``include_sources=False`` resolves nothing and writes neither ``sources/``
    nor ``source_index.parquet`` — the "corpus-only" export. Raises
    ``ValueError``, before any write, if *run_id* disagrees with the run's stamp
    or sources are wanted with no single local root to resolve against.
    """
    run_root = Path(run_root)
    shard_dir = run_root / "documents"
    manifest_path = run_root / RUN_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} is missing — finalise the run "
            "(`womblex manifest --shards` or the end of `womblex run`) before egress"
        )
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"{shard_dir} is missing — nothing to bundle")

    prefix = (bundle_prefix or run_id).strip("/")
    if not prefix or not all(is_safe_run_id(segment) for segment in prefix.split("/")):
        raise ValueError(
            f"unsafe bundle destination {prefix!r} — each path segment must be a "
            "plain name (no '..', '/', or null byte)"
        )
    corpus_prefix = f"{prefix}/{CORPUS_DIRNAME}"

    # Everything that can refuse the export is checked before the first write,
    # so a refusal never leaves a half-built bundle at the destination.
    shard_files = sorted(shard_dir.glob("*.parquet")) + [manifest_path]
    stamp = stamp_from_footers(shard_files, EGRESS_STAGE)
    if stamp is not None and stamp.run_id != run_id:
        raise ValueError(f"run_id {run_id!r} does not match the run's own stamp {stamp.run_id!r}")
    manifest = load_manifest(run_root)
    rows = manifest.select(["source_hash", "doc_id", "filename", "ext"]).to_pylist()
    resolver = _resolver(run_root, manifest, source_root) if include_sources else None

    for path in shard_files:
        store.upload_file(path, f"{corpus_prefix}/{path.name}")
    logger.info("Copied %d corpus file(s) -> %s/%s", len(shard_files), store.root, corpus_prefix)

    sources_copied = 0
    by_status: dict[str, int] = {}
    if resolver is not None:
        resolutions: dict[str, Resolution] = {r.source_hash: r for r in resolver.resolve_all()}
        # Keyed by source_hash: rows sharing one hash can disagree on `ext`, so
        # every such row's raw_key comes from the one file uploaded — its own
        # suffix, lowercased as the manifest's `ext` is.
        raw_keys: dict[str, str] = {}
        index_rows: list[dict] = []
        for row in rows:
            source_hash = row["source_hash"]
            resolution = resolutions.get(source_hash)
            status = resolution.status if resolution is not None else "not_found"
            by_status[status] = by_status.get(status, 0) + 1
            raw_key = None
            if resolution is not None and resolution.ok and resolution.path is not None:
                if source_hash not in raw_keys:
                    raw_keys[source_hash] = raw_key_for(source_hash, resolution.path.suffix.lower())
                    store.upload_file(resolution.path, f"{prefix}/{raw_keys[source_hash]}")
                    sources_copied += 1
                raw_key = raw_keys[source_hash]
            index_rows.append({**row, "raw_key": raw_key, "status": status})

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
        corpus_files=len(shard_files), sources_copied=sources_copied,
        sources_by_status=by_status,
    )


def _resolver(
    run_root: Path, manifest: pa.Table, source_root: str | Path | None,
) -> SourceResolver:
    """The run's resolver. A run with no file-hashed row never reads a root (every
    row is ``unsupported_basis``), so a records ingest with no ``ingest_root`` exports."""
    methods = manifest.column("extraction_method").to_pylist()
    if source_root is None and all(
        hash_basis_for(str(m or "")) != HASH_BASIS_FILE_BYTES for m in methods
    ):
        return SourceResolver(manifest, run_root)
    return SourceResolver.for_run(run_root, root=source_root)


def _write_egress_manifest(
    store: RemoteStore, prefix: str, *, run_id: str, by_status: dict[str, int],
    documents: int, sources_copied: int,
) -> None:
    descriptor = {
        "run_id": run_id,
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
    "EgressResult",
    "build_bundle",
]
