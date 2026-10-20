"""Write a distributed run's ``manifest.parquet`` in its store.

One body for both callers (Requirement 2): `womblex finalize`, on demand, and
the coordinator, as its last step once the run's stages have finished. It
stages in every ``*._manifest.parquet`` under ``<output_prefix>/documents``,
reads every shard's footer in place for the run record, consolidates them with
:func:`~womblex.store.run_manifest.write_run_manifest` and uploads
``<output_prefix>/manifest.parquet``. Idempotent: a re-run over the same shards
writes the same manifest, and one over more batches covers them too.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from womblex.store.run_manifest import RUN_MANIFEST_FILENAME, write_run_manifest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.cloud.jobs import RunBoard
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)


class NoShardManifests(Exception):
    """The run has no ``*._manifest.parquet`` under its shard prefix: nothing to consolidate."""


@dataclass(frozen=True)
class FinalizeResult:
    """What one finalisation wrote."""

    manifest_key: str
    shard_manifests: int
    footers_read: int
    footers_listed: int


def published_checksums(board: RunBoard, run_id: str, output_prefix: str) -> tuple[dict[str, str] | None, list[str]]:
    """The run's published files, run-relative, with their digests, and the workflows that may
    have published without one.

    Run-relative, so the record reads the same wherever the store is mounted. An empty read is
    a run recorded before checksums, not a run of no files, so it is ``None``.
    """
    published = board.file_checksums(run_id)
    checksums = {k.removeprefix(f"{output_prefix}/"): v for k, v in published.checksums.items()} or None
    if published.undigested:
        logger.warning("%d job(s) may have published files with no checksum: %s",
                       len(published.undigested), ", ".join(published.undigested))
    return checksums, published.undigested


def finalize_run(
    store: RemoteStore, output_prefix: str, *,
    checksums: dict[str, str] | None = None, undigested: list[str] | None = None,
) -> FinalizeResult:
    """Consolidate the shard manifests under *output_prefix* into its ``manifest.parquet``.

    Raises :class:`NoShardManifests` when there are none.
    """
    output_prefix = output_prefix.strip("/")
    shard_prefix = f"{output_prefix}/documents"
    manifest_keys = store.list_files(shard_prefix, "*._manifest.parquet")
    if not manifest_keys:
        raise NoShardManifests(f"no *._manifest.parquet under {shard_prefix}")

    # The run record is built from every shard's footer, but only the manifests
    # are staged in — so the footers are read where the shards actually live.
    # Without this the record would name extraction and nothing else, and a
    # missing stage is indistinguishable from one that never ran.
    shard_keys = store.list_files(shard_prefix, "*.parquet")
    footers = [observed for observed in (store.read_parquet_footer(k) for k in shard_keys) if observed is not None]
    logger.info("Read %d/%d shard footer(s) for the run record", len(footers), len(shard_keys))

    target = f"{output_prefix}/{RUN_MANIFEST_FILENAME}"
    with tempfile.TemporaryDirectory(prefix="womblex-finalize-") as tmp:
        docs = Path(tmp) / "documents"
        store.download_to_dir(manifest_keys, docs)
        local = write_run_manifest(docs, footers=footers, checksums=checksums, undigested=undigested)
        store.upload_file(local, target)
    return FinalizeResult(target, len(manifest_keys), len(footers), len(shard_keys))


__all__ = ["FinalizeResult", "NoShardManifests", "finalize_run", "published_checksums"]
