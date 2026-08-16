"""Thin pyarrow readers over shard sidecars, local and store-backed.

Local reads go straight through ``store/retention.py`` / ``store/output.py``.
Remote reads stage the (small) manifest into a temp dir and hand it to the
same local readers — the pattern ``cli/cloud.py``'s ``finalize`` already
uses, so there is one code path for the parquet logic.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import _SHARD_SUFFIX, read_manifest
from womblex.store.retention import STAGE_SUFFIXES, RunDescription, describe_run, list_runs
from womblex.store.run_manifest import RUN_MANIFEST_FILENAME
from womblex.store.shard_audit import ARCHIVE_SUFFIX, audit_shard_directory
from womblex.ui.deps import UISettings

if TYPE_CHECKING:
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)


def list_run_summaries(settings: UISettings) -> list[RunDescription]:
    """All runs the settings can see, newest run_id first."""
    if settings.is_remote:
        runs = _list_remote_runs(cast(str, settings.store_uri))
    else:
        runs = [describe_run(p) for p in list_runs(cast(Path, settings.output_root))]
    return sorted(runs, key=lambda d: d.run_id, reverse=True)


def get_manifest_rows(settings: UISettings, run_id: str) -> list[dict] | None:
    """Manifest rows (one per document) for run_id, or None if it doesn't exist."""
    if settings.is_remote:
        return _remote_manifest_rows(cast(str, settings.store_uri), run_id)
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    return list(_local_manifest_table(run_dir).to_pylist())


def get_stage_presence(settings: UISettings, run_id: str, stage: str) -> list[str] | None:
    """``source_hash`` values with a ``stage`` sidecar row in run_id.

    None if the run doesn't exist, else a (possibly empty) sorted list —
    the lifecycle-checkpoint switcher's data (docs/ui-plan.md §3 "lifecycle
    checkpoints are sidecar presence"). Reads only the ``source_hash``
    column of each sidecar, so this stays cheap even for large chunks /
    enrichment sidecars.
    """
    suffix = STAGE_SUFFIXES[stage]
    if settings.is_remote:
        return _remote_stage_presence(cast(str, settings.store_uri), run_id, suffix)
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    shard_dir = run_dir / "documents"
    if not shard_dir.is_dir():
        return []
    return _scan_stage_presence(shard_dir, suffix)


def get_shard_audit(settings: UISettings, run_id: str) -> dict | None:
    """Cross-batch structural audit for run_id — the verify-shards action's data."""
    if settings.is_remote:
        return _remote_shard_audit(cast(str, settings.store_uri), run_id)
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    return audit_shard_directory(run_dir / "documents").as_dict()


# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------


def _local_manifest_table(run_dir: Path) -> pa.Table:
    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    if manifest_path.exists():
        return pq.read_table(str(manifest_path))
    return read_manifest(run_dir / "documents")


def _scan_stage_presence(shard_dir: Path, suffix: str) -> list[str]:
    """``source_hash`` values across every ``*<suffix>`` sidecar in *shard_dir*.

    An unreadable sidecar is warned about and skipped rather than failing the
    whole screen: presence is an annotation on the documents grid, and one
    corrupt batch should not blank the other batches' answer. ``verify-shards``
    is the surface that reports corruption as such.
    """
    hashes: set[str] = set()
    for p in sorted(shard_dir.glob(f"*{suffix}")):
        if p.name.endswith(ARCHIVE_SUFFIX):
            continue
        try:
            col = pq.read_table(str(p), columns=["source_hash"]).column("source_hash")
        except Exception as e:
            logger.warning("stage-presence: skipping unreadable sidecar %s: %s", p.name, e)
            continue
        hashes.update(h for h in col.to_pylist() if h)
    return sorted(hashes)


# ---------------------------------------------------------------------------
# Remote (store-backed)
# ---------------------------------------------------------------------------


def _open_store(store_uri: str) -> RemoteStore:
    from womblex.store.remote import RemoteStore

    return RemoteStore.from_uri(store_uri)


def _list_remote_runs(store_uri: str) -> list[RunDescription]:
    store = _open_store(store_uri)
    return [_describe_remote_run(store, run_id) for run_id in store.list_dirs("runs")]


def _remote_manifest_rows(store_uri: str, run_id: str) -> list[dict] | None:
    store = _open_store(store_uri)
    if run_id not in store.list_dirs("runs"):
        return None
    return list(_remote_manifest_table(store, f"runs/{run_id}").to_pylist())


def _describe_remote_run(store: RemoteStore, run_id: str) -> RunDescription:
    prefix = f"runs/{run_id}"
    table = _remote_manifest_table(store, prefix)
    return RunDescription(
        run_id=run_id,
        document_count=table.num_rows,
        stages=_remote_stages_present(store, prefix),
        created_at=None,  # object stores don't expose this uniformly (docs/ui-plan.md §4)
        updated_at=None,
    )


def _remote_manifest_table(store: RemoteStore, prefix: str) -> pa.Table:
    """Stage *prefix*'s manifest into a temp dir and read it with the local reader.

    Prefers the consolidated ``manifest.parquet`` and falls back to the
    per-batch shard manifests, matching :func:`describe_run`'s local
    precedence — so a distributed run reads correctly both before and after
    ``womblex finalize`` has published the consolidated file.

    Assumes *prefix* names an existing run; callers check that first against
    ``list_dirs("runs")``.
    """
    manifest_key = f"{prefix}/{RUN_MANIFEST_FILENAME}"
    with tempfile.TemporaryDirectory(prefix="womblex-ui-") as tmp:
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


def _remote_stages_present(store: RemoteStore, prefix: str) -> tuple[str, ...]:
    return tuple(
        stage for stage, suffix in STAGE_SUFFIXES.items()
        if store.list_files(f"{prefix}/documents", f"*{suffix}")
    )


def _remote_stage_presence(store_uri: str, run_id: str, suffix: str) -> list[str] | None:
    store = _open_store(store_uri)
    if run_id not in store.list_dirs("runs"):
        return None
    keys = store.list_files(f"runs/{run_id}/documents", f"*{suffix}")
    if not keys:
        return []
    with tempfile.TemporaryDirectory(prefix="womblex-ui-") as tmp:
        tmp_dir = Path(tmp)
        store.download_to_dir(keys, tmp_dir)
        return _scan_stage_presence(tmp_dir, suffix)


def _remote_shard_audit(store_uri: str, run_id: str) -> dict | None:
    """Stage the four extraction-role shards and audit them locally.

    Restricted to ``_SHARD_SUFFIX`` rather than ``*.parquet`` on purpose:
    the audit reads only those roles, and a run's embeddings / enrichment
    sidecars are the largest files in it — globbing everything would pull
    the whole corpus over the network to count element kinds.
    """
    store = _open_store(store_uri)
    if run_id not in store.list_dirs("runs"):
        return None
    prefix = f"runs/{run_id}/documents"
    with tempfile.TemporaryDirectory(prefix="womblex-ui-") as tmp:
        tmp_dir = Path(tmp)
        for suffix in _SHARD_SUFFIX.values():
            keys = store.list_files(prefix, f"*{suffix}")
            if keys:
                store.download_to_dir(keys, tmp_dir)
        return audit_shard_directory(tmp_dir).as_dict()
