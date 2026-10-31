"""A deployment's runs as its store or output root holds them, whatever the queue knows.

Every reader takes a `DeploymentSettings` and answers for either run source:
a local ``output_root`` (``<root>/<run_id>/``) or a store (``runs/<run_id>/``).
A store read stages only the small files it needs into a temp dir and hands
them to the local reader, so the parquet logic has one code path. A per-run
reader returns ``None`` when the run does not exist, and for a run id that is
not a single safe path component (`is_safe_run_id`), so ``..`` never reaches the
parent of ``output_root``; an unreachable store is `StoreUnreachable`.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow.parquet as pq

from womblex.store.enrichment_output import LEGACY_HASH_COLUMN
from womblex.store.output import _SHARD_SUFFIX
from womblex.store.remote_manifest import open_store, remote_manifest_table
from womblex.store.retention import STAGE_SUFFIXES, RunDescription, describe_run, is_safe_run_id, list_runs
from womblex.store.shard_audit import ARCHIVE_SUFFIX, audit_shard_directory

if TYPE_CHECKING:
    from womblex.cloud.deployment import DeploymentSettings
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)

RUNS_PREFIX = "runs"


def hash_column(path: str, *, filesystem: object | None = None) -> str:
    """``source_hash``, or the pre-0.6.0 name when the file carries that instead (one footer read)."""
    names = pq.read_schema(path, filesystem=filesystem).names
    return "source_hash" if "source_hash" in names else LEGACY_HASH_COLUMN


def store_with_run(settings: DeploymentSettings, run_id: str) -> RemoteStore | None:
    """The deployment's store, or ``None`` when it holds no run *run_id*."""
    store = open_store(cast(str, settings.store_uri), settings.store_options)
    return store if run_id in store.list_dirs(RUNS_PREFIX) else None


def list_run_summaries(settings: DeploymentSettings) -> list[RunDescription]:
    """Every run the source holds, with its document count and stages present, newest id first."""
    if settings.is_remote:
        store = open_store(cast(str, settings.store_uri), settings.store_options)
        runs = [_describe_remote_run(store, run_id) for run_id in store.list_dirs(RUNS_PREFIX)]
    else:
        runs = [describe_run(p) for p in list_runs(cast(Path, settings.output_root))]
    return sorted(runs, key=lambda d: d.run_id, reverse=True)


def stage_presence(settings: DeploymentSettings, run_id: str, stage: str) -> list[str] | None:
    """The ``source_hash`` values with a *stage* sidecar row in *run_id*, sorted.

    Reads only the hash column. ``KeyError`` for a stage with no sidecar of its own.
    """
    suffix = STAGE_SUFFIXES[stage]
    if not is_safe_run_id(run_id):
        return None
    if settings.is_remote:
        store = store_with_run(settings, run_id)
        if store is None:
            return None
        keys = store.list_files(f"{RUNS_PREFIX}/{run_id}/documents", f"*{suffix}")
        if not keys:
            return []
        with tempfile.TemporaryDirectory(prefix="womblex-") as tmp:
            store.download_to_dir(keys, Path(tmp))
            return _scan_stage_presence(Path(tmp), suffix)
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    shard_dir = run_dir / "documents"
    return _scan_stage_presence(shard_dir, suffix) if shard_dir.is_dir() else []


def shard_audit(settings: DeploymentSettings, run_id: str) -> dict | None:
    """`audit_shard_directory` over *run_id*'s extraction shards, as a dict.

    From a store, only the four extraction roles are staged: the audit reads
    nothing else, and the embedding and enrichment sidecars are the run's
    largest files.
    """
    if not is_safe_run_id(run_id):
        return None
    if settings.is_remote:
        store = store_with_run(settings, run_id)
        if store is None:
            return None
        prefix = f"{RUNS_PREFIX}/{run_id}/documents"
        with tempfile.TemporaryDirectory(prefix="womblex-") as tmp:
            for suffix in _SHARD_SUFFIX.values():
                keys = store.list_files(prefix, f"*{suffix}")
                if keys:
                    store.download_to_dir(keys, Path(tmp))
            return audit_shard_directory(Path(tmp)).as_dict()
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    return audit_shard_directory(run_dir / "documents").as_dict()


def _describe_remote_run(store: RemoteStore, run_id: str) -> RunDescription:
    prefix = f"{RUNS_PREFIX}/{run_id}"
    return RunDescription(
        run_id=run_id,
        document_count=remote_manifest_table(store, prefix).num_rows,
        stages=tuple(s for s, suffix in STAGE_SUFFIXES.items() if store.list_files(f"{prefix}/documents", f"*{suffix}")),
        created_at=None,  # an object store exposes no uniform creation time
        updated_at=None,
    )


def _scan_stage_presence(shard_dir: Path, suffix: str) -> list[str]:
    """``source_hash`` values across every ``*<suffix>`` file in *shard_dir*.

    An unreadable sidecar is warned about and skipped: presence annotates
    documents, and `verify-shards` is what reports corruption.
    """
    hashes: set[str] = set()
    for p in sorted(shard_dir.glob(f"*{suffix}")):
        if p.name.endswith(ARCHIVE_SUFFIX):
            continue
        try:
            column = hash_column(str(p))
            hashes.update(h for h in pq.read_table(str(p), columns=[column]).column(column).to_pylist() if h)
        except Exception as e:
            logger.warning("stage presence: skipping unreadable sidecar %s: %s", p.name, e)
    return sorted(hashes)
