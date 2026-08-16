"""Thin pyarrow readers over shard sidecars, local and store-backed.

Local reads go straight through ``store/retention.py`` / ``store/output.py``.
Remote reads stage the (small) manifest into a temp dir and hand it to the
same local readers — the pattern ``cli/cloud.py``'s ``finalize`` already
uses, so there is one code path for the parquet logic.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import read_manifest
from womblex.store.retention import STAGE_SUFFIXES, RunDescription, describe_run, list_runs
from womblex.store.run_manifest import RUN_MANIFEST_FILENAME
from womblex.ui.deps import UISettings

if TYPE_CHECKING:
    from womblex.store.remote import RemoteStore


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


# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------


def _local_manifest_table(run_dir: Path) -> pa.Table:
    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    if manifest_path.exists():
        return pq.read_table(str(manifest_path))
    return read_manifest(run_dir / "documents")


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
