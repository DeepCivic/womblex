"""A run's batch logs: the per-document failure lines a worker or `womblex run` publishes to ``logs/``.

`utils/run_log.capture_batch_log` writes ``batch-NNNN.log`` beside a run's
shards. A name is checked against that pattern before any path join, so a
caller cannot reach outside the run's ``logs/`` prefix. ``None`` means the run
does not exist; an empty list, a run written before logs were published.
"""
from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from womblex.store.run_reads import RUNS_PREFIX, store_with_run

if TYPE_CHECKING:
    from womblex.cloud.deployment import DeploymentSettings
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)

_LOG_NAME = re.compile(r"batch-\d{4}\.log\Z")


def is_safe_log_name(name: str) -> bool:
    """True when *name* is a batch log's filename and nothing else."""
    return bool(_LOG_NAME.fullmatch(name))


def list_run_logs(settings: DeploymentSettings, run_id: str) -> list[dict[str, Any]] | None:
    """``name``, ``size`` and ``modified`` (UTC, or ``None`` where a store omits it) per log, newest first."""
    if settings.is_remote:
        store = store_with_run(settings, run_id)
        if store is None:
            return None
        entries = [
            _store_entry(store, key, key.rsplit("/", 1)[-1])
            for key in store.list_files(f"{RUNS_PREFIX}/{run_id}/logs", "*")
            if is_safe_log_name(key.rsplit("/", 1)[-1])
        ]
    else:
        run_dir = cast(Path, settings.output_root) / run_id
        if not run_dir.is_dir():
            return None
        logs_dir = run_dir / "logs"
        entries = [
            {"name": p.name, "size": p.stat().st_size, "modified": _utc(p.stat().st_mtime)}
            for p in (logs_dir.iterdir() if logs_dir.is_dir() else ())
            if p.is_file() and is_safe_log_name(p.name)
        ]
    return sorted(entries, key=lambda e: e["name"], reverse=True)


def read_run_log(settings: DeploymentSettings, run_id: str, name: str) -> str | None:
    """One log's text; ``None`` when the run or the log is absent, or *name* is not a log's name."""
    if not is_safe_log_name(name):
        return None
    if settings.is_remote:
        store = store_with_run(settings, run_id)
        key = f"{RUNS_PREFIX}/{run_id}/logs/{name}"
        return store.read_text(key) if store is not None and store.exists(key) else None
    path = cast(Path, settings.output_root) / run_id / "logs" / name
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None


def _utc(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _store_entry(store: RemoteStore, key: str, name: str) -> dict[str, Any]:
    """Size and modified time for one stored log, tolerating a backend that omits either."""
    size, modified = 0, None
    try:
        info: dict = store.fs.info(f"{store.root}/{key}")  # type: ignore[attr-defined]
        size = int(info.get("size") or info.get("Size") or 0)
        mtime = info.get("LastModified") or info.get("mtime")
        if isinstance(mtime, datetime):
            modified = mtime.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif isinstance(mtime, int | float):
            modified = _utc(mtime)
    except Exception as e:
        logger.warning("run logs: could not stat %s: %s", key, e)
    return {"name": name, "size": size, "modified": modified}
