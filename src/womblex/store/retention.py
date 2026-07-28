"""Run-output retention: list, generate, and purge run directories.

Each pipeline run writes its outputs under ``<output_root>/<run_id>/`` and
its checkpoint state under ``<checkpoint_dir>/<run_id>/``. Multiple runs
co-exist for stage-vs-stage comparison; retention controls how many old
runs are kept on disk.

Two policies:

- ``rolling`` — keep the ``keep`` most-recent runs (including current),
  purge older. Default ``keep=2`` (current + previous).
- ``keep_all`` — disable auto-purge; the user manages purges manually.

The current run is *always* preserved regardless of policy or position.
Retention is normally applied at the start of a new run, before any
output is written.
"""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_run_id() -> str:
    """Generate a sortable, filesystem-safe run id from current UTC time.

    Format: ``run-YYYYMMDDTHHMMSSZ`` — sorts lexically by creation order
    and parses back as an ISO-8601 basic-format timestamp.
    """
    return "run-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def list_runs(output_root: Path) -> list[Path]:
    """Return run directories under ``output_root``, sorted oldest-first.

    A "run directory" is any subdirectory whose name starts with ``run-``
    (matching the auto-generated ``run-YYYYMMDDTHHMMSSZ`` convention).
    Subdirectories with other names are preserved — they are not subject
    to auto-retention. This protects legacy / hand-named output dirs
    (e.g. ``output/documents/`` from a pre-run_id layout, or
    ``output/baseline-snapshot/``) from being purged when a new run lands.

    To bring a hand-named run under the retention policy, name it with a
    ``run-`` prefix (either via ``dataset.run_id`` in config or
    ``--run-id`` on the CLI).

    Sort key is directory name (so timestamp-default ids sort by
    creation order) with mtime as a secondary key for user-supplied
    ``run-*`` names.
    """
    if not output_root.is_dir():
        return []
    runs = [
        p for p in output_root.iterdir()
        if p.is_dir() and p.name.startswith("run-")
    ]
    runs.sort(key=lambda p: (p.name, p.stat().st_mtime))
    return runs


def most_recent_run(output_root: Path) -> Path | None:
    """Return the newest run directory under ``output_root``, or None."""
    runs = list_runs(output_root)
    return runs[-1] if runs else None


def apply_retention(
    output_root: Path,
    checkpoint_dir: Path,
    *,
    current_run_id: str,
    policy: str,
    keep: int,
) -> list[Path]:
    """Purge old run directories per policy.

    Args:
        output_root: Parent directory holding per-run subdirs.
        checkpoint_dir: Parent of per-run checkpoint subdirs (purged in lockstep).
        current_run_id: Name of the active run. Always preserved.
        policy: ``rolling`` (default) or ``keep_all``.
        keep: Number of runs to retain under ``rolling`` (>= 1).

    Returns:
        List of purged run directories (empty under ``keep_all`` or if
        nothing exceeds the retention window).

    Raises:
        ValueError: Unknown policy or ``keep < 1``.
    """
    if policy == "keep_all":
        return []
    if policy != "rolling":
        raise ValueError(f"unknown retention policy: {policy!r}")
    if keep < 1:
        raise ValueError(f"retention.keep must be >= 1, got {keep}")

    runs = list_runs(output_root)
    # Newest-first
    runs.sort(key=lambda p: (p.name, p.stat().st_mtime), reverse=True)

    others = [p for p in runs if p.name != current_run_id]
    # Reserve one slot for the current run (whether or not it's on disk yet)
    retain_others = max(keep - 1, 0)
    to_purge = others[retain_others:]

    purged: list[Path] = []
    for run_dir in to_purge:
        logger.info("retention: purging old run %s", run_dir)
        shutil.rmtree(run_dir)
        ckpt = checkpoint_dir / run_dir.name
        if ckpt.is_dir():
            shutil.rmtree(ckpt)
        purged.append(run_dir)
    return purged
