"""Each stage's checkpoint progress for one run, in pipeline order.

Every shard stage writes its `CheckpointState` to a dot-directory inside the
run (``STAGE_CONTRACTS``' ``checkpoint_dirname``), so progress is readable
from the run itself, locally or from a store, with or without a queue. Only
stages with a checkpoint appear; an absent one has not run.
"""
from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from womblex.cloud.stage_contracts import CHECKPOINT_DIRNAMES
from womblex.store.checkpoint import CHECKPOINT_GLOB, CheckpointProgress, read_checkpoints
from womblex.store.retention import is_safe_run_id
from womblex.store.run_reads import RUNS_PREFIX, store_with_run

if TYPE_CHECKING:
    from womblex.cloud.deployment import DeploymentSettings


def stage_progress(settings: DeploymentSettings, run_id: str) -> list[dict[str, Any]] | None:
    """One row per stage checkpoint (``stage`` plus `CheckpointProgress`); ``None`` when there is no such run."""
    if not is_safe_run_id(run_id):
        return None
    found: dict[str, list[CheckpointProgress]] = {}
    if settings.is_remote:
        store = store_with_run(settings, run_id)
        if store is None:
            return None
        with tempfile.TemporaryDirectory(prefix="womblex-") as tmp:
            for stage, dirname in CHECKPOINT_DIRNAMES.items():
                keys = store.list_files(f"{RUNS_PREFIX}/{run_id}/{dirname}", CHECKPOINT_GLOB)
                if keys:
                    (Path(tmp) / dirname).mkdir()
                    store.download_to_dir(keys, Path(tmp) / dirname)
                    found[stage] = read_checkpoints(Path(tmp) / dirname)
    else:
        run_dir = cast(Path, settings.output_root) / run_id
        if not run_dir.is_dir():
            return None
        found = {stage: read_checkpoints(run_dir / dirname) for stage, dirname in CHECKPOINT_DIRNAMES.items()}
    return [{"stage": stage, **asdict(p)} for stage, entries in found.items() for p in entries]
