"""Retry a failed stage by forking its workflow (`docs/decisions-operations.md`, D-2.1).

DBOS keeps the steps before the one a fork starts at, so a stage retried from its first
failed unit does not redo the units that finished. The fork is `<run>:stage:<s>:retry:N`,
carries the original's attributes, and runs on the stage's queue. A retry is a launch, so
every check that can refuse runs first and nothing is forked once one does.

A stage that ran after a failure ran on partial output, so once the failed stage's latest
attempt has succeeded those stages are offered a retry too, in the order the operator runs
them: a stage is retryable when it failed, or when it ran after a failure whose stage has
since been re-run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import sqlalchemy.exc

from womblex.cloud import dbos_app
from womblex.cloud.jobs import RunBoard, _attr, _board_status

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dbos import WorkflowStatus

    from womblex.config import WomblexConfig
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)


class RetryCode(StrEnum):
    NOT_FOUND = "not_found"
    NOT_FAILED = "not_failed"
    STILL_RUNNING = "still_running"
    EARLIER_FAILED = "earlier_failed"
    CONFIG_INVALID = "config_invalid"
    STEP_ORDER = "step_order"
    ALREADY_RETRIED = "already_retried"


class RetryRefusedError(Exception):
    """A retry that was refused before anything was forked."""

    def __init__(self, message: str, code: RetryCode):
        super().__init__(f"[{code.value}] {message}")
        self.code = code


@dataclass(frozen=True)
class StageRetry:
    """A forked retry: its id, what it forked from, and the stages offered a retry once it succeeds."""

    run_id: str
    stage: str
    retry_id: str
    forked_from: str
    start_step: int
    later: tuple[str, ...]


def _latest(attempts: dict[str, list[WorkflowStatus]], stage: str) -> WorkflowStatus:
    return attempts[stage][-1]


def _ran_after(attempts: dict[str, list[WorkflowStatus]], stage: str) -> list[str]:
    """The stages whose latest attempt succeeded having run after *stage* failed."""
    return sorted(s for s in attempts if s != stage and _board_status(_latest(attempts, s)) == "done"
                  and stage in (_attr(_latest(attempts, s), "ran_after_failed") or []))


def _check_retryable(attempts: dict[str, list[WorkflowStatus]], run_id: str, stage: str) -> None:
    """Raise unless *stage* failed, or ran on the partial output of a failure since re-run."""
    if stage not in attempts:
        raise RetryRefusedError(f"run {run_id!r} has no dispatched stage {stage!r}", RetryCode.NOT_FOUND)
    latest = _latest(attempts, stage)
    status = _board_status(latest)
    if status in ("pending", "running"):
        raise RetryRefusedError(f"{latest.workflow_id} is still {status}", RetryCode.STILL_RUNNING)
    if status == "failed":
        return
    after = [f for f in _attr(latest, "ran_after_failed") or [] if f in attempts]
    not_cured = [f for f in after if _board_status(_latest(attempts, f)) != "done"]
    if not_cured:
        raise RetryRefusedError(
            f"{stage} ran after {', '.join(not_cured)} failed, and that stage has not succeeded since: "
            f"retry {not_cured[0]} first", RetryCode.EARLIER_FAILED)
    if not any((_latest(attempts, f).created_at or 0) > (latest.created_at or 0) for f in after):
        raise RetryRefusedError(
            f"{latest.workflow_id} succeeded and ran on output no retry has changed", RetryCode.NOT_FAILED)


def _stage_config(base: WomblexConfig, workflow: dict[str, Any] | None) -> WomblexConfig:
    """The config the original attempt ran under, as the worker builds it (`workflows._stage_config`)."""
    if workflow is None:
        return base
    from womblex.cloud.workflow_runner import config_from_overlay

    return config_from_overlay(workflow["preset"], workflow["config"], base.paths.model_dump(mode="json"))


def retry_stage(
    board: RunBoard, run_id: str, stage: str, *, config: WomblexConfig,
    store: RemoteStore | None = None, owner: str | None = None,
) -> StageRetry:
    """Fork *stage*'s latest attempt from its first failed unit as the next `:retry:N`.

    Refused (`RetryRefusedError`) before the fork if the stage is unknown, still running, not failed
    (and not a stage that ran on a failure since re-run), or fails the checks a launch makes: its
    config loads with registered models, and its required inputs are in *store*. `RunOwnedError`
    if *owner* is named and does not own the run.
    """
    from womblex.cloud.workflow_order import StageOrderError, check_dispatch_order
    from womblex.utils.model_check import check_registered

    attempts = board.stage_attempts(run_id, owner=owner)
    _check_retryable(attempts, run_id, stage)
    latest = _latest(attempts, stage)
    shard_prefix, workflow = board.stage_arguments(latest.workflow_id)
    try:
        stage_config = _stage_config(config, workflow)
        check_registered(stage_config)  # registration only: this host may not carry the workers' models
    except ValueError as e:
        raise RetryRefusedError(f"config does not load: {e}", RetryCode.CONFIG_INVALID) from e
    try:
        check_dispatch_order([stage], stage_config, store, shard_prefix)
    except StageOrderError as e:
        raise RetryRefusedError(f"the stage's inputs are not there: {e}", RetryCode.STEP_ORDER) from e

    original, number = dbos_app.stage_attempt(latest.workflow_id) or (latest.workflow_id, 0)
    new_id = dbos_app.stage_retry_id(run_id, stage, number + 1)
    start = board.first_failed_step(latest.workflow_id)
    try:
        board.fork_stage(latest.workflow_id, stage, new_id, start)
    except sqlalchemy.exc.IntegrityError as e:  # two retries raced for the same attempt number
        raise RetryRefusedError(f"{new_id} already exists", RetryCode.ALREADY_RETRIED) from e
    logger.info("run %s: %s forked from step %d as %s (original %s)", run_id, stage, start, new_id, original)
    return StageRetry(run_id, stage, new_id, latest.workflow_id, start, tuple(_ran_after(attempts, stage)))


__all__ = ["RetryCode", "RetryRefusedError", "StageRetry", "retry_stage"]
