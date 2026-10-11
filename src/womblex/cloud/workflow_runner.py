"""Local multi-stage runner: execute a stored workflow's steps over a shard directory.

Per step it composes what `womblex run-stage --shards` uses (`prepare_stage_context`,
`run_stage_local`); no `*_shards()` call is made here. Everything that can refuse does
so before the first step (spec, preset, config, every step's context), so a workflow
that cannot complete spends nothing. Once a step has run nothing halts: a failed step
is recorded and logged, later steps still run labelled with the failures they follow,
and the exit code is the account. Steps run in list order, and the order is checked first
(`workflow_order`): an input no earlier step produces, or that a later step will rewrite,
refuses the workflow with stable codes. Conditions, destinations and triggers are stored but not acted on, and the log says so.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from womblex.cloud.stage_contracts import STAGE_CONTRACTS, RunContext
from womblex.cloud.stage_runner import (
    StagePreconditionError,
    StageRunSummary,
    prepare_stage_context,
    run_stage_local,
)
from womblex.cloud.workflow_order import (
    OrderIssue,
    Present,
    StageOrderError,
    check_step_order,
    optional_order_warnings,
    present_from_names,
)
from womblex.cloud.workflow_store import StoredWorkflow, check_spec, combine_config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.config import WomblexConfig
    from womblex.ui.presets import Preset

logger = logging.getLogger(__name__)


class RefusalCode(str, Enum):
    """Stable codes for a refusal that is not an ordering issue (those carry `OrderCode`)."""

    WORKFLOW_UNAVAILABLE = "workflow_unavailable"
    """No stored workflow by that id, or the store could not be read (`womblex validate --json`)."""

    SPEC_INVALID = "spec_invalid"
    PRESET_UNKNOWN = "preset_unknown"
    CONFIG_INVALID = "config_invalid"
    STEP_ORDER = "step_order"
    """The steps cannot complete in this order; `issues` carries each `OrderCode`."""

    STEP_PRECONDITION = "step_precondition"
    """A step's preflight, Isaacus or model check failed; `steps` names each."""


class WorkflowRefusedError(Exception):
    """The workflow cannot start. Raised before any step has run, so nothing was spent.

    `issues` carries the ordering refusals, each with its stable `code`, when that was the
    cause; `steps` is `(stage, message)` for every step whose preconditions failed.
    """

    def __init__(
        self,
        message: str,
        issues: tuple[OrderIssue, ...] = (),
        *,
        code: RefusalCode = RefusalCode.CONFIG_INVALID,
        steps: tuple[tuple[str, str], ...] = (),
    ):
        super().__init__(message)
        self.issues = issues
        self.code = code
        self.steps = steps


@dataclass
class StepOutcome:
    """What one step did; `ran_after_failed` names the earlier failed steps it ran after."""

    stage: str
    status: Literal["ok", "failed"]
    summary: StageRunSummary | None = None
    error: str | None = None
    ran_after_failed: tuple[str, ...] = ()


@dataclass
class WorkflowRunResult:
    """Every step in order, and the stored parts the runner did not act on."""

    workflow_id: str
    digest: str
    steps: list[StepOutcome] = field(default_factory=list)
    not_acted_on: tuple[str, ...] = ()

    @property
    def exit_code(self) -> int:
        return 1 if any(s.status == "failed" for s in self.steps) else 0

    def log(self) -> None:
        for s in self.steps:
            line = f"  {s.stage}: {s.status}" + (f" — {s.error}" if s.error else "")
            if s.ran_after_failed:
                line += f" (ran after failed: {', '.join(s.ran_after_failed)})"
            logger.info("%s", line)
        failed = [s.stage for s in self.steps if s.status == "failed"]
        logger.info("workflow %s (%s): %d step(s), %d failed %s",
                    self.workflow_id, self.digest, len(self.steps), len(failed), failed or "")


def resolve_workflow_config(
    stored: StoredWorkflow, resolve_preset: Callable[[str], Preset | None],
) -> WomblexConfig:
    """The config a workflow runs under: its preset with its own `config` over it, validated.

    *resolve_preset* finds built-in and operator-saved presets alike. `dataset.name` is the
    preset name, so run stamps record the recipe. Raises `WorkflowRefusedError` for an
    unknown preset or a combination that does not load.
    """
    from womblex.config import WomblexConfig

    spec = stored.spec
    preset = resolve_preset(spec.preset)
    if preset is None:
        raise WorkflowRefusedError(
            f"workflow {stored.id!r}: unknown preset {spec.preset!r}", code=RefusalCode.PRESET_UNKNOWN,
        )
    overlay = combine_config(preset.config, spec.config)
    try:
        return WomblexConfig.model_validate({
            "dataset": {"name": spec.preset},
            "paths": {"input_root": ".", "output_root": ".", "checkpoint_dir": "."},
            **overlay,
        })
    except ValueError as e:
        raise WorkflowRefusedError(
            f"workflow {stored.id!r}: preset {spec.preset!r} with the workflow's config does not load: {e}",
            code=RefusalCode.CONFIG_INVALID,
        ) from e


def not_acted_on(stored: StoredWorkflow) -> tuple[str, ...]:
    spec = stored.spec
    return tuple(
        name for name, present in (
            ("conditions", spec.conditions),
            ("destinations", spec.destinations),
            ("triggers", spec.triggers),
            ("input_binding.ingest_prefix", spec.input_binding.ingest_prefix),
        ) if present
    )


@dataclass
class PreparedWorkflow:
    """What is checked before a workflow's first step, and what that built."""

    config: WomblexConfig
    stages: list[str]
    contexts: list[RunContext]
    order_warnings: list[str]


def check_workflow(
    stored: StoredWorkflow,
    *,
    resolve_preset: Callable[[str], Preset | None],
    present: Present | None,
    models_check: str | None = None,
) -> tuple[WomblexConfig, list[str]]:
    """The offline half of the pre-run path: spec, preset, merged config, step order.

    Makes no model request and builds no client. `womblex plan` stops here; the runner and
    `validate` continue into `prepare_workflow`. Raises `WorkflowRefusedError`.
    """
    try:
        check_spec(stored.spec)
    except ValueError as e:
        raise WorkflowRefusedError(
            f"workflow {stored.id!r} is no longer valid: {e}", code=RefusalCode.SPEC_INVALID,
        ) from e
    config = resolve_workflow_config(stored, resolve_preset)
    if models_check:
        config.processing.models_check = models_check  # type: ignore[assignment]
    stages = [s.stage for s in stored.spec.steps]
    if order := check_step_order(stages, config, present=present):
        err = StageOrderError(order)
        raise WorkflowRefusedError(
            f"workflow {stored.id!r}: step order cannot complete: {err}", err.issues,
            code=RefusalCode.STEP_ORDER,
        ) from err
    return config, stages


def prepare_workflow(
    stored: StoredWorkflow,
    *,
    resolve_preset: Callable[[str], Preset | None],
    present: Present | None,
    models_check: str | None = None,
) -> PreparedWorkflow:
    """The one pre-run path: `check_workflow`, then every step's context.

    Shared by `run_workflow_local` and `womblex validate`, so they refuse alike. *present*
    is the disk state (`None`: unknown, so only the order among the steps is checked).
    Raises `WorkflowRefusedError`, naming every step whose preconditions failed.
    """
    config, stages = check_workflow(
        stored, resolve_preset=resolve_preset, present=present, models_check=models_check,
    )
    contexts: list[RunContext] = []
    failures: list[tuple[str, str]] = []
    for stage in stages:
        try:
            contexts.append(prepare_stage_context(STAGE_CONTRACTS[stage], config))
        except StagePreconditionError as e:
            failures.append((stage, str(e)))
    if failures:
        raise WorkflowRefusedError(
            f"workflow {stored.id!r}: " + "; ".join(m for _, m in failures),
            code=RefusalCode.STEP_PRECONDITION, steps=tuple(failures),
        )
    return PreparedWorkflow(config, stages, contexts, optional_order_warnings(stages, config))


def run_workflow_local(
    stored: StoredWorkflow,
    shard_dir: Path,
    *,
    resolve_preset: Callable[[str], Preset | None],
    source_root: str | None = None,
    force: bool = False,
    models_check: str | None = None,
) -> WorkflowRunResult:
    """Run *stored*'s steps in list order over *shard_dir*.

    Raises `WorkflowRefusedError` before any step runs; after that, returns a
    `WorkflowRunResult` whatever the steps did.
    """
    prepared = prepare_workflow(
        stored, resolve_preset=resolve_preset, models_check=models_check,
        present=present_from_names(p.name for p in shard_dir.glob("*.parquet")),
    )
    config, contexts = prepared.config, prepared.contexts
    for warning in prepared.order_warnings:
        logger.warning("workflow %r: %s", stored.id, warning)

    result = WorkflowRunResult(stored.id, stored.digest, not_acted_on=not_acted_on(stored))
    if result.not_acted_on:
        logger.warning(
            "workflow %s: %s stored but not acted on by the local runner",
            stored.id, ", ".join(result.not_acted_on),
        )
    logger.info("workflow %s (%s): %s over %s", stored.id, stored.digest, " -> ".join(s.stage for s in stored.spec.steps), shard_dir)

    for step, ctx in zip(stored.spec.steps, contexts, strict=True):
        ctx.source_root, ctx.force = source_root, force
        earlier = tuple(s.stage for s in result.steps if s.status == "failed")
        if earlier:
            logger.warning("%s: running after failed step(s) %s", step.stage, ", ".join(earlier))
        summary: StageRunSummary | None = None
        try:
            summary = run_stage_local(STAGE_CONTRACTS[step.stage], shard_dir, config, ctx=ctx)
            error = summary.error
            status: Literal["ok", "failed"] = "failed" if summary.exit_code else "ok"
        except Exception as e:
            logger.exception("%s: failed", step.stage)
            error, status = f"{type(e).__name__}: {e}", "failed"
        result.steps.append(StepOutcome(step.stage, status, summary, error, earlier))
    return result


__all__ = [
    "PreparedWorkflow",
    "RefusalCode",
    "StepOutcome",
    "WorkflowRefusedError",
    "WorkflowRunResult",
    "check_workflow",
    "not_acted_on",
    "prepare_workflow",
    "resolve_workflow_config",
    "run_workflow_local",
]
