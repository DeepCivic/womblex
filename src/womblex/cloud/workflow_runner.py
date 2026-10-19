"""Local multi-stage runner: execute a stored workflow's steps over a shard directory.

Per step it composes what `womblex run-stage --shards` uses (`prepare_stage_context`,
`run_stage_local`); no `*_shards()` call is made here. Everything that can refuse does
so before the first step (spec, preset, config, every step's context), so a workflow
that cannot complete spends nothing. Once a step has run nothing halts: a failed step
is recorded and logged, later steps still run labelled with the failures they follow,
and the exit code is the account. Steps run in list order, and the order is checked first
(`workflow_order`): an input no earlier step produces, or that a later step will rewrite,
refuses the workflow with stable codes. Triggers are not acted on here, and the log says so.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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
from womblex.store.run_stamp import workflow_context

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

    STAGE_NOT_DISPATCHABLE = "stage_not_dispatchable"
    """The fleet runs only `DOWNSTREAM_STAGES`; `steps` names the others (`layout`, `pii`, `quality`)."""

    ORDER_NOT_PIPELINE = "order_not_pipeline"
    """The fleet runs stages in pipeline order, so a list in another order is not dispatched."""

    NO_DOCUMENTS = "no_documents"
    """A firing found no supported document under its ingest prefix."""

    RUN_CONFLICT = "run_conflict"
    """The run already dispatched a stage under a different workflow digest."""

    BATCHES_CANCELLED = "batches_cancelled"
    """A firing's run already holds cancelled batches, which a re-enqueue would attach to and never run."""


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


def config_from_overlay(
    preset_name: str, overlay: dict[str, Any], paths: dict[str, Any] | None = None,
) -> WomblexConfig:
    """The config a workflow's resolved *overlay* stands for: `dataset.name` is the preset name.

    The one construction the local runner and a fleet worker share, so a stage runs under
    byte-identical settings in both. *paths* is the worker's own; a local run has none.
    """
    from womblex.config import WomblexConfig

    return WomblexConfig.model_validate({
        **overlay,  # first, so run identity cannot come from the overlay
        "dataset": {"name": preset_name},
        "paths": paths or {"input_root": ".", "output_root": ".", "checkpoint_dir": "."},
    })


def preset_resolver(presets_dir: Path | None) -> Callable[[str], Preset | None]:
    """Resolve a preset name: an operator-saved one in *presets_dir* shadows a built-in.

    Imports `ui.presets`, not `ui.readers` (which needs fastapi, an extra).
    """
    from womblex.ui import presets as pm

    def resolve(name: str) -> Preset | None:
        path = presets_dir / pm.preset_filename(name) if presets_dir and pm.is_safe_preset_name(name) else None
        saved = pm.parse_saved_preset(name, path.read_text(encoding="utf-8")) if path and path.is_file() else None
        return saved or pm.PRESETS.get(name)

    return resolve


def workflow_overlay(
    stored: StoredWorkflow, resolve_preset: Callable[[str], Preset | None],
) -> dict[str, Any]:
    """The resolved recipe: the preset's config with the workflow's own `config` over it.

    Raises `WorkflowRefusedError` for an unknown preset.
    """
    preset = resolve_preset(stored.spec.preset)
    if preset is None:
        raise WorkflowRefusedError(
            f"workflow {stored.id!r}: unknown preset {stored.spec.preset!r}",
            code=RefusalCode.PRESET_UNKNOWN,
        )
    return combine_config(preset.config, stored.spec.config)


def resolve_workflow_config(
    stored: StoredWorkflow, resolve_preset: Callable[[str], Preset | None],
) -> WomblexConfig:
    """The config a workflow runs under: its preset with its own `config` over it, validated.

    *resolve_preset* finds built-in and operator-saved presets alike. `dataset.name` is the
    preset name, so run stamps record the recipe. Raises `WorkflowRefusedError` for an
    unknown preset or a combination that does not load.
    """
    spec = stored.spec
    try:
        return config_from_overlay(spec.preset, workflow_overlay(stored, resolve_preset))
    except ValueError as e:
        raise WorkflowRefusedError(
            f"workflow {stored.id!r}: preset {spec.preset!r} with the workflow's config does not load: {e}",
            code=RefusalCode.CONFIG_INVALID,
        ) from e


def not_acted_on(stored: StoredWorkflow) -> tuple[str, ...]:
    spec = stored.spec
    return tuple(
        name for name, present in (
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

    with workflow_context(stored.digest):
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
    "config_from_overlay",
    "not_acted_on",
    "prepare_workflow",
    "preset_resolver",
    "resolve_workflow_config",
    "run_workflow_local",
    "workflow_overlay",
]
