"""Local multi-stage runner: execute a workflow file's steps over a shard directory.

Per step it composes what `womblex run-stage --shards` uses (`prepare_stage_context`,
`run_stage_local`); no `*_shards()` call is made here. Everything that can refuse does
so before the first step (spec, preset, config, every step's context), so a workflow
that cannot complete spends nothing. Once a step has run nothing halts: a failed step
is recorded and logged, later steps still run labelled with the failures they follow,
and the exit code is the account. Steps run in list order, and the order is checked first
(`workflow_order`): an input no earlier step produces, or that a later step will rewrite,
refuses the workflow with stable codes. Triggers are not acted on here, and the log says so.
After the last step, failed or not, the run is delivered to the workflow's destinations
(`cloud/delivery.py`), each checked before the first step.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from womblex.cloud.delivery import DeliveryOutcome, check_destinations, deliver, record_deliveries
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
    from womblex.presets import Preset

logger = logging.getLogger(__name__)


class RefusalCode(str, Enum):
    """Stable codes for a refusal that is not an ordering issue (those carry `OrderCode`)."""

    WORKFLOW_UNAVAILABLE = "workflow_unavailable"
    """The workflow file could not be read (`womblex validate --json`, `womblex plan --json`)."""

    SPEC_INVALID = "spec_invalid"
    PRESET_UNKNOWN = "preset_unknown"
    CONFIG_INVALID = "config_invalid"
    STEP_ORDER = "step_order"
    """The steps cannot complete in this order; `issues` carries each `OrderCode`."""

    STEP_PRECONDITION = "step_precondition"
    """A step's preflight, Isaacus or model check failed; `steps` names each."""

    STAGE_NOT_DISPATCHABLE = "stage_not_dispatchable"
    """The fleet runs only `WORKFLOW_FLEET_STAGES`, and a firing no run-scoped stage (`quality`); `steps` names them."""

    ORDER_NOT_PIPELINE = "order_not_pipeline"
    """The fleet runs stages in pipeline order, so a list in another order is not dispatched."""

    NO_DOCUMENTS = "no_documents"
    """A firing found no supported document under its ingest prefix."""

    NO_WORKER = "no_worker"
    """No live worker serves a step's (or a firing's extraction's) work under the workflow's config."""

    RUN_OWNED = "run_owned"
    """The run a firing appends to is owned by another owner than the stored workflow's."""

    BATCHES_CANCELLED = "batches_cancelled"
    """A replayed firing's own batches were cancelled; re-queueing would attach to them and never run."""

    DESTINATION_UNAVAILABLE = "destination_unavailable"
    """A destination's secrets do not resolve or its dependencies are not installed; the message names each."""


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
    deliveries: list[DeliveryOutcome] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        failed = any(s.status == "failed" for s in self.steps) or any(d.status != "delivered" for d in self.deliveries)
        return 1 if failed else 0

    def log(self) -> None:
        for s in self.steps:
            line = f"  {s.stage}: {s.status}" + (f" — {s.error}" if s.error else "")
            if s.ran_after_failed:
                line += f" (ran after failed: {', '.join(s.ran_after_failed)})"
            logger.info("%s", line)
        for d in self.deliveries:
            logger.info("  deliver %s: %s%s", d.destination, d.status, f" — {d.error}" if d.error else "")
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


def config_payload(config: WomblexConfig) -> dict[str, Any]:
    """What a fleet run carries of *config* (Phase R, D7): everything but `paths`, which are each worker's."""
    return config.model_dump(mode="json", exclude={"paths"})


def config_from_payload(payload: dict[str, Any], paths: dict[str, Any]) -> WomblexConfig:
    """The config a fleet run was dispatched with, on the worker that runs it, with that worker's *paths*."""
    from womblex.config import WomblexConfig

    return WomblexConfig.model_validate({**payload, "paths": paths})


def preset_resolver(presets_dir: Path | None) -> Callable[[str], Preset | None]:
    """Resolve a preset name: an operator-saved one in *presets_dir* shadows a built-in."""
    from womblex import presets as pm

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
            ("input_binding.store", spec.input_binding.store),
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
    shard_dir: Path | None = None,
) -> PreparedWorkflow:
    """The one pre-run path: `check_workflow`, then every step's context.

    Shared by `run_workflow_local` and `womblex validate`, so they refuse alike. *present*
    is the disk state (`None`: unknown, so only the order among the steps is checked), and
    *shard_dir* the directory the steps run over, where known.
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
    if problems := check_destinations(stored.spec.destinations):
        raise WorkflowRefusedError(
            f"workflow {stored.id!r}: " + "; ".join(f"destination {d}: {m}" for d, m in problems),
            code=RefusalCode.DESTINATION_UNAVAILABLE,
        )
    if stored.spec.destinations and shard_dir is not None and shard_dir.name != "documents":
        raise WorkflowRefusedError(
            f"workflow {stored.id!r} delivers a run, so it runs over a run's documents/ directory, not {shard_dir}",
            code=RefusalCode.DESTINATION_UNAVAILABLE,
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
    `WorkflowRunResult` whatever the steps did. A workflow with destinations delivers the run
    whose ``documents/`` *shard_dir* is once the steps are done.
    """
    prepared = prepare_workflow(
        stored, resolve_preset=resolve_preset, models_check=models_check,
        present=present_from_names(p.name for p in shard_dir.glob("*.parquet")), shard_dir=shard_dir,
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

    from womblex.cloud.webhook import RunEvents

    run_id = local_run_id(shard_dir)
    events = RunEvents(stored, run_id, f"{run_id}:local-{uuid.uuid4().hex[:12]}")
    events.emit("extraction_settled", "running")  # a local run's extraction is done before its steps
    with workflow_context(stored.digest, stored.id, stored.spec.model_dump(mode="json")):
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
            events.emit("stage_failed" if status == "failed" else "stage_finished", "running", step.stage,
                        stage=step.stage, error=error, ran_after_failed=list(earlier))
    if stored.spec.destinations:
        result.deliveries = deliver_local(stored, shard_dir, tuple(s.stage for s in result.steps if s.status == "failed"))
    for o in result.deliveries:
        events.emit("delivered" if o.status == "delivered" else "delivery_failed", "running", o.destination,
                    destination=o.destination, error=o.error, after_failed=o.after_failed)
    # The run's status as the board words it: its steps', not its deliveries' (those have their own events).
    events.emit("run_finished", "failed" if any(s.status == "failed" for s in result.steps) else "done")
    if events.active:
        result.deliveries += _record_local(shard_dir, events.records())
    return result


def local_run_id(shard_dir: Path) -> str:
    """The run a local shard directory holds: its stamp's run id, else its run root's name."""
    from womblex.store.run_stamp import stamp_from_footers

    stamp = stamp_from_footers(sorted(shard_dir.glob("*._manifest.parquet")), "egress")
    return stamp.run_id if stamp else shard_dir.parent.name


def _record_local(shard_dir: Path, outcomes: list[DeliveryOutcome]) -> list[DeliveryOutcome]:
    """Record *outcomes* on the local run and rewrite its manifest so the run record lists them."""
    from womblex.store.remote import RemoteStore
    from womblex.store.run_manifest import write_run_manifest

    try:
        record_deliveries(RemoteStore.from_uri(str(shard_dir.parent)), outcomes)
        write_run_manifest(shard_dir)
    except Exception:  # the run has finished; a record that could not be written is logged
        logger.exception("delivery records could not be written under %s", shard_dir.parent)
    return outcomes


def deliver_local(stored: StoredWorkflow, shard_dir: Path, failed: tuple[str, ...] = ()) -> list[DeliveryOutcome]:
    """Deliver the local run whose ``documents/`` is *shard_dir*: the manifest first, so the bundle
    holds a current one, then each destination, its record, and the manifest again, so the run
    record lists the deliveries. *failed* names the failed steps it follows. Never raises."""
    from womblex.store.run_manifest import write_run_manifest

    try:
        write_run_manifest(shard_dir)
        outcomes = deliver(stored, shard_dir.parent, local_run_id(shard_dir), after_failed=failed)
        _record_local(shard_dir, outcomes)
    except Exception:  # the steps have run; nothing here may stop or unwind them
        logger.exception("workflow %s: delivery could not complete", stored.id)
        return [DeliveryOutcome(d.name, d.kind, str(getattr(d, "uri", None) or getattr(d, "table", "")), stored.id,
                                "failed", 0, list(failed), error="delivery could not complete; see the log")
                for d in stored.spec.destinations]
    return outcomes


__all__ = [
    "PreparedWorkflow",
    "RefusalCode",
    "StepOutcome",
    "WorkflowRefusedError",
    "WorkflowRunResult",
    "check_workflow",
    "config_from_overlay",
    "config_from_payload",
    "config_payload",
    "deliver_local",
    "local_run_id",
    "not_acted_on",
    "prepare_workflow",
    "preset_resolver",
    "resolve_workflow_config",
    "run_workflow_local",
    "workflow_overlay",
]
