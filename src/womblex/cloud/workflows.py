"""The DBOS workflows a worker runs: an extraction batch, a stage, and the coordinator.

An extraction batch stages its inputs from the ingest store, runs the shared
``womblex.batch.process_batch`` body (identical to ``womblex run``) and
publishes the shards. A stage plans its units (one per batch base, or one for
the whole run), then runs each as its own step, so a unit that finished is
recorded and a re-run starts after it. The coordinator waits for the run's
extraction to settle, then runs the requested stages in pipeline order.

A launched run is never stopped part-way (`docs/decisions-operations.md`, "A run is an
arrow"): a unit that fails does not stop the stage's other units, and a stage
that fails does not stop the stages after it. Each is recorded where it
happened and named again at the end, so the board shows it failed once
everything that could run has.

Step results are storage keys, never data. The worker's config and stores are
process state (:func:`set_context`), not workflow arguments: arguments are
stored with the workflow, so they stay plain JSON-shaped values.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dbos import DBOS, SetWorkflowAttributes, SetWorkflowID, StepOptions
from dbos import error as dbos_error

from womblex.batch import process_batch
from womblex.cloud import dbos_app
from womblex.cloud.stage_contracts import PRODUCER_OF, STAGE_CONTRACTS
from womblex.cloud.stage_runner import (
    InputContractError,
    NotReady,
    plan_units,
    prepare_stage_context,
    run_unit,
)
from womblex.store.run_stamp import RunStamp, workflow_context
from womblex.store.source_provenance import IngestProvenance
from womblex.utils.log_format import log_context
from womblex.utils.run_log import capture_batch_log

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.config import WomblexConfig
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)

#: Seconds between looks at whether a run's extraction has settled.
SETTLE_POLL_SECONDS = 30.0
_UNSETTLED = ["ENQUEUED", "DELAYED", "PENDING"]


class StageNotReady(Exception):
    """An upstream sidecar the stage reads is absent: the stage was run out of order."""


class StageIncomplete(Exception):
    """Every unit of a stage ran and some failed; raised last, so the stage reads as failed."""


class RunIncomplete(Exception):
    """Every stage of a run ran and some failed; raised last, so the coordinator reads as failed."""


#: What a unit or a stage can fail with and the run still carry on: the work's own error,
#: surfaced by DBOS once retries are spent or a child workflow ends. Any other DBOS error
#: (nondeterminism, recovery) is the engine's and propagates.
_CARRIED: tuple[type[BaseException], ...] = (
    dbos_error.DBOSMaxStepRetriesExceeded,
    dbos_error.DBOSAwaitedWorkflowCancelledError,
    dbos_error.DBOSAwaitedWorkflowMaxRecoveryAttemptsExceeded,
)


def _carried(e: Exception) -> str:
    """*e* as one line for the record, or re-raise it when it is the engine's own."""
    if isinstance(e, dbos_error.DBOSException) and not isinstance(e, _CARRIED):
        raise e
    if isinstance(e, dbos_error.DBOSMaxStepRetriesExceeded) and e.errors:
        e = e.errors[-1]
    return f"{type(e).__name__}: {e}"


@dataclass
class WorkerContext:
    """What this worker process runs with; set once before DBOS launches."""

    config: WomblexConfig
    store: RemoteStore
    ingest: RemoteStore
    ingest_root: str
    dsn: str | None = None  # the system database a firing reads stored workflows from


_ctx: WorkerContext | None = None


def set_context(ctx: WorkerContext | None) -> None:
    global _ctx
    _ctx = ctx


def _context() -> WorkerContext:
    if _ctx is None:
        raise RuntimeError("no worker context: call set_context() before launching DBOS")
    return _ctx


def _not_a_failure_to_retry(e: BaseException) -> bool:
    """A missing input is a wiring fact, not a transient fault: retrying cannot fix it."""
    return not isinstance(e, NotReady | StageNotReady | InputContractError)


def _retry(name: str, max_attempts: int) -> StepOptions:
    return StepOptions(
        name=name, retries_allowed=True, max_attempts=max_attempts,
        interval_seconds=5.0, backoff_rate=2.0, should_retry=_not_a_failure_to_retry,
    )


def _output_prefix(shard_prefix: str) -> str:
    """The run's output dir — ``shard_prefix`` (``…/documents``) less its leaf."""
    return shard_prefix.rsplit("/", 1)[0]


def _logged[T](
    name: str, run_id: str, shard_prefix: str, stage: str | None,
    body: Callable[[Path], T],
) -> T:
    """Run *body* with its log captured, and publish the log on success or failure.

    A failed job still leaves its ``logs/<name>.log`` in the store — that is
    the case the operator most needs it. A failed *upload* of the log never
    masks the body's own error.
    """
    store = _context().store
    job_id = DBOS.workflow_id
    with tempfile.TemporaryDirectory(prefix="womblex-job-") as tmp:
        root = Path(tmp)
        log_path = root / f"{name}.log"
        try:
            with log_context(run_id=run_id, job_id=job_id, stage=stage), \
                    capture_batch_log(log_path):
                return body(root)
        finally:
            try:
                store.upload_file(log_path, f"{_output_prefix(shard_prefix)}/logs/{name}.log")
            except Exception:  # publishing the log must never mask the job's own error
                logger.exception("[%s] failed to publish job log", name)


# --- extraction --------------------------------------------------------------


def _extract(
    run_id: str, batch_num: int, input_keys: list[str], shard_prefix: str, ingest_root: str,
) -> dict[str, str]:
    """Stage this batch's inputs, extract them, publish the shards; return key -> SHA-256."""
    ctx = _context()

    def body(root: Path) -> dict[str, str]:
        shards_dir = root / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        # Nested: a job's keys come from a recursive listing, so two documents
        # under different prefixes routinely share a basename.
        files = ctx.ingest.download_to_dir(input_keys, root / "inputs", nested=True)
        declared = ingest_root or ctx.ingest_root
        provenance = (
            IngestProvenance.declare(
                declared, ctx.config.dataset.name,
                relpaths=dict(zip(files, input_keys, strict=True)),
            )
            if declared else None
        )
        stamp = RunStamp.declare(run_id, ctx.config, stage="extract") if run_id.strip() else None
        outcome = process_batch(
            files, ctx.config, batch_num=batch_num, shard_dir=shards_dir,
            provenance=provenance, stamp=stamp,
        )
        # Glob off the shard path the batch reported, so the naming scheme lives
        # only in womblex.batch.
        published = ctx.store.upload_glob(shards_dir, f"{outcome.shard_path.stem}.*", shard_prefix)
        logger.info(
            "[batch %d] %d ok, %d failed -> %s",
            batch_num, outcome.batch.succeeded, outcome.batch.failed, shard_prefix,
        )
        return published

    return _logged(f"batch-{batch_num:04d}", run_id, shard_prefix, None, body)


@DBOS.workflow(name=dbos_app.EXTRACT_WORKFLOW)
def extract_batch(
    run_id: str, batch_num: int, input_keys: list[str], shard_prefix: str,
    ingest_root: str, max_attempts: int,
) -> dict[str, Any]:
    recorded = DBOS.run_step(
        _retry("extract", max_attempts), _extract,
        run_id, batch_num, input_keys, shard_prefix, ingest_root,
    )
    return {"published": len(recorded), "checksums": _digests(recorded)}


def _digests(recorded: Any) -> dict[str, str]:
    """A step's published keys with digests; one recorded before checksums
    returned a key list, so its files carry none (the board names that gap)."""
    return recorded if isinstance(recorded, dict) else {}


# --- stages ------------------------------------------------------------------


def _stage_config(workflow: dict[str, Any] | None) -> WomblexConfig:
    """The config a stage runs under: the worker's own, or a dispatched workflow's recipe.

    A workflow's overlay is built exactly as the local runner builds it
    (`config_from_overlay`), keeping this worker's `paths`.
    """
    ctx = _context()
    if workflow is None:
        return ctx.config
    from womblex.cloud.workflow_runner import config_from_overlay

    return config_from_overlay(
        workflow["preset"], workflow["config"], ctx.config.paths.model_dump(mode="json"),
    )


def _plan(
    stage: str, shard_prefix: str, workflow: dict[str, Any] | None = None, bases: list[str] | None = None,
) -> list[list[str]]:
    ctx = _context()
    return plan_units(STAGE_CONTRACTS[stage], ctx.store, shard_prefix, _stage_config(workflow), bases=bases)


def _unit(
    stage: str, run_id: str, shard_prefix: str, unit: list[str],
    workflow: dict[str, Any] | None = None,
) -> dict[str, str]:
    ctx = _context()
    config = _stage_config(workflow)
    contract = STAGE_CONTRACTS[stage]
    label = unit[0] if len(unit) == 1 else f"{len(unit)}-bases"

    def body(_root: Path) -> dict[str, str]:
        # Built per unit: the client holds a connection, and a recovered
        # workflow resumes in a process that has none.
        run_ctx = prepare_stage_context(contract, config)
        try:
            with workflow_context(workflow["digest"]) if workflow else nullcontext():
                return run_unit(contract, config, run_ctx, ctx.store, shard_prefix, unit, ctx.ingest)
        except NotReady as nr:
            producer = PRODUCER_OF.get(nr.suffix, "the upstream stage")
            raise StageNotReady(
                f"{stage}: {nr.stem} is missing {nr.suffix} — {producer} has not produced it"
            ) from nr

    return _logged(f"stage-{stage}-{label}", run_id, shard_prefix, stage, body)


@DBOS.workflow(name=dbos_app.STAGE_WORKFLOW)
def run_stage(
    run_id: str, stage: str, shard_prefix: str, max_attempts: int,
    workflow: dict[str, Any] | None = None, bases: list[str] | None = None,
) -> dict[str, Any]:
    units = DBOS.run_step({"name": "plan"}, _plan, stage, shard_prefix, workflow, bases)
    published = 0
    checksums: dict[str, str] = {}
    failed: list[str] = []
    for unit in units:
        label = unit[0] if len(unit) == 1 else "run"
        try:
            recorded = DBOS.run_step(
                _retry(f"unit:{label}", max_attempts), _unit, stage, run_id, shard_prefix, unit,
                workflow,
            )
        except Exception as e:  # one bad unit must not stop the others
            failed.append(f"{label} ({_carried(e)})")
            logger.warning("%s: %s failed; the stage's other units still run: %s", stage, label, failed[-1])
            continue
        published += len(recorded)
        checksums.update(_digests(recorded))
    if failed:
        raise StageIncomplete(
            f"{stage}: {len(failed)} of {len(units)} unit(s) failed; every unit ran — {'; '.join(failed)}"
        )
    return {"stage": stage, "units": len(units), "published": published, "checksums": checksums}


# --- scheduled firings ---------------------------------------------------------


def firing_label(scheduled_time: datetime) -> str:
    """The UTC firing time: two triggers firing at the same instant share a firing."""
    return f"{scheduled_time.astimezone(UTC):%Y%m%dT%H%M%SZ}"


def _fire(run_id: str, context: dict[str, Any], firing: str) -> dict[str, Any]:
    """Load the stored workflow and fire it (`workflow_dispatch.fire_workflow`)."""
    from womblex.cloud.workflow_dispatch import fire_workflow
    from womblex.cloud.workflow_runner import RefusalCode, WorkflowRefusedError, preset_resolver
    from womblex.cloud.workflow_store import WorkflowStore

    ctx = _context()
    if not context.get("store"):  # scheduled before workflows named their input store
        raise WorkflowRefusedError(
            f"[{RefusalCode.SPEC_INVALID.value}] workflow {context['workflow_id']!r} was scheduled before "
            "workflows named their input store; add input_binding.store to its file and run "
            "`womblex schedule-workflow` on it again", code=RefusalCode.SPEC_INVALID)
    if (context.get("store") or "").rstrip("/") != ctx.ingest_root.rstrip("/"):  # routing keeps this from happening
        raise RuntimeError(f"firing of {context['workflow_id']!r} reached a worker on {ctx.ingest_root}, "
                           f"not the workflow's input store {context.get('store')}")
    with WorkflowStore(ctx.dsn) as wf_store:
        stored = wf_store.get(context["workflow_id"])
    if stored is None:
        raise RuntimeError(f"scheduled workflow {context['workflow_id']!r} is no longer stored")
    presets = os.environ.get("WOMBLEX_UI_PRESETS_DIR")
    try:
        sent = fire_workflow(
            stored, run_id=run_id, firing=firing, resolve_preset=preset_resolver(Path(presets) if presets else None),
            store=ctx.store, ingest=ctx.ingest, dsn=ctx.dsn,
        )
    except WorkflowRefusedError as e:  # the board shows the message, so it carries the code
        raise WorkflowRefusedError(f"[{e.code.value}] {e}", e.issues, code=e.code, steps=e.steps) from e
    return {"run_id": run_id, "firing": firing, "stages": list(sent.stages), "batches": sent.bases,
            "unchanged": sent.unchanged}  # read back by `RunBoard.seen_inputs`


@DBOS.workflow(name=dbos_app.TRIGGER_WORKFLOW)
def fire_trigger(run_id: str, context: dict[str, Any], firing: str) -> dict[str, Any]:
    return DBOS.run_step({"name": "fire"}, _fire, run_id, context, firing)


@DBOS.workflow(name=dbos_app.SCHEDULED_WORKFLOW)
def run_scheduled(scheduled_time: datetime, context: dict[str, Any]) -> dict[str, str]:
    """Start the firing as `<run>:trigger:<firing>`, so a firing that fails is on the board under its
    run, not only as DBOS's unattributed `sched-…` row. A workflow's firings append to one run,
    named for the workflow. A workflow has no owner, so the run is admin-only, like a CLI run."""
    run_id, firing = context["workflow_id"], firing_label(scheduled_time)
    attrs: dict[str, Any] = {"run_id": run_id, "kind": dbos_app.KIND_TRIGGER,
                             "workflow_id": context["workflow_id"], "firing": firing}
    with SetWorkflowID(dbos_app.trigger_id(run_id, firing)), SetWorkflowAttributes(attrs):
        # A schedule synced before workflows named a store has none: no worker serves an
        # unkeyed firing queue, so it goes where every worker serves, to be refused visibly.
        store = context.get("store")
        queue = dbos_app.firing_queue(store) if store else dbos_app.DOWNSTREAM_QUEUE
        DBOS.enqueue_workflow(queue, fire_trigger, run_id, context, firing)
    return {"run_id": run_id, "firing": dbos_app.trigger_id(run_id, firing)}


# --- coordinator -------------------------------------------------------------


def _unsettled_batches(run_id: str) -> int:
    return len(DBOS.list_workflows(
        workflow_id_prefix=f"{run_id}:batch:", status=_UNSETTLED,
        load_input=False, load_output=False,
    ))


@DBOS.workflow(name=dbos_app.DOWNSTREAM_WORKFLOW)
def run_downstream(
    run_id: str, stages: list[str], shard_prefix: str, max_attempts: int, owner: str | None,
    workflow: dict[str, Any] | None = None, firing: str | None = None, bases: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run *stages* over the run once its extraction has settled, in the order given.

    A scheduled *firing* runs them over its own batch *bases* only, as stages of its own.

    A failed batch does not hold the run: its absence surfaces as the
    not-ready or missing-input error of the stage that needed it, for that
    batch only. A failed stage does not stop the stages after it: each runs
    over what exists, recorded with the failures it follows (`ran_after_failed`
    on its row, as the local runner records), and the coordinator fails last
    if any did, naming them.
    """
    while DBOS.run_step({"name": "await-extraction"}, _unsettled_batches, run_id):
        DBOS.sleep(SETTLE_POLL_SECONDS)

    results: list[dict[str, Any]] = []
    failed: list[str] = []
    for stage in stages:
        attrs: dict[str, Any] = {"run_id": run_id, "kind": dbos_app.KIND_STAGE, "stage": stage,
                                 "max_attempts": max_attempts}
        if owner is not None:
            attrs["owner"] = owner
        if workflow is not None:
            attrs["workflow_digest"] = workflow["digest"]
        if firing is not None:
            attrs["firing"] = firing
        if failed:
            attrs["ran_after_failed"] = list(failed)
        # Enqueued by name, not registered: the workers that serve the stage
        # own the queue's settings.
        scope = dbos_app.stage_scope(firing, workflow["digest"] if workflow else None)
        with SetWorkflowID(dbos_app.stage_id(run_id, stage, scope)), SetWorkflowAttributes(attrs):
            handle = DBOS.enqueue_workflow(
                dbos_app.stage_queue(stage), run_stage, run_id, stage, shard_prefix, max_attempts,
                workflow, bases,
            )
        try:
            results.append({**handle.get_result(), "ran_after_failed": list(failed)})
        except Exception as e:  # a failed stage must not stop the ones after it
            results.append({"stage": stage, "error": _carried(e), "ran_after_failed": list(failed)})
            logger.warning("run %s: %s failed; later stages still run: %s", run_id, stage, results[-1]["error"])
            failed.append(stage)
    if failed:
        after = [f"{r['stage']} after {', '.join(r['ran_after_failed'])}"
                 for r in results if r["ran_after_failed"]]
        raise RunIncomplete(
            f"run {run_id}: stage(s) {', '.join(failed)} failed; every stage ran"
            + (f"; ran after a failure: {'; '.join(after)}" if after else "")
        )
    return results
