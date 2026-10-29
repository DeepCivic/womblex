"""The DBOS workflows a worker runs: an extraction batch, a stage, and the coordinator.

An extraction batch stages its inputs from the ingest store, runs the shared
``womblex.batch.process_batch`` body (identical to ``womblex run``) and
publishes the shards. A stage plans its units (one per batch base, or one for
the whole run), then runs each as its own step, so a unit that finished is
recorded and a re-run starts after it. The coordinator waits for the run's
extraction to settle, runs the requested stages in pipeline order, then writes
the run's ``manifest.parquet`` through the code `womblex finalize` runs and, for a
workflow that names destinations, delivers the run to them (`cloud/delivery.py`).

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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dbos import DBOS, SetWorkflowAttributes, SetWorkflowID, StepOptions
from dbos import error as dbos_error

from womblex.batch import process_batch
from womblex.cloud import dbos_app, run_events
from womblex.cloud.finalize import NoShardManifests, finalize_from_board
from womblex.cloud.stage_contracts import PRODUCER_OF, STAGE_CONTRACTS
from womblex.cloud.stage_runner import (
    InputContractError,
    NotReady,
    plan_units,
    prepare_stage_context,
    run_unit,
)
from womblex.cloud.unit_receipt import already_published
from womblex.cloud.workflow_schedules import trigger_attributes
from womblex.store.run_stamp import RunStamp, workflow_context
from womblex.store.source_provenance import IngestProvenance
from womblex.utils.log_format import log_context
from womblex.utils.model_check import models_key
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
    return not isinstance(e, NotReady | StageNotReady | InputContractError | NoShardManifests)


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


def _provenance_collection(run_config: WomblexConfig, workflow: dict[str, Any] | None) -> str:
    """The collection a workflow's documents belong to (its `collection`, else its id); the preset name otherwise."""
    if workflow and workflow.get("id"):
        return str((workflow.get("spec") or {}).get("collection") or workflow["id"])
    return run_config.dataset.name


def _extract(
    run_id: str, batch_num: int, input_keys: list[str], shard_prefix: str, ingest_root: str,
    config: dict[str, Any] | None = None, workflow: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Stage this batch's inputs, extract them under the run's config, publish the shards; return key -> SHA-256.

    A scheduled firing's *workflow* is named in the shards' footers, as a stage step names it in its sidecars.
    """
    ctx = _context()
    run_config = _stage_config(None, config)

    def body(root: Path) -> dict[str, str]:
        shards_dir = root / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        # Nested: a job's keys come from a recursive listing, so two documents
        # under different prefixes routinely share a basename.
        files = ctx.ingest.download_to_dir(input_keys, root / "inputs", nested=True)
        declared = ingest_root or ctx.ingest_root
        provenance = (
            IngestProvenance.declare(
                declared, _provenance_collection(run_config, workflow),
                relpaths=dict(zip(files, input_keys, strict=True)),
            )
            if declared else None
        )
        stamp = RunStamp.declare(run_id, run_config, stage="extract") if run_id.strip() else None
        with (workflow_context(workflow["digest"], workflow.get("id"), workflow.get("spec"))
              if workflow else nullcontext()):
            outcome = process_batch(
                files, run_config, batch_num=batch_num, shard_dir=shards_dir,
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
    ingest_root: str, max_attempts: int, config: dict[str, Any] | None = None,
    workflow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recorded = DBOS.run_step(
        _retry("extract", max_attempts), _extract,
        run_id, batch_num, input_keys, shard_prefix, ingest_root, config, workflow,
    )
    return {"published": len(recorded), "checksums": _digests(recorded)}


def _digests(recorded: Any) -> dict[str, str]:
    """A step's published keys with digests; one recorded before checksums
    returned a key list, so its files carry none (the board names that gap)."""
    return recorded if isinstance(recorded, dict) else {}


# --- stages ------------------------------------------------------------------


def _stage_config(workflow: dict[str, Any] | None, config: dict[str, Any] | None = None) -> WomblexConfig:
    """The config a run's work runs under: a dispatched workflow's recipe, or the config the run
    was dispatched with (Phase R, D7), each with this worker's `paths`.

    A workflow's overlay is built exactly as the local runner builds it (`config_from_overlay`).
    A row enqueued before runs carried their config has neither, and runs under the worker's own.
    """
    from womblex.cloud.workflow_runner import config_from_overlay, config_from_payload

    ctx = _context()
    paths = ctx.config.paths.model_dump(mode="json")
    if workflow is not None:
        return config_from_overlay(workflow["preset"], workflow["config"], paths)
    return config_from_payload(config, paths) if config is not None else ctx.config


def _plan(
    stage: str, shard_prefix: str, workflow: dict[str, Any] | None = None, bases: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> list[list[str]]:
    ctx = _context()
    return plan_units(STAGE_CONTRACTS[stage], ctx.store, shard_prefix, _stage_config(workflow, config), bases=bases)


def _unit(
    stage: str, run_id: str, shard_prefix: str, unit: list[str],
    workflow: dict[str, Any] | None = None, run_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    ctx = _context()
    config = _stage_config(workflow, run_config)
    contract = STAGE_CONTRACTS[stage]
    label = unit[0] if len(unit) == 1 else f"{len(unit)}-bases"

    def body(_root: Path) -> dict[str, str]:
        # Built per unit: the client holds a connection, and a recovered
        # workflow resumes in a process that has none.
        # A retry (`<stage id>:retry:N`, a fork or a later stage's re-run) runs every unit it holds.
        wf_id = DBOS.workflow_id or ""
        first = (dbos_app.stage_attempt(wf_id) or ("", 0))[1] == 0
        if first and already_published(contract, config, ctx.store, shard_prefix, unit, by=wf_id):
            logger.info("%s: %s already published under this config and these inputs; skipped", stage, label)
            return {}  # a unit that runs publishes at least one key, so empty reads as skipped
        run_ctx = prepare_stage_context(contract, config)
        try:
            with (workflow_context(workflow["digest"], workflow.get("id"), workflow.get("spec"))
                  if workflow else nullcontext()):
                return run_unit(contract, config, run_ctx, ctx.store, shard_prefix, unit, ctx.ingest, by=wf_id)
        except NotReady as nr:
            producer = PRODUCER_OF.get(nr.suffix, "the upstream stage")
            raise StageNotReady(
                f"{stage}: {nr.stem} is missing {nr.suffix} — {producer} has not produced it"
            ) from nr

    return _logged(f"stage-{stage}-{label}", run_id, shard_prefix, stage, body)


@DBOS.workflow(name=dbos_app.STAGE_WORKFLOW)
def run_stage(
    run_id: str, stage: str, shard_prefix: str, max_attempts: int,
    workflow: dict[str, Any] | None = None, bases: list[str] | None = None, config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    units = DBOS.run_step({"name": "plan"}, _plan, stage, shard_prefix, workflow, bases, config)
    published = skipped = done = 0
    checksums: dict[str, str] = {}
    failed: list[str] = []
    events: list[dict[str, Any]] = []  # the `batch_progress` events sent, for the coordinator's record
    for unit in units:
        label = unit[0] if len(unit) == 1 else "run"
        try:
            recorded = DBOS.run_step(
                _retry(f"unit:{label}", max_attempts), _unit, stage, run_id, shard_prefix, unit,
                workflow, config,
            )
        except Exception as e:  # one bad unit must not stop the others
            failed.append(f"{label} ({_carried(e)})")
            logger.warning("%s: %s failed; the stage's other units still run: %s", stage, label, failed[-1])
        else:
            published += len(recorded)
            skipped += not recorded
            checksums.update(_digests(recorded))
            done += 1
        events += run_events.record(
            run_id, workflow, "batch_progress", f"batch_progress:{stage}:{done + len(failed)}",
            stage=stage, done=done, failed=len(failed), total=len(units))
    if failed:
        raise StageIncomplete(
            f"{stage}: {len(failed)} of {len(units)} unit(s) failed; every unit ran — {'; '.join(failed)}"
        )
    return {"stage": stage, "units": len(units), "published": published, "skipped": skipped, "checksums": checksums,
            **({"events": events} if events else {})}


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
    named for the workflow's collection, else for the workflow (a schedule synced before the
    context carried `run_id` falls back to the workflow, which was its run id). The firing runs
    under the stored workflow's owner (`fire_workflow`); a CLI-stored workflow's run is admin-only."""
    run_id, firing = context.get("run_id") or context["workflow_id"], firing_label(scheduled_time)
    attrs = trigger_attributes(context, run_id, firing)
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


def _finalize(run_id: str, shard_prefix: str) -> str:
    """Write the run's ``manifest.parquet`` over every batch present, as `womblex finalize` does."""
    ctx = _context()
    written = finalize_from_board(ctx.store, run_id, _output_prefix(shard_prefix), ctx.dsn)
    logger.info("run %s: wrote %s from %d shard manifest(s)", run_id, written.manifest_key, written.shard_manifests)
    return written.manifest_key


def _deliver(run_id: str, shard_prefix: str, workflow: dict[str, Any], failed: list[str]) -> list[dict[str, Any]]:
    """Deliver the run to its workflow's destinations; each outcome, as its record holds it. Never raises."""
    from womblex.cloud.delivery import deliver, dispatched_workflow

    run = _context().store.sub(_output_prefix(shard_prefix))
    return [asdict(o) for o in deliver(dispatched_workflow(workflow), run, run_id, after_failed=tuple(failed))]


def _run_status(run_id: str) -> str:
    """The run's status in the API's words, this coordinator, still running, left out."""
    from womblex.cloud.jobs import RunBoard, run_state

    with RunBoard(_context().dsn) as board:
        counts = board.stats(run_id)
    counts["running"] = max(counts.get("running", 0) - 1, 0)
    return run_state(counts)


def _record(run_id: str, shard_prefix: str, workflow: dict[str, Any], delivered: list[dict[str, Any]],
            sent: list[dict[str, Any]]) -> None:
    """Record each delivery and each webhook's events, then rewrite the manifest so the run record
    lists them. Logs, never raises: the deliveries have happened."""
    from womblex.cloud.delivery import DeliveryOutcome, dispatched_workflow, record_deliveries
    from womblex.cloud.webhook import webhook_records

    run = _context().store.sub(_output_prefix(shard_prefix))
    outcomes = [DeliveryOutcome(**d) for d in delivered] + webhook_records(dispatched_workflow(workflow), sent)
    try:
        record_deliveries(run, outcomes)
        _finalize(run_id, shard_prefix)
    except Exception:
        logger.exception("run %s: delivery records could not be written", run_id)


@DBOS.workflow(name=dbos_app.DOWNSTREAM_WORKFLOW)
def run_downstream(
    run_id: str, stages: list[str], shard_prefix: str, max_attempts: int, owner: str | None,
    workflow: dict[str, Any] | None = None, firing: str | None = None, bases: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run *stages* over the run once its extraction has settled, in the order given.

    A scheduled *firing* runs them over its own batch *bases* only, as stages of its own.

    A failed batch does not hold the run: its absence surfaces as the
    not-ready or missing-input error of the stage that needed it, for that
    batch only. A failed stage does not stop the stages after it: each runs
    over what exists, recorded with the failures it follows (`ran_after_failed`
    on its row, as the local runner records), and the coordinator fails last
    if any did, naming them. Then it writes the run's ``manifest.parquet``
    (`cloud/finalize.py`), after a failed stage too; a manifest it could not
    write is named among the failures. Last, a dispatched workflow's run is
    delivered to its destinations; a failed delivery is recorded, not raised.
    A workflow's webhooks are sent the run's events as they happen, each a
    recorded step whose id is this coordinator's, so a recovery does not resend.
    """
    destinations = (workflow or {}).get("spec", {}).get("destinations") or []
    hooks = run_events.has_webhook(workflow)
    sent: list[dict[str, Any]] = []

    def event(name: str, subject: str = "", status: str = "running", **fields: Any) -> None:
        """One run event to the workflow's webhooks, as a recorded step, so a recovered coordinator
        does not send it again; its id is this coordinator's, so a resend keeps it."""
        sent.extend(run_events.record(run_id, workflow, name, f"{name}:{subject}" if subject else name,
                                      status, **fields))

    settled = 0
    while DBOS.run_step({"name": "await-extraction"}, _unsettled_batches, run_id):
        settled, progress = run_events.extraction_progress(run_id, workflow, bases, settled)
        sent.extend(progress)
        DBOS.sleep(SETTLE_POLL_SECONDS)
    sent.extend(run_events.extraction_progress(run_id, workflow, bases, settled)[1])
    event("extraction_settled")

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
        # Routed by the models the stage needs under the run's config, so only a worker checked for them claims it.
        models = models_key(_stage_config(workflow, config), stage) if workflow or config else ""
        event("stage_started", stage, stage=stage)
        with SetWorkflowID(dbos_app.stage_id(run_id, stage, scope)), SetWorkflowAttributes(attrs):
            handle = DBOS.enqueue_workflow(
                dbos_app.stage_queue(stage, models), run_stage, run_id, stage, shard_prefix, max_attempts,
                workflow, bases, config,
            )
        try:
            finished = handle.get_result()
            sent.extend(finished.pop("events", []))
            results.append({**finished, "ran_after_failed": list(failed)})
        except Exception as e:  # a failed stage must not stop the ones after it
            results.append({"stage": stage, "error": _carried(e), "ran_after_failed": list(failed)})
            logger.warning("run %s: %s failed; later stages still run: %s", run_id, stage, results[-1]["error"])
            event("stage_failed", stage, stage=stage, error=results[-1]["error"], ran_after_failed=list(failed))
            failed.append(stage)
        else:  # outside the try, so an event step that raised could never mark a finished stage failed
            event("stage_finished", stage, stage=stage, ran_after_failed=list(failed))
    # Last, and whatever failed: the manifest covers every batch present (Requirement 2).
    unwritten = ""
    try:
        DBOS.run_step(_retry("finalize", max_attempts), _finalize, run_id, shard_prefix)
    except Exception as e:  # recorded, as a failed stage is; the run has landed either way
        unwritten = f"the run manifest was not written ({_carried(e)}); `womblex finalize` writes it"
        logger.warning("run %s: %s", run_id, unwritten)
    if workflow is not None and destinations:
        # After a failed stage too, and never failing the run: each destination's outcome is its own
        # record (`cloud/delivery.py`), marked with the failed stages it follows.
        delivered = DBOS.run_step({"name": "deliver"}, _deliver, run_id, shard_prefix, workflow, failed)
        for o in delivered:
            event("delivered" if o["status"] == "delivered" else "delivery_failed", o["destination"],
                  destination=o["destination"], error=o["error"], after_failed=o["after_failed"])
        if hooks:
            event("run_finished", status=DBOS.run_step({"name": "run-status"}, _run_status, run_id))
        DBOS.run_step({"name": "record-deliveries"}, _record, run_id, shard_prefix, workflow, delivered, sent)
    if failed:
        after = [f"{r['stage']} after {', '.join(r['ran_after_failed'])}"
                 for r in results if r["ran_after_failed"]]
        raise RunIncomplete(
            f"run {run_id}: stage(s) {', '.join(failed)} failed; every stage ran"
            + (f"; ran after a failure: {'; '.join(after)}" if after else "")
            + (f"; {unwritten}" if unwritten else "")
        )
    if unwritten:
        raise RunIncomplete(f"run {run_id}: every stage finished, but {unwritten}")
    return results
