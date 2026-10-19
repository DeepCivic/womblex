"""Object-store execution of a workflow: dispatch its steps to the fleet.

The fleet's counterpart to `run_workflow_local`, through the coordinator path
(`RunBoard.enqueue_stages` -> `run_downstream`). Everything that can refuse does so
here, before a queue row exists, with the local runner's refusal type and codes:
the spec, preset and config (`workflow_runner`), the step order against the store's
listing (`workflow_order.check_dispatch_order`), and three things only a fleet has to
say: the stage is not dispatchable, the list is not in pipeline order (the coordinator
sorts, so another order would not be the order the workflow declares), or the run
already ran a stage under a different workflow. The workflow's resolved overlay and
digest travel with the dispatch (`enqueue_stages(workflow=…)`), so workers run the
stages under that recipe and stamp its digest.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from womblex.cloud.workflow_order import StageOrderError, check_dispatch_order
from womblex.cloud.workflow_runner import (
    RefusalCode,
    WorkflowRefusedError,
    config_from_overlay,
    config_payload,
    not_acted_on,
    workflow_overlay,
)
from womblex.cloud.workflow_store import StoredWorkflow, check_spec
from womblex.pipeline_order import DOWNSTREAM_STAGES, in_pipeline_order
from womblex.store.retention import is_safe_run_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.store.remote import RemoteStore
    from womblex.ui.presets import Preset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowDispatch:
    run_id: str
    workflow_id: str
    digest: str
    stages: tuple[str, ...]
    shard_prefix: str
    newly_enqueued: int
    dry_run: bool
    firing: str | None = None
    bases: list[str] | None = None  # a firing's own batches; None is every batch of the run
    unchanged: dict[str, str] = field(default_factory=dict)  # seen keys whose bytes it found unchanged -> fingerprint


def dispatch_workflow(
    stored: StoredWorkflow,
    *,
    resolve_preset: Callable[[str], Preset | None],
    run_id: str,
    store_uri: str | None,
    dsn: str | None,
    output_prefix: str | None = None,
    max_attempts: int = 3,
    dry_run: bool = False,
    store: RemoteStore | None = None,
    acted_on: tuple[str, ...] = (),
    firing: str | None = None,
) -> WorkflowDispatch:
    """Check *stored* against the store and the board, then enqueue it for *run_id*.

    Raises `WorkflowRefusedError` before any row is written. *dry_run* runs every check and
    writes nothing. *acted_on* names spec fields the caller acts on itself (a firing), so
    they are not warned as left alone. A scheduled *firing* runs the steps over the batches it
    queued, under stage ids of its own; a workflow dispatched outside a firing runs them over
    every batch, under ids scoped by its digest (`dbos_app.stage_scope`), so a run worked under
    another workflow runs this one rather than returning that one's results.
    """
    def refuse(message: str, code: RefusalCode, **kw: object) -> WorkflowRefusedError:
        return WorkflowRefusedError(f"workflow {stored.id!r}: {message}", code=code, **kw)  # type: ignore[arg-type]

    if not is_safe_run_id(run_id):
        raise refuse(f"unsafe run_id {run_id!r}", RefusalCode.CONFIG_INVALID)
    try:
        check_spec(stored.spec)
    except ValueError as e:
        raise refuse(f"is no longer valid: {e}", RefusalCode.SPEC_INVALID) from e
    overlay = workflow_overlay(stored, resolve_preset)
    try:
        config = config_from_overlay(stored.spec.preset, overlay)
    except ValueError as e:
        raise refuse(f"its config does not load: {e}", RefusalCode.CONFIG_INVALID) from e

    stages = tuple(s.stage for s in stored.spec.steps)
    if bad := [s for s in stages if s not in DOWNSTREAM_STAGES]:
        raise refuse(
            f"the fleet does not dispatch {', '.join(bad)} (run them with `run-stage`, or "
            "`run-workflow` locally)", RefusalCode.STAGE_NOT_DISPATCHABLE,
            steps=tuple((s, "not dispatchable") for s in bad),
        )
    if stages != tuple(in_pipeline_order(stages)):
        raise refuse(
            f"steps {list(stages)} are not in pipeline order {list(in_pipeline_order(stages))}; "
            "the coordinator runs stages in pipeline order, so reorder the steps",
            RefusalCode.ORDER_NOT_PIPELINE,
        )
    from womblex.utils.model_check import check_registered

    try:
        check_registered(config)  # registration only: this host may not carry the workers' models
    except ValueError as e:
        raise refuse(str(e), RefusalCode.CONFIG_INVALID) from e

    shard_prefix = f"{(output_prefix or f'runs/{run_id}').strip('/')}/documents"
    from womblex.store.remote import RemoteStore

    store = store or (RemoteStore.from_uri(store_uri) if store_uri else None)
    try:
        check_dispatch_order(list(stages), config, store, shard_prefix)
    except StageOrderError as e:
        raise refuse(f"step order cannot complete: {e}", RefusalCode.STEP_ORDER) from e
    from womblex.cloud.worker_registry import NoLiveWorker, needed_queues, require_served

    try:
        require_served(dsn, needed_queues(config, stages))
    except NoLiveWorker as e:
        raise refuse(str(e), RefusalCode.NO_WORKER) from e

    from womblex.cloud.dbos_app import ensure_schema, local_database_path
    from womblex.cloud.jobs import RunBoard

    if not dry_run:
        ensure_schema(dsn)
    # A dry run does not create the local database just to find it empty.
    unread = dry_run and not dsn and not local_database_path().exists()
    with nullcontext() if unread else RunBoard(dsn) as board:
        bases = None if firing is None or board is None else [
            f"batch-{n:04d}" for n in board.firing_batches(run_id, firing)]
        if firing is not None and not dry_run and not bases:
            raise refuse(f"run {run_id!r} has no batches from firing {firing!r}", RefusalCode.NO_DOCUMENTS)
        newly = 0 if dry_run or board is None else board.enqueue_stages(
            run_id, list(stages), shard_prefix, max_attempts=max_attempts,
            workflow={"id": stored.id, "digest": stored.digest, "preset": stored.spec.preset,
                      "config": overlay}, firing=firing, bases=bases,
        )
    if left := [f for f in not_acted_on(stored) if f not in acted_on]:
        logger.warning("workflow %s: %s stored but not acted on by dispatch", stored.id, ", ".join(left))
    return WorkflowDispatch(
        run_id, stored.id, stored.digest, stages, shard_prefix, newly, dry_run, firing, bases,
    )


def _firing_kwargs(
    stored: StoredWorkflow, run_id: str, resolve_preset: Callable[[str], Preset | None],
    store: RemoteStore, dsn: str | None,
) -> dict[str, Any]:
    return {"resolve_preset": resolve_preset, "run_id": run_id, "store_uri": None, "store": store,
            "dsn": dsn, "acted_on": ("triggers", "input_binding.store", "input_binding.ingest_prefix")}


def check_firing(
    stored: StoredWorkflow, *, run_id: str, resolve_preset: Callable[[str], Preset | None],
    store: RemoteStore, dsn: str | None, ingest: RemoteStore | None = None,
) -> list[str]:
    """Every check a firing makes before it writes a row; the keys it would extract.

    A dry-run dispatch, a named input store, a safe input prefix, and at least one document
    there. *ingest* is the input store opened with the caller's credentials (a worker's own);
    `None` opens it from the workflow's URI. Raises
    `WorkflowRefusedError`; a firing and `schedule-workflow` both call it, so a workflow is
    refused at scheduling for exactly what its firing would refuse.
    """
    from womblex.cli._shared import normalise_prefix
    from womblex.cloud.dispatch import supported_keys

    dispatch_workflow(stored, dry_run=True, **_firing_kwargs(stored, run_id, resolve_preset, store, dsn))
    if not (ingest_root := stored.spec.input_binding.store):
        raise WorkflowRefusedError(f"workflow {stored.id!r}: a scheduled workflow names its input store "
                                   "(input_binding.store)", code=RefusalCode.SPEC_INVALID)
    try:
        prefix = normalise_prefix(stored.spec.input_binding.ingest_prefix)
    except ValueError as e:
        raise WorkflowRefusedError(f"workflow {stored.id!r}: {e}", code=RefusalCode.SPEC_INVALID) from e
    from womblex.cloud.worker_registry import NoLiveWorker, needed_queues, require_served
    from womblex.store.remote import RemoteStore

    try:
        recipe = config_from_overlay(stored.spec.preset, workflow_overlay(stored, resolve_preset))
        require_served(dsn, needed_queues(recipe, ingest_root=ingest_root))
    except NoLiveWorker as e:
        raise WorkflowRefusedError(f"workflow {stored.id!r}: {e}", code=RefusalCode.NO_WORKER) from e
    location, keys = supported_keys(ingest or RemoteStore.from_uri(ingest_root), ingest_root, prefix)
    if not keys:
        raise WorkflowRefusedError(f"workflow {stored.id!r}: no supported documents under {location}",
                                   code=RefusalCode.NO_DOCUMENTS)
    return keys


def fire_workflow(
    stored: StoredWorkflow,
    *,
    run_id: str,
    firing: str,
    resolve_preset: Callable[[str], Preset | None],
    store: RemoteStore,
    ingest: RemoteStore,
    dsn: str | None,
) -> WorkflowDispatch:
    """One firing of a standing trigger: append the input prefix's new documents to *run_id*, then
    dispatch the steps over them.

    A document is new when the run has not queued its key with its current fingerprint
    (`RemoteStore.fingerprints`) and, for a seen key whose fingerprint moved, its bytes differ from
    the run's recorded `source_hash` (`_changed_content`), so a changed file is a new document, the
    earlier version stays, and a rewrite of the same bytes is not re-paid. A seen key found unchanged
    goes back with its fingerprint (`WorkflowDispatch.unchanged`, the firing's recorded output), so
    the next firing counts it seen and does not hash it again. A firing with nothing new queues and dispatches nothing. A replayed *firing* reuses
    the batches it queued. The launch, so every check runs first (`check_firing`, then that this
    firing's batches were not cancelled) and a firing that cannot complete writes no row. A
    dispatch that fails after the batches are queued leaves them running: the run reads
    `orphaned` on the board until the firing's stages are dispatched.
    """
    from womblex.cli._shared import normalise_prefix
    from womblex.cloud.dispatch import batch_specs
    from womblex.cloud.jobs import RunBoard

    keys = check_firing(stored, run_id=run_id, resolve_preset=resolve_preset, store=store,
                        ingest=ingest, dsn=dsn)
    binding = stored.spec.input_binding
    shard_prefix = f"runs/{run_id}/documents"
    unchanged: dict[str, str] = {}
    with RunBoard(dsn) as board:
        if cancelled := board.cancelled_batches(run_id, firing):  # a replayed firing: they would never run
            raise WorkflowRefusedError(f"firing {firing} of run {run_id!r} has cancelled batches "
                                       f"({', '.join(cancelled)})", code=RefusalCode.BATCHES_CANCELLED)
        if not board.firing_batches(run_id, firing):  # a replay reuses its batches and fingerprints nothing
            prints = ingest.fingerprints(keys, under=normalise_prefix(binding.ingest_prefix))
            seen = board.seen_inputs(run_id)
            new = [k for k in keys if (k, prints[k]) not in seen]
            new, same = _changed_content(stored.id, new, {k for k, _ in seen}, ingest, store, shard_prefix)
            unchanged = {k: prints[k] for k in same}  # recorded on the firing, so later ones do not hash them again
            if not new:
                logger.info("workflow %s: no new documents under its input prefix; firing %s runs nothing",
                            stored.id, firing)
                return WorkflowDispatch(run_id, stored.id, stored.digest, (), "", 0, False, firing, [], unchanged)
            specs = batch_specs(new, shard_prefix=shard_prefix, ingest_root=binding.store or "",
                                batch_size=binding.batch_size, max_attempts=3, start=board.next_batch_num(run_id))
            recipe = config_from_overlay(stored.spec.preset, workflow_overlay(stored, resolve_preset))
            board.enqueue(run_id, specs, firing=firing, fingerprints=prints, config=config_payload(recipe))
    try:
        sent = dispatch_workflow(stored, firing=firing, **_firing_kwargs(stored, run_id, resolve_preset, store, dsn))
    except Exception:
        logger.error("firing %s of run %s failed after queueing its batches; extraction runs on and the run reads "
                     "orphaned: dispatch its stages with `womblex dispatch-workflow <the %s workflow file> --run-id %s "
                     "--firing %s --store <store>`", firing, run_id, stored.id, run_id, firing)
        raise
    logger.info("workflow %s fired: run %s, firing %s, batches %s, stages %s", stored.id, run_id, firing,
                ", ".join(sent.bases or ()), ", ".join(sent.stages))
    return replace(sent, unchanged=unchanged)


def _changed_content(
    workflow_id: str, new: list[str], seen_keys: set[str], ingest: RemoteStore, store: RemoteStore,
    shard_prefix: str,
) -> tuple[list[str], set[str]]:
    """*new* without the seen keys whose bytes are unchanged, and those keys, so a rewrite of the same bytes is not re-paid.

    A seen key's fingerprint can move while its bytes stay (a re-copy, a multipart re-upload), so
    its bytes are hashed and compared with the `source_hash` the run's manifests record for it. A
    key with no manifest row (its earlier batch still extracting, or failed) cannot be compared and
    is queued, so a changed file is never held back.
    """
    import tempfile
    from pathlib import Path

    import pyarrow.parquet as pq

    moved = [k for k in new if k in seen_keys]
    if not moved:
        return new, set()
    recorded: dict[str, set[str]] = {}
    with tempfile.TemporaryDirectory(prefix="womblex-seen-") as tmp:
        for key in store.list_files(shard_prefix, "*._manifest.parquet"):
            local = store.download_file(key, Path(tmp) / key.rsplit("/", 1)[-1])
            rows = pq.read_table(local, columns=["source_relpath", "source_hash"]).to_pylist()
            for row in rows:
                recorded.setdefault(row["source_relpath"], set()).add(row["source_hash"])
    same = {k for k in moved if k in recorded and ingest.sha256(k) in recorded[k]}
    for key in sorted(same):
        logger.info("workflow %s: %s was rewritten with the same bytes; not queued again", workflow_id, key)
    return [k for k in new if k not in same], same


__all__ = ["WorkflowDispatch", "check_firing", "dispatch_workflow", "fire_workflow"]
