"""The run board: enqueue work and read its progress, over DBOS workflow rows.

Replaces the ``womblex_jobs`` table. One workflow is one row of the old
board: an extraction batch (``kind='batch'``), or the coordinator that runs a
run's downstream stages in order (``kind='downstream'``) with one child
workflow per stage (``kind='stage'``). Workflow ids are deterministic
(``<run_id>:batch:0001``), so enqueueing twice attaches to the existing row
and a re-enqueue resumes a run rather than duplicating it.

Ownership travels as the ``owner`` workflow attribute. DBOS's own statuses are
mapped back to the board's vocabulary (``pending`` / ``running`` / ``done`` /
``failed`` / ``cancelled``), which the API speaks.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Self, TypeGuard

import sqlalchemy.exc

from womblex.cloud import dbos_app
from womblex.cloud.run_owner import owned_rows, run_owners
from womblex.cloud.run_status import run_sentence, stage_counts

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dbos import EnqueueOptions, WorkflowStatus

logger = logging.getLogger(__name__)

_STATUS = {
    "ENQUEUED": "pending",
    "DELAYED": "pending",
    "PENDING": "running",
    "SUCCESS": "done",
    "ERROR": "failed",
    "CANCELLED": "cancelled",
    "MAX_RECOVERY_ATTEMPTS_EXCEEDED": "failed",
}


def run_state(counts: dict[str, int]) -> str:
    """Collapse a status rollup into one state; any failed job fails the run, and a finished run
    with a cancelled job is cancelled. A skipped stage counts as finished. The API's words, which
    webhook events use too."""
    if counts.get("failed"):
        return "failed"
    if counts.get("running") or (counts.get("pending") and (counts.get("done") or counts.get("skipped"))):
        return "running"
    if counts.get("pending"):
        return "pending"
    return "cancelled" if counts.get("cancelled") else "done"


@dataclass
class JobSpec:
    """A batch to enqueue: which input keys, and where its shards go."""

    batch_num: int
    input_keys: list[str]
    shard_prefix: str
    ingest_root: str
    max_attempts: int = 3


@dataclass(frozen=True)
class JobRow:
    """One workflow as read — the job list's grain. Timestamps are ISO-8601 strings."""

    id: str
    run_id: str
    batch_num: int
    status: str
    attempts: int
    max_attempts: int
    locked_by: str | None
    locked_at: str | None
    error: str | None
    created_at: str | None
    updated_at: str | None
    kind: str = "batch"
    stage: str | None = None
    retried_by: str | None = None  # a failed stage's next attempt (`<run>:stage:<s>:retry:N`)


@dataclass(frozen=True)
class RunSummary:
    """One run's rows rolled up by status — the run list's grain."""

    run_id: str
    owner: str | None
    counts: dict[str, int]
    created_at: str | None
    updated_at: str | None
    orphaned: bool = False  # extraction queued by a firing that failed before its stages were dispatched
    stages: dict[str, dict[str, int]] = field(default_factory=dict)  # status counts per stage (`run_status`)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def sentence(self) -> str:
        """The run's status as one sentence (`run_status.run_sentence`)."""
        return run_sentence(self.run_id, run_state(self.counts), self.stages, orphaned=self.orphaned)


class RunOwnedError(Exception):
    """A run id is already owned by a different owner."""


@dataclass(frozen=True)
class WorkerState:
    """A worker's live hold on the board, derived from the workflow's executor.

    Not liveness: a stopped worker's unfinished workflows stay here until DBOS
    recovers them, so an old entry means orphaned work, not a busy worker.
    """

    worker_id: str
    running: int
    oldest_locked_at: str | None
    newest_locked_at: str | None


@dataclass(frozen=True)
class Throughput:
    """Completions inside a trailing window — the dashboard's rate tile."""

    window_seconds: float
    completed: int
    per_minute: float
    last_completed_at: str | None


def _iso(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat() if ms else None


def _attr(status: WorkflowStatus, key: str, default: Any = None) -> Any:
    return (status.attributes or {}).get(key, default)


def _error_text(error: BaseException | None) -> str | None:
    """The failure as the board shows it, with the cause DBOS wraps on exhausted retries."""
    if error is None:
        return None
    causes = getattr(error, "errors", None)
    last = f" — last error: {type(causes[-1]).__name__}: {causes[-1]}" if causes else ""
    return f"{error}{last}"[:2000]


def schema_missing(e: Exception) -> bool:
    """The DBOS system tables have not been created yet (Postgres or SQLite)."""
    orig = getattr(e, "orig", None)
    return getattr(orig, "sqlstate", None) in ("42P01", "3F000") or "no such table" in str(orig)


def _latest_attempts(rows: list[WorkflowStatus]) -> list[WorkflowStatus]:
    """*rows* without the stage attempts a retry superseded: the run's rollup counts the latest."""
    newest: dict[str, int] = {}
    for s in rows:
        if attempt := dbos_app.stage_attempt(s.workflow_id):
            newest[attempt[0]] = max(newest.get(attempt[0], 0), attempt[1])
    return [s for s in rows
            if not (attempt := dbos_app.stage_attempt(s.workflow_id)) or attempt[1] == newest[attempt[0]]]


def _rollup(rows: list[WorkflowStatus]) -> list[tuple[WorkflowStatus, str]]:
    """Each row a run's rollup counts, with the status it counts as: its latest attempt's.

    Superseded stage attempts are dropped. A failed coordinator whose stages' latest attempts
    have all succeeded since counts `done`, and so does a failed firing whose stages were
    dispatched by hand: the coordinator then carries their outcome. Rows keep their own status
    in `list_jobs`; only the run's counts follow the repair.
    """
    latest = _latest_attempts(rows)
    stage_status = {(_attr(s, "run_id", ""), a[0]): _board_status(s)
                    for s in latest if (a := dbos_app.stage_attempt(s.workflow_id))}
    dispatched = {(_attr(s, "run_id", ""), _attr(s, "firing"))
                  for s in latest if _attr(s, "kind") == dbos_app.KIND_DOWNSTREAM}
    out: list[tuple[WorkflowStatus, str]] = []
    for s in latest:
        status, kind, run = _shown_status(s), _attr(s, "kind"), _attr(s, "run_id", "")
        if status == "failed" and kind == dbos_app.KIND_DOWNSTREAM:
            scope = dbos_app.stage_scope(_attr(s, "firing"), _attr(s, "workflow_digest"))
            stages = _attr(s, "stages") or []
            if stages and all(stage_status.get((run, dbos_app.stage_id(run, st, scope))) == "done" for st in stages):
                status = "done"
        elif status == "failed" and kind == dbos_app.KIND_TRIGGER and (run, _attr(s, "firing")) in dispatched:
            status = "done"
        out.append((s, status))
    return out


def _attempt_number(status: WorkflowStatus) -> int:
    attempt = dbos_app.stage_attempt(status.workflow_id)
    return attempt[1] if attempt else 0


def _retried_by(rows: list[WorkflowStatus]) -> dict[str, str]:
    """Each superseded stage attempt's id -> the id of the attempt after it."""
    attempts: dict[str, dict[int, str]] = {}
    for s in rows:
        if attempt := dbos_app.stage_attempt(s.workflow_id):
            attempts.setdefault(attempt[0], {})[attempt[1]] = s.workflow_id
    return {ids[n]: ids[min(m for m in ids if m > n)]
            for ids in attempts.values() for n in ids if any(m > n for m in ids)}


def _orphaned(rows: list[WorkflowStatus]) -> bool:
    """A run with a firing that failed after queueing its batches: its extraction and no coordinator.

    Read from the rows themselves, so it holds even when the database that failed the
    dispatch could not have taken a label either; dispatching the firing's stages clears it.
    """
    failed = {_attr(s, "firing") for s in rows if _attr(s, "kind") == dbos_app.KIND_TRIGGER and _board_status(s) == "failed"}
    batched = {_attr(s, "firing") for s in rows if _attr(s, "kind") == dbos_app.KIND_BATCH}
    dispatched = {_attr(s, "firing") for s in rows if _attr(s, "kind") == dbos_app.KIND_DOWNSTREAM}
    return bool(failed & batched - dispatched)


def _board_status(status: WorkflowStatus) -> str:
    return _STATUS.get(status.status, "failed")


def _shown_status(status: WorkflowStatus) -> str:
    """The status a row is shown and counted with: a stage that succeeded with every unit
    skipped (`cloud/unit_receipt.py`) reads `skipped`. Retries and checksums read `_board_status`,
    where it is `done`. Needs the row's output; a row listed without it reads `done`."""
    board = _board_status(status)
    out = status.output if isinstance(status.output, dict) else {}
    if board == "done" and _attr(status, "kind") == dbos_app.KIND_STAGE and out.get("units") \
            and out.get("skipped") == out["units"]:
        return "skipped"
    return board


def queue_models(config: dict[str, Any] | None, scope: str) -> str:
    """The models key (`model_check.models_key`) a run's *config* payload routes *scope*'s work by."""
    if config is None:  # a run that carries no config: the unkeyed queue, served as before
        return ""
    from womblex.cloud.workflow_runner import config_from_payload
    from womblex.utils.model_check import models_key

    return models_key(config_from_payload(config, {"input_root": ".", "output_root": ".", "checkpoint_dir": "."}), scope)


def _job_row(s: WorkflowStatus) -> JobRow:
    running = _board_status(s) == "running"
    return JobRow(
        id=s.workflow_id,
        run_id=_attr(s, "run_id", ""),
        batch_num=int(_attr(s, "batch_num", 0)),
        status=_shown_status(s),
        attempts=0 if _board_status(s) == "pending" else 1,
        max_attempts=int(_attr(s, "max_attempts", 3)),
        locked_by=s.executor_id if running else None,
        locked_at=_iso(s.updated_at) if running else None,
        error=_error_text(s.error),
        created_at=_iso(s.created_at),
        updated_at=_iso(s.updated_at),
        kind=_attr(s, "kind", "batch"),
        stage=_attr(s, "stage"),
    )


@dataclass(frozen=True)
class FileChecksums:
    """A run's published files with their digests, and the workflows that may
    have published files without one (failed, unfinished, or recorded before
    checksums existed)."""

    checksums: dict[str, str]
    undigested: list[str]


def _is_digest_map(value: Any) -> TypeGuard[dict[str, str]]:
    return isinstance(value, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    )


class RunBoard:
    """Enqueue and read runs through one DBOS client."""

    def __init__(self, dsn: str | None, *, connect_timeout: float | None = None):
        self._client = dbos_app.open_client(dsn, connect_timeout=connect_timeout)

    def close(self) -> None:
        self._client.destroy()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- writes ---------------------------------------------------------------

    def _list(
        self, run_id: str | None = None, *, owner: str | None = None,
        status: list[str] | str | None = None, **kwargs: Any,
    ) -> list[WorkflowStatus]:
        kwargs.setdefault("load_input", False)
        kwargs.setdefault("load_output", False)
        if run_id:
            kwargs["workflow_id_prefix"] = dbos_app.run_prefix(run_id)
        try:
            rows = self._client.list_workflows(status=status, **kwargs)
        except sqlalchemy.exc.DBAPIError as e:
            # Readers do not migrate: a database no writer has touched holds no runs.
            if not schema_missing(e):
                raise
            return []
        if owner is None:
            return rows
        # Filtered here, not by DBOS: filtering on attributes is Postgres-only. A run is the
        # owner its first row names (`cloud/run_owner.py`), so a filtered listing reads all rows.
        filtered = status or any(k in kwargs for k in ("limit", "completed_after"))
        whole = self._list(run_id) if filtered else rows
        return owned_rows(rows, run_owners(whole), owner)

    def check_owner(self, run_id: str, owner: str | None) -> None:
        """Raise :class:`RunOwnedError` where :meth:`enqueue` for *owner* would; write nothing."""
        self._claim_run_owner(run_id, owner)

    def _claim_run_owner(self, run_id: str, owner: str | None) -> str | None:
        """Resolve the owner new rows of *run_id* carry; refuse a different owner.

        An unscoped caller (``owner=None``: the CLI) is not
        checked and inherits the run's existing owner. Two submitters racing to
        create the same run can both pass this check; the first row written
        names the owner and the second attaches to its deterministic ids.
        """
        existing = self._list(run_id, limit=1)
        if not existing:
            return owner
        current = _attr(existing[0], "owner")
        if owner is not None and current != owner:
            raise RunOwnedError(f"run {run_id!r} is owned by another owner")
        return owner if owner is not None else current

    def _enqueue(self, sends: list[tuple[EnqueueOptions, tuple]]) -> int:
        """Enqueue each workflow; return how many were not already on the board.

        The workflow id is deterministic, so enqueueing one that exists attaches
        to it rather than starting a second.
        """
        ids = [options["workflow_id"] for options, _ in sends]
        present = {s.workflow_id for s in self._client.list_workflows(
            workflow_ids=ids, load_input=False, load_output=False)}
        for options, args in sends:
            self._client.enqueue(options, *args)
        return len(set(ids) - present)

    def enqueue(
        self, run_id: str, jobs: list[JobSpec], *, owner: str | None = None,
        firing: str | None = None, fingerprints: dict[str, str] | None = None,
        config: dict[str, Any] | None = None, workflow: dict[str, Any] | None = None,
    ) -> int:
        """Enqueue *jobs* for *run_id*; idempotent per batch number.

        *config* (`workflow_runner.config_payload`) is the config the batches extract under, on
        any worker. A scheduled *firing* labels its batches, each with its keys' *fingerprints*, so a
        later firing of the same run can tell which documents it has already seen. A firing's
        *workflow* (``id``, ``digest``, ``spec``) is named in the footer of every file its batches write.

        Returns the number of batches that were not already on the board.
        Raises :class:`RunOwnedError` if *owner* is named and the run already
        belongs to someone else, including a run with no owner.
        """
        row_owner = self._claim_run_owner(run_id, owner)
        from womblex.utils.model_check import SCOPE_EXTRACT

        models = queue_models(config, SCOPE_EXTRACT)
        sends = []
        for spec in jobs:
            attrs = {"run_id": run_id, "kind": dbos_app.KIND_BATCH, "batch_num": spec.batch_num,
                     "max_attempts": spec.max_attempts}
            if row_owner is not None:
                attrs["owner"] = row_owner
            if firing is not None:
                attrs["firing"] = firing
                attrs["fingerprints"] = {k: (fingerprints or {}).get(k, "") for k in spec.input_keys}
            options: EnqueueOptions = {
                "workflow_name": dbos_app.EXTRACT_WORKFLOW,
                "queue_name": dbos_app.extract_queue(spec.ingest_root, models),
                "workflow_id": dbos_app.batch_id(run_id, spec.batch_num),
                "attributes": attrs,
            }
            sends.append((options, (run_id, spec.batch_num, spec.input_keys, spec.shard_prefix,
                                    spec.ingest_root, spec.max_attempts, config, workflow)))
        inserted = self._enqueue(sends)
        logger.info("Enqueued %d new job(s) for run %s (of %d submitted)",
                    inserted, run_id, len(jobs))
        return inserted

    def enqueue_stages(
        self, run_id: str, stages: list[str], shard_prefix: str, *,
        max_attempts: int = 3, owner: str | None = None, workflow: dict[str, Any] | None = None,
        firing: str | None = None, bases: list[str] | None = None, config: dict[str, Any] | None = None,
    ) -> int:
        """Enqueue the coordinator that runs *stages* in pipeline order for *run_id*.

        *config* (`workflow_runner.config_payload`) is the config every stage runs under, on any
        worker, when no *workflow* is given.

        A scheduled *firing* runs its stages over its own batch *bases* only, under stage
        ids of its own, so a run a workflow appends to never re-runs an earlier firing's work.

        *workflow* (`id`, `digest`, `preset`, `config`: the resolved overlay) makes the fleet
        run the stages under that config and stamp its digest on what they write, rather
        than under the worker's own config; it is part of the coordinator's identity.

        Order is not taken from the list: the coordinator sorts by
        ``stage_rank``, so an out-of-order list still runs in pipeline order and
        a duplicated name collapses. The same stage list dispatched twice is the
        same coordinator, so pressing the button again re-runs nothing; a
        different list starts a new coordinator whose already-finished stages
        return their recorded result. Returns the number of stages newly
        dispatched. *owner* is checked as in :meth:`enqueue`.
        """
        from womblex.pipeline_order import in_pipeline_order

        ordered = list(in_pipeline_order(stages))
        row_owner = self._claim_run_owner(run_id, owner)
        attrs: dict[str, Any] = {"run_id": run_id, "kind": dbos_app.KIND_DOWNSTREAM,
                                 "max_attempts": max_attempts, "stages": ordered}
        if row_owner is not None:
            attrs["owner"] = row_owner
        if workflow is not None:
            attrs["workflow_digest"] = workflow["digest"]
        if firing is not None:
            attrs["firing"] = firing
        options: EnqueueOptions = {
            "workflow_name": dbos_app.DOWNSTREAM_WORKFLOW,
            "queue_name": dbos_app.DOWNSTREAM_QUEUE,
            "workflow_id": dbos_app.downstream_id(
                run_id, ordered, workflow["digest"] if workflow else None, firing),
            "attributes": attrs,
        }
        fresh = self._enqueue([(options, (run_id, ordered, shard_prefix, max_attempts, row_owner, workflow,
                                          firing, bases, config))])
        newly = len(ordered) if fresh else 0
        logger.info("Enqueued %d new stage job(s) for run %s (of %d submitted)",
                    newly, run_id, len(stages))
        return newly

    def seen_inputs(self, run_id: str) -> set[tuple[str, str]]:
        """Every `(key, fingerprint)` *run_id*'s firings have seen: queued for extraction, or found to
        hold bytes the run already has (a firing's recorded `unchanged`)."""
        rows = self._list(run_id, load_output=True)
        queued = {(k, fp) for s in rows if _attr(s, "kind") == dbos_app.KIND_BATCH
                  for k, fp in (_attr(s, "fingerprints") or {}).items()}
        return queued | {(k, fp) for s in rows if _attr(s, "kind") == dbos_app.KIND_TRIGGER
                         and isinstance(s.output, dict) for k, fp in (s.output.get("unchanged") or {}).items()}

    def firing_batches(self, run_id: str, firing: str) -> list[int]:
        """The batch numbers *firing* queued in *run_id*: a replayed firing reuses them."""
        return sorted(int(_attr(s, "batch_num", 0)) for s in self._list(run_id)
                      if _attr(s, "kind") == dbos_app.KIND_BATCH and _attr(s, "firing") == firing)

    def next_batch_num(self, run_id: str) -> int:
        """One past the highest batch number in *run_id*, so a firing appends rather than attaches."""
        return 1 + max((int(_attr(s, "batch_num", 0)) for s in self._list(run_id)
                        if _attr(s, "kind") == dbos_app.KIND_BATCH), default=0)

    def stage_attempts(
        self, run_id: str, *, owner: str | None = None,
    ) -> dict[str, list[WorkflowStatus]]:
        """Each stage of *run_id* with its attempts, the original first.

        Raises :class:`RunOwnedError` for a named *owner* that does not own the run.
        """
        self._claim_run_owner(run_id, owner)
        bases: dict[str, list[WorkflowStatus]] = {}
        for s in self._list(run_id):
            if _attr(s, "kind") == dbos_app.KIND_STAGE and (attempt := dbos_app.stage_attempt(s.workflow_id)):
                bases.setdefault(attempt[0], []).append(s)
        # A run a workflow appends to holds one base per firing; a stage's attempts are those of its
        # newest unfinished firing (else its newest), so a retry reaches each failure in turn.
        found: dict[str, list[WorkflowStatus]] = {}
        for rows in sorted(bases.values(), key=lambda r: min(s.created_at or 0 for s in r)):
            rows.sort(key=_attempt_number)
            stage, current = _attr(rows[0], "stage"), found.get(_attr(rows[0], "stage"))
            if current is None or _board_status(current[-1]) == "done" or _board_status(rows[-1]) != "done":
                found[stage] = rows
        return found

    def stage_arguments(self, workflow_id: str) -> dict[str, Any]:
        """A stage workflow's recorded arguments by name (`shard_prefix`, `workflow`, `bases`, `config`)."""
        (s,) = self._client.list_workflows(workflow_ids=[workflow_id], load_input=True, load_output=False)
        given = s.input or {"args": (), "kwargs": {}}  # read by name, whether passed by position or keyword
        names = ("run_id", "stage", "shard_prefix", "max_attempts", "workflow", "bases", "config")
        return dict(zip(names, given["args"])) | given["kwargs"]

    def first_failed_step(self, workflow_id: str) -> int:
        """The function id of the workflow's first failed step; 1 (the whole workflow) when none is recorded."""
        failed = [st["function_id"] for st in self._client.list_workflow_steps(workflow_id)
                  if st["error"] is not None]
        return min(failed, default=1)

    def unfinished_bases(self, workflow_id: str) -> list[str] | None:
        """The bases a stage attempt planned and did not finish; `None` when it recorded no plan."""
        steps = self._client.list_workflow_steps(workflow_id)
        plan = next((st for st in steps if st["function_name"] == "plan"), None)
        if plan is None or plan["error"] is not None or not isinstance(plan["output"], list):
            return None
        done = {st["function_name"].removeprefix("unit:") for st in steps
                if st["function_name"].startswith("unit:") and st["error"] is None}
        return [b for unit in plan["output"] for b in unit if (unit[0] if len(unit) == 1 else "run") not in done]

    def rerun_stage(
        self, workflow_id: str, stage: str, new_id: str, bases: list[str] | None, models: str = "",
    ) -> bool:
        """A new attempt, *new_id*, of *workflow_id*'s stage over *bases* alone (`None`: every base),
        with its recorded arguments and attributes, less `ran_after_failed`: it runs on input that
        is now complete. False, and nothing enqueued, if *new_id* already exists."""
        if self._client.list_workflows(workflow_ids=[new_id], load_input=False, load_output=False):
            return False
        (s,) = self._client.list_workflows(workflow_ids=[workflow_id], load_input=False, load_output=False)
        args = self.stage_arguments(workflow_id)
        attrs = {k: v for k, v in (s.attributes or {}).items() if k != "ran_after_failed"}
        options: EnqueueOptions = {"workflow_name": dbos_app.STAGE_WORKFLOW, "queue_name": dbos_app.stage_queue(stage, models),
                                   "workflow_id": new_id, "attributes": attrs}
        self._client.enqueue(options, args["run_id"], stage, args["shard_prefix"], args["max_attempts"],
                             args.get("workflow"), bases, args.get("config"))
        return True

    def fork_stage(self, workflow_id: str, stage: str, new_id: str, start_step: int, models: str = "") -> None:
        """Fork *workflow_id* from *start_step* as *new_id* on the stage's queue for *models*.

        The steps before *start_step* are kept, and the fork carries the original's attributes
        (owner, workflow digest, the stages it ran after), so it reads as the same stage.
        """
        from dbos import SetWorkflowID

        with SetWorkflowID(new_id):
            self._client.fork_workflow(workflow_id, start_step, queue_name=dbos_app.stage_queue(stage, models))

    def cancelled_batches(self, run_id: str, firing: str | None = None) -> list[str]:
        """*run_id*'s extraction batches (*firing*'s alone, if named) that were cancelled: a
        re-enqueue would attach to them and never run."""
        return [s.workflow_id for s in self._list(run_id, status="CANCELLED")
                if _attr(s, "kind") == dbos_app.KIND_BATCH and (firing is None or _attr(s, "firing") == firing)]

    # --- reads ----------------------------------------------------------------

    def _rows(self, run_id: str | None, owner: str | None, **kwargs: Any) -> list[WorkflowStatus]:
        return self._list(run_id, owner=owner, **kwargs)

    def stats(self, run_id: str | None = None, *, owner: str | None = None) -> dict[str, int]:
        """Count jobs by status (optionally for one run, one owner), each by its latest attempt (`_rollup`)."""
        return dict(Counter(status for _, status in _rollup(self._rows(run_id, owner, load_output=True))))

    def runs(
        self, owner: str | None = None, *, run_id: str | None = None, limit: int = 100,
    ) -> list[RunSummary]:
        """Runs with their jobs rolled up by status, latest activity first.

        *owner* narrows to one caller's runs; ``None`` lists every run,
        including CLI-submitted ones that have no owner.
        """
        grouped: dict[str, list[tuple[WorkflowStatus, str]]] = {}
        for s, status in _rollup(self._rows(run_id, owner, load_output=True)):
            grouped.setdefault(_attr(s, "run_id", ""), []).append((s, status))
        summaries = [
            RunSummary(
                rid, _attr(rows[0][0], "owner"),
                dict(Counter(status for _, status in rows)),
                _iso(min(s.created_at or 0 for s, _ in rows)),
                _iso(max(s.updated_at or 0 for s, _ in rows)),
                _orphaned([s for s, _ in rows]),
                stage_counts((_attr(s, "kind"), _attr(s, "stage"), status) for s, status in rows),
            )
            for rid, rows in grouped.items() if rid
        ]
        return sorted(summaries, key=lambda r: r.updated_at or "", reverse=True)[:limit]

    def list_jobs(
        self, run_id: str | None = None, *, status: str | None = None, limit: int = 200,
        owner: str | None = None,
    ) -> list[JobRow]:
        """Recent jobs, newest activity first, optionally scoped by run and status.

        A stage a retry superseded stays listed, `failed`, naming the attempt after it.
        """
        rows = self._rows(run_id, owner, load_output=True)
        rows.sort(key=lambda s: s.updated_at or 0, reverse=True)
        retried = _retried_by(rows)
        jobs = [replace(_job_row(s), retried_by=retried.get(s.workflow_id)) for s in rows]
        return [j for j in jobs if status is None or j.status == status][:limit]

    def file_checksums(self, run_id: str) -> FileChecksums:
        """Every file the run's extraction and stage workflows published, key -> SHA-256.

        A finished workflow's digests come from its output; one that failed or
        has not finished contributes the steps it completed. Read oldest first,
        so a key a later workflow rewrote (an in-place stage, a re-dispatched
        stage) carries the newest digest.
        """
        every = sorted(self._rows(run_id, None, load_output=True), key=lambda s: s.created_at or 0)
        rows = [s for s in every if _attr(s, "kind", dbos_app.KIND_BATCH) in (dbos_app.KIND_BATCH, dbos_app.KIND_STAGE)]
        checksums: dict[str, str] = {}
        undigested: list[str] = []
        for s in rows:
            digests: dict[str, str] = {}
            if _board_status(s) == "done":
                out = s.output if isinstance(s.output, dict) else {}
                if _is_digest_map(recorded := out.get("checksums")):
                    digests = recorded
                # A list output, or a count above the digests, is a workflow
                # recorded before checksums: its files carry none.
                if not isinstance(s.output, dict) or out.get("published", 0) > len(digests):
                    undigested.append(s.workflow_id)
            else:
                for step in self._client.list_workflow_steps(s.workflow_id):
                    if step["error"] is None and _is_digest_map(output := step["output"]):
                        digests.update(output)
                # A step that stopped part-way may have published without returning.
                undigested.append(s.workflow_id)
            checksums.update(digests)
        return FileChecksums(checksums, sorted(undigested))

    def stale_jobs(
        self, older_than_seconds: float, run_id: str | None = None, *,
        owner: str | None = None,
    ) -> list[JobRow]:
        """Running jobs with no update for longer than the threshold, oldest first.

        Read-only: DBOS recovers a dead executor's workflows when it restarts;
        this only names what has gone quiet.
        """
        cutoff = (datetime.now(UTC) - timedelta(seconds=older_than_seconds)).timestamp() * 1000
        quiet = [s for s in self._rows(run_id, owner, status="PENDING")
                 if (s.updated_at or 0) < cutoff]
        quiet.sort(key=lambda s: s.updated_at or 0)
        return [_job_row(s) for s in quiet]

    def workers(
        self, run_id: str | None = None, *, owner: str | None = None,
    ) -> list[WorkerState]:
        """Which executors hold which running workflows right now — the fleet view."""
        held: dict[str, list[int]] = {}
        for s in self._rows(run_id, owner, status="PENDING"):
            if s.executor_id:
                held.setdefault(s.executor_id, []).append(s.updated_at or 0)
        return [
            WorkerState(worker_id=w, running=len(t), oldest_locked_at=_iso(min(t)),
                        newest_locked_at=_iso(max(t)))
            for w, t in sorted(held.items())
        ]

    def throughput(
        self, run_id: str | None = None, *, window_seconds: float = 3600.0,
        owner: str | None = None,
    ) -> Throughput:
        """Jobs completed in the trailing window, as a rate."""
        since = (datetime.now(UTC) - timedelta(seconds=window_seconds)).isoformat()
        done = self._rows(run_id, owner, status="SUCCESS", completed_after=since)
        completed = len(done)
        return Throughput(
            window_seconds=window_seconds,
            completed=completed,
            per_minute=completed / (window_seconds / 60.0) if window_seconds > 0 else 0.0,
            last_completed_at=_iso(max((s.updated_at or 0 for s in done), default=0)),
        )


__all__ = [
    "JobRow",
    "JobSpec",
    "RunBoard",
    "RunOwnedError",
    "RunSummary",
    "Throughput",
    "WorkerState",
    "run_state",
]
