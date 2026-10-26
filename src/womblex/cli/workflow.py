"""Workflow CLI subcommands, each over a workflow file (`cloud.workflow_file`): ``run-workflow``,
``validate``, ``plan``, ``dispatch-workflow``, and ``schedule-workflow``, the one that stores a copy."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

from womblex.cli._shared import Command, add_models_check_argument

logger = logging.getLogger("womblex")


def _add_workflow_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("workflow", type=Path, help="Workflow file (YAML)")
    p.add_argument("--presets-dir", type=Path, default=None,
                   help="Directory of operator-saved presets (or $WOMBLEX_UI_PRESETS_DIR). "
                        "Built-in presets always resolve.")
    add_models_check_argument(p)


def _register_run_workflow(p: argparse.ArgumentParser) -> None:
    _add_workflow_arguments(p)
    p.add_argument("--shards", type=Path, required=True,
                   help="Local shard dir the steps run over (extraction has already produced it).")
    p.add_argument("--ingest", default=None,
                   help="`layout` step only: a local corpus root, when it is not where the manifest recorded it.")
    p.add_argument("--force", action="store_true",
                   help="Re-run bases whose declared outputs already exist")


def _register_validate(p: argparse.ArgumentParser) -> None:
    _add_workflow_arguments(p)
    p.add_argument("--shards", type=Path, default=None,
                   help="Local shard dir: check each step's inputs against what is on disk. "
                        "Without it only the order among the steps is checked.")
    p.add_argument("--json", action="store_true", help="Print the report as one JSON object")


def _register_plan(p: argparse.ArgumentParser) -> None:
    _add_workflow_arguments(p)
    p.add_argument("--shards", type=Path, required=True,
                   help="Local shard dir the steps would run over.")
    p.add_argument("--assume-price", action="append", default=[], metavar="STAGE=USD_PER_MILLION",
                   help="An assumed rate for `enrich`, `embed` or `chunk` (AI chunking) tokens (repeatable). Womblex holds no "
                        "rate card: any cost shown is this assumption times the exact token count.")
    p.add_argument("--json", action="store_true", help="Print the plan as one JSON object")


def _register_dispatch(p: argparse.ArgumentParser) -> None:
    p.add_argument("workflow", type=Path, help="Workflow file (YAML) to dispatch")
    p.add_argument("--run-id", required=True, help="Run whose extraction the stages follow")
    p.add_argument("--firing", default=None,
                   help="A scheduled firing whose stages failed to dispatch: run the steps over its batches only")
    _dispatch_options(p)


def _register_rerun(p: argparse.ArgumentParser) -> None:
    p.add_argument("workflow", type=Path, help="Workflow file (YAML) whose steps to run over its whole collection")
    _dispatch_options(p)


def _dispatch_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", default=None,
                   help="Object-store URI the run's shards are in (or $WOMBLEX_STORE_URI)")
    p.add_argument("--output-prefix", default=None, help="Default: runs/<run-id>")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="Run every check, write no queue row")
    p.add_argument("--dsn", default=None,
                   help="DBOS system database holding the queue "
                        "(or $WOMBLEX_DB_DSN / $DATABASE_URL). Default: the local SQLite file.")
    p.add_argument("--presets-dir", type=Path, default=None,
                   help="Directory of operator-saved presets (or $WOMBLEX_UI_PRESETS_DIR).")


def _register_schedule(p: argparse.ArgumentParser) -> None:
    p.add_argument("workflow", help="Workflow file (YAML) to schedule; with --remove or --status, "
                                    "its file or its name")
    p.add_argument("--dsn", default=None,
                   help="DBOS system database (or $WOMBLEX_DB_DSN / $DATABASE_URL). Default: the local SQLite file.")
    p.add_argument("--store", default=None,
                   help="Object-store URI the firings write to (or $WOMBLEX_STORE_URI). The firings read the "
                        "workflow's own input store (input_binding.store).")
    p.add_argument("--presets-dir", type=Path, default=None,
                   help="Directory of operator-saved presets (or $WOMBLEX_UI_PRESETS_DIR).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--remove", action="store_true",
                      help="Unschedule every trigger of the workflow; its stored copy stays (`remove-workflow` deletes it)")
    mode.add_argument("--status", action="store_true", help="List its schedules; change nothing")


def _load_workflow(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    """The workflow in the file *args* names, or `None` (the reason is logged and kept on *args*)."""
    from womblex.cloud.workflow_file import WorkflowFileError, load_workflow_file

    try:
        return load_workflow_file(Path(args.workflow))
    except WorkflowFileError as e:
        logger.error("%s", e)
        args.file_error = e
        return None


def _file_refusal(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    """The refusal a workflow file that did not load stands for, with its stable code."""
    from womblex.cloud.workflow_runner import RefusalCode, WorkflowRefusedError

    code = RefusalCode.SPEC_INVALID if args.file_error.readable else RefusalCode.WORKFLOW_UNAVAILABLE
    return WorkflowRefusedError(str(args.file_error), code=code)


def _preset_resolver(args: argparse.Namespace) -> Callable[[str], object]:
    from womblex.cloud.workflow_runner import preset_resolver

    presets_dir = args.presets_dir or (Path(d) if (d := os.environ.get("WOMBLEX_UI_PRESETS_DIR")) else None)
    return preset_resolver(presets_dir)


def cmd_run_workflow(args: argparse.Namespace) -> int:
    """Run a workflow file's steps in order; exit 1 if it was refused or any step failed."""
    from womblex.cloud.workflow_runner import WorkflowRefusedError, run_workflow_local

    if not args.shards.is_dir():
        logger.error("--shards path is not a directory: %s", args.shards)
        return 1
    stored = _load_workflow(args)
    if stored is None:
        return 1
    try:
        result = run_workflow_local(
            stored, args.shards,
            resolve_preset=_preset_resolver(args),  # type: ignore[arg-type]
            source_root=args.ingest, force=args.force, models_check=args.models_check,
        )
    except WorkflowRefusedError as e:
        logger.error("[%s] %s", e.code.value, e)
        for issue in e.issues:
            logger.error("[%s] %s", issue.code.value, issue.message)
        return 1
    result.log()
    return result.exit_code


def cmd_validate(args: argparse.Namespace) -> int:
    """Run a workflow's pre-run checks without running it; exit 1 if it would be refused."""
    from womblex.cloud.workflow_validate import validate_workflow

    if args.shards is not None and not args.shards.is_dir():
        logger.error("--shards path is not a directory: %s", args.shards)
        return 1
    stored = _load_workflow(args)
    if stored is None:
        if args.json:  # a script reading stdout gets a report, not nothing
            refusal = _file_refusal(args)
            print(json.dumps({
                "workflow_id": None, "valid": False, "code": refusal.code.value, "message": str(refusal),
            }, indent=2))
        return 1
    report = validate_workflow(
        stored, resolve_preset=_preset_resolver(args),  # type: ignore[arg-type]
        shard_dir=args.shards, models_check=args.models_check,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        head = f"workflow {report.workflow_id} ({report.digest}): " + " -> ".join(report.stages)
        print(head)
        if report.valid:
            print("valid: it would not be refused")
        else:
            # The message joins the issues and steps listed below, so print it only when there are none.
            print(f"refused [{report.code}]" + ("" if report.issues or report.steps else f": {report.message}"))
            for issue in report.issues:
                print(f"  [{issue['code']}] {issue['message']}")
            for step in report.steps:
                print(f"  [{report.code}] {step['stage']}: {step['message']}")
        for w in report.warnings:
            print(f"warning: {w}")
        if report.not_acted_on:
            print(f"stored but not acted on: {', '.join(report.not_acted_on)}")
        if not report.shards_checked:
            print("note: no --shards, so inputs on disk were not checked")
    return 0 if report.valid else 1


def _parse_prices(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        stage, _, value = item.partition("=")
        if stage not in ("enrich", "embed", "chunk"):
            raise ValueError(f"--assume-price takes enrich=…, embed=… or chunk=…, got {item!r}")
        out[stage] = float(value)
    return out


def cmd_plan(args: argparse.Namespace) -> int:
    """Say what a workflow would run and spend, calling no paid API; exit 1 if it would be refused."""
    from womblex.cloud.workflow_plan import plan_workflow
    from womblex.cloud.workflow_runner import WorkflowRefusedError

    if not args.shards.is_dir():
        logger.error("--shards path is not a directory: %s", args.shards)
        return 1
    try:
        prices = _parse_prices(args.assume_price)
    except ValueError as e:
        logger.error("%s", e)
        return 1
    stored = _load_workflow(args)
    try:
        if stored is None:
            raise _file_refusal(args)
        plan = plan_workflow(
            stored, args.shards, resolve_preset=_preset_resolver(args), prices=prices,  # type: ignore[arg-type]
        )
    except WorkflowRefusedError as e:
        # The message joins the issues listed after it, so say it only when there are none.
        logger.error("refused [%s]%s", e.code.value, "" if e.issues else f": {e}")
        for issue in e.issues:
            logger.error("[%s] %s", issue.code.value, issue.message)
        if args.json:  # a script reading stdout gets an answer, as `validate --json` does
            print(json.dumps({"workflow_id": stored.id if stored else None, "refused": True, "code": e.code.value,
                              "message": str(e), "issues": [i.as_dict() for i in e.issues]}, indent=2))
        return 1
    if args.json:
        print(json.dumps(plan.as_dict(), indent=2))
        return 0
    print(f"workflow {plan.workflow_id} ({plan.digest}) over {plan.shard_dir}")
    for s in plan.steps:
        print(f"  {s.action:<4} {s.stage} [{'paid' if s.paid else 'free, local'}]"
              + (f" — {s.reason}" if s.reason else ""))
        if (t := s.tokens) is not None:
            if t.basis in ("pending", "unavailable"):
                print(f"       tokens: {t.basis}: {t.note}")
            elif t.tokens is None:
                print(f"       {t.texts:,} text(s) in {t.requests:,} request(s) [{t.basis}]; tokens not counted: {t.note}")
            else:
                print(f"       tokens: {t.tokens:,} over {t.texts:,} text(s) in {t.requests:,} "
                      f"request(s) [{t.basis}, the provider's own token counter]")
            if t.assumed_cost is not None:
                print(f"       cost: ${t.assumed_cost:,.2f} — ASSUMPTION: ${t.assumed_price_per_million}/M "
                      "tokens supplied by --assume-price; not a published rate")
    for w in plan.warnings:
        print(f"warning: {w}")
    if plan.not_acted_on:
        print(f"stored but not acted on: {', '.join(plan.not_acted_on)}")
    print("no paid API was called; `womblex validate` checks models and Isaacus availability")
    return 0


def cmd_dispatch_workflow(args: argparse.Namespace) -> int:
    """Queue a workflow file's steps for the fleet; exit 1 if it is refused before any row."""
    from womblex.cloud.workflow_dispatch import dispatch_workflow
    from womblex.cloud.workflow_runner import WorkflowRefusedError

    stored = _load_workflow(args)
    if stored is None:
        return 1
    dsn = args.dsn or os.environ.get("WOMBLEX_DB_DSN") or os.environ.get("DATABASE_URL")
    try:
        result = dispatch_workflow(
            stored, resolve_preset=_preset_resolver(args),  # type: ignore[arg-type]
            run_id=args.run_id, store_uri=args.store or os.environ.get("WOMBLEX_STORE_URI"),
            dsn=dsn, output_prefix=args.output_prefix, max_attempts=args.max_attempts,
            dry_run=args.dry_run, firing=getattr(args, "firing", None),
        )
    except WorkflowRefusedError as e:
        logger.error("[%s] %s", e.code.value, e)
        for issue in e.issues:
            logger.error("[%s] %s", issue.code.value, issue.message)
        return 1
    logger.info(
        "workflow %s (%s): %s -> %s%s", result.workflow_id, result.digest,
        " -> ".join(result.stages), result.shard_prefix,
        " (dry run: nothing queued)" if result.dry_run else f", {result.newly_enqueued} newly dispatched",
    )
    return 0


def cmd_rerun_collection(args: argparse.Namespace) -> int:
    """Run a workflow file's current steps over its whole collection: `dispatch-workflow` outside a firing.

    The run is the workflow's collection (else its name). A unit whose version, stage config and
    inputs are unchanged since it was published is skipped; an in-place stage, `graph-refresh` and
    an output with no receipt run again over every batch.
    """
    if (stored := _load_workflow(args)) is None:
        return 1
    args.run_id = stored.run_id
    return cmd_dispatch_workflow(args)


def _check_firing(args: argparse.Namespace, stored, dsn: str | None) -> bool:  # type: ignore[no-untyped-def]
    """Run the checks a firing would make, so a workflow it would refuse is not scheduled."""
    from womblex.cloud.workflow_dispatch import check_firing
    from womblex.cloud.workflow_runner import WorkflowRefusedError
    from womblex.store.remote import RemoteStore

    store_uri = args.store or os.environ.get("WOMBLEX_STORE_URI")
    if not store_uri:
        logger.error("no store URI to check the firing against (pass --store or set WOMBLEX_STORE_URI)")
        return False
    try:
        check_firing(
            stored, run_id=stored.run_id,
            resolve_preset=_preset_resolver(args),  # type: ignore[arg-type]
            store=RemoteStore.from_uri(store_uri), dsn=dsn,
        )
    except WorkflowRefusedError as e:
        logger.error("not scheduled, a firing would refuse it: [%s] %s", e.code.value, e)
        return False
    return True


def _workflow_name(arg: str) -> str | None:
    """The workflow *arg* names: its file's `name`, else *arg* itself as a name (`None` if neither)."""
    from womblex.cloud.workflow_file import WorkflowFileError, workflow_file_name
    from womblex.cloud.workflow_store import is_safe_workflow_id

    if Path(arg).is_file():
        try:
            return workflow_file_name(Path(arg))
        except WorkflowFileError as e:
            logger.error("%s", e)
            return None
    if is_safe_workflow_id(arg):
        return arg
    logger.error("%s is neither a workflow file nor a workflow name", arg)
    return None


def cmd_schedule_workflow(args: argparse.Namespace) -> int:
    """Store a workflow's copy and make the board's schedules equal its triggers (workers fire them)."""
    from womblex.cloud import dbos_app
    from womblex.cloud.workflow_schedules import list_schedules, remove_schedules, schedule_workflow
    from womblex.cloud.workflow_store import WorkflowStore, ensure_workflow_schema

    dsn = args.dsn or os.environ.get("WOMBLEX_DB_DSN") or os.environ.get("DATABASE_URL")
    # --status and --remove take a name too: a workflow's file may be gone while its schedules stay.
    if args.status or args.remove:
        stored = None
        if (name := _workflow_name(args.workflow)) is None:
            return 1
    elif (stored := _load_workflow(args)) is None:
        return 1
    else:
        name = stored.id
    if stored is not None and stored.spec.triggers and not _check_firing(args, stored, dsn):
        return 1
    if args.status and not dsn and not dbos_app.local_database_path().exists():
        return 0  # nothing scheduled, and --status creates no database to say so
    client = dbos_app.open_client(dsn) if args.status else None
    if args.status:
        for s in list_schedules(client, name):  # type: ignore[arg-type]
            print(f"{s['schedule_name']}  {s['schedule']} {s.get('cron_timezone') or 'UTC'}  "
                  f"{s['status']}  last fired {s.get('last_fired_at') or 'never'}")
        return 0
    dbos_app.ensure_schema(dsn)
    ensure_workflow_schema(dsn)
    client = dbos_app.open_client(dsn)
    with WorkflowStore(dsn) as wf_store:
        if args.remove:
            print(f"unscheduled {remove_schedules(client, name)} trigger(s) of {name}")
            return 0
        assert stored is not None  # neither --status nor --remove
        names = schedule_workflow(client, wf_store, stored)
    print(f"{stored.id} ({stored.digest}): {len(names)} trigger(s) scheduled"
          + (f" ({', '.join(names)}); its copy is stored for the firings" if names else "; nothing stored"))
    return 0


def _register_collection_commands(p: argparse.ArgumentParser) -> None:
    p.add_argument("workflow", help="The workflow's file or its name")
    p.add_argument("--dsn", default=None,
                   help="DBOS system database (or $WOMBLEX_DB_DSN / $DATABASE_URL). Default: the local SQLite file.")
    p.add_argument("--store", default=None,
                   help="Object-store URI the firing writes to (or $WOMBLEX_STORE_URI)")
    p.add_argument("--presets-dir", type=Path, default=None,
                   help="Directory of operator-saved presets (or $WOMBLEX_UI_PRESETS_DIR).")


def cmd_add_to_collection(args: argparse.Namespace) -> int:
    """Fire a stored workflow once, now: add its input prefix's new documents to its collection.

    The stored copy is what fires (`schedule-workflow` stores it), checked as a schedule's firing
    is before any row; a workflow not stored, or one a firing would refuse, queues nothing.
    """
    import sqlalchemy.exc

    from womblex.cloud import dbos_app
    from womblex.cloud.jobs import schema_missing
    from womblex.cloud.workflow_schedules import add_to_collection
    from womblex.cloud.workflow_store import WorkflowStore

    dsn = args.dsn or os.environ.get("WOMBLEX_DB_DSN") or os.environ.get("DATABASE_URL")
    if (name := _workflow_name(args.workflow)) is None:
        return 1
    if not dbos_app.local_database_path().exists() and not dsn:
        logger.error("workflow %s is not stored: run `womblex schedule-workflow` on its file first", name)
        return 1
    try:
        with WorkflowStore(dsn) as wf_store:
            stored = wf_store.get(name)
    except sqlalchemy.exc.DBAPIError as e:
        if not schema_missing(e):
            raise
        stored = None  # a database no `schedule-workflow` has written to holds no workflow
    if stored is None:
        logger.error("workflow %s is not stored: run `womblex schedule-workflow` on its file first", name)
        return 1
    if Path(args.workflow).is_file() and (loaded := _load_workflow(args)) and loaded.digest != stored.digest:
        logger.warning("%s differs from the stored copy of %s; the firing uses the stored copy "
                       "(`womblex schedule-workflow` stores the file's)", args.workflow, name)
    if not _check_firing(args, stored, dsn):
        return 1
    print(add_to_collection(dbos_app.open_client(dsn), stored))
    return 0


def cmd_remove_workflow(args: argparse.Namespace) -> int:
    """Remove a workflow's schedules and its stored copy. The run it fed, and its shards, are untouched."""
    from womblex.cloud import dbos_app
    from womblex.cloud.workflow_schedules import unschedule_workflow
    from womblex.cloud.workflow_store import WorkflowStore, ensure_workflow_schema

    dsn = args.dsn or os.environ.get("WOMBLEX_DB_DSN") or os.environ.get("DATABASE_URL")
    if (name := _workflow_name(args.workflow)) is None:
        return 1
    dbos_app.ensure_schema(dsn)
    ensure_workflow_schema(dsn)
    with WorkflowStore(dsn) as wf_store:
        print(f"removed {name}: {unschedule_workflow(dbos_app.open_client(dsn), wf_store, name)} schedule(s) "
              "and its stored copy")
    return 0


def _register_deliver(p: argparse.ArgumentParser) -> None:
    p.add_argument("workflow", type=Path, help="Workflow file (YAML) naming the destinations")
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--shards", type=Path, help="A local run's documents/ directory")
    where.add_argument("--run-id", help="A fleet run in --store")
    p.add_argument("--store", default=None, help="Object-store URI the run is in (or $WOMBLEX_STORE_URI)")
    p.add_argument("--output-prefix", default=None, help="Default: runs/<run-id>")
    p.add_argument("--dsn", default=None,
                   help="DBOS system database, for the run record's file checksums "
                        "(or $WOMBLEX_DB_DSN / $DATABASE_URL). Default: the local SQLite file.")


def cmd_deliver(args: argparse.Namespace) -> int:
    """Deliver a finished run to its workflow's destinations again, replacing what was delivered.

    Refused before anything is sent when the workflow names no destination or one fails its check.
    Exit 1 if any delivery failed; each is recorded on the run either way.
    """
    from womblex.cloud.delivery import check_destinations, deliver, record_deliveries
    from womblex.cloud.finalize import finalize_from_board
    from womblex.cloud.workflow_runner import deliver_local
    from womblex.store.remote import RemoteStore

    stored = _load_workflow(args)
    if stored is None:
        return 1
    if not stored.spec.destinations:
        logger.error("workflow %s names no destinations", stored.id)
        return 1
    if problems := check_destinations(stored.spec.destinations):
        for name, problem in problems:
            logger.error("[destination_unavailable] destination %s: %s", name, problem)
        return 1
    if args.shards is not None:
        if args.shards.name != "documents" or not args.shards.is_dir():
            logger.error("--shards must be a run's documents/ directory: %s", args.shards)
            return 1
        outcomes = deliver_local(stored, args.shards)
    else:
        store_uri = args.store or os.environ.get("WOMBLEX_STORE_URI")
        if not store_uri:
            logger.error("No store URI (pass --store or set WOMBLEX_STORE_URI)")
            return 1
        prefix = (args.output_prefix or f"runs/{args.run_id}").strip("/")
        store = RemoteStore.from_uri(store_uri)
        dsn = args.dsn or os.environ.get("WOMBLEX_DB_DSN") or os.environ.get("DATABASE_URL")
        # As the coordinator does: the manifest first, so the bundle holds one covering a retry
        # made since the run finished, then again so the run record lists these deliveries.
        finalize_from_board(store, args.run_id, prefix, dsn)
        run = store.sub(prefix)
        outcomes = deliver(stored, run, args.run_id)
        record_deliveries(run, outcomes)
        finalize_from_board(store, args.run_id, prefix, dsn)
    for o in outcomes:
        logger.info("deliver %s: %s%s", o.destination, o.status, f" — {o.error}" if o.error else "")
    return 1 if any(o.status != "delivered" for o in outcomes) else 0


COMMANDS = [
    Command("run-workflow", "Run a workflow file's steps over a local shard directory",
            _register_run_workflow, cmd_run_workflow),
    Command("dispatch-workflow", "Queue a workflow file's steps for the fleet (object-store run)",
            _register_dispatch, cmd_dispatch_workflow),
    Command("rerun-collection", "Run a workflow file's steps over its whole collection (receipts skip unchanged units)",
            _register_rerun, cmd_rerun_collection),
    Command("schedule-workflow", "Store a workflow file's copy and schedule its cron triggers (workers fire them)",
            _register_schedule, cmd_schedule_workflow),
    Command("add-to-collection", "Fire a stored workflow once now, adding its input prefix's new documents to its collection",
            _register_collection_commands, cmd_add_to_collection),
    Command("remove-workflow", "Remove a workflow's schedules and stored copy (its run and shards stay)",
            _register_collection_commands, cmd_remove_workflow),
    Command("plan", "Say what a workflow file would run and spend, calling no paid API",
            _register_plan, cmd_plan),
    Command("validate", "Check a workflow file would run, without running it",
            _register_validate, cmd_validate),
    Command("deliver", "Deliver a finished run to its workflow file's destinations again",
            _register_deliver, cmd_deliver),
]
