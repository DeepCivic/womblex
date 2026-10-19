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
                   help="An assumed rate for `enrich` or `embed` tokens (repeatable). Womblex holds no "
                        "rate card: any cost shown is this assumption times the exact token count.")
    p.add_argument("--json", action="store_true", help="Print the plan as one JSON object")


def _register_dispatch(p: argparse.ArgumentParser) -> None:
    p.add_argument("workflow", type=Path, help="Workflow file (YAML) to dispatch")
    p.add_argument("--run-id", required=True, help="Run whose extraction the stages follow")
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
                      help="Unschedule every trigger of the workflow and delete its stored copy")
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
    """Run a stored workflow's steps in order; exit 1 if it was refused or any step failed."""
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
        if stage not in ("enrich", "embed"):
            raise ValueError(f"--assume-price takes enrich=… or embed=…, got {item!r}")
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
        print(f"  {s.action:<4} {s.stage}" + (f" — {s.reason}" if s.reason else ""))
        if (t := s.tokens) is not None:
            if t.basis in ("pending", "unavailable"):
                print(f"       tokens: {t.basis}: {t.note}")
            else:
                print(f"       tokens: {t.tokens:,} over {t.texts:,} text(s) in {t.requests:,} "
                      f"request(s) [{t.basis}, offline kanon-2 tokeniser]")
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
    """Queue a stored workflow's steps for the fleet; exit 1 if it is refused before any row."""
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
            dry_run=args.dry_run,
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


def _check_firing(args: argparse.Namespace, stored, dsn: str | None) -> bool:  # type: ignore[no-untyped-def]
    """Run the checks a firing would make, so a workflow it would refuse is not scheduled."""
    from datetime import UTC, datetime

    from womblex.cloud.workflow_dispatch import check_firing
    from womblex.cloud.workflow_runner import WorkflowRefusedError
    from womblex.cloud.workflows import scheduled_run_id
    from womblex.store.remote import RemoteStore

    store_uri = args.store or os.environ.get("WOMBLEX_STORE_URI")
    if not store_uri:
        logger.error("no store URI to check the firing against (pass --store or set WOMBLEX_STORE_URI)")
        return False
    try:
        check_firing(
            stored, run_id=scheduled_run_id(stored.id, datetime.now(UTC)),
            resolve_preset=_preset_resolver(args),  # type: ignore[arg-type]
            store=RemoteStore.from_uri(store_uri), dsn=dsn,
        )
    except WorkflowRefusedError as e:
        logger.error("not scheduled, a firing would refuse it: [%s] %s", e.code.value, e)
        return False
    return True


def _workflow_name(arg: str) -> str | None:
    """The workflow *arg* names: its file's `name`, else *arg* itself as a name (`None` if neither)."""
    from womblex.cloud.workflow_store import is_safe_workflow_id

    if Path(arg).is_file():
        stored = _load_workflow(argparse.Namespace(workflow=arg))
        return stored.id if stored else None
    if is_safe_workflow_id(arg):
        return arg
    logger.error("%s is neither a workflow file nor a workflow name", arg)
    return None


def cmd_schedule_workflow(args: argparse.Namespace) -> int:
    """Store a workflow's copy and make the board's schedules equal its triggers (workers fire them)."""
    from womblex.cloud import dbos_app
    from womblex.cloud.workflow_schedules import list_schedules, schedule_workflow, unschedule_workflow
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
            print(f"unscheduled {unschedule_workflow(client, wf_store, name)} trigger(s) of {name}")
            return 0
        assert stored is not None  # neither --status nor --remove
        names = schedule_workflow(client, wf_store, stored)
    print(f"{stored.id} ({stored.digest}): {len(names)} trigger(s) scheduled"
          + (f" ({', '.join(names)}); its copy is stored for the firings" if names else "; nothing stored"))
    return 0


COMMANDS = [
    Command("run-workflow", "Run a workflow file's steps over a local shard directory",
            _register_run_workflow, cmd_run_workflow),
    Command("dispatch-workflow", "Queue a workflow file's steps for the fleet (object-store run)",
            _register_dispatch, cmd_dispatch_workflow),
    Command("schedule-workflow", "Store a workflow file's copy and schedule its cron triggers (workers fire them)",
            _register_schedule, cmd_schedule_workflow),
    Command("plan", "Say what a workflow file would run and spend, calling no paid API",
            _register_plan, cmd_plan),
    Command("validate", "Check a workflow file would run, without running it",
            _register_validate, cmd_validate),
]
