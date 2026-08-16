"""UI CLI subcommand: ``womblex ui`` (console sidecar).

Binds to loopback by default. The console has no authentication by design
(docs/ui-plan.md §6) — it is kept undiscoverable at the network layer
instead — so exposing it beyond localhost is an explicit ``--host`` choice
the operator makes alongside whatever fronts it.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command

logger = logging.getLogger("womblex")


def _register_ui(p: argparse.ArgumentParser) -> None:
    target = p.add_mutually_exclusive_group()
    target.add_argument(
        "--output-root", type=Path, default=None,
        help="Local run root — the parent of run-*/ dirs (or $WOMBLEX_UI_OUTPUT_ROOT).",
    )
    target.add_argument(
        "--store", default=None,
        help="Object-store base URI containing runs/ (or $WOMBLEX_STORE_URI). "
             "Needs the 'cloud' extra.",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind address. Default: 127.0.0.1.")
    p.add_argument("--port", type=int, default=8080, help="Bind port. Default: 8080.")
    p.add_argument(
        "--feedback-dir", type=Path, default=None,
        help="Local mode only: writable dir for report-action files "
             "(or $WOMBLEX_UI_FEEDBACK_DIR). Defaults to <output-root>/feedback. "
             "Remote mode always writes under the store's own feedback/ prefix.",
    )
    p.add_argument(
        "--allow-execute", action="store_true",
        help="Reserved for the execution endpoints (docs/ui-plan.md merge 11). "
             "None exist yet, so today every deployment is audit-only either way.",
    )


def cmd_ui(args: argparse.Namespace) -> int:
    """Serve the read-only console sidecar over the resolved run source."""
    try:
        import uvicorn

        from womblex.ui.app import create_app
        from womblex.ui.deps import resolve_settings
    except ImportError:
        logger.error("`womblex ui` requires the 'ui' extra. Install with: pip install womblex[ui]")
        return 1

    try:
        settings = resolve_settings(
            args.output_root, args.store,
            allow_execute=args.allow_execute, feedback_dir=args.feedback_dir,
        )
    except ValueError as e:
        logger.error("%s", e)
        return 1

    app = create_app(
        output_root=settings.output_root,
        store_uri=settings.store_uri,
        allow_execute=settings.allow_execute,
        feedback_dir=settings.feedback_dir,
    )
    source = settings.store_uri or settings.output_root
    logger.info("womblex ui: serving %s on %s:%d", source, args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


COMMANDS = [
    Command("ui", "Serve the read-only console sidecar over a run's artefacts", _register_ui, cmd_ui),
]
