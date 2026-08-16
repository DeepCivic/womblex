"""FastAPI app factory for the console sidecar.

Serves the read API only in this merge — the built SPA (docs/ui-plan.md
merges 3-4) mounts as static files once ``ui/`` exists to build one from.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from womblex.ui.deps import UISettings
from womblex.ui.routes import runs


def create_app(
    *,
    output_root: Path | None = None,
    store_uri: str | None = None,
    allow_execute: bool = False,
) -> FastAPI:
    """Build the console app, bound to one run source for its lifetime.

    Exactly one of ``output_root`` / ``store_uri`` must be given;
    :class:`~womblex.ui.deps.UISettings` enforces that and raises
    ``ValueError`` otherwise. Binding at construction rather than per request
    is what keeps the run source out of the URL space — no endpoint can be
    talked into reading a directory the operator did not mount.
    """
    settings = UISettings(output_root=output_root, store_uri=store_uri, allow_execute=allow_execute)
    app = FastAPI(title="Womblex Console")
    app.state.settings = settings
    app.include_router(runs.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "mode": "remote" if settings.is_remote else "local"}

    return app
