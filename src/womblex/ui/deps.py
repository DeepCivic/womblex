"""Where the console reads run state from.

The sidecar reaches run state exactly the way a worker does
(docs/ui-plan.md §2), so it adds no configuration surface of its own: a
local deployment points at an ``output_root`` that is bind-mounted
read-only, and a cloud deployment sets ``WOMBLEX_STORE_URI`` — the same
variable ``womblex-cloud`` already reads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import Request


@dataclass(frozen=True)
class UISettings:
    """Where the console reads run state from. Exactly one of the two is set.

    ``db_dsn`` is the optional job queue the dashboard reads. It is
    orthogonal to the run source rather than paired with it: a queue is
    present whenever one is configured (env or argument) and absent
    otherwise, and a deployment with no queue falls back to the per-stage
    checkpoints inside the run itself (docs/ui-plan.md §2).

    ``feedback_dir`` is local-mode only — the report action's writable
    surface (docs/ui-plan.md §4). ``None`` means the default,
    ``<output_root>/feedback``; an explicit value is the escape hatch for a
    deployment that mounts ``output_root`` read-only and needs feedback to
    land somewhere else entirely. Remote mode has no equivalent field: it
    always writes under the store's own ``feedback/`` prefix, a sibling of
    ``runs/`` in the same bucket.

    ``presets_dir`` is where the Pipeline Composer *saves* operator-authored
    presets (docs/ui-plan.md merge 9) — one JSON file per preset. ``None``
    disables saving: the built-in presets still serve, but
    ``POST /api/composer/presets`` refuses with 409.
    """

    output_root: Path | None
    store_uri: str | None
    audit_only: bool = False
    feedback_dir: Path | None = None
    db_dsn: str | None = None
    presets_dir: Path | None = None

    def __post_init__(self) -> None:
        if bool(self.output_root) == bool(self.store_uri):
            raise ValueError("UISettings needs exactly one of output_root or store_uri")

    @property
    def is_remote(self) -> bool:
        return self.store_uri is not None


def resolve_settings(
    output_root: Path | None,
    store_uri: str | None,
    *,
    audit_only: bool = False,
    feedback_dir: Path | None = None,
    db_dsn: str | None = None,
    presets_dir: Path | None = None,
) -> UISettings:
    """Resolve settings from explicit arguments, falling back to env vars.

    ``$WOMBLEX_UI_OUTPUT_ROOT`` names a local run root;
    ``$WOMBLEX_STORE_URI`` names an object store. Raises ``ValueError`` when
    both or neither resolve — the console reads exactly one run source, and
    silently preferring one over the other would hide a misconfigured
    deployment behind an empty run list. ``$WOMBLEX_UI_FEEDBACK_DIR`` is the
    env fallback for ``feedback_dir``, read only when the explicit argument
    is absent.

    ``$WOMBLEX_UI_PRESETS_DIR`` is the env fallback for ``presets_dir`` — the
    directory the composer saves operator-authored presets into. Absent means
    saving is disabled (built-in presets still serve).

    ``$WOMBLEX_DB_DSN`` / ``$DATABASE_URL`` name the job queue, the same
    pair ``womblex worker`` reads. Absent is not an error: the dashboard
    falls back to checkpoints.
    """
    root = output_root
    if root is None and "WOMBLEX_UI_OUTPUT_ROOT" in os.environ:
        root = Path(os.environ["WOMBLEX_UI_OUTPUT_ROOT"])
    store = store_uri or os.environ.get("WOMBLEX_STORE_URI")
    fb_dir = feedback_dir
    if fb_dir is None and "WOMBLEX_UI_FEEDBACK_DIR" in os.environ:
        fb_dir = Path(os.environ["WOMBLEX_UI_FEEDBACK_DIR"])
    if root and store:
        raise ValueError("pass only one of --output-root / --store (or their env vars)")
    if not root and not store:
        raise ValueError(
            "no run source: pass --output-root or --store "
            "(or set $WOMBLEX_UI_OUTPUT_ROOT / $WOMBLEX_STORE_URI)"
        )
    dsn = db_dsn or os.environ.get("WOMBLEX_DB_DSN") or os.environ.get("DATABASE_URL")
    presets = presets_dir
    if presets is None and "WOMBLEX_UI_PRESETS_DIR" in os.environ:
        presets = Path(os.environ["WOMBLEX_UI_PRESETS_DIR"])
    return UISettings(
        output_root=root, store_uri=store, audit_only=audit_only,
        feedback_dir=fb_dir, db_dsn=dsn, presets_dir=presets,
    )


def get_settings(request: Request) -> UISettings:
    """FastAPI dependency: the app-wide settings resolved at startup."""
    settings: UISettings = request.app.state.settings
    return settings
