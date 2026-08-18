"""Where the console reads run state from.

The sidecar reaches run state exactly the way a worker does
(docs/ui-plan.md §2), so it adds no configuration surface of its own: a
local deployment points at an ``output_root`` that is bind-mounted
read-only, and a cloud deployment sets ``WOMBLEX_STORE_URI`` — the same
variable ``womblex-cloud`` already reads.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from fastapi import Request

from womblex.store.remote import assert_disjoint_locations
from womblex.ui.settings_store import SavedLocations, locations_path, read_saved_locations

logger = logging.getLogger(__name__)

#: Sentinel for "no mtime checked yet" — distinct from ``None`` (the file is
#: absent) so the very first request always rebuilds the overlay once.
_UNSET = object()


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
    presets (docs/ui-plan.md merge 9) — one JSON file per preset. It applies
    to *local* mode only, and mirrors ``feedback_dir``: remote mode has no
    equivalent field because it always writes presets under the store's own
    ``presets/`` prefix, a sibling of ``runs/`` and ``feedback/`` in the same
    bucket — so a store-backed console needs no writable mount to save presets.
    In local mode, ``None`` disables saving: the built-in presets still serve,
    but ``POST /api/composer/presets`` refuses with 409.

    ``ingest_uri`` names where source documents arrive from, distinct from
    ``store_uri``/``output_root`` (docs/ui-ingest-plan.md). ``None`` means the
    Execution Controls cannot dispatch; everything read-only still serves.
    When both are set they must be disjoint of the output store's effective
    ``runs/`` prefix, enforced here at construction.

    ``settings_dir`` holds an operator-saved location override — one
    ``locations.json``, mirroring ``presets_dir``. ``None`` keeps the
    Resources Console's location cards read-only. Unlike ``presets_dir``
    there is no remote-mode fallback: the override file is what *names* the
    store, so it cannot live inside it.
    """

    output_root: Path | None
    store_uri: str | None
    audit_only: bool = False
    feedback_dir: Path | None = None
    db_dsn: str | None = None
    presets_dir: Path | None = None
    ingest_uri: str | None = None
    settings_dir: Path | None = None

    def __post_init__(self) -> None:
        if bool(self.output_root) == bool(self.store_uri):
            raise ValueError("UISettings needs exactly one of output_root or store_uri")
        if self.ingest_uri and self.store_uri:
            assert_disjoint_locations(self.ingest_uri, self.store_uri)

    @property
    def is_remote(self) -> bool:
        return self.store_uri is not None

    @property
    def presets_writable(self) -> bool:
        """Whether this deployment can save presets at all.

        Remote mode always can (it writes to the store's ``presets/`` prefix,
        like feedback); local mode can only when a writable ``presets_dir`` was
        configured. The composer's save/delete routes 409 when this is false.
        """
        return self.is_remote or self.presets_dir is not None

    @property
    def settings_writable(self) -> bool:
        """Whether this deployment can save an ingest/output location override.

        Needs a configured ``settings_dir`` in *either* mode — remote or
        local — since the override is what names the store itself.
        """
        return self.settings_dir is not None


def apply_saved_locations(base: UISettings, saved: SavedLocations) -> UISettings:
    """*base* with a saved ingest/output override layered on top.

    A saved ``store_uri`` clears ``output_root``, keeping the XOR invariant.
    Going through ``dataclasses.replace`` re-runs ``__post_init__``, so
    disjointness is revalidated against the *effective* pair rather than the
    flag/env values alone.
    """
    output_root = base.output_root
    store_uri = base.store_uri
    if saved.store_uri:
        store_uri = saved.store_uri
        output_root = None
    ingest_uri = saved.ingest_uri or base.ingest_uri
    return replace(base, output_root=output_root, store_uri=store_uri, ingest_uri=ingest_uri)


def resolve_settings(
    output_root: Path | None,
    store_uri: str | None,
    *,
    audit_only: bool = False,
    feedback_dir: Path | None = None,
    db_dsn: str | None = None,
    presets_dir: Path | None = None,
    ingest_uri: str | None = None,
    settings_dir: Path | None = None,
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

    ``$WOMBLEX_INGEST_URI`` is the env fallback for ``ingest_uri`` — the same
    variable ``womblex enqueue``/``worker`` read. Absent means no ingest is
    configured. When resolved alongside a store, ``UISettings`` itself checks
    the two are disjoint and raises ``ValueError`` naming both.

    ``$WOMBLEX_UI_SETTINGS_DIR`` is the env fallback for ``settings_dir``.
    A saved override is validated here — a bad one fails at start-up, naming
    the file to delete — but the settings returned are the *pre-overlay*
    base; :func:`get_settings` applies the live override per request, which
    is what lets an edit take effect with no restart.
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
    ingest = ingest_uri or os.environ.get("WOMBLEX_INGEST_URI")
    settings = settings_dir
    if settings is None and "WOMBLEX_UI_SETTINGS_DIR" in os.environ:
        settings = Path(os.environ["WOMBLEX_UI_SETTINGS_DIR"])
    base = UISettings(
        output_root=root, store_uri=store, audit_only=audit_only,
        feedback_dir=fb_dir, db_dsn=dsn, presets_dir=presets, ingest_uri=ingest,
        settings_dir=settings,
    )
    if settings is not None:
        try:
            apply_saved_locations(base, read_saved_locations(settings))
        except ValueError as e:
            raise ValueError(
                f"{e} — this came from the saved override at "
                f"{locations_path(settings)}; delete that file to reset to the "
                "flag/env defaults."
            ) from e
    return base


def get_settings(request: Request) -> UISettings:
    """FastAPI dependency: this deployment's settings, re-resolved per request.

    ``app.state.settings`` holds the base (flags + env). When a settings dir
    is configured the override file's mtime is checked and the overlay
    rebuilt only when it changed, so an edit takes effect on the next request
    at the cost of one ``stat()`` on every other.
    """
    base: UISettings = request.app.state.settings
    if base.settings_dir is None:
        return base
    state = request.app.state
    path = locations_path(base.settings_dir)
    try:
        mtime: float | None = path.stat().st_mtime
    except OSError:
        mtime = None
    if getattr(state, "_locations_mtime", _UNSET) != mtime:
        saved = read_saved_locations(base.settings_dir)
        try:
            state._resolved_settings = apply_saved_locations(base, saved)
        except ValueError as e:
            # Same skip-and-continue as an unparseable file: an override that
            # no longer validates (the flags it was saved against changed, or
            # it was hand-edited) must degrade to the flag/env defaults, not
            # 500 every request until someone deletes it off the volume.
            logger.warning("deps: saved locations at %s rejected, serving defaults: %s", path, e)
            state._resolved_settings = base
        state._locations_mtime = mtime
    return cast(UISettings, state._resolved_settings)


def get_base_settings(request: Request) -> UISettings:
    """FastAPI dependency: the flags/env settings this process started with.

    The fallback a cleared location resets *to*, and what
    ``resources.save_locations`` validates a new override against. Every
    other route wants :func:`get_settings`.
    """
    settings: UISettings = request.app.state.settings
    return settings
