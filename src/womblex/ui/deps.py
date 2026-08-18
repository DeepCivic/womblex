"""Where the console reads run state from.

The sidecar reaches run state exactly the way a worker does
(docs/ui-plan.md §2), so it adds no configuration surface of its own: a
local deployment points at an ``output_root`` that is bind-mounted
read-only, and a cloud deployment sets ``WOMBLEX_STORE_URI`` — the same
variable ``womblex-cloud`` already reads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from fastapi import Request

from womblex.store.remote import assert_disjoint_locations
from womblex.ui.settings_store import SavedLocations, locations_path, read_saved_locations

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
    ``store_uri``/``output_root`` (docs/ui-ingest-plan.md). ``None`` means no
    ingest is configured: the Execution Controls cannot dispatch, but
    everything read-only still serves. When both are set they must be
    disjoint of the output store's effective ``runs/`` prefix — enforced at
    construction so a misconfigured deployment fails at start-up.

    ``settings_dir`` is where an operator-saved ingest/output location
    override lives (docs/ui-ingest-plan.md merge 3a) — one ``locations.json``,
    following the ``presets_dir`` pattern exactly. ``None`` disables saving:
    the Resources Console's location cards stay read-only and explain the
    flag, the same degradation preset saving already has. Unlike
    ``presets_dir``, there is no remote-mode fallback — the override file is
    what *names* the store, so it cannot live inside it, in either mode.
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

    A saved ``store_uri`` switches the deployment out of legacy
    ``output_root`` mode entirely — the two stay mutually exclusive, so
    setting one clears the other, exactly as building a fresh ``UISettings``
    with ``--store`` instead of ``--output-root`` would. Going through
    ``dataclasses.replace`` re-runs ``__post_init__``, which is what
    revalidates disjointness against the *effective* pair, not just the
    flag/env values — an overlapping saved override raises here, the same
    ``ValueError`` a bad flag/env pair would.
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

    ``$WOMBLEX_UI_SETTINGS_DIR`` is the env fallback for ``settings_dir`` —
    the writable directory an operator-saved ingest/output override lives in
    (docs/ui-ingest-plan.md merge 3a). When one resolves, the saved override
    is applied and validated here too, so a bad saved override fails at
    start-up exactly like a bad flag/env pair — but the object this function
    returns is still the *pre-overlay* base. The live override is applied
    fresh on every request instead (:func:`get_settings`), which is what
    lets an edit through the Resources Console take effect with no restart
    and makes "reset to default" exactly deleting the override file: baking
    the overlay into this return value would freeze it for the process's
    lifetime instead.
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
        apply_saved_locations(base, read_saved_locations(settings))  # validate; result discarded
    return base


def get_settings(request: Request) -> UISettings:
    """FastAPI dependency: this deployment's settings, re-resolved per request.

    ``app.state.settings`` holds the *base* — flags + env, no saved override
    — fixed for the process's lifetime (see :func:`get_base_settings`). When
    a settings dir is configured, the override file's mtime is checked (one
    cheap ``stat()``) and the overlay only rebuilt when it has changed, so an
    edit made through the Resources Console takes effect on the very next
    request with no restart, at the cost of one syscall on every other
    request. Absent a settings dir there is nothing to overlay, so the base
    settings serve untouched.
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
        state._resolved_settings = apply_saved_locations(base, read_saved_locations(base.settings_dir))
        state._locations_mtime = mtime
    return cast(UISettings, state._resolved_settings)


def get_base_settings(request: Request) -> UISettings:
    """FastAPI dependency: the flags/env settings this process started with.

    No saved override applied, unlike :func:`get_settings` — this is the
    fallback a cleared location resets *to*, and the object
    ``resources.save_locations`` validates a new override against. Every
    other route wants the effective settings (:func:`get_settings`); only the
    location-save route wants the pre-overlay base.
    """
    settings: UISettings = request.app.state.settings
    return settings
