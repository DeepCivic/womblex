"""Named pre-configured pipelines the Pipeline Composer offers as starting points.

A *preset* is a partial :class:`~womblex.config.WomblexConfig` — the stage
toggles and stage settings for a common end-to-end shape — that the composer
form loads so an operator does not hand-assemble it section by section. It is
deliberately *partial*: ``dataset`` and ``paths`` name a real run and have no
sensible default (they are the one thing the operator must still supply), so a
preset never carries them. Merging a preset into the form's current config
therefore only ever flips stage `enabled` flags and their settings, leaving the
run's identity and filesystem paths untouched.

Presets are just data here for the same reason the DAG is: the composer's
validation and YAML download go through the same ``WomblexConfig(**raw)``
construction ``load_config`` uses, so a preset that would not load is a test
failure, not a runtime surprise (``tests/test_ui.py`` builds each one).

`DEFAULT-Isaacus` is the reference Isaacus pipeline:
``extract → chunk → enrich → build_graph → money → done``, with the entity
graph (enrich + the ``graph-refresh`` mention→chunk edge rebuild) and monetary
amounts produced over the one run. It targets PDF and DOCX sources — the two
narrative formats among ``SUPPORTED_EXTENSIONS`` — which is what makes AI
chunking and enrichment meaningful (spreadsheets are structured, not narrative).

The runnable, CLI-first source of truth for that shape is
``configs/default-isaacus.yaml`` — a complete config plus the per-stage command
sequence a dev runs it with (``womblex run`` alone cannot: enrich / build_graph
/ money are per-stage commands, and enrich must precede chunk). The overlay
here mirrors that file's stage toggles and settings; keep the two in step.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from womblex.config import DatasetConfig, PathsConfig, WomblexConfig


@dataclass(frozen=True)
class Preset:
    """A named partial config the composer loads into its form.

    ``config`` is the partial ``WomblexConfig`` overlay (no ``dataset`` /
    ``paths``); ``formats`` is the file extensions the pipeline is intended for.
    ``source`` is ``"builtin"`` (code) or ``"saved"`` (operator-authored) — the
    screen offers delete only on the latter.
    """

    name: str
    description: str
    formats: tuple[str, ...]
    config: dict[str, Any]
    source: str = "builtin"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "formats": list(self.formats),
            "config": self.config,
            "source": self.source,
        }


#: `DEFAULT-Isaacus` — extract → chunk → enrich → build_graph → money → done.
#:
#: Mirrors ``configs/default-isaacus.yaml`` (the runnable, CLI-first source of
#: truth) — edit both together. Settings beyond the bare `enabled` flags are
#: carried so the composer form seeds sensible values, not just toggles:
#:
#: - `chunking.chunking_model = kanon-2-enricher` selects semchunk-4 AI
#:   chunking (boundaries follow the enricher's structure). With `enrich` also
#:   on, `WomblexConfig` auto-enables `enrichment.persist_document` so `chunk`
#:   reuses the enrich Document rather than enriching twice — run enrich BEFORE
#:   chunk (docs/decisions.md). The resulting graph then gains its mention→chunk
#:   edges from the offline `graph-refresh` stage (the "build_graph" step).
#:   `chunk_size = 480` is the Kanon-2 window; `overlap = 0.1` shares 10% across
#:   boundaries so a mention straddling one still lands in a chunk.
#: - `money.enabled` runs the offline amount annotator over the same run, so the
#:   graph and money sidecars land side by side (`graph + money over the one run`);
#:   `default_currency = AUD` matches Australian-government `$` convention.
_DEFAULT_ISAACUS = Preset(
    name="DEFAULT-Isaacus",
    description=(
        "Reference Isaacus pipeline: extract, semantic chunking, Kanon-2 "
        "enrichment, entity-graph edge rebuild (build_graph) and monetary-amount "
        "annotation — graph and money produced over the one run. For PDF and "
        "DOCX sources. Runnable from the CLI as configs/default-isaacus.yaml "
        "(a per-stage sequence: enrich before chunk, then graph-refresh, then money)."
    ),
    formats=(".pdf", ".docx"),
    config={
        "chunking": {
            "enabled": True,
            "chunking_model": "kanon-2-enricher",
            "chunk_size": 480,
            "overlap": 0.1,
        },
        "enrichment": {"enabled": True},
        "money": {"enabled": True, "default_currency": "AUD"},
        # Redaction/PII/embed/link stay at their defaults (off, except redaction
        # flagging) — the preset names the four stages the shape calls for and
        # leaves everything else to the config schema's own defaults.
    },
)

#: Registry, insertion-ordered. Add a preset by defining it above and listing
#: it here; the API and form pick it up with no other change.
PRESETS: dict[str, Preset] = {p.name: p for p in (_DEFAULT_ISAACUS,)}


def list_presets() -> list[dict[str, Any]]:
    """Every built-in preset, in registration order — the composer's preset dropdown."""
    return [p.as_dict() for p in PRESETS.values()]


def get_preset(name: str) -> Preset | None:
    """The built-in preset called *name*, or ``None`` if there is no such preset."""
    return PRESETS.get(name)


# ---------------------------------------------------------------------------
# Operator-saved presets (docs/ui-plan.md merge 9)
# ---------------------------------------------------------------------------
#
# Besides the built-ins above (code), an operator can *save* a composed config
# as a named starting point of their own. Each lands as one JSON file — one file
# per record, like ``store.feedback_output``, so two saves can't lose each
# other. *Where* that file sits is ``womblex.ui.readers``' call, not this
# module's: locally under a writable ``presets_dir``, remotely under the store's
# own ``presets/`` prefix (a sibling of ``runs/`` and ``feedback/``), so a
# store-backed console needs no writable mount. This module owns only the
# *format* — the filename, the record bytes, and parsing one file back into a
# :class:`Preset` — exactly the split ``feedback_output`` keeps.

#: The store prefix operator-saved presets live under in remote mode — a
#: sibling of ``runs/`` and ``feedback/`` in the same bucket, so the container
#: stays ``read_only`` (docs/ui-plan.md merge 9).
PRESETS_DIRNAME = "presets"

_PRESET_SUFFIX = ".preset.json"

#: A preset name becomes a filename, so it is constrained like a run id
#: (``feedback_output.is_safe_run_id``): letters, digits, dot, dash, underscore,
#: no leading dot — enough for ``My-Run_v2`` without admitting ``..`` or a
#: separator.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def is_safe_preset_name(name: str) -> bool:
    """True if *name* is a single, filesystem-safe token (it becomes a filename).

    Refused rather than sanitised, like ``feedback_output.is_safe_run_id``: a
    ``..``, separator or leading dot could climb out of the ``presets_dir``
    join or hide the file.
    """
    return bool(name) and name not in {".", ".."} and _SAFE_NAME.match(name) is not None


def _validate_overlay(config: dict[str, Any]) -> dict[str, Any]:
    """Return *config* minus ``dataset``/``paths``, having asserted it loads.

    A preset is an overlay, so the run-identity sections are dropped and the
    rest is checked against ``WomblexConfig`` (placeholder ``dataset``/``paths``)
    — the same construction ``load_config`` uses, so a preset that would not
    load is refused here rather than 500-ing whoever later picks it.
    """
    overlay = {k: v for k, v in config.items() if k not in {"dataset", "paths"}}
    WomblexConfig(
        dataset=DatasetConfig(name="preset-check"),
        paths=PathsConfig(input_root=Path("."), output_root=Path("."), checkpoint_dir=Path(".")),
        **overlay,
    )
    return overlay


# ---------------------------------------------------------------------------
# Format: filename, record bytes, and parsing one file back into a Preset.
# `readers.py` owns *where* these files sit (local dir vs. store prefix); this
# module owns *what* one is, exactly the split `feedback_output` keeps.
# ---------------------------------------------------------------------------


def preset_filename(name: str) -> str:
    """The single filename a preset called *name* is stored as.

    Assumes *name* is already :func:`is_safe_preset_name`-checked by the caller
    (it becomes this path segment); the saver validates before reaching here.
    """
    return f"{name}{_PRESET_SUFFIX}"


def preset_name_from_filename(filename: str) -> str | None:
    """The preset name a ``*.preset.json`` file encodes, or ``None`` if it is not one.

    ``None`` also for a name that is not :func:`is_safe_preset_name`-safe, so a
    smuggled or hand-placed file never becomes a listable preset.
    """
    if not filename.endswith(_PRESET_SUFFIX):
        return None
    name = filename[: -len(_PRESET_SUFFIX)]
    return name if is_safe_preset_name(name) else None


def build_preset_record(
    *, name: str, description: str, formats: tuple[str, ...], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a save request into ``(record, overlay)`` — the on-disk shape and the overlay.

    ``dataset``/``paths`` are stripped and the overlay validated
    (:func:`_validate_overlay`) — the same ``WomblexConfig(**raw)`` construction
    the built-ins are checked with, so a preset that would not load is refused
    here rather than 500-ing whoever later picks it. Raises ``ValueError`` on an
    unsafe *name*; ``pydantic.ValidationError`` on an overlay that does not load.
    """
    if not is_safe_preset_name(name):
        raise ValueError(f"unsafe preset name: {name!r}")
    overlay = _validate_overlay(config)
    record = {"name": name, "description": description, "formats": list(formats), "config": overlay}
    return record, overlay


def serialise_preset_record(record: dict[str, Any]) -> str:
    """The bytes one preset file carries — one definition for local and store writes."""
    return json.dumps(record, indent=2)


def parse_saved_preset(name: str, raw_bytes: str) -> Preset | None:
    """Parse one preset file's contents into a saved :class:`Preset`, or ``None``.

    ``None`` (skipped, not fatal) when the file will not parse or its overlay
    will not load — one corrupt preset must not blank the dropdown, the same
    skip-and-continue the readers apply to a corrupt sidecar.
    """
    try:
        raw = json.loads(raw_bytes)
        _validate_overlay(raw.get("config", {}))
    except Exception:
        return None
    return Preset(
        name=name,
        description=str(raw.get("description", "")),
        formats=tuple(raw.get("formats", ())),
        config=raw.get("config", {}),
        source="saved",
    )


def merge_saved(saved: dict[str, Preset]) -> list[dict[str, Any]]:
    """Built-in presets, then *saved* (a saved name shadows a built-in)."""
    merged: dict[str, Preset] = dict(PRESETS)
    merged.update(saved)
    return [p.as_dict() for p in merged.values()]


def resolve_one(saved: dict[str, Preset], name: str) -> Preset | None:
    """One preset by name — a *saved* one shadowing a built-in — or ``None``."""
    return saved.get(name) or PRESETS.get(name)


def _validate_presets() -> None:
    """Assert every built-in preset loads — called once at import so it fails fast."""
    for preset in PRESETS.values():
        _validate_overlay(preset.config)


_validate_presets()
