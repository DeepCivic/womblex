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
# as a named starting point of their own. Those land as one JSON file each under
# a writable ``presets_dir`` (``UISettings.presets_dir``, ``None`` when the
# deployment configured none) — one file per record, like
# ``store.feedback_output``, so two saves can't lose each other.

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


def _load_saved_presets(presets_dir: Path) -> dict[str, Preset]:
    """Read every ``*.preset.json`` under *presets_dir*, keyed by filename stem.

    An unreadable or invalid file is skipped, not fatal — one corrupt preset
    must not blank the dropdown (the same skip-and-continue the readers apply).
    """
    saved: dict[str, Preset] = {}
    if not presets_dir.is_dir():
        return saved
    for path in sorted(presets_dir.glob(f"*{_PRESET_SUFFIX}")):
        name = path.name[: -len(_PRESET_SUFFIX)]
        if not is_safe_preset_name(name):
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            _validate_overlay(raw.get("config", {}))
        except Exception:
            continue
        saved[name] = Preset(
            name=name,
            description=str(raw.get("description", "")),
            formats=tuple(raw.get("formats", ())),
            config=raw.get("config", {}),
            source="saved",
        )
    return saved


def list_all_presets(presets_dir: Path | None) -> list[dict[str, Any]]:
    """Built-in presets, then any saved under *presets_dir* (a saved name shadows)."""
    merged: dict[str, Preset] = dict(PRESETS)
    if presets_dir is not None:
        merged.update(_load_saved_presets(presets_dir))
    return [p.as_dict() for p in merged.values()]


def get_any_preset(presets_dir: Path | None, name: str) -> Preset | None:
    """One preset by name — a saved one shadowing a built-in — or ``None``."""
    if presets_dir is not None and is_safe_preset_name(name):
        saved = _load_saved_presets(presets_dir)
        if name in saved:
            return saved[name]
    return PRESETS.get(name)


def save_preset(
    presets_dir: Path,
    *,
    name: str,
    description: str,
    formats: tuple[str, ...],
    config: dict[str, Any],
) -> Preset:
    """Write *config* as a named preset under *presets_dir*; return what was saved.

    ``dataset``/``paths`` are stripped and the overlay validated
    (:func:`_validate_overlay`). Raises ``ValueError`` on an unsafe *name*;
    ``pydantic.ValidationError`` on an overlay that does not load.
    """
    if not is_safe_preset_name(name):
        raise ValueError(f"unsafe preset name: {name!r}")
    overlay = _validate_overlay(config)
    presets_dir.mkdir(parents=True, exist_ok=True)
    record = {"name": name, "description": description, "formats": list(formats), "config": overlay}
    (presets_dir / f"{name}{_PRESET_SUFFIX}").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return Preset(name=name, description=description, formats=formats, config=overlay, source="saved")


def delete_saved_preset(presets_dir: Path, name: str) -> bool:
    """Delete a saved preset; True if removed, False if absent.

    Only ever removes a file under *presets_dir* — a built-in is code, so a name
    matching only a built-in (or an unsafe name) returns False untouched.
    """
    if not is_safe_preset_name(name):
        return False
    path = presets_dir / f"{name}{_PRESET_SUFFIX}"
    if not path.is_file():
        return False
    path.unlink()
    return True


def _validate_presets() -> None:
    """Assert every built-in preset loads — called once at import so it fails fast."""
    for preset in PRESETS.values():
        _validate_overlay(preset.config)


_validate_presets()
