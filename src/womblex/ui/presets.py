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

from dataclasses import dataclass
from typing import Any

from womblex.config import DatasetConfig, PathsConfig, WomblexConfig


@dataclass(frozen=True)
class Preset:
    """A named partial config the composer loads into its form.

    ``config`` is the partial ``WomblexConfig`` overlay (no ``dataset`` /
    ``paths``); ``formats`` is the file extensions the pipeline is intended for,
    surfaced so the screen can label the preset and warn if pointed at other
    inputs. Neither is enforced here — a preset is a starting point, not a lock.
    """

    name: str
    description: str
    formats: tuple[str, ...]
    config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "formats": list(self.formats),
            "config": self.config,
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
    """Every preset, in registration order — the composer's preset dropdown."""
    return [p.as_dict() for p in PRESETS.values()]


def get_preset(name: str) -> Preset | None:
    """The preset called *name*, or ``None`` if there is no such preset."""
    return PRESETS.get(name)


def _validate_presets() -> None:
    """Assert every preset overlays onto a loadable ``WomblexConfig``.

    Called once at import so a preset that would not load fails fast (and is
    covered by ``tests/test_ui.py``) rather than 500-ing when an operator picks
    it. Uses placeholder ``dataset`` / ``paths`` — the two sections a preset
    never carries — exactly as the composer's structural-graph default does.
    """
    from pathlib import Path

    for preset in PRESETS.values():
        WomblexConfig(
            dataset=DatasetConfig(name="preset-check"),
            paths=PathsConfig(
                input_root=Path("."), output_root=Path("."), checkpoint_dir=Path(".")
            ),
            **preset.config,
        )


_validate_presets()
