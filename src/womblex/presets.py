"""Named pre-configured pipelines a workflow or a dispatched run starts from.

A *preset* is a partial :class:`~womblex.config.WomblexConfig` — the stage
toggles and stage settings for a common end-to-end shape. It is deliberately
*partial*: ``dataset`` and ``paths`` name a real run and have no sensible
default, so a preset never carries them. Merging a preset into a config
therefore only ever flips stage `enabled` flags and their settings, leaving the
run's identity and filesystem paths untouched.

Presets are just data, so a preset that would not load is a test failure, not a
runtime surprise: every one is validated through the same
``WomblexConfig(**raw)`` construction ``load_config`` uses.

`DEFAULT-Isaacus` is the reference Isaacus pipeline:
``extract → normalise → enrich → chunk → build_graph → embed →
money → link → done``, with text cleaning first (normalise, selected via
``processing.text_source``), the entity graph (enrich + the
``graph-refresh`` mention→chunk edge rebuild), chunk embeddings (the
kanon-2-embedder retrieval index), monetary amounts and entity links produced
over the one run. It targets PDF, DOCX and Markdown sources — the narrative formats
among ``SUPPORTED_EXTENSIONS`` — which is what makes AI chunking and enrichment
meaningful (spreadsheets are structured, not narrative). The vendored demo run
(``run-synthetic-demo``, ``womblex seed-demo``) is a completed run of this
shape over the synthetic fixture documents, bar ``link`` (it needs a corpus
reference register).

The runnable, CLI-first source of truth for that shape is
``configs/default-isaacus.yaml`` — a complete config plus the per-stage command
sequence a dev runs it with (``womblex run`` alone cannot: normalise,
enrich, build_graph, money and link are per-stage commands, normalise
run before enrich, and enrich must precede chunk). The overlay here mirrors
that file's stage toggles and settings; keep the two in step.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from womblex.config import DatasetConfig, PathsConfig, WomblexConfig

if TYPE_CHECKING:
    from womblex.cloud.deployment import DeploymentSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Preset:
    """A named partial config a run or workflow starts from.

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


#: `DEFAULT-Isaacus` — extract → normalise → enrich → chunk →
#: build_graph → embed → money → link.
#:
#: Mirrors ``configs/default-isaacus.yaml`` (the runnable, CLI-first source of
#: truth) and the vendored demo run (``run-synthetic-demo``, a completed run of
#: this shape bar ``link``) — edit all three together. Settings beyond the bare
#: `enabled` flags are carried so a preset seeds sensible values, not
#: just toggles:
#:
#: - `processing.text_source = normalised` selects the cleaned text layer as
#:   the single string BOTH enrich and chunk reassemble from. It is also the
#:   gate that puts `normalise` into the downstream-stage dispatch: it has no
#:   `enabled` flag of its own, so `enabled_downstream_stages` treats it as
#:   wanted precisely when a consumer selects its layer. It runs BEFORE enrich
#:   in `PIPELINE_ORDER`, so the cleaned text is what enrichment and AI chunking
#:   see.
#: - `spellfix` is deliberately absent. Dictionary-gated OCR repair rewrites
#:   source text on a dictionary judgement, and Australian government documents
#:   are dense with proper nouns, statute short titles and agency acronyms that
#:   are legitimately out-of-dictionary — so it is a per-corpus call to make
#:   after measuring, not a reference default. The stage and its config are
#:   unchanged.
#: - `chunking.chunking_model = kanon-2-enricher` selects semchunk-4 AI
#:   chunking (boundaries follow the enricher's structure). With `enrich` also
#:   on, `WomblexConfig` auto-enables `enrichment.persist_document` so `chunk`
#:   reuses the enrich Document rather than enriching twice — run enrich BEFORE
#:   chunk (docs/decisions.md). The resulting graph then gains its mention→chunk
#:   edges from the offline `graph-refresh` stage (the "build_graph" step).
#:   `chunk_size = 480` is the Kanon-2 window; `overlap = 0.1` shares 10% across
#:   boundaries so a mention straddling one still lands in a chunk.
#: - `embedding.enabled` vectorises the chunks with the kanon-2-embedder (the
#:   retrieval index), leaving `task`/`dimensions` at the model defaults. This
#:   is the stage the demo run carries (`*.embeddings.parquet`) and was the one
#:   the earlier preset omitted, so the sample corpus and the preset disagreed.
#: - `money.enabled` runs the offline amount annotator over the same run, so the
#:   graph and money sidecars land side by side (`graph + money over the one run`);
#:   `default_currency = AUD` matches Australian-government `$` convention.
#: - `linking.enabled` matches the enrichment entities to a corpus reference
#:   register. The register is corpus-specific and is NOT carried here (like
#:   `dataset`/`paths`, it names a real file): the operator supplies
#:   `linking.reference` at run time, and until they do the stage warns and
#:   no-ops. Enabling it without a reference still builds a valid `WomblexConfig`
#:   (there is no model-level requirement), so the preset stays loadable.
_DEFAULT_ISAACUS = Preset(
    name="DEFAULT-Isaacus",
    description=(
        "Reference Isaacus pipeline: extract, text cleaning (normalise), "
        "Kanon-2 enrichment, semantic (AI) chunking, entity-graph "
        "edge rebuild (build_graph), chunk embeddings, monetary-amount "
        "annotation and entity linking — cleaning first, then graph, embeddings, "
        "money and links over the one run. For PDF, DOCX and Markdown sources. Runnable "
        "from the CLI as configs/default-isaacus.yaml (a per-stage sequence: "
        "normalise, enrich before chunk, then graph-refresh, embed, "
        "money, link). Linking needs a corpus reference register (set "
        "linking.reference) supplied at run time."
    ),
    formats=(".pdf", ".docx", ".md", ".markdown"),
    config={
        "chunking": {
            "enabled": True,
            "chunking_model": "kanon-2-enricher",
            "chunk_size": 480,
            "overlap": 0.1,
        },
        "enrichment": {"enabled": True},
        "embedding": {"enabled": True},
        "money": {"enabled": True, "default_currency": "AUD"},
        "linking": {"enabled": True},
        "processing": {"text_source": "normalised"},
        # Redaction/PII stay at their defaults (off, except redaction flagging).
        # normalise carries no toggle — it is gated by text_source above, and
        # spellfix stays off (see the note above).
    },
)

#: Registry, insertion-ordered. Add a preset by defining it above and listing
#: it here; the API and form pick it up with no other change.
PRESETS: dict[str, Preset] = {p.name: p for p in (_DEFAULT_ISAACUS,)}


def list_presets() -> list[dict[str, Any]]:
    """Every built-in preset, in registration order."""
    return [p.as_dict() for p in PRESETS.values()]


def get_preset(name: str) -> Preset | None:
    """The built-in preset called *name*, or ``None`` if there is no such preset."""
    return PRESETS.get(name)


# ---------------------------------------------------------------------------
# Operator-saved presets
# ---------------------------------------------------------------------------
#
# Besides the built-ins above (code), an operator can *save* a config
# as a named starting point of their own. Each lands as one JSON file — one file
# per record, so two saves can't lose each other. Locally the files sit under a
# ``presets_dir``, remotely under the store's own ``presets/`` prefix (a sibling
# of ``runs/``). This module owns the *format* — the filename, the record bytes,
# and parsing one file back into a :class:`Preset` — and reading, saving and
# deleting them.

#: The store prefix operator-saved presets live under in remote mode — a
#: sibling of ``runs/`` in the same bucket.
PRESETS_DIRNAME = "presets"

_PRESET_SUFFIX = ".preset.json"

#: A preset name becomes a filename, so it is constrained like a run id
#: (``retention.is_safe_run_id``): letters, digits, dot, dash, underscore,
#: no leading dot — enough for ``My-Run_v2`` without admitting ``..`` or a
#: separator.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def is_safe_preset_name(name: str) -> bool:
    """True if *name* is a single, filesystem-safe token (it becomes a filename).

    Refused rather than sanitised, like ``retention.is_safe_run_id``: a
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
# `read_saved_presets` below owns *where* these files sit (local dir vs. store
# prefix); this section owns *what* one is.
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
    except Exception as exc:
        logger.warning("saved preset %r skipped: %s", name, exc)
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


def read_saved_presets(settings: DeploymentSettings) -> dict[str, Preset]:
    """Every operator-saved preset, keyed by name — the store prefix or the local dir.

    A file that will not parse or whose overlay will not load is skipped (not
    fatal): one corrupt preset must not blank the list.
    """
    saved: dict[str, Preset] = {}
    if settings.is_remote:
        from womblex.store.remote_manifest import open_store

        store = open_store(str(settings.store_uri))
        for key in store.list_files(PRESETS_DIRNAME, "*"):
            name = preset_name_from_filename(key.rsplit("/", 1)[-1])
            if name is None:
                continue
            preset = parse_saved_preset(name, store.read_text(key))
            if preset is not None:
                saved[name] = preset
        return saved
    presets_dir = settings.presets_dir
    if presets_dir is None or not presets_dir.is_dir():
        return saved
    for path in sorted(presets_dir.glob("*")):
        name = preset_name_from_filename(path.name)
        if name is None:
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:
            continue
        preset = parse_saved_preset(name, body)
        if preset is not None:
            saved[name] = preset
    return saved


def list_all_presets(settings: DeploymentSettings) -> list[dict[str, Any]]:
    """Built-in presets, then saved ones (a saved name shadows a built-in)."""
    return merge_saved(read_saved_presets(settings))


class PresetsNotWritable(Exception):
    """This deployment has nowhere to save presets: local, with no ``presets_dir``."""


def save_preset(
    settings: DeploymentSettings, *, name: str, description: str, formats: tuple[str, ...], config: dict[str, Any],
) -> Preset:
    """Validate and save one preset under the store's ``presets/`` prefix or the local ``presets_dir``.

    Replaces a saved preset of the same name. Raises `PresetsNotWritable`, ``ValueError``
    on an unsafe name and ``pydantic.ValidationError`` on an overlay that would not load.
    """
    if not settings.presets_writable:
        raise PresetsNotWritable("no presets_dir is configured, so presets cannot be saved")
    record, overlay = build_preset_record(name=name, description=description, formats=formats, config=config)
    body = serialise_preset_record(record).encode("utf-8")
    filename = preset_filename(name)
    if settings.is_remote:
        import io

        from womblex.store.remote_manifest import open_store

        open_store(str(settings.store_uri)).write_stream(io.BytesIO(body), f"{PRESETS_DIRNAME}/{filename}")
    else:
        presets_dir = cast(Path, settings.presets_dir)
        presets_dir.mkdir(parents=True, exist_ok=True)
        (presets_dir / filename).write_bytes(body)
    return Preset(name=name, description=description, formats=formats, config=overlay, source="saved")


def delete_saved_preset(settings: DeploymentSettings, name: str) -> bool:
    """Delete a saved preset; ``False`` when no saved one has that name (a built-in is code, never removed)."""
    if not settings.presets_writable:
        raise PresetsNotWritable("no presets_dir is configured, so there are no saved presets")
    if not is_safe_preset_name(name):
        return False
    filename = preset_filename(name)
    if settings.is_remote:
        from womblex.store.remote_manifest import open_store

        store = open_store(str(settings.store_uri))
        key = f"{PRESETS_DIRNAME}/{filename}"
        if not store.exists(key):
            return False
        store.delete(key)
        return True
    path = cast(Path, settings.presets_dir) / filename
    if not path.is_file():
        return False
    path.unlink()
    return True


def get_any_preset(settings: DeploymentSettings, name: str) -> Preset | None:
    """One preset by name (a saved one shadowing a built-in), or ``None``."""
    if not is_safe_preset_name(name):
        return PRESETS.get(name)
    return resolve_one(read_saved_presets(settings), name)


def _validate_presets() -> None:
    """Assert every built-in preset loads — called once at import so it fails fast."""
    for preset in PRESETS.values():
        _validate_overlay(preset.config)


_validate_presets()
