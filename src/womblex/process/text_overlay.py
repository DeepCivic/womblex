"""Element-text overlay resolution — the shared composability primitive.

Several stages produce a per-element *text layer* over the verbatim element
stream, all with the same ``(source_hash, elem_order, text)`` shape:

- ``normalised`` → ``*.normalised_text.parquet`` (formatting cleanup)
- ``spellfix``   → ``*.spellfix_text.parquet``   (OCR character-confusion repair)

A consuming stage selects one via ``text_source`` and overlays it onto the
``Element`` list *before* reassembly, so both the chunk branch
(``build_chunk_input``) and the enrichment branch (``reassemble_narrative``)
operate on the same repaired/cleaned text in one coordinate space. ``"elements"``
(the default) means verbatim — no overlay. A selected overlay that hasn't been
written yet resolves to ``None`` (graceful passthrough), so ordering the stages
is the only requirement, not a hard dependency.

That graceful fallback is a composability convenience for the transform
stages (chunk / enrich / money), where re-running with the sidecar present
simply refines the result. It is **not** universal: a caller that must not
render or persist verbatim text under a declared cleaning layer passes
``required=True`` and gets a loud :class:`FileNotFoundError` instead of the
silent fallback (the render path is the one such caller today).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow.parquet as pq

from womblex.ingest.elements import Element
from womblex.store.normalise_output import NORMALISED_TEXT_SUFFIX
from womblex.store.spellfix_output import SPELLFIX_TEXT_SUFFIX

logger = logging.getLogger(__name__)

TEXT_SOURCES = ("elements", "normalised", "spellfix")

_SUFFIX = {
    "normalised": NORMALISED_TEXT_SUFFIX,
    "spellfix": SPELLFIX_TEXT_SUFFIX,
}


def load_overlay(
    base_path: Path, text_source: str, *, warn_if_missing: bool = True, required: bool = False,
) -> dict[tuple[str, int], str] | None:
    """Return ``{(source_hash, elem_order): text}`` for *text_source*, or ``None``.

    ``None`` means "use verbatim element text": either ``text_source='elements'``
    or the selected overlay sidecar isn't present for this batch. Pass
    ``warn_if_missing=False`` when a missing overlay is expected (e.g. spellfix
    chaining off a normalise layer that may not have been run).

    ``required=True`` refuses that silent fallback: a declared non-``elements``
    overlay that is missing raises :class:`FileNotFoundError` rather than
    returning ``None``, so the caller renders the declared text layer or fails.
    ``text_source='elements'`` still returns ``None`` under ``required`` —
    verbatim *is* the declared layer there, not a fallback.
    """
    if text_source == "elements":
        return None
    if text_source not in _SUFFIX:
        raise ValueError(f"text_source must be one of {TEXT_SOURCES}, got {text_source!r}")

    path = base_path.parent / f"{base_path.stem}{_SUFFIX[text_source]}"
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"text_source={text_source!r} declared but {path.name} is missing; "
                f"run the {text_source} stage first. This caller renders the declared "
                f"text layer and never falls back to verbatim."
            )
        if warn_if_missing:
            logger.warning(
                "text_source=%r selected but %s missing — using verbatim element text. "
                "Run the %s stage first.", text_source, path.name, text_source,
            )
        return None

    table = pq.read_table(str(path), columns=["source_hash", "elem_order", "text"])
    return {
        (sh, eo): tx
        for sh, eo, tx in zip(
            table.column("source_hash").to_pylist(),
            table.column("elem_order").to_pylist(),
            table.column("text").to_pylist(),
        )
    }


def apply_overlay(
    source_hash: str, elements: list[Element], overrides: dict[tuple[str, int], str] | None,
) -> None:
    """Override each element's ``text`` from *overrides* in place (no-op if ``None``)."""
    if not overrides:
        return
    for e in elements:
        replacement = overrides.get((source_hash, e.order))
        if replacement is not None:
            e.text = replacement


__all__ = ["TEXT_SOURCES", "apply_overlay", "load_overlay"]
