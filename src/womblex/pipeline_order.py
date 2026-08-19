"""The order the pipeline's stages run in — declared once.

`StageContract` knows each stage's *inputs*, not its *place*. That is
deliberate (`run-stage` documents "ordering is yours to pick"), and it holds
because the one edge that matters most is not expressible as an input at all:
`chunk` reads `.enrichment_doc` only when `chunking.chunking_model` is set, so
enrich→chunk is config-derived and a topological sort over `required_inputs`
alone will happily put chunk first.

So the order has to be written down. It was — three times, in three places,
disagreeing: `STAGE_CONTRACTS`'s dict order, `store.retention.STAGE_SUFFIXES`
(commented "in pipeline order") and the README's command sequence. The first
two both ran chunk before enrich, and the console rendered its lifecycle
checkpoints in that order. This module is the README's sequence — the correct
one — as the single declaration the others now derive from.

`sort_by_pipeline` rather than an exported list of pairs, because every
consumer so far holds its own stage→something mapping (a sidecar suffix, a
checkpoint dirname) and wants that mapping ordered, not replaced.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeVar

#: Every pipeline stage, in the order a full run executes them. Extraction
#: leads (it produces what all the rest read); the two non-obvious edges are:
#:
#: - **normalise and spellfix precede enrich and chunk**, because they write
#:   the element-text overlays `processing.text_source` makes *both* of those
#:   stages reassemble from. Running them later leaves the overlay on disk and
#:   unread — the silent no-op that makes a config setting look broken.
#:   spellfix chains on top of normalised, so it follows it.
#: - **enrich precedes chunk**, so AI chunking reuses the persisted ILGS
#:   Document rather than paying for a second enrichment pass (README: "enrich
#:   must precede chunk so AI chunking reuses the enrichment (no double
#:   cost)").
#:
#: The rest follow their hard input edges: `graph-refresh` re-links mentions
#: onto the chunks `chunk` just wrote, `embed` vectorises those chunks, `link`
#: matches the entities `enrich` wrote, and `pii` is terminal — its masking is
#: irreversible and must not precede the stages that need the raw text.
#: `money` reads only extraction output, so its position is free; it sits with
#: the other offline stages. `quality` is last because it is the one
#: run-scoped stage: its duplicate-cluster ids are corpus-wide, so it is
#: meaningful only over a drained run.
PIPELINE_ORDER: tuple[str, ...] = (
    "extract",
    "normalise",
    "spellfix",
    "enrich",
    "chunk",
    "graph-refresh",
    "embed",
    "money",
    "link",
    "pii",
    "quality",
)

#: Rank per stage, for sorting. Unknown names sort last (rather than raising)
#: so a caller's mapping that carries an extra key still round-trips — this is
#: a presentation order, and dropping or exploding on an unrecognised stage
#: would be a worse failure than putting it at the end.
_RANK: dict[str, int] = {name: i for i, name in enumerate(PIPELINE_ORDER)}

_V = TypeVar("_V")


def sort_by_pipeline(mapping: Mapping[str, _V]) -> dict[str, _V]:
    """*mapping*, re-keyed in `PIPELINE_ORDER`. Values are untouched."""
    return {k: mapping[k] for k in sorted(mapping, key=stage_rank)}


def stage_rank(stage: str) -> int:
    """Position of *stage* in the pipeline; unknown stages sort last."""
    return _RANK.get(stage, len(PIPELINE_ORDER))


def in_pipeline_order(stages: Iterable[str]) -> tuple[str, ...]:
    """*stages*, ordered. For a bare collection with no mapping to carry."""
    return tuple(sorted(stages, key=stage_rank))


__all__ = ["PIPELINE_ORDER", "in_pipeline_order", "sort_by_pipeline", "stage_rank"]
