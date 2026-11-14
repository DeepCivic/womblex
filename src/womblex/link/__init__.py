"""Entity linking: resolve enrichment candidates to canonical reference entities.

- :mod:`womblex.link.normalise` — minimal name/address normalisation.
- :mod:`womblex.link.reference` — reference register loading (CSV, Parquet).
- :mod:`womblex.link.matcher` — generic record-linkage matcher.
- :mod:`womblex.link.blocking` — candidate narrowing for a large register.
- :mod:`womblex.link.text_candidates` — register names found in the text.
- :mod:`womblex.link.stage` — per-stage ``link_shards`` over a shard dir.
"""

from womblex.link.matcher import Candidate, Link, Matcher, resolve
from womblex.link.reference import ReferenceEntity, ReferenceTable, load_reference
from womblex.link.stage import LinkStageResult, link_shards

__all__ = [
    "Candidate",
    "Link",
    "LinkStageResult",
    "Matcher",
    "ReferenceEntity",
    "ReferenceTable",
    "link_shards",
    "load_reference",
    "resolve",
]
