"""Reference register consumption for entity linking.

A reference register is a table of canonical entities, read from CSV or Parquet.
Which columns play which role (id, name, exact-match, fuzzy-match, parent) is
declared corpus-side via :class:`womblex.config.ReferenceConfig` — the library
knows nothing about specific registers.

**One entity, many names.** A register that gives an entity a legal name,
trading names and aliases does not fit one name column per row, so names
accumulate from three places: every ``match_fuzzy_cols`` column of a row, every
further row that repeats the same id, and every row of an optional long-format
``names`` table. All of them are scored as names of the one entity.

Parquet is read column-wise — only the declared columns are loaded, so a
register with a geometry or payload column costs nothing for it.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from womblex.config import ReferenceConfig, _format_of
from womblex.link.normalise import normalise_address, normalise_name

logger = logging.getLogger(__name__)


@dataclass
class ReferenceEntity:
    """One canonical reference entity with precomputed match keys."""

    entity_id: str
    name: str
    entity_type: str
    parent_id: str
    exact_key: str          # normalised concat of match_exact_cols (e.g. address)
    fuzzy_keys: list[str]    # normalised names: legal, trading, aliases — every one it goes by
    # exact_key of every further row that repeats this id (a second address)
    more_exact_keys: list[str] = field(default_factory=list)


@dataclass
class ReferenceTable:
    """Resolved reference: entities + curated alias overrides."""

    entities: list[ReferenceEntity]
    aliases: dict[str, str] = field(default_factory=dict)  # normalised alias -> entity_id
    _by_id: dict[str, ReferenceEntity] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for e in self.entities:
            self._by_id.setdefault(e.entity_id, e)

    def entity_by_id(self, entity_id: str) -> ReferenceEntity | None:
        return self._by_id.get(entity_id)

    @property
    def name_count(self) -> int:
        return sum(len(e.fuzzy_keys) for e in self.entities)


def load_reference(config: ReferenceConfig) -> ReferenceTable:
    """Load a reference register into a normalised :class:`ReferenceTable`."""
    wanted = [config.id_col, config.name_col, *config.match_exact_cols, *config.match_fuzzy_cols]
    if config.parent_id_col:
        wanted.append(config.parent_id_col)
    by_id: dict[str, ReferenceEntity] = {}
    for r in _read_rows(config.path, config.file_format(), wanted):
        entity_id = (r.get(config.id_col) or "").strip()
        if not entity_id:
            continue
        names = [normalise_name(r.get(c)) for c in config.match_fuzzy_cols]
        entity = by_id.get(entity_id)
        exact_parts = [r.get(c, "") or "" for c in config.match_exact_cols]
        exact_key = normalise_address(" ".join(p for p in exact_parts if p))
        if entity is None:
            entity = by_id[entity_id] = ReferenceEntity(
                entity_id=entity_id,
                name=(r.get(config.name_col) or "").strip(),
                entity_type=config.entity_type,
                parent_id=(
                    (r.get(config.parent_id_col) or "").strip() if config.parent_id_col else ""),
                exact_key=exact_key,
                fuzzy_keys=[],
            )
        elif exact_key and exact_key != entity.exact_key and exact_key not in entity.more_exact_keys:
            entity.more_exact_keys.append(exact_key)
        _add_names(entity, names)

    if config.names is not None:
        _load_names(config, by_id)
    aliases = _load_aliases(config.alias_table) if config.alias_table else {}
    return ReferenceTable(entities=list(by_id.values()), aliases=aliases)


def _add_names(entity: ReferenceEntity, names: list[str]) -> None:
    for name in names:
        if name and name not in entity.fuzzy_keys:
            entity.fuzzy_keys.append(name)


def _load_names(config: ReferenceConfig, by_id: dict[str, ReferenceEntity]) -> None:
    """Add every row of the long-format names table to the entity its id names."""
    names = config.names
    assert names is not None
    orphans = 0
    fmt = names.format or _format_of(names.path)
    for r in _read_rows(names.path, fmt, [names.id_col, names.name_col]):
        entity = by_id.get((r.get(names.id_col) or "").strip())
        if entity is None:
            orphans += 1
            continue
        _add_names(entity, [normalise_name(r.get(names.name_col))])
    if orphans:
        logger.warning(
            "reference %s: %d name row(s) in %s name an id the register does not hold; ignored",
            config.path, orphans, names.path,
        )


def _read_rows(path: Path, fmt: str, columns: list[str]) -> list[dict[str, str]]:
    """Rows of a register file as string dicts. A declared column the file lacks is an error."""
    if fmt == "csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            missing = sorted(set(columns) - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f"reference file {path} has no column(s) {missing}")
            return list(reader)
    wanted = list(dict.fromkeys(columns))
    missing = sorted(set(wanted) - set(pq.read_schema(str(path)).names))
    if missing:
        raise ValueError(f"reference file {path} has no column(s) {missing}")
    table = pq.read_table(str(path), columns=wanted)
    data = {name: table.column(name).to_pylist() for name in wanted}
    return [
        {name: "" if (v := data[name][i]) is None else str(v) for name in wanted}
        for i in range(table.num_rows)
    ]


def _load_aliases(path: Path) -> dict[str, str]:
    """Load a corpus alias table (columns ``alias``, ``entity_id``).

    Aliases cover entities the register doesn't carry (e.g. a prior
    trustee). Keyed on the normalised alias name.
    """
    out: dict[str, str] = {}
    for r in _read_rows(path, _format_of(path), ["alias", "entity_id"]):
        alias = normalise_name(r.get("alias"))
        eid = (r.get("entity_id") or "").strip()
        if alias and eid:
            out[alias] = eid
    return out
