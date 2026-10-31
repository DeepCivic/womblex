"""Document versions across a collection's batches, for the run-root manifest.

A collection keeps every version of a file its input prefix has held (Requirement 34). Versions
are chained per `source_relpath`, in the order the manifest rows arrive: `read_manifest` reads the
per-batch shards by batch number, so row order is batch order and never time. A moved or renamed
file starts a new chain, and identical bytes at two paths are two chains. A row with no
`source_relpath` (a records ingest, a shard written before the column) is its own chain.

Per-batch manifests are never rewritten; the three columns exist only on the consolidated one
(`store/run_manifest.write_run_manifest`) and are null on a per-batch file.
"""

from __future__ import annotations

import pyarrow as pa

VERSION_COLUMNS = ("replaces", "replaced_by", "is_current")


def add_versions(table: pa.Table) -> pa.Table:
    """*table* with `replaces`, `replaced_by` and `is_current` filled from its rows' order.

    `replaces` is the `source_hash` of the previous version in the row's chain, `replaced_by`
    the next one's, and `is_current` is true for the last row of the chain. A row repeating the
    hash just before it is the same version, not a new one.
    """
    hashes = table.column("source_hash").to_pylist()
    relpaths = table.column("source_relpath").to_pylist()
    n = len(hashes)
    replaces: list[str | None] = [None] * n
    replaced_by: list[str | None] = [None] * n
    is_current = [True] * n
    held: dict[str, list[int]] = {}  # relpath -> the rows of the chain's latest version
    for i, relpath in enumerate(relpaths):
        if not relpath:
            continue
        rows = held.setdefault(relpath, [])
        newer = bool(rows) and hashes[rows[0]] != hashes[i]
        if rows:
            replaces[i] = hashes[rows[0]] if newer else replaces[rows[0]]
        for j in rows:  # a new version replaces every row of the last; a repeat leaves them superseded by this row
            is_current[j] = False
            if newer:
                replaced_by[j] = hashes[i]
        if newer:
            rows.clear()
        rows.append(i)
    out = table
    for name in VERSION_COLUMNS:
        if name in out.schema.names:
            out = out.drop_columns(name)
    return (out.append_column("replaces", pa.array(replaces, pa.string()))
            .append_column("replaced_by", pa.array(replaced_by, pa.string()))
            .append_column("is_current", pa.array(is_current, pa.bool_())))


__all__ = ["VERSION_COLUMNS", "add_versions"]
