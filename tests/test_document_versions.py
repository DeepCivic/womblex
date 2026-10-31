"""Document versions on the run-root manifest: chains per path, in batch order."""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.document_versions import add_versions
from womblex.store.output import MANIFEST_SCHEMA, read_manifest
from womblex.store.run_manifest import write_run_manifest


def _table(rows: list[tuple[str, str]]) -> pa.Table:
    return pa.table({"source_hash": [h for h, _ in rows], "source_relpath": [r for _, r in rows]})


def _chain(rows: list[tuple[str, str]]) -> list[tuple[str | None, str | None, bool]]:
    t = add_versions(_table(rows))
    return list(zip(t["replaces"].to_pylist(), t["replaced_by"].to_pylist(), t["is_current"].to_pylist(), strict=True))


def test_a_changed_file_chains_to_its_earlier_version_and_only_the_last_is_current() -> None:
    assert _chain([("h1", "a.csv"), ("h2", "a.csv"), ("h3", "a.csv")]) == [
        (None, "h2", False), ("h1", "h3", False), ("h2", None, True)]


def test_a_moved_file_and_identical_bytes_at_two_paths_are_separate_chains() -> None:
    assert _chain([("h1", "a.csv"), ("h1", "b.csv"), ("h2", "c.csv")]) == [
        (None, None, True), (None, None, True), (None, None, True)]


def test_a_row_with_no_relpath_is_its_own_chain_and_a_repeated_hash_is_one_version() -> None:
    assert _chain([("h1", ""), ("h2", ""), ("h3", "a"), ("h3", "a"), ("h4", "a")]) == [
        (None, None, True), (None, None, True), (None, "h4", False), (None, "h4", False), ("h3", None, True)]


def test_an_empty_manifest_keeps_its_schema_and_gains_the_columns() -> None:
    t = add_versions(_table([]))
    assert t.num_rows == 0 and t.schema.field("is_current").type == pa.bool_()


def _shard(directory: Path, batch: int, rows: list[tuple[str, str]], *, old: bool = False) -> None:
    schema = pa.schema([f for f in MANIFEST_SCHEMA if not (old and f.name in {"replaces", "replaced_by", "is_current"})])
    data = [{f.name: ("" if pa.types.is_string(f.type) else (0 if pa.types.is_integer(f.type) else None))
             for f in schema} | {"source_hash": h, "source_relpath": r, "filename": Path(r).name} for h, r in rows]
    pq.write_table(pa.Table.from_pylist(data, schema=schema), str(directory / f"batch-{batch:04d}._manifest.parquet"))


def test_consolidation_orders_by_batch_number_and_per_batch_files_are_untouched(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    _shard(docs, 2, [("h2", "inbox/a.csv")])
    _shard(docs, 1, [("h1", "inbox/a.csv"), ("x", "inbox/b.csv")], old=True)  # written before the columns
    before = (docs / "batch-0001._manifest.parquet").read_bytes()
    target = write_run_manifest(docs)
    run = pq.read_table(str(target)).to_pylist()
    by_hash = {r["source_hash"]: r for r in run}
    assert (by_hash["h1"]["replaced_by"], by_hash["h1"]["is_current"]) == ("h2", False)
    assert (by_hash["h2"]["replaces"], by_hash["h2"]["is_current"]) == ("h1", True)
    assert by_hash["x"]["is_current"] is True
    assert (docs / "batch-0001._manifest.parquet").read_bytes() == before
    per_batch = read_manifest(docs)  # an older shard reads with the columns null
    assert per_batch.column("is_current").null_count == per_batch.num_rows
