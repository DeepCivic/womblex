"""Thin pyarrow readers over shard sidecars, local and store-backed.

Local reads go straight through ``store/retention.py`` / ``store/output.py``.
Remote reads stage the (small) manifest into a temp dir and hand it to the
same local readers — the pattern ``cli/cloud.py``'s ``finalize`` already
uses, so there is one code path for the parquet logic.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.enrichment_output import ENRICHMENT_ENTITIES_SUFFIX, ENTITY_SCHEMA
from womblex.store.feedback_output import (
    FEEDBACK_DIRNAME,
    build_feedback_record,
    feedback_filename,
    write_feedback_record,
)
from womblex.store.money_output import MONEY_SPANS_SCHEMA, MONEY_SPANS_SUFFIX
from womblex.store.output import (
    _SHARD_SUFFIX,
    CHUNKS_SCHEMA,
    CHUNKS_SUFFIX,
    read_manifest,
)
from womblex.store.pii_output import PII_SPANS_SCHEMA, PII_SPANS_SUFFIX
from womblex.store.quality_output import CHUNK_QUALITY_SCHEMA, CHUNK_QUALITY_SUFFIX
from womblex.store.retention import STAGE_SUFFIXES, RunDescription, describe_run, list_runs
from womblex.store.run_manifest import RUN_MANIFEST_FILENAME
from womblex.store.shard_audit import ARCHIVE_SUFFIX, audit_shard_directory
from womblex.ui.deps import UISettings

if TYPE_CHECKING:
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)


def list_run_summaries(settings: UISettings) -> list[RunDescription]:
    """All runs the settings can see, newest run_id first."""
    if settings.is_remote:
        runs = _list_remote_runs(cast(str, settings.store_uri))
    else:
        runs = [describe_run(p) for p in list_runs(cast(Path, settings.output_root))]
    return sorted(runs, key=lambda d: d.run_id, reverse=True)


def get_manifest_rows(settings: UISettings, run_id: str) -> list[dict] | None:
    """Manifest rows (one per document) for run_id, or None if it doesn't exist."""
    if settings.is_remote:
        return _remote_manifest_rows(cast(str, settings.store_uri), run_id)
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    return list(_local_manifest_table(run_dir).to_pylist())


def get_stage_presence(settings: UISettings, run_id: str, stage: str) -> list[str] | None:
    """``source_hash`` values with a ``stage`` sidecar row in run_id.

    None if the run doesn't exist, else a (possibly empty) sorted list —
    the lifecycle-checkpoint switcher's data (docs/ui-plan.md §3 "lifecycle
    checkpoints are sidecar presence"). Reads only the ``source_hash``
    column of each sidecar, so this stays cheap even for large chunks /
    enrichment sidecars.
    """
    suffix = STAGE_SUFFIXES[stage]
    if settings.is_remote:
        return _remote_stage_presence(cast(str, settings.store_uri), run_id, suffix)
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    shard_dir = run_dir / "documents"
    if not shard_dir.is_dir():
        return []
    return _scan_stage_presence(shard_dir, suffix)


def get_shard_audit(settings: UISettings, run_id: str) -> dict | None:
    """Cross-batch structural audit for run_id — the verify-shards action's data."""
    if settings.is_remote:
        return _remote_shard_audit(cast(str, settings.store_uri), run_id)
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    return audit_shard_directory(run_dir / "documents").as_dict()


# The Chunk Inspector's overlay sidecars (docs/ui-plan.md §3), each filtered
# to one document: (response key, suffix, canonical schema, join column).
# Suffix and schema are taken from the same store module so a renamed suffix
# cannot drift away from the schema it describes. `entities` alone joins on
# `document_id` — see `enrichment_output.py`'s note that the sharded layout
# writes the source_hash into that column.
_CHUNK_DETAIL_SIDECARS: tuple[tuple[str, str, pa.Schema, str], ...] = (
    ("chunks", CHUNKS_SUFFIX, CHUNKS_SCHEMA, "source_hash"),
    ("entities", ENRICHMENT_ENTITIES_SUFFIX, ENTITY_SCHEMA, "document_id"),
    ("pii_spans", PII_SPANS_SUFFIX, PII_SPANS_SCHEMA, "source_hash"),
    ("money_spans", MONEY_SPANS_SUFFIX, MONEY_SPANS_SCHEMA, "source_hash"),
    ("quality", CHUNK_QUALITY_SUFFIX, CHUNK_QUALITY_SCHEMA, "source_hash"),
)


def get_chunk_detail(settings: UISettings, run_id: str, source_hash: str) -> dict | None:
    """Chunks + overlay sidecars for one document — the Chunk Inspector's data.

    None if run_id doesn't exist. An empty ``chunks`` list is a legitimate
    answer (chunk stage not yet run, or source_hash absent from this run) —
    the same "present, not missing" shape as :func:`get_stage_presence`.
    """
    if settings.is_remote:
        return _remote_chunk_detail(cast(str, settings.store_uri), run_id, source_hash)
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    shard_dir = run_dir / "documents"  # glob yields nothing if it doesn't exist yet
    paths = {
        suffix: [str(p) for p in sorted(shard_dir.glob(f"*{suffix}"))]
        for _key, suffix, _schema, _column in _CHUNK_DETAIL_SIDECARS
    }
    return _chunk_detail(paths, source_hash)


def write_feedback(
    settings: UISettings,
    run_id: str,
    *,
    record_type: str,
    source_hash: str,
    chunk_index: int | None,
    row: dict,
    note: str,
    reported_by: str | None,
) -> dict | None:
    """Write one report-action file for run_id; None if the run doesn't exist.

    Local and remote both land at a ``feedback/<run_id>/`` sibling of the
    run directories, never inside one (docs/ui-plan.md §4) — see
    :mod:`womblex.store.feedback_output`'s module docstring for why. The
    two branches differ only in where that sibling sits: locally it nests
    under ``output_root`` (a plain directory, not a run-id one, so
    retention's ``run-*`` purge never touches it); remotely it is the
    store's own ``feedback/`` prefix, parallel to ``runs/``.
    """
    record = build_feedback_record(
        run_id=run_id, record_type=record_type, source_hash=source_hash,
        chunk_index=chunk_index, row=row, note=note, reported_by=reported_by,
    )
    if settings.is_remote:
        store = _open_store(cast(str, settings.store_uri))
        if run_id not in store.list_dirs("runs"):
            return None
        with tempfile.TemporaryDirectory(prefix="womblex-ui-") as tmp:
            name = feedback_filename()
            tmp_path = Path(tmp) / name
            tmp_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            store.upload_file(tmp_path, f"{FEEDBACK_DIRNAME}/{run_id}/{name}")
        return record
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    feedback_root = settings.feedback_dir or cast(Path, settings.output_root) / FEEDBACK_DIRNAME
    write_feedback_record(feedback_root, run_id, record)
    return record


# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------


def _local_manifest_table(run_dir: Path) -> pa.Table:
    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    if manifest_path.exists():
        return pq.read_table(str(manifest_path))
    return read_manifest(run_dir / "documents")


def _conform(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Reindex *table* to *schema*, null-filling columns a drifted shard predates.

    The same compat shim ``store.output._read_chunks_shard`` applies (see
    ``_CHUNKS_BACKFILL``): a run written before a column existed must not
    hand the frontend rows whose key set differs batch to batch.
    """
    for field in schema:
        if field.name not in table.schema.names:
            table = table.append_column(field, pa.nulls(table.num_rows, type=field.type))
    return table.select([f.name for f in schema]).cast(schema)


def _read_filtered(
    paths: list[str],
    schema: pa.Schema,
    column: str,
    value: str,
    *,
    filesystem: object | None = None,
) -> list[dict]:
    """Rows with ``column == value`` across *paths*, conformed to *schema*.

    The predicate is pushed into the parquet reader rather than applied after
    a full read, so a whole-corpus sidecar costs only the row groups whose
    statistics admit *value*. Same skip-unreadable-and-warn policy as
    :func:`_scan_stage_presence` — one corrupt batch narrows a document's
    overlay data, it doesn't blank the screen.
    """
    rows: list[dict] = []
    for p in paths:
        try:
            table = pq.read_table(p, filesystem=filesystem, filters=[(column, "=", value)])
            rows.extend(_conform(table, schema).to_pylist())
        except Exception as e:
            logger.warning("chunk-detail: skipping unreadable sidecar %s: %s", p, e)
            continue
    return rows


def _chunk_detail(
    paths_by_suffix: dict[str, list[str]],
    source_hash: str,
    *,
    filesystem: object | None = None,
) -> dict:
    """Assemble the Chunk Inspector payload from already-located sidecar paths.

    Local and store-backed reads share this body; only path resolution and
    the filesystem differ.
    """
    detail: dict = {}
    for key, suffix, schema, column in _CHUNK_DETAIL_SIDECARS:
        detail[key] = _read_filtered(
            paths_by_suffix.get(suffix, []), schema, column, source_hash,
            filesystem=filesystem,
        )
    # The sharded enrichment layout writes source_hash into `document_id`;
    # present it under the name every other sidecar joins on.
    for entity in detail["entities"]:
        entity["source_hash"] = entity.pop("document_id")
    # chunk_index is re-sequenced per document across narrative *and* table
    # chunks (process/chunker.py), so it totally orders a document's chunks.
    detail["chunks"].sort(key=lambda r: r["chunk_index"])
    # Only narrative-locus money anchors to chunk text; table_cell / sheet_cell
    # spans anchor to the cell sidecars and have no offset to overlay here.
    detail["money_spans"] = [r for r in detail["money_spans"] if r["locus"] == "narrative"]
    # `value` is decimal128(38,4) — exact by contract. FastAPI's encoder turns
    # a Decimal into a float, which silently loses that exactness, so it goes
    # over the wire as a string.
    for span in detail["money_spans"]:
        span["value"] = None if span["value"] is None else str(span["value"])
    return detail


def _scan_stage_presence(shard_dir: Path, suffix: str) -> list[str]:
    """``source_hash`` values across every ``*<suffix>`` sidecar in *shard_dir*.

    An unreadable sidecar is warned about and skipped rather than failing the
    whole screen: presence is an annotation on the documents grid, and one
    corrupt batch should not blank the other batches' answer. ``verify-shards``
    is the surface that reports corruption as such.
    """
    hashes: set[str] = set()
    for p in sorted(shard_dir.glob(f"*{suffix}")):
        if p.name.endswith(ARCHIVE_SUFFIX):
            continue
        try:
            col = pq.read_table(str(p), columns=["source_hash"]).column("source_hash")
        except Exception as e:
            logger.warning("stage-presence: skipping unreadable sidecar %s: %s", p.name, e)
            continue
        hashes.update(h for h in col.to_pylist() if h)
    return sorted(hashes)


# ---------------------------------------------------------------------------
# Remote (store-backed)
# ---------------------------------------------------------------------------


def _open_store(store_uri: str) -> RemoteStore:
    from womblex.store.remote import RemoteStore

    return RemoteStore.from_uri(store_uri)


def _list_remote_runs(store_uri: str) -> list[RunDescription]:
    store = _open_store(store_uri)
    return [_describe_remote_run(store, run_id) for run_id in store.list_dirs("runs")]


def _remote_manifest_rows(store_uri: str, run_id: str) -> list[dict] | None:
    store = _open_store(store_uri)
    if run_id not in store.list_dirs("runs"):
        return None
    return list(_remote_manifest_table(store, f"runs/{run_id}").to_pylist())


def _describe_remote_run(store: RemoteStore, run_id: str) -> RunDescription:
    prefix = f"runs/{run_id}"
    table = _remote_manifest_table(store, prefix)
    return RunDescription(
        run_id=run_id,
        document_count=table.num_rows,
        stages=_remote_stages_present(store, prefix),
        created_at=None,  # object stores don't expose this uniformly (docs/ui-plan.md §4)
        updated_at=None,
    )


def _remote_manifest_table(store: RemoteStore, prefix: str) -> pa.Table:
    """Stage *prefix*'s manifest into a temp dir and read it with the local reader.

    Prefers the consolidated ``manifest.parquet`` and falls back to the
    per-batch shard manifests, matching :func:`describe_run`'s local
    precedence — so a distributed run reads correctly both before and after
    ``womblex finalize`` has published the consolidated file.

    Assumes *prefix* names an existing run; callers check that first against
    ``list_dirs("runs")``.
    """
    manifest_key = f"{prefix}/{RUN_MANIFEST_FILENAME}"
    with tempfile.TemporaryDirectory(prefix="womblex-ui-") as tmp:
        tmp_dir = Path(tmp)
        if store.exists(manifest_key):
            local = store.download_file(manifest_key, tmp_dir / RUN_MANIFEST_FILENAME)
            return pq.read_table(str(local))
        documents_dir = tmp_dir / "documents"
        documents_dir.mkdir()
        manifest_keys = store.list_files(f"{prefix}/documents", "*._manifest.parquet")
        if manifest_keys:
            store.download_to_dir(manifest_keys, documents_dir)
        return read_manifest(documents_dir)


def _remote_stages_present(store: RemoteStore, prefix: str) -> tuple[str, ...]:
    return tuple(
        stage for stage, suffix in STAGE_SUFFIXES.items()
        if store.list_files(f"{prefix}/documents", f"*{suffix}")
    )


def _remote_stage_presence(store_uri: str, run_id: str, suffix: str) -> list[str] | None:
    store = _open_store(store_uri)
    if run_id not in store.list_dirs("runs"):
        return None
    keys = store.list_files(f"runs/{run_id}/documents", f"*{suffix}")
    if not keys:
        return []
    with tempfile.TemporaryDirectory(prefix="womblex-ui-") as tmp:
        tmp_dir = Path(tmp)
        store.download_to_dir(keys, tmp_dir)
        return _scan_stage_presence(tmp_dir, suffix)


def _remote_chunk_detail(store_uri: str, run_id: str, source_hash: str) -> dict | None:
    """Read the overlay sidecars in place, filtered to one document.

    The only reader here that does **not** stage into a temp dir first. The
    others answer whole-run questions about small files (manifests) or must
    read every shard anyway (the audit); this one wants one document out of
    sidecars that span the entire corpus, so staging them would transfer
    every chunk of every document to render one page. Reading through the
    store's own fsspec filesystem pushes the ``source_hash`` predicate into
    the parquet reader instead.
    """
    store = _open_store(store_uri)
    if run_id not in store.list_dirs("runs"):
        return None
    prefix = f"runs/{run_id}/documents"
    # `list_files` returns store-relative keys; pyarrow needs them rooted the
    # way RemoteStore roots its own filesystem calls.
    paths = {
        suffix: [f"{store.root}/{key}" for key in store.list_files(prefix, f"*{suffix}")]
        for _key, suffix, _schema, _column in _CHUNK_DETAIL_SIDECARS
    }
    return _chunk_detail(paths, source_hash, filesystem=store.fs)


def _remote_shard_audit(store_uri: str, run_id: str) -> dict | None:
    """Stage the four extraction-role shards and audit them locally.

    Restricted to ``_SHARD_SUFFIX`` rather than ``*.parquet`` on purpose:
    the audit reads only those roles, and a run's embeddings / enrichment
    sidecars are the largest files in it — globbing everything would pull
    the whole corpus over the network to count element kinds.
    """
    store = _open_store(store_uri)
    if run_id not in store.list_dirs("runs"):
        return None
    prefix = f"runs/{run_id}/documents"
    with tempfile.TemporaryDirectory(prefix="womblex-ui-") as tmp:
        tmp_dir = Path(tmp)
        for suffix in _SHARD_SUFFIX.values():
            keys = store.list_files(prefix, f"*{suffix}")
            if keys:
                store.download_to_dir(keys, tmp_dir)
        return audit_shard_directory(tmp_dir).as_dict()
