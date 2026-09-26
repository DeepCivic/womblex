# Egress bundle contract — Womblex

The one integration surface between Womblex (the producer) and its two
downstream consumers, Numbatch and Echidnet. Womblex exports one finalised
run plus its raw source documents to a destination as a single on-disk
*bundle*; Numbatch reads a bundle for its data-team review panel; Echidnet
consumes a bundle's extracted corpus with no code change.

## The bundle layout

Defined once by Womblex, read by both consumers:

```
<dest>/<run_id>/
  corpus/                     # a standard Womblex shard directory, unchanged
    batch-*.elements.parquet
    batch-*.table_cells.parquet
    batch-*.form_fields.parquet
    batch-*._manifest.parquet
    batch-*.chunks.parquet    # + any downstream-stage sidecars the run produced
    manifest.parquet          # consolidated documents table
  sources/
    <source_hash>.<ext>       # the raw file each document was extracted from
  source_index.parquet        # source_hash -> raw key, ext, doc_id, filename, status
  egress_manifest.json        # bundle descriptor + per-document resolution report
```

Everything content-addresses by `source_hash`, so raw and extracted resolve to
each other in any consumer with no call back to the producer.

## Boundaries

- **Womblex owns the bundle format and nothing downstream of it.** It writes
  the bundle and stops — no retention, serving, versioning, or auth. It does
  not know who reads a bundle, does not import Numbatch or Echidnet, and never
  reads any consumer's state. If the format changes, that repository is where
  it changes.
- **The contract is data on disk, not an API.** The three repositories share
  the *bundle layout* only. No repository imports another; no repository calls
  another at run time. A consumer reads files; it does not reach the producer.
- **Numbatch consumes a bundle; its outputs stay in Numbatch.** Reviewer
  corrections (extraction or classification) are stored Numbatch-side and are
  **never** written back into a bundle, into `corpus/`, or into Womblex. The
  bundle is read-only to Numbatch.
- **Echidnet consumes a bundle's `corpus/` read-only.** It reads the shard
  directory it already understands and treats every byte as untrusted input.
  It writes nothing into the bundle and adds no dependency on Womblex or
  Numbatch.
- **Raw sources are original documents.** A bundle may contain unredacted
  material. Scoping access to a bundle destination is the responsibility of
  whoever runs egress and whoever hosts the destination — it is a property of
  the bundle, not of any one consumer.

## What Womblex does

Womblex gains the producer half of the contract: an `egress` capability that
takes one finalised **local** run and writes a bundle to any `RemoteStore`
destination (local directory, S3, MinIO, GCS), then stops.

### Modules

- `store/egress.py` — `build_bundle(run_root, store, *, run_id, bundle_prefix=None,
  include_sources=True, source_root=None)` orchestrates one run into one
  bundle: mirror `documents/` (recursively, not just the top-level `*.parquet`
  shards) plus `manifest.parquet` under `corpus/`, resolve and copy raw sources,
  write `source_index.parquet`, write `egress_manifest.json`. Reuses
  `RemoteStore` for all I/O, `SourceResolver.resolve_all()` for local-run hash
  verification, and the existing manifest consolidation.
- `store/egress_output.py` — `source_index.parquet` schema and IO,
  self-contained in the manner of the other `store/*_output.py` modules.

### Source resolution and copy

Raw sources are copied through `RemoteStore` off each manifest row's
`ingest_root` and `source_relpath`, so a run ingested from `file://` and one
ingested from `s3://` are handled by one path. On a locally-ingested run,
`SourceResolver.resolve_all()` is layered over that copy to verify bytes by
hash and to populate the resolution report. **Scoped to a local run** — a run
ingested from an object store is refused before anything is written; stage it
locally first (`womblex finalize` then a sync-down).

Resolution is non-fatal per document. Each `source_index.parquet` row carries a
status drawn from the existing `SourceResolver` vocabulary — `resolved`,
`hash_mismatch`, `not_found`, `unsupported_basis` — plus one `egress.py` adds
of its own, `upload_failed`, for a source that resolved but whose copy to the
destination raised; each is isolated per source, so one failed upload never
aborts the export. A records-ingested document (hashed over id and text, not
file bytes) resolves `unsupported_basis` and has no entry under `sources/`; this
is reported, never an error.

Raw files are deduplicated by `source_hash`: two manifest rows sharing one hash
(the same bytes ingested under two names) point at one file under `sources/`,
and the index carries both documents against that shared key.

Re-exporting into a bundle folder that already has a `sources/` directory
removes any file under it that the fresh export neither wrote nor attempted —
so a document dropped from the manifest since the last export does not linger
as an orphan `source_index.parquet` no longer names. A source whose upload
just failed is left as-is either way, since its destination content (if any
landed before the failure) is of unknown provenance.

### CLI

```
womblex egress <run> --to <dest> [--run-id ID] [--bundle-prefix PREFIX]
                      [--sources | --no-sources | --corpus-only]
                      [--source-root ROOT]
```

- `<run>` — a finished local run root (holds `documents/` and
  `manifest.parquet`, as `womblex run` or `womblex manifest` produces).
- `--to` — any `RemoteStore` URI: a local directory for an air-gapped handoff,
  a bucket for a live consumer.
- `--run-id` defaults to `<run>`'s own directory name and must agree with the
  run's own stamp, if it has one.
- `--bundle-prefix` defaults to `--run-id`; overriding it lets the bundle land
  somewhere other than a directory named after the run.
- `--sources` (default) resolves and copies raw source documents;
  `--no-sources` / `--corpus-only` skip resolution entirely and write no
  `sources/` or `source_index.parquet` — a corpus-only export.
- `--source-root` overrides where raw sources resolve from (a moved corpus);
  default is the run's own recorded ingest root.

`finalize` consolidates a run's manifest *in place*; `egress` *exports* a run
to a destination.

### Out of scope for v1

Rendered page images. A consumer that wants pixels renders the raw file
itself; the bundle carries raw plus extracted only.

## Resolved review questions

Two questions were open while this contract was under review; both were
resolved pragmatically for v1 and are settled unless Numbatch/Echidnet
feedback disagrees:

- Sources resolve via `SourceResolver` against a **local** run only — a run
  ingested from an object store is refused before anything is written.
- `source_index.parquet` keeps the four-status `SourceResolver` vocabulary
  (plus `egress.py`'s own `upload_failed`), rather than a leaner
  resolved/unresolved flag.
