# Egress bundle contract — Womblex (REVIEW COPY)

> **Status: REVIEW COPY — partially implemented.** Of the three merges under
> "Change sizing", two have landed: `store/egress_output.py` (the
> `source_index.parquet` schema and IO) and `store/egress.py` (the bundle
> builder). The CLI verb (`womblex egress`) has not; once it lands with
> `docs/egress.md` as the canonical contract, this review copy retires. Do not
> treat it as a record of shipped behaviour.
>
> The two open review questions below were resolved pragmatically for v1:
> sources resolve via `SourceResolver` against a **local** run only (a run
> ingested from an object store is refused before anything is written — stage
> it locally first), and `source_index.parquet` keeps the four-status
> vocabulary. Revisit both if Numbatch/Echidnet feedback disagrees.

## Shared context (identical across Womblex, Numbatch, Echidnet)

<!-- BEGIN SHARED CONTEXT — keep verbatim in all three repos -->

**The change.** A single on-disk *egress bundle* becomes the one integration
surface between Womblex (the producer) and its two downstream consumers,
Numbatch and Echidnet. Womblex gains an `egress` capability that exports one
finalised run plus its raw source documents to a destination; Numbatch gains a
data-team panel that reads a bundle; Echidnet consumes a bundle's extracted
corpus with no code change.

**The bundle layout** (defined once by Womblex, read by both consumers):

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

**The boundaries between the repos are explicit and one-directional:**

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

<!-- END SHARED CONTEXT -->

## What changes in Womblex

Womblex gains the producer half of the contract: a new `egress` capability that
takes one finalised run and writes a bundle to any `RemoteStore` destination
(local directory, S3, MinIO, GCS), then stops.

### New modules

- `store/egress.py` — orchestrates one run into one bundle: copy `corpus/`,
  resolve and copy raw sources, write `source_index.parquet`, write
  `egress_manifest.json`. Reuses `RemoteStore` for all I/O,
  `SourceResolver.resolve_all()` for local-run hash verification, and the
  existing manifest consolidation.
- `store/egress_output.py` — `source_index.parquet` schema and IO,
  self-contained in the manner of the other `store/*_output.py` modules.

### Source resolution and copy

Raw sources are copied through `RemoteStore` off each manifest row's
`ingest_root` and `source_relpath`, so a run ingested from `file://` and one
ingested from `s3://` are handled by one path. On a locally-ingested run,
`SourceResolver.resolve_all()` is layered over that copy to verify bytes by
hash and to populate the resolution report.

Resolution is non-fatal per document. Each `source_index.parquet` row carries a
status drawn from the existing `SourceResolver` vocabulary — `resolved`,
`hash_mismatch`, `not_found`, `unsupported_basis` — with a null raw key where no
file was produced. A records-ingested document (hashed over id and text, not
file bytes) resolves `unsupported_basis` and has no entry under `sources/`; this
is reported, never an error.

Raw files are deduplicated by `source_hash`: two manifest rows sharing one hash
(the same bytes ingested under two names) point at one file under `sources/`,
and the index carries both documents against that shared key.

### CLI

A new verb beside `finalize` in `cli/cloud.py`:

```
womblex egress <run> --to <dest> [--sources/--no-sources] [--corpus-only]
```

`finalize` consolidates a run's manifest *in place*; `egress` *exports* a run to
a destination. `<dest>` is any `RemoteStore` URI, so a bucket for a live
consumer and a local directory for an air-gapped handoff are the same code path.

### Out of scope for v1

Rendered page images. A consumer that wants pixels renders the raw file itself;
the bundle carries raw plus extracted only.

### Documentation to land with the code

`docs/egress.md` as the canonical contract, plus the module additions to
`docs/architecture.md`, `docs/project-structure.md`, and the `CLAUDE.md` module
table.

### Change sizing

Over the 500-line merge cap as one change; split into sequential merges that
each pass on their own — `egress_output` schema/IO/tests, then the bundle
builder (both copy paths) with tests, then the CLI verb with docs.

## Review questions specific to Womblex

- Is copying raw sources by `(ingest_root, source_relpath)` through
  `RemoteStore` the right primary path, with `SourceResolver` as the local-only
  verification layer?
- Is the `source_index.parquet` status column (reusing the four `SourceResolver`
  statuses) the right shape for consumers, versus a leaner resolved/unresolved
  flag?
