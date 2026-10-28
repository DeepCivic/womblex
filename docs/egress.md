# Egress bundle contract — Womblex

The one integration surface between Womblex (the producer) and its two
downstream consumers, Numbatch and Echidnet. Womblex exports one finalised
run plus its raw source documents to a destination as a single on-disk
*bundle* (by `womblex egress`, or as a workflow's delivery, below); Numbatch reads a bundle for its data-team review panel; Echidnet
consumes a bundle's extracted corpus with no code change.

The per-file contract version, sensitivity labels and join keys are in
[contract.md](contract.md).

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
    <source_hash><ext>        # the raw file each document was extracted from (ext keeps its dot)
  source_index.parquet        # source_hash -> raw key, ext, doc_id, filename, status
  egress_manifest.json        # bundle descriptor (incl. contract_version) + aggregate source counts by status
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

Womblex implements the producer half of the contract: `egress` takes one
finalised run, local or held in an object store, and writes a bundle to any
`RemoteStore` destination (local directory, S3, MinIO, GCS), then stops.

### Modules

- `store/egress.py` — `build_bundle(run_root, store, *, run_id, bundle_prefix=None,
  include_sources=True, source_root=None)` orchestrates one run into one
  bundle: mirror `documents/` (recursively, not just the top-level `*.parquet`
  shards) plus `manifest.parquet` under `corpus/`, resolve and copy raw sources,
  write `source_index.parquet`, write `egress_manifest.json`. The run is read
  through a `RemoteStore` (a local directory through fsspec's local backend),
  so a local run and one held in an object store are one code path, copied
  store to store with no local staging; it reads the already-consolidated
  `manifest.parquet` in place (it consolidates nothing itself).
- `store/egress_sources.py` — one outcome per `source_hash`: `LocalSources`
  (a local corpus, through `SourceResolver.resolve_all()`) and `StoreSources`
  (the object store the run was ingested from, by `source_relpath`).
  `egress_manifest.json` holds aggregates only — `run_id`, `contract_version`,
  `womblex_version`, `created_at_iso`, `documents`, `sources_copied`,
  `sources_by_status`; per-document status lives in `source_index.parquet`.
- `store/egress_output.py` — `source_index.parquet` schema and IO,
  self-contained in the manner of the other `store/*_output.py` modules.

### Source resolution and copy

Sources come from the manifest's single `ingest_root` (or a `--source-root`
override, a local path or a store URI); a run declaring none or several is
refused before anything is written.

- **A `file://` root** — `SourceResolver.resolve_all()` verifies each
  document's bytes by hash against the local corpus, and each resolved path is
  uploaded to the destination.
- **An object-store root** — each document is fetched from that store by its
  `source_relpath`, streamed to a temporary key in the bundle while it is
  hashed, and moved to `sources/<source_hash><ext>` only when the hash equals
  `source_hash`; a mismatched copy is deleted and reported `hash_mismatch`, a
  missing key `not_found`. A store that cannot be read at all is refused before
  the first write. Credentials come from the environment, as for any store.

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

- `<run>` — a finished run root holding `documents/` and `manifest.parquet`:
  a local directory (as `womblex run` or `womblex manifest` produces) or an
  object-store URI (`s3://bucket/runs/<run>`, as a fleet run's coordinator or
  `womblex finalize` writes it), copied store to store.
- `--to` — any `RemoteStore` URI: a local directory for an air-gapped handoff,
  a bucket for a live consumer.
- `--run-id` defaults to `<run>`'s own directory name (its last path segment) and must agree with the
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

## Delivery to a workflow's destinations

A workflow file can name destinations ([workflows.md](workflows.md)); after its last
step, failed or not, the run is delivered to each. `run-workflow` delivers a local
run, a fleet run's coordinator delivers from the store, and `womblex deliver` delivers
again. This section is what each destination is sent. Credentials are never in the
workflow: each destination names the environment variables that hold them, and they
are checked before the run starts.

**Object store** (`kind: object_store`) — the bundle above, under `<uri>/<run_id>/`,
exactly as `womblex egress` writes it, with or without `sources/` and
`source_index.parquet` as `include_sources` says. Its `egress_manifest.json` also
carries a `delivery` block: the workflow, the destination's name and `after_failed`,
the steps that had failed when it was delivered (empty when none had). Raw sources
are original documents, as above.

**Postgres** (`kind: postgres`) — one row per embedded chunk in `table` (default
`womblex_chunks`), created if absent (with the `vector` extension, which pgvector lets
only a superuser create, so for any other role it is created once beforehand):

| Column | Holds |
|---|---|
| `run_id`, `source_hash`, `content_type`, `chunk_index` | The key; `source_hash` joins the bundle and the manifest |
| `text` | The chunk's text in `text_layer` |
| `text_layer` | `clean_text` (masked by the `pii` step; the default) or `chunks` (raw, only with `text_layer: raw`) |
| `mask_status` | For masked text: `masked`, `no_entity` or `not_masked` (no candidate source covered it, so the text is verbatim) |
| `model`, `embedding` | The embedding model and its vector (a pgvector `vector`) |
| `delivered_at` | When this delivery wrote the row |

A chunk the layer holds no text for is not sent from the other layer; it is left out
and counted in the delivery record (`rows_skipped`). No sources and no other sidecar
are sent.

**Webhook** (`kind: webhook`) — the run's events, never its data: `extraction_settled`,
`stage_started`, `batch_progress`, `stage_finished`, `stage_failed`, `delivered`,
`delivery_failed` and `run_finished`; an optional `events` list narrows them.
Each is a JSON `POST` with `event`, `event_id`, `run_id`, `workflow`, `status` (the
run's state as `/v1/runs` words it: `running`, then `done`, `failed` or `cancelled`), `at`, and the event's own `stage`, `destination`,
`error`, `ran_after_failed` or `after_failed`; `batch_progress` carries `stage`, `done`, `failed` and `total`. Headers: `X-Womblex-Event`,
`X-Womblex-Event-Id` and `X-Womblex-Signature: sha256=<hex>`, the HMAC-SHA256 of the raw
body under the key in `secret_env`. A receiver recomputes the signature before
trusting a body and drops an `event_id` it has seen, since an event is resent under
the same id.

**Retries and records.** A delivery or event is tried three times with backoff; one
that still fails is recorded and never stops the run. Each destination's record is
`<run>/deliveries/<name>.json` (status, attempts, error, `after_failed`, counts, and a
webhook's `events`), and the run record in `manifest.parquet` lists every one.
Delivering again replaces: the bundle is rewritten, the rows upserted (the run's rows
the delivery did not write are removed), and the record replaced. Nothing redelivers
by itself; after a successful `retry-stage`, the command prints the `womblex deliver`
line.

### Out of scope

Rendered page images. A consumer that wants pixels renders the raw file
itself; the bundle carries raw plus extracted only.

## Limitations

- `source_index.parquet` carries the four-status `SourceResolver` vocabulary
  plus `upload_failed`, not a resolved/unresolved flag. A consumer that wants
  the binary view treats every status but `resolved` as unresolved.
