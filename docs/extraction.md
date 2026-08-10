# Extraction output

Extraction reads a source file and produces an ordered stream of
elements. An element is one thing a reader sees: a paragraph, a
heading, a table, a form, an image. Spreadsheets get cell-grained
elements because cells are what a spreadsheet *is*.

This document is the canonical reference for what comes out of the
extraction step.

---

## In-memory model

[`womblex.ingest.elements.Element`](../src/womblex/ingest/elements.py)
is the canonical structural unit. An `ExtractionResult` carries:

| field | role |
|---|---|
| `elements: list[Element]` | the ordered structural stream — what the parquet writer serialises |
| `pages: list[PageResult]` | per-page concatenated text; mutable so downstream PII / redaction can rewrite `page.text` |
| `metadata` | document-level capture metadata (strategy, confidence, page count, content mix) |
| `warnings`, `error`, `document_id`, `redaction_report` | provenance and status |

The legacy view properties `.text_blocks` / `.tables` / `.forms` /
`.images` remain on `ExtractionResult` as read-only derivations from
`elements`. They exist for downstream stages that have not migrated.
Mutating them has no effect on the underlying elements.

`pages` is **not** derived. PII and redaction stages mutate
`page.text` in place. The on-disk parquet retains extraction-time
verbatim text because the writer reads `elements`, not `pages`.

## Element kinds

| kind | when |
|---|---|
| `paragraph` | prose block (default for unclassified text) |
| `heading` | heading-styled prose (large font, or bold short non-sentence text) |
| `list_item` | sub-paragraph marker `(a)` / `(i)` / `(1)` / bullet `•·-*` at start of block |
| `caption` | figure / table caption — emitted by the DocLayNet layout model's `Caption` class on OCR'd pages |
| `header` | short text in top 8% of page (letterhead-style content) |
| `footer` | page-number footer or short text in bottom 8% of page |
| `signature` | signatory block (reserved; not currently emitted) |
| `figure` | layout-detected visual region (no extracted image data). A full-page scan whose dominant layout region is a figure but which OCR's to substantial text (≥5 words) is reclassified to `paragraph` so its content reaches chunking — only sparse regions (page-number stamps, bare logos) stay `figure`. See [decisions.md](decisions.md) "Element-kind classification" |
| `image` | extracted image with alt text |
| `table` | table; cells nest on `Element.cells` in memory, flatten to a sidecar in parquet |
| `form` | form region; fields nest on `Element.fields`, flatten to a sidecar in parquet |
| `page_break` | one per page transition (N-1 for an N-page document); `text`/`bbox` empty, `page` is the page just begun |
| `sheet_meta` | one per worksheet in a spreadsheet (carries sheet index, dimensions; title/metadata rows found above the real header land verbatim on `meta["preamble"]`) |
| `sheet_cell` | one per non-empty spreadsheet cell; row 0 is always the detected header row (`meta["is_header"]`) |

## Text policy

Text is **verbatim from the producing extractor**. Extraction applies
no post-processing — no footer stripping, no quote-glyph fixes, no
whitespace normalisation. If a source has typos, the extraction
preserves them. If an extractor produces wrong bytes due to its own
bug (e.g. broken ToUnicode font maps), the fix belongs in the
extractor, not as a post-processing pass.

This is a hard reversal of the prior `_normalise_text` behaviour.
Downstream stages may apply their own cleaning to `pages[i].text`,
but the on-disk parquet always reflects extraction-time content.

**Scope of the verbatim guarantee.** The guarantee covers
`*.elements.parquet` only. As of I2 (2026-05-27), chunks are also
built from `elements` (via `reassemble_narrative` over TEXT_KINDS
elements joined with `\n\n`), so `*.chunks.parquet` text is
*extraction-verbatim* too. In-memory `pages[i].text` mutations from
PII / redact-blackout under `womblex run` no longer flow to chunks;
downstream consumers that need post-rewrite text will read it from
a future `*.clean_text.parquet` sidecar (P1, not yet written).

---

## Parquet output

The extraction stage writes four sibling parquet files per batch. The
shard base name is the caller's choice (e.g. `batch-0001`):

```
batch-0001.elements.parquet     # one row per element
batch-0001.table_cells.parquet  # children of kind='table' elements
batch-0001.form_fields.parquet  # children of kind='form' elements
batch-0001._manifest.parquet    # one row per source file
```

Downstream stages add their own siblings — the chunking stage
(`womblex chunk --shards`) writes a fifth file `batch-0001.chunks.parquet`.
See [dataflow.md](dataflow.md) for the chunks schema.

#### Chunking adapter boundary

`process/chunker.py` is a thin adapter over **semchunk 3.x** (audited
I5, 2026-05-30). semchunk owns all chunking; Womblex handles only what
semchunk can't — parquet I/O, element-stream → `ChunkInput` projection,
source-hash plumbing, and `<REDACTED>` cross-boundary repair. Every
`ChunkingConfig` field either maps directly to a semchunk parameter
(`tokenizer`, `chunk_size`, `memoize`, `cache_maxsize`,
`max_token_chars` → `chunkerify`; `overlap`, `processes`, `progress` →
`Chunker.__call__`) or is a Womblex-only concern (`enabled` stage gate,
`chunk_tables` projection). There is **no** Womblex toggle that
re-exposes a semchunk feature under a different name — semchunk's
parameters *are* the feature surface. Three defaults diverge from
upstream, each for a measured corpus reason:
`tokenizer="isaacus/kanon-2-tokenizer"` (matches the analysis side),
`chunk_size=480` (Kanon-2 window — upstream defaults to `None`, which
auto-derives the size from the tokeniser's `model_max_length`; that
path still passes through if `chunk_size` is set to `null`),
`processes=1` (Chromebook portability). The Kanon-2 tokeniser is free on
Hugging Face (vendored under `_models/kanon-2-tokenizer`, resolved
locally), so chunk-size counting is exact and offline; the chunk stage
still gates on a configured Isaacus deployment — `ISAACUS_API_KEY` or
`ISAACUS_SAGEMAKER_ENDPOINTS`
(`womblex.utils.availability.isaacus_available`) — because **AI chunking**
(`chunking_model`) calls the API, and that gate is conservative for plain
token chunking. `offsets=True` is pinned in the adapter because Womblex
always needs char offsets for page mapping.

### elements.parquet

| column | type | notes |
|---|---|---|
| `source_hash` | string | SHA-256 of the source file bytes |
| `collection_id` | string | caller-supplied batch / dataset identifier |
| `elem_order` | int32 | monotonic across the whole source |
| `kind` | string | one of the kinds in the table above |
| `extractor` | string | producing extractor (`native_text`, `ocr_paddle`, `docx`, `xlsx`, `spreadsheet_print`, `figure_image`, …) |
| `confidence` | float32 | 0–1, extractor-reported |
| `page`, `bbox` | int32 / struct | document layout; nullable for non-PDF / non-DOCX |
| `text`, `alt_text` | string | content for text-bearing kinds and images |
| `header_rows` | list&lt;int32&gt; | for `kind='table'`, the row indices that act as headers |
| `sheet`, `row`, `col` | string / int32 | spreadsheet location |
| `value`, `value_type`, `formula`, `number_format`, `merge_range` | string | spreadsheet cell payload |
| `meta` | map&lt;string,string&gt; | parser-specific overflow |

### table_cells.parquet

Joined to `elements.parquet` by `(source_hash, parent_elem_order)`
matching `(source_hash, elem_order)` with `kind='table'`.

| column | type |
|---|---|
| `source_hash` | string |
| `parent_elem_order` | int32 |
| `row`, `col`, `rowspan`, `colspan` | int32 |
| `value`, `value_type` | string |

### form_fields.parquet

Joined by `(source_hash, parent_elem_order)` matching elements with
`kind='form'`.

| column | type |
|---|---|
| `source_hash` | string |
| `parent_elem_order` | int32 |
| `field_index` | int32 |
| `name`, `value`, `field_type` | string |

### _manifest.parquet

One row per source file in the batch.

| column | type |
|---|---|
| `source_hash`, `collection_id`, `filename`, `ext` | string |
| `extraction_method` | string |
| `elements_count`, `table_cells_count`, `form_fields_count` | int64 |
| `status` | string — `completed` or `error` |
| `error` | string — empty on success |
| `extracted_at_iso` | string |
| `parser_version` | string |

### *.redactions.parquet (optional sidecar)

Written by `womblex.redact.batch.annotate_redactions_for_shards` as an
opt-in 5th sibling alongside the four canonical shards. One row per
element on a page where redactions were detected; elements without nearby
redactions have no row.

| column | type | notes |
|---|---|---|
| `source_hash` | string | FK to elements.parquet |
| `elem_order` | int32 | FK to elements.parquet |
| `has_redaction` | bool | always `true` in this artefact; absence-from-table means `false` |

Not part of the `verify_shard_persistence` integrity set. Consumers should
LEFT JOIN and treat `has_redaction IS NULL` as `false`:

```sql
SELECT e.*, COALESCE(r.has_redaction, FALSE) AS has_redaction
FROM elements e
LEFT JOIN redactions r
  ON r.source_hash = e.source_hash AND r.elem_order = e.elem_order
WHERE e.source_hash = :h
ORDER BY e.elem_order;
```

**Sidecar pattern.** Post-extraction operations follow the same shape:
sparse parquet keyed by `source_hash` (plus `elem_order` for
element-level sidecars, or offset ranges for chunk-level), LEFT-JOIN-
with-default semantics, opt-in (absence is a valid state). Keeps the
elements shards canonical and avoids rewriting them when downstream
annotations land. As of I2 (2026-05-27), chunks land via this pattern
(`batch-NNNN.chunks.parquet`, schema in
[dataflow.md](dataflow.md)). PII / `clean_text` sidecars (P1) are
next.

---

## Reassembly

Read elements in `elem_order` and render each by `kind`. This
single query reproduces a faithful structural rendering of the
source for any document or spreadsheet:

```sql
SELECT elem_order, kind, page, text, value, sheet, row, col
FROM elements
WHERE source_hash = :h
ORDER BY elem_order;
```

For tables, join the cells sidecar:

```sql
SELECT e.elem_order, c.row, c.col, c.value
FROM elements e
JOIN table_cells c
  ON c.source_hash = e.source_hash
 AND c.parent_elem_order = e.elem_order
WHERE e.source_hash = :h AND e.kind = 'table'
ORDER BY e.elem_order, c.row, c.col;
```

For forms, replace `table_cells` with `form_fields` and `kind='form'`.

---

## Integrity

`verify_shard_persistence` runs after every batch write and checks:

- All four shard files exist and are non-empty.
- `manifest` row count matches `expected_docs`.
- Every `(source_hash, parent_elem_order)` in `table_cells` resolves
  to an element with `kind='table'`.
- Every `(source_hash, parent_elem_order)` in `form_fields` resolves
  to an element with `kind='form'`.
- The cumulative shard-directory size has not shrunk relative to
  prior batches (catches the canonical overwrite-bug signature).

Failures raise `ShardVerificationError` and halt the batch run.

---

## What changed from the previous schema

The previous output was one parquet file per batch with nested struct
lists for tables / forms / images / text_blocks. That shape made the
budget-statement class of documents (narrative interleaved with
tables) unrepresentable — the on-disk order between a table and the
paragraph before it was lost. It also forced spreadsheets to
masquerade as documents by emitting one ExtractionResult per row, a
shape that doesn't match how spreadsheets are queried.

The element-stream shape solves both: `elem_order` preserves
interleaving; spreadsheet cells are first-class. Sidecars give dense
typed access to cell content without exploding the elements table.
