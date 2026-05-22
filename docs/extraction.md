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
| `caption` | figure / table caption (reserved; not currently emitted — see STATUS.md K6) |
| `header` | short text in top 8% of page (letterhead-style content) |
| `footer` | page-number footer or short text in bottom 8% of page |
| `signature` | signatory block (reserved; not currently emitted — see STATUS.md K1) |
| `figure` | layout-detected visual region (no extracted image data) |
| `image` | extracted image with alt text |
| `table` | table; cells nest on `Element.cells` in memory, flatten to a sidecar in parquet |
| `form` | form region; fields nest on `Element.fields`, flatten to a sidecar in parquet |
| `page_break` | one per page transition (N-1 for an N-page document); `text`/`bbox` empty, `page` is the page just begun |
| `sheet_meta` | one per worksheet in a spreadsheet (carries sheet index, dimensions) |
| `sheet_cell` | one per non-empty spreadsheet cell |

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
`*.elements.parquet` only. Chunks are built from `full_text` (derived
from `pages[i].text`) after downstream stages have mutated it in
place — so chunks reflect PII replacement, blackout redaction, and
any other `pages`-level mutation. They are *post-stage* text, not
extraction-time verbatim.

---

## Parquet output

Each batch writes four sibling parquet files. The shard base name
is the caller's choice (e.g. `batch-0001`):

```
batch-0001.elements.parquet     # one row per element
batch-0001.table_cells.parquet  # children of kind='table' elements
batch-0001.form_fields.parquet  # children of kind='form' elements
batch-0001._manifest.parquet    # one row per source file
```

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

**Sidecar pattern.** Future post-extraction operations (PII, chunk
persistence, etc) can follow the same shape: sparse parquet with
`(source_hash, elem_order)` as the join key, LEFT-JOIN-with-default
semantics, opt-in (absence is a valid state). Keeps the elements shards
canonical and avoids rewriting them when downstream annotations land.

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
