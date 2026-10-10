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
| `caption` | figure / table / chart caption — emitted by the PP-DocLayout-M layout model's `figure_title` / `table_title` / `chart_title` classes on OCR'd pages (`ingest/layout_onnx.py` `LABEL_MAP`) |
| `header` | short text in top 8% of page (letterhead-style content) |
| `footer` | page-number footer or short text in bottom 8% of page |
| `footnote` | sub-paragraph note — emitted by the PP-DocLayout-M layout model's `footnote` class on OCR'd pages |
| `signature` | signatory block (reserved; not currently emitted) |
| `figure` | layout-detected visual region (no extracted image data). A full-page scan whose dominant layout region is a figure but which OCR's to substantial text (≥5 words) is reclassified to `paragraph` so its content reaches chunking — only sparse regions (page-number stamps, bare logos) stay `figure` |
| `image` | extracted image with alt text |
| `table` | table; cells nest on `Element.cells` in memory, flatten to a sidecar in parquet |
| `form` | form region; fields nest on `Element.fields`, flatten to a sidecar in parquet |
| `page_break` | one per page transition (N-1 for an N-page document); `text`/`bbox` empty, `page` is the page just begun |
| `sheet_meta` | one per worksheet in a spreadsheet (carries sheet index, dimensions; title/metadata rows found above the real header land verbatim on `meta["preamble"]`) |
| `sheet_cell` | one per non-empty spreadsheet cell, **plus** every cell a merge covers (blank in the source but structurally part of the merge). Row 0 is always the detected header row (`meta["is_header"]`). A cell that is the top-left (anchor) of a merged range carries the merge's address (e.g. `"A1:C1"`) on `merge_range`; the covered cells carry an empty `value` and no `merge_range`. Ordinary blank cells (outside any merge) stay absent |

## Text policy

Text is **verbatim from the producing extractor**. Extraction applies
no post-processing — no footer stripping, no quote-glyph fixes, no
whitespace normalisation. If a source has typos, the extraction
preserves them. If an extractor produces wrong bytes due to its own
bug (e.g. broken ToUnicode font maps), the fix belongs in the
extractor, not as a post-processing pass.

Downstream stages may apply their own cleaning to `pages[i].text`,
but the on-disk parquet always reflects extraction-time content.

**Scope of the verbatim guarantee.** The guarantee covers
`*.elements.parquet` only. Chunks are built from `elements` (via `reassemble_narrative` over TEXT_KINDS
elements joined with `\n\n`), so `*.chunks.parquet` text is
*extraction-verbatim* as well when `processing.text_source` is `elements`
(the default). With `text_source` set to `normalised` or `spellfix`,
that cleaning layer's element-text overlay is applied before
reassembly (`process/text_overlay.py`), so chunk text is the cleaned
text, not the raw extraction. In-memory `pages[i].text` mutations from
PII / redact-blackout do not flow to chunks;
consumers that need masked text read the `*.clean_text.parquet`
sidecar the `pii` stage writes (`store/pii_output.py`).

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

### elements.parquet

| column | type | notes |
|---|---|---|
| `source_hash` | string | SHA-256 of the source file bytes |
| `collection_id` | string | caller-supplied batch / dataset identifier |
| `elem_order` | int32 | monotonic across the whole source |
| `kind` | string | one of the kinds in the table above |
| `extractor` | string | producing extractor (`native_text`, `ocr_paddle`, `docx`, `xlsx`, `spreadsheet_print`, `figure_image`, …) |
| `confidence` | float32 | 0–1, extractor-reported |
| `page`, `bbox` | int32 / struct | document layout; null for anything not laid out by the PDF / image path (DOCX, text, spreadsheets) |
| `text`, `alt_text` | string | content for text-bearing kinds and images |
| `header_rows` | list&lt;int32&gt; | for `kind='table'`, the row indices that act as headers |
| `sheet`, `row`, `col` | string / int32 | spreadsheet location |
| `value`, `value_type`, `formula`, `number_format`, `merge_range` | string | spreadsheet cell payload; `merge_range` is set on a merge's anchor cell only; `formula` is in the schema but never set by ingest (always null) |
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
| `bbox` | struct&lt;x, y, width, height&gt; (float32), nullable |

`bbox` locates the cell on its page, normalised 0–1 with a top-left origin
like `elements.bbox`. It is set for tables the native PDF finder (pdfplumber
`TableFinder`) produced, from the finder's own cell rectangles; a merged-away
cell the finder reports as absent, DOCX cells, OCR-reconstructed tables and
spreadsheet-print tables carry null. Files written before contract `1.3`
read back with a null `bbox`.

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
| `source_hash`, `collection_id`, `doc_id`, `filename`, `ext` | string |
| `ingest_root` | string — scheme-qualified corpus root (`file://…`, `s3://…`) |
| `source_relpath` | string — path under that root; root + relpath names the document |
| `extraction_method` | string |
| `elements_count`, `table_cells_count`, `form_fields_count` | int64 |
| `status` | string — `completed` (only completed results are written; a failed document gets no row) |
| `error` | string — empty on success |
| `extracted_at_iso` | string |
| `parser_version` | string |
| `content_digest` | string (SHA-256 over ordered elements; null in shards written before the column) |

All four shard files also carry the ingest root, the collection and the
batch's relative paths in their Parquet footer key-value metadata, under
`womblex.*` keys, alongside the run-stamp and contract keys every pipeline
Parquet carries. Footer metadata is additive: a reader that ignores it reads
the file unchanged. `ingest_root` is declared by `paths.ingest_root` (or,
unset, by `paths.input_root`) and is never inferred from the working
directory; both `ingest_root` and `source_relpath` are empty for a writer that
declared no root. Masking never rewrites the manifest — it is not a masking
surface, and a `pii` run over a completed run leaves it byte-identical.

### *.redactions.parquet (optional sidecar)

Written by `womblex.redact.batch.annotate_redactions_for_shards` as an
opt-in sibling alongside the four canonical shards. One row per
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
annotations land.

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
