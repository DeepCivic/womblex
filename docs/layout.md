# Layout

Layout analysis finds the regions of a page (paragraph, heading, table, figure,
header, footer and so on) and records them in `*.layout_regions.parquet`. One
model runs once per selected page in the extraction batch, so what it found can
be inspected on disk and compared across models and settings.

Two things read the regions. **OCR:** table reconstruction and the block types
on a scanned page come from the layout step's output; a page whose layout is
`empty` or `error` falls back to whole-page OCR text. **Redaction:** the same
regions are its exclusion zones. Neither runs a model of its own, and the old
`extraction.ocr.layout_model` / `redaction.layout_model` keys (and their
`layout_options`) are refused at config load: use `layout.model` /
`layout.options`. Why layout is shaped this way is in
[decisions.md](decisions.md).

## Settings

A top-level `layout:` section in the config:

| Setting | Default | Meaning |
|---|---|---|
| `model` | `pp-doclayout-m` | A name registered in the layout slot. A tuned model is a `womblex.models.layout` plugin ([model-plugins.md](model-plugins.md)). An unknown name stops the run, listing the known ones. |
| `options` | `{}` | Passed unchanged to the model's factory. |
| `page_scope` | `consumers` | Which pages are analysed; see below. |

Pages are rendered at `extraction.ocr.dpi`, so regions and OCR describe the same
render. The pre-run model check (`processing.models_check`) loads `layout.model`
whenever the step would use it.

### Page scope

- `consumers` analyses only the pages something reads layout on: OCR-routed
  pages (an OCR engine with the `markdown` trait bypasses layout, so it
  contributes none), plus, when `redaction.enabled` and
  `redaction.use_layout_filter` are on, every page with no vector redaction.
- `all` analyses every page of every PDF or image. Use it to measure a model.

Documents the PDF seam does not open (DOCX, spreadsheets, text, and the records
ingest) are never analysed and write no rows.

## Redaction's exclusion zones

On a page with no vector redaction, redaction renders the page and runs a
contour detector. With `redaction.use_layout_filter` on, the page's `figure` and
`table` regions are exclusion zones: a dark region whose centre falls inside one
is not reported. Regions are normalised, so they map onto redaction's own render
(150 dpi) whatever dpi layout ran at. `redaction.use_layout_filter` decides
whether redaction reads the regions; `layout.model` decides which model made
them. `womblex redact --shards` reads each batch's `*.layout_regions.parquet`;
`womblex run` hands over the regions it just found.

A page whose layout row has `status = error`, or that has no row, is detected
with no exclusion zones; one warning per document names those pages, and the
report carries them as `RedactionReport.unfiltered_pages`. An `empty` page is a
real "nothing to exclude" and does not count.

### Listing unfiltered pages

`store.layout_output.unfiltered_redaction_pages(path)` returns the
`(source_hash, page)` pairs redaction ran on without exclusion zones, for one
file or a directory holding the layout sidecars and any `*.redactions.parquet`:

```python
from womblex.store.layout_output import unfiltered_redaction_pages

unfiltered_redaction_pages("shards/")   # [("<source_hash>", 2), ...]
```

It counts a page when its layout row has `status = error` **and** the file's
footer key `womblex.layout_redaction_consumed` is `true`; a file whose footer
says `false`, or has none, asked nothing of the regions. The equivalent query:
read the footer, skip the file unless it says `true`, then select the distinct
`(source_hash, page)` where `status = 'error'`. The list over-reports slightly,
because a page redaction resolved from vector drawings never needed the filter.

`womblex redact --shards` can meet pages the run never analysed (shards
extracted with redaction off have no layout rows for native pages), so it
records the pages it ran unfiltered in each `*.redactions.parquet` footer under
`womblex.redaction_unfiltered_pages` (`{source_hash: [page, ...]}`), and the
query adds those. Rerunning `run-stage layout` with redaction enabled gives
those pages layout.

## The sidecar

`batch-NNNN.layout_regions.parquet` sits beside the batch's other shards. It is
written whenever the batch ran with a layout configuration, even with no rows,
so a shard set is complete.

| Column | Meaning |
|---|---|
| `source_hash`, `page` | Key. `page` is 0-based, as on elements. |
| `region_order` | Order within the page, top to bottom. Null on a status row. |
| `bbox` | `{x, y, width, height}`, normalised 0-1 from the top-left, the same convention as element `bbox`. Null on a status row. |
| `label` | The model's own class name (for example `paragraph_title`). |
| `block_type` | Womblex vocabulary: `paragraph`, `heading`, `list_item`, `caption`, `header`, `footer`, `footnote`, `signature`, `figure`, `table`. |
| `confidence` | 0-1. Present on every region row. |
| `status` | `ok`, `empty` or `error`. |
| `error` | Why, on an `error` row. |

An analysed page always has a row. `empty` means the model ran and found
nothing; `error` means analysis failed (the model could not load, the render
failed, or the model's output was malformed). The two are never the same row. A
page with no row at all was not selected by `page_scope`. A failed page does not
fail its document.

The file holds no document text, so its contract sensitivity is `none`.

### Reading it

```python
import pyarrow.compute as pc
from womblex.store.layout_output import read_layout_regions

regions = read_layout_regions("out/documents")  # a shard directory, or one file
tables = regions.filter(pc.equal(pc.cast(regions["block_type"], "string"), "table"))
failed = regions.filter(pc.equal(pc.cast(regions["status"], "string"), "error"))
```

To place a region on a page image, multiply `x` and `width` by the image width
and `y` and `height` by its height; any render of the page works.

## Fingerprint

The footer of the sidecar, and of the batch's `*.elements.parquet`, carries a
fingerprint under `womblex.layout_fingerprint`: the model name, a model digest,
a digest of `options`, the render dpi, the page scope, its `consumers` (`ocr`,
`redaction`: what selects pages under `consumers` scope) and the schema version.
Equal fingerprints mean the same model and settings over the same pages.

`model_digest` is the content digest of the model's local files for a built-in
model, and `<distribution>==<version>` for a plugin, which Womblex cannot digest.

The sidecar's footer also records, under `womblex.layout_redaction_consumed`,
whether redaction was configured to consume layout (`redaction.enabled` and
`use_layout_filter`); always `false` once `run-stage layout` rewrites it.

The elements footer records the layout that extraction ran with. A shard written
before this existed has no fingerprint there; that reads as provenance unknown,
not as a mismatch.

### Is a sidecar out of date?

```python
from womblex.store.layout_output import layout_fingerprint_statuses

layout_fingerprint_statuses("shards/")   # {"batch-0001": "match", ...}
```

For each batch it compares the sidecar's fingerprint with the one in the
elements footer:

| Status | Meaning |
|---|---|
| `match` | Both carry a fingerprint and they are equal: the sidecar describes the layout extraction consumed. |
| `mismatch` | Both carry one and they differ: the sidecar was rerun under another model or setting, so the elements (tables, redaction) were built from different regions. |
| `unknown` | One side carries none: a run extracted before the layout stage, or a batch with no sidecar. This is "provenance unknown", never a mismatch. |

## Rerunning layout

```bash
womblex run-stage --stage layout --shards shards/ --config config.yaml
```

Reruns the layout step against the source documents and replaces each batch's
`*.layout_regions.parquet`. Use it to measure another model or setting on a run
already extracted: change `layout.model`, `layout.options` or
`layout.page_scope` in the config (or turn on redaction's layout filter, which
widens the pages `consumers` selects) and rerun. A batch whose sidecar already
carries the config's fingerprint is skipped; `--force` reruns it anyway.

What a rerun changes, and what it does not:

- Only the layout sidecar is replaced. Elements, tables and redaction results
  stay as extracted, so a rerun under a different model reads as `mismatch`.
  Applying a new model to tables or redaction means re-extracting.
- Sources are found through the manifest's recorded ingest root and paths, and
  checked by hash. `--ingest <dir>` names the corpus where it has moved. A
  document that cannot be found or analysed is logged with its source hash, and
  its batch is left exactly as it was (a sidecar is never half one model and
  half another).
- DOCX, spreadsheets and text carry no layout and are not opened.
- `layout` is not in the stages `enqueue-stages` dispatches: a rerun is
  deliberate.

### On a distributed run

```bash
womblex run-stage --stage layout --store s3://bucket --run-id <run_id> \
    --ingest s3://bucket/ingest --config config.yaml
```

The runner takes each batch's shards from the store, and for a batch that needs
rerunning fetches each PDF or image from the ingest location (`--ingest`, else
`$WOMBLEX_INGEST_URI`, else the store itself). A document's key is the
`source_relpath` its manifest row recorded at extraction; the downloaded bytes
are checked against `source_hash`. A document that is missing from the ingest
store, has moved, or whose bytes differ is an error for that document, which
fails its batch and leaves the published sidecar as it was. A batch whose
sidecar fingerprint already matches fetches nothing. The new sidecar is
published atomically, and a failed batch does not stop the others.
