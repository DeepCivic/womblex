# Layout

Layout analysis finds the regions of a page (paragraph, heading, table, figure,
header, footer and so on) and records them in `*.layout_regions.parquet`. One
model runs once per selected page in the extraction batch, so what it found can
be inspected on disk and compared across models and settings.

This is the first step of moving layout out of OCR and redaction into a stage of
its own (see [plan-permissive-deps.md](plan-permissive-deps.md)). **Today
nothing reads the sidecar.** OCR table reconstruction and redaction's exclusion
zones still run their own analyser (`extraction.ocr.layout_model`,
`redaction.layout_model`), so the regions here do not change any element. The
sidecar is what a later release will switch them over to.

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
a digest of `options`, the render dpi, the page scope and the sidecar schema
version. Two files with equal fingerprints were produced by the same layout
model and settings.

`model_digest` is the content digest of the model's local files for a built-in
model, and `<distribution>==<version>` for a plugin, which Womblex cannot digest.

The sidecar's footer also records, under `womblex.layout_redaction_consumed`,
whether redaction was configured to consume layout (`redaction.enabled` and
`use_layout_filter`).

The elements footer records the layout that extraction ran with. A shard written
before this existed has no fingerprint there; that reads as provenance unknown,
not as a mismatch.
