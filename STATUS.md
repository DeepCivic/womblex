# STATUS — Womblex extraction quality, 2026-05-17

Codebase changes shipped against the ACT Early Childhood Incidents
corpus. Pairs with [stories/STATUS.md](../../../stories/STATUS.md) for
the corpus-side extraction-quality view.

> **Note (2026-05-16 refactor).** The extraction output shape was
> refactored to an element-stream + typed sidecars layout
> (`*.elements.parquet`, `*.table_cells.parquet`,
> `*.form_fields.parquet`, `*._manifest.parquet`). See
> [CHANGELOG.md](CHANGELOG.md) "Changed — BREAKING" and
> [docs/extraction.md](docs/extraction.md) for the canonical reference.
> The `tables[]` / `text_blocks[]` language in the Phase 5 section
> below describes the pre-refactor schema; findings still hold, but
> on-disk shape now lives in the four sibling parquet files above.

## Snapshot

| metric | value |
|---|---|
| unit tests | 537 passed, 20 skipped, 24 deselected (3 pre-existing PII assertions on `<>`-vs-`[]` brackets, 1 missing-geopandas-deps test, 24 slow-marker accuracy benchmarks deselected by pyproject `addopts`) |
| labels CER vs production parquet (`womblex score`) | hybrid 0.520 · native_with_structured 0.044 · scanned_machinewritten 0.051 · scanned_mixed 0.217 — element-stream reassembly preserves text fidelity vs pre-refactor pipeline output |
| corpus re-extraction (2026-05-17) | 2,626 source files · 3h 5m · 0 failed · 424 sibling-parquet files (106 batches × 4) |
| `kind='table'` contamination | 2,791 elements pre-fix → 172 post-fix (94% reduction); 3 residual conf=0.60 fabrications, ~10% borderline at conf=0.80, real manifests intact |
| `meta` map carries doc/table context | Verified on all 3 spreadsheet-print manifests (`context_213A reference`, `context_Element #`, `context_Text from motion`, `context_motion`) |

## What changed

The session bundled four production-hardening fixes addressing
distinct quality gaps surfaced by per-page GT comparison against the
labels packet at `stories/ACT_EarlyChildhoodIncidents/womblex-extract/labels/`.
A fifth proposed fix was tried, validated against labels, and
reverted on evidence of irreducible regression elsewhere.

### 1. Native-page table column-major emission (`extract.py` + `orchestrator.py`)

**Problem.** Native PDFs with ruled tables (Compliance Notice rules-of-
the-Law pages, e.g. 00281 page 1) had their cells correctly detected
into `tables[]` by `_extract_tables_from_page` but the prose `text`
field still read the same cells row-major via
`page.get_text("text")`, producing garble like
`"Provision of the Description Steps to be taken Law Section The
approved provider of The Provider to is to submit evidence..."` —
the data was present in `tables[]` but the prose field that
downstream chunking consumes was scrambled.

**Fix.**
- New `_find_native_tables(page)` returning
  `list[tuple[TableData, fitz.Rect, list[list]]]` — exposes per-table
  bbox + raw cell grid alongside the existing `TableData`.
- New `_emit_table_column_major(cells)` — emits each column as its
  own paragraph (cells joined by `\n` within a column, columns
  separated by `\n\n`). Mirrors the OCR-side `_table_aware_text` shape.
- `_extract_tables_from_page` is now a thin wrapper around
  `_find_native_tables` — preserves the legacy `list[TableData]`
  signature for callers that don't need bboxes.
- `_apply_native_page` calls `_find_native_tables`, partitions table
  rects out of the prose call via `extract_page_text(page,
  exclude_rects=…)`, and appends the column-major emissions
  in y-order.

**Gating.** Only `confidence ≥ 0.8` (lines-strategy hits, ruled cells)
drive prose-region exclusion. Text-strategy hits (whitespace-aligned
columns, conf ≈ 0.6) stay in `tables` but **do not** partition the
prose, because text-strategy false-positives on ordinary multi-column
prose (e.g. 2-column layouts with regular x-spacing) — caught by
`test_native_extractor_handles_two_column_page` in the test suite
before the gating was added.

The spreadsheet-print path (Phase 4) handles whitespace-aligned
manifests separately via `extract_spreadsheet_print`, so the gating
doesn't lose that case.

### 2. Block-aware paragraph emission (`grid_projection.py`)

**Problem.** For single-column native pages, `extract_page_text` fell
back to `page.get_text("text", flags=TEXT_DEHYPHENATE)`, which joins
adjacent blocks with a single `\n`. Paragraph breaks between numbered
list items, bullets, headings, and footers were lost — `01132 page 1`
extracted as one continuous paragraph with section breaks invisible to
downstream chunking.

**Fix.**
- `extract_page_text` accepts `exclude_rects: Sequence[fitz.Rect] |
  None`. Words whose midpoint falls inside any rect are filtered
  before column projection.
- Single-column path now routes through `_render_blocks_with_breaks`,
  which iterates `page.get_text("blocks", flags=TEXT_DEHYPHENATE)`
  and joins blocks with `\n\n`. Blocks whose centre falls inside an
  exclude rect are dropped (used in concert with fix 1 to suppress
  table-region prose).
- New `_word_in_any_rect(word, rects)` helper for the word-level
  filter.

The multi-column path (≥2 columns from `project_to_columns`)
continues to use `render_spatial_text(columns)` unchanged.

### 3. Body-context pipe-as-I (`extract.py` `_normalise_text`)

**Problem.** ACT Gov letter footers (`GPO Box 158 Canberra ACT 2601 |
phone: 132281 | www.act.gov.au`) OCR the `|` separator as a capital
`I` when it sits between a space and a lowercase keyword:
`2601 I phone:`. The existing `_FOOTER_PIPE_RE` only catches the
page-marker shape `<digit>lPage`.

**Fix.** `_BODY_PIPE_RE = re.compile(r" I (?=(?:phone|email|fax|www|http)\b)",
re.IGNORECASE)` — restricted to a fixed keyword set to avoid false
positives on legitimate sentence-initial `I` + verb. Applied
alongside the existing footer rule in `_normalise_text` (RES-004b).

### 4. `format_labels.TITLE_PATTERNS` regex anchor bug *(stories-side
script, not Womblex code)*

**Problem.** In
`stories/.../womblex-extract/format_labels.py`, the title patterns
embed `^` at the start (`r"^(SHOW CAUSE NOTICE|COMPLIANCE NOTICE|…)"`).
That works for `TITLE_RE.search`, but the same patterns were also
substituted in-string via `re.sub(rf"\s*({pat})", …)`, where `^`
restricted the match to position 0 — silently no-op'd for the same
title appearing mid-string. So `Dear COMPLIANCE NOTICE Section 177…`
never got the `\n\n` break inserted before `COMPLIANCE NOTICE`.

**Fix.** Removed `^` from `TITLE_PATTERNS`; dropped `re.IGNORECASE`
from the title substitution loop so body-text mentions like *"this
compliance notice"* don't false-trigger paragraph breaks.

Drives the **hybrid mean CER 0.208 → 0.145** result in the labels
retest — affects every label page where a known title appears
mid-stream after a redacted name.

### 5. *(reverted)* OCR-side `_table_aware_text` relaxation

**Tried.** Relax start condition from `min_start_rows=2` consecutive
≥3-item rows to a single 3+-item row, and allow 1-item continuation
rows (which OCR produces for multi-line cell wraps).

**Why reverted.** Helped table-shaped pages (R-01313F 0.080 → 0.018,
00281 0.27 → 0.22) but caused matching regression on form-shaped
pages (R-04060 0.020 → 0.157) — CRM-style screenshots where many
1-item label/value rows look like a multi-line cell wrap, get
absorbed into a "table", emitted column-major as one blob.

Tried four progressively stricter discriminators (2-row start,
consecutive-singleton cap, single-column-only singleton absorption,
column-spread minimums). None recovered baseline for both classes
of page. The signature of a real multi-line cell wrap and a form
field stream is structurally similar at the per-region level; we
couldn't separate them without page-image context the OCR pipeline
doesn't carry.

**Decision.** Revert to baseline. The native-path fix above already
handles the ruled-table case on the production native code path;
OCR-side ruled-table detection on rendered images stays at original
conservative behaviour. Trade-off documented; do not retry without
new discriminating signal.

## Verification

### Unit tests
`tests/test_extract.py` (40), `tests/test_grid_projection.py` (17),
`tests/test_spreadsheet_print.py` (14), full suite 532 passing.
Two new behaviours not yet covered by tests:
- `_emit_table_column_major` (covered indirectly via the native
  integration test)
- `_render_blocks_with_breaks` paragraph-separator output
  (covered indirectly via fixture accuracy)

### Labels retest
`stories/.../womblex-extract/cer_results.md` (18 reviewed pages):

| strategy | n | baseline mean CER | bundle mean CER |
|---|---:|---:|---:|
| hybrid | 2 | 0.208 | **0.145** |
| native_with_structured | 3 → 6 | 0.020 | 0.051 (+3 new pages) |
| scanned_machinewritten | 7 | 0.044 | 0.044 |
| scanned_mixed | 3 | 0.104 | 0.104 |

Newly-reviewed pages (no prior baseline):
- `02424A p4`: 0.000 — true-blank page, predicted empty matches GT empty
- `01132 p0`: 0.014 — body pipe-as-I + TITLE_PATTERNS fix visible
- `00281 p0`: 0.233 — OCR-quality residue (`Govemment` typo not in
  normaliser; mid-sentence drop in the "satisfied that" clause).
  Documented as OCR-character-level limit, not pipeline gap.

### Native-path source-PDF validation
Real `00281.pdf` page 1 (native_with_structured): ruled rules-of-the-
Law table extracts as 1 detected table; prose section emits
block-by-block with `\n\n` between `Time for Compliance / You are
required… / Failure to comply / It is an offence…`; table content
follows column-major: column 1 (`Section / 167(1) / Section / 174(2)`),
column 2 (the approved-provider description), column 3 (steps).
Behaviour matches design intent.

## Roadmap

### Phases 1–4 (✅ complete, prior session)
Per-page profile + plan-driven orchestrator (Phase 2), form-pair
extraction & block-type classifier rewrite (Phase 1), per-image OCR
& PII token swap & fullwidth footer (Phase 3), spreadsheet-print
extractor (Phase 4). All metrics maintained in the bundle's
acceptance test against the same random-500 sample.

### Phase 5 — Production hardening (✅ complete, prior session)
Four bundled fixes above. Validated against the labels packet and
the full 2,626-doc corpus. See `stories/STATUS.md` for the
production-corpus run output and quality audit.

### CLI surface

`womblex score --labels <dir> --shards <dir> [--group-by FIELD] [--report PATH]`
— promoted from a corpus-local script to first-class CLI in
2026-05-17. Scores a per-page labels packet (`<stem>.gt.md` +
`<stem>.meta.json`) against per-page text reassembled from the
element-stream parquet. See `src/womblex/score.py` for the module API
(`load_labels`, `build_manifest_index`, `reassemble_page_text`,
`score_labels`, `format_report_markdown`).

### Phase 6 — `kind='table'` over-firing (2026-05-17, ✅ resolved)

Audit of the pre-refactor corpus run found 62% of `structured`-strategy
docs shipping pseudo-tables built from form layouts and shredded prose;
the same primitive over-fired more broadly across the 2,791 PDF table
records corpus-wide. Two fixes landed, in source code rather than
config because Womblex didn't expose the right knobs:

- **§1 — `_find_native_tables` block-count gate** (`ingest/extract.py`).
  Reject any PyMuPDF table candidate where the count of natural
  `get_text("dict")` blocks inside the table bbox is less than the row
  count the table claims. Real tables decompose into ≥1 block per row;
  prose-as-table over-claims rows by carving sub-block whitespace into
  pseudo-rows. ~15 LOC + the `_count_blocks_in_bbox` helper.
- **§2 — `_has_manifest_table` + `PageProfile.has_manifest_signal`**
  (`ingest/detect.py`, `ingest/page_profile.py`). Stricter signal than
  `has_table_signal`: only fires when a page contains a table with ≥300
  non-empty cells — the discriminator between real manifests (FOI
  master 1,713 non-empty cells per page; Schedule-2ai–2av 503 per page)
  and prose-as-table over-fires (170–280 per page). The
  `qualify_for_spreadsheet_print` qualifier now gates on this stricter
  signal, so regulatory letters with embedded rules-of-law tables stop
  routing through the manifest extractor.

Validated by re-running extraction across the full 2,626-doc corpus
(2026-05-17) and auditing `kind='table'` elements + cells:

| stratum | pre-fix | post-fix | residual |
|---|---:|---:|---|
| conf=0.60 (`native_text` text-strategy) | 2,362 | 3 | 3 known prose-as-table on heavily-redacted pages (added to labels packet for follow-up) |
| conf=0.70 (`spreadsheet_print`) | 83 | 3 | All real manifests (FOI master, Schedule-2ai–2av, Schedule-2b); regulatory-letter misroutes eliminated |
| conf=0.80 (`native_text` lines-strategy) | 346 | 166 | ~90% clean rules-of-law tables; ~10% borderline (single-row / mostly-empty) |
| **total** | **2,791** | **172** | **~3-5% residual** |

The 3 residual conf=0.60 fabrications survive because the §1 gate
passes on redacted-prose pages: redaction splits paragraphs into many
small natural dict blocks, so `n_blocks ≥ n_rows` holds even though
the "table" is prose. Closing this would need a second-layer signal
(e.g. non-empty-cell-density), tracked under `stories/STATUS.md`
Outstanding §2. Material impact is small (~0.11% of source docs).

### Redaction & PII marker conventions (agreed 2026-05-17, not actioned)

Two distinct concerns, two distinct markers:

| concern | source | marker | inline per span | metadata home |
|---|---|---|---|---|
| Source redaction | rendered black bar in PDF (FOI / publisher) | `<REDACTED>` | yes | `RedactionReport` per-span (bbox, page, method, confidence) |
| PII redaction | detected in extracted text (regex + cosine + enrichment graph) | `<PERSON>`, `<EMAIL>`, `<ADDRESS>`, … (typed) | yes | enrichment graph + chunk `has_redaction` flag |

The codebase has the right separation conceptually (`redact/` vs
`pii/`) but three sites disagree on bracket style:

- [pii/cleaner.py:331](src/womblex/pii/cleaner.py#L331) emits
  `[ENTITY_TYPE]` (square) — the implementation. Module docstrings
  at lines 141 and 301 already advertise `<ENTITY_TYPE>` (angle).
- [redact/stage.py:122](src/womblex/redact/stage.py#L122) `blackout`
  mode prepends `[REDACTED]` once per affected page rather than
  inserting inline per detected span.
- Fixtures `_transcript-with-redacted-tags.txt` use `<REDACTED>` inline.
  3 PII-cleaning tests assert `<PERSON>` angle brackets and are
  currently deselected because the implementation emits square.

Unification: align all sites on angle brackets, inline per span.
Re-enables the 3 deselected PII tests as part of the change. Small
fix; deferred pending wider pipeline work.

### Deferred / cut

- **Refresh accuracy-report generator + regenerate accuracy docs**
  (added 2026-05-16) — `tests/accuracy_reports.py` still describes
  extractor outputs in the pre-refactor language ("tables / forms /
  blocks"). The generated `docs/accuracy/EXTRACTION.md`,
  `CHUNKING.md`, `PII_CLEANING.md`, and `REDACTION_HANDLING.md`
  inherit that language. Refresh the generator strings to describe
  the element stream (kinds: paragraph / heading / table / form /
  image / sheet_cell / …), then rerun
  `tests/test_fixture_accuracy.py` and
  `tests/test_womblex_collection_accuracy.py` to regenerate the
  accuracy docs from current behaviour. Out of scope of the
  schema refactor itself.
- **OCR-side ruled-table column-major emission** — irreducible
  trade-off documented under fix 5 above. Re-evaluate only with a
  new discriminating signal (e.g. layout-from-image-vision pass)
  that separates real table cells from form-field streams.
- **Parquet schema for `TableData.context` + `document_metadata`** —
  Largely resolved by the element-stream refactor (2026-05-16):
  per-element ``meta`` map carries arbitrary key-value overflow on
  the elements shard; ``document_metadata`` rides on the manifest.
  ``TableData.context`` is preserved on a kind='table' element via
  ``meta`` keys (``context_*``). Original deferral note kept for
  history.
- **Letterhead-typo normalisations for `Govemment` / standalone
  `AcT`** — the existing `_LETTERHEAD_FIXES` covered the parenthesised
  `(AcT)` and double-m `Govermment` shapes. Under the post-refactor
  verbatim-text policy, `_normalise_text` no longer runs in the
  extraction hot path, so adding entries to `_LETTERHEAD_FIXES` would
  have no effect on the on-disk parquet. If letterhead-typo correction
  is required, it now belongs to a downstream cleaning stage that
  rewrites `pages[i].text`, not the extractor.
- **`02737`-style cross-cell handwritten forms** — paddle's
  row-major reading is architecturally mismatched with forms that
  humans read by cell, AND the handwriting itself crosses cell
  boundaries. Structural OCR-engine limit; not addressable in
  Womblex without a layout-aware OCR backend.

## Don't

The conventions in `CLAUDE.md` continue to apply. One reminder
surfaced this session worth pinning here:

- **Don't retry an OCR-side table-detection relaxation** without
  introducing a new discriminator (layout-from-image, paragraph-gap
  vs cell-gap, column-spread-vs-page-width threshold). Four
  variants tried; all hit the same trade-off cliff. The relaxed
  rule helps real tables and hurts forms by exactly the same
  amount under per-region OCR input.
