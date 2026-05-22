# STATUS — Womblex extraction quality, 2026-05-20

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
| unit tests | All unit tests pass. Fast-suite snapshot (excludes long-running accuracy benchmarks `test_fixture_accuracy.py`, `test_integration.py`, `test_womblex_collection_accuracy.py`): 484 passed, 6 skipped, 24 deselected in ~3 min. Headline per-file counts post K-cluster: `test_extract` 37 (was 19; +18 from `TestClassifyNativeBlock` / `TestFormLabelDenylist` / `TestYoloLabelMapDefault` / `TestPageBreakEmission`), `test_chunker` 60, `test_pii` 37, `test_pii_enrichment` 33, `test_redaction` 58 (was 55; +3 from `exclude_rects` coverage). |
| labels CER vs production parquet (`womblex score`) | hybrid 0.520 · native_with_structured 0.044 · scanned_machinewritten 0.051 · scanned_mixed 0.217 — element-stream reassembly preserves text fidelity vs pre-refactor pipeline output |
| corpus re-extraction (2026-05-17) | 2,626 source files · 3h 5m · 0 failed · 424 sibling-parquet files (106 batches × 4) |
| `kind='table'` contamination | 2,791 elements pre-fix → 172 post-fix (94% reduction); 3 residual conf=0.60 fabrications, ~10% borderline at conf=0.80, real manifests intact |
| `meta` map carries doc/table context | Verified on all 3 spreadsheet-print manifests (`context_213A reference`, `context_Element #`, `context_Text from motion`, `context_motion`) |
| raster-path layout filter | YOLO COCO regions consumed as exclusion zones via `RedactionConfig.use_layout_filter` (default true). Vector path unchanged. See "Detector — raster-path layout filter (2026-05-22)" for design + cost. |
| accuracy-doc generators | Refreshed 2026-05-22 to describe orchestrator + element-stream + vector-first detector + YOLO exclusion zones. `EXTRACTION.md` / `REDACTION_HANDLING.md` / `PII_CLEANING.md` regenerated cleanly. |
| non-`table` element kind audit | Surfaced 2026-05-22 (90,843 elements audited). Eight concrete fixes K1-K8 tracked in "Open follow-ups". Headline: `kind='signature'` 100% mis-classified (442 closing-phrase matches, no signatory blocks); `kind='figure'` 65% mis-classified on scanned pages (1,044 of 1,600); `list_item` / `caption` / `page_break` declared but never produced. (K2's "all-zero bbox" claim turned out to be a probe-formatter artefact — bboxes are populated for all native-text kinds; real narrower issue is OCR-derived forms with no bbox, tracked as K2′.) Full audit in `stories/STATUS.md`. |
| element-kind audit fix cluster | K1 / K3 / K4 / K5 / K7(a) / K8 landed 2026-05-22. K2 retracted 2026-05-23 as a false finding (probe formatter artefact). K2′ (OCR-form bbox loss) and K7(b) (document-layout YOLO swap) remain open. |
| post-K-cluster corpus re-extraction | Completed 2026-05-23 (2h 35m · 2,626 succeeded · 0 failed). All K-cluster effects landed cleanly: `signature` 442→0, `list_item` 0→4,015, `page_break` 0→6,561, `header` 0→335, `form` 5,183→4,391 (−792 spurious), `figure` 1,600→1,587 (small — K7(a) was bounded by COCO's explicit screen-class mappings, not just the unknown default). Strategy distribution unchanged, all known limitations preserved (no regressions). Text fidelity preserved: per-strategy mean CER identical to pre-re-extraction (hybrid 0.520, native_with_structured 0.044, scanned_machinewritten 0.051, scanned_mixed 0.217). |

## Open follow-ups

Tracked here so they don't drift back into "Deferred". Update or strike-through when resolved.

### Element-kind audit fix cluster (K1-K8)

Surfaced by the corpus-wide audit completed 2026-05-22 — see `stories/STATUS.md` "Non-`table` element kind audit" for the full data set. **K1, K3, K4, K5, K7(a), K8 landed 2026-05-22 as a single change set; K2 and K7(b) remain as separate larger tracks below.**

| ID | Status | Fix | Code site |
|---|---|---|---|
| **K1** ✓ | landed | `_SIGNATURE_RE` removed from `_classify_native_block`. Closing phrases ("Yours sincerely") no longer emit `kind='signature'` — they fall to `paragraph` until a proper signatory-block detector lands. | [extract.py:325-346](src/womblex/ingest/extract.py#L325-L346) |
| ~~K2~~ | retracted | **Originally "every native-text element bbox is zero" — that was a probe-script formatter artefact (`:.0f` rounding normalised 0-1 floats to "0"). Corpus measurement confirms bbox population at 100% for paragraph / heading / footer / signature / figure / image / table / native_with_structured forms.** The narrower real issue is K2′ below. | (verified 2026-05-23) |
| **K2′** | open | **OCR-form bbox loss.** `kind='form'` elements from line-based extraction (OCR strategies: `scanned_machinewritten`, `scanned_mixed`, `scanned_handwritten`, and the OCR-page subset of `hybrid`) carry `bbox=(0,0,0,0)` by design — `_extract_form_pairs_from_lines` operates on assembled text without per-word bboxes. Affects 4,184 of 5,183 form elements (80.7%). Fix: wire PaddleOCR per-word bboxes through to the line-based form-pair extractor. Same infrastructure unlocks #C inline-per-span on raster pages. | [forms.py:146](src/womblex/ingest/forms.py#L146) |
| **K3** ✓ | landed | Label denylist added to `_looks_like_form_label`: `Penalty`, `OFFICIAL`, `Note`, `Caution` — captures the regulation-citation / document-banner patterns that drove ~250-500 spurious forms in hybrid + structured. | [forms.py:38-46](src/womblex/ingest/forms.py#L38-L46) |
| **K4** ✓ | landed | `header` added to `ElementKind` literal, `TEXT_KINDS` frozenset, and `_BLOCK_TYPE_TO_KIND` mapping. `_classify_native_block` was already returning `"header"`; now it round-trips into `kind='header'` instead of silently demoting to `paragraph`. | [elements.py:26](src/womblex/ingest/elements.py#L26); [orchestrator.py:212](src/womblex/ingest/orchestrator.py#L212) |
| **K5** ✓ | landed | `_LIST_ITEM_RE` added to `_classify_native_block`: matches `(a)` / `(b)` / `(i)` / `(1)` / bullets `•·-*`. Bare `1. `-prefix excluded (ambiguous with numbered paragraphs in this corpus). | [extract.py:320](src/womblex/ingest/extract.py#L320) |
| K6 | deferred | Caption detection (image-adjacent). Needs layout-aware adjacency analysis; subsumed by K7(b). | — |
| **K7(a)** ✓ | landed | `_YOLO_COCO_LABEL_MAP` default changed from `figure` to `paragraph`. Unknown COCO classes (the dominant case on scanned pages, since COCO doesn't have document classes) now bucket to text. Explicit screen/keyboard/etc. mappings preserved. | [paddle_ocr.py:232](src/womblex/ingest/paddle_ocr.py#L232) |
| **K7(b)** | open | **Document-layout YOLO swap** (DocLayNet / PubLayNet checkpoint). Full fix — same lever closes both `kind='figure'` mis-classification AND the redaction-precision gap on scanned_mixed (#6 cohort measurement showed COCO yields only 5%; 02737 unchanged). One model swap, two precision wins. | `ingest/paddle_ocr.py` |
| **K8** ✓ | landed | Orchestrator emits `kind='page_break'` between consecutive pages in `extract_with_plan`. N-1 breaks for N pages. | [orchestrator.py:401-406](src/womblex/ingest/orchestrator.py#L401-L406) |

**Validation.** Six landed fixes verified via:
- Unit tests: `tests/test_extract.py` adds `TestClassifyNativeBlock` (10 cases for K1/K4/K5), `TestFormLabelDenylist` (4 cases for K3), `TestYoloLabelMapDefault` (2 cases for K7(a)), `TestPageBreakEmission` (2 cases for K8). All 37 test_extract pass; broader 484-test fast-suite pass with no regressions.
- Sample re-extraction on 12 docs (3 per stratum) shows expected per-kind deltas vs the pre-K-cluster production parquet:
  | kind | sample delta | direction |
  |---|---:|---|
  | `page_break` | **+25** | K8 emitting per page transition |
  | `list_item` | **+20** | K5 picking up regulation sub-paragraph markers |
  | `paragraph` | **−17** | net reduction from list_item reclassification + signature drops |
  | `signature` | **−4** | K1 — all 4 sampled signatures were "Yours sincerely" closings, now dropped |
  | `form` | **−3** | K3 denylist filtering Penalty/OFFICIAL on sampled docs |
  | `header` | **+1** | K4 round-tripping top-of-page short text into the canonical kind |

**Corpus-scale measured** (post-K-cluster re-extraction 2026-05-23, 2h 35m, 2,626 succeeded, 0 failed):

| kind | pre (2026-05-17) | post (2026-05-23) | Δ | note |
|---|---:|---:|---:|---|
| paragraph | 40,980 | 37,107 | −3,873 | net of list_item / header reclassification, partial offset from K1 |
| sheet_cell | 36,780 | 36,780 | 0 | unchanged |
| form | 5,183 | 4,391 | **−792** | K3 denylist removed more than projected (~250-500 estimated) |
| page_break | 0 | **6,561** | +6,561 | K8 — averaging ~2.5/doc (corpus skews to 1-3 page notification forms) |
| footer | 2,695 | 2,695 | 0 | unchanged |
| heading | 2,656 | 2,634 | −22 | small reclassification to list_item / header |
| list_item | 0 | **4,015** | +4,015 | K5 |
| figure | 1,600 | 1,587 | **−13** | K7(a) impact was small — figure count is driven by explicit COCO screen-class mappings (`tv` / `laptop` / `monitor`), not the unknown-class default. **K7(b) remains the right lever for closing this.** |
| image | 331 | 331 | 0 | unchanged |
| signature | **442** | **0** | **−442** | K1 — every closing-phrase mis-classification gone |
| table | 172 | 172 | 0 | §1 limitations preserved (3 residual conf=0.60 fabrications still present, as documented) |
| header | 0 | **335** | +335 | K4 |
| sheet_meta | 4 | 4 | 0 | unchanged |
| **total** | **90,843** | **96,612** | **+5,769** | net of new kinds + reclassifications |

**Strategy distribution identical** (detection logic unchanged): 1,777 scanned_machinewritten · 628 hybrid · 165 structured · 40 native_with_structured · 11 scanned_mixed · 4 spreadsheet · 1 scanned_handwritten.

**Text fidelity preserved exactly** — per-strategy mean CER on the 18-page labels packet matches 2026-05-21 numbers to 3 decimal places. K-cluster reclassified element kinds without touching text content.

**Spot-check on known limitations**: §1 residual fabrications (01093, 01094, 01349) still carry their 1 junk table each — behaviour preserved, no regressions. §11 02737 unchanged except +1 page_break. FOI master manifest (`Schedule-of-documents-Part-2b`) intact.

**K7(a) impact was smaller than originally projected** (−13 vs expected near-zero on scanned_machinewritten figure count). Root cause: COCO YOLO's explicit screen-class mappings (`tv` / `laptop` / `monitor` / `keyboard` / `mouse` / `scissors` / `clock` → `figure`) catch most scanned-page hits, not the unknown-class default. Closing the remaining 1,587 figure mis-classifications requires K7(b) (document-layout YOLO swap) — confirms the dependency framing.

Previous on-disk parquet preserved at `stories/.../womblex-extract/output-pre-kcluster-2026-05-17/` for historical comparison.

### Larger tracks (separate work)

- **#A — Document-layout YOLO swap (K7(b) above).** Now actionable post-#6 measurement. Joint payoff: redaction precision on scanned_mixed (closes §11) + `kind='figure'` reclassification on scanned pages (closes K7 fully). Model swap + `_YOLO_COCO_LABEL_MAP` rewrite + re-measure on both the 11-doc redaction cohort and a figure-quality sample. External dependency: a public DocLayNet / PubLayNet YOLOv8 checkpoint.
- **#B — Native-text footer whitespace artefacts** (stories §8). `3|P age` / `3| Page` from sub-glyph kerning. Belongs to a downstream cleaning op that rewrites `pages[i].text` — verbatim-text policy means extraction won't normalise.
- **#C — Inline-per-span source redactions.** Bracket-only unification (2026-05-21) is page-prefix only. Inline-per-span needs bbox-to-text-position mapping. Two paths with two different requirements:
  - **Raster pages**: depends on K2′ (wire PaddleOCR per-word bboxes through extraction).
  - **Native pages**: needs a separate text-to-bbox character-position mapping over `page.get_text("dict")` spans.
- **#D — Redaction-induced paragraph breaks on native pages** (stories §9). PyMuPDF `blocks` join with `\n\n` either side of a redaction. Fix paths: (a) refine `_render_blocks_with_breaks` adjacent-baseline detection or (b) downstream orphan-line re-join.
- **#E — CHUNKING.md generator.** Hand-maintained today (numbers from 2026-03-22; framings refreshed 2026-05-22). The other three accuracy docs have generators; this one doesn't. Adding a `generate_chunking_report` closes a quiet drift vector.
- **#F — Full corpus re-extraction.** Refresh the on-disk production parquet at `stories/.../womblex-extract/output/` with current K-cluster code behaviour. Not blocking anything (nothing downstream consumes the parquet yet), and the 12-doc sample already proved the K-cluster works. Defer until downstream pipeline work starts or until multiple code changes have stacked.

### Closed (this session)

- ~~K2 investigation~~ (2026-05-23; **retracted as a false finding** — bboxes are populated correctly at 100% for all native-text kinds; the audit's all-zero observation came from a `:.0f` formatter rounding 0-1 normalised floats. Narrower real issue surfaced as K2′ above.)
- ~~K-cluster QA pass~~ (2026-05-23; 33 modified files audited; 6 minor doc/STATUS fixes landed; 484 unit tests pass, 0 regressions)
- ~~Element-kind audit fix cluster~~ (2026-05-22; K1 / K3 / K4 / K5 / K7(a) / K8 landed — see fix cluster table above)
- ~~Measure #6 precision gain~~ (2026-05-22; −8 / −5.0% on the 11-doc cohort, 02737 unchanged — see "Detector — raster-path layout filter" cohort measurement section above)
- ~~Audit non-`table` element kinds at corpus scale~~ (2026-05-22; surfaced K1-K8 above)
- ~~Decide `use_layout_filter` default~~ (2026-05-22; keep `True` — no doc regressed, modest per-doc cost; full-benchmark 7× slowdown acceptable)
- ~~Marker convention unification~~ (2026-05-21; see "Redaction & PII marker conventions" section below)
- ~~PDF annotation read probe~~ (2026-05-21; not viable for this corpus — see `stories/STATUS.md` Outstanding §4(b))
- ~~Doc-drift audit~~ (2026-05-21; 23 stale claims fixed across CLAUDE.md / README.md / architecture / dataflow / steering / accuracy generators)

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
`tests/test_spreadsheet_print.py` (14); the full-suite snapshot lives
in the Snapshot table above (numbers shift as new tests land).
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

### Library — `redact/batch.py` (2026-05-19, batch redaction operations)

Promoted from a corpus-local validation script. Two entry points:

- `annotate_redactions_for_shards(shard_dir, pdf_dir, config, output_dir,
  checkpoint_path)` — batch-detect redactions across extracted parquet
  shards; write a sparse `*.redactions.parquet` sidecar per batch with
  `(source_hash, elem_order, has_redaction)` rows for elements on
  affected pages. Resumable via the optional checkpoint JSON.
- `validate_redactions_against_labels(labels_dir, pdf_dir, config)` —
  run detection over PDFs referenced in a labels packet; return per-doc
  `ValidationSummary` objects. Used for detector tuning / sanity checks.

CLI wrappers landed 2026-05-20 — see CLI restructure below.

### CLI — `cli.py` → `cli/` subpackage (2026-05-20)

Single 728-line `cli.py` (at the 750-line cap) split into a per-topic
subpackage. Each topic module exposes a ``COMMANDS: list[Command]`` and
``cli/__init__.py`` aggregates them, wires up argparse subparsers, and
dispatches by name.

```
src/womblex/cli/
├── __init__.py      main() + ALL_COMMANDS aggregation + dispatch
├── _shared.py       Command NamedTuple, setup_logging, discover_files, format_eta
├── pipeline.py      run, extract, chunk
├── redact.py        redact, annotate-redactions, validate-redactions
├── ingest.py        ingest-gnaf, ingest-geo
├── score.py         score
└── profile.py       profile
```

Two new CLI subcommands landed in `cli/redact.py`:

- `womblex annotate-redactions <shards> <pdfs> [--output DIR] [--checkpoint PATH]`
  — invokes `redact.batch.annotate_redactions_for_shards`. Resumable.
- `womblex validate-redactions --labels DIR --pdfs DIR [--report PATH]`
  — invokes `redact.batch.validate_redactions_against_labels`. JSON or
  markdown output.

Entry point `womblex = "womblex.cli:main"` unchanged (cli/__init__.py
exports `main`). All existing subcommand surfaces preserved verbatim
(args, help text, behaviour). Heavy double-line spacing from the old
file normalised during the move.

Largest file post-split: `cli/pipeline.py` at 283 lines (38% of cap).

### Detector — vector-first detection (2026-05-19)

`redact/stage.py:detect_redactions` was extended to try
`page.get_drawings()` first for filled near-black rectangles; falls back
to the existing CV2 contour detector on rasterised pages when the
vector path finds nothing. Both paths return `RedactionInfo` with
bboxes in pixel coordinates at the configured DPI.

Filters (each surfaced by a measured false-positive class during Phase
2 validation):

- Near-black fill (max channel ≤ 0.1 RGB; CMYK K ≥ 0.9 + others ≤ 0.1) —
  baseline filter.
- `min_width ≥ 3pt` — excludes narrow vertical separator lines that
  appear in manifest-style tables (FOI master regression on 37 pages).
- `min_height ≥ 8pt` — excludes glyph-rendering small filled rects on
  PDFs that draw text as filled-path glyphs rather than vector text
  (01125-class regression: 14,184 false positives → 144 actual).

Closes Outstanding §4(a) in `stories/STATUS.md`. Validated against the
labels packet: §1 residual recall improved 6→14, 7→13, 3→68 without
regressing the FOI master manifest (0 regions preserved) or any
plain-scanned doc. The 02737-class scanned_mixed false-positive cohort
falls back to the raster path and is unchanged (documented limitation,
see `stories/STATUS.md` §11).

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

The 3 residual conf=0.60 fabrications survive because PyMuPDF's
text-strategy `find_tables` clusters the text fragments left over
after large redaction blocks fragment the surrounding prose into
whitespace-aligned columns. The bars themselves do not register as
cells (measured cell-vs-fill overlap ≤ 1% across all three pages);
they cause the misread indirectly via the gap pattern they create.
The §1 gate's `n_blocks ≥ n_rows` premise is satisfied because the
count of surviving paragraph dict-blocks happens to match the count
of synthesised rows. Content impact is contained: text-bearing
elements on these pages capture the prose verbatim (the
`kind='table'` element is additive noise, not corruption). A
region-level black-fill signal could close the strong-signal page
(01349, 28% bbox coverage) but is on a threshold tightrope for
01093/01094 (~3.5% coverage) where false-positives on real native
tables become a real risk. Accepted as documented limitation; see
`stories/STATUS.md` Outstanding §2. Material impact ~0.11% of source
docs.

### Redaction & PII marker conventions (✅ unified 2026-05-21, bracket-only)

Two distinct concerns, two distinct markers:

| concern | source | marker | inline per span | metadata home |
|---|---|---|---|---|
| Source redaction | rendered black bar in PDF (FOI / publisher) | `<REDACTED>` | not yet — see below | `RedactionReport` per-span (bbox, page, method, confidence) |
| PII redaction | detected in extracted text (regex + cosine + enrichment graph) | `<PERSON>`, `<EMAIL>`, `<ADDRESS>`, … (typed) | yes | enrichment graph + chunk `has_redaction` flag |

Bracket-style unification on angle brackets landed across:

- [pii/cleaner.py:331,406](src/womblex/pii/cleaner.py#L331) — emits `<ENTITY_TYPE>` (PII spans). Module docstring rationale rewritten; the prior BPE/SentencePiece tokenisation argument for square brackets didn't survive scrutiny (neither bracket style is single-piece in standard pretrained tokenisers without explicit special-token registration).
- [redact/stage.py:187](src/womblex/redact/stage.py#L187) — `blackout` mode emits `<REDACTED>` (still page-prefix, not inline per span — see below).
- [process/chunker.py:227](src/womblex/process/chunker.py#L227) — `_repair_redaction_splits` marker constant flipped to `<REDACTED>`; cross-file coupling comment preserved.
- [operations.py:298](src/womblex/operations.py#L298), [config.py:86](src/womblex/config.py#L86) — docstrings.
- [docs/architecture.md](docs/architecture.md), [docs/dataflow.md](docs/dataflow.md) — manual references aligned. [docs/accuracy/REDACTION_HANDLING.md](docs/accuracy/REDACTION_HANDLING.md) is generated by `test_fixture_accuracy.py` and will regenerate on next accuracy-benchmark run.
- Tests across `test_pii.py`, `test_pii_enrichment.py`, `test_chunker.py`, `test_redaction.py`, `test_womblex_collection_accuracy.py`, `accuracy_reports.py` — all assertions migrated. The 3 previously-failing ADDRESS angle-bracket tests now pass.

**Inline-per-span for source redactions is deferred** — flipping the bracket style alone is bracket-only behaviour-preserving. Going inline-per-span requires a bbox-to-text-position mapping that doesn't exist for raster-path redactions (pixel-only bboxes with no character index). Native-path (PDF vector) detection has PDF coords and could be mapped; OCR/raster path needs word-bbox routing wired through to the redact stage. Tracked as a follow-up.

No on-disk data migration was needed — production runs to date are flag-mode, no `blackout` text mutation has been applied, and PII cleaning hasn't been applied to the corpus.

### Detector — raster-path layout filter (2026-05-22)

`RedactionDetector.detect()` accepts a new `exclude_rects` parameter; candidates whose bbox centre falls inside any rect are dropped. `redact/stage.py:detect_redactions()` runs YOLO layout analysis on raster-fallback pages and passes regions of `_LAYOUT_EXCLUSION_CLASSES` (`tv`, `laptop`, `monitor`, `cell phone`, `keyboard`, `mouse`, `book`, `dining table` — the COCO classes that heuristically land on form-field backgrounds and chart regions) as exclusion zones to the contour detector. Best-effort: try/except, falls back to raw raster pass on any error (missing ultralytics, model weights absent).

Gated by `RedactionConfig.use_layout_filter: bool = True` (default on). Threaded through `operations.py:run_redaction`, `redact/batch.py:annotate_redactions_for_shards`, `redact/batch.py:validate_redactions_against_labels`. CLI / `configs/example.yaml` exposes the flag.

**Cost:** Vector-path detection (native PDFs) is unchanged — YOLO never runs there. Raster-fallback path triggers YOLO inference per page, which materially slowed the accuracy benchmark (3 min → 22 min for `test_womblex_collection_accuracy.py`). For production batch runs on scanned_mixed cohorts the trade is correct (precision over speed).

**Cohort measurement (2026-05-22).** Ran `detect_redactions` on the 11 scanned_mixed docs from the corpus with the filter off vs on (config: `RedactionConfig(max_area_ratio=0.05)`, the corpus tune):

| metric | off | on | Δ |
|---|---:|---:|---:|
| total regions across 11 docs | 159 | 151 | **−8 (−5.0%)** |
| docs with any region | 10 | 10 | 0 |
| 02737-213A (the signature case) | 10 | 10 | **0** |
| runtime | 9.5s | 12.6s | +3.1s (1.3×) |

**Interpretation.** The COCO YOLO model produces useful exclusion zones on 6 of 11 docs but at very small magnitudes (−1 to −3 per doc). It does **not** touch the worst case — 02737's 10 regions across 2 pages, the cohort's most egregious false positives, are entirely missed by the COCO classes the filter listens for. The hypothesis that `tv` / `laptop` / `monitor` etc. would heuristically land on dark form-field backgrounds was too weak: YOLO either doesn't detect those classes on rendered CRM-form pages, or detects them but not on the regions where contour detection misfires.

The filter is net-positive (more precise without regressing any doc) but the magnitude is too small to close §11's `scanned_mixed` false-positive gap on its own. A document-layout-trained checkpoint (DocLayNet / PubLayNet, with `Figure` / `Table` / `Form` classes) would be a better fit — see "Open follow-ups" item 8 in this STATUS.

**Default decision.** Keeping `use_layout_filter=True` is defensible (no doc regressed; some precision gain) but the value is modest and the test-suite runtime cost is real. Best path forward is the YOLO swap — see "Open follow-ups" K7(b) / track #A.

### Non-`table` element kind audit (2026-05-22)

Corpus-wide audit of element kinds beyond the resolved `kind='table'` work. Counts (90,843 elements across 2,626 docs) and per-strategy distribution live in `stories/STATUS.md` "Non-`table` element kind audit". Top findings, in summary form (full data + per-fix code-site references in the stories STATUS):

- **`kind='signature'` is 100% semantically wrong.** `_SIGNATURE_RE` matches `Yours sincerely` / `faithfully` / `truly` — i.e. the closing phrase, not the signatory block. All 442 signature elements in the corpus are closing-phrase matches. Actual signatory blocks (name + title + redaction bar) are filed as `paragraph`.
- **`kind='figure'` is 65% mis-classified on scanned pages** (1,044 of 1,600). Root cause: `_YOLO_COCO_LABEL_MAP` defaults unknown COCO classes to `figure`, and on rendered scanned-page images YOLO finds plenty of unknown classes. These "figures" contain OCR text — they're text-bearing elements filed under a non-text kind.
- **Three element kinds declared but never produced.** `list_item`, `caption`, `page_break` — schema enum + mapping table both present, no producer. Lists are extremely common in regulatory documents; the capability is silently absent.
- **`header` block_type falls through to `paragraph`.** `_classify_native_block` returns `"header"` for `y_norm < 0.08`, but `"header"` isn't in `ElementKind` — it gets demoted into `meta['block_type']`, downstream consumers reading `kind` miss it.
- **`kind='form'` over-fires on regulatory letters.** `Penalty: $10 000, in the case of an individual` (regulation citation) and `OFFICIAL: Sensitive - Legislative Secrecy` (document banner) get matched as form pairs. ~250-500 spurious in hybrid + structured cohorts.
- **Bboxes for native-text elements serialise as `(0, 0, 0×0)`** — silent data-loss bug. Visible in every native sample but separate from kind classification. Tracked as K2.

Eight concrete fixes K1-K8 written up in "Open follow-ups" above. K1 / K3 / K4 / K5 / K7(a) / K8 land naturally as one low-effort change set. K2 and K7(b) are separate larger tracks.

### Deferred / cut

- ~~**Refresh accuracy-report generator + regenerate accuracy docs**~~ (resolved 2026-05-22). `tests/accuracy_reports.py` generator strings refreshed: `generate_redaction_report` now describes the vector-first detector + YOLO exclusion zones + `run_redaction` operation; `generate_extraction_report` now describes the per-page orchestrator + element-stream kinds + four sibling parquet shards; the strategy-matrix column reflects per-page dispatch. The four generated accuracy docs (`EXTRACTION.md`, `REDACTION_HANDLING.md`, `PII_CLEANING.md`) were regenerated cleanly. `docs/accuracy/CHUNKING.md` framings updated by hand for the post-refactor spreadsheet shape; a generator for this doc is still to-be-written (numbers still date from 2026-03-22).
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
