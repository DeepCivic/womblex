# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`money` annotation op** (`womblex money --shards`). Recovers monetary
  amounts from the extraction parquet and writes two siblings per batch:
  `*.money_spans.parquet` (one row per amount) and `*.money_columns.parquet`
  (the column-classification audit). Offline, API-free, no ordering dependency
  on enrich, and it never rewrites element or chunk text. Implements the design
  in [docs/money.md](docs/money.md).

  Two evidence paths, because most of this corpus's amounts carry no currency
  marker at all:

  - **Self-evidencing** (`process/money.py`) — a symbol, ISO 4217 code or
    currency word sits with the number. The pattern set is applied in priority
    order with overlap resolution, magnitude expansion (`$4.2bn`, 97% of marked
    narrative amounts carry a scale suffix), range endpoints linked rather than
    collapsed, qualifiers (`up to`, `~`) stored separately from the value, and
    accounting negatives gated — an unanchored bracketed-number scan is the
    corpus's worst false-positive source (`s167(1)`, `(02) 6203 7300`).
    Candidates embedded in Australian false-positive classes (dates, times,
    phone numbers, ABNs/ACNs, legislative references, measurements,
    percentages) are rejected.
  - **Column-evidenced** (`process/money_columns.py`) — a bare number whose
    money-ness comes from its column: number format carrying a currency symbol
    (definitive), else money-vocabulary header plus predominantly numeric
    cells. Numeric cells never promote a column alone; whole-word vetoes
    suppress one (`age` vetoes `Age`, `Average Cost` survives); null markers
    are absent values excluded from the numeric fraction; the header supplies
    the column's scale (`$m`, `$'000`) and currency. A column with no
    recoverable header is left un-extracted.

  Values are exact `decimal128(38, 4)`, not floats — aggregating a 48,997-row
  register accumulates float error. Three loci are anchored in two coordinate
  spaces and never mixed: `narrative` spans are character offsets into the
  reassembled narrative in the `processing.text_source` layer (stamped on every
  row, so they join enrichment mentions and map to chunks), `table_cell` is
  `(parent_elem_order, row, col)`, `sheet_cell` is `(sheet, row, col)`.

  Header continuation rows are folded into the header: PDF financial tables
  wrap `Approved` / `Budget $m` across two rows and declare only the first,
  which previously left the column looking like a nameless run of bare
  numbers. One leading non-numeric row is absorbed when the rest of the column
  is numeric, so a genuine text data row is never eaten.

  No new dependencies. Pattern 10 (bare numbers near financial vocabulary in
  narrative) and continental number formats ship off by default.

  A number in a header no longer declares a thousands scale: the `'000`
  pattern matched the `000` inside any number, so a `Grants over $10,000`
  header multiplied every cell beneath it by 1,000.

  Tier-3 ISO codes are gated on surrounding context rather than merely
  scored lower, because several are ordinary English words in capitals:
  ungated, `TOP 10 projects` reads as ten Tongan paʻanga and
  `ALL OTHER COMPENSATION ($)` resolves to Albanian lek. A tier-3 code needs a
  currency symbol or financial trigger word nearby; in a header it must be
  parenthesised (`Value (PGK)`). Tier 1/2 codes stand alone.

  Count columns are no longer read as money. A financial table marks a count
  column `(#)` exactly as it marks a money column `($)` — the same page can
  carry `Threshold ($)` and `Threshold (#)` — but the header tokeniser dropped
  `#` entirely and promoted the count column on the vocabulary term alone.

  A veto term no longer suppresses a column whose header declares its own
  currency: `Grant Date Fair Value of Stock and Option Awards ($)` is a money
  column that happens to contain the word "date", and vetoing it lost all five
  amounts beneath it. Count columns on the same page carry `(#)` rather than
  `($)` and stay vetoed. The overridden term is still recorded in the column
  audit.

  First real-document run (four benchmark fixtures through the real pipeline,
  every span hand-checked): all 42 marked narrative amounts recovered from the
  ANAO Major Projects Report, and its `Approved Budget $m` column reconciles
  three ways — 25 project amounts summing to the table's own total row and to
  the narrative's independently written "$78.7 billion". Details and the
  measured limits in [docs/money.md](docs/money.md).

- **Table-cell reconstruction on OCR'd pages (#17), step A0 — scope and
  plumbing.** The layout pass (`_layout_blocks_and_tables`) now receives the
  per-detection OCR regions and the OCR render's pixel dimensions from the
  orchestrator, the raw material for reconstructing cells inside a detected
  table rect. Both arguments are optional, so callers that don't supply them
  keep their exact previous behaviour, and **no tables are produced yet** —
  `tables` is still returned empty on every path.

  The scope this fixes is which engines can ever reach reconstruction:
  region-based ones only. LLM/VLM engines (`mistral-ocr`, `ollama`) resolve
  reading order natively, return markdown with no regions, and are dispatched
  to `_markdown_page_block` — there are no quads to bin, so a markdown
  pipe-table parser is their separate, deferred feeder. The accuracy suite's
  extraction calls now pin `engine="paddleocr"` accordingly; under a config
  default of an LLM engine its numbers would describe a different pipeline.

  Two pieces of the reconstructor's foundation come with the seam:
  `_regions_in_rect()`, the OCR-quad → table-rect intersection by centroid
  containment, and a coordinate-space guard — the OCR render and the layout
  render are the same page at the same dpi, so unless the OCR render's
  dimensions are supplied and match, the coordinates are not known to be
  comparable and the regions are dropped with a warning rather than binned.
  Losing inputs is the correct failure; a mis-binned grid would be confidently
  wrong downstream. Deskewed pages are a separate hazard this check does not
  catch (dims survive `warpAffine`), handled by a later page-level refusal.
  Per-table debug logging records how many OCR regions fall inside each
  detected table, so the size of the gap is traceable per page before the
  reconstructor lands.
  Plan and sequencing in
  [docs/table-cell-reconstruction-plan.md](docs/table-cell-reconstruction-plan.md).

- **Table-cell reconstruction on OCR'd pages (#17), step A1 — one shared
  grid algorithm, two feeders.** The row/column inference that
  `spreadsheet_print` already used (y-band binning, data-anchored column
  clustering, x-left cell assignment) is lifted into `ingest/table_grid.py`
  and consumed unchanged by `spreadsheet_print`; the point-space tolerances
  became parameters so pixel-space callers scale them by `dpi/72` instead of
  running ~2.8× too tight at 200 dpi. The OCR-side row-clustering preamble
  that `_spatial_sort_regions` and `_table_aware_text` each carried nearly
  line-for-line is now one shared helper (`rows_from_spans`), closing the
  repo's third table-ish duplication.

  `ingest/ocr_tables.py` is the new second feeder:
  `reconstruct_table(regions, table_rect, dpi, conf)` reduces the OCR quads
  inside a layout-detected table rect to spans and reconstructs the grid as
  a `TableData` — or returns `None`, never a partial, below its precision
  gates (minimum columns/rows, a left-edge column-fit ratio, header text
  actually recovering, and a row-fill density floor added in B2; each refusal
  debug-logged). Refusal on a hard shape is a correct round-1 outcome. The
  header band and body bands bin separately, so a first body row with a blank
  leading cell — an indented or grouped row — is no longer folded into the
  header and lost by the wrapped-cell continuation rule. Element lineage is
  deliberate: confidence comes
  from the constituent region confidences capped by the detector's, and
  `context["producer"] = "table_grid"` distinguishes reconstructed tables
  from PyMuPDF-fallback ones in the parquet. Nothing is wired into the
  layout pass yet — `tables` is still returned empty on every extraction
  path until A3.

- **Table reconstruction benchmark (#17, steps B0 + B3 + B1.2).** The
  table metric is fixed before anything is measured against it (B0): the
  DocLayNet GT aggregation no longer charges stray sub-3-span
  Table-labelled runs (footnote lines mislabelled Table) as false
  negatives — table recall on the vendored fixtures was 25% largely by
  annotation artefact, 50% after the fix — and steering's stale
  "0 predictions" layout claim is corrected.

  `tests/test_table_benchmark.py` (B3 + B1.2, `benchmark`-marked) measures
  reconstruction *conditioned on a correct table rect*, no layout detector
  in the loop: deterministic table pages rendered from the two vendored
  spreadsheet sources (3 pages × 30 rows of the Approved-providers CSV;
  one page per MSO fuel sheet), rasterised at 200 dpi, OCR'd with
  paddleocr, reconstructed with the rect known by construction, scored
  positionally against the drawn strings under a declared normalisation
  (NFKC + dash folding + whitespace collapse). Round-1 baseline: all six
  rendered-clean fixtures reconstruct with exact structure and full header
  recovery; cell accuracy 84–99%, every mismatch glyph-level OCR
  recognition rather than grid binning. The hard scan fixture
  `dense_text_548` tracks without a gate and **refuses** (the row-fill
  density gate B2 added rejects its sparse ~0.45-fill grid; pre-B2 it
  yielded a 12×12 partial against the 39×11 GT). The off-spec
  `sparse_text_344` CSV/meta GT is removed (declared non-GT).

- **Table-cell reconstruction on OCR'd pages (#17), steps A3 + A2 — the
  OCR-PDF path now produces cells.** A layout-detected table region on an
  OCR'd PDF page is handed to `reconstruct_table`, and where the grid clears
  its precision gates the page gains a `kind="table"` element with cells
  instead of the table's text being swallowed into page narrative. This is
  the first path on which `_layout_blocks_and_tables` returns a non-empty
  `tables` list. Nothing downstream changed to accommodate it: the element
  goes through the same `_table_to_element` → writer → `table_cells.parquet`
  route as native and spreadsheet-print tables, so the chunker's markdown
  projection and the money stage's `table_cell` locus pick it up as-is.

  The double-count this had to avoid is the **narrative fallback**, not the
  `[TABLE]` placeholder. Layout-derived blocks carry no text, so the "no
  block has text" fallback fires on essentially every layout-successful page
  and emits one block holding the whole page's OCR text — table content
  included. Where a table reconstructs, that narrative is now rebuilt from
  the OCR regions *outside* its rect, so the chunker sees the table once
  (as markdown) rather than twice. A page that is only a table emits no
  narrative block at all. The same absorbed regions are withheld from
  form-pair extraction, so a colon-bearing cell can't land in both a form
  element and the table. On refusal — the precision gates, or A2 below —
  the page keeps its previous behaviour exactly, byte for byte.

  `PageResult.text` stays the verbatim full-page OCR text. The subtraction
  is an element-stream concern; page text feeds text-coverage and the CER
  metrics, which compare against a transcript of the whole page.

  **A2 — deskewed pages refuse rather than mis-bin.** `preprocess_for_ocr`
  deskews via `warpAffine` before OCR when |angle| > 0.5°, so the region
  coordinates are in rotated space while the layout pass renders the raw
  page. `warpAffine` preserves the frame, so A0's dimension guard cannot
  catch this. The orchestrator now reads `"deskew" ∈ steps` off `_ocr_page`
  and the layout pass drops its cell source on such pages — a page-level
  refusal consistent with precision-first. Mapping the layout rect into
  deskewed space is deferred to the round that targets real scans. Flat
  contemporary documents, round 1's target, almost never trip deskew.

  The image path is untouched by A3 — that was A4's scope, which turned
  out to be a no-op (below).

- **Table-cell reconstruction on OCR'd pages (#17), step A4 — images were
  never a separate path; the dead extractor is gone.** A4 was scoped to
  route `ImageExtractor` through the layout pass, on the premise that
  standalone images bypassed table reconstruction. The premise was wrong:
  `extract_text` gates the legacy path-based dispatch on
  `(SPREADSHEET, DOCX, TEXT)`, and `IMAGE` is not in it — it falls through
  to `fitz.open()` + `extract_pdf_with_plan`, because PyMuPDF opens an
  image as a one-page document. Images have always reached
  `_apply_ocr_page`, so **A3 already gave them table reconstruction**;
  verified by driving a real `.png` through `extract_text` and observing a
  cellified `table` element carrying `context_producer=table_grid` beside a
  narrative paragraph with the table text subtracted.

  `ImageExtractor` was therefore unreachable from every production and
  measurement path (the accuracy suites call `extract_text` or
  `get_paddle_reader` directly). It is deleted, along with `get_extractor`'s
  unreachable `DocumentType.IMAGE` case and the `strategies.py` re-export.
  `get_extractor`'s `dpi` / `lang` / `engine` / `engine_options` parameters
  go with it — they existed only to construct `ImageExtractor`, and a
  function that silently ignores an `engine=` argument is a trap; the
  signature is now `get_extractor(profile)`, returning
  `PathExtractionStrategy`.

  This is a **breaking change for direct importers** of
  `womblex.ingest.strategies.ImageExtractor`,
  `womblex.ingest.strategies_scanned.ImageExtractor`, or `get_extractor`'s
  removed keyword arguments. Nothing inside womblex used any of them. Route
  images through `extract_text` instead — it is what the pipeline does.

  `TestImageDocumentsRouteThroughTheOrchestrator` pins the routing from the
  `extract_text` entry point, so a future change reintroducing an image
  bypass fails there rather than silently losing table reconstruction on
  every image input. `table_to_element` moved from the orchestrator to
  `ingest/views.py`, joining the reverse projections so the whole
  view↔element mapping is in one file; its body is unchanged.

  Stale claims corrected in the same pass, all of which predated this work:
  steering's "every image input … is still unchanged"; `money.md`'s note
  that `dense_text_548` is out of reach because it is a PNG (it is reached
  — what limits it is grid quality on a stacked-header table, which #17 B2
  owns); `get_extractor`'s docstring; CLAUDE.md's and dataflow's
  "non-PDFs via `get_extractor`"; and the generated EXTRACTION.md
  strategy-matrix row `| IMAGE | ImageExtractor (legacy) | Direct PaddleOCR |`.

- **Table-cell reconstruction on OCR'd pages (#17), step B2 — precision
  gates calibrated.** The reconstructor's precision gates were provisional
  structural constants; B2 calibrated them against the rendered-clean
  cohort (must reconstruct) and a false-table cohort (must refuse). A new
  `MIN_ROW_FILL_RATIO = 0.75` gate in `ingest/ocr_tables.py` — mean cell
  occupancy across the reconstructed body — is the load-bearing signal:
  measured, the clean fixtures fill 0.98–1.00 and the hard/false shapes
  0.375–0.49, so 0.75 sits in the empty gap. The over-segmented,
  over-merged grid a hierarchical or form shape produces is structurally
  large but mostly empty, which the existing `MIN_*` count gates and the
  left-edge `MIN_ASSIGNED_RATIO` could not see; density does.

  Effect: all six rendered-clean fixtures still reconstruct, the eight
  false-table probes (3 non-table DocLayNet pages + 5 FUNSD forms) all
  refuse — closing three false positives the provisional gates had let
  through (`diverse_layout_49` 32×3, `funsd/82200067_0069` 15×8,
  `funsd/87528321` 21×6) — and `dense_text_548` refuses rather than
  emitting its pre-B2 12×12 partial. The plan's right-edge overflow
  question is resolved by measurement: the overflow signal is 0 on every
  fixture (`column_for_x` absorbs right-of-last-column content), so density,
  not assigned-ratio symmetry, is the guardrail.

  Benchmark additions in `tests/test_table_benchmark.py`: an alignment
  projection (`cells → DataFrame`, header row → uniquified column names)
  feeding `utils/tabular_metrics.py` (`structural_fidelity` +
  `data_integrity`), so a reconstructed OCR grid is scored by the same
  metrics the spreadsheet ingest uses; the false-table cohort
  (`TestFalseTableCohort`, false-positive count gates the build); and the
  Appendix A.6 GT acceptance checker (`TestGroundTruthAcceptance`,
  parametrised over every `*_table.csv` beside a DocLayNet fixture). Wiring
  these into the generated `EXTRACTION.md` and the CI-level regression gate
  remains B4/B5.

### Fixed
- **A declined continental number no longer leaks its decimal tail as an
  amount.** In Australian (default) mode `find_money` correctly refuses to read
  `1.234,56` — the reading is ambiguous and `international_numbers` is the
  deliberate opt-in — but declining the candidate that *starts* at the run left
  the rest of it exposed, and `,56 EUR` is itself a complete match for the
  suffix patterns. `1.234,56 EUR` came back as `56 EUR`, a value wrong by 10³,
  which is precisely the failure the guard exists to prevent. Ambiguous numeric
  runs (continental decimals, and malformed thousands groups like `$1,23`) are
  now blocked whole, so the amount is missed rather than misread.

  Only the ISO-suffix, currency-word and symbol-suffix patterns leaked;
  prefix-marker forms were always safe, because the tail has no leading marker
  to match. That asymmetry is why the existing locale test (`€1.000,50`, a
  prefix form) passed throughout. International mode is unaffected — there the
  continental reading is the correct one.

- **CI runs the type-check and test steps again.** `ruff` was declared
  unpinned, so CI resolved 0.16.0, whose expanded *default* rule set reported
  297 errors across a tree that had been green — 233 of them pre-existing and
  unrelated, surfaced by the release rather than by any change. Lint runs
  before mypy and pytest, so both were being skipped entirely and the `money`
  op merged without CI ever executing its tests. `ruff` is now bounded
  (`>=0.16,<0.17`) so an upstream release can no longer turn CI red on its own;
  raising that ceiling is a deliberate commit that also clears whatever the new
  defaults flag.

  The tree is now clean under 0.16.0's defaults. `--fix` resolved 174 findings
  mechanically; `BLE001`, `S110` and `S112` are suppressed in
  `[tool.ruff.lint]` because the codebase deliberately does what they flag —
  every site is a batch- or per-document isolation boundary, and narrowing
  those handlers to named exception types would let one malformed document
  abort a 1500-document run. The remaining 55 were resolved individually.
  Two are worth noting beyond the mechanical: the readability smoke-tests in
  `store/output.py` / `store/shard_audit.py` (`pq.ParquetFile(p).metadata`,
  whose whole purpose is to raise on a corrupt footer) now bind their result
  rather than being deleted as useless expressions, and `analyse/graph.py`
  carried a crossreference edge whose `source` was the same value on both
  arms of its conditional — collapsed to the value it already produced, so
  behaviour is unchanged, but the condition looks like it was meant to
  distinguish something.

- **The Isaacus test suites run in CI.** CI installed `.[dev,cloud]`, omitting
  the `isaacus` extra, so the enrich / graph / query / embed modules hit their
  module-level `importorskip("isaacus")` and skipped wholesale — 66 tests that
  need no API key never ran. CI now installs the extra; only the 10 tests
  requiring a live endpoint still skip on the missing key. Installing the SDK
  also unmasked a real typing error in `process/chunker.py`, where
  `isaacus_client` is deliberately `object | None` to keep the module SDK-free;
  the narrowing to semchunk's concrete client type now happens at the call
  boundary.

- **`mypy` passes with `openpyxl` installed.** The new read-only openpyxl pass
  below was the codebase's first import of it and had no entry in the
  `ignore_missing_imports` override list, so the type-check leg failed on a
  missing stub package.

- **Spreadsheet extraction preserves `number_format` and a numeric
  `value_type`.** `ingest/spreadsheet.py` read cells with pandas
  (`dtype=str`), which discards both, so every `sheet_cell` element landed with
  `value_type="text"` and `number_format=None` despite `ELEMENT_SCHEMA` having
  columns for each. A second read-only openpyxl pass now supplies them.
  Values are untouched — the pandas read stays authoritative, so the verbatim
  contract ("1,234" stays "1,234") is unchanged.

  This matters because a register's money column is frequently identifiable
  *only* from its format: a GrantConnect award export carries `$#,##0.00` on
  48,997 cells whose text is a bare `50000`, and no cell, header or value in
  that workbook contains a currency symbol. The format was the sole
  unambiguous currency marker in the file and was being dropped at the
  extraction boundary, where no downstream stage could recover it. Only
  non-`General` formats are retained, keeping the lookup small. CSV sheets have
  no cell formats and are unaffected; a failed openpyxl pass logs a warning and
  leaves the fields unset rather than failing extraction.

## [0.2.0] - 2026-07-19

### Added
- **Pre-extracted records ingest (`ingest/records.py`).** Turns already-clean
  text records (a JSONL corpus; the Open Australian Legal Corpus) straight into
  the standard element-shard layout (`*.elements.parquet` + sidecars +
  `*._manifest.parquet`) so the `enrich → chunk → embed → graph-refresh`
  pipeline runs over a pre-extracted corpus unchanged — unlike the register
  ingests (`gnaf`/`abn`/`geo`) which *bypass* the NLP pipeline, this one *feeds*
  it. `source_hash = sha256(record_id + text)` is content-addressed (unchanged
  records are cache hits on re-ingest); text is split into paragraph blocks so
  the reassembled narrative round-trips. Corpus-agnostic — a
  `RecordFieldMapping` (declared by a thin `stories/<corpus>` config) names the
  id / text / provenance fields. Record metadata lands in a
  `*.provenance.parquet` sidecar (`store/provenance_output.py`) consolidated
  into a run-root `manifest.parquet` (source_hash → provenance).
- **Token-budget request packer (`utils/token_packer.py`).** Isaacus rate
  limits bind on *tokens per request/window*, not request count, so requests
  are packed by exact local token counts from the kanon-2 tokenizer:
  `pack_by_tokens` groups items to `min(max_items, token_budget)`; an
  over-budget item is sent solo; `split_on_boundaries` splits an over-ceiling
  document on blank-line boundaries into offset-tagged segments. `TokenCounter`
  is a cached, offline wrapper over the tokenizer.
- **Enrichment — token-aware batching + long-doc split (`enrich_stage.py`).**
  Replaces the one-doc-per-call loop with packer-driven requests of
  `min(max_texts_per_request=8, token_budget)` (8× fewer requests for small
  docs; token-aware so a batch of long judgments never overpacks a
  429-triggering request). A doc over `split_ceiling` is split and its
  per-segment results offset-merged (`analyse/enrich_merge.py`). `enrich.py`
  honours a `Retry-After` header on 429. New `EnrichmentConfig` knobs:
  `tokenizer`, `max_texts_per_request`, `token_budget`, `split_ceiling`.
- **Graph-edge refresh stage (`analyse/graph_refresh.py`, `womblex
  graph-refresh`).** Offline, deterministic rebuild of mention→chunk edges from
  the entity + chunk sidecars (both carry char offsets) — needed because AI
  chunking runs *after* enrichment, so the enrich-time graph has no chunk edges
  yet. Populates `enrichment_entities.chunk_index` and refreshes
  `*.graph_edges.parquet`, preserving hierarchy/citation edges. Idempotent.
- **Offline kanon-2 tokenizer.** The tokenizer is vendored under
  `_models/kanon-2-tokenizer` and resolved locally by both the token packer and
  `create_chunker` — no Hugging Face round-trip per run, offline-safe.
- **Distributed / cloud execution (`womblex[cloud]`).** Optional scale-out for
  long batch runs without changing the local CPU-first default. Three pieces:
  (1) `store/remote.py` — an fsspec stage-in/stage-out object-storage adapter
  (S3/MinIO/GCS/local) that confines all remote-storage knowledge to one place
  so the `Path`-based stages stay untouched; (2) `cloud/queue.py` — a Postgres
  `FOR UPDATE SKIP LOCKED` job queue over one `womblex_jobs` table where the row
  `status` *is* the distributed checkpoint (idempotent re-enqueue on
  `(run_id, batch_num)`, per-job retry, crashed-worker requeue); (3)
  `cloud/worker.py` — a worker that claims a batch, stages its inputs, runs the
  shared `batch.process_batch` body, and publishes `batch-NNNN.*.parquet` shards
  back. New CLI: `womblex enqueue` / `worker` / `jobs` / `finalize` (the last
  consolidates a distributed run's shard manifests into
  `<run>/manifest.parquet` in the store — the explicit end-step `womblex run`
  performs locally). Outputs are the ordinary shard layout, so `manifest` /
  `chunk --shards` consume a distributed run exactly like a local one.
  `process_batch` is also now the single shared body behind `womblex run`, so
  local and distributed modes cannot diverge.
- **Container image + compose stack.** `Dockerfile` (extraction + `[cloud]`)
  and `docker-compose.yml` bundling Postgres (queue), MinIO (object store), and
  horizontally scalable workers (`docker compose up --scale worker=N`).
- **CI security job.** `ci.yml` gains a `security` job: Semgrep SAST over `src/`
  with the Python + OWASP Top Ten rulesets (blocking) and `pip-audit`
  dependency scanning (informational — the ML dep tree carries advisories we
  can't action directly). The test job now also installs the `cloud` extra so
  the object-storage tests run in CI.
- **ABN Lookup bulk extract ingest.** New `ingest/abn_bulk.py` stream-parses
  the ABR bulk extract XML files (`yyyymmddPublicNN.xml`, ~6 GB uncompressed
  across 20 files) with constant memory and writes two Parquet files per
  input: `<stem>.parquet` (one row per ABR record — ABN/status, entity type,
  main entity name or legal-entity name parts with given names as separate
  `given_name_1` / `given_name_2` columns since a single given name may
  itself contain a space, state/postcode, ACN, GST) and
  `<stem>_names.parquet` (one row per registered name — main/legal, business,
  trading and DGR fund names keyed by ABN, ready for `link/` register
  consumption). Values are verbatim strings, absent optionals are `""`, and
  provenance (schema version, source file, MD5, row counts) rides as parquet
  metadata — the `ingest/gnaf.py` pattern. Failures are isolated per file:
  any error (malformed XML, read/write failure) logs with the source name,
  removes partial output, and lets the directory ingest continue. New
  `womblex ingest-abn <file|dir>` CLI command; bypasses the NLP pipeline.
  (`ingest/abn_bulk.py`, `cli/ingest.py`, `tests/test_abn_bulk.py` —
  all-synthetic fixtures.) The shared MD5 helper moved to
  `utils/checksum.py`, replacing the per-module copies in `ingest/gnaf.py`
  and `ingest/geospatial.py`.
- **Spreadsheet preamble/header detection.** Export products that open with
  title rows, generated-date lines or `key: value` metadata blocks above the
  real header (e.g. AusTender contract-notice exports, agency stats
  workbooks) previously had the first row parsed as the header, with pandas
  fabricating `Unnamed: N` column names that landed verbatim-violating cell
  values on the element stream — and ragged CSVs (one-field title row above
  a wide header) failed outright. Sheets are now read with `header=None`
  (CSVs via a new field-count-sniffing `read_csv_raw`, capped at `nrows`
  when sampling) and split via `split_preamble`: the header is the
  candidate row (≥2 non-empty cells in a 10-row window) that starts the
  longest run of table-consistent rows below it — a blank separator or the
  wider table below breaks a title/metadata row's run, ties prefer the
  wider candidate, single-cell section rows are neutral, and a width-ratio
  rule plus row-0 fallback covers header-only and single-column sheets.
  Preamble rows land verbatim on the sheet_meta element
  (`meta["preamble"]`) and the row-0-is-header contract of the cell grid is
  preserved for downstream table views. Header-first, single-column and
  uniformly narrow (key/value, glossary) sheets are unaffected. Detection
  (`_detect_spreadsheet`) shares the same reader and split with headroom so
  the 500-row classification sample is unchanged, and `SheetInfo.key_column`
  resolves against the real header. (`ingest/spreadsheet.py`,
  `ingest/detect.py`.)
- **Run-level document manifest.** `womblex run` now consolidates the per-batch
  `batch-NNNN._manifest.parquet` sidecars into a single
  `<run_root>/manifest.parquet` at the end of the run — the published
  documents table mapping `source_hash` (the join key on every chunk/sidecar
  row) back to `doc_id`, `filename`, extraction method, counts and status, so
  shipped chunks are attributable to their source documents. A new
  `womblex manifest --shards <dir> [-o PATH]` command regenerates it for
  existing runs. (`store/run_manifest.py`, `cli/pipeline.py`.)
- **Shippable enrichment graph.** `enrich_shards` now writes a
  `*.graph_edges.parquet` sibling per batch alongside the entities/meta
  sidecars — the Kanon-2 document graph (containment, segment hierarchy,
  person/location hierarchy, citations, cross-references, contact-info and
  date relations) flattened to the existing `GRAPH_EDGE_SCHEMA`, with
  `document_id` carrying the `source_hash` so it joins the other sidecars.
  When the batch already has a `*.chunks.parquet` sibling, narrative chunks
  are mapped in so the graph includes mention→chunk edges. On resume, a batch
  missing its graph sidecar is re-enriched so prior runs gain it (the graph is
  only buildable from the live enrichment result). New
  `write_graph_edges_shard` / `read_graph_edges` / `graph_edges_path_for`.
  (`analyse/enrich_stage.py`, `store/enrichment_output.py`.)
- **CLI fix — `womblex chunk --shards` + `--config` combinable.** The two
  flags were in a `required=True` mutually exclusive argparse group, which made
  the `--shards` branch's config handling (chunking settings, `chunking_model`
  for AI chunking, `processing.text_source`) unreachable from the CLI —
  per-stage AI chunking was dead-ended. They now combine: `--shards` with
  `--config` sources chunking settings from the YAML; `--config` alone remains
  the E2E composition mode. (`cli/pipeline.py`.)
- **Single-enrichment reuse for AI chunking.** When AI chunking
  (`chunking.chunking_model`) and the `enrich` stage are both on, the enrich
  stage now persists the raw ILGS Document per doc to a new
  `*.enrichment_doc.parquet` sidecar (opt-in `enrichment.persist_document`,
  auto-enabled by `WomblexConfig`), and the chunk stage reuses it for semchunk's
  AI path instead of re-enriching — eliminating the double Kanon-2 call. Reuse is
  gated by a **byte-identity guard**: a persisted `Document.text` is used only
  when it equals the chunk stage's freshly reassembled narrative for that
  `source_hash`; on any mismatch (different `text_source`, stale/corrupt blob,
  absent sidecar) the doc falls back to self-enrich, so offsets can never desync
  the PII mention↔chunk mapping. Requires running `enrich` before `chunk`; the
  `WomblexConfig` validator now warns about that ordering rather than about
  double-enrichment. New self-contained `store/enrichment_doc.py`;
  `enrich_documents_raw` / `enrich_document_raw` expose the raw SDK Document;
  `chunk_batch` gains `narrative_overrides`. Verified live against
  `kanon-2-enricher` (gates in `docs/decisions.md`). (`store/enrichment_doc.py`,
  `analyse/enrich.py`, `analyse/enrich_stage.py`, `process/chunker.py`,
  `process/chunk_stage.py`, `config.py`, `cli/link.py`.)
- **AI chunking pass-through (semchunk 4).** `ChunkingConfig.chunking_model`
  (default `null`) enables semchunk 4's AI chunking — chunk boundaries follow
  the Isaacus enricher's (`kanon-2-enricher`) structure spans instead of the
  offline token/recursive split. Opt-in and off by default, so callers using a
  non-Kanon tokeniser keep purely offline chunking (composable). `create_chunker`
  now forwards `chunking_model`, `isaacus_client`, and `tokenizer_kwargs`
  straight to `semchunk.chunkerify` (thin-adapter doctrine — semchunk's params
  are the feature surface); threaded through both the E2E `run_chunking` path and
  the per-stage `chunk_shards`. Graph-reuse across the chunk + enrich stages (so
  the narrative is enriched once, not twice) is now implemented — see
  "Single-enrichment reuse for AI chunking" above. Bumps `semchunk>=3.0` →
  `>=4.0`. (`process/chunker.py`, `config.py`, `process/chunk_stage.py`,
  `operations/chunk.py`.)
- **`spellfix` stage — dictionary-gated OCR character-confusion repair
  (`womblex spellfix`).** A separate, opt-in cleaning op (distinct from the
  fidelity-neutral `normalise`) that fixes digit/letter glyph confusions
  (`chi1d`→`child`). Validates candidates against the bundled en_AU Hunspell
  dictionary (`spylls`, harvested from the Australian Writing MCP; MIT/SCOWL) and
  rewrites a token only on three gates: out-of-dictionary trigger,
  single-character in-dictionary candidate, and a *unique* such candidate.
  Default Tier A swaps only OCR digit→letter homoglyphs (length-preserving);
  Tier B general edit-distance-1 is opt-in (`--general` / `general_edits`,
  carries proper-noun risk). Repairs at the **element layer** (reads
  `*.elements.parquet`, chaining off the normalise overlay when present) and
  writes a `*.spellfix_text.parquet` element-text overlay + a
  `*.spellfix_corrections.parquet` audit — raw elements untouched. New deps:
  `spylls`; bundled dict under `_models/en_AU`. (`process/spellfix.py`,
  `process/spellfix_stage.py`, `store/spellfix_output.py`, `cli/spellfix.py`,
  `SpellfixConfig`.)
- **Composable element-text overlays via one `processing.text_source`.** New
  `process/text_overlay.py` resolves the normalise / spellfix element-text layer
  selected by a single pipeline setting (`elements` | `normalised` | `spellfix`)
  and applies it before reassembly at **both** the chunk and enrich sites, so
  chunking, embeddings, Kanon-2 enrichment and PII all consume the same repaired
  text in one offset coordinate space. Deliberately one knob (not per-stage):
  enrichment runs on the whole document and PII maps mention offsets onto chunks
  via `chunk.start_char`, so the enricher input and chunk source must match.
- **Enricher `overflow_strategy` (default `auto`).** `enrich_documents` /
  `EnrichmentConfig` now pass `overflow_strategy` to `enrichments.create`,
  defaulting to `auto` (vs upstream `null`, which errors on >16k-token inputs).
  Kanon-2 chunks long documents internally and stitches the ILGS graph back into
  a single prediction; returned span offsets still index the full source, so the
  PII offset mapping is unaffected. Fixes long FOI bundles erroring on enrichment.
- **`score --text-source` — CER of extraction vs normalisation.** `womblex
  score` (and `score_labels`) now accept `text_source={elements,normalised}`:
  `normalised` reassembles the labelled page from the `*.normalised_text.parquet`
  sidecar instead of the verbatim element stream, so a caller can measure how
  the cleanup/normalisation stage changes CER against the same GT.
- **Benchmark: ACT-ECI labelled-pages raw-vs-normalised CER.** New
  `TestActEciLabelledPages` (`-m benchmark`) extracts each labelled page, scores
  raw extraction and normalise-stage output against the per-page GT, and reports
  a per-strategy `Raw CER / Norm CER / Δ` table in `docs/accuracy/EXTRACTION.md`.
  Degenerate GT (<20 chars) excluded; a regression guard asserts normalisation
  never worsens CER. (Fixtures cohort expanded 7→19 labelled pages.)
- **`quality` stage — chunk-quality annotation sidecar (`womblex quality`).**
  Reads `*.chunks.parquet` and writes a `*.chunk_quality.parquet` sibling
  (joined on `(source_hash, chunk_index)`) with ML-readiness flags
  (`char_len`, `alpha_frac`, `is_short`, `boilerplate_flag`) and cross-batch
  duplicate cluster ids (`exact_dup_id`, `near_dup_id`). Duplicate clustering
  is self-contained (no datasketch dep): `exact_dup_id` over
  whitespace/case/punctuation-normalised text, `near_dup_id` via a fixed-seed
  MinHash+LSH (default 64 perms / 4 bands ≈ Jaccard 0.92). Annotation only —
  chunk text is never mutated; runs as a single global pass since dedup is
  corpus-wide. `boilerplate_patterns` are corpus-driven config, never
  hardcoded. New `QualityConfig`; 5 unit tests.
- **`normalise` stage — `unicode_hygiene` transform.** Folds unicode
  whitespace (NBSP, en/em spaces, ideographic space, U+2028/9 separators) to
  ASCII space/newline and strips zero-width marks, BOM and stray control
  chars; smart quotes and em/en dashes are preserved. New `unicode_hygiene`
  toggle on `NormaliseConfig` (default on), composed ahead of the existing
  transforms. 4 new unit tests.

### Fixed
- **OpenCV 5 compatibility in skew detection.** `detect_skew_angle`
  (`ingest/heuristics_cv2.py`) unpacked `HoughLinesP` segments as `line[0]`,
  which assumes OpenCV 4's `(N, 1, 4)` layout; OpenCV 5 flattens to `(N, 4)`,
  turning `line[0]` into a scalar and crashing every OCR-path extraction
  (`TypeError: cannot unpack non-iterable numpy.int32`). The segments are now
  reshaped to `(-1, 4)` before unpacking, which accepts both layouts. New
  direct unit tests pin both shapes so the regression no longer needs the
  full OCR fixture suite to surface.
- **mypy no longer pins `python_version = "3.11"`.** The pin forced the CI
  3.12 matrix leg to re-check under 3.11 grammar — redundant with the 3.11
  leg, and broken once numpy ≥ 2.5 (3.12-only) began shipping PEP 695 `type`
  statements in its stubs, which mypy rejects under a 3.11 target. Each leg
  now checks at its own interpreter version.
- **Register manifest now covers `ingest-geo` and derives roles from footer
  metadata only.** `cmd_ingest_geo` never called `write_register_manifest`
  despite the documented `abn`/`gnaf`/`geo` coverage, and the module's
  namespace whitelist said `geo` while `ingest/geospatial.py` writes
  `geospatial.*` footer keys — so geo outputs could not be indexed at all.
  The namespace is now taken from whichever `<ns>.source_file` footer key is
  present (no per-register registry to keep in sync), geo ingest writes the
  manifest like ABN/G-NAF, and the ABN ingest tags each output with an
  `abn.role` footer key (`records`/`names`) so the manifest's role column
  comes from metadata rather than the `_names` filename suffix — the exact
  glob-style fragility the manifest exists to remove. ABN outputs written
  before this change lack the role key and re-index as `records`; re-run
  `ingest-abn` to restore the distinction. Also renames the module's
  `RUN_MANIFEST_FILENAME` constant to `REGISTER_MANIFEST_FILENAME` — it had
  borrowed the *run* manifest's constant name from `store/run_manifest.py`
  while naming a different artefact.
- **`RemoteStore` no longer leaks s3fs-shaped options into non-S3 backends.**
  `storage_options_from_env()` built AWS-style kwargs (`key`, `secret`,
  `client_kwargs.endpoint_url`) and `from_uri` applied them to *any* remote
  protocol, so a `gs://`/`az://` store with AWS env vars set (common in mixed
  environments) got misconfigured. The helper now takes the target URI and
  returns options only for `s3://`; other backends authenticate via their own
  native mechanisms. Also: `womblex enqueue`'s batch-size fallback now reads
  `ProcessingConfig()` instead of restating the default, and the worker
  derives its shard-upload glob from the `BatchOutcome.shard_path` the batch
  reported, so the `batch-NNNN` naming scheme lives only in `womblex.batch`.
- **K9-fig — full-page scans no longer dropped from chunking as `figure`.**
  The dominant-region fallback in `_layout_blocks_and_tables` collapsed a
  whole page's OCR onto one block tagged with the largest layout region's
  kind; when that was DocLayNet `Picture` → `figure`, a full-page scanned
  document became a single `figure` element, which is excluded from the
  chunk narrative (`figure` ∉ `TEXT_KINDS`) — silently losing the document.
  New shared helper `_ocr_region_block_type(text, layout_kind)` promotes a
  non-text fallback kind to `paragraph` when the OCR yields ≥5 words; sparse
  output (page-number stamps, bare logos) keeps `figure`. (The original E4
  audit mis-attributed this to `_ocr_image_regions`; that path now routes
  through the same helper.) On the ACT-ECI corpus: `figure` 1,200→154,
  `paragraph` +1,046, and all 16 previously zero-chunk complaint documents
  now produce chunks (docs-with-chunks 2,610→2,626). 4 new unit tests.

### Added
- **I7 — entity-link sidecar: `womblex enrich` + `womblex link` per-stage CLIs.**
  Two new per-stage stages mirroring `womblex chunk --shards`, each with an
  independent `CheckpointManager` and per-batch sibling parquets.
  `womblex enrich --shards <dir>` reassembles each doc's narrative
  (`reassemble_narrative`), calls the Kanon-2 enricher one doc at a time
  (per-doc failure isolation), and writes `*.enrichment_entities.parquet` +
  `*.enrichment_meta.parquet` (reusing `store/enrichment_output.py` schemas,
  keyed on `source_hash`). `womblex link --shards <dir> --config <yaml>`
  resolves enrichment candidates (corporate persons + address locations) to a
  reference register and writes `*.entity_links.parquet`. **Generic by design:**
  the schema uses an `entity_type` discriminator (no domain columns), the
  matcher (`link/matcher.py`) is generic record-linkage (alias → address-exact
  → token-set name-fuzzy, stdlib `difflib`, no new dependency), and the corpus
  declares register column-roles via the new `linking`/`reference` config — the
  library knows nothing about specific registers. Reference loading
  (`link/reference.py`) is bundle-aware by interface (CSV implemented; the
  multi-file/geospatial seam is reserved, not built). Doc-grain attribution is
  a derived read view over the persisted mention-grain rows, not a second file.
  New `isaacus` is the optional extra (`uv sync --extra isaacus`). Live smoke
  over the 17-doc Artemis set attributed 15/17 to the correct canonical service
  (`SE-40002132`); the 2 misses are an enrichment-recall gap and an
  OCR-typo+no-address doc, not matcher faults. New `tests/test_link.py` (23) +
  `tests/test_enrich_stage.py`; full fast suite green.
  - **Matcher** uses stdlib `difflib` only (no rapidfuzz dependency): alias →
    address-exact → name-fuzzy, where name-fuzzy combines a token-set ratio
    (suburb-suffix recall, cross-brand precision) with OCR-tolerant per-token
    char similarity (folds "Earty"→"Early" while still rejecting a different
    brand). With OCR tolerance the Artemis smoke reaches **16/17**.
  - **`enrich`** isolates per-doc failures and, critically, does **not**
    checkpoint a doc whose enrichment errored — a transient/connection failure
    stays unprocessed so a resume retries it (regression-tested).
- **`womblex embed --shards` — chunk embeddings stage (Kanon-2 embedder).**
  `analyse/embed.py` (thin `embeddings.create` wrapper: 128-text batching, 429
  retry, order-preserving, `retrieval/document`/`query` task-aware) +
  `analyse/embed_stage.py` (`embed_shards` over `*.chunks.parquet` →
  `*.embeddings.parquet`, one vector per chunk, per-stage `CheckpointManager`,
  batch-level failure isolation) + `cli/embed.py` + `EmbeddingConfig` +
  `EMBEDDINGS_SCHEMA`/IO. The substrate for downstream search/clustering and a
  doc→entity attribution backstop for no-extraction docs. `tests/test_embed_stage.py`.
- **I8–I10 — `womblex pii --shards` per-stage CLI: graph-driven detection +
  `<PERSON_n>` masking.** Reads `*.chunks.parquet` + `*.enrichment_entities.parquet`
  and writes `*.pii_spans.parquet` (audit; one row per span with `entity_id` +
  `replacement`) plus a masked `*.clean_text.parquet` (publishable text layer,
  drop-in for chunks). The Kanon-2 graph is the primary entity source
  (`natural`→PERSON, `address`→ADDRESS); graph spans map onto narrative chunks
  via `chunk.start_char`. Masking is **terminal** — applied after enrich + embed,
  never rewriting the raw chunks that feed Isaacus. Tags are typed + numbered per
  document off the graph entity (`<PERSON_1>`…). `pii/cleaner.py` refactored to a
  span-returning `detect_spans()` + shared `_anonymize()`; the regex/cosine
  backstop is now opt-in (`PIIConfig.use_regex_backstop`, default off — low
  precision). New `store/pii_output.py`, `pii/pii_stage.py`, `cli/pii.py`,
  `PIIConfig.write_clean_text`. `tests/test_pii_stage.py`.
- **I3 — `womblex redact --shards` per-stage CLI.** `womblex redact` is
  now dual-mode, mirroring `womblex chunk`: `--shards <dir> --pdfs <dir>`
  runs per-stage redaction detection over an existing extraction shard
  directory and writes `*.redactions.parquet` siblings; `--config <yaml>`
  runs the E2E extract+redact path unchanged. The `--shards`/`--config`
  group is mutually exclusive and required. `--pdfs` is mandatory in
  `--shards` mode because detection rasterises the source pages (unlike
  chunking, which works purely from the element stream). The per-stage
  path calls the existing `redact.batch.annotate_redactions_for_shards`
  engine via a shared `_run_redact_shards` helper. 8 new CLI tests.

### Changed
- **`operations.py` split into an `operations/` package.** The 902-line module
  (over the 750-line cap) became one module per independent operation
  (`models`/`extract`/`redact`/`chunk`/`pii`/`enrich`/`persist`), each ≤90
  lines. The flat import surface (`from womblex.operations import run_extraction`,
  …) is preserved by `operations/__init__` re-exports — no caller changes.
  Behaviour-neutral; `test_integration.py` patch targets for `create_chunker`
  moved to `womblex.operations.chunk`.
- **Resume-integrity self-heal generalised across stages.** `store/shard_audit.py`
  gains `reconcile_stage_checkpoint_with_shards(mgr, dir, *, suffix)`; the chunk
  reconcile now delegates to it, and `enrich`/`link`/`embed` wire it (+
  `--no-verify-resume`) — so every `CheckpointManager`-backed stage drops +
  re-does batches with corrupt sidecars on resume, identically.
- **I5 — SemChunk wrapper audit (P2).** Audited `process/chunker.py`
  against semchunk 3.2.5: `create_chunker` exposes every `chunkerify`
  creation parameter and `chunk_batch` passes every relevant
  `Chunker.__call__` parameter through (`offsets=True` pinned because
  Womblex needs char offsets for page mapping). No semchunk-native
  surface is reimplemented or shadowed. **Removed the dead
  `ChunkingConfig.batch` flag** — it mapped to no semchunk parameter,
  was consumed by no code path, and its description referred to the
  pre-I2 per-document-vs-batch behaviour that I2 deleted (chunk_batch
  always batches the whole input list). **Widened `chunk_size` to
  `int | None`** so semchunk's auto-derive path (`None` → size from the
  tokeniser's `model_max_length`) passes through faithfully; the default
  stays `480` (the Kanon-2 window), so behaviour is unchanged unless a
  config explicitly sets `chunk_size: null`. Documented the adapter
  boundary explicitly in the `chunker.py` module docstring, the
  `ChunkingConfig` docstring, and `docs/extraction.md`; the three
  default divergences from upstream (`tokenizer`, `chunk_size=480`,
  `processes=1`) are each annotated with their corpus reason. Pure
  thin-adapter cleanup — chunk output is byte-identical to I2 by
  construction (the removed field was never read; the new `None` path is
  opt-in). 98 chunker/config/pipeline/output tests pass; 79 integration
  tests pass.
- **`annotate-redactions` is now a deprecated back-compat alias** for
  `redact --shards <dir> --pdfs <dir>`. Its positional-argument surface
  (`annotate-redactions <shards> <pdfs>`) is preserved verbatim and routes
  through the same `_run_redact_shards` helper, so existing scripts keep
  working with byte-identical output. New callers should prefer
  `redact --shards`. The redact stage retains the engine's JSON
  `--checkpoint` rather than the `CheckpointManager` used by `chunk`;
  unifying the two is a deferred P1 follow-up.

### Added
- **I2 — `womblex chunk --shards` per-stage CLI + `chunks.parquet`
  sidecar.** New `CHUNKS_SCHEMA` (source_hash, chunk_index, text,
  start_char/end_char, content_type, has_redaction, page_start /
  page_end) in `store/output.py` with `write_chunks` / `read_chunks` /
  `verify_chunks_persistence`. New `process/chunk_stage.py` walks a
  shard directory, reassembles narrative + tables from the element
  stream per source_hash, calls the single `chunk_batch` engine, and
  writes a `*.chunks.parquet` sibling per batch. Per-stage
  `CheckpointManager` keyed `<dataset>_chunk_checkpoint.json`;
  chunks-side resume integrity in `shard_audit.scan_chunks_directory`
  / `reconcile_chunk_checkpoint_with_shards` archives corrupt
  `*.chunks.parquet` independently of the element-stream files. The
  shared `chunk_batch` powers both per-stage `chunk_shards` and E2E
  `operations.run_chunking`, so `--shards` and `--config` modes feed
  semchunk identical inputs.

### Changed
- **`process/chunker.py` collapsed against semchunk v3+ surface.**
  Deleted per-doc wrappers `chunk_text`, `chunk_texts_batch`,
  `chunk_document` (+ `_chunk_document_sequential` /
  `_chunk_document_batch` dispatchers) — semchunk already batches
  across a list of texts and parallelises over `processes` workers
  when handed one. The new single entry point `chunk_batch(inputs,
  chunker, ...)` flattens every doc's narrative into one semchunk
  call (with `overlap`) and every doc's table markdowns into another
  (no overlap), so `processes` and the progress bar parallelise
  across the entire batch instead of being thrown away per-document.
  `TextChunk` gained `page_start` / `page_end` (nullable); the
  redaction-split repair pass propagates page spans across a merge.
  New helpers `reassemble_narrative`, `collect_tables_from_elements`,
  `build_chunk_input` formalise the "element stream → ChunkInput"
  projection shared by both invocation paths.
- **`operations.run_chunking` rewired through `chunk_batch`.** Builds
  one `ChunkInput` per completed result from
  `dr.extraction.elements` (canonical), not `dr.extraction.full_text`
  (which is derived from `pages` and reflects in-memory mutations).
  Behaviour change: in-memory PII / redact-blackout mutations to
  `pages[i].text` no longer flow to chunks under `womblex run`.
  Aligns the E2E path with the per-stage one (both consume the
  element stream); future PII / redact stages will reattach via
  their sidecars per P1.

### Added
- **Shard integrity scan on `--resume` (E1).** New
  `womblex.store.shard_audit.reconcile_checkpoint_with_shards` runs at
  the top of `cmd_run` when `--resume` is given. Walks every batch's
  four sibling parquet files: confirms presence + non-empty + parquet-
  readable, and that manifest `elements_count` / `table_cells_count` /
  `form_fields_count` sums match the actual sidecar row counts. Any
  batch failing a check has its `doc_id`s dropped from the checkpoint
  and its files renamed with a `.corrupt` suffix so reader globs
  (`*.elements.parquet` etc.) skip them; the dropped docs get re-
  extracted into new batches past the high-water mark. Batches whose
  manifest is itself unreadable can't be reconciled automatically (no
  way to enumerate `doc_id`s) — they're logged loudly and the operator
  is told to re-run without `--resume`. Defaults on; opt out with
  `--no-verify-resume`. Closes the silent-failure class of post-write
  filesystem corruption (drive glitch, partial sync, manual deletion)
  that motivated the i1b batch-0087 0-byte incident.

- **`womblex verify-shards` CLI (E2).** Audits a run / shard directory
  for corruption + cross-batch consistency; takes a shard dir or a run
  root (auto-detects `documents/`). Reports per-batch integrity, total
  elements / methods / kind counts, dupe and empty hashes. With
  `--compare-to <other>` produces a side-by-side diff against another
  run (useful for K-cluster-style "what changed between two
  extractions" investigations — promotes the ad-hoc `i1b_audit.py`
  pattern to first-class library + CLI). Optional `--input-dir <pdfs>`
  surfaces source-vs-manifest count drift. Exits 2 when corruption is
  detected so CI / cron pipelines can fail loudly. New module:
  `womblex.store.shard_audit`. New tests in `tests/test_shard_audit.py`
  (19).

### Changed
- **Manifest schema gains `doc_id` column.** `MANIFEST_SCHEMA` now
  carries the extraction's `doc_id` directly, removing the implicit
  `Path(filename).stem == checkpoint.doc_id` coincidence that previously
  bound the resume reconcile join. The reader (`read_manifest`) is
  backward-compatible: manifests written before the bump derive
  `doc_id` from `Path(filename).stem` on read so existing runs reconcile
  without re-extraction. Parser version bump is intentionally deferred
  — the schema is additive and reads gracefully.

### Added
- **K7(b) — Document-layout YOLO model (DocLayNet).** New default layout
  checkpoint `yolo11n_doc_layout.pt` (5.37 MB,
  [Armaggheddon/yolo11-document-layout](https://huggingface.co/Armaggheddon/yolo11-document-layout),
  MIT) replaces the COCO-trained `yolov8n.pt` as the primary layout
  backend. `YOLOLayoutAnalyzer` detects the loaded model's taxonomy from
  its class names: DocLayNet's 11 document classes (Caption, Footnote,
  Formula, List-item, Page-footer, Page-header, Picture, Section-header,
  Table, Text, Title) map directly into womblex `ElementKind` values
  via the new `_YOLO_DOCLAYNET_LABEL_MAP`. COCO weights remain as a
  best-effort fallback when the DocLayNet checkpoint isn't resolvable.
  Inference imgsz follows a per-taxonomy default (DocLayNet: 832, COCO:
  640) — empirically equivalent on this corpus to the model card's 1280
  recommendation while matching COCO speed; override to 1280 when
  small-class (Caption / Footnote) recall matters. Closes the
  1,587-element `kind='figure'` mis-classification on scanned pages
  tracked in docs/decisions.md; unlocks Caption / Footnote producers
  (K6 closes as a side effect).

- **`footnote` ElementKind.** New text-bearing kind added to
  `ElementKind`, `TEXT_KINDS`, and `_BLOCK_TYPE_TO_KIND`. Primary
  producer is the DocLayNet `Footnote` class via the new label map.
  Downstream stages (PII / redact / chunk) operate on text kinds and
  pick up the new kind automatically through `TEXT_KINDS`. Future
  iterations may refine signatory / footnote separation now that the
  distinction is preserved.

- **K2′ — OCR form-pair bboxes.** New `_extract_form_pairs_from_regions`
  in `ingest/forms.py` walks per-region OCR detections (PaddleOCR /
  RapidOCR per-line bboxes) and produces `FormField` entries with real
  positions. `_apply_ocr_page` prefers this path; the legacy
  `_extract_form_pairs_from_lines` survives as a fallback for LLM-OCR
  engines that resolve reading order natively and don't emit per-region
  bboxes. Closes the K2′ silent-zero-bbox issue on 4,184 of 5,183 OCR
  form elements (80.7%). Same plumbing unblocks inline-per-span
  redaction markers on raster pages (P6 option (c); see docs/decisions.md).

### Changed
- **`@pytest.mark.slow` tests now run by default.** Removed the
  `-m 'not slow'` default from `[tool.pytest.ini_options].addopts`. The
  24 OCR-fixture tests in `tests/test_fixtures.py` were originally marked
  slow because they invoked EasyOCR (30+ seconds each). The backend has
  since moved to rapidocr-onnxruntime and the whole cohort completes in
  ~7 seconds, so excluding them was costing coverage without saving
  meaningful time. The `slow` marker is retained (description updated)
  so users can still pass `-m 'not slow'` or `-m slow` for ad-hoc
  filtering.

### Added
- **`run_id` + retention plumbing (I1 of publishable-corpus track).**
  Pipeline runs now write outputs to `<output_root>/<run_id>/documents/`
  rather than `<output_root>/documents/`. Run id resolution order:
  `--run-id` CLI flag → `dataset.run_id` in config → auto-generated
  `run-YYYYMMDDTHHMMSSZ` timestamp. `--resume` without a run id picks
  the most-recent existing run dir. Checkpoints follow under
  `<checkpoint_dir>/<run_id>/`.

  New `processing.retention` config block: `policy: rolling | keep_all`
  (default `rolling`) and `keep: int` (default `2`). On fresh runs (not
  `--resume`), old run dirs beyond the retention window are purged in
  lockstep with their checkpoint dirs. The current run is always
  preserved regardless of position.

  Foundation for stage-aware sidecar persistence (I2+); no extraction
  output content has changed.

  New module: `womblex.store.retention` (`generate_run_id`, `list_runs`,
  `most_recent_run`, `apply_retention`). New tests in
  `tests/test_retention.py` (20) and `tests/test_config.py` (8).

  Retention only considers subdirectories whose name starts with `run-`
  — legacy / hand-named output dirs (e.g. `output/documents/` from a
  pre-run_id layout) are preserved unconditionally and must be removed
  manually if no longer wanted. To bring a hand-named run under the
  policy, name it with a `run-` prefix.

### Changed — BREAKING
- **Extraction output reshaped to element-stream + typed sidecars.**
  `ExtractionResult` now carries `elements: list[Element]` as the
  canonical structural stream. Per-batch parquet output is split across
  four sibling files: `batch-NNNN.elements.parquet` plus
  `.table_cells.parquet`, `.form_fields.parquet`, `._manifest.parquet`.
  Sidecars are joinable via `(source_hash, parent_elem_order)`. The
  previous one-parquet-per-batch shape (`EXTRACTION_SCHEMA` with nested
  struct lists for tables / forms / images / text_blocks) is removed.
  See `docs/extraction.md` for the canonical reference.
- **Narrative ↔ table interleaving preserved.** DOCX bodies are walked
  in OOXML order so paragraphs and tables emit in their true position;
  PDF per-page outputs are sorted by `(y, x)` so within-page reading
  order survives.
- **Spreadsheets: one ExtractionResult per workbook.** Cells are the
  element grain (`kind='sheet_cell'`); sheets emit a leading
  `sheet_meta` element. The previous one-result-per-row shape is
  removed.
- **Verbatim text at the schema boundary.** `_normalise_text` no longer
  runs in the extraction hot path. Page text and element text are
  emitted verbatim from the producing extractor. Downstream stages may
  still mutate `pages[i].text`; the on-disk parquet is unaffected.
- **Legacy view properties remain.** `ExtractionResult.text_blocks` /
  `.tables` / `.forms` / `.images` are now read-only derived views over
  `elements` so downstream PII / redact / chunk stages continue working
  without change.
- **Sidecar integrity check.** `verify_shard_persistence` now confirms
  every `(source_hash, parent_elem_order)` in `table_cells` /
  `form_fields` resolves to an element with the matching kind. Drift
  raises `ShardVerificationError`.

### Added
- **Per-page extraction plan + orchestrator.** PDFs now route via per-page
  profiling (`ingest/page_profile.py`) and a doc-wide orchestrator
  (`ingest/orchestrator.py`) that dispatches operations page-by-page,
  rather than picking one document-level strategy. Correctly handles
  heterogeneous FOI bundles (e.g. one image-only page in an otherwise-
  native doc) that the previous sample-based detector missed.
- **Spreadsheet-print extractor** (`ingest/spreadsheet_print.py`) for
  native PDFs printed from a spreadsheet (CSV/Excel) — common shape for
  government FOI manifests, schedules, and registers. Extracts row-by-row
  data plus a metadata block (the label-value fields above the first
  data row). Handles 90°-rotated landscape pages, multi-line headers,
  centred header text within wider cells, and frozen headers across
  pages. Triggered by a cheap qualifier (filename hints + table signal)
  before structural vetting. Config-driven via
  `extraction.native.spreadsheet_print` (`metadata_location`,
  `filename_hints`).
- **Form-pair text extractor** (`ingest/forms.py`) — picks up "Field:
  value" form pairs in native PDFs (`_extract_form_pairs_from_text`)
  and OCR'd text (`_extract_form_pairs_from_lines`). Forms-column
  populated for ~94 % of docs in the ACT FOI corpus (was 0 %).
- **`block_type` classifier rewrite.** Native blocks now typed by
  position + typography (footer, header, signature, heading, paragraph)
  instead of length-only fallback. `caption` reserved for image-adjacent
  blocks downstream.
- **Per-image OCR on native pages** with embedded image regions (e.g.
  letterhead text). Generalised from hybrid-only via the existing
  `_ocr_image_regions` helper; recovers text-in-images that previously
  fell through extraction.
- **Multi-strategy table detection** — PyMuPDF `find_tables(strategy=
  "lines")` falls back to `strategy="text"` when the document uses
  whitespace-aligned columnar layout instead of ruled cells.
- **`TableData.context` and `ExtractionResult.document_metadata`** —
  per-table and document-level metadata blocks for spreadsheet-print
  layouts.
- `womblex profile` CLI command for per-column schema inference on
  CSV / XLSX / XLS / Parquet / NDJSON files.

### Changed
- **PDF dispatch flips from doc-level to page-level.** PDF strategy
  classes (`Native*Extractor`, `Scanned*Extractor`, `HybridExtractor`,
  `StructuredExtractor`) are deleted; `get_extractor()` now only
  dispatches non-PDF types (DOCX, spreadsheet, text, image). The
  orchestrator is the canonical PDF path.
- **Source redaction and PII tokens unified on angle brackets.**
  `redact/stage.py` blackout mode emits `<REDACTED>` (was `[REDACTED]`);
  `pii/cleaner.py` emits `<ENTITY_TYPE>` (e.g. `<PERSON>`, `<ADDRESS>`,
  `<EMAIL>`). Aligns with the labelling-packet ground-truth fixtures
  (`_transcript-with-redacted-tags*.txt`). The BPE/SentencePiece
  tokenisation argument that previously motivated square brackets didn't
  hold up — neither bracket style is single-piece in standard pretrained
  tokenisers without explicit special-token registration. Source
  redactions remain page-prefix (not inline per span) — inline-per-span
  requires bbox-to-text mapping not currently routed through to the
  redact stage; tracked as a follow-up.

### Internal
- Moved page-image morphology helpers from `detect.py` into
  `ingest/morphology.py` to keep `detect.py` under the 750-line cap.
- `_normalise_text` regex extended to catch fullwidth Unicode `Page`
  variants in OCR'd footers (`5lｐａge`, `3lＰａｇｅ`).

### Packaging
- MANIFEST.in no longer packages all of `src/womblex/_models/` — only the
  en_AU Hunspell dictionary ships, matching the declared package-data. The
  previous rule pulled every local model artefact (~110 MB compressed) into
  both the sdist and, via setuptools' `include-package-data` default, the
  wheel — over PyPI's 100 MB per-file limit. Large models remain resolved
  via `WOMBLEX_MODELS_DIR`.
- PyPI metadata completed: `readme`, project URLs; setuptools floor raised
  to >=77 for PEP 639 SPDX licence string support.
- The kanon-2 tokenizer (~5 MB) is bundled in the wheel alongside the en_AU
  dictionary, so pip-installed token packing and chunking resolve it locally
  with no Hugging Face round-trip, as already documented.

## [0.1.0] - 2026-04-28

### Added
- Initial public release.
- PDF extraction with document-type routing (native, scanned, hybrid, forms, tables).
- DOCX and spreadsheet (CSV/XLSX/XLS) ingestion.
- Standalone G-NAF and geospatial (SHP) ingest paths.
- Post-extraction redaction stage with configurable insertion points.
- Optional PII cleaning (PERSON detection via regex + cosine-similarity validation).
- semchunk-based chunking with table-aware splitting.
- Optional Isaacus enrichment via the `[isaacus]` extra.
- Local model resolution via `WOMBLEX_MODELS_DIR` for offline / edge deployment.

## Historical engineering notes (migrated from STATUS.md 2026-05-28)

> These sections record the point-in-time engineering detail behind the
> structured `[Unreleased]` / `[0.1.0]` entries above: the extraction-quality
> session bundle (native-page tables, block-aware paragraphs, footer
> pipe-as-I, the reverted OCR-table relaxation), the Phase 1–6 roadmap
> history, the CLI restructure, the detector evolution (vector-first +
> raster-path layout filter), and the redaction/PII marker-convention
> unification. The durable decisions, dead-ends, limitations and deferred
> backlog now live in [docs/decisions.md](docs/decisions.md); corpus-specific
> run history is a corpus concern (see the corpus's own status notes).
> (These notes were originally migrated from a now-removed `STATUS.md`.)

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

The filter is net-positive (more precise without regressing any doc) but the magnitude is too small to close §11's `scanned_mixed` false-positive gap on its own. A document-layout-trained checkpoint (DocLayNet / PubLayNet, with `Figure` / `Table` / `Form` classes) would be a better fit (the DocLayNet swap later landed — see CHANGELOG `[Unreleased]`).

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

[Unreleased]: https://github.com/DeepCivic/womblex/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/DeepCivic/womblex/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DeepCivic/womblex/releases/tag/v0.1.0
