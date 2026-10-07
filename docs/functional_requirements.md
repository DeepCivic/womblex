# Womblex — Functional Requirements

This document is the single source of truth for **what** Womblex must do, expressed
as user stories with testable acceptance criteria. It deliberately does **not**
cover *how* the system is built, why decisions were made, or measured accuracy.

## 1. Local Deployment Optimisation

**As** a user,
**I want** to run the full pipeline locally without any cloud account, API key, or network access,
**so that** I can process documents immediately and at low cost.

**Given** a machine with Python installed and no internet connection at runtime
**When** the operator installs Womblex and runs the local pipeline commands
**Then** extraction, OCR, chunking, and PII operations run CPU-only against the local filesystem with no external calls.

**Acceptance criteria:**

- A default local run requires no cloud account, object store, database, or API key.
- Models are bundled or resolved from a local directory, ensuring no network access is needed at runtime.
- The base installation includes all modules required for local text processing, OCR, chunking, and PII detection.
- External enrichment services and cloud APIs remain dormant (no outbound calls) until explicitly configured.
- PII detection runs locally, but its primary candidate source is the enrichment graph: a local-only run with no enrichment and the default config detects nothing. Local-only detection requires opting in to the low-precision regex/context backstop (`pii.use_regex_backstop: true`).
- A standard CPU-only local environment can successfully execute the supported local pipeline commands.

## 2. Scale-Out and Environment-Agnostic Execution

**As** a user,
**I want** to scale the pipeline to a cluster and move stages between local and cloud environments,
**so that** I gain throughput without rewriting jobs or re-extracting documents.

**Given** a configured object store and a transactional queue
**When** the operator enqueues work and starts additional workers
**Then** workers claim batches concurrently without duplication, writing the standard shard layout.

**Acceptance criteria:**

- Cloud workers and local runs use identical batch-processing logic and output identical shard layouts.
- Workers coordinate via a transactional queue lock so a batch is never double-processed, and workers can scale dynamically.
- Distributed run shards can be synced locally and consumed unchanged by local per-stage commands.
- A stage can execute in-place over object storage using the exact same contract as local file execution.
- The pipeline natively resolves standard local and cloud storage URIs.
- Object-storage stage writes are all-or-none per stage: every declared output must exist before any is published; a set left partial by a failed upload never reads as complete, so the next run redoes that base and overwrites it; and a stage that rewrites outputs in place commits atomically, so an interrupted publish leaves the previous outputs intact.

**TO-DO:**

- **Azure storage URIs aren't credential-wired.** `storage_options_from_env` (`store/remote.py`) wires only `s3://`; `az://`/`abfs://` fall through to adlfs unauthed and untested. Add an `az`/`abfs` branch reading `AZURE_STORAGE_*`, plus coverage.
- **No Azure ML connection in the enrichment paths.** `utils/isaacus_client.py` handles the hosted API and AWS SageMaker (`ISAACUS_SAGEMAKER_ENDPOINTS`) but has no Azure ML equivalent. Add an Azure ML connection option so those paths can reach an Azure ML endpoint (`AZUREML_ENDPOINTS`, client factory, `unserved_models` pre-flight).

## 3. Ingest-First Data Flow and Operation Composition

**As** a user,
**I want** a clear two-phase execution model (ingest first, then optional composed operations),
**so that** workflows are predictable and invalid stage configurations fail early.

**Given** raw input files and a chosen set of downstream operations
**When** the pipeline is executed
**Then** ingest runs first to produce base extraction results, followed by caller-composed downstream operations in any order the stage-dependency DAG permits.

**Acceptance criteria:**

- Ingest is format-dependent and strictly precedes any transform operation.
- Operations are independent functions that callers compose directly based on business need.
- Stage ordering is a **partial order** (a dependency DAG), not a single fixed sequence: a stage is a valid next step whenever the inputs it requires already exist. The default dispatch order is one valid topological sort of that DAG, not the sole valid execution order.
- Each operation enforces its preconditions as required-input edges.
- A config-disabled stage acts as a passthrough without raising an error.
- A failed stage preflight (a missing reference register, an unresolvable enrichment deployment, a failed model check) refuses before any document is processed.
- Invalid compositions (a stage run before the inputs it requires exist, or with a config-selected strict input missing) fail early, with a message naming the producing stage.
- Register ingests and text-only extraction (`extract` → `.txt`) are terminal — they have no valid downstream text stages.

**TO-DO:**

- **A missing required input is not a fail-fast, before-processing error (re: "Invalid compositions … naming the producing stage").** `stage_runner.run_stage_remote` (`cloud/stage_runner.py`) resolves required inputs *per base* and raises `NotReady`, which is caught and logged as a warning; the run still exits `0` unless the count of not-ready bases equals the total discovered bases (`StageRunSummary.exit_code`). This is deliberate — a still-draining fleet must not read as a stage-ordering error — but it means a genuinely mis-ordered composition over a *partially* processed run neither raises immediately nor fails the run, and the operator only sees a per-base warning. Decide whether to (a) add an explicit up-front composition check (verify the whole DAG's required-input edges against what the store already holds before processing any base, distinguishing "upstream still draining" from "upstream will never run" via the dispatched-stage set), or (b) document the per-base `NotReady`/warning behaviour as the intended contract and soften the acceptance criterion accordingly. **Partly resolved on the queue side:** a *stage job* whose every base is blocked now raises `StageNotReady` (`cloud/worker.py`), which the worker loop **releases** instead of failing — the attempt is not consumed, so a stage claimed ahead of its upstream no longer burns its retry budget and lands terminally failed. That fixes the queue semantics; the up-front whole-DAG composition check above is still open.
- **A missing strict overlay does not fail early, and locally does not fail at all (re: "invalid stage configurations fail early").** `_resolve_inputs` (`cloud/stage_runner.py`) raises `InputContractError` per base inside the `run_stage_remote` loop, so bases processed before the gap is reached still publish and the remaining bases continue; `run_stage_local` and the per-stage `*_shards()` commands call `load_overlay` without `required=True`, so a declared `processing.text_source` overlay that is missing yields a warning and a sidecar built from verbatim text with a zero exit. Decide whether to (a) check strict conditional inputs across every base before processing any, and pass `required=True` on the local path, or (b) document the per-base remote refusal and the local verbatim fallback as the intended contract.

## 4. File Profiling, Detection, and Routing

**As** a user,
**I want** the system to profile each file and route it to the correct extractor,
**so that** inappropriate extraction methods (like OCR on native spreadsheets) are avoided.

**Given** a batch containing mixed file formats (PDFs, Word documents, spreadsheets, images)
**When** ingest runs
**Then** each file is profiled to determine its true type and signals, then automatically routed to the appropriate extraction engine.

**Acceptance criteria:**

- The system generates a document-level profile capturing format, per-page visual signals, and OCR confidence.
- PDFs and images are explicitly routed to the page-level orchestrator.
- Path-based formats (DOCX, spreadsheets, text, Markdown) are explicitly routed to their respective format extractors.
- Visual signals like handwriting, ruled lines, and layout regularity inform the extraction profile.
- Spreadsheets are accurately classified per sheet by reading a sample of leading rows.

## 5. PDF and Image Extraction with OCR Quality Controls

**As** a user,
**I want** reliable text extraction from native, scanned, hybrid, and redacted visual documents,
**so that** the corpus accurately reflects content despite poor scan quality.

**Given** a PDF or image input
**When** extraction runs through the page-level orchestrator
**Then** native text is extracted directly, scanned regions undergo OCR with preprocessing, and low-confidence results are flagged.

**Acceptance criteria:**

- Native document pages extract selectable text and logical structure directly.
- Scanned pages undergo layout analysis, deskewing, and dynamic binarisation prior to OCR.
- OCR processes producing an average confidence score below an acceptable threshold (e.g., 40%) raise a warning.
- The extraction outputs distinct elements (paragraphs, headings, tables, forms, images) with logical reading order preserved.

## 6. Native Office and Spreadsheet Extraction

**As** a user,
**I want** native office documents and spreadsheets extracted with their structure and cell granularity preserved,
**so that** logical paragraphs, embedded tables, and tabular records are usable for semantic analysis.

**Given** a DOCX, TXT, CSV, or Excel input
**When** extraction runs
**Then** text, body order, and cell-grained tabular grids are extracted, correctly detecting table headers and structural boundaries.

**Acceptance criteria:**

- DOCX inputs are walked in body order, extracting paragraphs and embedded tables seamlessly interleaved. A Word table's header band is read from the authored `w:tblHeader` ("repeat as header row") property: the leading contiguous run of rows carrying it becomes the element's `header_rows`, so a genuine two-row hierarchical header is emitted as more than one index rather than collapsed. A table with no declared header row falls back to a single header at row 0.
- Spreadsheets produce a cell-grained element stream with a dedicated meta-element per sheet.
- Spreadsheet headers are programmatically detected based on layout, placing the real header (a single header row, at row 0) at the top of the grid and preamble in metadata.
- Cell elements capture value, value type, formula, and number format. A merged region's extent is captured on its `merge_range` field (the openpyxl address, e.g. `"A1:C1"`), set on the merge's top-left (anchor) cell; the cells the merge covers are emitted too — blank, since Excel leaves a merged region's covered cells empty — so the merge's footprint survives in the grid.
- Text boundaries are preserved natively without the need for visual OCR orchestration.

**TO-DO:**

- **Multi-row / hierarchical headers are emitted for DOCX only; spreadsheets and the legacy table view still collapse to row 0.** `DocxExtractor` (`strategies_file.py`) now populates `header_rows` with the full leading run of `w:tblHeader`-declared rows, so a multi-row Word table header is preserved. Two producers remain single-row: spreadsheet headers are hard-coded to row 0 by the producer (`ingest/spreadsheet.py`: `split_preamble` picks one header row and `_emit_sheet` keeps it at row 0; `_sheet_rows` in `process/money_stage.py` is a consumer that inherits it), so a genuine multi-row header is collapsed by pandas into one row and the extra header rows become data rows; and `table_to_element` (`views.py`) reconstructs a legacy `TableData`, whose flat `headers` list cannot carry more than one header row. No **cell-level** header-row coordinate is carried yet — `header_rows` lives on the table element, not on each `Cell`. Decide whether to (a) detect and emit multi-row headers from the spreadsheet layout too and add a per-cell header-row coordinate (a `Cell`/sidecar schema change), or (b) document single-row-header as the intended contract for the spreadsheet and legacy-view producers.
- **`formula` is never captured.** The element schema carries `formula`, but no spreadsheet producer sets it, so it is always null. The XLSX openpyxl pass in `ingest/spreadsheet.py` already loads formulas (no `data_only`), so it can carry them alongside pandas' cached values.

## 7. Standalone Reference Register Ingestion

**As** a user,
**I want** standalone ingestion for reference registers,
**so that** structured relational and spatial data bypasses NLP stages and is immediately queryable.

**Given** structured data files (e.g., G-NAF, ABN bulk-extract, Shapefiles)
**When** the operator runs the specific register-ingest commands
**Then** the registers are converted directly to standard or geospatial Parquet with complete provenance.

**Acceptance criteria:**

- Outputs are strictly schema-typed Parquet (or GeoParquet) files representing the structured data.
- Large XML streams are parsed in constant memory to prevent out-of-memory errors on bulk extracts.
- Spatial files preserve their original geometry, attributes, and coordinate reference systems.
- File-level malformations isolate failures, logging the error and discarding partial output without halting the directory ingest.
- Register ingestion bypasses the extraction/NLP pipeline: its output carries no extraction sidecars, so text-based downstream stages never discover it. The bypass is structural; no guard rejects a downstream stage pointed at a register directory.

**TO-DO:**

- **G-NAF and geospatial writes are not guarded against partial output (re: "discarding partial output without halting the directory ingest").** `gnaf.ingest_psv` calls `pq.write_table` unguarded, so a write failure propagates out of `ingest_gnaf_directory` and halts the directory ingest, possibly leaving a partial file. `geospatial.ingest_shapefile` catches a `to_parquet` failure and records it on the result but does not remove the partial file, and its follow-up provenance rewrite (`pq.read_table` / `pq.write_table`) is unguarded. Decide whether to (a) guard both writes and unlink partial output on failure, as `abn_bulk` does, or (b) narrow the criterion to read-failure isolation for these two ingests.

## 8. Output Data Contract & Persistence Integrity

**As** a user,
**I want** a consistent, immutable output contract with a universal join key and automated integrity checks,
**so that** disparate outputs from any stage can be joined predictably and trusted.

**Given** a completed extraction batch or downstream operation
**When** results are written to the filesystem or store
**Then** outputs adhere to standard Parquet layouts, preserve original text verbatim, join on a universal key, and pass integrity verification.

**Acceptance criteria:**

- Multi-unit inputs and all downstream stages output standard Parquet files universally keyed by `source_hash`.
- A strict verbatim text policy applies: downstream operations (redaction, PII, etc.) write separate sidecar overlays and never rewrite the base extracted text.
- Standard extraction yields a parent elements file, with complex child rows (table cells, form fields) joining via `source_hash` and element order.
- A unified run-level manifest correctly consolidates provenance, statuses, and counts for all source files.
- A persistence verifier ensures all required shard files exist, are readable, match expected document counts, and prevent accidental overwrites.
- `source_hash` is content-addressed (SHA-256 of the source bytes) and stable across environments and runs; logical row order within a shard is stable given the same inputs and iteration order.
- Output is **not** byte-for-byte reproducible across runs; content is. For a given `source_hash`, version, configuration and model set, extraction content and row order are stable, and the manifest's per-document `content_digest` lets a later run confirm it. A mismatch is explained by the stamped version, configuration and model digests and never blocks output.
- Every pipeline Parquet and `egress_manifest.json` declare a contract version, versioned apart from the package, and every pipeline Parquet declares its sensitivity (`raw`, `masked` or `none`). Additive schema changes are back-filled for older files; renames and removals ship a reader shim.
- Only files whose footer says `masked` or `none` are safe to hand onward; anything `raw`, or with no sensitivity key, stays inside the trust boundary.
- `womblex.__all__` declares the stable Python API, pinned by a test; `import womblex` does not load the extraction stack. A name in it is removed or changed incompatibly only after one minor release emitting a `DeprecationWarning` that names the replacement.

## 9. Redaction Handling

**As** a user,
**I want** redacted regions detected visually and handled according to a configurable policy,
**so that** sensitive visual blocks are managed consistently.

**Given** an extracted PDF document
**When** the operator runs the redaction stage
**Then** redactions are detected per page, applied based on the chosen mode, and (via the standalone shard CLI) written as an independent sidecar.

**Acceptance criteria:**

- Solid redaction regions are detected per page, whether drawn as native filled shapes or present only in the page image.
- Multiple modes are supported: `flag` (annotation only, no text change), `blackout` (prepend a `<REDACTED>` marker to affected page text), and `delete` (clear affected page text). The `<REDACTED>` marker is a content marker, distinct from the human-readable warning strings described below.
- A redaction report is attached to the extraction result, and per-page warning strings (e.g. `page N: K redacted region(s) detected`) are appended to the extraction's warnings.
- Detected redactions are written as an independent Parquet sidecar (`*.redactions.parquet`).
- Redaction detection runs at a single fixed point, immediately after extraction.

**TO-DO:**

- **Redaction has no configurable pipeline point, contrary to earlier documentation.** `RedactionConfig` (`config/__init__.py`) carries no `pipeline_point`/timing field, and `batch.py` runs redaction detection at one fixed position (immediately after extraction, before any downstream stage). The claim that redaction runs "at configurable pipeline points (post_chunk, post_enrichment)" in `CLAUDE.md` and older docs appears to be conflated with `PIIConfig.pipeline_point` — that configurability exists for the **PII** stage, not redaction. Decide whether to (a) add a genuine `RedactionConfig.pipeline_point` (post_extraction / post_chunk / post_enrichment) mirroring the PII stage if before/after-chunking placement is actually wanted, or (b) treat the fixed post-extraction position as the intended contract and correct the stale `CLAUDE.md` wording accordingly.
- **The independent redactions sidecar is only written on the standalone shard CLI, not the E2E `run` path.** `redact/batch.py::annotate_redactions_for_shards` writes `*.redactions.parquet`, but `operations/redact.py::run_redaction` (the path `batch.py` invokes during `womblex run`) persists nothing standalone — redaction survives only as in-memory annotations on the extraction result. Decide whether to (a) have the E2E path also emit `*.redactions.parquet` so the "independent sidecar" contract holds uniformly, or (b) document the two paths' divergence (E2E = in-line annotation; shard CLI = independent sidecar) as intended and keep the acceptance criterion qualified as above.

## 10. Chunking and AI Chunking Reuse

**As** a user,
**I want** extracted text split into semantically meaningful, token-bounded chunks,
**so that** downstream semantic analysis fits within model context limits.

**Given** an existing extraction result
**When** the operator runs the chunking stage
**Then** narrative text and tables are intelligently split within a token budget and saved as a chunks sidecar.

**Acceptance criteria:**

- Narrative text and markdown-converted tables are chunked independently with specific tags.
- Token counting uses a local tokenizer to ensure no network dependency.
- Partial redaction markers that split across boundaries are automatically repaired in the chunk overlay.
- Optional AI chunking uses semantic boundaries, leveraging previously persisted enrichment data if it matches the source text.
- If AI chunking detects a mismatch with previously persisted enrichment, it falls back to self-enrichment rather than using mismatched data.

## 11. PII Detection and Masking

**As** a user,
**I want** PII detected, masked, and retained reversibly for authorised audits,
**so that** sensitive data is protected for publication without destroying internal provenance.

**Given** a chunked shard directory, normally already enriched
**When** the operator runs the PII stage
**Then** sensitive spans are replaced with typed tags, generating an auditable sidecar and a clean-text layer.

**Acceptance criteria:**

- On the per-stage path (`womblex pii`), candidates are the enrichment graph's person and address entities, mapped onto narrative chunks. The regex/context backstop runs only when `pii.use_regex_backstop` is set (default `false`); with no enrichment and the backstop off, nothing is detected.
- The stage is terminal: it runs after enrich and embed, never rewrites `*.chunks.parquet`, and writes the masked layer as a separate `*.clean_text.parquet` (when `pii.write_clean_text`, the default).
- The in-memory PII operation runs at a configurable point (`pii.pipeline_point`): before enrichment it uses the regex/context detector, and after enrichment it merges graph spans with it.
- Identified PII spans in the clean-text layer are replaced with normalised typed tags (e.g., `<PERSON_1>`).
- The spans are written to an independent Parquet sidecar (`*.pii_spans.parquet`) that retains each span's original text plus its chunk offsets and graph `entity_id` — the audit record from which an authorised reversal can be reconstructed against the clean-text layer. The sidecar is a reversal-enabling audit layer; no automatic un-masking operation is implemented.

## 12. Knowledge Graph and External Enrichment

**As** a user,
**I want** entities and relationships extracted via external APIs and synchronised into a unified document graph,
**so that** structured mentions accurately map back to their source chunks regardless of run order.

**Given** a chunked extraction result and configured external enrichment credentials
**When** the operator runs enrichment and graph generation
**Then** entities are extracted, retrying on API limits, and a graph is built linking entities to specific chunks.

**Acceptance criteria:**

- Enrichment securely calls external services (retrying on rate limits) to produce entities and relationships.
- The graph generation builds nodes and edges, producing an explicit mention-to-chunk link layer.
- Graph refresh operations explicitly rewrite mention-to-chunk links in place if chunking is executed or modified after initial enrichment.
- Graph generation is idempotent and never skips its refresh based solely on file existence.

## 13. Money Annotation

**As** a user,
**I want** monetary amounts identified in narrative text and tabular cells,
**so that** values can be queried and joined to specific document contexts downstream.

**Given** standard extraction shards
**When** the operator runs the money annotation stage
**Then** monetary spans and columns are identified and written as a sidecar without modifying element text.

**Acceptance criteria:**

- The annotation scans baseline element and table-cell streams (chunking is not a precondition).
- Narrative spans index the same coordinate space as enrichment mentions, allowing seamless joins at query time.
- Output is order-independent within a run: the money sidecars carry identical rows whether the stage runs before or after the knowledge-graph stages.
- Cell annotations carry their sheet, parent element order, row, and column coordinates. Where an amount sits in a merged region, the merge extent lives on the source cell's `merge_range` (Requirement 6), not on the money span row, and the column's evidencing header is recorded only as joined text on the `money_columns` sidecar. A header-row coordinate is still not carried.

**TO-DO:**

- **Cell annotations lack a merged-cell extent and header-row coordinate on the span row (re: "cell annotations carry their … coordinates").** `_cell_row` (`process/money_stage.py`) emits `(row, col)` / `(parent_elem_order, row, col)` / `(sheet, row, col, elem_order)` and can now recover a cell's merge extent by joining back to the source element's `merge_range` (populated per Requirement 6), but it does not copy that extent — nor a header-row index — onto the money span row itself. Multi-row *table* headers are now emitted for DOCX (`w:tblHeader`) but spreadsheets and the legacy table view still collapse to `header_rows=[0]` (the remaining Requirement 6 gap), and no per-cell header-row coordinate exists yet. Decide whether to (a) copy the source cell's `merge_range` onto the span/cell rows and add a header-row column once multi-row headers are emitted, or (b) document the join-back-to-`merge_range` path as the intended contract and keep the span row free of the extent.

## 14. Batch Processing, Resiliency, and Operational Controls

**As** a user,
**I want** reliable batching, checkpointing, and discrete execution controls (CLI/UI),
**so that** I can manage large runs, recover from failures, and inspect outputs easily.

**Given** a configured run (local or distributed)
**When** the operator uses the CLI or Web UI
**Then** documents process in isolated, resumable batches, with independent stage controls and visual inspection capabilities.

**Acceptance criteria:**

- Processing occurs in configurable batches, appending results and writing checkpoints upon batch completion.
- Resuming an interrupted run automatically reconciles checkpoints and skips already-completed documents.
- Individual document errors are isolated, recorded in the manifest, and do not crash the wider batch.
- CLI commands allow stages to be dispatched independently, with idempotent queueing ensuring dependent stages sequence properly.
- Logs can be emitted as JSON lines (`womblex --log-format json` or `WOMBLEX_LOG_FORMAT=json`) with no added dependency; worker and batch records carry `run_id`, `job_id`, `stage`, and `source_hash` where known, and the default text format is unchanged.

## 15. Web Console Shell, Navigation, and Deployment Modes

**As** a developer or administrator,
**I want** an optional admin and debugging console that reads the artefacts the pipeline already writes,
**so that** I can inspect and debug every run domain in one place without a separate tool or a live pipeline connection.

**Given** a `womblex[ui]` install bound to a single run source (a local output root or an object-store URI)
**When** the operator launches the console and opens it in a browser
**Then** a persistent shell routes between the five console domains and serves the read API even when no frontend build is present.

**Acceptance criteria:**

- The console is an admin and debugging utility, not an integration surface: other software submits and reads work through the authenticated, owner-scoped `womblex serve` `/v1` API or reads the on-disk data contract. The console has no authentication and binds to loopback unless told otherwise.
- The console binds to exactly one run source at construction (local output root or store URI) so no endpoint can be steered to read an unmounted directory.
- A persistent top bar (global search, run selector, execution controls) and a side-nav rail route between the Dashboard, Corpus Inspector, Semantic Chunk Inspector, Pipeline Composer, and Resources Console.
- The console is a reader over persisted artefacts and never edits a stage output; its only writable surfaces are dispatch and preset saving.
- Dispatch requires a store, an ingest location, and a job queue to be configured; a console missing any of these still serves the full inspection surface (it simply cannot enqueue work).
- A bare install with no SvelteKit build still serves the read API; the SPA is mounted only when a build exists alongside it.

## 16. Dashboard — Queue and Stage Progress

**As** an administrator,
**I want** a run-scoped dashboard of queue state and per-stage progress,
**so that** I can monitor throughput and spot stalled jobs while operating or debugging a run, without touching the queue.

**Given** a selected run, with an optional job queue and the run's own per-stage checkpoints
**When** the operator opens the Dashboard
**Then** it presents queue counts and per-stage completion, polling while the tab is visible and pausing while it is hidden.

**Acceptance criteria:**

- Queue and stage state are read from sources the pipeline already writes (the job queue and per-stage checkpoints); with no queue configured, the dashboard falls back to checkpoints.
- Job status is rendered with the exact values the system writes (`pending` / `running` / `done` / `failed`), plus `stale` for a running row past its lock timeout and `skipped`.
- The dashboard only *names* a stalled job for a worker to recover; it never requeues, cancels, or claims work itself.
- Polling is paused while the browser tab is hidden so a backgrounded console stops hitting the queue.
- Per-stage progress renders in the declared pipeline order rather than an ad-hoc frontend ordering.
- Batch and stage logs for the run are listed newest-first and are individually viewable.

## 17. Corpus Inspector — Document Grid and Integrity Audit

**As** a developer or administrator,
**I want** a dense, virtualised document grid with checkpoint and shard-integrity views,
**so that** I can inspect thousands of documents and confirm a run's outputs are complete and readable when debugging it.

**Given** a selected run's manifest and shard directories
**When** the operator opens the Corpus Inspector
**Then** each document appears as a row with its status, a stage-checkpoint switcher, and an on-demand shard-integrity audit.

**Acceptance criteria:**

- The grid uses real table semantics with a sticky header, row virtualisation, and an announced total row count for accessibility.
- Document status is conveyed by a status pill (icon plus label), never by row background tint alone.
- A checkpoint switcher reports which stages are present for the run, in the declared pipeline order.
- A verify-shards action runs the persistence audit, confirming required shard files exist, are readable, and match expected document counts.
- The grid supports a failed-only filter and a user-selectable density (comfortable / default / compact) persisted locally.

**TO-DO:**

- **Row virtualisation and the announced total row count are not implemented (re: "row virtualisation, and an announced total row count").** `DocumentGrid.svelte` renders a real `<table>` with a `<caption>`, `<th scope="col">` header, and a sticky header, but puts *every* row in the DOM rather than virtualising a window over them — an explicit code comment records that `aria-rowcount`/`aria-rowindex` are omitted because "every row is in the DOM here, so the browser's own count is correct … They become necessary when virtualisation lands." As built, the grid will not stay dense over thousands of rows (the story's own motivation), and there is no `aria-rowcount` announcing a total beyond the rendered window. Decide whether to (a) add windowed row virtualisation and the `aria-rowcount`/`aria-rowindex` announcement it requires, or (b) drop "virtualised" from the story and "row virtualisation, and an announced total row count" from this criterion until it is implemented. Resolve alongside the matching Requirement 21 TO-DO, which restates the same gap as an accessibility rule.

## 18. Semantic Chunk Inspector — Chunk, Entity, PII, and Money Overlays

**As** a developer,
**I want** to read a document's chunks with entity, PII, and money overlays rendered inline,
**so that** I can debug chunking quality and confirm sensitive spans are masked correctly.

**Given** a document chosen from the run's manifest
**When** the operator opens the Semantic Chunk Inspector for that document
**Then** its chunks and sidecar overlays are read per-`source_hash` and rendered as one card each.

**Acceptance criteria:**

- Chunk detail is read keyed on a single `source_hash`, pushing the predicate into the Parquet read so per-document inspection stays cheap over a corpus-wide sidecar.
- Each chunk card shows chunk index, token count, character range, and content type in a monospace, sunken well.
- Entity mentions are underlined with a hover tooltip, and PII masks (e.g., `<PERSON_n>`) are rendered as inline pills.
- Money spans surfaced by the money-annotation sidecar are rendered inline alongside entity and PII overlays.
- Overlays are read-only projections of the sidecars and never rewrite the base extracted text.

## 19. Pipeline Composer — Configuration, Validation, and Dispatch

**As** a developer or administrator,
**I want** to compose and validate a pipeline configuration visually and dispatch an admin run from it,
**so that** I can author correct configs and enqueue work without hand-editing YAML or re-implementing guardrails.

**Given** the served stage graph and the `WomblexConfig` JSON Schema
**When** the operator edits the config form and presses to enqueue or run downstream stages
**Then** the config is validated and dispatched through the same code paths the CLI uses, writing queue rows byte-identical to the documented commands.

**Acceptance criteria:**

- The stage graph is rendered from the served `STAGE_CONTRACTS` (nodes wired by their required-input edges); disabled stages drop to reduced opacity while keeping their edges so a broken chain reads as a gap.
- The form is rendered from `WomblexConfig`'s JSON Schema, and validation plus YAML download go through the same `WomblexConfig` construction the CLI's config loader uses — the console cannot accept a config the CLI would reject.
- Named presets are offered as starting points; operator-authored presets can be saved when a presets directory is configured, and preset saving refuses cleanly when it is not.
- Enqueuing an extraction run and dispatching downstream stages call the same enabled-stage gate and queue-enqueue paths as the equivalent CLI commands, and dispatch is idempotent per `(run_id, stage)`.
- Which downstream stages run is decided server-side (never re-derived in the frontend); the result panel reports what was dispatched in claim order, and irreversible or run-scoped stages (PII, quality) remain undispatchable.
- Runs dispatched from the console carry no owner, so through the service API they are visible to `admin` callers only.

## 20. Resources Console — Connections

**As** an administrator,
**I want** connection cards for the store, ingest, queue, and enrichment service with reachability tests,
**so that** I can confirm a deployment is wired correctly without exposing secrets.

**Given** the deployment's configured store, ingest, queue, and Isaacus connections
**When** the operator opens the Resources Console
**Then** each connection is shown as a credential-masked card with a live "Test" action.

**Acceptance criteria:**

- Four connection cards (store, ingest, queue, Isaacus) render deployment configuration, not any single run's artefacts.
- Connection strings are shown with secrets masked (never rendered in full in the DOM or a copy buffer), and each card offers a live reachability test.
- The store and ingest locations are deploy-time configuration (flags / env) and are shown read-only, as are the queue and Isaacus cards; an empty worker fleet is a normal resting state rather than an error.

## 21. Console Design System and Accessibility

**As** a developer or administrator,
**I want** the console to default to a dense, dark, state-legible design that runs with no network access and meets accessibility standards,
**so that** I can read large grids and chunk text reliably in both themes and at every density.

**Given** the console rendered in a browser, possibly air-gapped
**When** the operator uses it across themes, densities, and input methods
**Then** colour carries state (not decoration), assets are self-hosted, and every interactive element is keyboard-reachable with visible focus.

**Acceptance criteria:**

- The console defaults to a dark theme with a supported light theme, using semantic design tokens only (never hardcoded colour values).
- Colour communicates pipeline/queue state through a measured status palette, and status is never encoded by colour alone — every status carries an icon and a text label.
- Fonts and icons are self-hosted in the UI bundle with no runtime network requests, consistent with the pipeline's local-first model.
- Grids are real tables with `<th scope>` and `<caption>`, and virtualised rows announce total counts to assistive technology.
- The keyboard reaches everything the mouse does (grid arrow-key navigation, `/` to focus search, `Esc` to close drawers) with a visible focus indicator, and interactive controls keep a ≥ 44×44px hit area even at 32px density.

**TO-DO:**

- **Virtualised rows do not announce total counts (re: "virtualised rows announce total counts to assistive technology").** The `<table>`/`<th scope>`/`<caption>` half of this criterion holds (`DocumentGrid.svelte`), but the grid is not virtualised and emits no `aria-rowcount`, so there is no announced total beyond the rendered rows. This is the same gap as the Requirement 17 TO-DO, stated here as its accessibility consequence: until windowed virtualisation lands with `aria-rowcount`/`aria-rowindex`, assistive technology hears only the DOM row count. Resolve together — either implement virtualisation with the row-count announcement, or drop the "virtualised rows announce total counts" clause from both requirements.

## 22. Service API — Submission and Retrieval

**As** a developer integrating another system,
**I want** an authenticated HTTP API to upload documents, submit runs, and read their results,
**so that** my software can use Womblex as a backend service without shell access to the pipeline host.

**Given** `womblex serve` bound to a store, a job queue, and an ingest location, with a client registry
**When** a caller presents a service token and uploads documents, submits a run, and polls it
**Then** the run is queued under the caller's ownership and its status, files, and text are readable only by that caller (or an admin), gated by scope and file sensitivity.

**Acceptance criteria:**

- Every `/v1` endpoint except `health` and `ready` requires a bearer token whose SHA-256 is in the registry; a missing or unknown token is 401, and tokens are compared in constant time.
- Scopes are `submit`, `read`, `read_raw`, and `admin` (which implies the rest); a request outside the caller's scopes is 403.
- The service refuses to start with an empty registry unless explicitly run with `--insecure-no-auth`, and binds to loopback by default.
- An upload is written to `<ingest>/<client_id>/<upload_id>/` keeping only bare file names; an unusable, repeated, or unsupported name is 400 with nothing written, an over-limit request is 413, and the response names the folder to submit as `input_prefix`.
- A non-admin caller can submit runs only over its own `<client_id>/` tree (403 otherwise); the run id is minted server-side and unique across concurrent submissions.
- A submitted `preset` or `config` is validated before any queue row is written (400 on failure) and selects downstream stages only; a config enabling none is an extraction-only run.
- A run belongs to the client that submitted it: another client's run is 404, an admin sees all runs, and runs dispatched from the CLI or console carry no owner and are visible to admins only. A run never has mixed ownership.
- Document text defaults to the masked layer; raw layers (`chunks`, `elements`) require `read_raw`, decided by the layer's contract sensitivity.
- Run file listings report each file's row count, `contract_version`, and `sensitivity`, so a caller can fetch Parquet directly under the data contract.
- `ready` reports store and queue reachability and answers 503 when either is down; an unreachable queue, store, or ingest location on any other endpoint is 503, not 500.
- The OpenAPI surface is pinned by a test so breaking changes are visible in review.

## 23. Egress Bundle — Export to Downstream Consumers

**As** an operator handing a run to another system,
**I want** one finished run exported with its raw source documents as a self-describing bundle,
**so that** consumers can review and reuse the extracted corpus alongside its sources without calling back into Womblex.

**Given** a finished local run root (holding `documents/` and `manifest.parquet`)
**When** the operator runs `womblex egress <run> --to <dest>`
**Then** a bundle is written under `<dest>/<bundle_prefix>/` holding the unchanged corpus, deduplicated raw sources, a source index, and a descriptor, and Womblex stops.

**Acceptance criteria:**

- The destination may be any `RemoteStore` URI (local directory, S3, MinIO, GCS) through one code path.
- `corpus/` mirrors the run's shard directory recursively, including every downstream sidecar, plus the consolidated `manifest.parquet`, unchanged.
- Raw sources are written to `sources/<source_hash><ext>`, deduplicated by hash; two documents sharing a hash point at one file.
- `source_index.parquet` has one row per document with `source_hash`, `doc_id`, `filename`, `ext`, `raw_key`, and a status of `resolved`, `hash_mismatch`, `not_found`, `unsupported_basis`, or `upload_failed`; `raw_key` is null wherever no file was produced.
- Source resolution is non-fatal per document: a missing, mismatched, records-ingested, or failed-upload source is reported in the index and in `egress_manifest.json`, never aborting the export.
- `egress_manifest.json` carries the contract version and a per-document resolution report.
- A corpus-only export (`--no-sources` / `--corpus-only`) skips resolution and writes no `sources/` or `source_index.parquet`.
- `--source-root` resolves sources from a moved corpus; `--run-id` defaults to the run root's directory name and must agree with the run's own stamp, if it has one.
- Re-exporting into an existing bundle removes source files the fresh export neither wrote nor attempted, and leaves a source whose upload failed untouched.
- A run ingested from an object store is refused before anything is written.
- Womblex writes the bundle and nothing downstream of it: no retention, serving, or write-back of consumer corrections.

## 24. Swappable Models Without a Schema Change

**As** a user with my own models, local or API-backed,
**I want** to swap the model behind a pipeline slot without forking Womblex,
**so that** I can use models suited to my corpus while a packaged CPU-friendly group stays the default.

**Given** an installed package that registers a model for a slot
**When** the operator names that model in the config
**Then** the run uses it in place of the default, with no change to any Parquet schema.

**Acceptance criteria:**

- Swappable slots: OCR engine, layout analyser, PII context model, chunk tokeniser, and spellfix dictionary.
- Installing a third-party model package makes its models selectable by name, with no change to Womblex's own dependencies.
- The built-in models (PaddleOCR, Mistral, Ollama, PP-DocLayout-M) keep their existing names and aliases.
- An installed package can supply model files, and they resolve offline the same way bundled models do.
- A config names registered models only; an unknown name is an error that lists the known names. Import paths are refused.
- An engine's own options pass through to it unchanged, and Womblex adds no per-library toggles.
- An OCR model may return either regions or page markdown, and the pipeline handles each the same way it handles the built-in engines of that kind.
- A layout model emits the Womblex `block_type` vocabulary, checked by a conformance test.
- The layout model applies to redaction detection as well as extraction.
- Changing the PII context model is supported. The docs state that `context_similarity_threshold` must be recalibrated when it changes.
- With no plugin configured, `content_digest` is unchanged on every synthetic fixture.
- The model-plugin guide documents each slot's interface and how a package registers a model, with a minimal example.

**Out of scope:** model provenance (26), `/v1` submission controls (a submitted config only selects stages; workers run under the operator's config), container-image packaging of plugins, and slots that need a Parquet schema change, each needing its own requirement (25 and 26, and the rest): the enrichment provider, AI chunking with a non-Isaacus model, graph-driven PII detection, link-stage candidates, `graph_refresh`, the enrichment token-budget tokeniser, per-page or per-element OCR engine recording, and the embedder provider (including any CPU embedding baseline).

## 25. Pre-Run Model Check

**As** an operator,
**I want** configured models checked before any document is processed,
**so that** a run never fails part-way, or mixes models, because a model was missing.

**Given** a config that names models for one or more slots
**When** a run, worker, or API submission starts
**Then** each named model is checked, and a failure stops the run before its first document with the reason.

**Acceptance criteria:**

- The check has three levels, set by `processing.models_check` or `--models-check`: `off`; `load` (resolve and load the models, the default); `smoke` (also run one inference on a small built-in input).
- `womblex run` checks the models of every enabled stage before batch one, before any output is written, so a run whose later stages cannot complete does not start.
- A worker checks at startup and refuses jobs whose models fail the check, through the existing refused path. A job is refused only for the models it needs: a batch for the extraction models, a stage job for that stage's.
- A model behind a service is checked with one minimal request at either level: each Isaacus model an enabled stage calls (enrich, embed, AI chunking), and an API-backed OCR engine (Bedrock, Ollama), so a run whose later paid stage cannot reach its service does not start.
- An API submission checks only that each named model is registered, because the API host may not carry the models.
- Downstream stage preflight includes the model check, including the standalone `chunk`, `pii`, `spellfix` and `redact` commands, which check before writing anything.
- The existing in-slot fallback (PaddleOCR v5 to the wheel's v4) stays, and the check reports which variant resolved.
- The check result is written into the run record.

## 26. Model Provenance

**As** a consumer of a run's output,
**I want** to know which models produced it,
**so that** I can reproduce, compare, or audit a run.

**Given** a run that used default or plugin models
**When** I read the Parquet footers or the run record
**Then** each slot's model is named with the distribution and version that supplied it.

**Acceptance criteria:**

- Every pipeline Parquet's footer records, per slot, the model name and the distribution and version that supplied it. No column is added.
- Plugin model weights appear in the loaded-model record alongside the built-ins.
- The run record no longer lists the OCR engine as unestablished.
- A model the pre-run check builds only to confirm it loads is not recorded as used — a run whose check built the model but whose stages never actually called it shows no entry for it; only a run's own build of it does.
- Benchmark reports (womblex-benchmark) name the models that produced them, suites can run against a named model, and `docs/accuracy/` stays labelled as the default group.

## Outstanding

Requirements not yet fully met. The detail lives in the **TO-DO** block under each requirement; this list is the index. When a gap is closed (or the criterion is deliberately narrowed), its TO-DO block and its entry here are removed in the same PR.


- **Requirement 2 (Scale-Out):** Azure storage URIs (`az://` / `abfs://`) are not credential-wired, and the enrichment paths have no Azure ML connection.
- **Requirement 3 (Ingest-First Data Flow):** no up-front whole-DAG composition check; a missing strict `text_source` overlay is refused per base remotely and falls back to verbatim locally with only a warning.
- **Requirement 6 (Native Office and Spreadsheet Extraction):** multi-row headers are emitted for DOCX only; spreadsheets and the legacy table view collapse to row 0, no per-cell header-row coordinate exists, and cell formulas are never captured.
- **Requirement 7 (Standalone Reference Register Ingestion):** G-NAF and geospatial writes are not guarded against partial output.
- **Requirement 9 (Redaction Handling):** no redaction `pipeline_point`, and the E2E `run` path writes no `*.redactions.parquet`.
- **Requirement 13 (Money Annotation):** the money span row carries no merge extent or header-row coordinate.
- **Requirements 17 and 21 (Corpus Inspector; Console Design System):** `DocumentGrid.svelte` is not virtualised and emits no `aria-rowcount`.

---
