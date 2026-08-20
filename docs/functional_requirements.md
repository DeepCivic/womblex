# Womblex — Functional Requirements

This document is the single source of truth for **what** Womblex must do, expressed
as user stories with testable acceptance criteria. It deliberately does **not**
cover *how* the system is built, why decisions were made, or measured accuracy —
those concerns live in the related documents listed below.

## Related documents

These requirements describe intended behaviour only. The following documents own
the adjacent concerns and should be consulted rather than duplicated here:

| Document | Owns |
|---|---|
| [`README.md`](../README.md) | Project overview, installation, and command usage. |
| [`docs/architecture.md`](architecture.md) | High-level system architecture and component responsibilities. |
| [`docs/composable-design.md`](composable-design.md) | The composable-operations design and stage-contract model. |
| [`docs/dataflow.md`](dataflow.md) | End-to-end data movement, from raw input to Parquet output. |
| [`docs/extraction.md`](extraction.md) | The extraction output schema (element streams and child rows). |
| [`docs/money-extraction.md`](money-extraction.md) | The canonical reference for the `money` annotation op. |
| [`docs/heuristics_disambiguation.md`](heuristics_disambiguation.md) | CV2/NumPy heuristics used for classification and routing. |
| [`docs/project-structure.md`](project-structure.md) | File-level map of the source tree. |
| [`docs/decisions.md`](decisions.md) | Design decisions, rejected alternatives, and known limitations. |
| [`docs/steering.md`](steering.md) | Current state and prioritisation of upcoming work. |
| [`docs/evaluation.md`](evaluation.md) | Evaluation metrics and candidate-technology mapping. |
| [`docs/accuracy/`](accuracy/) | Generated accuracy reports (extraction, chunking, PII, redaction). |
| [`DESIGN.md`](../DESIGN.md) | The console design system — tokens, components, and accessibility rules. |

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

**TO-DO:**

- **In-place object-storage writes are not atomic.** No atomic multipart upload; durability comes from all-or-none-per-stage publish + idempotent overwrite (`stage_runner._publish`, `MutationMode`, `RemoteStore.upload_file`). Either document idempotent-overwrite as the contract, or add true atomic writes (temp-key-then-copy / `If-None-Match`).
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
- Stage ordering is a **partial order** (a dependency DAG), not a single fixed sequence: a stage is a valid next step whenever the sidecars it declares as required inputs are already present. Several orderings are therefore valid — e.g. after `enrich`, `link` (which needs only the enrichment entities) is immediately valid, while `graph-refresh` (the mention→chunk edge rebuild, `build_graph` in `configs/default-isaacus.yaml`) additionally requires that `chunk` has already run; `chunk` → `enrich` is valid because `enrich` reads the extraction text, not the chunk output; independent sidecar ops (`money`, and `embed` once chunks exist) may be appended after any earlier stage that produced their required sidecars.
- Each operation enforces clear preconditions expressed as required-input edges (e.g., `chunk`/`money` require the extraction sidecars; `embed`/`pii` require chunks; `link` requires enrichment entities). The single linear `PIPELINE_ORDER` is a **presentation and default-dispatch order only** — it is one valid topological sort of the DAG, not the sole valid execution order.
- A config-disabled stage acts as a passthrough without raising an error.
- Invalid compositions (a stage run before the sidecars it requires exist) are surfaced with a message naming the producing stage. The surface differs by failure class: a *strict* conditional input that config selected but is absent (e.g. a `processing.text_source` overlay) and a failed pre-flight both refuse **before any base is processed** (`InputContractError` / `StagePreconditionError`). A missing *required* input is handled **per base** by the runner as a `NotReady` state — logged with the producing stage and skipped — and only escalates to a non-zero exit when *every* discovered base is blocked (the still-draining-fleet case is otherwise a warning, not a failure).
- Register ingests (`ingest_gnaf_directory` / `ingest_abn_xml` / `ingest_abn_directory` / the geo ingest) and text-only extraction (`extract` → `.txt`) are terminal — they have no valid downstream text stages (see Requirement 7).

**TO-DO:**

- **A missing required input is not a fail-fast, before-processing error (re: "Invalid compositions … naming the producing stage").** `stage_runner.run_stage_remote` (`cloud/stage_runner.py`) resolves required inputs *per base* and raises `NotReady`, which is caught and logged as a warning; the run still exits `0` unless the count of not-ready bases equals the total discovered bases (`StageRunSummary.exit_code`). This is deliberate — a still-draining fleet must not read as a stage-ordering error — but it means a genuinely mis-ordered composition over a *partially* processed run neither raises immediately nor fails the run, and the operator only sees a per-base warning. Decide whether to (a) add an explicit up-front composition check (verify the whole DAG's required-input edges against what the store already holds before processing any base, distinguishing "upstream still draining" from "upstream will never run" via the dispatched-stage set), or (b) document the per-base `NotReady`/warning behaviour as the intended contract and soften the acceptance criterion accordingly.

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
- Path-based formats (DOCX, spreadsheets, text) are explicitly routed to their respective format extractors.
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
- Scanned pages undergo layout analysis, deskewing, and dynamic binarization prior to OCR.
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

- DOCX inputs are walked in body order, extracting paragraphs and embedded tables seamlessly interleaved.
- Spreadsheets produce a cell-grained element stream with a dedicated meta-element per sheet.
- Spreadsheet headers are programmatically detected based on layout, placing the real header (a single header row, at row 0) at the top of the grid and preamble in metadata.
- Cell elements capture value, value type, formula, and number format. A `merge_range` field exists in the schema but is **not currently populated** by any extractor.
- Text boundaries are preserved natively without the need for visual OCR orchestration.

**TO-DO:**

- **Merged-cell coordinates are not captured (re: "cell elements capture … merge ranges").** `merge_range` is defined on the `Element` schema (`ingest/elements.py`) and round-tripped by the writer/reader, but **no extractor ever assigns it** — the spreadsheet extractor (`ingest/spreadsheet.py`) reads via pandas `header=None` plus an openpyxl pass that captures only `number_format`/`value_type`, and never inspects `ws.merged_cells`. As built, a merged region yields the top-left value with the remaining cells empty and no merge extent recorded. Decide whether to (a) populate `merge_range` from openpyxl's `merged_cells` and back-fill the empty cells' provenance, or (b) drop `merge_range` from the schema and the acceptance criterion until it is implemented.
- **Multi-row / hierarchical headers are not emitted (re: "spreadsheet headers … a single header row").** Spreadsheet headers are hard-coded to row 0 (`_emit_sheet`, `_sheet_rows`), so a genuine multi-row header is collapsed by pandas into one row and the extra header rows become data rows. Table extractors likewise only ever set `header_rows=[0]` (`strategies_file.py`, `views.py`); the downstream `header_rows` *list* and header-continuation folding exist but are never fed more than one row. Decide whether to detect and emit multi-row headers (populating `header_rows` with more than one index, and a header-row coordinate on cell elements), or to document single-row-header as the intended contract.

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
- Register ingestion explicitly bypasses the extraction/NLP pipeline. The bypass is **structural, not enforced**: register ingests write self-contained (Geo)Parquet with their own schemas and no `source_hash`/element layout, so the text-based downstream stages — which discover work by globbing the extraction sidecars (`*.elements.parquet` / `*.chunks.parquet`) — never see register output. Nothing actively rejects an operator who points a downstream stage at a register directory; it is incompatible by construction rather than blocked by a guard.

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
- Output is **not** byte-for-byte reproducible across runs: the manifest stamps a wall-clock `extracted_at_iso`, and Parquet writes do not pin the writer's embedded metadata (`created_by`, timestamp coercion, statistics).

**TO-DO:**

- **Cross-run Parquet determinism is not specified or achieved (re: the two determinism criteria above).** Hashes and logical row ordering are deterministic, but two runs over identical inputs produce different bytes because `write_results` stamps `extracted_at_iso` from wall-clock `time.gmtime()`, and `_write_rows` calls `pq.write_table(..., compression="zstd")` without pinning encoding (`store_schema`, `coerce_timestamps`) or suppressing the version-bearing `created_by` metadata (`store/output.py`; `_source_hash`; `_write_rows`). Decide whether byte-for-byte reproducibility is a requirement: if so, source the timestamp deterministically (e.g. from the run id) and pin the Parquet writer options; if not, replace the reproducibility expectation with an explicit "row order + `source_hash` are stable; file bytes are not" contract and add a test asserting the stable subset.

## 9. Redaction Handling

**As** a user,
**I want** redacted regions detected visually and handled according to a configurable policy,
**so that** sensitive visual blocks are managed consistently.

**Given** an extracted PDF document
**When** the operator runs the redaction stage
**Then** redactions are detected per page, applied based on the chosen mode, and (via the standalone shard CLI) written as an independent sidecar.

**Acceptance criteria:**

- Pages are rendered visually to detect solid redaction regions (a vector `get_drawings()` fast path for native filled rectangles, falling back to CV2 contour detection on the rasterised page).
- Multiple modes are supported: `flag` (annotation only, no text change), `blackout` (prepend a `<REDACTED>` marker to affected page text), and `delete` (clear affected page text). The `<REDACTED>` marker is a content marker, distinct from the human-readable warning strings described below.
- A redaction report is attached to the extraction result, and per-page warning strings (e.g. `page N: K redacted region(s) detected`) are appended to the extraction's warnings.
- Detected redactions are written as an independent Parquet sidecar (`*.redactions.parquet`) **only on the standalone `womblex redact --shards --pdfs` path** (`redact/batch.py`). The in-process E2E `run` path (`batch.py` → `run_redaction`) does **not** emit a redactions sidecar — it annotates the in-memory extraction result (element `meta['has_redaction']`, warning strings, and `has_redaction` folded onto chunks downstream) and, for `blackout`/`delete`, mutates page text in place.
- Redaction runs at a single fixed point in the pipeline (extraction → redaction detection, per `batch.py`); there is **no** timing/pipeline-point configuration for it. `RedactionConfig` exposes only `enabled`, `mode`, `threshold`, `min_area_ratio`, `max_area_ratio`, `dpi`, and `use_layout_filter`.

**TO-DO:**

- **Redaction has no configurable pipeline point, contrary to earlier documentation.** `RedactionConfig` (`config.py`) carries no `pipeline_point`/timing field, and `batch.py` runs redaction detection at one fixed position (immediately after extraction, before any downstream stage). The claim that redaction runs "at configurable pipeline points (post_chunk, post_enrichment)" in `CLAUDE.md` and older docs appears to be conflated with `PIIConfig.pipeline_point` — that configurability exists for the **PII** stage, not redaction. Decide whether to (a) add a genuine `RedactionConfig.pipeline_point` (post_extraction / post_chunk / post_enrichment) mirroring the PII stage if before/after-chunking placement is actually wanted, or (b) treat the fixed post-extraction position as the intended contract and correct the stale `CLAUDE.md` wording accordingly.
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
- Token counting utilizes a local tokenizer to ensure no network dependency.
- Partial redaction markers that split across boundaries are automatically repaired in the chunk overlay.
- Optional AI chunking uses semantic boundaries, leveraging previously persisted enrichment data if it matches the source text.
- If AI chunking detects a mismatch with previously persisted enrichment, it falls back to self-enrichment rather than using mismatched data.

## 11. PII Detection and Masking

**As** a user,
**I want** PII detected, masked, and retained reversibly for authorized audits,
**so that** sensitive data is protected for publication without destroying internal provenance.

**Given** an extracted document (and optionally post-enrichment spans)
**When** the operator runs the PII stage
**Then** sensitive spans are replaced with typed tags, generating an auditable sidecar and a clean-text layer.

**Acceptance criteria:**

- Candidates for entities like persons and addresses are generated contextually and validated via embedding similarities.
- If run post-enrichment, external entity spans are merged into the PII candidates.
- Identified PII spans in the clean-text layer are replaced with normalized typed tags (e.g., `<PERSON_1>`).
- The spans are written to an independent Parquet sidecar (`*.pii_spans.parquet`) that retains each span's original text plus its chunk offsets and graph `entity_id` — the audit record from which an authorised reversal can be reconstructed against the clean-text layer. The sidecar is a reversal-enabling audit layer; no automatic un-masking operation is implemented.

## 12. Knowledge Graph and External Enrichment

**As** a user,
**I want** entities and relationships extracted via external APIs and synchronized into a unified document graph,
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
- Output is order-independent within a run: the money sidecars carry identical rows (same `source_hash`, same logical order) whether the stage runs before or after the knowledge-graph stages, because money reads only the extraction sidecars and never stamps the wall-clock `extracted_at_iso`. The timestamp half of the Requirement 8 cross-run caveat therefore does not apply to it; the Parquet-writer-metadata half (`created_by`, encoding) is shared with every sidecar.
- Cell annotations carry their sheet, parent element order, row, and column coordinates. They do **not** carry a merged-cell extent or a header-row coordinate — an amount inside a merged region cannot be tied back to the merge span, and the column's evidencing header is recorded only as joined text on the `money_columns` sidecar.

**TO-DO:**

- **Cell annotations lack merged-cell and header-row coordinates (re: "cell annotations carry their … coordinates").** `_cell_row` (`process/money_stage.py`) emits `(row, col)` / `(parent_elem_order, row, col)` / `(sheet, row, col, elem_order)` but no merge extent and no header-row index — a direct consequence of the two Requirement 6 gaps (`merge_range` never populated; `header_rows` never more than `[0]`). Multi-row *table* headers are consumed correctly downstream (`_header_texts` joins every declared header row; `_fold_continuation` folds continuations), so this is blocked on extractors emitting the coordinates, not on the annotator. Resolve alongside the Requirement 6 TO-DOs; once merge/header-row data is emitted, add a merge-span and header-row column to the money span/cell rows.

## 14. Batch Processing, Resiliency, and Operational Controls

**As** a user,
**I want** reliable batching, checkpointing, and discrete execution controls (CLI/UI),
**so that** I can manage large runs, recover from failures, and inspect outputs easily.

**Given** a configured run (local or distributed)
**When** the operator utilizes the CLI or Web UI
**Then** documents process in isolated, resumable batches, with independent stage controls and visual inspection capabilities.

**Acceptance criteria:**

- Processing occurs in configurable batches, appending results and writing checkpoints upon batch completion.
- Resuming an interrupted run automatically reconciles checkpoints and skips already-completed documents.
- Individual document errors are isolated, recorded in the manifest, and do not crash the wider batch.
- CLI commands allow stages to be dispatched independently, with idempotent queueing ensuring dependent stages sequence properly.

## 15. Web Console Shell, Navigation, and Deployment Modes

**As** a user,
**I want** an optional web console that reads the artefacts the pipeline already writes,
**so that** I can navigate every run domain in one place without a separate tool or a live pipeline connection.

**Given** a `womblex[ui]` install bound to a single run source (a local output root or an object-store URI)
**When** the operator launches the console and opens it in a browser
**Then** a persistent shell routes between the five console domains and serves the read API even when no frontend build is present.

**Acceptance criteria:**

- The console binds to exactly one run source at construction (local output root or store URI) so no endpoint can be steered to read an unmounted directory.
- A persistent top bar (global search, run selector, execution controls) and a side-nav rail route between the Dashboard, Corpus Inspector, Semantic Chunk Inspector, Pipeline Composer, and Resources Console.
- The console is a reader over persisted artefacts and never edits a stage output; its only writable surfaces are dispatch, location/preset saving, and the report action.
- Dispatch requires a store, an ingest location, and a job queue to be configured; a console missing any of these still serves the full inspection surface (it simply cannot enqueue work).
- A bare install with no SvelteKit build still serves the read API; the SPA is mounted only when a build exists alongside it.

## 16. Dashboard — Queue and Stage Progress

**As** a user,
**I want** a run-scoped dashboard of queue state and per-stage progress,
**so that** I can monitor throughput and spot stalled jobs without touching the queue.

**Given** a selected run, with an optional job queue and the run's own per-stage checkpoints
**When** the operator opens the Dashboard
**Then** it presents queue counts and per-stage completion, polling while the tab is visible and pausing while it is hidden.

**Acceptance criteria:**

- Queue and stage state are read from sources the pipeline already writes (the job queue and per-stage checkpoints); with no queue configured, the dashboard falls back to checkpoints.
- Job status is rendered with the exact values the system writes (`pending` / `running` / `done` / `failed`), plus `stale` for a running row past its lock timeout and `skipped`.
- The dashboard only *names* a stalled job for a worker to recover; it never requeues, cancels, or claims work itself.
- Polling is paused while the browser tab is hidden so a backgrounded console stops hitting the queue.
- Per-stage progress renders in the declared presentation order (`PIPELINE_ORDER`, e.g. enrich before chunk) rather than an ad-hoc frontend ordering — a display convention over the dependency DAG, not a claim that this is the only valid execution order (see Requirement 3).
- Batch and stage logs for the run are listed newest-first and are individually viewable.

## 17. Corpus Inspector — Document Grid and Integrity Audit

**As** a user,
**I want** a dense, virtualised document grid with checkpoint and shard-integrity views,
**so that** I can inspect thousands of documents and confirm the run's outputs are complete and readable.

**Given** a selected run's manifest and shard directories
**When** the operator opens the Corpus Inspector
**Then** each document appears as a row with its status, a stage-checkpoint switcher, and an on-demand shard-integrity audit.

**Acceptance criteria:**

- The grid uses real table semantics with a sticky header, row virtualisation, and an announced total row count for accessibility.
- Document status is conveyed by a status pill (icon plus label), never by row background tint alone.
- A checkpoint switcher reports which stages are present for the run, ordered by the declared presentation order (`PIPELINE_ORDER`; see Requirement 3 for why this is a display convention, not the sole valid execution order).
- A verify-shards action runs the persistence audit, confirming required shard files exist, are readable, and match expected document counts.
- The grid supports a failed-only filter and a user-selectable density (comfortable / default / compact) persisted locally.

**TO-DO:**

- **Row virtualisation and the announced total row count are not implemented (re: "row virtualisation, and an announced total row count").** `DocumentGrid.svelte` renders a real `<table>` with a `<caption>`, `<th scope="col">` header, and a sticky header, but puts *every* row in the DOM rather than virtualising a window over them — an explicit code comment records that `aria-rowcount`/`aria-rowindex` are omitted because "every row is in the DOM here, so the browser's own count is correct … They become necessary when virtualisation lands." As built, the grid will not stay dense over thousands of rows (the story's own motivation), and there is no `aria-rowcount` announcing a total beyond the rendered window. Decide whether to (a) add windowed row virtualisation and the `aria-rowcount`/`aria-rowindex` announcement it requires, or (b) drop "virtualised" from the story and "row virtualisation, and an announced total row count" from this criterion until it is implemented. Resolve alongside the matching Requirement 21 TO-DO, which restates the same gap as an accessibility rule.

## 18. Semantic Chunk Inspector — Chunk, Entity, PII, and Money Overlays

**As** a user,
**I want** to read a document's chunks with entity, PII, and money overlays rendered inline,
**so that** I can verify chunking quality and confirm sensitive spans are masked correctly.

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

**As** a user,
**I want** to compose and validate a pipeline configuration visually and dispatch a run from it,
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

## 20. Resources Console — Connections and Reporting

**As** a user,
**I want** connection cards for the store, ingest, queue, and enrichment service with reachability tests and a credential-safe reporting action,
**so that** I can confirm the environment is wired correctly and flag bad records without exposing secrets.

**Given** the deployment's configured store, ingest, queue, and Isaacus connections
**When** the operator opens the Resources Console or files a report on a record
**Then** each connection is shown as a credential-masked card with a live "Test" action, and a report appends a note to an append-only feedback log.

**Acceptance criteria:**

- Four connection cards (store, ingest, queue, Isaacus) render deployment configuration, not any single run's artefacts.
- Connection strings are shown with secrets masked (never rendered in full in the DOM or a copy buffer), and each card offers a live reachability test.
- The store and ingest locations are editable when a settings directory is configured; the queue and Isaacus cards stay read-only, and an empty worker fleet is a normal resting state rather than an error.
- The report action files a reviewer's note plus the record to an append-only feedback log stored as a sibling of the runs, never inside one, confirmed by a toast rather than a modal.
- Reporting a record leaves its appearance unchanged — a report is an observation, not a state change.

## 21. Console Design System and Accessibility

**As** a user,
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

---

