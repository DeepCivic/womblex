# Womblex — Functional Requirements

This document is the single source of truth for **what** Womblex must do, expressed
as user stories with testable acceptance criteria. It deliberately does **not**
cover *how* the system is built, why decisions were made, or measured accuracy.

**Paid features are tagged.** A requirement whose behaviour can start a call billed by a third party carries a **Paid:** line under its heading, saying what is billed and which setting switches it on. A requirement without one engages nothing paid. The paid models are the Kanon-2 Enricher and Kanon-2 Embedder (Isaacus, on the hosted API or a SageMaker endpoint) and Mistral Pixtral Large on AWS Bedrock (`mistral-ocr`). Everything else runs locally and costs nothing per call, including semchunk (MIT), the Kanon-2 tokeniser (Apache-2.0), PaddleOCR, PP-DocLayout-M and a self-hosted Ollama. A merge that adds, removes or changes a paid call updates the line in the same merge.

Acceptance criteria name outcomes a user can observe. How a criterion is tested goes on the requirement's **Verification:** line.

## Glossary

Terms the requirements use, as the code defines them.

- **batch** — a numbered group of documents extracted together; it writes one `batch-NNNN` set of shard files.
- **shard layout** — each batch's four extraction Parquet files (`elements`, `table_cells`, `form_fields`, `_manifest`) plus the sidecars later stages write beside them, with the schemas in `src/womblex/store/output.py`.
- **unit** — what a stage runs as one step: one batch's shard files for a per-batch stage, or the whole run for a whole-run stage (`plan_units` / `run_unit` in `src/womblex/cloud/stage_runner.py`). A unit's outputs are published all or none.
- **affected unit** — a unit of a later stage whose inputs came from a unit the failed stage did not finish.
- **job** — one row on the queue (an extraction batch, a stage or a stage coordinator), as `womblex jobs` and the API list it (`src/womblex/cloud/jobs.py`).
- **stage attempt** — one run of a stage over a run: its first dispatch, or a retry named `<run>:stage:<stage>:retry:<N>`. Counts take the latest attempt.
- **staleness threshold** — `stale_after_seconds`, default 900: a `running` job with no update for longer than this is listed `stale`.
- **extraction settled** — no extraction batch of the run is still pending or running; the coordinator waits for this before it runs the downstream stages.
- **DBOS** — the durable workflow system: the queue, the workflow rows and the recovery of a stopped worker's work, kept in one system database (Postgres for a fleet, a local SQLite file otherwise; `src/womblex/cloud/dbos_app.py`).
- **semchunk / Kanon-2** — semchunk is the local, MIT-licensed chunker and the Kanon-2 tokeniser runs locally; the Kanon-2 Enricher and Embedder are Isaacus's paid models.

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

**Paid:** Launches workers that run every paid stage the run's config enables (Requirements 10, 12, 29, the `embed` stage, `mistral-ocr`). A SageMaker deployment of the Kanon-2 Enricher or Embedder is billed by AWS while its endpoint runs, called or not.

**As** a user,
**I want** to scale the pipeline to a cluster and move stages between local and cloud environments,
**so that** I gain throughput without rewriting jobs or re-extracting documents.

**Given** a configured object store and a DBOS system database (Postgres, or a local SQLite file)
**When** the operator enqueues work and starts additional workers
**Then** workers claim batches concurrently without duplication, write the standard shard layout, and write the run manifest when the stages finish.

**Acceptance criteria:**

- Cloud workers and local runs run the same batch-processing code and, over the same input, write the same shard layout with the same `content_digest` for each document.
- Work runs as DBOS workflows, so a batch is never double-processed, a unit that finished is not run again, and workers can be added or stopped while a run is in progress.
- A stage dispatched again under new stage ids (a changed workflow, a later firing) skips a unit whose outputs were published by the same Womblex version under the same stage config from unchanged inputs, so identical work is neither redone nor re-paid. Any difference runs the unit, and a retry runs every unit it holds.
- Distributed run shards can be synced locally and consumed unchanged by local per-stage commands.
- A stage can execute in place over object storage under the same contract as local execution.
- A storage location may be a local path or a `file://`, `s3://` (S3 and MinIO), `s3a://`, `gs://`, `gcs://`, `az://` or `abfs://` URI.
- Object-storage writes are all-or-none per unit: every declared output exists before any is published, and a unit left partial by a failed upload never reads as complete, so the next run redoes and overwrites it. A failed unit is recorded and the rest of the run continues. A stage that rewrites outputs in place commits atomically, so an interrupted publish leaves the previous outputs intact.
- A distributed run writes `<run>/manifest.parquet` when its stages finish. `womblex finalize` still writes it on demand, with the same result.

**Verification:** `womblex verify-shards <local run> --compare-to <fleet run>` over one input reports no difference.

**TO-DO:**

- **`verify-shards --compare-to` does not compare `content_digest`.** It prints each run's aggregate counts (`store/shard_audit.py`, `format_audit_diff`) side by side and flags nothing, so a document whose `content_digest` differs between a local and a fleet run goes unreported. Compare the manifests' `content_digest` per `source_hash` and report each difference.
- **Azure storage URIs aren't credential-wired.** `storage_options_from_env` (`store/remote.py`) wires only `s3://`; `az://`/`abfs://` fall through to adlfs unauthed and untested. Add an `az`/`abfs` branch reading `AZURE_STORAGE_*`, plus coverage.
- **No Azure ML connection in the enrichment paths.** `utils/isaacus_client.py` handles the hosted API and AWS SageMaker (`ISAACUS_SAGEMAKER_ENDPOINTS`) but has no Azure ML equivalent. Add an Azure ML connection option so those paths can reach an Azure ML endpoint (`AZUREML_ENDPOINTS`, client factory, `unserved_models` pre-flight).

## 3. Ingest-First Data Flow and Operation Composition

**Paid:** Composes stages that include paid ones: `enrich`, `embed`, AI chunking in `chunk`, and `mistral-ocr` in extraction. None runs unless configured.

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

- **A missing required input is not a fail-fast, before-processing error for a single `run-stage --store`.** `stage_runner.run_stage_remote` resolves required inputs *per base* and raises `NotReady`, caught and logged as a warning; the run exits `0` unless every discovered base is not ready (`StageRunSummary.exit_code`). Deliberate: a still-draining fleet must not read as an ordering error. The whole-sequence check now exists for a stored workflow and for stage dispatch (`cloud/workflow_order.py`, refusals `input_missing` / `input_out_of_order` / `input_stale`): option (a) from the earlier note, with extraction's siblings treated as arriving. A stored workflow dispatched with `dispatch-workflow` is checked the same way before any row; a lone `run-stage` keeps its per-base behaviour.

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

**Paid:** With `mistral-ocr` as the OCR engine, every OCR'd page is a billed AWS Bedrock call (Mistral Pixtral Large). The default engine (PaddleOCR) and Ollama run locally and cost nothing per call.

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

## 10. Chunking, and AI Chunking When Asked For

**Paid:** AI chunking (`chunking.chunking_model`) calls the Kanon-2 Enricher, billed per call, for each document with no matching persisted Document. Default chunking (semchunk, local) costs nothing.

**As** a user,
**I want** extracted text split into token-bounded chunks by semchunk, locally,
**so that** downstream analysis fits model limits, and chunking costs nothing unless I switch on AI chunking.

**Given** an extraction result
**When** the operator runs the chunking stage
**Then** narrative text and tables are split by semchunk within the token budget and saved as a chunks sidecar.

**Acceptance criteria:**

- Chunking is semchunk, an open-source library (MIT) that runs locally. The tokeniser that measures chunk size is a dataset config choice (any Hugging Face tokeniser, or a callable). Default chunking needs no network and no key.
- Narrative text and markdown-converted tables are chunked independently, each chunk tagged with its kind.
- Redaction markers split across a chunk boundary are repaired.
- AI chunking is a semchunk mode that only `chunking.chunking_model` switches on. It takes its boundaries from a Kanon-2 Enricher Document, so it is a paid feature unless a matching Document already exists.
- A Document the `enrich` stage persisted is reused, with no call, when its text is byte-identical to the narrative being chunked.
- Where no persisted Document matches, the chunk stage makes the Enricher call itself, because `chunking_model` asked for AI chunking. Every such call is logged per document, counted in the chunks footer and counted by `plan`. Nothing calls it a fallback.
- AI chunking needs an enrichment provider that returns a Document. With any other provider, `chunking_model` is set aside with a warning and chunking is semchunk's local splitting.
- A run with AI chunking whose Isaacus deployment cannot serve the Enricher is refused before launch.

## 11. PII Detection and Masking

**Paid:** Makes no paid call, but its main candidate source is the enrichment graph, the output of a paid stage (Requirement 12). Without enrichment, only the opt-in regex backstop detects anything.

**As** a user,
**I want** PII detected, masked, and retained reversibly for authorised audits,
**so that** sensitive data is protected for publication without destroying internal provenance.

**Given** a chunked shard directory, normally already enriched
**When** the operator runs the PII stage
**Then** sensitive spans are replaced with typed tags, generating an auditable sidecar and a clean-text layer.

**Acceptance criteria:**

- On the per-stage path (`womblex pii`), candidates are the enrichment graph's person and address entities, mapped onto narrative chunks, and each table's own enrichment mapped onto that table's chunks. The regex/context backstop runs only when `pii.use_regex_backstop` is set (default `false`); with no enrichment and the backstop off, nothing is detected. Each `*.clean_text.parquet` row carries `mask_status`: `masked` (a span was replaced), `no_entity` (a candidate source covered the chunk and found nothing) or `not_masked` (no candidate source covered it, so the text is verbatim), so unmasked text is never labelled masked.
- The stage is terminal: it runs after enrich and embed, never rewrites `*.chunks.parquet`, and writes the masked layer as a separate `*.clean_text.parquet` (when `pii.write_clean_text`, the default).
- The in-memory PII operation runs at a configurable point (`pii.pipeline_point`): before enrichment it uses the regex/context detector, and after enrichment it merges graph spans with it.
- Identified PII spans in the clean-text layer are replaced with normalised typed tags (e.g., `<PERSON_1>`).
- The spans are written to an independent Parquet sidecar (`*.pii_spans.parquet`) that retains each span's original text plus its evidence reference (the span's position in the source, at the precision the stage could establish) and graph `entity_id` — the audit record from which an authorised reversal can be reconstructed against the clean-text layer. The sidecar is a reversal-enabling audit layer; no automatic un-masking operation is implemented.

## 12. Enrichment and Knowledge Graph — Orchestration

**Paid:** Each request to the enrichment provider is billed when the provider is a paid model, as the default enrichment provider is (Requirement 29). Graph refresh is local and costs nothing.

**As** a user,
**I want** whichever enrichment provider is configured driven the same way, and its entities and relationships joined to my chunks,
**so that** mentions map back to their source chunks whatever the provider and whatever the run order.

**Given** an extraction result and a configured enrichment provider
**When** the operator runs enrichment, and graph refresh after any rechunk
**Then** Womblex sends each document to the provider within the limits the provider declares, writes what the provider returns as sidecars, and links each mention to its chunk without rewriting earlier stages' files.

**Acceptance criteria:**

- Womblex owns, for every provider alike: packing requests to the provider's declared limits (texts per request, token ceiling, the provider's own token counter), sending each table's markdown as its own text beside the narrative, splitting a text only when it exceeds the provider's declared ceiling, offset-merging the parts, the entity, edge and metadata sidecars, evidence references, and the mention-to-chunk links.
- Retries, back-off and fitting long input to the model are the provider adapter's concern; Womblex adds none of its own around a provider.
- A request that still fails is recorded against its documents, and the rest of the stage continues.
- The provider's entities and relationships are written as returned. Womblex does not build or alter the provider's graph; the mention-to-chunk link layer is the one part Womblex derives, from character offsets, for any provider.
- Graph refresh writes its links to a sidecar of its own, which records the chunk version it linked against, and never rewrites earlier stages' files.
- Readers can tell when refreshed links no longer match the current chunks, and use the matching links wherever they exist.
- Runs written before the refresh sidecar existed read as before.
- Graph refresh is idempotent and never skips on file existence alone.
- Enrichment is not chunking. Whether chunking reuses an enrichment result is Requirement 10's, never this stage's.

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
- Cell annotations carry the shared evidence reference: the element (the table, or the `sheet_cell`), the sheet, and the cell's row and column. Where an amount sits in a merged region, the merge extent lives on the source cell's `merge_range` (Requirement 6), not on the money span row, and the column's evidencing header is recorded only as joined text on the `money_columns` sidecar. A header-row coordinate is still not carried.

**TO-DO:**

- **Cell annotations lack a merged-cell extent and header-row coordinate on the span row (re: "cell annotations carry their … coordinates").** `_cell_row` (`process/money_stage.py`) emits the evidence reference (`elem_order`, `sheet`, `cell_row`, `cell_col`) and can now recover a cell's merge extent by joining back to the source element's `merge_range` (populated per Requirement 6), but it does not copy that extent — nor a header-row index — onto the money span row itself. Multi-row *table* headers are now emitted for DOCX (`w:tblHeader`) but spreadsheets and the legacy table view still collapse to `header_rows=[0]` (the remaining Requirement 6 gap), and no per-cell header-row coordinate exists yet. Decide whether to (a) copy the source cell's `merge_range` onto the span/cell rows and add a header-row column once multi-row headers are emitted, or (b) document the join-back-to-`merge_range` path as the intended contract and keep the span row free of the extent.

## 14. Batch Processing, Resiliency, and Operational Controls

**Paid:** A retry or resume reruns stages; each unit of a paid stage it redoes is billed again.

**As** a user,
**I want** reliable batching, checkpointing, and discrete execution controls (CLI/UI),
**so that** I can manage large runs, recover from failures, and inspect outputs easily.

**Given** a configured run (local or distributed)
**When** the operator uses the CLI or Web UI
**Then** documents process in isolated, resumable batches, with independent stage controls and inspection.

**Acceptance criteria:**

- Processing occurs in configurable batches, appending results and writing a checkpoint as each batch completes.
- Resuming an interrupted run reconciles checkpoints and skips completed documents.
- A document error is isolated, recorded in the manifest, and does not stop its batch.
- Stages can be dispatched independently from the CLI; idempotent queueing keeps dependent stages in order.
- `womblex retry-stage` forks a failed distributed stage from its first failed unit, so finished units are not rerun.
- A retry is a launch, refused before the fork with one of these codes: `not_found` (no such run or stage), `still_running`, `not_failed`, `earlier_failed` (a stage it ran after has not succeeded since), `already_retried`, `config_invalid` (the config does not load with registered models), `step_order` (an input the stage needs is missing from the store or out of order) or `no_worker` (no live worker passes the model check for the stage). A run another owner holds is refused before the fork too.
- A stage that ran after a failure ran on partial output, so it is retryable once the failed stage's latest attempt has succeeded, earliest stage first. Its retry redoes only its affected units, so finished work is not paid for again.
- Logs can be emitted as JSON lines (`womblex --log-format json` or `WOMBLEX_LOG_FORMAT=json`) with no added dependency; worker and batch records carry `run_id`, `job_id`, `stage`, and `source_hash` where known, and the default text format is unchanged.

## 15 to 21. The console

These requirements described the admin console (`womblex ui`). The console now lives in its own repository, [womblex-console](https://github.com/DeepCivic/womblex-console), which carries them in `docs/requirements.md`. Womblex ships no console; the numbers stay so that 22 onward keep theirs.

## 22. Service API — Submission and Retrieval

**Paid:** A submitted run launches the paid stages its config enables, billed to the operator's accounts, not the caller's.

**As** a developer integrating another system,
**I want** an authenticated HTTP API to upload documents, submit runs, and read their results,
**so that** my software can use Womblex as a service without shell access to its host.

**Given** `womblex serve` bound to a store, a job queue and an ingest location, with a client registry
**When** a caller presents a service token, uploads documents, submits a run and polls it
**Then** the run is queued under the caller's ownership and runs under the submitted config, and its status, files and text are readable only by that caller (or an admin), gated by scope and file sensitivity.

**Acceptance criteria:**

- Every `/v1` endpoint except `health` and `ready` requires a bearer token whose SHA-256 is in the registry; a missing or unknown token is 401.
- Scopes are `submit`, `read`, `read_raw` and `admin` (which implies the rest); a request outside the caller's scopes is 403.
- The service refuses to start with an empty registry unless run with `--insecure-no-auth`, and binds to loopback by default.
- An upload is written to `<ingest>/<client_id>/<upload_id>/`, bare file names only. An unusable, repeated or unsupported name is 400 with nothing written, an over-limit request is 413, and the response names the `input_prefix` to submit.
- A non-admin caller submits runs only over its own `<client_id>/` tree (403 otherwise). The run id is minted server-side and unique under concurrent submissions.
- A submitted `preset` or `config` is validated before any queue row (400 on failure) and governs every stage of the run, not only which stages run. A config enabling no downstream stage is an extraction-only run.
- A run belongs to the client that submitted it: another client's run is 404, an admin sees all runs, and runs dispatched from the CLI or a workflow carry no owner and are admin-only. A run never has mixed ownership.
- Document text defaults to the masked layer; raw layers (`chunks`, `elements`) require `read_raw`, by the layer's contract sensitivity.
- Run file listings report each file's row count, `contract_version` and `sensitivity`, so a caller can fetch Parquet directly under the data contract.
- A run's status is the same sentence the CLI shows (`summary`), with its counts.
- A run started by a workflow that names a webhook sends its events there (Requirement 27). Polling still works.
- Everything an operator surface needs is a `/v1` call, so a user interface is a renderer over the API and holds no Womblex logic: the pipeline graph, a config's JSON Schema, validation and YAML (`read`); presets listed and fetched (`read`), saved and deleted (`admin`); an ingest prefix previewed before submission (`submit`, the caller's tree); every run the store holds (`admin`); per run, the documents each stage wrote, the shard audit, the batch logs and each stage's checkpoint progress (`read`) and a document's chunks with their annotations (`read_raw`); the queue in one read (`read`, the caller's runs); and what the deployment is wired to, credentials masked, with a live check of each piece (`admin`).
- A per-run read over the store answers an `admin` for a run only the store holds; any other caller reads only the runs it owns.
- `ready` reports store and queue reachability and answers 503 when either is down; any other endpoint answers 503, not 500, when the queue, store or ingest location is unreachable.

**Verification:** the token is checked against every registry entry with `hmac.compare_digest` and no early exit; a test fails on any change to the OpenAPI surface.

## 23. Egress Bundle — Export to Downstream Consumers

**As** an operator handing a run to another system,
**I want** one finished run exported with its raw source documents as a self-describing bundle,
**so that** consumers can review and reuse the extracted corpus alongside its sources without calling back into Womblex.

**Given** a finished run root (holding `documents/` and `manifest.parquet`), local or in an object store
**When** the operator runs `womblex egress <run> --to <dest>`
**Then** a bundle is written under `<dest>/<bundle_prefix>/` holding the unchanged corpus, deduplicated raw sources, a source index and a descriptor, and Womblex stops.

**Acceptance criteria:**

- The destination may be any storage location Requirement 2 lists (a `RemoteStore` URI), through one code path.
- `corpus/` mirrors the run's shard directory, every downstream sidecar included, plus the consolidated `manifest.parquet`, unchanged.
- Raw sources are written to `sources/<source_hash><ext>`, deduplicated by hash; two documents sharing a hash point at one file.
- `source_index.parquet` has one row per document (`source_hash`, `doc_id`, `filename`, `ext`, `raw_key`, and a status of `resolved`, `hash_mismatch`, `not_found`, `unsupported_basis` or `upload_failed`); `raw_key` is null wherever no file was produced.
- Source resolution never aborts the export: a source whose status is `not_found`, `hash_mismatch`, `unsupported_basis` (a records-ingested document) or `upload_failed` is reported in the index and in `egress_manifest.json`.
- `egress_manifest.json` carries the contract version and a per-document resolution report.
- A corpus-only export (`--no-sources` / `--corpus-only`) skips resolution and writes no `sources/` or `source_index.parquet`.
- `--source-root` resolves sources for a moved corpus; `--run-id` defaults to the run root's name and must agree with the run's stamp, if any.
- Re-exporting removes source files the fresh export neither wrote nor attempted, and leaves a failed-upload source untouched.
- A run held in an object store is copied store to store, with no local staging.
- A run ingested from an object store resolves each source from that store by `source_relpath`, checked against `source_hash`, with the same statuses.
- Womblex writes the bundle and nothing downstream of it: no retention, serving, or write-back of consumer corrections. Delivery to a workflow's destinations is Requirement 27.

## 24. Swappable Models Without a Schema Change

**Paid:** The OCR, enrichment and embedding slots can hold paid models; swapping a slot can add or remove a paid call.

**As** a user with my own models, local or API-backed,
**I want** to swap the model behind a pipeline slot without forking Womblex,
**so that** I can use models suited to my corpus while a packaged CPU-friendly group stays the default.

**Given** an installed package that registers a model for a slot
**When** the operator names that model in the config
**Then** the run uses it in place of the default, with no change to any Parquet schema.

**Acceptance criteria:**

- Swappable slots: OCR engine, layout analyser, PII context model, chunk tokeniser, spellfix dictionary, enrichment provider, and embedding provider.
- An enrichment provider declares its limits (texts per request, token ceiling, token counter, network need, whether it returns a Document). Womblex keeps the packing, splitting above a declared ceiling, table pass, sidecars, evidence and chunk mapping (Requirement 12); retries and long-input handling stay inside the provider's adapter. A provider changes who recognises entities and nothing downstream. The Isaacus adapter is Requirement 29. An embedding provider declares its texts per request and network need. Only a provider that needs Isaacus is gated on an Isaacus deployment.
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

**Out of scope:** model provenance (26), container-image packaging of plugins, and slots that need a Parquet schema change, each needing its own requirement (25 and 26, and the rest): AI chunking with a non-Isaacus model, graph-driven PII detection, link-stage candidates, `graph_refresh`, the enrichment token-budget tokeniser, per-page or per-element OCR engine recording.

## 25. Pre-Run Model Check

**Paid:** At `load` the Kanon-2 Enricher and Kanon-2 Embedder are checked for connectivity only, with no billed request; at `smoke` each one an enabled stage uses is also sent one minimal request, which is billed. Mistral on Bedrock is pinged at either level, a one-token request, billed. A warning names each billed request before it is sent. Ollama's check is free.

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

## 27. Workflows — Check, Run, Dispatch, Schedule and Deliver

**Paid:** `run-workflow`, `dispatch-workflow` and every scheduled firing launch the workflow's paid steps; each firing is billed for the documents it processes. `validate` and `plan` make no paid call.

**As** an operator,
**I want** to declare a recipe of stages once and run it locally, on the fleet or on a schedule,
**so that** every run follows the same steps, refusals happen before any spend, scheduled firings pay only for new documents, and results reach downstream systems.

**Given** a workflow file (name, preset, config overlay, input location, ordered steps, triggers and destinations)
**When** the operator validates, plans, runs, dispatches or schedules it
**Then** every refusal happens before the first step, each firing processes only new documents, every sidecar a workflow step writes and every extraction file a scheduled firing writes names its workflow, and each destination receives the results.

**Acceptance criteria:**

- A workflow is a plain-text file; creating or editing one needs no code.
- Every command validates the file first (steps, cron and timezone, config overlay) and rejects unknown keys.
- The workflow digest ignores formatting and comments.
- Its `config` is laid over its preset's key by key and validated as a full config.
- `validate` runs every check `run-workflow` runs before its first step, and writes nothing.
- `plan` counts the tokens and requests of every paid call, AI chunking included, with each provider's own token counter, and shows local steps as free. It calls no paid API.
- Out-of-order steps are refused before launch with the code `step_order`, each issue naming `input_missing`, `input_out_of_order` or `input_stale`.
- Once a step has run, a failed step does not stop later steps; each records the failure it follows, and the run exits non-zero.
- Every sidecar a workflow step writes names the workflow that wrote it, and so does every extraction file a scheduled firing writes; the run record holds the workflow's name and full spec. Extraction a workflow did not start (`womblex run`, `enqueue`) names none.
- `dispatch-workflow` is refused before any row for anything the local runner refuses, a step the fleet cannot run, or an order the coordinator would change; workers run each stage under the workflow's config.
- `schedule-workflow` saves the workflow and syncs its schedules in one action, so stored triggers and the queue's schedules always match. It runs every check a firing makes (`check_firing`, against `--store` and the workflow's input store), so a workflow its firing would refuse is not scheduled.
- Each firing runs the saved copy against the workflow's input store (`input_binding.store`), on a worker that uses that store, and is refused before any row when the prefix holds no document.
- A scheduled workflow appends to one run: each firing processes only documents the run has not seen, and never reprocesses or re-pays for one it has.
- After an edit and reschedule, the new version applies to new documents only; earlier documents keep their results.
- A firing with no new documents runs nothing and logs that.
- A file whose content changed is a new document; the earlier version stays in the run. A file rewritten with the same bytes is not a new document.
- A stored workflow has an owner, the API caller that stored it (none for the CLI); the runs it starts belong to that owner, and a run already owned by another owner is refused. A workflow with no owner starts admin-only runs, like CLI runs.
- Fleet runs perform `pii` when a workflow includes it; runs without a workflow never do.
- A workflow can name destinations: an object store, a Postgres database with embeddings, and a webhook.
- A workflow names where secrets are found, never the secrets. No secret is written to the file or the stored copy.
- Every destination is checked before launch (its secrets resolve, its dependencies are installed); a failed check refuses the run.
- A Postgres destination requires the `embed` step, and sending masked text requires `pii`; a workflow missing either is refused before launch.
- Results are delivered after the last step, even when a step failed, and a delivery after a failure names the failed steps.
- A failed delivery is retried a set number of times, then recorded on the run. It never stops the run.
- After a failed stage is retried successfully, the retry command prints how to redeliver. Redelivery replaces: files are rewritten and rows upserted; the delivery record is kept as a trace (Requirement 34).
- An object-store destination receives the Requirement 23 bundle; each destination chooses whether raw sources are included.
- A Postgres destination gets one row per chunk with its embedding: masked text by default, raw only where the workflow asks, each row naming its layer.
- A webhook receives six events: `extraction_settled`, `stage_finished`, `stage_failed`, `run_finished`, `delivered` and `delivery_failed`. Each carries an `event_id` (kept when the event is resent, so the receiver can drop a duplicate), the `run_id` and the run's status in the API's words, and is signed with a key held on the worker.
- There are no per-document or per-batch events.
- The run record lists every delivery.

**Out of scope:** creating, editing or removing workflows through the API (listing, adding to and rerunning a collection through a stored workflow are Requirement 34), pruning changed or deleted documents, resolving the source of a changed file's earlier version (its bytes are gone from the store, so `SourceResolver` reads it as `hash_mismatch`), backfill of missed firings, per-step destinations, databases other than Postgres (each needs its own requirement), registering webhooks through the API, and write-back from consumers.

## 28. Fleet Runs — Honour the Dispatch, Report the Truth

**Paid:** A retry redoes the affected units of paid stages, and each is billed again.

**As** an operator, or a system submitting work,
**I want** fleet runs to use the config they were sent with, run only on workers checked for them, and show their real status after a fix,
**so that** fleet results match local ones and a repaired run no longer reads as failed.

**Given** a run dispatched from the CLI, the API or a workflow
**When** workers process it, a stage fails, or the operator retries a failed stage
**Then** every stage runs under the dispatched config on a worker checked for it, a retry redoes only affected units, and the run shows its real status.

**Acceptance criteria:**

- Every stage of a fleet run runs under the config it was dispatched with; a worker's own config supplies only its store, paths, credentials and model locations.
- Config submitted through the API or CLI governs how stages run, not only which run.
- A worker serves a stage only once its model check passes for that stage: work is queued by the models the run's config names for it, and a worker checks the local models of every stage in its own config, enabled or not, and an Isaacus model only for a stage its config enables, and joins only the queues for the models it checked.
- A run no live worker can serve is refused before launch.
- A retry is refused before the fork for each refusal Requirement 14 lists.
- Retrying a failed stage makes later stages redo only their affected units; finished work is not sent to a paid service again.
- Once every failed stage has been retried successfully, the run shows `done`.
- A scheduled firing whose dispatch failed and was later completed shows its real status, not `failed`.

**Out of scope:** automatic retries, and retrying from the API.

## 29. Isaacus Enrichment Adapter

**Paid:** Every request is a billed Kanon-2 Enricher call: per call on the hosted API, and through AWS for a SageMaker endpoint.

**As** a user with an Isaacus deployment,
**I want** Kanon-2 Enricher reached through a thin adapter over the Isaacus SDK,
**so that** I get the API's own retry and long-input handling, and pay only for the calls I configured.

**Given** the `enrichment` slot set to `kanon-2-enricher` and a declared Isaacus deployment (`ISAACUS_API_KEY`, or `ISAACUS_SAGEMAKER_ENDPOINTS`)
**When** the `enrich` stage runs
**Then** each request goes through the Isaacus SDK and each returned ILGS Document is converted to Womblex's enrichment result.

**Acceptance criteria:**

- Three things stay separate, in code and in docs. semchunk is an open-source library (MIT) that runs locally with no key. The Kanon-2 tokeniser is open source (Apache-2.0), vendored, and counts tokens locally with no call. Kanon-2 Enricher is a paid model, billed per call. Only the last is this adapter.
- Retries, including on rate limits, are the SDK's (`max_retries`, upstream default unless the corpus has a measured reason to diverge); the adapter has no retry loop of its own.
- Input longer than the model's context is handled by the API's `overflow_strategy` (passed through, `auto`). The adapter declares the API's limits (8 texts per request) and no token ceiling that makes Womblex split a document the API accepts whole.
- The hosted API and SageMaker go through the same adapter. A deployment that serves no enricher endpoint is refused before the run.
- Only the `enrich` stage calls the adapter, and only when configured. It is never a fallback for another provider or another stage.
- The adapter declares that it returns a Document. The raw Document is written only when `persist_document` is set; what reads it is Requirement 10's.

**Out of scope:** AI chunking (Requirement 10) and the Isaacus embedder (`embed`).

## 34. Collections — A Long-Lived Run Named for What It Holds

**Paid:** A firing that adds a collection's documents launches the workflow's paid steps, billed for the documents it adds (Requirement 27).

**As** an operator,
**I want** a workflow's documents kept in a run named for a collection, not for the workflow,
**so that** a collection keeps its identity when the workflow that feeds it is renamed or replaced.

**Given** a workflow file with a `collection`
**When** the operator schedules it and it fires
**Then** every firing appends to the one run named for the collection.

**Acceptance criteria:**

- A workflow may name a `collection`: one safe token, as a workflow name is. Without one, the run is named for the workflow, as before.
- A scheduled firing appends to the run named for the collection. Renaming the workflow while keeping the collection fires into the same run.
- A schedule synced before the `collection` field existed still fires into the run named for its workflow.
- The workflow's digest does not cover the collection, so naming or changing it leaves the digest of an otherwise unchanged workflow as it was.
- The source-provenance collection recorded in a firing's extraction files is the collection, else the workflow's name; extraction from a run no workflow started still records the preset name. Files written earlier keep what they were written with.

**Verification:** `tests/test_workflow_store.py` (the digest and an unsafe collection) and `tests/test_workflow_schedules.py` (a renamed workflow firing into its collection, with the provenance footer; a schedule synced without `run_id` firing into the run named for its workflow; on-demand adds, the kept copy, `remove-workflow` and `rerun-collection`; owners) with `tests/test_unit_receipt.py` for the skip rule.

- `add-to-collection` fires the stored workflow once, now, on the queue of its input store, as a schedule's firing: it makes the same checks before any row (a workflow not stored, or one the firing would refuse, queues nothing), adds only the documents the collection has not seen, and a repeat within the same second attaches rather than firing twice.
- A stored workflow's owner is the client it was stored for (`schedule-workflow --owner <client_id>`; none, admin, otherwise), and a copy stored before owners reads as unowned. Its firings run under that owner, so the collection's run is owned like an API run, and a firing for a run another owner holds is refused.
- A workflow with a collection and no triggers keeps its stored copy. `schedule-workflow --remove` unschedules and keeps the copy; `remove-workflow` removes the schedules and the copy by name, even when the file is gone, and leaves the collection's run and shards.

- `rerun-collection` runs the workflow file's current steps over the whole collection, outside a firing. A unit whose version, stage config and inputs are unchanged since it was published is skipped, so adding `pii` runs only `pii` and a changed setting reruns the stages it changes. It is refused before any row for what `dispatch-workflow` refuses.
- Limits: an in-place stage, `graph-refresh` and an output with no receipt run again over the whole collection on every rerun. Re-extraction is out: batch ids are deterministic and attach rather than rerun.

- The service API lists a caller's collections and shows one with its workflows, document count and the workflows its files record; another owner's collection is 404 and an admin sees all, including a collection whose run the store holds after its workflows were removed (adding to or rerunning that one is 404). Adding to and rerunning a collection (`submit` scope) act through the stored workflow, make the checks of `add-to-collection` and `rerun-collection`, and refuse before any row with the workflow refusal's code (409). A collection fed by several workflows needs one named.
- Every outbound webhook event carries the workflow's `collection` (null when it names none).

- A redelivery to a destination writes a new delivery record with the next `seq` and the `seq` it `replaces`, and keeps the earlier record as it was. A record names the documents whose `content_digest` changed since the delivery it replaces, and the run record (version 9) lists every delivery with the one that replaced it.
- The run manifest marks each document's version: `replaces` and `replaced_by` name the earlier and later version of the same `source_relpath` (by `source_hash`), and `is_current` is true for the latest. Versions follow batch order, never time. Per-batch manifests are not rewritten; they carry the columns null, as do shards written before them. The collection's detail reports how many documents are current.
- Limits: a moved or renamed file starts a new chain, and identical bytes at two paths form two chains.

**Out of scope:** creating, editing or removing a workflow through the API (so a non-admin owner is, for now, whoever the deployment stored the workflow for), document versions and document versions, each a later piece of the same requirement.

## Outstanding

Requirements not yet fully met. The detail lives in the **TO-DO** block under each requirement; this list is the index. When a gap is closed (or the criterion is deliberately narrowed), its TO-DO block and its entry here are removed in the same PR.


- **Requirement 2 (Scale-Out):** `verify-shards --compare-to` does not compare `content_digest`; Azure storage URIs (`az://` / `abfs://`) are not credential-wired, and the enrichment paths have no Azure ML connection.
- **Requirement 3 (Ingest-First Data Flow):** up-front composition check covers workflows (local and `dispatch-workflow`) and dispatch, not a lone `run-stage --store` (per-base `NotReady`).
- **Requirement 6 (Native Office and Spreadsheet Extraction):** multi-row headers are emitted for DOCX only; spreadsheets and the legacy table view collapse to row 0, no per-cell header-row coordinate exists, and cell formulas are never captured.
- **Requirement 7 (Standalone Reference Register Ingestion):** G-NAF and geospatial writes are not guarded against partial output.
- **Requirement 9 (Redaction Handling):** no redaction `pipeline_point`, and the E2E `run` path writes no `*.redactions.parquet`.
- **Requirement 13 (Money Annotation):** the money span row carries no merge extent or header-row coordinate.

---
