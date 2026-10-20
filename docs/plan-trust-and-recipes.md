# Trust and Recipes — Plan

*Status: in progress, October 2026.*

Phase 1 is done. Phase 2 is done but has design mistakes; Phase R repairs them before Phase 3.

## Target state
What Womblex does once this plan is done. Each requirement is given whole, as it will read in [functional_requirements.md](functional_requirements.md); requirements the plan does not touch are omitted, and gaps it does not close stay recorded there. At the end, the Outstanding list has no entry for Requirements 23 or 27.

### Glossary
Terms the requirements below use, as the code defines them.

- **batch** — a numbered group of documents extracted together; it writes one `batch-NNNN` set of shard files.
- **shard layout** — each batch's four extraction Parquet files (`elements`, `table_cells`, `form_fields`, `_manifest`) plus the sidecars later stages write beside them, with the schemas in `src/womblex/store/output.py`.
- **unit** — what a stage runs as one step: one batch's shard files for a per-batch stage, or the whole run for a whole-run stage (`plan_units` / `run_unit` in `src/womblex/cloud/stage_runner.py`). A unit's outputs are published all or none.
- **affected unit** — a unit of a later stage whose inputs came from a unit the failed stage did not finish.
- **job** — one row on the queue (an extraction batch, a stage or a stage coordinator), as `womblex jobs`, the dashboard and the API list it (`src/womblex/cloud/jobs.py`).
- **stage attempt** — one run of a stage over a run: its first dispatch, or a retry named `<run>:stage:<stage>:retry:<N>`. Counts take the latest attempt.
- **staleness threshold** — `stale_after_seconds`, default 900: a `running` job with no update for longer than this is listed `stale`.
- **extraction settled** — no extraction batch of the run is still pending or running; the coordinator waits for this before it runs the downstream stages.
- **durable workflow system** — DBOS: the queue, the workflow rows and the recovery of a stopped worker's work, kept in one system database (Postgres for a fleet, a local SQLite file otherwise; `src/womblex/cloud/dbos_app.py`).
- **semchunk / Kanon-2** — semchunk is the local, MIT-licensed chunker and the Kanon-2 tokeniser runs locally; the Kanon-2 Enricher and Embedder are Isaacus's paid models.

### 2. Scale-Out and Environment-Agnostic Execution

**Paid:** Launches workers that run every paid stage the run's config enables (Requirements 10, 12, 29, the `embed` stage, `mistral-ocr`). A SageMaker deployment of the paid models (Glossary) is billed by AWS while its endpoint runs, called or not.

**As** a user,
**I want** to scale the pipeline to a cluster and move stages between local and cloud environments,
**so that** I gain throughput without rewriting jobs or re-extracting documents.

**Given** a configured object store and a durable workflow system database (Postgres, or a local SQLite file)
**When** the operator enqueues work and starts additional workers
**Then** workers claim batches concurrently without duplication, write the standard shard layout, and write the run manifest when the stages finish.

**Acceptance criteria:**

- Cloud workers and local runs run the same batch-processing code and, over the same input, write the same shard layout with the same `content_digest` for each document.
- Work runs on the durable workflow system, so a batch is never double-processed, a unit that finished is not run again, and workers can be added or stopped while a run is in progress.
- Distributed run shards can be synced locally and consumed unchanged by local per-stage commands.
- A stage can execute in place over object storage under the same contract as local execution.
- A storage location may be a local path or a `file://`, `s3://` (S3 and MinIO), `s3a://`, `gs://`, `gcs://`, `az://` or `abfs://` URI.
- Object-storage writes are all-or-none per unit: every declared output exists before any is published, and a unit left partial by a failed upload never reads as complete, so the next run redoes and overwrites it. A failed unit is recorded and the rest of the run continues. A stage that rewrites outputs in place commits atomically, so an interrupted publish leaves the previous outputs intact.
- A distributed run writes `<run>/manifest.parquet` when its stages finish. `womblex finalize` still writes it on demand, with the same result.

**Verification:** `womblex verify-shards <local run> --compare-to <fleet run>` over one input reports no difference.

### 3. Ingest-First Data Flow and Operation Composition

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

### 5. PDF and Image Extraction with OCR Quality Controls

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

### 10. Chunking, and AI Chunking When Asked For

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

### 11. PII Detection and Masking

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

### 12. Enrichment and Knowledge Graph — Orchestration

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

### 14. Batch Processing, Resiliency, and Operational Controls

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
- A retry is a launch, refused before the fork with one of these codes: `not_found` (no such run or stage, or the run is another owner's), `still_running`, `not_failed`, `earlier_failed` (an earlier stage's latest attempt failed), `already_retried`, `config_invalid`, or `step_order` (an input is missing or out of order). It is also refused before the fork when no live worker passes the model check for the stage.
- A stage that ran after a failure is retryable once the failed stage's latest attempt has succeeded, earliest stage first. Its retry redoes only its affected units, so finished work is not paid for again.
- Logs can be JSON lines (`--log-format json` or `WOMBLEX_LOG_FORMAT=json`) with no added dependency; worker and batch records carry `run_id`, `job_id`, `stage` and `source_hash` where known. The default text format is unchanged.

### 16. Dashboard — Queue and Stage Progress

**As** an administrator,
**I want** a run-scoped dashboard of queue state and per-stage progress,
**so that** I can monitor throughput and spot stalled jobs without touching the queue.

**Given** a selected run, with an optional job queue and the run's per-stage checkpoints
**When** the operator opens the Dashboard
**Then** it shows the run's status sentence, queue counts and per-stage completion, polling while the tab is visible.

**Acceptance criteria:**

- Queue and stage state come from what the pipeline already writes (the job queue and per-stage checkpoints); with no queue, the dashboard falls back to checkpoints.
- A run's status is one sentence built from exact per-stage counts of done, pending and failed work, shown beside the counts. The CLI, API and dashboard show the same sentence.
- The sentence uses the API's status words (`pending` / `running` / `done` / `failed` / `cancelled`), plus a warning when a scheduled firing could not dispatch its stages.
- Job status is one of `pending`, `running`, `done`, `failed`, `cancelled`, `skipped` (a unit whose outputs were already published) or `stale` (a `running` job past `stale_after_seconds`, default 900).
- A retried stage counts once: counts take its latest stage attempt, and the superseded attempt stays listed `failed`, its `retried_by` field holding the next attempt's id.
- The run record's lineage event reports the run's real outcome (complete, failed or cancelled), not always complete.
- The dashboard only lists a `stale` job; the durable workflow system resumes it when its worker restarts. The dashboard never requeues, cancels or claims work.
- Polling pauses while the browser tab is hidden.
- Per-stage progress renders in pipeline order.
- Batch and stage logs are listed newest first and are individually viewable.

### 19. Pipeline Composer — Configuration, Validation, and Dispatch

**Paid:** Dispatching a composed pipeline launches its enabled paid stages.

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

### 22. Service API — Submission and Retrieval

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
- A run belongs to the client that submitted it: another client's run is 404, an admin sees all runs, and runs dispatched from the CLI, the console or a workflow carry no owner and are admin-only. A run never has mixed ownership.
- Document text defaults to the masked layer; raw layers (`chunks`, `elements`) require `read_raw`, by the layer's contract sensitivity.
- File listings report each file's row count, `contract_version` and `sensitivity`.
- A run's status is the same sentence the CLI and dashboard show, with its counts.
- A run started by a workflow that names a webhook sends its events there. Polling still works.
- `ready` reports store and queue reachability and answers 503 when either is down; any other endpoint answers 503, not 500, when the queue, store or ingest location is unreachable.

**Verification:** the token is checked against every registry entry with `hmac.compare_digest` and no early exit; a test fails on any change to the OpenAPI surface.

### 23. Egress Bundle — Export to Downstream Consumers

**As** an operator handing a run to another system,
**I want** one finished run exported with its raw sources as a self-describing bundle,
**so that** consumers can reuse the corpus beside its sources without calling back into Womblex.

**Given** a finished run root, local or in an object store
**When** the operator runs `womblex egress <run> --to <dest>`
**Then** a bundle is written holding the unchanged corpus, deduplicated raw sources, a source index and a descriptor, and Womblex stops.

**Acceptance criteria:**

- The destination may be any storage location Requirement 2 lists, through one code path.
- `corpus/` mirrors the run's shard directory, every downstream sidecar included, plus `manifest.parquet`, unchanged.
- Raw sources are written to `sources/<source_hash><ext>`, deduplicated by hash.
- `source_index.parquet` has one row per document (`source_hash`, `doc_id`, `filename`, `ext`, `raw_key`, and a status of `resolved`, `hash_mismatch`, `not_found`, `unsupported_basis` or `upload_failed`); `raw_key` is null where no file was produced.
- Source resolution never aborts the export: a source whose status is `not_found`, `hash_mismatch`, `unsupported_basis` or `upload_failed` is reported in the index and in `egress_manifest.json`.
- `egress_manifest.json` carries the contract version and a per-document resolution report.
- A corpus-only export (`--no-sources` / `--corpus-only`) skips resolution and writes no `sources/` or `source_index.parquet`.
- `--source-root` resolves sources for a moved corpus; `--run-id` defaults to the run root's name and must agree with the run's stamp, if any.
- Re-exporting removes source files the fresh export neither wrote nor attempted, and leaves a failed-upload source untouched.
- A run held in an object store is copied store to store, with no local staging.
- A run ingested from an object store resolves each source from that store by `source_relpath`, checked against `source_hash`, with the same statuses.
- Womblex writes the bundle and nothing downstream of it: no retention, serving or write-back. Delivery to a workflow's destinations is Requirement 27.

### 24. Swappable Models Without a Schema Change

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

### 25. Pre-Run Model Check

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

### 27. Workflows — Check, Run, Dispatch, Schedule and Deliver

**Paid:** `run-workflow`, `dispatch-workflow` and every scheduled firing launch the workflow's paid steps; each firing is billed for the documents it processes. `validate` and `plan` make no paid call.

**As** an operator,
**I want** to declare a recipe of stages once and run it locally, on the fleet or on a schedule,
**so that** every run follows the same steps, refusals happen before any spend, scheduled firings pay only for new documents, and results reach downstream systems.

**Given** a workflow file (preset, config overlay, input location, ordered steps, triggers and destinations)
**When** the operator validates, plans, runs, dispatches or schedules it
**Then** every refusal happens before the first step, each firing processes only new documents, every output names its workflow, and each destination receives the results.

**Acceptance criteria:**

- A workflow is a plain-text file; creating or editing one needs no code.
- Every command validates the file first and rejects unknown keys.
- The workflow digest ignores formatting and comments.
- Its `config` is laid over its preset's key by key and validated as a full config.
- `validate` runs every check `run-workflow` runs before its first step, and writes nothing.
- `plan` counts the tokens and requests of every paid call, AI chunking included, with each provider's own token counter, and shows local steps as free. It calls no paid API.
- Out-of-order steps are refused before launch with the code `step_order`, each issue naming `input_missing`, `input_out_of_order` or `input_stale`.
- Once a step has run, a failed step does not stop later steps; each records the failure it follows, and the run exits non-zero.
- Every output names the workflow that wrote it, and the run record holds the workflow's name and full spec.
- `dispatch-workflow` is refused before any row for anything the local runner refuses, a step the fleet cannot run, or an order the coordinator would change.
- `schedule-workflow` saves the workflow and syncs its schedules in one action, so stored triggers and the durable workflow system's schedules always match. It runs every check a firing makes, against the workflow's input location, and does not schedule a workflow a firing would refuse.
- Each firing runs the saved copy against the workflow's input location, on a worker that uses that store.
- A scheduled workflow appends to one run: each firing processes only documents the run has not seen, and never reprocesses or re-pays for one it has.
- After an edit and reschedule, the new version applies to new documents only; earlier documents keep their results.
- A firing with no new documents runs nothing and logs that.
- A file whose content changed is a new document; the earlier version stays in the run.
- A workflow has no owner; the runs it starts are admin-only, like CLI runs.
- Fleet runs perform `pii` when a workflow includes it; runs without a workflow never do.
- A workflow can name destinations: an object store, a Postgres database with embeddings, and a webhook.
- A workflow names where secrets are found, never the secrets. No secret is written to the file or the stored copy.
- Every destination is checked before launch (its secrets resolve, its dependencies are installed); a failed check refuses the run.
- A Postgres destination requires the `embed` step, and sending masked text requires `pii`; a workflow missing either is refused before launch.
- Results are delivered after the last step, even when a step failed, and a delivery after a failure names the failed steps.
- A failed delivery is retried a set number of times, then recorded on the run. It never stops the run.
- After a failed stage is retried successfully, the retry command prints how to redeliver. Redelivery replaces: files are rewritten and rows upserted.
- An object-store destination receives the Requirement 23 bundle; each destination chooses whether raw sources are included.
- A Postgres destination gets one row per chunk with its embedding: masked text by default, raw only where the workflow asks, each row naming its layer.
- A webhook receives six events: `extraction_settled`, `stage_finished`, `stage_failed`, `run_finished`, `delivered` and `delivery_failed`. Each carries an `event_id` (kept when the event is resent, so the receiver can drop a duplicate), the `run_id` and the run's status in the API's words, and is signed with a key held on the worker.
- There are no per-document or per-batch events.
- The run record lists every delivery.

**Out of scope:** pruning changed or deleted documents, backfilling missed firings, managing workflows through the API, workflow owners, per-step destinations, databases other than Postgres (each needs its own requirement), registering webhooks through the API, and write-back from consumers.

### 28. Fleet Runs — Honour the Dispatch, Report the Truth

**Paid:** A retry redoes the affected units of paid stages, and each is billed again.

**As** an operator, or a system submitting work,
**I want** fleet runs to use the config they were sent with, run only on workers checked for them, and show their real status after a fix,
**so that** fleet results match local ones and a repaired run no longer reads as failed.

**Given** a run dispatched from the CLI, the console, the API or a workflow
**When** workers process it, a stage fails, or the operator retries a failed stage
**Then** every stage runs under the dispatched config on a worker checked for it, a retry redoes only affected units, and the run shows its real status.

**Acceptance criteria:**

- Every stage of a fleet run runs under the config it was dispatched with; a worker's own config supplies only its store, paths, credentials and model locations.
- Config submitted through the API, console or CLI governs how stages run, not only which run.
- A worker serves a stage only once its model check passes for that stage.
- A run no live worker can serve is refused before launch.
- A retry is refused before the fork for each refusal Requirement 14 lists.
- Retrying a failed stage makes later stages redo only their affected units; finished work is not sent to a paid service again.
- Once every failed stage has been retried successfully, the run shows `done`.
- A scheduled firing whose dispatch failed and was later completed shows its real status, not `failed`.

**Out of scope:** automatic retries, and retrying from the API or console.

### 29. Isaacus Enrichment Adapter

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

## How this plan works
The plan lists only outstanding work; finished work lives in the CHANGELOG and the docs. Merge size and the end-of-phase stocktake follow CLAUDE.md. An item is removed when done, and the plan is deleted with its last item.

A change that reverses a decision deletes the old decision from the decision docs rather than keeping it as a rejected alternative. The mistakes list below is the record of what went wrong.

Each work item names the target-state requirements it delivers. When an item meets a requirement, the requirement moves into the requirements document.

Acceptance criteria name outcomes a user can observe. How a criterion is tested goes on the requirement's **Verification:** line.

**Paid features are tagged.** A requirement whose behaviour can start a call billed by a third party carries a **Paid:** line under its heading, saying what is billed and which setting switches it on. A requirement without one engages nothing paid. The paid models are the Kanon-2 Enricher and Kanon-2 Embedder (Isaacus, on the hosted API or a SageMaker endpoint) and Mistral Pixtral Large on AWS Bedrock (`mistral-ocr`). Everything else runs locally and costs nothing per call, including semchunk (MIT), the Kanon-2 tokeniser (Apache-2.0), PaddleOCR, PP-DocLayout-M and a self-hosted Ollama. A merge that adds, removes or changes a paid call updates the line in the same merge.

## Mistakes to avoid
Phase 2 made each of these. Check every item, here and in future plans, against this list before building. Phase 2 made the first eight; the ninth has recurred in almost every merge.

1. **A setting before anything uses it.** Workflow `destinations` were stored and did nothing. *Rule:* a setting lands in the merge that makes it act.
2. **A file nobody reads.** Workflow conditions wrote a label sidecar nothing reads, derivable from the manifest anyway. *Rule:* every new output names its reader; anything derivable is derived on read.
3. **Automation that pays twice.** Each scheduled firing re-extracts every document and re-pays enrichment. *Rule:* anything that runs unattended processes only unseen documents.
4. **Tools taken from the operator.** A workflow could only be created in code. *Rule:* anything an operator sets up has a file and a command.
5. **A justification never delivered.** Workflows went into a database because a file "cannot hold a schedule", and the schedule then repeated all work (mistake 3). *Rule:* a feature that justifies a decision must work in the same phase, or the decision reopens.
6. **Halting a run after launch, then reverting.** Two changes stopped a run mid-flight over one document; both were reverted. *Rule:* every refusal happens before launch.
7. **Secrets in stored settings.** A planned destination would have stored a database password in the workflow. *Rule:* settings name where a secret lives, never the secret.
8. **Asking the owner what the design answers.** Reviews treated open items as settled and asked the owner to pick options. *Rule:* ask only when the answer changes what users get; otherwise the builder decides and cites where the design says so.
9. **Isaacus's open-source parts treated as its paid models.** Requirements and docs ran semchunk, the Kanon-2 tokeniser and the Kanon-2 Enricher together. They described enrichment as if Isaacus were the only provider, put AI chunking inside enrichment, and called a paid chunk-time call a "fallback". *Rule:* semchunk (MIT, local), the Kanon-2 tokeniser (Apache-2.0, local) and the Kanon-2 Enricher and Embedder (paid, per call) are named separately in every sentence that touches them. Enrichment is written provider-neutral, with Isaacus as one adapter. AI chunking belongs to chunking. Every paid call names the setting that asked for it.

## Phase R — Repairs

### Decisions made 17 October 2026
Each merge records its decision in [decisions-operations.md](decisions-operations.md) and deletes any decision it reverses.

- **D1. Workflows are files.** Every workflow command takes a file path; the file is the master copy. Only scheduling stores a copy (firings read it). Dispatch stores none, since the run receives the whole workflow. Reverses code-only creation.
- **D2. A scheduled workflow appends to one run.** Each firing lists the input location, processes only documents the run lacks, and runs the steps on those. A changed file is a new document; the old version stays (pruning is out of scope). Each firing uses the workflow as it stands at firing, so an edit applies to new documents only. The refusal of a run already worked under another workflow digest is removed.
- **D3. Conditions are removed.** The manifest answers everything they labelled.
- **D4. `destinations` is removed** until Phase 3 can deliver.

### Decisions proposed 17 October 2026
From a review of the work merged 10–17 October.

- **D5. Run status takes each unit's latest attempt.** Today a run whose failed stage was retried successfully, or a scheduled run repaired by hand, still reads `failed`. Reverses the rule that the original failure keeps counting.
- **D6. A retry redoes only affected work.** Today retrying a stage that ran after a failure redoes every batch and re-pays for all of them.
- **D7. Every fleet run carries its config.** Today only workflows do; API, console and CLI runs use each worker's own config, so one run can behave differently per worker. Reverses Requirement 22's "selects downstream stages only".
- **D8. Work goes only to workers checked for it.** Today a worker checks only the models its own config enables, yet accepts work needing others. After:
  - a unit is routed only to workers whose model check covers it;
  - a worker checks the models for every stage in its config, enabled or not;
  - each worker records the work it serves;
  - work no live worker can serve is refused before launch;
  - a retry runs every check a dispatch does.
- **D9. A workflow names its input store.** Today a firing reads whichever store its worker uses. After, only workers on the named store run it. Credentials still come from the worker.
- **D10. Scheduling saves the workflow and its schedules together.** Today a stored workflow can change without its schedules. After, only `schedule-workflow` saves, checking and syncing in one action.
- **D11. A run records its whole workflow.** Today only the digest is recorded, and the spec is lost once the workflow changes. After, every sidecar and the run record carry the name and full spec.
- **D12. `plan` counts every paid call.** Today it misses AI-chunking calls and counts every provider with the Kanon-2 tokeniser. After, AI chunking is counted, each provider's own counter is used, and local steps show as free.
- **D13. Workflows have no owner.** Nothing sets one, so the field does nothing and is removed. Workflow runs are admin-only, like CLI runs. An owner returns if the API ever creates workflows. Reverses owner-scoped workflows.

### Work items
Each item is one merge under the size cap; an item that outgrows it splits into two that each stand alone.

- **R.1 Remove conditions** (D3; Req 27), local and fleet, with their docs.
- **R.2 Remove `destinations`** (D4; Req 27) and its TO-DO.
- **R.2b Remove the workflow owner** (D13; Reqs 22, 27), including the firing's owner check.
- **R.3 Workflows as files** (D1, D10; Req 27). Every command takes a file; an example workflow joins `configs/`. Only scheduling stores a copy, syncing schedules in the same action. The code example leaves the workflow guide.
- **R.3b A workflow names its input store** (D9; Req 27). Firings route only to workers on that store.
- **R.4 Firings append only new documents** (D2; Req 27); the mixed-digest refusal goes. If too large: the label change first, new-documents-only second.
- **R.5 Run status from the latest attempt** (D5; Req 28).
- **R.6 Every fleet run carries its config** (D7; Reqs 22, 28), with the API guide.
- **R.7 Work only to checked workers** (D8; Reqs 14, 28). If too large: routing first, the up-front refusal second.
- **R.8 Retries redo only affected work** (D6; Reqs 14, 28).
- **R.9 Runs record their whole workflow** (D11; Req 27). The run record's version bumps.
- **R.10 `plan` counts every paid call** (D12; Reqs 10, 27), AI chunking's included.
- **R.11 Stocktake.**

## Phase 3 — Delivery
A finished run delivers its results to the destinations its workflow names. Each merge adds one destination kind end to end; no kind can be named before it delivers.

### Decisions made 17 October 2026
Each is recorded with the merge that builds it.

- **Fleet runs perform `pii` when a workflow asks.** Runs without a workflow still never do. Reverses "the fleet never runs `pii`".
- **Fleet runs write their own manifest on finishing.** `womblex finalize` stays and is idempotent.
- **An object-store run egresses store to store**, with no local staging.
- **Status uses the API's existing words** (`pending` / `running` / `done` / `failed` / `cancelled`, plus a warning for a firing that could not dispatch), in the status sentence, events and lineage. Replaces an earlier vocabulary.
- **Delivery follows the retry rules.** A delivery after a failure is marked with it; after a successful retry, `retry-stage` prints how to redeliver. Nothing redelivers automatically.
- **Destinations are checked before launch**: secrets, the webhook signing key and extra dependencies, beside the model check.
- **A Postgres destination requires embeddings**; a workflow without `embed` is refused before launch.
- **Delivery requirements belong in Requirement 27.**

### Work items
- **3.1 Dependencies only.** The webhook client becomes core and the Postgres driver an extra. Needs human approval.
- **3.2a Fleet runs write their own manifest** (Req 2), through the same code as `finalize`.
- **3.2b Egress from an object store** (Req 23): store-to-store copy; sources fetched by `source_relpath` and checked against `source_hash`.
- **3.2c Object-store destination and the delivery mechanism** (Req 27). `destinations` returns with this kind only. Delivery after the last step, even on failure; bounded retries; failures recorded on the run, never stopping it; every delivery in the run record. The pre-launch checks and redelivery hint land here.
- **3.3 `pii` in fleet runs** (Req 27).
- **3.4 Postgres destination with embeddings** (Req 27). The workflow names where the password lives. One row per chunk, upserted on redelivery; masked text by default, raw only on request, each row naming its layer. Masked text without `pii`, or Postgres without `embed`, is refused before launch.
- **3.5 Webhook events** (Reqs 22, 27), with bounded retries and signing by a key held on the worker.
- **3.5.5 Egress guide covers delivery**, including what is sent. Docs only.
- **3.6 Stocktake.**

## Phase 4 — Day-to-day operation

### Decisions made 19 October 2026
Each is recorded in [decisions.md](decisions.md) with the merge that builds it.

- **Isaacus's parts are named separately.** semchunk is open source (MIT) and local. The Kanon-2 tokeniser is open source (Apache-2.0, per its Hugging Face card) and local. The Kanon-2 Enricher and Embedder are paid models, billed per call.
- **Enrichment is two requirements.** Requirement 12 is what Womblex does for any provider. Requirement 29 is the Isaacus adapter.
- **The Isaacus adapter is thin.** Retries are the SDK's own (429 and 5xx, `max_retries`). Long input is the API's `overflow_strategy`. The adapter declares the API's limit of 8 texts per request and no smaller split ceiling.
- **AI chunking belongs to chunking.** A chunk-time Enricher call is paid work that `chunking_model` asked for. It is logged and counted, never called a fallback.

### Work items
- **4.0 Split `cli/cloud.py`.** `jobs` moves to its own module ahead of the cap; commands and output are unchanged.
- **4.1 One-sentence run status** (Reqs 16, 22), from exact per-stage counts, shown by the CLI, API and console. The lineage event reports complete, failed or cancelled instead of always complete.
- **4.2 Graph refresh writes its own sidecar** (Req 12), stamped with the chunk version it linked against; readers use it where it matches. Until 4.3 it also keeps rewriting the old files.
- **4.3 Graph refresh stops rewriting earlier files** (Req 12). Older runs read as before. `layout` still replaces its own sidecar.
- **4.3.5 Docs for 4.2 and 4.3.**
- **4.3.6 The Isaacus adapter goes thin** (Reqs 12, 29). The retry loops in `analyse/enrich.py` and `analyse/embed.py` are deleted in favour of the SDK's `max_retries`. The adapter declares the API's limits (8 texts per request, no split ceiling below what the API accepts), and Womblex splits only above a declared ceiling.
- **4.3.7 AI chunking says when it pays** (Req 10). Every chunk-time Enricher call is logged per document and counted in the `*.chunks.parquet` footer. Code, comments and logs stop calling it a fallback (`process/chunker.py`, `process/chunk_stage.py`).
- **4.3.8 Docs name Isaacus's parts separately, and paid features are tagged** (Reqs 3, 5, 10, 11, 12, 19, 24, 25, 29). Docs only. Requirements 10, 12 and 29 move into [functional_requirements.md](functional_requirements.md). [models.md](models.md) records the tokeniser's Apache-2.0 licence. CLAUDE.md gets a Common Pitfalls entry with the three-way split and the **Paid:** tag rule. Requirements 3, 5, 11, 19, 24 and 25 move in as the target state gives them, with their **Paid:** lines, and [functional_requirements.md](functional_requirements.md) states the tag rule at its top. [decisions.md](decisions.md) records this phase's decisions. A grep for "fallback", "self-enrich" and "over Isaacus" across `docs/`, `src/` and CLAUDE.md is the evidence.
- **4.4 Final stocktake.** The mistakes list moves to [decisions-dead-ends.md](decisions-dead-ends.md); the rest goes to the decision docs and CHANGELOG, and the plan is deleted.
