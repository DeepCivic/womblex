# Trust and Recipes — Plan

*Status: in progress, October 2026.*

Phase 1 is done. Phase 2 is done but has design mistakes; Phase R repairs them before Phase 3.

## Target state
What Womblex does once this plan is done. Each requirement is given whole, as it will read in [functional_requirements.md](functional_requirements.md); requirements the plan does not touch are omitted, and gaps it does not close stay recorded there. At the end, the Outstanding list has no entry for Requirements 23 or 27.

### 2. Scale-Out and Environment-Agnostic Execution

**As** a user,
**I want** to scale the pipeline to a cluster and move stages between local and cloud environments,
**so that** I gain throughput without rewriting jobs or re-extracting documents.

**Given** a configured object store and a DBOS system database (Postgres, or a local SQLite file)
**When** the operator enqueues work and starts additional workers
**Then** workers claim batches concurrently without duplication, write the standard shard layout, and write the run manifest when the stages finish.

**Acceptance criteria:**

- Cloud workers and local runs use identical batch-processing logic and output identical shard layouts.
- Work runs as DBOS workflows, so a batch is never double-processed, a unit that finished is not run again, and workers can scale dynamically.
- Distributed run shards can be synced locally and consumed unchanged by local per-stage commands.
- A stage can execute in place over object storage under the same contract as local execution.
- Standard local and cloud storage URIs resolve natively.
- Object-storage writes are all-or-none per unit: every declared output exists before any is published, and a unit left partial by a failed upload never reads as complete, so the next run redoes and overwrites it. 
- A failed unit is recorded and the rest of the run continues. 
- A stage that rewrites outputs in place commits atomically, so an interrupted publish leaves the previous outputs intact.
- A distributed run writes `<run>/manifest.parquet` when its stages finish. `womblex finalize` still writes it on demand, with the same result.

### 12. Knowledge Graph and External Enrichment

**As** a user,
**I want** entities and relationships extracted via external APIs and joined into one document graph,
**so that** mentions map back to their source chunks regardless of run order.

**Given** a chunked extraction result and configured enrichment credentials
**When** the operator runs enrichment and graph generation
**Then** entities are extracted, retrying on API limits, and a graph links entities to chunks without rewriting earlier stages' files.

**Acceptance criteria:**

- Enrichment sends each document's text, in token-budgeted batches, to the configured provider (default Kanon-2 over Isaacus) and retries on rate limits.
- Only a provider that returns a Document persists one for AI-chunking reuse. With any other, chunking is semchunk's local splitting and no paid call replaces it.
- Graph generation builds nodes and edges, with a separate mention-to-chunk link layer.
- Graph refresh writes its links to a sidecar of its own, stamped with the chunk version it linked against, and never rewrites earlier stages' files.
- Readers can tell when refreshed links no longer match the current chunks, and use the matching links wherever they exist.
- Runs written before the refresh sidecar existed read as before.
- Graph refresh is idempotent and never skips on file existence alone.

### 14. Batch Processing, Resiliency, and Operational Controls

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
- A retry is a launch: before the fork it is refused for a stage that is unknown, still running, not failed or owned by another owner, and for anything a dispatch refuses (config, models, no live worker serving the stage, missing inputs).
- A stage that ran after a failure is retryable once the failed stage's latest attempt has succeeded, earliest stage first. Its retry redoes only the units the failure affected, so finished work is not paid for again.
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
- Job status uses the values the system writes, plus `stale` for a running workflow past the staleness threshold, and `skipped`.
- A retried stage counts once: counts take its latest attempt, and the superseded attempt stays listed `failed` with `retried_by` naming the next.
- The run record's lineage event reports the run's real outcome (complete, failed or cancelled), not always complete.
- The dashboard only names a stalled job; DBOS recovers it when its executor restarts. The dashboard never requeues, cancels or claims work.
- Polling pauses while the browser tab is hidden.
- Per-stage progress renders in pipeline order.
- Batch and stage logs are listed newest first and are individually viewable.

### 22. Service API — Submission and Retrieval

**As** a developer integrating another system,
**I want** an authenticated HTTP API to upload documents, submit runs, and read their results,
**so that** my software can use Womblex as a service without shell access to its host.

**Given** `womblex serve` bound to a store, a job queue and an ingest location, with a client registry
**When** a caller presents a service token, uploads documents, submits a run and polls it
**Then** the run is queued under the caller's ownership and runs under the submitted config, and its status, files and text are readable only by that caller (or an admin), gated by scope and file sensitivity.

**Acceptance criteria:**

- Every `/v1` endpoint except `health` and `ready` requires a bearer token whose SHA-256 is in the registry; a missing or unknown token is 401. Tokens are compared in constant time.
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
- The OpenAPI surface is pinned by a test.

### 23. Egress Bundle — Export to Downstream Consumers

**As** an operator handing a run to another system,
**I want** one finished run exported with its raw sources as a self-describing bundle,
**so that** consumers can reuse the corpus beside its sources without calling back into Womblex.

**Given** a finished run root, local or in an object store
**When** the operator runs `womblex egress <run> --to <dest>`
**Then** a bundle is written holding the unchanged corpus, deduplicated raw sources, a source index and a descriptor, and Womblex stops.

**Acceptance criteria:**

- The destination may be any `RemoteStore` URI (local directory, S3, MinIO, GCS) through one code path.
- `corpus/` mirrors the run's shard directory, every downstream sidecar included, plus `manifest.parquet`, unchanged.
- Raw sources are written to `sources/<source_hash><ext>`, deduplicated by hash.
- `source_index.parquet` has one row per document (`source_hash`, `doc_id`, `filename`, `ext`, `raw_key`, and a status of `resolved`, `hash_mismatch`, `not_found`, `unsupported_basis` or `upload_failed`); `raw_key` is null where no file was produced.
- Source resolution never aborts the export: a missing, mismatched, records-ingested or failed-upload source is reported in the index and in `egress_manifest.json`.
- `egress_manifest.json` carries the contract version and a per-document resolution report.
- A corpus-only export (`--no-sources` / `--corpus-only`) skips resolution and writes no `sources/` or `source_index.parquet`.
- `--source-root` resolves sources for a moved corpus; `--run-id` defaults to the run root's name and must agree with the run's stamp, if any.
- Re-exporting removes source files the fresh export neither wrote nor attempted, and leaves a failed-upload source untouched.
- A run held in an object store is copied store to store, with no local staging.
- A run ingested from an object store resolves each source from that store by `source_relpath`, checked against `source_hash`, with the same statuses.
- Womblex writes the bundle and nothing downstream of it: no retention, serving or write-back. Delivery to a workflow's destinations is Requirement 27.

### 27. Workflows — Check, Run, Dispatch, Schedule and Deliver

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
- Out-of-order steps are refused before launch with a stable code.
- Once a step has run, a failed step does not stop later steps; each records the failure it follows, and the run exits non-zero.
- Every output names the workflow that wrote it, and the run record holds the workflow's name and full spec.
- `dispatch-workflow` is refused before any row for anything the local runner refuses, a step the fleet cannot run, or an order the coordinator would change.
- `schedule-workflow` saves the workflow and syncs its schedules in one action, so stored triggers and DBOS schedules always match. It runs every check a firing makes, against the workflow's input location, and does not schedule a workflow a firing would refuse.
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
- A webhook receives five events: extraction settled, stage finished or failed, run finished, delivered, and delivery failed. Each is signed with a key held on the worker.
- Events carry the run's status in the API's words. There are no per-document or per-batch events.
- The run record lists every delivery.

**Out of scope:** pruning changed or deleted documents, backfilling missed firings, managing workflows through the API, workflow owners, per-step destinations, databases other than Postgres (each needs its own requirement), registering webhooks through the API, and write-back from consumers.

### 28. Fleet Runs — Honour the Dispatch, Report the Truth

**As** an operator, or a system submitting work,
**I want** fleet runs to use the config they were sent with, run only on workers checked for them, and show their real status after a fix,
**so that** fleet results match local ones and a repaired run no longer reads as failed.

**Given** a run dispatched from the CLI, the console, the API or a workflow
**When** workers process it, a stage fails, or the operator retries a failed stage
**Then** every stage runs under the dispatched config on a worker checked for it, a retry redoes only affected units, and the run shows its real status.

**Acceptance criteria:**

- Every stage of a fleet run runs under the config it was dispatched with; a worker's own config supplies only its paths.
- Config submitted through the API, console or CLI governs how stages run, not only which run.
- A worker serves a stage only once its model check passes for that stage.
- A run no live worker can serve is refused before launch.
- A retry is refused before the fork for anything a new dispatch would refuse.
- Retrying a failed stage makes later stages redo only the units the failure affected; finished work is not sent to a paid service again.
- Once every failed stage has been retried successfully, the run shows `done`.
- A scheduled firing whose dispatch failed and was later completed shows its real status, not `failed`.

**Out of scope:** automatic retries, and retrying from the API or console.

## How this plan works
The plan lists only outstanding work; finished work lives in the CHANGELOG and the docs. Merge size and the end-of-phase stocktake follow CLAUDE.md. An item is removed when done, and the plan is deleted with its last item.

A change that reverses a decision deletes the old decision from the decision docs rather than keeping it as a rejected alternative. The mistakes list below is the record of what went wrong.

Each work item names the target-state requirements it delivers. When an item meets a requirement, the requirement moves into the requirements document.

## Mistakes to avoid
Phase 2 made each of these. Check every item, here and in future plans, against this list before building.

1. **A setting before anything uses it.** Workflow `destinations` were stored and did nothing. *Rule:* a setting lands in the merge that makes it act.
2. **A file nobody reads.** Workflow conditions wrote a label sidecar nothing reads, derivable from the manifest anyway. *Rule:* every new output names its reader; anything derivable is derived on read.
3. **Automation that pays twice.** Each scheduled firing re-extracts every document and re-pays enrichment. *Rule:* anything that runs unattended processes only unseen documents.
4. **Tools taken from the operator.** A workflow could only be created in code. *Rule:* anything an operator sets up has a file and a command.
5. **A justification never delivered.** Workflows went into a database because a file "cannot hold a schedule", and the schedule then repeated all work (mistake 3). *Rule:* a feature that justifies a decision must work in the same phase, or the decision reopens.
6. **Halting a run after launch, then reverting.** Two changes stopped a run mid-flight over one document; both were reverted. *Rule:* every refusal happens before launch.
7. **Secrets in stored settings.** A planned destination would have stored a database password in the workflow. *Rule:* settings name where a secret lives, never the secret.
8. **Asking the owner what the design answers.** Reviews treated open items as settled and asked the owner to pick options. *Rule:* ask only when the answer changes what users get; otherwise the builder decides and cites where the design says so.

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
- **R.10 `plan` counts every paid call** (D12; Req 27).
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

### Work items
- **4.0 Split `cli/cloud.py`.** `jobs` moves to its own module ahead of the cap; commands and output are unchanged.
- **4.1 One-sentence run status** (Reqs 16, 22), from exact per-stage counts, shown by the CLI, API and console. The lineage event reports complete, failed or cancelled instead of always complete.
- **4.2 Graph refresh writes its own sidecar** (Req 12), stamped with the chunk version it linked against; readers use it where it matches. Until 4.3 it also keeps rewriting the old files.
- **4.3 Graph refresh stops rewriting earlier files** (Req 12). Older runs read as before. `layout` still replaces its own sidecar.
- **4.3.5 Docs for 4.2 and 4.3.**
- **4.4 Final stocktake.** The mistakes list moves to [decisions-dead-ends.md](decisions-dead-ends.md); the rest goes to the decision docs and CHANGELOG, and the plan is deleted.
