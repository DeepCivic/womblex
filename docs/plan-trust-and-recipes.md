# Trust baseline and recipes — plan

*Status: in progress (2026-10). Phase 1 has shipped. Phase 2 shipped with design failures, and Phase R repairs them before Phase 3 starts. This document lists outstanding work only. Completed work is recorded in `CHANGELOG.md` and the enduring docs, and the conventions every merge holds to (merge size, stocktakes, stacked PRs, dependency merges) are in CLAUDE.md. Each merge removes its item from this document as it lands, and the document is deleted once its last item ships. A merge that reverses a recorded decision deletes that entry from the decision docs. It does not keep it as a "rejected" alternative: the design failures below are the record. Each phase ends with draft requirements written in the form of `functional_requirements.md`. The merge that satisfies a criterion moves it there, and removes the TO-DO and Outstanding entries it closes.*

## Design failures — do not repeat
Phase 2 shipped each of these. Before building, check every merge in this plan and every new plan against this list.

1. **A field stored before anything uses it.** `destinations` was added to the stored workflow as "stored, not acted on yet". It was a TODO in schema form, and it produced design questions about rows that do nothing. *Rule:* a field lands in the merge that acts on it, and only with the kinds that merge builds.
2. **An output nothing reads.** Conditions write `*.conditions.parquet`, which no module reads. Every label in it is a filter over `manifest.parquet`, which can be computed at any time. *Rule:* a new output names its reader in the same merge. Anything derivable from data already persisted is computed when read, not written.
3. **Unattended automation that repeats paid work.** Each trigger firing re-extracts the whole ingest prefix under a new run id, and re-enriches and re-embeds it at Isaacus prices. *Rule:* anything that runs unattended processes only documents it has not seen, keyed by `source_hash`.
4. **The operator surface was removed.** D-2.2 made a workflow creatable only by writing Python against `WorkflowStore`. "No specific YAML format" was misread as "no file at all". *Rule:* every object an operator defines has a file form beside `configs/*.yaml` and a CLI path. An operator never writes Python to use a feature.
5. **A justification that was never delivered.** Storing workflows in the database won over a recipe file because a file "cannot carry standing triggers". The triggers were then built so that they repeat all work at every firing (failure 3). *Rule:* when a decision is justified by a feature, that feature ships in a useful form in the same phase, or the decision is reopened.
6. **Refusals after launch, built and then reverted.** #186 refused a batch over failed evidence receipts, and #189 refused a run over one document's condition result. Both contradicted "a run is an arrow" and cost merges to undo. *Rule:* check a refusal against "a run is an arrow" before building it. Refusals go only in the pre-run checks.
7. **Credentials in stored rows.** `WorkflowDestination.target: str` would have stored a Postgres DSN, password included, in `womblex_workflows`. *Rule:* a stored object names an environment variable, never a secret.
8. **Decisions brought to the owner that the design already answers.** Plan reviews accepted stubs as premises and turned them into options for the owner. *Rule:* a question goes to the owner only if its answer changes what an operator or consumer gets. Anything the documented design answers is settled by the implementer and cited.

## Phase R — repair (branch)
Decisions taken 2026-10-17. Each merge records its decision in `decisions-operations.md` and deletes the entries it reverses.

- **D-R.1 Workflows are YAML files.** A workflow is a YAML file with the `StoredWorkflowSpec` fields. `run-workflow`, `validate`, `plan`, `dispatch-workflow` and `schedule-workflow` take its path. The file is the source. The `womblex_workflows` table holds the copy that scheduling writes and that a firing reads. `dispatch-workflow` writes no copy: the coordinator already receives the spec as its arguments. Reverses D-2.2 and the "recipe file" rejection.
- **D-R.2 A workflow's run grows; it does not restart.** A trigger firing adds to one long-lived run per workflow. It hashes each file under the ingest prefix, extracts only `source_hash` values the run's manifest does not hold, as batches numbered on from the last, and runs the steps over those batches only. Stage ids and units are scoped to the firing, because `run:stage` would attach a second firing to the finished stage. A file whose bytes change is a new document, and the old one stays. Handling superseded documents is out of scope. Replaces "No incrementality".
  Each firing runs under the digest of the stored copy it reads. After an edit, later firings apply the new recipe to new documents only. Documents processed under an earlier digest are not processed again, and the run record lists each digest with the stages it wrote. Merge 7b's `run_conflict` refusal is deleted, because its reason (stage ids of the form `run:stage`) goes with firing-scoped ids.
- **D-R.3 Conditions are removed.** The code, the sidecar, the spec field and their docs go. A filter over `manifest.parquet` gives the same answer.
- **D-R.4 The `destinations` field is removed.** It returns in 3.2c with the object-store kind only.

*Proposed 2026-10-17, not yet decided.* These repair Phase 2 and earlier merges found in the review of #182 to #222.
- **D-R.5 A run's state comes from the latest attempt of each piece of work.** A coordinator's state is derived from the latest attempts of its stages. A failed firing row stops counting once its run has a coordinator. A run whose failures have all been retried successfully, or whose orphaned firing has been dispatched, reads `done`. Reverses D-2.1's "the coordinator's own row stays `failed`".
- **D-R.6 A retry resends only the work the failure affected.** `ran_after_failed` is recorded per batch as well as per stage. Retrying a later stage runs only its units over batches whose inputs the failed stage's retry rewrote, plus units that failed or were not ready. Completed units over unchanged inputs are never sent to a paid service again.
- **D-R.7 Every dispatch carries its config.** Extraction batches, `enqueue-stages`, the console and `POST /v1/runs` pass the resolved config as a workflow argument, as merge 7a does for workflows. A worker contributes only its `paths`. The order check and the stamped `config_digest` are over the config the work actually runs under. This reverses Requirement 22's "selects downstream stages only".
- **D-R.8 Work runs only where its models were checked.**
  - A stage queue is named for the stage and the models its config names, as an extraction queue is named for its ingest root.
  - A worker checks the models of every stage contract in its own config, enabled or not, and joins only the queues it passed.
  - A worker records the queues it serves.
  - Dispatch, a firing and `retry-stage` are refused before any row when no live worker serves a queue the work needs.
  - A retry runs every check a dispatch runs, not only the model-registration check.
- **D-R.9 A workflow's input names its ingest root.** `input_binding` holds the ingest root URI beside the prefix. Credentials come from the environment, per failure 7. A firing is enqueued on that root's queue, so only a worker serving that root runs it. `schedule-workflow` checks against the same root, and its `--ingest` flag goes.
- **D-R.10 Scheduling writes the copy and its schedules together.** `schedule-workflow` is the only writer of the stored copy. In one call it runs `check_firing` and makes the schedules equal the copy's triggers. A copy with no triggers has no schedules.
- **D-R.11 A run's files name the recipe that wrote them.** Beside its digest, each sidecar written by a workflow step carries the workflow id and the full spec. The run record lists each observed workflow by id, digest and spec, so the recipe can be recovered from the run alone after the file or its stored copy is edited.
- **D-R.12 `plan` counts every paid call through the provider that makes it.**
  - It counts AI chunking's enricher requests for documents whose Document cannot be reused.
  - `EmbeddingLimits` declares a `token_counter`, as `ProviderLimits` does, and `plan` counts embed tokens with it.
  - A step whose provider declares `needs_network=False` is reported as making no paid calls.
- **D-R.13 A workflow has no owner.** No operator surface sets one (the CLI has no owner flag, and a service-API route for workflows is out of scope). So the copy holds no `owner`, firings run unowned like CLI and console runs, and the owner-prefix check on firings goes. An owner returns with a service-API route for workflows, if one is built. Reverses "owner-scoped as runs are" for workflows.

**Merges:**
- **R.1 Remove conditions** (D-R.3). Delete `cloud/workflow_conditions.py` and `store/conditions_output.py`, the labelling step in `cloud/workflows.py` and in the runner, the spec field, the condition digests in `cloud/jobs.py`, and their tests. Docs: the conditions entry in `decisions-operations.md`, `workflows.md`, Requirement 27, CLAUDE.md rows, `architecture.md`, `project-structure.md`. Split into fleet path then spec and modules if it goes over the cap.
- **R.2 Remove `destinations`** (D-R.4). The spec field, the runner's "not acted on" warning, tests. Docs: `workflows.md`, Requirement 27's destinations TO-DO and its Outstanding entry.
- **R.2b Remove the workflow's owner** (D-R.13). The `owner` column and argument in `cloud/workflow_store.py`, `WorkflowOwnedError`, the owner-prefix check in `check_firing` (`prefix_not_owned`), and the owner copied onto a firing's rows. Docs: `workflows.md`, Requirement 27, the CLAUDE.md rows.
- **R.3 Workflows from YAML** (D-R.1, D-R.10). The five commands take a path. An example workflow sits beside `configs/default-isaacus.yaml` and replaces its commented commands. `dispatch-workflow` stops writing the stored copy. `schedule-workflow` writes the copy, runs `check_firing` and syncs the schedules in one call. Docs: `workflows.md` (the "Saving one" Python block goes), README command table, D-2.2 deleted.
- **R.3b A workflow names its ingest root** (D-R.9). An `input_binding` field, the firing enqueued on the root's queue, and `schedule-workflow --ingest` removed. Docs: `workflows.md`, `decisions-operations.md` ("A trigger fires a dispatch").
- **R.4 Incremental firings** (D-R.2). `workflow_dispatch.fire_workflow` lists the prefix and hashes against the run's manifest, `dispatch.batch_specs` numbers on from the last batch, and stage ids and `plan_units` are scoped to the firing. `cloud/retry.py` follows the new stage ids. `run_conflict` is deleted from `workflow_dispatch.py` and `RefusalCode`. Docs: `decisions-operations.md` ("A trigger fires a dispatch"; "No incrementality" and the `run_conflict` part of merge 7b deleted), `workflows.md`. Split mechanical (firing-scoped ids) then behaviour (hash-and-skip) if over the cap.
- **R.5 Run state from latest attempts** (D-R.5). `cloud/jobs.py` rollup and `_orphaned`, and the API run state. Docs: D-2.1's "Kept as is" deleted, Requirement 28.
- **R.6 Every dispatch carries its config** (D-R.7). `cloud/dispatch.py`, `RunBoard.enqueue` / `enqueue_stages`, `workflows.extract_batch` and `_stage_config`. Docs: Requirements 22 and 28, `service-api.md`, `decisions-operations.md` (merge 7a widened to every path).
- **R.7 Model-keyed stage queues** (D-R.8). `dbos_app.stage_queue`, `worker.served_queues` and its record of served queues, the unserved-queue refusal in dispatch, firing and retry, and `retry_stage` running the dispatch checks. Docs: Requirement 28, `decisions-operations.md`. Split the queue naming from the refusal if over the cap.
- **R.8 A retry resends only affected work** (D-R.6). Per-batch `ran_after_failed` on the coordinator, and a later stage's retry run over the affected units only. Docs: D-2.1, Requirement 28.
- **R.9 Recipes in files and the run record** (D-R.11). `store/run_stamp.workflow_context` carries the id and spec, and `run_manifest._observed_workflows` lists them in the next run-record version (3.2c's `deliveries` takes the one after). Docs: `contract.md`, Requirement 27.
- **R.10 `plan` counts every paid call** (D-R.12). `cloud/workflow_plan.py`, `EmbeddingLimits`, `embed_stage.plan_embedding`, and an AI-chunking count from `chunk_stage`. Docs: `workflows.md`, `model-plugins.md`.
- **R.11 Stocktake S-R.**

### Draft requirements — Phase R
**Requirement 27 is replaced by:**

#### 27. Workflows — Define, Check, Run, Dispatch and Schedule

**As** an operator,
**I want** to define a recipe of steps over a preset in a file and run it locally, on the fleet or on a schedule,
**so that** a run's sequence is declared once, checked before it spends anything, traceable from what it writes, and a schedule pays only for documents it has not processed.

**Given** a workflow file (a preset, a config overlay, an ingest location and prefix, ordered steps and cron triggers)
**When** the operator validates, plans, runs, dispatches or schedules it by its path
**Then** everything that can refuse does so before the first step, each firing processes only new documents, and the files each step writes identify the workflow that wrote them.

**Acceptance criteria:**

- A workflow is a YAML file, and every workflow command takes its path. Defining or changing a workflow needs no code.
- Every command checks the file (steps, cron and timezone, config overlay) and refuses an unknown field.
- A workflow's digest depends only on its content, so comments and formatting do not change it.
- The workflow's config is laid over its preset's key by key, and the whole is validated as a full config.
- `validate` runs exactly the checks `run-workflow` runs before its first step, and writes nothing.
- `plan` reports the requests and tokens of every step that calls a paid service, AI chunking included, counted with the tokeniser of the model that receives them. It calls no paid service, and reports a step whose model runs locally as free.
- Step order is checked before launch. A step whose input is missing, produced later, or rewritten later is refused with a stable code.
- Once a step has run, a failed step does not stop the steps after it. Each later step records the failure it follows, and the exit code is non-zero.
- Every file a step writes names the workflow and its digest. The run record holds the full workflow for each digest found in the run's files, so the workflow behind a run can be read after the file has changed.
- `dispatch-workflow` queues the steps for the fleet. Before anything is queued, it refuses whatever the local runner refuses, a step the fleet cannot run, and an order the fleet would change.
- `schedule-workflow` stores a copy of the workflow and makes the schedules match its triggers in the same command, so the stored triggers and the live schedules never differ.
- `schedule-workflow` runs every check a firing runs, against the ingest location the workflow names, and does not schedule a workflow its firing would refuse.
- A firing runs the stored copy and reads the ingest location the workflow names, whichever worker picks it up.
- A scheduled workflow has one run that grows. A firing extracts only documents the run does not hold, matched by content hash, and runs the steps over those documents only.
- No document already in the run is re-extracted or resent to a paid service by a firing.
- After a workflow is edited and rescheduled, later firings apply the new version to new documents only. Earlier documents keep the output of the version that processed them.
- A firing that finds no new document queues nothing and logs that it found none.
- A file whose bytes have changed is a new document, and the earlier version stays in the run.
- A workflow has no owner. The runs it starts are visible to admins, as runs started from the CLI are.

**Out of scope:** removing or superseding a document whose file changed or was deleted, backfill of missed firings, a service-API route for workflows, workflow ownership, and destinations (Phase 3).

**New requirement:**

#### 28. Fleet Runs — Run as Dispatched, Report as Finished

**As** an operator or a service-API caller,
**I want** work I send to the fleet to run under the settings I sent, on workers able to run it, and to report its true state after a retry,
**so that** a fleet run gives the same result as a local run, and a repaired run is not reported as failed.

**Given** a run dispatched to the fleet from the CLI, the console, the service API or a workflow
**When** workers run it, a stage fails, or the operator retries a stage
**Then** every piece of work runs under the dispatched config on a worker checked for it, a retry resends only the work the failure affected, and the run's state reflects the latest attempt of each piece of work.

**Acceptance criteria:**

- Extraction and every stage run under the config the run was dispatched with. A worker's own config supplies only its local paths.
- A config or preset submitted through the service API, the console or `enqueue-stages` is the config the run uses, not only a selection of stages.
- A worker runs a stage only if it has checked the models that stage's config names.
- Work that no running worker can serve is refused before anything is queued.
- A retry is refused before it starts for anything a dispatch would refuse.
- Retrying a stage that ran after a failure re-runs only the batches the failure affected. Batches that completed over unchanged inputs are not resent to a paid service.
- A run whose failures have all been retried successfully reads `done`.
- A run whose firing failed after queueing extraction reads by its stages once they are dispatched, not `failed`.

**Out of scope:** retrying a stage automatically, and retry through the service API or the console.

This replaces Requirement 22's "selects downstream stages only" and adds to Requirement 14's retry criterion.

## Phase 3 — integration (branch)
A finished run sends its output somewhere by itself (a "destination"), declared in the workflow file. Each merge adds one destination kind that works end to end. No kind is stored before the merge that delivers it.

Decisions taken 2026-10-17, recorded with the merge that builds each one:
- **`pii` is a fleet step when a workflow lists it.** Config-driven dispatch (`enqueue-stages`) still never runs it. Reverses merge 7b's `stage_not_dispatchable` for `pii` and the `pipeline_order.py` rule for workflows.
- **The coordinator finalises every fleet run.** It writes `manifest.parquet` to the store, and `womblex finalize` stays as an idempotent re-run.
- **Egress copies a store-hosted run store to store.** Reverses "stage-in/stage-out" for egress only.
- **Run status uses the existing `/v1/runs` state words** (`pending`, `running`, `done`, `failed`, `cancelled`, plus `orphaned`) for the summary sentence, webhook events and lineage event types. Replaces D-4.1's word set and renames D-3.3's events.
- **Delivery follows D-2.1.** A delivery is labelled `ran_after_failed` like a later stage. Once the failed stage's retry succeeds, `retry-stage` prints the delivery's retry line. No automatic re-run.
- **Destinations check before launch.** Credentials, the webhook secret and the `[pgvector]` extra are checked in `prepare_workflow` and `check_firing`, beside the model check.
- **pgvector needs embeddings.** `embeddings` is a strict input, so a workflow without `embed` is refused before launch.
- **Destination criteria live in Requirement 27.**

**Merges:**
- **3.1 Dependencies only** (per D-3.4). `httpx` moves from the `[dev]` extra to core, and a new `[pgvector]` extra holds pgvector's client. `pyproject.toml` and `uv.lock`, with human approval.
- **3.2a The coordinator finalises.** Move `cmd_finalize`'s logic into a library function that the CLI and the coordinator both call. Docs: `decisions-operations.md`, CLAUDE.md rows.
- **3.2b Egress from a store-hosted run.** `build_bundle` reads the corpus from the result store, store to store, and resolves sources from the ingest store by `source_relpath`, checked by hash (D-3.1). `womblex egress` works on a fleet run. Docs: `egress.md`, Requirement 23's TO-DO and its Outstanding entry.
- **3.2c Object-store destination and the delivery model.** `destinations` returns with `object_store` only (a prefix and `include_sources`). A delivery is a DBOS workflow with bounded retries, enqueued after the last stage whether or not one failed. The run record gains `deliveries` (the record version after R.9's). A failed delivery labels the run and never stops it. The pre-launch check and the D-2.1 retry line ship here. Docs: `decisions-operations.md`, Requirement 27, `architecture.md`, CLAUDE.md rows.
- **3.3 `pii` as a fleet step.** Docs: `decisions-operations.md` (merge 7b), `workflows.md`, the `pipeline_order.py` comment.
- **3.4 Postgres with pgvector** (per D-3.2). The destination names the environment variable that holds its DSN. An idempotent upsert keyed `(run_id, source_hash, chunk_index, content_type)`, masked text by default and raw where declared, with the layer recorded on each row. A masked destination without a `pii` step, or any pgvector destination without `embed`, is refused before launch. Docs: `contract.md` ("Safe to hand onward"), README installation extras, Requirement 27.
- **3.5 Webhooks** (per D-3.3, events named in the state words). Each POST is a DBOS workflow with bounded retries, signed with an environment secret. Docs: `service-api.md` ("Callbacks and webhooks (consumers poll)"), README (callers poll), `decisions-operations.md` ("Consumers poll"), Requirements 22 and 27.
- **3.5.5 `egress.md` becomes "Egress and destinations"**, with a payload reference (docs only).
- **3.6 Stocktake S-3.**

### Draft requirements — Phase 3
**Requirement 27 gains (with 3.2c, 3.3, 3.4 and 3.5):**

Given: "… ordered steps, cron triggers and destinations". Then: "… and each destination receives the run's output once its steps have run".

- A workflow may declare destinations. A kind is accepted only once Womblex delivers it: an object store (3.2c), Postgres with pgvector (3.4), a webhook (3.5).
- A destination names the environment variables that hold its credentials. A secret never appears in the workflow file or its stored copy.
- Every destination is checked before launch: its credentials and webhook secret are set and its extra is installed. A pgvector destination also needs an `embed` step, and a masked one a `pii` step. A failed check refuses the run before any row.
- Delivery starts after the last step, whether or not a step failed. A delivery that follows a failure is labelled with the failures it follows.
- A failed delivery is retried a bounded number of times, then recorded on the run and in the run record's `deliveries`. It never stops the run.
- Once a retried stage succeeds, `retry-stage` names the delivery's retry. A re-delivery replaces the earlier one at the destination: the bundle prefix is rewritten and pgvector rows are updated in place.
- An object-store destination writes the bundle Requirement 23 defines, with `include_sources` set per destination.
- A pgvector destination upserts one row per chunk, keyed `(run_id, source_hash, chunk_index, content_type)`, with its vector. Masked text is the default and raw text is sent where the destination declares it. Each row records its text layer.
- A webhook destination POSTs five events, each signed with HMAC under an environment secret: extraction settled, stage done or failed, run done, delivery done or failed. Each event carries the run's state in the words `/v1/runs` uses. No event is sent per document or per batch.
- `pii` runs on the fleet when a workflow lists it. Config-driven dispatch (`enqueue-stages`) never runs it.

**Requirement 23 changes (with 3.2b):** Given becomes "a finished run, local or held in an object store". Added: "A run held in an object store is copied store to store, and its sources are fetched from the ingest store." The TO-DO and its Outstanding entry go.

**Requirement 2 gains (with 3.2a):** "A fleet run's `manifest.parquet` is written by the coordinator once the run's stages finish. `womblex finalize` re-runs it idempotently."

**Requirement 22 gains (with 3.5):** "A run dispatched from a workflow with a webhook destination signals its events to the caller's endpoint. Polling stays available, and a caller that registers no webhook polls."

**Out of scope for Phase 3:** a destination per step, databases or vector stores other than Postgres with pgvector (each brings its own extra and requirement), registering webhooks through the service API, and writing consumer corrections back.

## Phase 4 — operation (branch)
**Merges:**
- **4.0 Split `cli/cloud.py`.** `jobs` moves to its own CLI module (the file is near the 750-line cap). Commands and output stay the same.
- **4.1 `RunBoard.describe` and its surfaces.** One sentence per run in the state words, over the exact per-stage counts: `womblex jobs --run`, an additive `summary` on `/v1/runs/{id}`, the console. The OpenLineage event (`store/lineage.py`, today always `COMPLETE`) takes its type from the run's state (`cancelled` is `ABORT`, `failed` is `FAIL`). Docs: `contract.md` (the lineage export), Requirements 16 and 22, `service-api.md`, `DESIGN.md`.
- **4.2 The graph-refresh sidecar** (per D-4.2). Define the chunks digest (`store/content_digest.py`), then a sidecar keyed by it. Readers (`analyse/query.py`, the Chunk Inspector's entities overlay) prefer it. Until 4.3, graph-refresh writes both the sidecar and the in-place files. Docs: `contract.md` (minor version), CLAUDE.md rows.
- **4.3 Graph-refresh stops rewriting in place.** Its contract moves from `MutationMode.IN_PLACE` to `SIDECAR`, with `PRODUCER_OF`, `store/retention.py`, `workflow_order` and `workflow_plan` updated. Older runs fall back to the in-place `chunk_index`. `layout` keeps its in-place replace. Docs: CLAUDE.md rows, `project-structure.md`.
- **4.3.5 Docs:** `composable-design.md` (the outputs table's "rewritten in place" and the `graph-refresh` "in place" bullet), Requirement 12, `decisions.md` ("Graph-edge refresh"), `architecture.md`, README.
- **4.4 Stocktake S-4 (final).** Move the design-failures list into `decisions-dead-ends.md`, retire this plan into the decision docs and `CHANGELOG.md`, then delete it.

### Draft requirements — Phase 4
**Requirements 16 and 22 gain (with 4.1):**

- A run's status is one sentence, built from its exact per-stage counts, in the state words `/v1/runs` uses (`pending`, `running`, `done`, `failed`, `cancelled`, and `orphaned`). The counts stay alongside it.
- The sentence is shown by `womblex jobs --run`, an additive `summary` field on `/v1/runs/{id}`, and the Dashboard. All three give the same sentence for the same run.
- A run's OpenLineage event takes its type from the run's state: `done` is `COMPLETE`, `failed` is `FAIL`, `cancelled` is `ABORT`.

**Requirement 12 changes (with 4.2 and 4.3):** "Graph refresh operations explicitly rewrite mention-to-chunk links in place …" becomes:

- Graph refresh writes its own sidecar, keyed by the digest of the chunks it linked against, and never rewrites enrichment's files.
- A reader can tell when the refresh no longer matches the chunks, and prefers the refresh sidecar wherever it matches.
- A run written before the sidecar existed is read through its in-place links, unchanged.
