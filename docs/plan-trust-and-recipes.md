# Trust baseline and recipes — plan

*Status: in progress (2026-10). Phase 1 has shipped. Phase 2 shipped with design failures, and Phase R repairs them before Phase 3 starts. This document lists outstanding work only. Completed work is recorded in `CHANGELOG.md` and the enduring docs, and the conventions every merge holds to (merge size, stocktakes, stacked PRs, dependency merges) are in CLAUDE.md. Each merge removes its item from this document as it lands, and the document is deleted once its last item ships. A merge that reverses a recorded decision deletes that entry from the decision docs. It does not keep it as a "rejected" alternative: the design failures below are the record.*

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

- **D-R.1 Workflows are YAML files.** A workflow is a YAML file with the `StoredWorkflowSpec` fields. `run-workflow`, `validate`, `plan`, `dispatch-workflow` and `schedule-workflow` take its path. The file is the source. The `womblex_workflows` table holds the copy that dispatch and scheduling write and that a firing reads. Reverses D-2.2 and the "recipe file" rejection.
- **D-R.2 A workflow's run grows; it does not restart.** A trigger firing adds to one long-lived run per workflow. It hashes each file under the ingest prefix, extracts only `source_hash` values the run's manifest does not hold, as batches numbered on from the last, and runs the steps over those batches only. Stage ids and units are scoped to the firing, because `run:stage` would attach a second firing to the finished stage. A file whose bytes change is a new document, and the old one stays. Handling superseded documents is out of scope. Replaces "No incrementality".
- **D-R.3 Conditions are removed.** The code, the sidecar, the spec field and their docs go. A filter over `manifest.parquet` gives the same answer.
- **D-R.4 The `destinations` field is removed.** It returns in 3.2c with the object-store kind only.

**Merges:**
- **R.1 Remove conditions** (D-R.3). Delete `cloud/workflow_conditions.py` and `store/conditions_output.py`, the labelling step in `cloud/workflows.py` and in the runner, the spec field, the condition digests in `cloud/jobs.py`, and their tests. Docs: the conditions entry in `decisions-operations.md`, `workflows.md`, Requirement 27, CLAUDE.md rows, `architecture.md`, `project-structure.md`. Split into fleet path then spec and modules if it goes over the cap.
- **R.2 Remove `destinations`** (D-R.4). The spec field, the runner's "not acted on" warning, tests. Docs: `workflows.md`, Requirement 27's destinations TO-DO and its Outstanding entry.
- **R.3 Workflows from YAML** (D-R.1). The five commands take a path. An example workflow sits beside `configs/default-isaacus.yaml` and replaces its commented commands. Docs: `workflows.md` (the "Saving one" Python block goes), README command table, D-2.2 deleted.
- **R.4 Incremental firings** (D-R.2). `workflow_dispatch.fire_workflow` lists the prefix and hashes against the run's manifest, `dispatch.batch_specs` numbers on from the last batch, and stage ids and `plan_units` are scoped to the firing. `cloud/retry.py` follows the new stage ids. Docs: `decisions-operations.md` ("A trigger fires a dispatch"; "No incrementality" deleted), `workflows.md`. Split mechanical (firing-scoped ids) then behaviour (hash-and-skip) if over the cap.
- **R.5 Stocktake S-R.**

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
- **3.2c Object-store destination and the delivery model.** `destinations` returns with `object_store` only (a prefix and `include_sources`). A delivery is a DBOS workflow with bounded retries, enqueued after the last stage whether or not one failed. The run record gains `deliveries` (record version 7). A failed delivery labels the run and never stops it. The pre-launch check and the D-2.1 retry line ship here. Docs: `decisions-operations.md`, Requirement 27, `architecture.md`, CLAUDE.md rows.
- **3.3 `pii` as a fleet step.** Docs: `decisions-operations.md` (merge 7b), `workflows.md`, the `pipeline_order.py` comment.
- **3.4 Postgres with pgvector** (per D-3.2). The destination names the environment variable that holds its DSN. An idempotent upsert keyed `(run_id, source_hash, chunk_index, content_type)`, masked text by default and raw where declared, with the layer recorded on each row. A masked destination without a `pii` step, or any pgvector destination without `embed`, is refused before launch. Docs: `contract.md` ("Safe to hand onward"), README installation extras, Requirement 27.
- **3.5 Webhooks** (per D-3.3, events named in the state words). Each POST is a DBOS workflow with bounded retries, signed with an environment secret. Docs: `service-api.md` ("Callbacks and webhooks (consumers poll)"), README (callers poll), `decisions-operations.md` ("Consumers poll"), Requirements 22 and 27.
- **3.5.5 `egress.md` becomes "Egress and destinations"**, with a payload reference (docs only).
- **3.6 Stocktake S-3.**

## Phase 4 — operation (branch)
**Merges:**
- **4.0 Split `cli/cloud.py`.** `jobs` moves to its own CLI module (the file is near the 750-line cap). Commands and output stay the same.
- **4.1 `RunBoard.describe` and its surfaces.** One sentence per run in the state words, over the exact per-stage counts: `womblex jobs --run`, an additive `summary` on `/v1/runs/{id}`, the console. The OpenLineage event (`store/lineage.py`, today always `COMPLETE`) takes its type from the run's state (`cancelled` is `ABORT`, `failed` is `FAIL`). Docs: `contract.md` (the lineage export), Requirements 16 and 22, `service-api.md`, `DESIGN.md`.
- **4.2 The graph-refresh sidecar** (per D-4.2). Define the chunks digest (`store/content_digest.py`), then a sidecar keyed by it. Readers (`analyse/query.py`, the Chunk Inspector's entities overlay) prefer it. Until 4.3, graph-refresh writes both the sidecar and the in-place files. Docs: `contract.md` (minor version), CLAUDE.md rows.
- **4.3 Graph-refresh stops rewriting in place.** Its contract moves from `MutationMode.IN_PLACE` to `SIDECAR`, with `PRODUCER_OF`, `store/retention.py`, `workflow_order` and `workflow_plan` updated. Older runs fall back to the in-place `chunk_index`. `layout` keeps its in-place replace. Docs: CLAUDE.md rows, `project-structure.md`.
- **4.3.5 Docs:** `composable-design.md` (the outputs table's "rewritten in place" and the `graph-refresh` "in place" bullet), Requirement 12, `decisions.md` ("Graph-edge refresh"), `architecture.md`, README.
- **4.4 Stocktake S-4 (final).** Move the design-failures list into `decisions-dead-ends.md`, retire this plan into the decision docs and `CHANGELOG.md`, then delete it.
