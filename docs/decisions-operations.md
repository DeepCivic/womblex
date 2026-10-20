# Operating decisions

The durable "why" behind operating Womblex: distributed execution, workflows, source resolution, deployment and the service surface.

### Distributed execution — stage-in/stage-out, not a filesystem abstraction
Cloud scale-out (2026-06) reuses the existing batch shape rather than rewriting
it. Womblex already shards, checkpoints per batch, and isolates per-doc
failures; the only gaps for horizontal scale were *shared state* and a *safe
claim*. Two deliberate choices:

- **Object storage by stage-in/stage-out, not by threading an abstraction
  through every stage.** A worker pulls a batch's inputs to a local scratch dir,
  runs the ordinary `Path`-based pipeline, and pushes the shard sidecars back
  (`store/remote.py`, fsspec). *Rejected:* making `shard_dir` a filesystem
  abstraction wired through all ~10 `*_shards()` stages + `output.py` +
  `shard_audit.py` — a large, risky refactor that fights the `Path` idiom for no
  gain over copying a handful of files at the job boundary. fsspec's local
  backend means the air-gapped/CPU default exercises the *same* code path with
  no S3 dependency touched.
- **DBOS workflows are the distributed checkpoint, not a second mechanism
  alongside the JSON `CheckpointManager`** (2026-10; D9 of the
  trust-and-recipes plan). Concurrent workers
  writing one checkpoint file would race, so the worker path does **not** use
  `CheckpointManager`. An extraction batch and a downstream stage are DBOS
  workflows on queues; a batch, and each stage unit, run as a recorded step, so
  a unit that finished is not run again after a crash, and re-running `enqueue`
  (deterministic workflow ids, `<run>:batch:NNNN`) is the resume. Step results
  are storage keys, never data. The system database is Postgres for a fleet and
  a SQLite file for a local run. *Replaced:* a hand-rolled Postgres
  `FOR UPDATE SKIP LOCKED` table (`womblex_jobs`) with its own claim, retry and
  stale-recovery code. *Rejected:* Redis/Celery (new infra), Temporal/Prefect/
  Hatchet (a separate server), Procrastinate/PgQueuer (no durable steps or
  events), Absurd (pre-1.0, no schedules in the Python SDK), Dramatiq (LGPL).
  *Behaviour that moved with it:* a worker no longer claims a job and hands it
  back — it joins only the queues it can serve (an extraction queue is named for
  the ingest root, a stage queue for the stage, joined only if the stage's
  models pass the model check). A failed stage is recorded and the stages
  after it still run (see "A run is an arrow"); re-dispatching it returns
  the recorded failure; `womblex retry-stage` forks it instead (D-2.1, below). A stage whose upstream
  sidecar is missing fails without step retries, as retrying cannot produce it.
  Queue settings are stored in the system database and set by the workers that
  serve the queue; the coordinator enqueues a stage by queue name only. A reader
  of a database no writer has migrated sees no runs. `--run-id` and
  `--stale-timeout` left `womblex worker`: DBOS has no per-run claim filter and
  recovers a stopped worker's workflows when its executor id restarts. Owner
  scoping is a workflow attribute, filtered client-side because attribute
  filtering is Postgres-only; two submitters racing to create one run can both
  pass the owner check (the advisory lock is gone) and the first row written
  names the owner. *Tested:* replay of a coordinator that
  caught a failed child (a recovered coordinator re-reads the child's recorded
  failure, runs the later stage once and ends `failed` with the same error; no
  unit runs again), and the board, worker and workflow tests run on SQLite and,
  where CI has a service, on a fresh Postgres database each (`tests/_postgres.py`).
  *Limits:* locally those tests run on SQLite alone unless `WOMBLEX_DB_DSN` names a Postgres server.

The load-bearing invariant: `cmd_run` and the worker call **one** shared
`batch.process_batch`, so local and distributed runs produce byte-identical
shards by construction — the same reasoning as "per-stage and E2E feed the same
engines".

### Trust and recipes: what an external requirements set does and does not change
2026-10. A business-analyst requirements set (FR-1.1 to FR-5.2: high-integrity extraction, recipe-based workflows, destinations and delivery, agent-friendly operation, safety and operability) was written without knowledge of the repository. Most of it mapped onto what already existed: evidence onto element `page` / `bbox` / `order`, content-addressed `source_hash` and the evidence reference; run artefacts and lineage onto the run stamp, `content_digest`, `manifest.parquet`, the run record and `source_index.parquet`; recipes and step ordering onto validated configs, presets, stored workflows and `STAGE_CONTRACTS`; delivery onto egress; agent operation onto typed exceptions and the owner-scoped `/v1` API; sensitive data onto the `sensitivity` footer key and the API's masked-by-default reads. What it did not change:

- **No accepted / flagged / rejected verdicts (FR-1.2).** Measuring quality is the benchmark's job; the rule against quality scoring stands.
- **No schema-driven field extraction.** "Extract invoices" or "summarise contracts" are not Womblex features; evidence requirements apply only to the annotations Womblex produces (money spans, enrichment mentions, entity links, PII spans). Topic annotation of elements is a later roadmap item and, when it lands, is one more annotation sidecar under the evidence reference.
- **Egress keeps delivering unredacted sources (FR-5.1, D10).** Access control stays with whoever runs egress and hosts the bundle (`egress.md`); per-step destinations do not change that.
- **Destinations come in this order (D11):** object storage (built on egress), Postgres with pgvector, then webhook notifications. `httpx`, already locked through the Isaacus SDK, is approved for direct declaration for webhooks; each database or vector-store destination brings its own client in a per-destination extra.

### Trust and recipes: the calls taken before Phases 2 to 4
2026-10, decided before building; each lands with the merge the plan names for it. Two reverse earlier positions: D-3.1 (egress resolves sources only for a local run) and D-3.3 (consumers poll). Womblex is a backend service: where an option exposes raw documents or text (D-3.1, D-3.2), whether to use it is the deploying system's decision, as access to an egress bundle already is (D10).

- **D-2.1 Retry.** A failed stage is retried by forking its DBOS workflow from the first failed unit. The latest attempt counts in the run's rollup, with the original listed `failed` and naming its retry; retry is CLI-only for now; a retry offers to re-run the later stages that ran after the failure. A retry is a launch, so it runs its stage's pre-launch checks again.
  *As built (2.1).* `cloud/retry.py`, CLI `retry-stage --run-id --stage --config [--store]` (`cli/retry.py`). DBOS 3.2's `fork_workflow(id, start_step)` keeps the steps before `start_step` and copies the original's attributes (owner, `workflow_digest`, `ran_after_failed`), so a fork reads as the same stage under the same owner and recipe; it takes no id argument, but honours `SetWorkflowID`, which names it `<run>:stage:<s>:retry:N`, and the fork is put on the stage's queue so a worker that serves the stage runs it. The start step is the first recorded failed step of the latest attempt (a unit, or `plan`); a workflow with none recorded restarts from step 1. Refused before the fork: unknown or still running stage, a stage that did not fail, another owner's run, a config that will not load (`check_registered`, a dispatched workflow's overlay rebuilt as the worker builds it), and `check_dispatch_order` for the stage's inputs against the store. A failed stage's retry that fails is retried again as `retry:N+1`. The rollup (`RunBoard.stats` / `runs`) drops superseded attempts; `list_jobs` keeps them with `retried_by`. *Calls taken:* the offer to re-run later stages is **ordered by hand**: a stage that succeeded having run after a failure (its `ran_after_failed` names it) is retryable once that stage's latest attempt has succeeded, and only while the failed stage's retry is newer than it, and the command prints each such stage's `retry-stage` line, and that retry redoes only the stage's affected units (Phase R, D6). *Rejected:* `resume_workflow` (it does not restart a workflow that recorded its failure), a `--rerun-later` flag with a waiter workflow (a new workflow for an ordering the operator already controls), and a print-only offer (names the problem with no way to act on it). *Kept as is:* the coordinator's own row stays `failed` in the job list (its error names the stages that failed when it ran); forking it would re-enqueue stage ids that exist, so it is the record of the original dispatch. The run's rollup counts it by its stages' latest attempts (Phase R, D5).
- **D-2.3 `schedule-workflow` refuses** a workflow its firing would refuse, rather than warning and failing per firing.
- **D-2.4 A workflow guide** is its own document, `docs/workflows.md`.
- **D-2.5 A doc-currency test** holds the README command table to the CLI parser and the CLAUDE.md module table to `src/`, so drift fails CI in the PR that causes it rather than waiting for a stocktake. *As built (2.3.7):* `tests/test_doc_currency.py`. A command is listed if the README table names it anywhere (a deprecated alias sits in a description); a path CLAUDE.md names must exist; and a `src/` module (not `__init__`, `cli/` or `ui/`) with no row fails unless it is on a recorded list of 28 modules the table has never listed, which can only shrink. *Rejected:* a complete table now (77 rows' worth of module descriptions in one merge) and a forward check alone (a new module would go undocumented).
- **D-3.1 Egress fetches sources from an object store.** A run ingested from an object store resolves each source from that store by the manifest's `source_relpath`, checked against `source_hash`, with the same per-document status as local resolution; `include_sources` stays the caller's choice per export or destination. *Rejected:* corpus-only object-store destinations, leaving consumers to fetch sources from the ingest bucket themselves.
- **D-3.2 pgvector holds either text layer.** Masked `clean_text` by default, which needs `pii` in the workflow (refused before launch otherwise); raw chunk text where the destination declares it. Each row records its layer. *Rejected:* masked only.
- **D-3.3 Webhooks** are HMAC-signed with an environment secret and send five events: extraction settled, stage done or failed, run done, delivery done or failed. No per-document or per-batch events. An undeliverable webhook is retried a bounded number of times, then recorded on the run; it never stops it.
- **D-3.4 Dependencies.** `httpx` is declared in core and pgvector's client sits in a `[pgvector]` extra, landed as their own merge.
- **D-4.1 Plain-language status.** One sentence per run, derived from the exact per-stage counts, which stay. Its words are one fixed set (waiting, extracting, processing, finished, finished with failures) shared with the webhook events and the lineage event types. Shown by `womblex jobs --run`, an additive `summary` on `/v1/runs/{id}`, and the console.
- **D-4.2 Graph-refresh writes its own sidecar,** keyed by the chunks' digest, and stops rewriting the enrichment sidecars. Enrich's files stay as enrich stamped them, and a reader can tell when the refresh no longer matches the chunks. *Rejected:* a version in the file name, which says that the file changed but not which chunks it matches.

### A workflow is a file; scheduling stores a copy (Phase R, D1 and D10)
2026-10; D8 of the trust-and-recipes plan, as repaired by Phase R item R.3. A preset is a partial config applied once per run; it holds no step sequence or inputs. A workflow references a preset and adds those, in a YAML file (`cloud/workflow_file.py`) that is the master copy: `validate`, `plan`, `run-workflow`, `dispatch-workflow` and `schedule-workflow` each take its path, and unknown keys are refused. Its digest, over the parsed spec, is stamped beside `config_digest`. Dispatch stores nothing, since the run receives the whole workflow. Only `schedule-workflow` stores a copy, in the database DBOS uses, because its firings read it: it checks the firing, saves the copy and syncs the schedules in one action (`workflow_schedules.schedule_workflow`), putting the previous copy back if the sync fails, so a stored copy never disagrees with its schedules. Reverses Phase 2's code-only creation (D-2.2, deleted): a workflow could only be saved through `WorkflowStore` in code, and a stored copy could change without its schedules.

- *Rejected:* **extending the preset** with steps. One schema and no new storage, but it mixes how a stage behaves with what runs, and leaves nothing for a standing trigger to act on.
- *Rejected:* **a workflow kept only in the database.** A schedule needs a copy its firings can read, but the operator edits and reviews a file; keeping both, with the file the master, gives each its job.

### A workflow runs locally: validate on save and load, refuse before work, record after
Phase 2 merge 2 (`cloud/workflow_runner.py`). Three calls the merge-1 review left open.

- **Validation sits at both ends.** `WorkflowStore.save` runs `check_spec` so a spec built without `validate_workflow_spec` cannot be stored unchecked, and the runner runs it again on load, because a row can outlive the rule it passed (a stage renamed or removed). The cost is a few dictionary checks. An unknown preset name still passes the save-time check: it may be an operator-saved preset the store cannot read, so the runner resolves it and refuses there.
- **A preset and a workflow's `config` merge key by key** (merge 2 replaced whole sections). Mappings merge recursively, a list or scalar replaces, and `null` drops the key so the field takes its schema default (not a literal `None`, which a non-optional field would refuse). Section-replace was predictable, but `enrichment: {model: x}` silently reset every other key the preset set (`enabled` included) to its default, and warning on dropped keys would have left that trap in place. The cost: a key reverts only by naming it `null`, and lists are not concatenated. One function (`combine_config`) serves the save-time overlay check and the runner. `POST /v1/runs` is different and stays so: its preset replaces the request's config outright, with no workflow overlay. The combination is validated as a full config, with `dataset.name` set to the preset name so the run stamp records the recipe.
- **A failed step does not stop the steps after it.** Everything that can refuse does so before the first step: spec, preset, config, and each step's preflight, Isaacus availability and model check (`prepare_stage_context`), so a workflow that cannot complete spends nothing. Once a step has run, halting would discard paid work, so it does not: a failed step (a `MissingOverlayError` refusal, which `run_stage_local` logs and counts, or any exception) is recorded with its error, later steps still run, and each carries `ran_after_failed` naming the failed steps whose output its inputs may lack. The result and a non-zero exit code are the account, as `mask_status` and `evidence_check` are for rows. Steps run in list order; checking that order across the whole list is merge 3.

### `womblex validate` is the runner's pre-run path, read-only
Phase 2 merge 4 (`cloud/workflow_validate.py`). The plan left the report's shape and the model-check scope open.

- **One path, not a copy.** `workflow_runner.prepare_workflow` holds everything before a first step; `run_workflow_local` and `validate_workflow` both call it, so a workflow validate passes is not refused by the run for a pre-run reason, and the codes are the same.
- **Every failed step is reported.** The runner used to stop at the first step whose preconditions failed; it now collects all of them before refusing, since nothing has spent anything and a fix-one-rerun loop costs a model check each time. The refusal message joins them.
- **Refusal codes beside the order codes.** `RefusalCode` (`spec_invalid`, `preset_unknown`, `config_invalid`, `step_order`, `step_precondition`) is the stable top-level code; `step_order` carries the `OrderCode` issues.
- **The model check runs as configured.** At `load` it checks an Isaacus model's connectivity, as a run's does, and at `smoke` it also sends the billed request a run's would, warned first; `--models-check off` makes validate fully offline. Validate never runs a step and never writes.
- **Without `--shards` only the order among the steps is checked**, and the report says so (`shards_checked`).

### `womblex plan` counts, it does not price
Phase 2 merge 5 (`cloud/workflow_plan.py`).

- **Counts come from the stages' own request builders**, so no packing logic can drift, in each provider's own tokens (Phase R, D12): the enrichment provider's declared `token_counter`, and the embedding provider's (`EmbeddingLimits.token_counter`; the Isaacus embedder declares the Kanon-2 tokeniser). A provider that declares no counter has its texts and requests counted, not its tokens.
- **Every paid call is counted (Phase R, R.10).** AI chunking's chunk-time Enricher calls are (`chunk_stage.plan_ai_chunking`: the stage's own inputs and reuse test, and semchunk's own rule — a narrative no persisted Document is reused for is enriched only when it holds more tokens than the chunk size, one call per million-character prechunk; tables chunk in token mode and are never enriched), and a step that calls no paid service is marked free.
- **Free earlier steps are produced in a scratch symlink directory** (`normalise`, `spellfix`, `chunk` sans AI chunking), so a fresh-extraction workflow gets a real embed count. An input only a paid step produces is `pending`, never guessed.
- **No model probe and no rate card.** Plan stops at `check_workflow`; `--assume-price` is the caller's rate, labelled an assumption. `skip` is only what is knowable offline (`layout` by fingerprint).

### A run records its whole workflow (Phase R, D11)
2026-10, Phase R item R.9. Only the digest was recorded, and once the workflow file changed nothing said what that digest had been. `workflow_context` now carries the workflow's name and full spec beside its digest, and `sidecar_footer` writes them as `womblex.workflow` (JSON) on every file a workflow step writes, locally and on the fleet, where the dispatch carries the spec. The run record (version 7) lists each observed workflow's `name` and `spec` beside its digest and stages; files written before the key give `null` for both.

### The workflow digest is a contextvar, observed in the record
Phase 2 merge 6. `sidecar_footer` is the one place every stage's writer already takes its stamp, so the digest rides it through `workflow_context` rather than a parameter on a dozen writers. A contextvar, not a global, so two workflows in one worker process cannot stamp each other's files. It is written even where the shard is unstamped (it is a fact about the write; the run id is not claimed). The run record lists workflows *observed* in files, like stages, never those merely stored.

### A dispatched workflow's config travels as a workflow argument
Phase 2 merge 7a. A worker's config is process state, so a workflow dispatched to the fleet would otherwise run under whatever config each worker was started with. The resolved overlay (`combine_config(preset, spec.config)`) goes to the coordinator and each stage as a plain-JSON argument, and the worker builds the config with the same `config_from_overlay` the local runner uses (recipe sections from the overlay, schema defaults elsewhere, the worker's own `paths`), so local and fleet stages run under identical settings. Not the worker's config merged under the overlay: that would make a stage's behaviour depend on which worker claimed it. The workflow digest is part of the coordinator id, and a workflow's stage ids are scoped by it (`dbos_app.stage_scope`), so a *changed* workflow dispatched over a run runs its own stages rather than returning another's recorded result.

### Dispatching a workflow refuses what the coordinator would quietly change
Phase 2 merge 7b. Two fleet-specific refusals, because the coordinator is not the local runner: it sorts stages by `PIPELINE_ORDER` (so a workflow listed in another order would run in an order it did not declare: `order_not_pipeline`), and it dispatches only `DOWNSTREAM_STAGES` (so `pii`, `quality`, `layout` are `stage_not_dispatchable`; `pii` is never on the queue). The third, `run_conflict` (a run already worked under another workflow digest), is gone (Phase R, D2): stage ids are scoped by firing or digest, so nothing returns another workflow's result. Re-dispatching the same workflow is the same coordinator and attaches. `--dry-run` runs every check and writes nothing: no row, no schema, and no local database file where none exists.

### A run is an arrow: refused before launch, never stopped after
A run is launched, then it lands. Everything that can refuse does so before launch (spec, preset, config, step order, each step's preflight and model check); once launched, nothing stops it — a failure is recorded where it happened and the rest of the run carries on, labelled with what it follows. This is why the pre-run checks matter, and why no check is added after launch.

- **The fleet carries on as the local runner does.** A unit that fails (once its retries are spent) does not stop the stage's other units, and a failed stage does not stop the stages after it; each later stage records `ran_after_failed`. The stage, and then the coordinator, fail last, naming what failed, so the board shows them failed and the run record lists them in `partial`. A batch missing an input only that failure would have written is not ready for that batch alone. A strict input missing from any batch still refuses that one stage as a whole, as `require_overlays` does locally and `run-stage --store` does, so a stage never mixes text layers.
- *Rejected:* **a failed stage ends the coordinator** (Phase 0), chosen so later stages would not report "not ready". It stopped a launched run over one batch, and the local runner already carried on. Engine errors (nondeterminism, recovery) still propagate; only the work's own failures are carried.

### A fleet run writes its own manifest (Phase 3, item 3.2a)
2026-10. A distributed run had no manifest until someone ran `womblex finalize`, so a fleet run's `manifest.parquet` and run record depended on an operator remembering a step. The coordinator now writes it as its last step, after the stages, failed ones included, through the body `finalize` runs (`cloud/finalize.py`): the shard manifests staged in, every footer read in place, the published checksums read off the board. A manifest it cannot write does not stop anything (the stages have run); the coordinator fails last naming it and the command that writes it. `womblex finalize` stays and is idempotent: an extraction-only run has no coordinator, and a stage retried after the coordinator finished changes files the manifest's record lists, so `retry-stage` prints the `finalize` line to run once the retry succeeds.

### Workflow conditions are removed (Phase R, D3)
2026-10, Phase R items R.1a (the fleet stops labelling), R.1b (`run-workflow` stops labelling) and R.1c (the field goes). A condition labelled each document `holds`, `fails` or `unevaluable` against a manifest column, in a `*.conditions.parquet` sidecar that nothing read. Every column it labelled is in `manifest.parquet`, so a reader filters the manifest instead. Reverses Phase 2 merges 8 and 9a.

### A workflow names its input store (Phase R, D9)
2026-10, Phase R item R.3b. A firing used to read whichever ingest store its worker was started with, so the same workflow listed different documents on different workers. `input_binding.store` names the store; a schedule puts its firing on that store's `firing_queue`, which only a worker whose ingest root it is serves, and the firing refuses (before any row) a workflow that names none. Credentials still come from the worker: the workflow names where documents are, never a secret. `schedule-workflow --ingest` is gone, since the workflow says where its documents are.

### Workflows have no owner (Phase R, D13)
2026-10, Phase R item R.2b. Nothing set a workflow's owner, so the field did nothing: it leaves the workflow table, the schedule context, the firing's rows and its prefix check (`prefix_not_owned` is gone). A workflow's run carries no owner and is admin-only, like a CLI run. An owner returns if the API ever creates workflows. Reverses owner-scoped stored workflows (Phase 2 merge 1) and the owned firing's `<owner>/` prefix rule (merge 9b).

### Workflow destinations are removed until they deliver (Phase R, D4)
2026-10, Phase R item R.2. `destinations` was stored on a workflow and nothing delivered to it, so the field is removed with its TO-DO; Phase 3 brings it back one destination kind at a time, each in the merge that delivers it. A stored row that still holds destinations loads with them dropped and a warning.

### A retry redoes only affected work (Phase R, D6)
2026-10, Phase R item R.8. Retrying a stage that ran after a failure forked it from step 1, so it redid every batch and re-paid for each. Now that retry is a new attempt (`RunBoard.rerun_stage`, still `<base>:retry:N`, the original's attributes less `ran_after_failed`) over its affected units alone: the bases each failed stage's attempt it ran after planned and did not finish (`RunBoard.unfinished_bases`, read from that attempt's recorded `plan` and `unit:` steps; every base where it recorded no plan). A failed stage is still forked from its first failed unit. `retry-stage` prints the units a re-run covers.

### Work goes only to workers checked for it (Phase R, D8)
2026-10, Phase R item R.7a (routing). A worker checked only the models its own config enabled, yet claimed any stage's work, and once runs carried their config (D7) a run could name models a worker never loaded. Now extraction and stage queues are named for the models their work needs under the run's config (`model_check.models_key`: a digest of slot, name and options; `""` for a stage with none, so `normalise`, `money`, `link` and `graph-refresh` keep their plain queues). A worker checks the local models of every stage in its config, enabled or not, and an Isaacus model only for a stage its config enables, and joins a scope's queue keyed by its own config's models only when that check passes (a stage whose Isaacus model it did not check is not joined), so work reaches only a worker checked for exactly those models. It also joins the unkeyed queues, which hold work enqueued before runs carried a config. Each worker records the queues it serves in `womblex_workers` (`cloud/worker_registry.py`), refreshed each minute and removed when it stops. A disabled stage costs no request at worker start.

R.7b adds the refusal. Every dispatch, before any row, names the queues its work needs (`worker_registry.needed_queues`: extraction, each stage and the coordinators, keyed by the run's config) and refuses what no live worker serves (`require_served`, `NoLiveWorker`): the API with 409, the console and CLI with their usual refusal, `dispatch-workflow` and a firing with `no_worker`, and `retry-stage` with `no_worker`, so a retry runs every check a dispatch does. Live is a registry row refreshed within the board's staleness threshold (900 s). A deployment that scales its workers to zero is refused until one is running.

### Every fleet run carries its config (Phase R, D7)
2026-10, Phase R item R.6. Only a workflow's stages ran under the config they were dispatched with; a run from the API, the console or `enqueue` / `enqueue-stages` ran under each claiming worker's own config, so one run could behave differently per worker, and the API's config chose which stages ran and nothing else. Now every dispatch carries its validated config, less `paths` (`workflow_runner.config_payload`), as an argument of each extraction batch and of the coordinator and its stages; the worker rebuilds it with its own `paths` (`config_from_payload`), the construction `config_from_overlay` uses for a workflow. A worker's config supplies only its store, paths, credentials and model locations. `womblex enqueue` without `--config`, an API run with neither preset nor config, and the console's extraction press without one carry the default config. A row enqueued before this change carries none and runs under the worker's config. Reverses Requirement 22's "selects downstream stages only".

### Run status takes each unit's latest attempt (Phase R, D5)
2026-10, Phase R item R.5. A run whose failed stage was retried successfully, or whose failed firing was dispatched by hand, still read `failed`, because the failed coordinator or firing row kept counting. The rollup (`RunBoard.stats` / `runs`, so `womblex jobs`, `/v1/runs` and the dashboard) now counts each row by its latest attempt (`jobs._rollup`): a superseded stage attempt is dropped, a failed coordinator whose stages' latest attempts all succeeded counts `done`, and so does a failed firing whose stages have since been dispatched. The job list keeps every row's own status. Reverses the rule that the original failure keeps counting.

### A scheduled workflow appends to one run (Phase R, D2)
2026-10, Phase R item R.4. A firing used to re-extract its whole prefix into a fresh run and re-pay for every document. Now a workflow's firings append to one run named for the workflow. Each lists its input prefix and queues only the documents the run lacks: a key is seen once the run has queued it with its current fingerprint (`RemoteStore.fingerprints`: the store's ETag or checksum, else size and modification time), recorded on the batch rows that queued it, so a changed file is a new document and its earlier version stays (pruning is out of scope). The firing's batches are numbered after the run's last, and its steps run over those batches only, under stage ids of its own (`<run>:fire-<UTC time>:stage:<s>`), with the workflow as it stood at that firing, so an edit applies to new documents only. A seen key whose fingerprint moved is hashed against the run's recorded `source_hash`; one whose bytes are unchanged is not queued, and the firing records its new fingerprint in its output (`unchanged`, read back by `RunBoard.seen_inputs`), so no later firing hashes it again. A firing with nothing new runs nothing and logs that. A replayed firing reuses the batches it queued. Firing queues run one firing at a time per input store, so two firings never number the same batch. `retry-stage` reaches a run's failed firings newest first. The refusal of a run already worked under another workflow digest (`run_conflict`) is removed.

- *Rejected:* **hashing every listed document** to tell new from seen. It downloads the whole prefix at every firing; the store's own fingerprint answers the same question from the listing.

### A trigger fires a dispatch
Phase 2 merge 9b (#197, re-landed after it merged into a stacked base that had already merged). A DBOS schedule per trigger, firing `womblex.run_scheduled` on the coordinator queue every worker serves. The firing is its run's launch (`check_firing`, then the batches, then the dispatch): it appends to the run named for the workflow (Phase R, D2), `input_binding.ingest_prefix` listed through the same `supported_keys` the console uses.

- **Check first.** A dry-run dispatch, a safe input prefix and at least one document are checked before any row, so a firing that cannot complete queues nothing and fails with the refusal's code.
- **No backfill** (`automatic_backfill` false).
- **Deleting a workflow file does not unschedule it.** `schedule-workflow <name> --remove` does, deleting the stored copy too.
- **A firing is a row of its run.** DBOS names a scheduled workflow `sched-…` with no attributes, so the firing runs as a child `<run>:trigger:<UTC time>` carrying the run's id. Two triggers firing at one instant share the firing, so the second attaches to the first and runs nothing. The firing body lives in `workflow_dispatch`, beside the dispatch it composes; `workflows.py` holds only the DBOS wrappers.
- **Scheduling checks the firing (D-2.3).** `schedule-workflow` calls `workflow_dispatch.check_firing`, the function a firing runs first (dry-run dispatch, a safe input prefix, at least one document), against `--store` (or its environment variable) and the workflow's input store and refuses before it writes a schedule; a workflow with no triggers needs neither. *Rejected:* a second copy of the checks in the CLI, and warning at scheduling while the firing fails each time. A firing can still refuse later, because the ingest prefix or the run may change after scheduling.
- **A firing that fails after its batches are queued leaves them running and the run `orphaned`.** Every check has passed by then, so the window is the dispatch write after the batches. The batches run on, as nothing stops work once launched, and the run reads `orphaned` (`RunBoard.runs`, `/v1/runs`, `womblex jobs`): a failed firing's trigger row and batch rows, and no coordinator for that firing. The label is read from the rows, not written, so it holds when the database that failed the dispatch could take no label either, and it clears once the stages are dispatched; the logged error names `dispatch-workflow <file> --run-id <run> --firing <time>` for that. A replayed firing refuses its own cancelled batches (`batches_cancelled`): it would attach to them, see extraction settled and dispatch stages over shards that never came. *Rejected:* cancelling the batches (it stops work already launched and leaves partial shards), and dispatching first (the coordinator would find no batch rows, read extraction as settled and run the stages over nothing).
- **`cancelled` is a board status.** DBOS's `CANCELLED` was read as `failed`, so a withdrawn job was indistinguishable from a broken one; it now reads `cancelled`, `/v1/runs` gives a finished run with a cancelled job the state `cancelled`, and the console shows the skipped pill. A cancelled stage stays retryable, as it was while read as `failed`.
- **Boundaries kept at the close of Phase 2 (stocktake S-2).** Retired: the retry deferred by the stored-workflows phase (D-2.1), `schedule-workflow` checking only the owner prefix (D-2.3), a firing that fails after its batches are queued, the missing Postgres tests and the unproven replay (#192). Kept deliberately: a deleted workflow file's schedules stay until `--remove`; no backfill; `womblex_workflows` is created if absent with no versioned migration; no test waits for DBOS's scheduler to start a firing itself; two submitters racing to create one run can both pass the owner check.
- **Limits.** The schedule tests run on SQLite and, where CI has a service, on a fresh Postgres database each. `--status` on a Postgres database whose DBOS tables are not yet created (#200's CI failure on the missing `workflow_schedules` table) now prints nothing, as a missing SQLite file does (`list_schedules` reads a database no writer has migrated as empty). No test waits for DBOS's scheduler to start a firing on its own: the tests set a cron that never comes due and enqueue each firing with the workflow name, queue and context its schedule holds.

### The remote shard-stage runner generalises `finalize`
Extraction went distributed in 2026-06; everything downstream of it stopped at
the local filesystem, so a store-backed run had to be pulled down before any
`*_shards()` stage could touch it. `womblex run-stage` (2026-08) closes that,
and it does so by generalising a shape that already existed rather than
inventing one. `finalize` lists one sidecar class under the run's shard prefix,
downloads it to a temp dir, calls an unchanged library function against that
`Path`, and uploads the result. The runner does exactly that, per batch, for the
downstream stages — which keeps the stage-in/stage-out decision above intact:
**no `*_shards()` signature changed and no filesystem abstraction was threaded
through them.**

Decisions inside it:

- **Stage contracts are data, and two fields are functions of *config*, not of
  stage name** (`cloud/stage_contracts.py`). `chunk` reads
  `*.enrichment_doc.parquet` only when `chunking.chunking_model` is set; `chunk`
  / `enrich` / `money` read the overlay sidecar named by `processing.text_source`
  (with `money.text_source` outranking it); `pii` writes `*.clean_text.parquet`
  only when `write_clean_text`; `enrich` writes `*.enrichment_doc.parquet` only
  when persisting. The Isaacus gate is config-aware for the same reason: `chunk`
  needs the API only under AI chunking (`requires_isaacus_api(config)` returns
  `chunking_model` truthiness), while the static `needs_isaacus_api=True` stays
  the coarse display answer for the composer graph. Resolving any of these from
  the stage name would download the wrong files and — worse — let the
  output-exists skip fire on an incomplete set, or refuse a keyless offline
  chunk run.
- **A selected-but-absent overlay is a hard failure, not a fallback.**
  Falling back to verbatim would publish a sidecar built from the wrong text
  layer and look like success. Both paths check every batch before processing
  any, so a gap in a late batch cannot leave earlier ones published: the
  runner refuses the stage, and locally chunk, enrich and money raise
  `MissingOverlayError`. The `spellfix → normalised` chain is exempt: it
  passes `warn_if_missing=False`, so absence there is legitimate.
- **Discovery is separate from required inputs.** Bases come from
  extraction-role siblings only; `*.form_fields.parquet` makes a base
  discoverable but is read by no stage, so it is never downloaded. A
  `*.chunks.parquet` with no extraction sibling is not a batch.
- **All declared outputs are verified before any is uploaded**, so the *stage*
  can never leave a half-written set behind. That is a pre-upload check, not an
  atomic multi-object write — object stores offer no such primitive, and a
  transport failure part-way through the uploads can still leave a partial set.
  What covers that case is the skip rule, not atomicity: skip fires only when
  **every** declared output is present, so a partial set never reads as complete
  and the next run redoes the base and overwrites. The runner is therefore
  idempotent and re-runnable as more batches land, exactly as `finalize` is.
  Nothing-to-do exits non-zero — a typo'd `--run-id` must not read as success in
  `run-stage … && next-step`.
- **`manifest` is excluded**, because `finalize` is it. Duplicating it in the
  runner inventory would give one job two commands.
- **`graph-refresh` is modelled explicitly as an in-place mutator.** Its outputs
  (`*.enrichment_entities.parquet`, `*.graph_edges.parquet`) are a *subset* of
  its inputs, so it breaks the disjoint required-input/produced-output model that
  output-exists skip depends on. Rather than exclude it, the contract carries
  `MutationMode.IN_PLACE`: never skipped, both sidecars re-uploaded
  unconditionally, correctness resting on the stage's existing idempotency
  (recomputing from the same character offsets yields the same `chunk_index` and
  the same `mentioned_in` edges). Atomicity is the same all-or-nothing publish,
  so a mid-batch failure leaves the store untouched.
- **`quality` is run-scoped, not per-batch.** Per-batch execution is not a weaker
  version of the same output; it is a differently-meaning column with the same
  name. `_cluster_ids` numbers clusters `0,1,2,…` in first-occurrence order
  *within a pass*, so batch-local passes both miss cross-batch duplicates and
  emit colliding ids that mean different things in different files. It gets
  `StageScope.WHOLE_RUN` and stages every batch's chunks in one pass —
  text-only sidecars, and the same download-compute-upload shape as `finalize`,
  just at run scope. It is also the one stage with no `CheckpointManager`, by its
  own design.
- **Runtime clients fail explicitly.** `chunk_shards` under **AI chunking**
  (`chunking.chunking_model` set) calls the enricher API, so it warns and
  returns `ChunkStageResult(0, 0, 0)` when Isaacus is unresolvable — remotely a
  clean-looking run that published nothing. The contract's config-aware
  `requires_isaacus_api(config)` gate makes the runner refuse *that* case
  (`prepare_stage_context`) and exit non-zero. Plain token chunking (no
  `chunking_model`) sizes chunks with the vendored tokeniser and runs offline,
  so it needs no API and is *not* gated — `chunk_shards` instead checks the
  tokeniser resolves locally (`tokenizer_available`). `enrich` / `embed`
  always need the API. `link`'s reference register is a worker-*local* file,
  not a store object, so it gets a preflight that resolves the path.
- **Checkpoint staging is opt-in and single-invocation.** Checkpoint state is a
  directory, not a shard suffix, and per-base temp staging gives it nowhere to
  live. The default is the runner-level output-exists skip — the right
  distributed idiom, and consistent with "concurrent workers writing one
  checkpoint file would race". `--stage-checkpoints` stages the dir in/out for
  the single-runner case.

*Known cost:* per-batch `chunk` loses the cross-batch `semchunk` memoise cache
(`chunk_shards` builds one chunker and reuses it across bases). Output is
byte-identical; only warm-cache speed is lost.

### A run ingests what is directly under the location it is given

Local and distributed enumerated a corpus differently. `womblex run` (and
`redact`, and the back-compat E2E chunk path) listed the immediate level of the
input root; the enqueue path in `cli/cloud.py` and the console's in
`ui/execute.py` listed recursively. A corpus laid out under dated or per-agency
prefixes therefore extracted in full through the workers and yielded nothing at
all locally, reported as "no supported files found" with no indication that
nested documents had been passed over.

Parity between local and cloud is the design invariant, so both conform to one
rule: `select_supported` in `cli/_shared.py` decides what a location yields, and
every entry point calls it. A corpus with supported documents in subdirectories
is refused — by the local CLI with a non-zero exit, by the console with a 400,
naming a subdirectory and its document count. A subdirectory holding no
supported document is ignored, so a corpus alongside `.git` or a notes directory
still runs. The refusal fires on a mixed layout too: ingesting only the top level
is a partial ingest, which is the same defect as ingesting nothing.

*Rejected: recursion everywhere.* It is the smaller diff and it breaks nothing
that works today, but it makes the location a run was pointed at no longer
describe what the run consumed — an operator who points at a parent directory
silently ingests every corpus beneath it. Refusal keeps the location honest and
is recoverable in one command.

*Rejected: a `--recursive` opt-out.* It reintroduces the two behaviours the rule
exists to remove, and a resolver reading back a run's provenance would have to
encode both again.

The cost is borne by the console, which has no ingest-prefix field on any screen:
an operator whose documents sit a level down cannot repoint at the subdirectory
from the UI and must change the configured ingest location. The preflight count
reports the refusal and its reason rather than a bare zero, so the reason is at
least visible; a prefix field is the follow-up if the layout turns out to be
common.

### Resolving a row back to its document — path first, hash second

`store/source_resolver.py` is the return leg of the provenance chain: a row's
`source_hash` back to the file it describes. Two things shaped it.

*Path first, then hash.* The obvious build is an index — scan the corpus, hash
every document, look the value up — and the requirement was written expecting
one. But a manifest row already records the path under the ingest root (P1), so
the common case, a corpus still sitting where the run found it, is answered by
one file read. The index is built only when the path has not answered, which is
what makes the resolver usable against a corpus of thousands rather than a tool
you run once and wait on. Both orders verify the bytes, so neither trades
correctness for the speed: the difference is only how much of the corpus is read
to reach the same answer.

*Every call returns a result, and the failures are told apart.* A resolver that
returns nothing for a row a consumer holds is indistinguishable from a corpus
that has been tampered with, so there is no empty return and no exception: a
`Resolution` carries a status and a reason. The statuses exist because the ways
resolution fails are not equally bad. A re-saved corpus — same content, new
bytes, which is what a bulk re-export or a virus scanner produces — reports the
mismatch *and the path it found*, because knowing the file is there and has
changed is worth more than being told nothing matched. A document moved within
the corpus is found by hash. Only a genuine absence is `not_found`.

The two hash bases are distinguished rather than conflated. A file ingest hashes
source bytes; the pre-extracted records ingest hashes record id plus text, so
there is no file to hash at all. Those rows are *declined by name*, naming
`*.provenance.parquet` as the back-link that does answer for them — a row that
is out of scope is accounted for, not missing, and reporting it as missing would
have made every records corpus look corrupt.

*Rejected: a second corpus walk.* The resolver could enumerate the corpus its
own way, and for a while that would agree with what a run ingests. It would stop
agreeing the first time either changed. Both paths go through `select_supported`
instead — the rule the entry above settled — so a document the resolver returns
is one a run could have processed, and a nested corpus is refused here exactly
as a run refuses it. That is also the answer to what the entry above left open:
a resolver reading back a run's provenance encodes one enumeration, not two.

*Rejected: inferring the corpus location.* An object-store root, or a directory
whose rows name several roots, raises at construction rather than resolving
nothing row by row. Guessing a local path for an `s3://` root would resolve
against whatever happened to be there. `--root` is how an operator names a local
copy, which is also what makes a moved corpus re-resolve.

### Isaacus deployment is environment-shaped, and endpoints are declared, not assumed
SageMaker support (2026-08) lets an air-gapped AWS account serve Kanon-2 with no
API key and no egress. Three choices behind `utils/isaacus_client.py`:

- **Environment, not YAML.** Which deployment answers a call is a property of
  *where Womblex is running*, like `ISAACUS_API_KEY`, `WOMBLEX_STORE_URI` and
  the AWS credentials themselves — not of the dataset being processed. So
  `ISAACUS_SAGEMAKER_ENDPOINTS` is read where the client is built and nothing
  is threaded through the stages. *Rejected:* an `isaacus:` config section,
  which would have to reach every call site (embed, enrich, chunk, run-stage)
  to configure something no config file should differ on.
- **The user declares the endpoint topology; Womblex assumes none.**
  Subscriptions are per model *and* there is a universal one, so deployments
  legitimately range from one endpoint serving everything to one per feature,
  mixed. The spec grammar (`name[@region][=model|model|...]`, no `=models`
  meaning "serves all") mirrors `IsaacusSageMakerRuntimeEndpoint` exactly, so
  the integration's own router — not a Womblex re-implementation — resolves
  each request. *Rejected:* probing SageMaker to discover what is deployed (an
  API call, IAM surface and failure mode in exchange for guessing).
- **Reachability is checked before the first request, not at it.** Stages pass
  the model ids they will call, so an undeployed subscription fails with the
  model named and the served set listed — not the integration's `No SageMaker
  endpoints registered for model` mid-batch, nor (when no region resolves) a
  bare `AssertionError` that vanishes under `python -O`. A malformed spec
  raises for the same reason: falling back to the hosted API means reaching for
  a key an air-gapped install does not have.

Only *API* calls move; the vendored tokeniser makes offline chunking identical
in both deployments.

### Container images are built in CI and stamped with the commit they came from
Two Dockerfiles existed that no workflow built, so a change that broke an image
was found on deploy; and the image build context excludes `.git`, so a
containerised run resolved its commit as `unavailable`. Three choices behind the
`images` CI job and the stamp block at the foot of each Dockerfile:

- **The commit is supplied, never discovered.** `SOURCE_COMMIT` is a build
  argument. A build context is a copy of a work tree rather than one, so a build
  that shelled out to git would be reading nothing — and putting `.git` into the
  context to make it readable would ship the repository's whole history in the
  image to answer a question one argument answers. *Rejected:* un-ignoring
  `.git`; deriving the commit from the package version, which is the same string
  for every commit between two releases.
- **Stamped after the install, not before.** The stamp is written into the
  *installed* package — the copy `store/build_info.py` actually asks — as the
  last layer. Writing it into the source tree before `pip install` would put a
  per-commit value underneath the dependency layer and re-resolve the whole
  vision/ML stack on every push, which is the difference between a check that
  runs on every change and one nobody waits for.
- **A published image must name its commit; a local build need not.**
  `REQUIRE_STAMP=1` fails a build given no commit. The publish path sets it, and
  a local build does not (they live in the compose override) — so an operator's
  local image still reports `unavailable` with a reason and stays frictionless,
  while a published image that could not say what it was becomes a build failure
  rather than an artefact making an honest-looking claim about nothing. CI builds
  both ways and asserts each: the stamped image's label and resolved commit
  against the commit, and an unstamped published build against a non-zero exit.

The label and the stamp carry the same fact for different readers — `docker
inspect` reads the image, `build_info` reads from inside a run. Neither is a
digest: what identifies a published *artefact* is deferred to the publication
that does not exist yet.

**A run's image digest is injected, and the record says so.** A digest is
content-addressed after the push, so it cannot be baked into the image it
names, and a container cannot read its own labels from inside — so the run
record's `image` block is operator-honest rather than self-verifying, and its
`partial` list says that even when the digest is present. That is the same
shape the declared-services entry already has: the record describes the
environment it was written in and never claims to have verified it. *Rejected:*
recording the image's tag as its identity, which is a pointer someone can move
presented as an artefact; inferring the container from cgroups, which vary by
kernel, runtime and orchestrator and would turn a plain fact into a guess — the
runtime's own marker file is asked instead, and an unrecognised runtime reports
"not containerised" rather than a claim about an image nobody named.

**The publication is GHCR, on a release tag, recorded as a digest.** GHCR
because it takes the workflow's ambient token for this repository's own
packages, so the "no credential is committed" property holds by construction
rather than by care. A release tag because that is when a version exists to
publish under; a manual dispatch publishes an `edge-<sha>` tag instead, so the
path can be exercised without moving `latest`. And a digest in
`deploy/images.env` rather than a tag in a deployment's head, because a tag is
a pointer someone can move and the artefact is what a run needs to name.
*Rejected:* a registry needing a provisioned secret, which would have put a
credential in repository settings to satisfy a requirement about not having
one; recording the digest only in release notes, which leaves the pin outside
the repository and manual. Publication is a separate workflow from the PyPI
release for a concrete reason — that one runs under Trusted Publishing and a
re-run would attempt a duplicate upload, so it cannot carry the
`workflow_dispatch` this needs.

**Which services that publication would cover is settled and written down.**
`docs/deployment-images.md` enumerates every compose service with a verdict
each, and a test holds it to the files. Two findings shaped it: the five
build-from-source services collapse to two images rather than five, because
four of them build the same Dockerfile and differ only in the command they run;
and a third-party service on a moving tag is the same unidentifiability as a
local build, so the audit covers those too and records that two of the three
carry no tag at all. *Not done:* pinning them — a database image's tag is a
deployment decision with an upgrade path attached, and an audit that fixed its
own findings would be making that decision without one.

### Womblex as a backend subsystem — data contract plus an owner-scoped service

Womblex serves other software two ways: a versioned on-disk contract
([contract.md](contract.md)) and a callable `/v1` API
([service-api.md](service-api.md)).

- **A separate `womblex serve` app, not an extended console.** The console
  stays an unauthenticated admin/debug utility; its location-override `PUT` and
  report/feedback action were retired so it holds no writable state beyond
  dispatch and presets.
- **Static service tokens.** This is a trusted-subsystem deployment on a
  private network, not a public cloud service, so a hashed YAML registry is
  enough; no identity provider.
- **Runs are owner-scoped on the run board.** Ownership is an attribute of the
  DBOS workflows, the existing source of truth for what runs exist, rather than
  a new store.
- **Determinism is an accuracy target, not a blocker.** The contract
  guarantees *content* stability (`content_digest`) for the same version,
  config and models, not file bytes. Drift is detectable, not prevented.
- **Consumers poll.** Callbacks, cancellation and per-client quotas were left
  out of v1.
