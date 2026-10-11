# Trust baseline and recipes — plan

*Status: in progress (2026-10). Outstanding work only: what has shipped (Phase 0 DBOS; all of Phase 1, the trust baseline) is recorded in `CHANGELOG.md`, `decisions.md` and the enduring docs. Each phase is a branch of sequential merges under the 500-line cap. Each merge updates this document as it lands, and the document is retired into `decisions.md` once its last phase ships.*

## Context
A business-analyst requirements set proposed five themes: high-integrity extraction, recipe-based workflow authoring, destinations and delivery, agent-friendly operation, and safety and operability. Its labels (FR-1.1 to FR-5.2) are kept below so this plan can be read against it. It was written without knowledge of the repository, so this plan maps each theme onto what Womblex already has, records what is out of scope, and orders the remaining gaps.

**Decisions taken:**
- **No confidence or quality scoring.** FR-1.2 (accepted / flagged / rejected verdicts) is out. Measuring quality is the benchmark's job; the contributor guidance's rule against quality scoring stands.
- **No schema-driven field extraction.** Requests such as "extract invoices" or "summarise contracts" are not Womblex features. FR-1.1 applies only to the annotations Womblex already produces (money spans, enrichment mentions, entity links, PII spans).
- **Topic annotation of elements is a later roadmap item**, not part of this plan. When it lands it is one more annotation sidecar and inherits the evidence rules below.
- **New dependencies go to review.** The assessment is below; nothing is added without approval.
- **The merge cap stays.** Phases ship as branches of sequential merges.

## What already exists
Verified against the code at the time of writing.

| Theme | Already in place |
|---|---|
| Evidence (FR-1.1) | Every `Element` carries `page`, a normalised `bbox`, `order` and `confidence`; `source_hash` is content-addressed; money, enrichment and PII sidecars carry character offsets; `SourceResolver` resolves a row back to its source file |
| Run artefacts (FR-1.3) | Run-stamp footer keys (run id, version, commit, config digest, stage, preset, models, model check); per-document `content_digest`; contract version and sensitivity footer keys; all-or-none stage publish; per-stage checkpoints |
| Lineage (FR-1.4) | `manifest.parquet`, `build_run_record`, the footer stamps and `source_index.parquet` already link source, run and outputs |
| Recipes (FR-2.1) | YAML config validated by `WomblexConfig`; built-in and saved presets; the composer UI |
| Step ordering (FR-2.2) | `STAGE_CONTRACTS` declares each stage's required inputs; `PIPELINE_ORDER` is one valid order |
| Delivery (FR-3) | Egress bundles to any `RemoteStore` destination |
| Agents (FR-4) | Typed exceptions (`InputContractError`, `StagePreconditionError`, `RunOwnedError`); the owner-scoped `/v1` API; queue status rollups |
| Sensitive data (FR-5.1) | The `sensitivity` footer key; the API reads the masked layer by default and gates raw layers behind `read_raw` |

## Findings
- **Only native PDF tables have cell boxes.** DOCX, OCR-reconstructed and spreadsheet-print tables carry a null cell `bbox`; OCR tables could take one from the `table_grid` bands and columns if a consumer needs it.
- **Each annotation sidecar defines its own anchor columns.** There is no shared shape for "where in the source this came from".
- **Two stages rewrite files in place** (`MutationMode.IN_PLACE`), the exceptions to write-once sidecars. `graph-refresh` rewrites the entity and edge sidecars. `layout` replaces the extraction batch's `*.layout_regions.parquet`, but only when the config's layout fingerprint differs from the file's, and all-or-nothing per batch.
- **The local stage sequence lives in comments.** `womblex run` extracts only; the per-stage order for a full pipeline is documented in comments in `configs/default-isaacus.yaml` and run by hand.
- **Egress copies unredacted material by design.** `docs/egress.md` makes access control the responsibility of whoever runs egress and hosts the bundle. FR-5.1's "masked outputs need an explicit policy before delivery" contradicts that decision; it is listed under open questions, not as a defect.

## Dependency assessment
Assessed against the repository's rules (DBOS was adopted in Phase 0 and is recorded in `decisions.md`): thin adapters only, delete Womblex code when a library takes a concern over, no heavyweight ML in core, and a core install that runs without a database.

| Candidate | Licence | Verdict | Reason |
|---|---|---|---|
| OpenLineage (`openlineage-python`) | Apache-2.0 | No dependency | Emit spec-conformant JSON from the run record and test it against the published schema. An optional extra only if pushing to a lineage server becomes a requirement |
| Docling (full converter) | MIT | Reject for core | Brings torch and its own layout models. Now that layout is its own stage ([`layout.md`](layout.md)), its layout model could be a `womblex.models.layout` plugin installed outside core and judged by the benchmark |
| docling-core | MIT | Optional egress format, later | Light, pydantic-based. Useful only as an output format for consumers that want Docling's schema |
| docling-parse | MIT | Hold | A fallback if pypdfium2 + pdfplumber miss their parity gates |
| Docling Graph | MIT | Reject; borrow the idea | Its deterministic provenance ledger informs the evidence reference below; the package brings full Docling plus LLM clients |
| Pydantic AI | MIT | Reject | Field extraction is out of scope; a future topic annotation would go through Isaacus |
| Unstructured | Apache-2.0 | Reject | Heavy, torch-based inference; duplicates the existing extractors |
| MarkItDown | MIT | Reject | Drops page geometry, which the evidence work needs |
| Temporal, Prefect, Haystack | MIT / Apache-2.0 | Reject | A separate service, or a pipeline model competing with `STAGE_CONTRACTS` |
| lakeFS | BUSL-1.1 | Reject | Not a permissive licence; a separate service |
| OpenInference | Apache-2.0 | Reject for now | Model calls are Isaacus and Bedrock; JSON logs with run context already exist |

Requirements that would need a dependency: webhooks use `httpx`, already in the lockfile through the Isaacus SDK and approved for direct declaration (2026-10); each database or vector-store destination brings its own client and belongs in a per-destination extra.

## Workflow model
Today a preset is a partial config: stage switches and settings, with no dataset or paths. Its name and the config digest are stamped into every output footer. It holds no step sequence, inputs, conditions or destinations, and it is applied once per run rather than existing anywhere.

| Option | Pros | Cons |
|---|---|---|
| **A. Extend the preset** with steps, conditions and destinations | One concept and one schema; lineage already covered by `config_digest`; no new storage | Mixes how a stage behaves with what runs and where results go; still nothing for `deploy` / `destroy` to act on |
| **B. Stored workflow object** referencing a preset, plus input binding, steps, destinations and triggers | Clean separation; one preset serves many workflows; `deploy` / `status` / `destroy` have a subject; standing triggers become possible | A second concept; a new table and migration; owner scoping on another object; a workflow digest must be stamped beside `config_digest`; triggers need an always-on scheduler |
| **C. Recipe file** holding a preset plus steps, conditions and destinations, kept in the user's repository | Diffable and reviewable; no storage or lifecycle; meets FR-2.1, FR-2.2 and most of FR-4.4; promotes to B later without breaking anything | No standing triggers; event subscriptions are declared per run |

Because `womblex run` extracts only, options B and C both need a **local multi-stage runner** that executes a step list in order through the existing stage functions. That runner also replaces the comment-documented sequence in `configs/default-isaacus.yaml`.

**Decided (D8, 2026-10):** B, a stored workflow object, so scheduled and standing triggers are possible. It still needs the local multi-stage runner, which executes a workflow's steps through the existing stage functions.

## Phases

### Phase 0 follow-up: retry a failed stage
Re-dispatching a stage that failed returns its recorded failure. DBOS's `resume_workflow` does not restart a failed workflow, but `fork_workflow` does: it starts a copy from a chosen step and keeps the steps before it, so only the failed unit and those after it run again. The copy has a new workflow id and the failed original stays on the board, so this merge decides how a run's status counts the original once its retry succeeds. Its own merge.

### Enrichment provider slot (own merge, after the trust baseline)
An `enrichment` slot in `utils/model_registry.py`, default `kanon-2-enricher`, with providers installed through `womblex.models.enrichment` entry points; the provider contract, the limits each provider declares and what Womblex keeps are in `decisions.md` ("Local enrichment fallback"). Embedding follows the same pattern. The GLiNER provider, in development by the maintainer, is the first alternative and follows this item. No provider other than Isaacus ships with it.

### Phase 2 — workflows (branch)
- **Shipped (merge 1):** stored workflow object (preset reference, input binding, ordered steps, conditions, destinations, triggers): a new table in the database DBOS uses (Postgres, or SQLite on a local run), owner scoping as runs have, validated by the same code path as configs. `cloud/workflow_store.py`. The table is created if absent; a versioned migration comes with its first column change. Schema and storage only — conditions/destinations round-trip but are not evaluated or dispatched, triggers are stored but not scheduled, and there is no CLI or API surface yet; the remaining bullets below act on this object.
- **Shipped (merge 2):** local multi-stage runner (`cloud/workflow_runner.py`, `womblex run-workflow <id> --shards DIR`): a workflow's steps run in list order through `prepare_stage_context` + `run_stage_local`. `WorkflowStore.save` now runs `check_spec`; the runner re-runs it on load. The preset (built-in or saved) and the workflow's `config` combine per top-level section and the whole is validated. Everything that can refuse does before the first step (spec, preset, config, every step's preflight and model check); once a step has run nothing halts — a failed step is recorded and logged and later steps still run, labelled `ran_after_failed`. Conditions, destinations and triggers are warned as stored but not acted on. Object-store execution of a workflow is not in it.
- **Shipped (merge 3):** the up-front ordering check (D4), `cloud/workflow_order.py`. Each step's required inputs and strict config-derived inputs must be on disk or produced by an earlier step; refusals carry stable codes `input_missing` / `input_out_of_order` / `input_stale`. A copy on disk is stale when a later step will rewrite it (so a misordered re-run over the same shard directory is refused, not read stale); `IN_PLACE` stages are not ordered; an optional input produced later against `PIPELINE_ORDER` is warned, not refused. Used by `run_workflow_local` and by stage dispatch against the store listing (`enqueue-stages --store`, the console, `POST /v1/runs`, whose 400 carries `detail.issues`). A stored workflow has no object-store runner yet, so that path is covered through dispatch only. Closes the Requirement 3 TO-DO's first bullet.
- **Shipped (merge 4):** `womblex validate <id> [--shards DIR] [--json]` (`cloud/workflow_validate.py`): the runner's pre-run checks through the shared `prepare_workflow`, read-only, with stable `RefusalCode`s beside the ordering codes; every failed step is reported.
- **Shipped (merge 7a):** the fleet carries a workflow's config: `enqueue_stages(workflow=…)` hands the resolved overlay and digest to the coordinator and each stage, which run under `config_from_overlay` and stamp the workflow digest. The dispatch command and its refusals are 7b.
- **Shipped (merge 7b):** `womblex dispatch-workflow <id> --run-id R --store URI [--dry-run]` (`cloud/workflow_dispatch.py`): object-store execution of a stored workflow through the coordinator, order-checked against the store with the local runner's refusal codes plus `stage_not_dispatchable`, `order_not_pipeline` and `run_conflict`. Closes the Requirement 3 gap for workflows.
- **Shipped (merge 8):** constrained conditions over manifest columns, evaluated and refused before any step spends anything (`cloud/workflow_conditions.py`); admit-or-refuse only, so none can skip a step or disable `pii`. Page-level profile is not persisted, so it is not a condition column.
- Standing triggers on DBOS schedules.
- **Shipped (merge 5):** `womblex plan <id> --shards DIR [--assume-price STAGE=USD] [--json]` (`cloud/workflow_plan.py`): run/skip per step and exact enrich and embed token counts via the offline tokeniser, never calling a paid API; prices only as caller-supplied, labelled assumptions.
- Constrained conditions over manifest and profile columns only. A condition that cannot be evaluated stops the run; no condition can disable `pii`.
- **Shipped (merge 6):** the workflow digest stamped beside the config digest: `womblex.workflow_digest` on every sidecar a workflow run writes, and `workflows` in the run record (version 6).

### Phase 3 — integration (branch)
- Destinations as delivery targets beside egress, each with its own status and bounded retries: object storage, Postgres with pgvector, and webhooks (D11).
- Delivery retries and webhook events run as DBOS workflows.
- An OpenLineage-shaped export of the run record.

### Phase 4 — operation (branch)
- Run status in plain language, consistent with events and lineage.
- `graph-refresh` writing a new versioned sidecar instead of rewriting in place. `layout` keeps its in-place replace: the file is a derived cache of the source document, its fingerprint records which model and settings produced it, and the replace is all-or-nothing per batch.

## Decisions
All settled (2026-10); nothing in this plan waits on a decision. Decisions whose work has shipped (D1 `mask_status`, D2 table-chunk enrichment, D6 the evidence reference, D9 DBOS) are in `decisions.md` and `CHANGELOG.md`. "Repo answer" records what existing conventions settled and what was decided.

| # | Gates | Decision | Repo answer |
|---|---|---|---|
| D4 | Phase 2 | When the up-front ordering check ships | **Decided 2026-10:** with workflows in Phase 2, so local and object-store runs get it at once |
| D8 | Phase 2 | Workflow model, and the condition language's scope | **Decided 2026-10:** option B, a stored workflow object. Condition scope as Phase 2 states |
| D10 | Phase 3 | Whether the egress decision changes once per-step destinations exist | **Confirmed 2026-10.** Bundles may hold unredacted sources; access control is the host's responsibility (`egress.md`) |
| D11 | Phase 3 | Which destinations come first | **Decided 2026-10:** object storage (S3, built on egress), Postgres with pgvector, and webhook notifications. `httpx` approved for webhooks |

## Conventions this plan holds to
- No quality or confidence scoring, and no scoring against ground truth in this repository.
- Extraction text stays verbatim; every new artefact is a sidecar.
- Each merge stays under 500 changed lines and updates `docs/architecture.md`, `docs/project-structure.md` and `docs/functional_requirements.md` where it changes them.
- No dependency is added without approval.
