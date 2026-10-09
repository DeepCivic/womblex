# Trust baseline and recipes — plan

*Status: in progress (2026-10). Phase 0 (DBOS) and Phase 1 items 1 and 2 have shipped; D1 to D11 are decided. Each phase is a branch of sequential merges under the 500-line cap. Each merge updates this document as it lands, and the document is retired into `decisions.md` once its last phase ships.*

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
- **The PII stage can label unmasked text `masked`.** The PII stage treats the enrichment entities sidecar as optional (`strict=False` in `cloud/stage_contracts.py`). When it is absent, `pii/pii_stage.py` finds no graph spans, logs nothing, and still writes `*.clean_text.parquet`, which `store/contract.py` labels `masked`. With the regex backstop off (the default) that file is the raw chunk text under the label that says it is safe to hand onward. Table chunks are in that state on every run: enrichment covers the reassembled narrative only, and `pii_stage` passes graph spans to narrative chunks only.
- **A missing `text_source` overlay changes the evidence layer silently on a local run.** Downstream offsets then index a different text layer than the config selected. *Fixed by Phase 1 item 1.*
- **Unknown config keys are ignored.** No model under `config/` sets `extra="forbid"`, so a misspelt key in a YAML config validates and is dropped. The only rejection today is the `_reject_removed` validators in `config/__init__.py`, which refuse retired sections by name.
- **Table cells cannot be located on the page.** `Cell` carries no `bbox`, so a cell value is locatable only to its parent table element. Most of the corpus's monetary amounts live in table cells.
- **Each annotation sidecar defines its own anchor columns.** There is no shared shape for "where in the source this came from".
- **Output files carry no file-level checksum.** `content_digest` covers a document's elements; nothing records the bytes of each output file.
- **Two stages rewrite files in place** (`MutationMode.IN_PLACE`), the exceptions to write-once sidecars. `graph-refresh` rewrites the entity and edge sidecars. `layout` replaces the extraction batch's `*.layout_regions.parquet`, but only when the config's layout fingerprint differs from the file's, and all-or-nothing per batch.
- **The local stage sequence lives in comments.** `womblex run` extracts only; the per-stage order for a full pipeline is documented in comments in `configs/default-isaacus.yaml` and run by hand.
- **Egress copies unredacted material by design.** `docs/egress.md` makes access control the responsibility of whoever runs egress and hosts the bundle. FR-5.1's "masked outputs need an explicit policy before delivery" contradicts that decision; it is listed under open questions, not as a defect.

## Dependency assessment
Assessed against the repository's rules: thin adapters only, delete Womblex code when a library takes a concern over, no heavyweight ML in core, and a core install that runs without a database.

| Candidate | Licence | Verdict | Reason |
|---|---|---|---|
| DBOS Transact | MIT | Adopt in Phase 0 (D9) | 3.2.0 (2026-09) covers schedules, step retries with backoff, rate-limited queues and workflow events on Postgres or SQLite. Adds `sqlalchemy` and `websockets` to the lockfile; the `pyproject.toml` change still goes to review. It overlaps `cloud/queue.py`, the worker, the stage runner and checkpoints, and stores step results in its database while Womblex's checkpoint unit is a Parquet shard in object storage, and it replaces all of them (D9) |
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

### Phase 0 — DBOS foundation (shipped)
DBOS replaced the job system outright (D9), and went first because every later phase builds on it. Shipped as one change because the queue, worker, stage runner and every reader moved together; the 500-line cap was waived for it by the maintainer. What differs from the bullets below: `cloud/stage_runner.py` stays as the stage engine (`plan_units`, `run_unit`, and `run_stage_remote` for `run-stage --store`) rather than being deleted; the order is one coordinator workflow per dispatch, with a child workflow per stage; owner scoping is a workflow attribute filtered client-side; `womblex worker` lost `--run-id` and `--stale-timeout`; DBOS keeps its default pickle serialiser, since the portable JSON one is not configured. Rationale is in `decisions.md`.
- **Dependency approved (2026-10).** `dbos` is added to `pyproject.toml`, bringing `sqlalchemy` and `websockets` into the lockfile, in its own dependency-scoped merge.
- **Replace, don't wrap.** `cloud/queue.py`, `cloud/worker.py` and `cloud/stage_runner.py` are rewritten on DBOS workflows and deleted as they are replaced: an extraction batch and a downstream stage unit are each a workflow, and the order `pipeline_order` declares becomes the workflow's step order rather than queue-row positioning.
- **Completion moves to DBOS's progress records.** Today a stage unit counts as done when all its declared outputs are published (skip-by-published-output, all-or-none publish). Under DBOS the publish is the unit's last step, so a recorded step is the done signal, and a re-run skips it. Step results are storage keys, never data, under DBOS's portable JSON serialiser, not its default pickle.
- **Kept as Womblex code:** owner scoping of runs (`RunOwnedError`, owner-filtered views), stage contracts and their preflight, stage-in/stage-out through `store/remote.py`, and the shared `batch.process_batch`, so local and distributed runs stay byte-identical.
- **Local runs** use DBOS's SQLite system database, so the core install still runs without a database server.
- **Carried over:** `enqueue` / `enqueue-stages` / `worker` / `jobs` / `finalize` / `run-stage` keep their CLI surface; the console and `/v1` API read run status from DBOS instead of the `womblex_jobs` table. Runs queued under the old table are drained or re-enqueued, not migrated.
- **Retires** the "Distributed execution" decision's queue-as-checkpoint half in `decisions.md`, rewritten in the same change that deletes `cloud/queue.py`.
- **Follow-up: retry a failed stage.** Re-dispatching a stage that failed returns its recorded failure, as the old queue did. DBOS's `resume_workflow` does not restart a failed workflow, but `fork_workflow` does: it starts a copy from a chosen step and keeps the steps before it, so only the failed unit and those after it run again. The copy has a new workflow id and the failed original stays on the board, so the follow-up decides how a run's status counts the original once its retry succeeds. Its own merge, after Phase 0.

### Phase 1 — trust baseline (branch)
1. **Strict `text_source`.** *Shipped.* A missing overlay the config selected refuses on both the local and object-store paths, before any base is processed (`require_overlays` / `MissingOverlayError` locally; an up-front pass in `run_stage_remote`).
2. **No false `masked` label.** *Shipped.* `*.clean_text.parquet` gains a per-row `mask_status` enum: `masked` (at least one span replaced), `no_entity` (a candidate source covered the chunk and found nothing) and `not_masked` (no candidate source covered the chunk, such as a batch with no enrichment entities). The column is additive: a minor contract bump with a reader back-fill (D1).
3. **Table chunks get candidates.** Each table chunk's own text is sent to the enricher, separately from the narrative (folding tables into the narrative is not pursued, see `decisions.md`), so narrative offsets do not move and a table chunk's spans index its own text. Table text so table chunks have graph spans of their own, and the Kanon-2 token spend this adds is measured and recorded (D2). Further entity-recognition, embedding and graph providers are planned and must fit beside Isaacus as alternatives.
4. **Strict config.** `extra="forbid"` on the config models, with presets and shipped configs checked against it. The `_reject_removed` validators stay for their named messages, a saved preset that no longer validates is logged by name (D3), and the merge records in `CHANGELOG.md` that a saved preset or user config with a stray key now fails validation.
5. **File checksums.** A SHA-256 per output file, computed at write time, returned with the file's key by the DBOS publish step (Phase 0), and written by `finalize` into the run's index (`manifest.parquet`), so a whole run is auditable without opening every file (D5).
6. **Evidence reference.** One shared anchor shape (source hash, element order, page, bbox, character span, text layer) *replaces* the anchor columns of the money, entity-link and PII sidecars, plus a check that each span's text equals the source text at its offsets. The check runs at write time, where a mismatch refuses the batch's publish, and as an on-demand verify command over finished runs, beside shard integrity verification. A major contract bump (D6).
7. **Reader migration.** Every reader of those sidecars moves to the new anchor shape in the same branch as item 6: `api/readers.py`, `ui/readers.py`, the benchmark's suites, and the reader shim `contract.md` requires for the old major.
8. **Cell bbox.** Add `bbox` to `Cell` and the table-cells sidecar: an additive column and a minor contract-version bump.

### Phase 2 — workflows (branch)
- Stored workflow object (preset reference, input binding, ordered steps, conditions, destinations, triggers): a new table with its migration in the database DBOS uses (Postgres, or SQLite on a local run), owner scoping as runs have, validated by the same code path as configs.
- Local multi-stage runner executing a workflow's steps through the existing stage functions.
- **Up-front ordering check**, moved from Phase 1 so local and object-store runs get it together (D4): verify the whole step sequence's required inputs before processing, with stable error codes shared by the CLI and the API. Closes the Requirement 3 TO-DO's first bullet.
- Standing triggers on DBOS schedules (Phase 0).
- `womblex validate` and `womblex plan` over a workflow. The plan reports steps to run or skip and exact token counts for enrich and embed via the offline tokeniser; prices are labelled as assumptions.
- Constrained conditions over manifest and profile columns only. A condition that cannot be evaluated stops the run; no condition can disable `pii`.
- The workflow digest stamped beside the config digest.

### Phase 3 — integration (branch)
- Destinations as delivery targets beside egress, each with its own status and bounded retries: object storage, Postgres with pgvector, and webhooks (D11).
- Delivery retries and webhook events run as DBOS workflows (Phase 0).
- An OpenLineage-shaped export of the run record.

### Phase 4 — operation (branch)
- Run status in plain language, consistent with events and lineage.
- `graph-refresh` writing a new versioned sidecar instead of rewriting in place. `layout` keeps its in-place replace: the file is a derived cache of the source document, its fingerprint records which model and settings produced it, and the replace is all-or-nothing per batch.

## Decisions
All settled (2026-10); nothing in this plan waits on a decision. "Repo answer" records what existing conventions settled and what was decided.

| # | Gates | Decision | Repo answer |
|---|---|---|---|
| D1 | Phase 1 item 2 | What the PII stage writes for a chunk with no PII candidate source | **Decided 2026-10:** option (c), a per-row `mask_status` enum: `masked` (at least one span replaced), `no_entity` (a candidate source covered the chunk and found nothing), `not_masked` (no candidate source covered the chunk). Existing readers keep working |
| D2 | Phase 1 item 3 | Whether table chunks get a candidate source of their own | **Decided 2026-10:** send table text to the enricher; the added spend is measured. Other entity-recognition, embedding and graph options come later |
| D3 | Phase 1 item 4 | Whether `extra="forbid"` supersedes the `_reject_removed` validators, and how a saved preset with a stray key is treated | **Answered.** Retired keys are refused naming their replacement (`CHANGELOG.md`, L3b and L3c), which a generic extra-field error cannot do, so the validators stay. `ui/presets.parse_saved_preset` skips a saved preset that will not validate rather than failing the list, so a stray key hides that preset; the merge adds a log line naming it |
| D4 | Phase 2 | When the up-front ordering check ships | **Decided 2026-10:** with workflows in Phase 2, so local and object-store runs get it at once |
| D5 | Phase 1 item 5 | Where file checksums are stored | **Decided 2026-10:** computed at write time, stored in the run's index (`manifest.parquet`) |
| D6 | Phase 1 items 6 and 7 | Whether the evidence reference replaces or sits beside each sidecar's anchor columns, and where the span check runs | **Decided 2026-10:** replace (a major contract bump), with reader migration as its own scope item. The span check runs both at write time and as an on-demand verify command |
| D7 | Phase 1 item 8 | Cell bbox coordinate space, sources with no page geometry, and branch split | **Answered.** `BBox` is normalised 0–1 with a top-left origin, and `Element.bbox` is already optional; cells follow both, so DOCX and spreadsheet cells carry null. The column is additive, a minor bump (`contract.md`). The merge cap requires splitting items 6 to 8 into sequential merges; whether they form a separately named Phase 1b is labelling only |
| D8 | Phase 2 | Workflow model, and the condition language's scope | **Decided 2026-10:** option B, a stored workflow object. Condition scope as Phase 2 states |
| D9 | Phase 0 | DBOS against the existing queue for delivery retries and events | **Decided 2026-10:** DBOS. Alternatives checked: Procrastinate and PgQueuer (scheduling and retries, no durable steps or events), Absurd (pre-1.0, no schedules in the Python SDK); Hatchet, Temporal and Prefect need a separate server; Dramatiq is LGPL |
| D10 | Phase 3 | Whether the egress decision changes once per-step destinations exist | **Confirmed 2026-10.** Bundles may hold unredacted sources; access control is the host's responsibility (`egress.md`) |
| D11 | Phase 3 | Which destinations come first | **Decided 2026-10:** object storage (S3, built on egress), Postgres with pgvector, and webhook notifications. `httpx` approved for webhooks |

## Conventions this plan holds to
- No quality or confidence scoring, and no scoring against ground truth in this repository.
- Extraction text stays verbatim; every new artefact is a sidecar.
- Each merge stays under 500 changed lines and updates `docs/architecture.md`, `docs/project-structure.md` and `docs/functional_requirements.md` where it changes them.
- No dependency is added without approval.
