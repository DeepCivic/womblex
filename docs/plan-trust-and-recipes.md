# Trust baseline and recipes — plan

*Status: proposed (2026-10). No merge has shipped. Each phase is a branch of sequential merges under the 500-line cap. Each merge updates this document as it lands, and the document is retired into `decisions.md` once its last phase ships.*

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
- **The PII stage can label unmasked text `masked`.** The PII stage treats the enrichment entities sidecar as optional (`strict=False` in `cloud/stage_contracts.py`). When it is absent, `pii/pii_stage.py` finds no graph spans, logs nothing, and still writes `*.clean_text.parquet`, which `store/contract.py` labels `masked`. With the regex backstop off (the default) that file is the raw chunk text under the label that says it is safe to hand onward.
- **A missing `text_source` overlay changes the evidence layer silently on a local run.** This is the open Requirement 3 TO-DO in `functional_requirements.md`. Downstream offsets then index a different text layer than the config selected.
- **Unknown config keys are ignored.** No model under `config/` sets `extra="forbid"`, so a misspelt key in a YAML config validates and is dropped.
- **Table cells cannot be located on the page.** `Cell` carries no `bbox`, so a cell value is locatable only to its parent table element. Most of the corpus's monetary amounts live in table cells.
- **Each annotation sidecar defines its own anchor columns.** There is no shared shape for "where in the source this came from".
- **Output files carry no file-level checksum.** `content_digest` covers a document's elements; nothing records the bytes of each output file.
- **`graph-refresh` rewrites files in place** (`MutationMode.IN_PLACE`), which is the one exception to write-once sidecars.
- **The local stage sequence lives in comments.** `womblex run` extracts only; the per-stage order for a full pipeline is documented in comments in `configs/default-isaacus.yaml` and run by hand.
- **Egress copies unredacted material by design.** `docs/egress.md` makes access control the responsibility of whoever runs egress and hosts the bundle. FR-5.1's "masked outputs need an explicit policy before delivery" contradicts that decision; it is listed under open questions, not as a defect.

## Dependency assessment
Assessed against the repository's rules: thin adapters only, delete Womblex code when a library takes a concern over, no heavyweight ML in core, and a core install that runs without a database.

| Candidate | Licence | Verdict | Reason |
|---|---|---|---|
| DBOS Transact | MIT | Defer to Phase 3 | Overlaps `cloud/queue.py`, the worker, the stage runner and checkpoints, all working. Adoption means deleting those, and stores step results in its database while Womblex's checkpoint unit is a Parquet shard in object storage. Phase 3 is the decision point: delivery retries and events in the existing queue, or DBOS replacing it |
| OpenLineage (`openlineage-python`) | Apache-2.0 | No dependency | Emit spec-conformant JSON from the run record and test it against the published schema. An optional extra only if pushing to a lineage server becomes a requirement |
| Docling (full converter) | MIT | Reject for core | Brings torch and its own layout models. After layout becomes its own stage (L3 in `plan-permissive-deps.md`), its layout model could be a `womblex.models.layout` plugin installed outside core and judged by the benchmark |
| docling-core | MIT | Optional egress format, later | Light, pydantic-based. Useful only as an output format for consumers that want Docling's schema |
| docling-parse | MIT | Hold | A fallback for the PyMuPDF replacement if pypdfium2 + pdfplumber miss their parity gates |
| Docling Graph | MIT | Reject; borrow the idea | Its deterministic provenance ledger informs the evidence reference below; the package brings full Docling plus LLM clients |
| Pydantic AI | MIT | Reject | Field extraction is out of scope; a future topic annotation would go through Isaacus |
| Unstructured | Apache-2.0 | Reject | Heavy, torch-based inference; duplicates the existing extractors |
| MarkItDown | MIT | Reject | Drops page geometry, which the evidence work needs |
| Temporal, Prefect, Haystack | MIT / Apache-2.0 | Reject | A separate service, or a pipeline model competing with `STAGE_CONTRACTS` |
| lakeFS | BUSL-1.1 | Reject | Not a permissive licence; a separate service |
| OpenInference | Apache-2.0 | Reject for now | Model calls are Isaacus and Bedrock; JSON logs with run context already exist |

Requirements that would need a dependency: webhooks can use `httpx`, already in the lockfile through the Isaacus SDK (declaring it directly needs approval); each database or vector-store destination brings its own client and belongs in a per-destination extra.

## Workflow model
Today a preset is a partial config: stage switches and settings, with no dataset or paths. Its name and the config digest are stamped into every output footer. It holds no step sequence, inputs, conditions or destinations, and it is applied once per run rather than existing anywhere.

| Option | Pros | Cons |
|---|---|---|
| **A. Extend the preset** with steps, conditions and destinations | One concept and one schema; lineage already covered by `config_digest`; no new storage | Mixes how a stage behaves with what runs and where results go; still nothing for `deploy` / `destroy` to act on |
| **B. Stored workflow object** referencing a preset, plus input binding, steps, destinations and triggers | Clean separation; one preset serves many workflows; `deploy` / `status` / `destroy` have a subject; standing triggers become possible | A second concept; a new table and migration; owner scoping on another object; a workflow digest must be stamped beside `config_digest`; triggers need an always-on scheduler |
| **C. Recipe file** holding a preset plus steps, conditions and destinations, kept in the user's repository | Diffable and reviewable; no storage or lifecycle; meets FR-2.1, FR-2.2 and most of FR-4.4; promotes to B later without breaking anything | No standing triggers; event subscriptions are declared per run |

Because `womblex run` extracts only, options B and C both need a **local multi-stage runner** that executes a step list in order through the existing stage functions. That runner also replaces the comment-documented sequence in `configs/default-isaacus.yaml`.

**Proposed:** C, scoped as recipe file plus local multi-stage runner. Promote to B when standing triggers become a requirement; the recipe digest stamped beside the config digest keeps lineage continuous across that change.

## Phases

### Phase 1 — trust baseline (branch)
1. **Strict `text_source`.** A missing overlay the config selected refuses on both the local and object-store paths, before any base is processed. Closes part of the Requirement 3 TO-DO.
2. **No false `masked` label.** The PII stage refuses, or writes no `clean_text`, when it has neither enrichment entities nor the regex backstop. Which of the two is an open question.
3. **Strict config.** `extra="forbid"` on the config models, with presets and shipped configs checked against it.
4. **Up-front ordering check.** Verify the whole stage sequence's required inputs before processing, with stable error codes shared by the CLI and the API. Closes the rest of the Requirement 3 TO-DO.
5. **File checksums.** A SHA-256 per output file in the run record.
6. **Evidence reference.** One shared anchor shape (source hash, element order, page, bbox, character span, text layer) adopted by the money, entity-link and PII sidecars, plus a check that each span's text equals the source text at its offsets.
7. **Cell bbox.** Add `bbox` to `Cell` and the table-cells sidecar: an additive column and a minor contract-version bump.

### Phase 2 — recipes (branch)
- Recipe file schema (preset reference or inline config, ordered steps, conditions, destinations), validated by the same code path as configs.
- Local multi-stage runner executing a recipe's steps through the existing stage functions.
- `womblex validate` and `womblex plan` over a recipe. The plan reports steps to run or skip and exact token counts for enrich and embed via the offline tokeniser; prices are labelled as assumptions.
- Constrained conditions over manifest and profile columns only. A condition that cannot be evaluated stops the run; no condition can disable `pii`.
- The recipe digest stamped beside the config digest.

### Phase 3 — integration (branch)
- Destinations as delivery targets beside egress, each with its own status and bounded retries.
- Delivery attempts and webhook events recorded in the existing queue, unless the DBOS decision above goes the other way.
- An OpenLineage-shaped export of the run record.

### Phase 4 — operation (branch)
- Run status in plain language, consistent with events and lineage.
- `graph-refresh` writing a new versioned sidecar instead of rewriting in place.
- Option B, only if standing triggers are needed.

## Open questions
- Should the PII stage refuse outright, or skip writing `clean_text`, when it has no candidate source?
- Does the egress decision (unredacted material, access control left to the host) change once per-step destinations exist?
- Which destinations come first: object storage only, or a database target as well?

## Conventions this plan holds to
- No quality or confidence scoring, and no scoring against ground truth in this repository.
- Extraction text stays verbatim; every new artefact is a sidecar.
- Each merge stays under 500 changed lines and updates `docs/architecture.md`, `docs/project-structure.md` and `functional_requirements.md` where it changes them.
- No dependency is added without approval.
