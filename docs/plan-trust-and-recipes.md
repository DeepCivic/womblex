# Trust baseline and recipes — plan

*Status: in progress (2026-10). Phase 1 item 1 has shipped. Each phase is a branch of sequential merges under the 500-line cap. Each merge updates this document as it lands, and the document is retired into `decisions.md` once its last phase ships.*

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
| DBOS Transact | MIT | Defer to Phase 3 | Overlaps `cloud/queue.py`, the worker, the stage runner and checkpoints, all working. Adoption means deleting those, and stores step results in its database while Womblex's checkpoint unit is a Parquet shard in object storage. Phase 3 is the decision point: delivery retries and events in the existing queue, or DBOS replacing it |
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
1. **Strict `text_source`.** *Shipped.* A missing overlay the config selected refuses on both the local and object-store paths, before any base is processed (`require_overlays` / `MissingOverlayError` locally; an up-front pass in `run_stage_remote`).
2. **No false `masked` label.** Covers both a batch with no enrichment entities and table chunks with the backstop off. Blocked on decision D1.
3. **Strict config.** `extra="forbid"` on the config models, with presets and shipped configs checked against it. The merge states whether it supersedes the `_reject_removed` validators or keeps them for their named messages, and records in `CHANGELOG.md` that a saved preset or user config with a stray key now fails validation.
4. **Up-front ordering check.** Verify the whole stage sequence's required inputs before processing, with stable error codes shared by the CLI and the API. Closes the Requirement 3 TO-DO's first bullet.
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
- `graph-refresh` writing a new versioned sidecar instead of rewriting in place. `layout` keeps its in-place replace: the file is a derived cache of the source document, its fingerprint records which model and settings produced it, and the replace is all-or-nothing per batch.
- Option B, only if standing triggers are needed.

## Decisions pending
Each is settled before the merge it gates starts, and recorded here when taken. "Repo answer" is what existing conventions already settle; only the remainder is open.

| # | Gates | Decision | Repo answer |
|---|---|---|---|
| D1 | Phase 1 item 2 | With no PII candidate source for a chunk (no enrichment entities for the batch; any table chunk with the backstop off), the PII stage (a) refuses with a typed error, (b) omits `clean_text` for the affected batch or chunks, or (c) writes the row with a per-row masked / unmasked column | **Partly.** `decisions.md` (PII) records that narrative graph spans never apply to table chunks, but not that their `clean_text` rows are labelled `masked`. Option (c) is an additive column, so a minor contract bump with a reader back-fill (`contract.md`). The choice is open |
| D2 | Phase 1 item 2 | Whether table chunks get a candidate source of their own, or stay outside masking under D1's rule | **Partly.** `decisions.md` (PII) settles that recall is raised by enrichment granularity, not a second detector, and keeps the backstop opt-in for its low precision. That rules out turning the backstop on for table chunks by default; enriching table text, or leaving tables outside masking, is open |
| D3 | Phase 1 item 3 | Whether `extra="forbid"` supersedes the `_reject_removed` validators, and how a saved preset with a stray key is treated | **Answered.** Retired keys are refused naming their replacement (`CHANGELOG.md`, L3b and L3c), which a generic extra-field error cannot do, so the validators stay. `ui/presets.parse_saved_preset` skips a saved preset that will not validate rather than failing the list, so a stray key hides that preset; the merge adds a log line naming it |
| D4 | Phase 1 item 4 | Whether the up-front ordering check ships in Phase 1 for the object-store path, or moves to Phase 2 with the recipe | **Partly.** Local and cloud parity is the design invariant (`decisions.md`, nested corpora), so a remote-only check runs against it. The Requirement 3 TO-DO leaves open (a) a check against the dispatched-stage set or (b) documenting per-base `NotReady` as the contract |
| D5 | Phase 1 item 5 | Where file checksums are computed: at write time, or when the run record is built | **Partly.** `utils/checksum.md5_file` is the shared checksum helper for the register ingests, and stage publish is all-or-none per unit, which makes write time the natural point. The run record reads remote footers in place without downloading files. Hashing at write time follows from those; where the hash is stored (footer or manifest) is open |
| D6 | Phase 1 item 6 | Whether the evidence reference replaces each sidecar's anchor columns or is added beside them, and where the span check runs | **Partly.** `contract.md` fixes the cost: adding is a minor bump with a reader back-fill; replacing is a major bump with a reader shim kept for the old major. The choice and the check's location are open |
| D7 | Phase 1 item 7 | Cell bbox coordinate space, sources with no page geometry, and branch split | **Answered.** `BBox` is normalised 0–1 with a top-left origin, and `Element.bbox` is already optional; cells follow both, so DOCX and spreadsheet cells carry null. The column is additive, a minor bump (`contract.md`). The merge cap requires splitting items 6 and 7 into sequential merges; whether they form a separately named Phase 1b is labelling only |
| D8 | Phase 2 | Workflow model option C (recipe file), and the condition language's scope | **Open.** Proposed in this plan, not decided |
| D9 | Phase 3 | DBOS against the existing queue for delivery retries and events | **Partly.** `decisions.md` makes the Postgres queue the distributed checkpoint and rejects new infrastructure beyond one datastore. DBOS also runs on Postgres, so that rule does not rule it out. Open |
| D10 | Phase 3 | Whether the egress decision changes once per-step destinations exist | **Answered for now.** `egress.md`: bundles may hold unredacted sources and access control is the host's responsibility. Revisit only if destinations change who holds the bundle |
| D11 | Phase 3 | Which destinations come first | **Open.** Nothing in the repository answers it |

## Conventions this plan holds to
- No quality or confidence scoring, and no scoring against ground truth in this repository.
- Extraction text stays verbatim; every new artefact is a sidecar.
- Each merge stays under 500 changed lines and updates `docs/architecture.md`, `docs/project-structure.md` and `docs/functional_requirements.md` where it changes them.
- No dependency is added without approval.
