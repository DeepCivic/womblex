# Trust baseline and recipes — plan

*Status: in progress (2026-10). Outstanding work only; completed work is recorded in `CHANGELOG.md`, `decisions.md` and the enduring docs, not here. Scope, settled decisions and the dependency assessment are in `decisions.md` ("Trust and recipes", "Dependencies assessed"). Phase 1 needs no design decision and goes first; each later phase opens with the decisions it needs before its first merge. Each merge removes its item from this document as it lands, and the document is deleted once its last item ships.*

## Findings
- **Limits the merged PRs recorded.** #197's are in `decisions.md` ("A trigger fires a dispatch"). #192: replay of a coordinator that caught a failed child was not crash-tested. Every DBOS merge exercises Postgres through CI only.
- **The local stage sequence still lives in comments.** `configs/default-isaacus.yaml` documents the full pipeline as commented per-stage commands. Nothing saves a stored workflow except code (no CLI or API), so no shipped workflow can replace them yet (decision D-2.2).
- **Enduring docs have drifted.** The README command table lacks `validate`, `plan` and `dispatch-workflow`; its documentation index lacks `egress.md`, `service-api.md`, `money-extraction.md`, `evaluation.md`, `deployment-images.md`, `heuristics_disambiguation.md` and `DESIGN.md`; `functional_requirements.md` has no stored-workflow requirement; `architecture.md` holds workflows in one paragraph; `steering.md` row 15 is stale; three tests cite docs that do not exist (`ui-plan.md`, `ground-truth-units.md`, `ground-truth-review.md`).

## Phase 1 — no decisions required (branch)
Everything here is settled in `decisions.md` or in a reviewed PR.

- **1.3 The `enrichment` registry slot.** Default `kanon-2-enricher`, the `womblex.models.enrichment` entry-point group, `enrichment.model`. The contract's Isaacus gate, `unserved_models` and the model check key off the provider; AI-chunking reuse and `persist_document` only for a provider that returns a Document, token splitting otherwise. Paired womblex-benchmark merge, since `is_default_group` labels its reports. Docs: Requirement 24 (the slot, and "the enrichment provider" out of its out-of-scope line), Requirement 12, `models.md`, `architecture.md` (Enrichment), `decisions.md` ("Local enrichment fallback" from position to shipped).
- **1.4 The `embedding` slot**, the same pattern: an Isaacus adapter, `embedding.model`, a paired benchmark merge. Docs: Requirement 24, `models.md`, CLAUDE.md rows.
- **1.4.5 `model-plugins.md` provider sections** (docs only): the enrichment and embedding interfaces, the declared limits, what Womblex keeps, and a minimal entry-point example.
- **1.5 OpenLineage-shaped export of the run record.** `store/lineage.py` maps the run record (observed, never config) to RunEvent JSON; `womblex manifest --shards DIR --lineage OUT`; tested against the published schema vendored under `fixtures/`. No dependency. Docs: `architecture.md` (the run record), `contract.md`, README command table, CHANGELOG.
- **1.6 Stocktake S-1 (baseline).** The full checklist below, plus the drift in Findings: README command table and documentation index; a new stored-workflows requirement absorbing the Requirement 3 TO-DO; a "Workflows and distributed execution" section in `architecture.md`; `steering.md` row 15; the three dead doc citations in tests; #198's summary that Phase 2 shipped, written before #197 re-landed. Split into S-1a and S-1b if it passes the cap.

## Phase 2 — retry and workflow operation (branch)
**Decisions before the first merge:**
- **D-2.1 Retry status.** Whether the latest attempt counts in a run's rollup, with the original listed `failed` and naming its retry; whether retry is CLI-only for now; whether a retry offers to re-run the later stages that ran after the failure.
- **D-2.2 Workflow authoring.** Whether `womblex workflow save | list | show | delete` and a shipped `configs/workflows/default-isaacus.yaml` are in scope.
- **D-2.3 Scheduling a workflow a firing would refuse.** Refuse at `schedule-workflow`, or warn.
- **D-2.4 Workflow guide.** A new `docs/workflows.md`, or sections in the README and `architecture.md`.
- **D-2.5 Doc-currency test.** Whether the stocktakes gain a test holding the README command table to the CLI parser and the CLAUDE.md module table to `src/`.

**Merges:**
- **2.1 Retry a failed stage by fork.** Re-dispatching a failed stage returns its recorded failure; DBOS's `resume_workflow` does not restart it, `fork_workflow` does, keeping the steps before the chosen one. Refused before the fork: a stage not failed, still running, owned by another owner, or failing its own pre-launch checks (a retry is a launch). Forks from the first failed unit as `run:stage:retry:N`; status per D-2.1. First task: confirm the DBOS 3.2 `fork_workflow` signature and whether it carries attributes. Docs: `decisions.md` (replaces "a retry is a plan follow-up"), Requirements 14 and 16, `architecture.md`, README, CLAUDE.md rows, `project-structure.md`.
- **2.2 Workflow authoring and the shipped default workflow** (per D-2.2). Replaces the comment sequence in `configs/default-isaacus.yaml`, keeping `ui/presets.py` and the demo run consistent as that file requires. Docs: README quick start and command table, `composable-design.md` (CLI), the stored-workflows requirement, CLAUDE.md rows, `project-structure.md`.
- **2.3 Schedule-time checks** (per D-2.3). Docs: the stored-workflows requirement, `decisions.md`.
- **2.3.5 Workflow guide** (per D-2.4, docs only).
- **2.4 Stocktake S-2**, with the currency test if D-2.5 adds it.

## Phase 3 — integration (branch)
**Decisions and approvals before the first merge:**
- **D-3.1 Object-store destination on fleet runs.** Corpus-only, which keeps the recorded egress boundary (egress does not read a run ingested from an object store); or reopen that decision so egress fetches sources from the ingest store, which changes Requirement 23's last two criteria.
- **D-3.2 pgvector text layer.** Masked `clean_text` only, or raw chunk text where a workflow declares it.
- **D-3.3 Webhooks.** HMAC signing with an environment secret, and the event list: extraction settled, stage done or failed, run done, delivery done or failed.
- **D-3.4 Dependency approval.** `httpx` declared in core (approved under D11) and a `[pgvector]` extra.

**Merges:**
- **3.1 Dependencies only.** `pyproject.toml` and `uv.lock`.
- **3.2 Delivery model.** Destination targets validated per kind at save, and an unbuilt kind refused there. A delivery is a DBOS workflow (`run:deliver:<n>`) with bounded retries, enqueued after the last stage whether or not one failed, and labelled `ran_after_failed`; locally, delivery follows the steps. The run record gains `deliveries` (record version 7), read by a new `store/` module beside `run_manifest.py`. A failed delivery labels the run and never stops it. Docs: `decisions.md`, Requirement 23 or a new destinations requirement, `architecture.md`, CLAUDE.md rows.
- **3.3 Object-store destination** (per D-3.1), built on `build_bundle`. Docs: `egress.md`, Requirement 23.
- **3.4 Postgres with pgvector** (per D-3.2): an idempotent upsert keyed `(run_id, source_hash, chunk_index, content_type)`. Docs: `contract.md` ("Safe to hand onward"), README installation extras.
- **3.5 Webhooks** (per D-3.3), each POST a DBOS workflow with bounded retries. Docs: `service-api.md` and README (both say callers poll and there are no webhooks), Requirement 22.
- **3.5.5 `egress.md` becomes "Egress and destinations"** with a payload reference (docs only).
- **3.6 Stocktake S-3.**

## Phase 4 — operation (branch)
**Decisions before the first merge:**
- **D-4.1 Plain-language status.** The vocabulary, shared with the webhook events and the lineage event types, and its surfaces: `womblex jobs --run`, an additive `summary` on `/v1/runs/{id}`, the console. It sits beside the exact status values Requirement 16 renders, never in place of them.
- **D-4.2 A versioned `graph-refresh` output.** A new sidecar keyed by the chunks' digest, leaving the enrichment sidecars as enrich wrote them, or a version in the file name.

**Merges:**
- **4.1 `RunBoard.describe` and its surfaces** (per D-4.1). Docs: Requirements 16 and 22, `service-api.md`, `DESIGN.md`.
- **4.2 The new graph-refresh sidecar** (per D-4.2); readers (`analyse/query.py`, the Chunk Inspector's entities overlay) prefer it. Docs: `contract.md` (minor version), CLAUDE.md rows.
- **4.3 Stop rewriting in place.** The contract moves from `MutationMode.IN_PLACE` to `SIDECAR`, with `PRODUCER_OF`, `store/retention.py`, `workflow_order` and `workflow_plan` updated; older runs fall back to the in-place `chunk_index`. `layout` keeps its in-place replace. Docs: CLAUDE.md rows, `project-structure.md`.
- **4.3.5 Docs:** `composable-design.md` (its three "rewritten in place" passages), Requirement 12, `decisions.md` ("Graph-edge refresh"), `architecture.md`, README.
- **4.4 Stocktake S-4 (final).** Retire this plan into `decisions.md` and `CHANGELOG.md`, then delete it.

## Stocktake checklist
The last merge of every phase, docs only, its record in the PR body rather than a new file. It covers every enduring document: README, CHANGELOG, CLAUDE.md, DESIGN.md, THIRD_PARTY_DATA.md, `fixtures/synthetic/README.md`, the PR template and every `docs/*.md`. It never edits `docs/accuracy/` (written by womblex-benchmark). Each check carries its evidence (a grep, a listing or a command):
1. Module tables in CLAUDE.md and `project-structure.md` against the `src/` tree.
2. The README command table against the CLI parser's subcommands.
3. The Requirements "Outstanding" index against the TO-DO blocks.
4. `decisions.md` entries marked "not built", "position" or "wanted" that have since shipped.
5. `CHANGELOG.md` Unreleased covers the phase's merges.
6. Every document cited (in docs, code comments and tests) exists.
7. No section-sign character and no line-number references.
8. Each mechanism claim the phase touched holds against the code.

## Conventions this plan holds to
- No quality or confidence scoring, and no scoring against ground truth in this repository.
- Extraction text stays verbatim; every new artefact is a sidecar.
- Each merge stays under 500 changed lines, passes `ruff`, `mypy src/` and the suite, and carries its own documentation: `CHANGELOG.md`, the CLAUDE.md module rows, `docs/architecture.md`, `docs/project-structure.md`, `docs/functional_requirements.md` and a `decisions.md` entry for any call it makes, wherever it changes them. A `.5` merge carries only further material: a guide or a restructured section.
- A dependency change is its own merge, approved before it lands.
- A run is an arrow: every refusal happens before launch; a retry is a launch and runs its stage's pre-launch checks again; delivery, webhook and lineage failures label the run and never stop it.
- Recorded boundaries stand unless a decision above reopens them.
- A change to `SLOTS` or `DEFAULT_MODELS` is raised with its womblex-benchmark pair.
- Files near the 750-line cap (`cli/cloud.py`, `cloud/stage_runner.py`, `store/run_manifest.py`, `cloud/stage_contracts.py`) take no new responsibilities: new CLI goes in `cli/workflow.py`, new run-record readers in their own `store/` modules.
- Every merge targets `main`. A stacked PR is retargeted to `main` before it merges, never merged into its base branch.
