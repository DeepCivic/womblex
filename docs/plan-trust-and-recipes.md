# Trust baseline and recipes — plan

*Status: in progress (2026-10). Outstanding work only; completed work is recorded in `CHANGELOG.md`, `decisions.md` and the enduring docs, not here. Scope, settled decisions and the dependency assessment are in `decisions.md` ("Trust and recipes", "Trust and recipes: the calls taken before Phases 2 to 4", "Dependencies assessed"); a merge cites its decision by number (D-2.1 to D-4.2). Each merge removes its item from this document as it lands, and the document is deleted once its last item ships.*

## Findings
- **Limits the merged PRs recorded.** #197's are in `decisions.md` ("A trigger fires a dispatch"). #192: replay of a coordinator that caught a failed child was not crash-tested. Every DBOS merge exercises Postgres through CI only.
- **The local stage sequence stays in comments.** `configs/default-isaacus.yaml` documents the full pipeline as commented per-stage commands. Nothing saves a stored workflow except code (no CLI or API), and D-2.2 keeps it that way for this plan, so the comments are the shipped sequence.

## Phase 2 — retry and workflow operation (branch)
**Merges:**
- **2.1 Retry a failed stage by fork.** Re-dispatching a failed stage returns its recorded failure; DBOS's `resume_workflow` does not restart it, `fork_workflow` does, keeping the steps before the chosen one. Refused before the fork: a stage not failed, still running, owned by another owner, or failing its own pre-launch checks (a retry is a launch). Forks from the first failed unit as `run:stage:retry:N`; status per D-2.1. First task: confirm the DBOS 3.2 `fork_workflow` signature and whether it carries attributes. Docs: `decisions.md` (replaces "a retry is a plan follow-up"), Requirements 14 and 16, `architecture.md`, README, CLAUDE.md rows, `project-structure.md`.
- **2.3 Schedule-time checks** (per D-2.3). Docs: Requirement 27's TO-DO and its Outstanding entry, `decisions.md` ("A trigger fires a dispatch", its limits).
- **2.3.5 Workflow guide** (per D-2.4, docs only), saying that a workflow is saved only through code (D-2.2).
- **2.3.7 Doc-currency test** (per D-2.5). Fixes any drift it finds on landing. Docs: the stocktake checklist below.
- **2.4 Stocktake S-2.**

## Phase 3 — integration (branch)
Phase 3 lets a finished run send its output somewhere by itself (a "destination"), rather than someone running `womblex egress` or reading the API. Destinations live on a stored workflow, and per D-2.2 a workflow is saved only through code, so a destination is configured the same way.

**Merges:**
- **3.1 Dependencies only.** `pyproject.toml` and `uv.lock`.
- **3.2 Delivery model.** Destination targets validated per kind at save, and an unbuilt kind refused there. A delivery is a DBOS workflow (`run:deliver:<n>`) with bounded retries, enqueued after the last stage whether or not one failed, and labelled `ran_after_failed`; locally, delivery follows the steps. The run record gains `deliveries` (record version 7), read by a new `store/` module beside `run_manifest.py`. A failed delivery labels the run and never stops it. Docs: `decisions.md`, Requirement 23 or a new destinations requirement, `architecture.md`, CLAUDE.md rows.
- **3.3 Sources from the ingest store** (per D-3.1). `build_bundle` resolves a run ingested from an object store against that store, by `source_relpath` and checked by hash, reusing the per-document fetcher the `layout` contract builds; the object-store refusal is removed. Docs: `egress.md`, Requirement 23's TO-DO and its Outstanding entry, CLAUDE.md rows.
- **3.3.1 Object-store destination**, built on `build_bundle`, with `include_sources` per destination. Docs: `egress.md`, Requirement 23.
- **3.4 Postgres with pgvector** (per D-3.2): an idempotent upsert keyed `(run_id, source_hash, chunk_index, content_type)`, masked text by default and raw where the destination declares it, the layer recorded on each row. Docs: `contract.md` ("Safe to hand onward" names the raw option and who decides it), README installation extras.
- **3.5 Webhooks** (per D-3.3), each POST a DBOS workflow with bounded retries; events are named in D-4.1's words from the start, so 4.1 renames nothing. Docs: `service-api.md` and README (both say callers poll and there are no webhooks), `decisions.md` ("Consumers poll"), Requirement 22.
- **3.5.5 `egress.md` becomes "Egress and destinations"** with a payload reference (docs only).
- **3.6 Stocktake S-3.**

## Phase 4 — operation (branch)
**Merges:**
- **4.1 `RunBoard.describe` and its surfaces** (per D-4.1). Docs: Requirements 16 and 22, `service-api.md`, `DESIGN.md`.
- **4.2 The new graph-refresh sidecar** (per D-4.2); readers (`analyse/query.py`, the Chunk Inspector's entities overlay) prefer it. Docs: `contract.md` (minor version), CLAUDE.md rows.
- **4.3 Stop rewriting in place.** The contract moves from `MutationMode.IN_PLACE` to `SIDECAR`, with `PRODUCER_OF`, `store/retention.py`, `workflow_order` and `workflow_plan` updated; older runs fall back to the in-place `chunk_index`. `layout` keeps its in-place replace. Docs: CLAUDE.md rows, `project-structure.md`.
- **4.3.5 Docs:** `composable-design.md` (its three "rewritten in place" passages), Requirement 12, `decisions.md` ("Graph-edge refresh"), `architecture.md`, README.
- **4.4 Stocktake S-4 (final).** Retire this plan into `decisions.md` and `CHANGELOG.md`, then delete it.

## Stocktake checklist
The last merge of every phase, docs only, its record in the PR body rather than a new file. It covers every enduring document: README, CHANGELOG, CLAUDE.md, DESIGN.md, THIRD_PARTY_DATA.md, `fixtures/synthetic/README.md`, the PR template and every `docs/*.md`. It never edits `docs/accuracy/` (written by womblex-benchmark). Each check carries its evidence (a grep, a listing or a command):
1. Module tables in CLAUDE.md and `project-structure.md` against the `src/` tree (CLAUDE.md's by the doc-currency test once 2.3.7 lands).
2. The README command table against the CLI parser's subcommands (by the doc-currency test once 2.3.7 lands).
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
- Recorded boundaries stand unless a numbered decision reopens them.
- A change to `SLOTS` or `DEFAULT_MODELS` is raised with its womblex-benchmark pair.
- Files near the 750-line cap (`cli/cloud.py`, `cloud/stage_runner.py`, `store/run_manifest.py`, `cloud/stage_contracts.py`) take no new responsibilities: new CLI goes in `cli/workflow.py`, new run-record readers in their own `store/` modules.
- Every merge targets `main`. A stacked PR is retargeted to `main` before it merges, never merged into its base branch.
