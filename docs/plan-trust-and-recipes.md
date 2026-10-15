# Trust baseline and recipes — plan

*Status: in progress (2026-10). Outstanding work only; completed work is recorded in `CHANGELOG.md`, `decisions.md` and the enduring docs, not here. Scope, settled decisions and the dependency assessment are in `decisions.md` ("Trust and recipes", "Dependencies assessed"). Each phase opens with the decisions it needs before its first merge. Each merge removes its item from this document as it lands, and the document is deleted once its last item ships.*

## Findings
- **Limits the merged PRs recorded.** #197's are in `decisions.md` ("A trigger fires a dispatch"). #192: replay of a coordinator that caught a failed child was not crash-tested. Every DBOS merge exercises Postgres through CI only.
- **The local stage sequence stays in comments.** `configs/default-isaacus.yaml` documents the full pipeline as commented per-stage commands. Nothing saves a stored workflow except code (no CLI or API), and D-2.2 keeps it that way for this plan, so the comments are the shipped sequence.

## Phase 2 — retry and workflow operation (branch)
**Decisions before the first merge:**
- **D-2.1 Retry status.** Whether the latest attempt counts in a run's rollup, with the original listed `failed` and naming its retry; whether retry is CLI-only for now; whether a retry offers to re-run the later stages that ran after the failure. ANSWERED: Yes

- **D-2.2 Workflow authoring.** Out of scope: no `womblex workflow save | list | show | delete` and no shipped default workflow. A stored workflow is still saved only through code (`WorkflowStore`), and the workflow guide (2.3.5) says so.

- **D-2.3 Scheduling a workflow a firing would refuse.** Refuse at `schedule-workflow`, or warn. ANSWERED: Refuse at `schedule-workflow`

- **D-2.4 Workflow guide.** A new `docs/workflows.md`, or sections in the README and `architecture.md`. ANSWERED: A new `docs/workflows.md`

- **D-2.5 Doc-currency test.** ANSWERED: add it. A test holds the README command table to the CLI parser and the CLAUDE.md module table to `src/`, so drift fails CI in the PR that causes it; stocktake checks 1 and 2 cite the test instead of being done by hand.

**Merges:**
- **2.1 Retry a failed stage by fork.** Re-dispatching a failed stage returns its recorded failure; DBOS's `resume_workflow` does not restart it, `fork_workflow` does, keeping the steps before the chosen one. Refused before the fork: a stage not failed, still running, owned by another owner, or failing its own pre-launch checks (a retry is a launch). Forks from the first failed unit as `run:stage:retry:N`; status per D-2.1. First task: confirm the DBOS 3.2 `fork_workflow` signature and whether it carries attributes. Docs: `decisions.md` (replaces "a retry is a plan follow-up"), Requirements 14 and 16, `architecture.md`, README, CLAUDE.md rows, `project-structure.md`.
- **2.3 Schedule-time checks** (per D-2.3). Docs: the stored-workflows requirement, `decisions.md`.
- **2.3.5 Workflow guide** (per D-2.4, docs only), saying that a workflow is saved only through code (D-2.2).
- **2.3.7 Doc-currency test** (per D-2.5). Fixes any drift it finds on landing. Docs: the stocktake checklist below, `decisions.md`.
- **2.4 Stocktake S-2.**

## Phase 3 — integration (branch)
**Decisions and approvals before the first merge:**
Phase 3 lets a finished run send its output somewhere by itself (a "destination"), rather than someone running `womblex egress` or reading the API. Destinations live on a stored workflow, and per D-2.2 a workflow is saved only through code, so a destination is configured the same way.

- **D-3.1 Sources on object-store runs.** ANSWERED: reopen the egress boundary. Egress, and the object-store destination built on it, can fetch a run's source documents from the ingest store it was read from: each by the manifest's `source_relpath`, checked against `source_hash`, with the same per-document status as local resolution. Including sources stays the caller's choice per export or destination (`include_sources`, as `build_bundle` has today). Womblex is a backend service; whether raw documents may go to a destination is the deploying system's call, not a limit Womblex sets. Requirement 23's object-store refusal goes.

- **D-3.2 pgvector text layer.** ANSWERED: both. Masked `clean_text` is the default; raw chunk text is written where the workflow's destination declares it. Each row and the table record which layer it holds, and the masked layer needs `pii` in the workflow (refused before launch otherwise); the raw layer does not. As with D-3.1, whether raw text is appropriate is the deploying system's call.

- **D-3.3 Webhooks.** ANSWERED: yes. HMAC signing with an environment secret, and five events: extraction settled, stage done or failed, run done, delivery done or failed. No per-document or per-batch events. A webhook that cannot be delivered is retried a bounded number of times, then recorded on the run; it never stops the run.

- **D-3.4 Dependency approval.** ANSWERED: approved. `httpx` declared in core (already installed through the Isaacus SDK and `huggingface-hub`; today declared only in `[dev]`) and a `[pgvector]` extra.

**Merges:**
- **3.1 Dependencies only.** `pyproject.toml` and `uv.lock`.
- **3.2 Delivery model.** Destination targets validated per kind at save, and an unbuilt kind refused there. A delivery is a DBOS workflow (`run:deliver:<n>`) with bounded retries, enqueued after the last stage whether or not one failed, and labelled `ran_after_failed`; locally, delivery follows the steps. The run record gains `deliveries` (record version 7), read by a new `store/` module beside `run_manifest.py`. A failed delivery labels the run and never stops it. Docs: `decisions.md`, Requirement 23 or a new destinations requirement, `architecture.md`, CLAUDE.md rows.
- **3.3 Sources from the ingest store** (per D-3.1). `build_bundle` resolves a run ingested from an object store against that store, by `source_relpath` and checked by hash, reusing the per-document fetcher the `layout` contract builds; the object-store refusal is removed. Docs: `egress.md`, Requirement 23 (its last two criteria), `decisions.md` (the egress boundary), CLAUDE.md rows.
- **3.3.1 Object-store destination**, built on `build_bundle`, with `include_sources` per destination. Docs: `egress.md`, Requirement 23.
- **3.4 Postgres with pgvector** (per D-3.2): an idempotent upsert keyed `(run_id, source_hash, chunk_index, content_type)`, masked text by default and raw where the destination declares it, the layer recorded on each row. Docs: `contract.md` ("Safe to hand onward" names the raw option and who decides it), README installation extras.
- **3.5 Webhooks** (per D-3.3), each POST a DBOS workflow with bounded retries. Docs: `service-api.md` and README (both say callers poll and there are no webhooks), Requirement 22.
- **3.5.5 `egress.md` becomes "Egress and destinations"** with a payload reference (docs only).
- **3.6 Stocktake S-3.**

## Phase 4 — operation (branch)
**Decisions before the first merge:**
- **D-4.1 Plain-language status.** The vocabulary, shared with the webhook events and the lineage event types, and its surfaces: `womblex jobs --run`, an additive `summary` on `/v1/runs/{id}`, the console. It sits beside the exact status values Requirement 16 renders, never in place of them. ANSWER: unclear on the decision here, yes plain language is good??
  *In plain terms:* the run status today is exact but technical: per-stage counts of `pending` / `running` / `done` / `failed`, with DBOS workflow ids. The proposal adds one plain sentence per run, such as "Extracted 1,204 of 1,210 documents; enrichment running; 6 documents failed (see list)", worked out from the exact counts, which stay where they are. Plain language is already agreed; the decision is only:
  - *Which words:* a short, fixed set of phrases (for example: waiting, extracting, processing, finished, finished with failures). The same set names the webhook events and the lineage events, so one word means one thing everywhere.
  - *Where it shows:* `womblex jobs --run`, a new `summary` field on `/v1/runs/{id}` (added, so existing callers are unaffected), and the console.
  - *Recommendation:* approve both. The cost is one function and three places that show it.
- **D-4.2 A versioned `graph-refresh` output.** A new sidecar keyed by the chunks' digest, leaving the enrichment sidecars as enrich wrote them, or a version in the file name. ANSWER: unclear why we need this or don't need it.
  *In plain terms:* `graph-refresh` is the only downstream stage that edits another stage's file. Enrich writes `*.enrichment_entities.parquet` before chunks exist, so each mention's `chunk_index` is `-1`. Then `graph-refresh` reopens that file and fills the numbers in. That causes three problems:
  - The file no longer matches what enrich wrote, so its footer stamp (stage, run, digest) describes a file that has since been changed.
  - If chunking is re-run with different settings, the filled-in numbers point at chunks that no longer exist until graph-refresh runs again, and nothing on the file says which chunks it was matched against.
  - Re-running enrich wipes the refresh, and the ordering checks have to treat the file as "stale" in special ways (`workflow_order`, `workflow_plan`).

  A separate sidecar written by graph-refresh fixes all three. Enrich's file stays as written, the new file records which chunks it matched (the chunks' digest), and a reader can tell when it is out of date. That is merge 4.3, "every new artefact is a sidecar", applied to the last exception.
  - *Need it?* Only if chunks get re-run on a run that has already been refreshed, or if anyone relies on footers to say who wrote a file. If neither ever happens, the in-place rewrite works and Phase 4's merges 4.2 and 4.3 can be dropped.
  - *Recommendation:* the new sidecar keyed by the chunks' digest, not a version in the file name. A version number says *that* the file changed, not *which chunks* it matches.

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
- Recorded boundaries stand unless a decision above reopens them.
- A change to `SLOTS` or `DEFAULT_MODELS` is raised with its womblex-benchmark pair.
- Files near the 750-line cap (`cli/cloud.py`, `cloud/stage_runner.py`, `store/run_manifest.py`, `cloud/stage_contracts.py`) take no new responsibilities: new CLI goes in `cli/workflow.py`, new run-record readers in their own `store/` modules.
- Every merge targets `main`. A stacked PR is retargeted to `main` before it merges, never merged into its base branch.
