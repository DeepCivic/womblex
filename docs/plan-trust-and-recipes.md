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


- **D-2.5 Doc-currency test.** Whether the stocktakes gain a test holding the README command table to the CLI parser and the CLAUDE.md module table to `src/`. ANSWERED: I need more information, I don't understand the implications.
  *In plain terms:* stocktake checks 1 and 2 are done by hand each phase: compare the README's list of commands with the commands `womblex` actually has, and CLAUDE.md's module table with the files under `src/`. A test would do the comparison on every CI run instead.
  - *Add it:* drift is caught in the PR that causes it, not up to a phase later, and stocktakes get shorter. The cost is one small test file. A PR that adds a command or a module then fails CI until its doc row is written, which is what the conventions already require.
  - *Don't:* nothing breaks; drift is still found, but only at the next stocktake, by hand.
  - *Recommendation:* add it. It enforces a rule the plan already holds, and fails before a merge, never during a run.

**Merges:**
- **2.1 Retry a failed stage by fork.** Re-dispatching a failed stage returns its recorded failure; DBOS's `resume_workflow` does not restart it, `fork_workflow` does, keeping the steps before the chosen one. Refused before the fork: a stage not failed, still running, owned by another owner, or failing its own pre-launch checks (a retry is a launch). Forks from the first failed unit as `run:stage:retry:N`; status per D-2.1. First task: confirm the DBOS 3.2 `fork_workflow` signature and whether it carries attributes. Docs: `decisions.md` (replaces "a retry is a plan follow-up"), Requirements 14 and 16, `architecture.md`, README, CLAUDE.md rows, `project-structure.md`.
- **2.3 Schedule-time checks** (per D-2.3). Docs: the stored-workflows requirement, `decisions.md`.
- **2.3.5 Workflow guide** (per D-2.4, docs only), saying that a workflow is saved only through code (D-2.2).
- **2.4 Stocktake S-2**, with the currency test if D-2.5 adds it.

## Phase 3 — integration (branch)
**Decisions and approvals before the first merge:**
Phase 3 lets a finished run send its output somewhere by itself (a "destination"), rather than someone running `womblex egress` or reading the API. Note for D-2.2: destinations live on a stored workflow, and a workflow is saved only through code, so until authoring exists a destination is configured the same way.

- **D-3.1 Object-store destination on fleet runs.** Corpus-only, which keeps the recorded egress boundary (egress does not read a run ingested from an object store); or reopen that decision so egress fetches sources from the ingest store, which changes Requirement 23's last two criteria.
  *In plain terms:* egress today builds a bundle of the extracted corpus *plus* the original PDFs, and finds the PDFs on local disk. A fleet run read its PDFs from an object store (S3), so egress refuses it. The question is what a fleet run's object-store destination sends.
  - *Corpus-only:* it sends the extracted parquet and manifest, not the PDFs. It's small and safe, and no recorded decision changes. A consumer that wants the PDFs fetches them from the ingest bucket itself, using the manifest's `source_relpath`.
  - *Fetch sources:* it copies the PDFs out of the ingest store too, checking each by hash, so the bundle is complete. It's more code, and it reverses a recorded boundary (Requirement 23 and `egress.md` change). It also copies raw, unmasked government documents to a second place, which matters if the destination is less locked down than the ingest bucket.
  - *Recommendation:* corpus-only now. Reopen it if a consumer needs the PDFs and can't reach the ingest bucket.

- **D-3.2 pgvector text layer.** Masked `clean_text` only, or raw chunk text where a workflow declares it.
  *In plain terms:* pgvector puts each chunk's text and embedding into a Postgres table, for search. The question is which text goes in the table: the PII-masked version (`<PERSON_1>`) or the original (real names).
  - *Masked only:* nothing with personal names leaves Womblex through this route, and it matches `contract.md`'s "safe to hand onward" rule. Search still works, because the embeddings were made from the raw text; only the returned text is masked. It needs the `pii` stage to have run first, so a workflow with this destination and no `pii` would be refused before launch.
  - *Raw if declared:* search results show real names, which reviewers may need. But a database holding raw text is now a store of personal information and needs the access controls that go with that, and one setting turns it on.
  - *Recommendation:* masked only. Add raw later, as its own decision, if a named use needs it.

- **D-3.3 Webhooks.** HMAC signing with an environment secret, and the event list: extraction settled, stage done or failed, run done, delivery done or failed.
  *In plain terms:* today a caller has to keep asking the API "is my run done?" (polling). A webhook makes Womblex POST to the caller's URL when something happens.
  - *HMAC signing:* each POST carries a signature made with a shared secret held in an environment variable. The receiver recomputes it to prove the message came from Womblex and wasn't altered. It's the standard approach (GitHub and Stripe do the same) and costs one secret to manage per deployment. Without it, anyone who learns the URL can fake a "run done".
  - *Event list:* fewer events are simpler and less noisy; more give finer progress. The proposed five cover "can I start reading?" (extraction settled, run done) and "did something go wrong?" (stage failed, delivery failed). Per-document or per-batch events are left out because a large run would send thousands.
  - *Failures:* a webhook that can't be delivered is retried a few times, then recorded on the run. It never stops the run.
  - *Recommendation:* yes to both as proposed.

- **D-3.4 Dependency approval.** `httpx` declared in core (approved under D11) and a `[pgvector]` extra.
  *In plain terms:* `httpx` is the HTTP client the webhooks send with. It's small and already installed on every install (the Isaacus SDK and `huggingface-hub` depend on it), but Womblex declares it only in `[dev]`, for the test client. The `[pgvector]` extra adds the Postgres vector driver only for installs that want that destination, so nobody else carries it. Per the conventions, this is its own merge (3.1), made only after you approve it.

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
