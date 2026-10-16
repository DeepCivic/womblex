# Trust baseline and recipes — plan

*Status: in progress (2026-10); Phase 1 has shipped. Outstanding work only: completed work is recorded in `CHANGELOG.md` and the enduring docs, and the conventions every merge holds to (merge size, stocktakes, stacked PRs, dependency merges) are in CLAUDE.md. Scope, settled decisions and the dependency assessment are in `decisions-operations.md` ("Trust and recipes", "Trust and recipes: the calls taken before Phases 2 to 4") and `decisions-dead-ends.md` ("Dependencies assessed"); a merge cites its decision by number (D-2.1 to D-4.2). A `.5` merge carries only further documentation (a guide or a restructured section). Each merge removes its item from this document as it lands, and the document is deleted once its last item ships.*

## Phase 2 — retry and workflow operation (branch)
This phase finishes the stored-workflows work (`CHANGELOG.md`'s "Phase 2 merges 1–9b", an earlier numbering of this plan): the retry it deferred and the limits it recorded in `decisions-operations.md` ("A trigger fires a dispatch" and the DBOS decision). **Phase 3 does not open until every merge here has landed**, since destinations sit on the dispatch and firing paths these merges harden. There is no 2.2: D-2.2 dropped workflow authoring.

**Merges:**
- **2.3.2 Workflow and schedule tests on Postgres.** The schedule tests use SQLite and the CLI tests clear CI's `WOMBLEX_DB_DSN`, so nothing runs them on Postgres; #200's CI showed `schedule-workflow --status` failing on a Postgres database whose DBOS tables are not yet created (missing `workflow_schedules`), where SQLite prints nothing. Fix that and run the tests against CI's Postgres service. Adds the crash test for replaying a coordinator that caught a failed child (#192). Docs: the limits in `decisions-operations.md` (DBOS decision and "A trigger fires a dispatch").
- **2.3.5 Workflow guide** (per D-2.4, docs only), saying that a workflow is saved only through code (D-2.2).
- **2.3.7 Doc-currency test** (per D-2.5). Fixes any drift it finds on landing. Docs: CLAUDE.md's stocktake (its checks 1 and 2 cite the test).
- **2.4 Stocktake S-2**, which also confirms that the limits recorded by the stored-workflows phase are retired or deliberately kept as boundaries.

## Phase 3 — integration (branch)
Phase 3 lets a finished run send its output somewhere by itself (a "destination"), rather than someone running `womblex egress` or reading the API. Destinations live on a stored workflow, and per D-2.2 a workflow is saved only through code, so a destination is configured the same way.

**Merges:**
- **3.1 Dependencies only** (per D-3.4). `httpx` moves from the `[dev]` extra to core; a new `[pgvector]` extra holds pgvector's client. `pyproject.toml` and `uv.lock`, with human approval.
- **3.2 Delivery model.** Destination targets validated per kind at save, and an unbuilt kind refused there. A delivery is a DBOS workflow (`run:deliver:<n>`) with bounded retries, enqueued after the last stage whether or not one failed, and labelled `ran_after_failed`; locally, delivery follows the steps. The run record gains `deliveries` (record version 7), read by a new `store/` module beside `run_manifest.py`. A failed delivery labels the run and never stops it. Docs: `decisions-operations.md`, Requirement 23 or a new destinations requirement (open: decide before the merge starts), `architecture.md`, CLAUDE.md rows.
- **3.3 Sources from the ingest store** (per D-3.1). `build_bundle` resolves a run ingested from an object store against that store, by `source_relpath` and checked by hash, reusing the per-document fetcher the `layout` contract builds; the object-store refusal is removed. Docs: `egress.md`, Requirement 23's TO-DO and its Outstanding entry, CLAUDE.md rows.
- **3.3.1 Object-store destination**, built on `build_bundle`, with `include_sources` per destination. Docs: `egress.md`, Requirement 23, Requirement 27's "Destinations are stored but not acted on" TO-DO and its Outstanding entry narrowed to the kinds not yet built.
- **3.4 Postgres with pgvector** (per D-3.2): an idempotent upsert keyed `(run_id, source_hash, chunk_index, content_type)`, masked text by default and raw where the destination declares it, the layer recorded on each row; a masked destination in a workflow without `pii` is refused before launch. Docs: `contract.md` ("Safe to hand onward" names the raw option and who decides it), README installation extras.
- **3.5 Webhooks** (per D-3.3), each POST a DBOS workflow with bounded retries; events are named in D-4.1's words from the start, so 4.1 renames nothing. Docs: `service-api.md` ("Callbacks and webhooks (consumers poll)"), README (callers poll), `decisions-operations.md` ("Consumers poll"), Requirement 22, and Requirement 27's destinations TO-DO and its Outstanding entry, retired now that every kind is built.
- **3.5.5 `egress.md` becomes "Egress and destinations"** with a payload reference (docs only).
- **3.6 Stocktake S-3.**

## Phase 4 — operation (branch)
**Merges:**
- **4.1 `RunBoard.describe` and its surfaces** (per D-4.1): `womblex jobs --run`, an additive `summary` on `/v1/runs/{id}`, the console; the OpenLineage event (`store/lineage.py`, today always `COMPLETE`) takes its type from the same words. Docs: `contract.md` (the lineage export), Requirements 16 and 22, `service-api.md`, `DESIGN.md`.
- **4.2 The new graph-refresh sidecar** (per D-4.2); readers (`analyse/query.py`, the Chunk Inspector's entities overlay) prefer it. Docs: `contract.md` (minor version), CLAUDE.md rows.
- **4.3 Stop rewriting in place.** The contract moves from `MutationMode.IN_PLACE` to `SIDECAR`, with `PRODUCER_OF`, `store/retention.py`, `workflow_order` and `workflow_plan` updated; older runs fall back to the in-place `chunk_index`. `layout` keeps its in-place replace. Docs: CLAUDE.md rows, `project-structure.md`.
- **4.3.5 Docs:** `composable-design.md` (the outputs table's "rewritten in place" and the `graph-refresh` "in place" bullet), Requirement 12, `decisions.md` ("Graph-edge refresh"), `architecture.md`, README.
- **4.4 Stocktake S-4 (final).** Retire this plan into the decision docs and `CHANGELOG.md`, then delete it.
