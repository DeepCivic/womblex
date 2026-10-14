# Trust baseline and recipes — plan

*Status: in progress (2026-10). Outstanding work only; completed work is recorded in `CHANGELOG.md`, `decisions.md` and the enduring docs, not here. Scope, settled decisions and the dependency assessment are in `decisions.md` ("Trust and recipes", "Dependencies assessed"). Each phase is a branch of sequential merges under the 500-line cap. Each merge removes its item from this document as it lands, and the document is deleted once its last item ships.*

## Findings
- **The local stage sequence still lives in comments.** `womblex run` extracts only. A stored workflow runs a step list locally (`run-workflow`), but `configs/default-isaacus.yaml` still documents the full pipeline as a commented sequence of per-stage commands run by hand.

## Phases

### Phase 0 follow-up: retry a failed stage
Re-dispatching a stage that failed returns its recorded failure. DBOS's `resume_workflow` does not restart a failed workflow, but `fork_workflow` does: it starts a copy from a chosen step and keeps the steps before it, so only the failed unit and those after it run again. The copy has a new workflow id and the failed original stays on the board, so this merge decides how a run's status counts the original once its retry succeeds. Its own merge.

### Enrichment provider slot (own merge, after the trust baseline)
An `enrichment` slot in `utils/model_registry.py`, default `kanon-2-enricher`, with providers installed through `womblex.models.enrichment` entry points; the provider contract, the limits each provider declares and what Womblex keeps are in `decisions.md` ("Local enrichment fallback"). Embedding follows the same pattern. The GLiNER provider, in development by the maintainer, is the first alternative and follows this item. No provider other than Isaacus ships with it.

### Phase 2 — workflows (branch)
- Standing triggers on DBOS schedules. Triggers are stored and validated but nothing schedules them.

### Phase 3 — integration (branch)
- Destinations as delivery targets beside egress, each with its own status and bounded retries: object storage, Postgres with pgvector, and webhooks, in that order (`decisions.md`, "Trust and recipes").
- Delivery retries and webhook events run as DBOS workflows.
- An OpenLineage-shaped export of the run record.

### Phase 4 — operation (branch)
- Run status in plain language, consistent with events and lineage.
- `graph-refresh` writing a new versioned sidecar instead of rewriting the entity and edge sidecars in place.

## Conventions this plan holds to
- No quality or confidence scoring, and no scoring against ground truth in this repository.
- Extraction text stays verbatim; every new artefact is a sidecar.
- Each merge stays under 500 changed lines and updates `docs/architecture.md`, `docs/project-structure.md` and `docs/functional_requirements.md` where it changes them.
- No dependency is added without approval.
