# Console split — plan

*Status: not started (2026-10). This document is deleted when the console's last trace leaves Womblex.*

## Context
The console (`womblex ui`) is two parts: the SvelteKit SPA in `ui/`, and the FastAPI backend in `src/womblex/ui/`, which reads run state and dispatches to the queue. It exists for local deployment experimentation, adds little for most users, and may be deprecated. It moves to its own repository, `womblex-console`.

The two projects stay **aligned, not coupled**. The console repository does almost all of the work. Womblex does only what it needs to stop shipping the console, and it gains no surface, test or doc that exists for the console's sake.

## Principles
- Dependency runs one way. The console depends on a pinned Womblex release. Womblex never imports, tests or documents the console.
- The console adapts. When Womblex changes, the console repository moves its pin and its imports; Womblex carries no obligation to keep it working.
- Deprecation is archiving the console repository. Nothing in Womblex changes.

## Step 1: create `womblex-console` (console repository; Womblex unchanged)
- Seed it with `git filter-repo` over `ui/`, `src/womblex/ui/`, `Dockerfile.ui`, `tests/test_ui_*.py` and `tests/_ui_helpers.py`, keeping history.
- Rename the package to `womblex_console`, so it cannot collide with the `womblex.ui` still present in the Womblex release it first pins. A `womblex-console` entry point keeps the `womblex ui` flags.
- Dependencies: `womblex` pinned to the current release, `fastapi`, `uvicorn`; `pytest` and `httpx` for dev.
- Anything it needs that Womblex treats as private is copied in, not made public in Womblex.
- Its own CI (frontend build and npm audit, Python lint, type check and tests, image build), image publishing, a compose file that points at an existing Womblex store and DSN, and its own docs.

## Step 2: remove the console from Womblex (Womblex)
Library code imports `womblex.ui` today. The pieces it uses are Womblex's own concerns, so they move into the library; this is the only Womblex work that is not deletion:
- `ui/presets.py` → `womblex/presets.py`. Workflows resolve presets (`cloud/workflow_runner.py`, `workflow_store.py`, `workflow_dispatch.py`, `workflow_plan.py`, `workflow_validate.py`).
- The `UISettings` dataclass and `deployment_paths`, used by `cloud/dispatch.py` and `api/app.py`, move to `cloud/` as fastapi-free deployment settings.
- What `womblex serve` reads from the console modules (`get_manifest_rows`, `StoreUnreachable`, saved-preset loading, the store and queue connectivity probes) moves to `store/`, `cloud/` or `presets.py`.
- `CHECKPOINT_DIRNAMES` moves beside `STAGE_CONTRACTS`.

Then deletion:
- `src/womblex/ui/`, `cli/ui.py` and its registration, the `test_ui_*` modules and `_ui_helpers.py`. Non-console tests that touch the moved pieces are repointed (`test_config.py`, `test_pipeline_order.py`, `test_model_check.py`, `test_demo_corpus.py`, the `test_workflow_*.py` set); the `_chunk_relative_pii` case in `test_evidence_sidecars.py` leaves with the console.
- `ui/`, `Dockerfile.ui`, the console jobs in `ci.yml` and `publish-images.yml`, and the `ui` service in `docker-compose*.yml`.
- The `[ui]` extra, as its own dependency merge with human approval.

Merges stay under the 500-line cap: the moves first, suite green; the pure-deletion merges exceed it and say so in their PR bodies for human approval. The phase closes with a docs stocktake: the README command table, the CLAUDE.md module table, `docs/architecture.md`, `docs/project-structure.md`, `docs/deployment-images.md`, a decision entry in `docs/decisions-operations.md`, Requirement 19 in `docs/functional_requirements.md`, and `CHANGELOG.md`.

## Step 3: the console takes the new Womblex release (console repository)
Bump the pin and repoint imports at the moved modules (`womblex.presets`, the deployment settings). Its CI checks it against the published release.

## Verification
- Womblex: `uv run ruff check src/ tests/`, `uv run mypy src/`, `uv run python -m pytest tests/ -m "not benchmark"`; `grep -rn "womblex.ui\|womblex ui" src tests` finds nothing; `uv lock --check` passes; `test_doc_currency.py` is green.
- Console: `npm run check && npm run build`, pytest against the pinned Womblex, the image build, and `womblex-console --output-root output/` serving a run written by `womblex run`.
