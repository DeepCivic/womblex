# Cloud flexibility — plan

*Status: Phases A to E are built (the provider packages, Requirements 36 to 39); Phase F, the stocktake, remains. A piece is removed from this plan when it is built, and this document is deleted when its last phase lands.*

## Context

Adding a vendor, a hosted model or a storage backend costs more than its adapter. The
registry side is already light: a factory and one `register()` call, or an entry point in a
plugin package (Requirement 24). The weight is elsewhere:

- **Isaacus has a path of its own.** "Is this Isaacus" is a model trait (`isaacus`), read
  through `calls_isaacus` by `cloud/stage_contracts.py`, `utils/model_check.py`,
  `cloud/worker.py`, `utils/availability.py` and `cloud/resources.py`. Other paid models are
  checked through a reader's `ping()` / `ping_is_billed`. Two mechanisms answer one question.
- **The Isaacus deployment is process-wide.** `make_isaacus_client` returns a SageMaker client
  whenever `ISAACUS_SAGEMAKER_ENDPOINTS` is set, else the hosted one, and each call site
  (`cli/embed.py`, `cli/link.py`, `cloud/stage_runner.py`, `utils/model_check.py`, AI chunking)
  builds its own client.
- **Object stores are S3-shaped.** `store/remote.py` fixes the schemes
  (`SUPPORTED_PROTOCOLS`, `_REMOTE_PROTOCOLS`) and only `s3://` gets options
  (`storage_options_from_env`). `gs://` and `az://` are accepted but no backend is installed,
  so they fail at first use.
- **Documentation steps are remembered, not checked.** A billed model's `docs/models.md` entry
  and **Paid:** line rely on the author.

**Requirement numbers:** The requirements this plan carried (36 to 39) are all written in `docs/functional_requirements.md`. As each requirement is delivered, it
is written there as built, in that file's conventions, and removed from this plan in the same
merge.

**In scope:**
- a deployment registry shared by every model a deployment serves
- object stores as whatever fsspec can open, configured by fsspec's own options
- provider packages and suite-checked documentation steps for a new provider
- an OpenAI-compatible embedding provider, as the second real vendor

**Out of scope:**
- Isaacus's reranking, classification and extractive QA functions
- an import-linter architecture guard
- moving `dbos` or `psycopg` behind a provider
- `gcsfs` / `adlfs` extras (a dependency change, its own merge, when a deployment needs one)

**Merge rules (CLAUDE.md):**
- Each merge is ≤500 changed lines and passes ruff, mypy and pytest on its own.
- A file near the 750-line cap gets no new responsibility; new code goes in a new module beside it:
  - `cloud/stage_runner.py`: 677
  - `store/run_manifest.py`: 635
  - `cloud/stage_contracts.py`: 600
  - `utils/model_check.py`: 543
- A new module lands in the CLAUDE.md table, `docs/architecture.md` and `docs/project-structure.md`
  in the same merge (`tests/test_doc_currency.py`).
- A merge that delivers a requirement writes it as built into `docs/functional_requirements.md`,
  records its decision in `docs/decisions.md` or `docs/decisions-operations.md`, adds its
  CHANGELOG line, and removes it from this plan.
- With the default configuration, every merge leaves `content_digest`, shard layout, refusal
  codes, queue keys and existing footer values unchanged.

## Requirements (working drafts)

## Phase F — Stocktake

The last merge is docs-only, recorded in its PR body, covering every check in CLAUDE.md's
Stocktake section. This plan is deleted in it.

## Order

Only F remains.
