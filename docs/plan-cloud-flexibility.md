# Cloud flexibility — plan

*Status: Phases A to C are built (the provider packages, Requirement 36 deployments registered once, and Requirement 37 object stores as fsspec opens them); Phases D to F are proposed. A piece is removed from this plan when it is built, and this document is deleted when its last phase lands.*

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

**Requirement numbers:** Requirements 38 and 39 are working labels for this plan only.
`docs/functional_requirements.md` stops at Requirement 37. As each requirement is delivered, it
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

### 38. Adding a Provider Is a Bounded Change

**As** a contributor adding a model, deployment or storage backend,
**I want** one checklist whose documentation steps the suite checks,
**so that** the cost of a new option is its adapter and its test, not a search across the docs.

**Given** a new built-in or plugin provider
**When** it is added
**Then** the checklist names every file it touches, and the suite fails on any step left out.

**Acceptance criteria:**

- The model-plugin guide (Requirement 24) carries the checklist for a built-in and for a plugin: the deployment (Requirement 36), if new; its model registrations; conformance tests; one `docs/models.md` entry per model; one CHANGELOG line.
- One plugin package can register a deployment and models in several slots, and installing it makes all of them selectable.
- A built-in vendor lives in its own package under `providers/` (its client module beside one adapter module per slot), and the module table covers `providers/` in one row, so a new vendor adds no row.
- A model on a billed deployment with no `docs/models.md` entry under its canonical name, or whose slot's requirement has no **Paid:** line naming it, fails the suite.
- Each slot's conformance test runs against every model registered in it, and the deployment conformance test against every registered deployment, so a new built-in is tested by registering it.
- The current import paths of the moved adapters (`ingest/llm_ocr.py`, the Isaacus adapters in `analyse/enrich.py` and `analyse/embed.py`, `utils/isaacus_client.py`) re-export for one release, and the CHANGELOG names the release that removes them.
- A womblex-benchmark pair is needed only when `SLOTS` or `DEFAULT_MODELS` change; a new non-default model needs none.

**Verification:** a fixture plugin under `tests/` registering one deployment and models in two slots touches only its own module and test, and every conformance test picks it up; `tests/test_doc_currency.py` fails for a billed model missing from `docs/models.md`.

### 39. A Second Embedding Provider

**Paid:** Adds one paid call when pointed at a hosted vendor: that vendor's embedding model, billed per request, called only when `embedding.model` names it. Pointed at a self-hosted server marked unbilled (Requirement 36), nothing is billed.

**As** a user without an Isaacus deployment, or wanting a second option,
**I want** a built-in embedding provider for any OpenAI-compatible endpoint,
**so that** hosted vendors and self-hosted servers are one setting away, and Requirement 36 is proven by a second real vendor.

**Given** `embedding.model` set to the OpenAI-compatible provider and its deployment declared
**When** the `embed` stage runs
**Then** `*.embeddings.parquet` is written with the same schema, `dim` taken from the vectors returned.

**Acceptance criteria:**

- A thin adapter over `/v1/embeddings` through `httpx`, adding no dependency; retries and limits are the server's, with no retry loop of Womblex's own. It serves hosted vendors and self-hosted servers (vLLM, Ollama, TEI) alike.
- It declares its deployment (Requirement 36) and its texts per request and network need (Requirement 24).
- Embedding roles are `document` and `query`; each provider maps them to its own terms, or to nothing when it has none, and says so in its `docs/models.md` entry. `embedding.task` keeps accepting `retrieval/document` and `retrieval/query`.
- A `dimensions` setting the provider cannot honour is refused before launch.
- A token ceiling is optional, from the provider's settings. A chunk the server rejects as too long is that batch's recorded error; the stage goes on.
- A workflow's Postgres destination stores vectors of any width beside their `model`, as its untyped `vector` column does today; `docs/models.md` notes that a similarity index needs one width.
- Isaacus remains the default; with nothing configured, outputs are unchanged.

**Verification:** the embedding conformance test passes for it with a mocked transport; a live test skips without its endpoint, like the Isaacus ones.

## Phase D — Checked documentation steps (Requirement 38)

**D1. The checks and the checklist.**
- A test fails for a model on a billed deployment with no `docs/models.md` entry under its canonical name, or no **Paid:** line naming it in its slot's requirement.
- Each slot's conformance test parametrised over every registered model.
- A fixture plugin under `tests/` registering one deployment and models in two slots.
- The checklist in `docs/model-plugins.md`.
- The CHANGELOG names the release that removes the re-exports of the moved paths (`utils/isaacus_client.py`, `ingest/llm_ocr.py`, and the Isaacus adapters in `analyse/enrich.py` and `analyse/embed.py`).
- Delivers Requirement 38.

## Phase E — OpenAI-compatible embedding (Requirement 39)

**E1. Embedding roles and capabilities.**
- `document` / `query` roles on the embedding provider contract; `embedding.task`'s Isaacus values accepted as aliases; the Isaacus adapter maps them to its tasks.
- A provider declares whether it honours `dimensions`; a setting it cannot honour is refused before launch.

**E2. The provider.**
- `providers/openai_compat/embedding.py` over `httpx`, and its deployment (base URL and key variable, `billed` from its own setting).
- Conformance with a mocked transport; a live test skipping without an endpoint.
- Its `docs/models.md` entry (roles, the one-width note for similarity indexes), the **Paid:** line, and the paid-models paragraph at the top of `docs/functional_requirements.md`.
- Delivers Requirement 39.

## Phase F — Stocktake

The last merge is docs-only, recorded in its PR body, covering every check in CLAUDE.md's
Stocktake section. This plan is deleted in it.

## Order

D → E. D's conformance and paid checks need the deployments built in Phase B;
E needs those deployments and D's conformance tests.
