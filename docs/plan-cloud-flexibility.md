# Cloud flexibility — plan

*Status: proposed; nothing is built. A piece is removed from this plan when it is built, and this document is deleted when its last phase lands.*

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

**Requirement numbers:** Requirements 36 to 39 are working labels for this plan only.
`docs/functional_requirements.md` stops at Requirement 35. As each requirement is delivered, it
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
  - `config/__init__.py`: 710 lines (holds `OCRConfig`, `EnrichmentConfig`, `EmbeddingConfig`)
  - `cloud/stage_runner.py`: 675
  - `store/run_manifest.py`: 636
  - `cloud/stage_contracts.py`: 605
  - `utils/model_check.py`: 558
- A new module lands in the CLAUDE.md table, `docs/architecture.md` and `docs/project-structure.md`
  in the same merge (`tests/test_doc_currency.py`).
- A merge that delivers a requirement writes it as built into `docs/functional_requirements.md`,
  records its decision in `docs/decisions.md` or `docs/decisions-operations.md`, adds its
  CHANGELOG line, and removes it from this plan.
- With the default configuration, every merge leaves `content_digest`, shard layout, refusal
  codes, queue keys and existing footer values unchanged.

## Requirements (working drafts)

### 36. Deployments Registered Once, Shared by Every Model They Serve

**Paid:** Adds no paid call by default. It moves the pre-run gating of the existing paid calls (`enrich`, `embed`, AI chunking in `chunk`, `mistral-ocr`) onto the deployment that serves them. A slot's `deployment:` setting can put two of one vendor's models on two deployments, billed two ways.

**As** a maintainer adding a vendor that offers several functions through one API,
**I want** the vendor's deployment registered once and every model it serves, in any slot, to reference it,
**so that** credentials, reachability, billing and the client are declared and checked once per deployment, not once per function or in a branch per vendor.

**Given** a registered deployment and models in several slots that reference it
**When** a run, worker or dispatch starts
**Then** the deployment is checked once, each model is checked as served by it, and every stage that uses one of its models is handed the same client.

**Acceptance criteria:**

- A deployment declares the environment variables it reads (alternatives allowed, so Isaacus reads `ISAACUS_API_KEY` or `ISAACUS_SAGEMAKER_ENDPOINTS`), whether it needs the network, whether its calls are billed, a free reachability check, a billed smoke call, the models it serves (all, or a declared list) and how its client is built. A declaration names variables, never a secret.
- A model registers in a slot against a deployment and a vendor model id. One model id may be registered in more than one slot (Kanon-2 Enricher as the `enrichment` provider and as the AI chunking model), and one model may be served by more than one deployment (Kanon-2 Embedder on the hosted API or a SageMaker endpoint).
- Each slot's config takes an optional `deployment:`. Left unset, a model is served as now: by SageMaker when `ISAACUS_SAGEMAKER_ENDPOINTS` is set, else the hosted API. A model the chosen deployment does not serve is refused, never routed to another deployment.
- Before launch, each configured model's deployment is checked once however many slots use it, and a model its deployment does not serve is refused naming the deployment and what it serves — for `run`, a worker, `POST /v1/runs` and `dispatch-workflow` alike. `load` runs the free check; `smoke` sends one billed call per deployment and model, warning first.
- A process builds one client per deployment and passes it to every provider that uses it.
- A worker serves a stage only when every model the stage uses passes its deployment's check.
- A deployment that needs the network is billed. Only the OpenAI-compatible deployment (Requirement 39) may be marked unbilled, by its own setting, for a self-hosted server.
- `GET /v1/resources` lists each declared deployment and the models it serves, values masked.
- Every sidecar records the deployment each model ran on, under a new footer key; an ARN is reduced to its endpoint name and region, as the run record does. Existing keys keep their values.
- A plugin can register a deployment as it registers a model; the run record names the distribution and version that supplied it.
- Existing plugins keep working: the `isaacus` trait means "served by the Isaacus deployment", and a reader with `ping()` / `ping_is_billed` and no declared deployment is given one built from them.
- With the default configuration the built-ins behave as now — Isaacus hosted and SageMaker as two deployments of one adapter (Requirement 29), Mistral on Bedrock, Ollama — with the same refusal codes, the same queues joined and the same `womblex.model_check` values.

**Verification:** a test registers one deployment and two models in two slots against it; with its variable unset, `run`, the worker and dispatch refuse before launch; set, the reachability check runs once and both stages receive the same client. The five gating modules read deployments, not `calls_isaacus`, `needs_isaacus_api` or `isaacus_models`.

### 37. Object Stores Are Whatever fsspec Can Open

**As** an operator,
**I want** any storage fsspec can open to be usable as a store, an ingest location or a destination, configured with fsspec's own options,
**so that** adding a storage backend is installing it and naming it, not editing Womblex.

**Given** an installed fsspec backend and its options in the deployment's settings
**When** a run, worker, API or egress names its scheme
**Then** it works with no change to Womblex.

**Acceptance criteria:**

- A scheme is usable when fsspec knows it and its backend is importable in this install. A scheme fsspec knows whose backend is not installed (`gs://` without `gcsfs`) is refused before launch, naming the package; an unknown scheme is refused naming the usable ones. The same check covers a store, an ingest location, the API's upload location and a destination.
- The existing typo checks stay: a missing `://`, an uppercase scheme, no bucket.
- Each location (store, ingest, destination) takes its own `storage_options`, set in the deployment's settings and, for containers, from one JSON environment variable per location. They reach fsspec unchanged; Womblex adds no per-backend option. A secret is written `{"env": "NAME"}` and resolved at use, so `GET /v1/resources` and the run record show the variable, never the value.
- S3-compatible stores (MinIO, Cloudflare R2, others) need configuration only. The ingest and output stores may be different accounts or vendors.
- The existing S3 variables (`WOMBLEX_S3_ENDPOINT`, `WOMBLEX_S3_ACCESS_KEY_ID`, `WOMBLEX_S3_SECRET_ACCESS_KEY`, the `AWS_*` fallbacks) keep working, below explicit `storage_options` in precedence, and still keep a store key out of the ambient AWS credentials the SageMaker signer relies on.
- The ingest/output disjointness check applies to every scheme.
- No dependency is added; an operator installs `gcsfs` or `adlfs` for the store they use.

**Verification:** a test runs extract → chunk → finalize → egress over fsspec's built-in `memory://` filesystem with no Womblex edit; a `gs://` store with `gcsfs` absent is refused before any output, naming `gcsfs`.

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

## Phase A — Provider packages (Requirement 38, mechanical)

Lands first so every later merge writes into the new layout. No behaviour change.

**A1. Isaacus package.**
- `utils/isaacus_client.py` → `providers/isaacus/client.py` (a rename; the old path re-exports).
- `IsaacusEnrichment` and its factory out of `analyse/enrich.py` → `providers/isaacus/enrichment.py`; `IsaacusEmbedding` out of `analyse/embed.py` → `providers/isaacus/embedding.py`. The provider protocols, limits and `make_*_provider` stay in `analyse/`.
- Registration still happens on import of `analyse/enrich.py` / `analyse/embed.py`, which import the adapters, so nothing that resolves a model changes.
- One `providers/` row in the CLAUDE.md table.

**A2. OCR engines.**
- A2a: `ingest/llm_ocr.py` → `providers/bedrock/ocr.py` as a rename, the old path re-exporting. Remove `ingest/llm_ocr.py` from `UNLISTED` in `tests/test_doc_currency.py`.
- A2b: the Ollama reader out of it → `providers/ollama/ocr.py`. `ingest/paddle_ocr.py` keeps the registrations, importing from the new modules.

## Phase B — Deployment registry (Requirement 36)

**B1. The registry.**
- New `utils/deployment_registry.py` beside `utils/model_registry.py`: the deployment declaration, `register` / `resolve` / `known_deployments`, the `womblex.deployments` entry-point group (reusing the model registry's loading), and the distribution and version of a plugin deployment.
- `ModelEntry` gains the deployments a model may be served by and its vendor model id.
- Built-ins declared: `isaacus-api`, `isaacus-sagemaker`, `bedrock`, `ollama`. A reader with `ping()` and no deployment is wrapped in one built from it.
- A deployment conformance test, run against every registered deployment. Nothing reads the registry yet.

**B2. The checks read deployments.**
- `utils/model_check.py`, `utils/availability.py` and `cloud/resources.py` check each configured model through its deployment, once per deployment. `calls_isaacus` stays as a read of "served by the Isaacus deployment".
- Pinned by tests: the same `SlotCheck` sources and `womblex.model_check` values for the default configuration, and the same masked resource view.
- `utils/model_check.py` is near the cap: the deployment checks go in a new module it calls, and `_check_isaacus` / `_ping_isaacus` are removed from it.

**B3. Contracts and workers read deployments.**
- `cloud/stage_contracts.py`: `needs_isaacus_api` / `needs_isaacus_api_for` become "the stage's models need their deployments", derived from the models the contract already names.
- `cloud/worker.py`: a stage is served when every model it uses passes its deployment's check.
- Pinned by a test: `model_check.models_key` gives the same key for an unchanged config, so rows already queued still route.

**B4. One client per deployment.**
- The registry caches one client per deployment per process. `cli/embed.py`, `cli/link.py`, `cloud/stage_runner.py`, `utils/model_check.py` and `make_ai_chunking_client` take it from there.

**B5. Choosing a deployment, and recording it.**
- An optional `deployment:` on the OCR, enrichment, embedding and AI chunking settings. `config/__init__.py` is at 710 lines: if the fields would take it near the cap, the slot settings move to a new `config/models.py` first, as their own mechanical merge.
- A new footer key naming the deployment each model ran on, an ARN reduced to endpoint name and region. The run record lists it; the record version is raised. `store/run_manifest.py` is near the cap, so the reduction is shared from a new module.
- Delivers Requirement 36.

## Phase C — Object stores (Requirement 37)

Independent of Phases A and B; may land in parallel as its own stack.

**C1. Schemes from fsspec.**
- `validate_location_uri` asks fsspec whether it knows the scheme and whether its backend imports; `SUPPORTED_PROTOCOLS` and `_REMOTE_PROTOCOLS` go. Its callers (`cloud/deployment.py`, `cloud/workflow_store.py`, `cloud/ingest_preview.py`, `store/source_provenance.py`, `store/egress.py`, `cli/cloud.py`) are unchanged.
- `cloud/delivery.check_destinations`'s backend check moves into `store/remote.py`, so store, ingest and destination share one.
- The disjointness check covers every scheme.
- The `memory://` end-to-end test.

**C2. Storage options per location.**
- `DeploymentSettings` gains options for the store and ingest locations; a destination's go in its spec. The environment form is one JSON variable per location.
- `{"env": "NAME"}` values resolved at use; `cloud/resources.py` shows the variable name.
- Precedence: explicit options, then the `WOMBLEX_S3_*` variables, then `AWS_*`, matching `storage_options_from_env` today.
- The compose files' comments show an R2 and an Azure example.
- Delivers Requirement 37 and closes Requirement 2's Azure storage gap (its Outstanding entry is narrowed to the Azure ML enrichment gap).

## Phase D — Checked documentation steps (Requirement 38)

**D1. The checks and the checklist.**
- A test fails for a model on a billed deployment with no `docs/models.md` entry under its canonical name, or no **Paid:** line naming it in its slot's requirement.
- Each slot's conformance test parametrised over every registered model.
- A fixture plugin under `tests/` registering one deployment and models in two slots.
- The checklist in `docs/model-plugins.md`.
- The CHANGELOG names the release that removes the re-exports from Phase A.
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

A → B → D → E, with C beside them. B needs A's layout; D's conformance and paid checks need
B's deployments; E needs B's deployments and D's conformance tests.
