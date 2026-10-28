# Cloud flexibility — plan

*Status: Phases A to D are built (the provider packages, Requirement 36 deployments registered once, Requirement 37 object stores as fsspec opens them, and Requirement 38 the checked provider checklist); Phases E and F are proposed. A piece is removed from this plan when it is built, and this document is deleted when its last phase lands.*

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

**Requirement numbers:** Requirement 39 is a working label for this plan only.
`docs/functional_requirements.md` stops at Requirement 38. As each requirement is delivered, it
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

E needs the deployments (built) and the conformance tests (built), then F.
