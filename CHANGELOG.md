# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Console Pipeline Composer — presets (`DEFAULT-Isaacus`).** The composer
  now offers named, selectable pre-configured pipelines from a dropdown above
  the stage graph, so an operator seeds a common end-to-end shape instead of
  hand-assembling it section by section. `DEFAULT-Isaacus` is the reference
  Isaacus pipeline — `extract → chunk → enrich → build_graph → money → done`,
  with the entity graph (enrich + the offline `graph-refresh` mention→chunk
  edge rebuild) and monetary-amount annotation produced over the one run — for
  PDF and DOCX sources.

  A preset is served as data (`ui/presets.py`, `GET /api/composer/presets`
  and `/presets/{name}`) and is a *partial* `WomblexConfig`: it carries only
  the stage toggles and settings its shape sets, never `dataset` / `paths`,
  which name the run and stay the operator's to supply. Loading one
  deep-merges its overlay onto the form's current config, so those two
  sections survive untouched. Each preset is validated at import against the
  same `WomblexConfig(**raw)` construction `load_config` uses (and covered by
  `tests/test_ui.py`), so a preset that would not load is a test failure, not
  a runtime surprise — the same rule the graph, schema and `/validate` already
  follow. Add a preset by declaring it in `ui/presets.py`; the API and the
  form pick it up with no other change.

  Runnable from the CLI, not UI-only: `configs/default-isaacus.yaml` ships the
  same shape as a complete, `load_config`-valid config, with the per-stage
  command sequence commented inside it. This matters because `womblex run`
  alone runs only `extract → redact → chunk → pii` — `enrich`, `build_graph`
  (graph-refresh) and `money` are per-stage commands, and `enrich` must precede
  `chunk` so AI chunking reuses the enrichment (no double Kanon-2 cost). The
  config carries reasonable settings for the shape (`chunk_size: 480` the
  Kanon-2 window, `overlap: 0.1` so a boundary-straddling mention still lands
  in a chunk, `money.default_currency: AUD`; embed/link/pii off). A test pins
  the config file against the console preset so the two cannot drift.
- **Console Execution Controls — screen (`docs/ui-plan.md` merge 11).**
  The screen over the backend that landed just below: `ui/src/routes/execute`
  replaces the sidebar's last stub with the console's one writable-to-a-run
  surface. It loads `GET /api/execute/status` (a cheap, network-free read)
  and, when the console cannot dispatch, shows a banner naming the *one*
  missing piece — audit-only (no `--allow-execute`), no store, or no queue —
  in the guard's own order, rather than a bare "disabled". The
  configure-and-run form (input prefix, optional run id, batch size, max
  attempts) posts to `POST /api/execute/enqueue`; a success reports the run
  id, document/batch counts and `newly_enqueued` (which distinguishes a fresh
  run from a resume), then points the run selector at the run just planned so
  the operator can switch straight to the Dashboard or Corpus Inspector to
  watch it drain.

  Dispatch is the only action: "log streaming" is the Dashboard's own
  queue-status + checkpoint feed (plan §4), so the screen links there rather
  than duplicating it. The primary button is lime `--primary`, per
  `DESIGN.md`'s role table — `--accent` is structural-only and never a fill.
  A capability change since load (the switch flipped off, say) surfaces as the
  enqueue's 403/409 and re-reads the status so the form disables and the
  banner explains, matching what the server saw. The new `EnqueueRefused`
  carries the HTTP status so the client tells the three failure shapes apart
  (403 audit-only, 409 unwired, 400 bad input) without parsing a message.
  With this, merge 11 is complete end to end and only the Dashboard screen (8)
  and the `ReportIssue` control (7) remain.
- **Console Execution Controls — backend (`docs/ui-plan.md` merge 11).**
  The console's first writable-to-a-run surface: `GET /api/execute/status`
  reports whether this deployment can dispatch work (and which piece is
  missing if not), and `POST /api/execute/enqueue` plans an extraction run
  into the job queue — the "configure-and-run" action. Both live in a new
  `ui/execute.py` + `ui/routes/execute.py`.

  Dispatch is always the queue (plan §4): `enqueue_extraction` is a thin
  wrapper over the same key-listing / batching `cli/cloud.cmd_enqueue` does,
  reached through the sidecar's own store, so no web request can become an
  arbitrary command — there is no command, only an idempotent queue row per
  batch. Workers a platform brings up do the work; the console runs no
  scheduler.

  `--allow-execute` is the switch, enforced in one place (`_guard`) that
  every write action calls before touching the store or the queue: off gives
  a pure auditing console (403). Execution is queue-only, so it also needs
  *both* a `--store` and a `--dsn` — a local `output_root`-only console can
  configure and audit but not dispatch, and refuses with 409 rather than
  half-working. `ExecutionCapability` carries the three flags so the screen
  can name the missing piece; the route maps its `ExecutionDisabled.reason`
  onto 403 (audit-only) vs 409 (unwired). "Log streaming" is the queue's own
  job-status transitions plus the per-stage checkpoints the Dashboard
  already serves — batch-granular, not a fabricated line-by-line log. The
  `womblex ui --allow-execute` flag, previously reserved, now does this.
- **Console Pipeline Composer — config form (`docs/ui-plan.md` merge 9).**
  The composer's second half: a form over `/api/composer/schema`, a Validate
  action, a YAML preview and a download, and stage toggles on the graph's
  nodes wired to the same config the form edits.

  `SchemaForm.svelte` renders `WomblexConfig`'s JSON Schema recursively — a
  `$ref` property becomes a collapsible subsection — so every field of every
  nested model is reachable without a hand-typed mirror of `config.py`, and a
  new config field appears in the console the moment Pydantic reports it.
  `X | None` is read through as Pydantic's optionality marker rather than
  rendered as a variant picker, and an optional subsection that defaults to
  null (`linking.reference`) stays null until an operator asks for it, so the
  composer never posts a section the library never had.

  Nothing about validity is decided in the browser: Validate and the YAML
  download both go to the endpoints, which build a `WomblexConfig` the way
  `load_config` does, and the downloaded file is the server's rendering — so
  it is byte-identical to what `womblex run --config` would read.

  Each node's `enabled` checkbox writes its config section's `enabled`, so
  the graph and the form are one state rather than two views that can
  disagree; disabled stages drop to 40% and keep their edges, per DESIGN.md's
  `StageNode`. Which section a stage belongs to is served as `config_section`
  from a new `CONFIG_SECTION` map in `ui/composer.py`: not derivable from
  `StageContract` (contracts name suffixes, not config fields), so declared
  beside the config models under a test that every name is a real
  `WomblexConfig` field, rather than typed into the frontend where a rename
  would drift in silence.
- **Console Pipeline Composer — stage graph (`docs/ui-plan.md` merge 9).**
  The composer screen replaces its `ScreenStub` with the pipeline DAG that
  `/api/composer/graph` already served, plus a detail panel for the selected
  stage (scope, mutation mode, Isaacus need, declared inputs and outputs).

  `StageGraph.svelte` lays nodes out by longest path from `extract`, so a
  stage sits one column right of its latest dependency and every edge points
  forward. Nothing about the ordering is typed in the frontend: the columns
  fall out of the `required_inputs` edges the endpoint serves, which is the
  plan's §3 rule ("do not hand-code the DAG in the frontend") holding in the
  one place it could have been broken.

  Nodes are HTML cards positioned from the same constants the edge SVG reads,
  rather than a measured layout — no `ResizeObserver`, and the geometry is
  deterministic in the first frame. Selecting a node emphasises the edges
  that touch it and fades the rest, which is how a stage's actual
  dependencies read at a glance in a graph with fifteen of them.

  Also fixes the frontend CI job, red on `main` before this: `svelte-check`
  rejected `let x: ChunkDetail | null = $state(null)` in the Chunk Inspector
  (TypeScript narrows the annotated `let` to `null`, so every `$derived` over
  it errored on `never`), and `eslint` flagged a plain `Map` built and
  returned inside a `$derived` — pinned with a scoped disable and its reason,
  since `SvelteMap` is for state edited in place.
- **Console Resources Console (`docs/ui-plan.md` merge 10).** `GET
  /api/resources` returns three connection cards — run store, job queue,
  Isaacus — plus `POST /api/resources/test/store` and `/test/queue` as the
  live checks behind each card's action, and the screen itself replaces its
  `ScreenStub`.

  No new detection logic: the plan's §3 row for this screen says connection
  *testing* already exists as library code, so the store card reads
  `is_remote_uri` / `storage_options_from_env`, the Isaacus card reads
  `unserved_models()`, and the queue card reuses `dashboard.queue_section`
  (renamed from a private helper) rather than reimplementing
  connect-and-read — so its fleet and queue-depth state is by construction
  the same view the Dashboard shows. The models checked for Isaacus coverage
  are read off `EmbeddingConfig` / `EnrichmentConfig`'s `model` field
  defaults rather than re-typed, the same rule `/schema` and
  `CHECKPOINT_DIRNAMES` already follow.

  Configuration is split from reachability deliberately: the `GET` makes no
  network call, so one dead connection cannot stall the page load, and only
  the card an operator actually clicks pays the timeout.

  **Credentials do not leave the process.** The store card reports whether
  AWS keys are configured, never their values, and the queue DSN is masked.
  Found by probing that masker rather than by reading it: it handled only
  the URI DSN form, so libpq's equally-valid keyword form (`host=…
  password=…`, which psycopg accepts and `JobQueue` passes straight through)
  has no netloc for `urlsplit` to find a password in and was returned
  verbatim — a full credential leak from an endpoint with no auth in front
  of it (plan §6). Both forms are masked now, along with a `?password=`
  query parameter and a secret too short to have a non-revealing tail.

  The store test answers a deliberately different question per deployment —
  locally that the mount landed, remotely that the listing completed, since
  an object store has no `runs/` prefix until its first run finishes. A
  genuinely unreachable store still fails, because the connection error
  propagates rather than flattening to an empty result. Documented and
  pinned, because the two verdicts look like a bug until you know which
  question each is answering.
- **Console Pipeline Composer read API (`docs/ui-plan.md` merge 9).**
  `GET /api/composer/graph`, `/schema`, `POST /validate` and `/yaml` — the
  composer's four data sources, none of them run-scoped, so unlike every
  other console route these take no `UISettings` dependency.

  The graph endpoint renders `STAGE_CONTRACTS` as nodes and edges instead of
  the frontend hand-coding a DAG (plan §3). Edges come from each stage's
  `required_inputs` — the ordering guardrail the plan calls out ("ensuring
  extraction precedes chunking") — resolved through the existing
  `PRODUCER_OF` map, with a synthetic `extract` node for the sidecars
  extraction itself writes. Config-derived `conditional_inputs` ride along
  per node rather than adding edges, since an edge for one would only be
  true for whatever config the form happens to hold. One edge per ordered
  pair carries every suffix justifying it (25 → 15); tests pin what matters
  — acyclic, every stage reachable from `extract`.

  `/schema` is `WomblexConfig.model_json_schema()` verbatim, so no hand-typed
  mirror of `config.py` can drift. `/validate` and `/yaml` both construct
  `WomblexConfig(**raw)`, the call `load_config` makes, so a config the
  composer accepts is one the CLI accepts; `/yaml` round-trips through
  `model_dump(mode="json")`, carrying Pydantic-applied defaults rather than
  just what the browser posted.

  **A typo'd key no longer validates clean and then vanishes.** Found by
  probing, not by reading the diff: Pydantic ignores unrecognised keys, so
  `chunkng:` (or `chnk_size:` inside a correct section) returned
  `{"valid": true, "errors": []}` and rendered a YAML with the setting gone
  and `chunk_size` back at its 480 default — for a config *editor*, the
  worst failure mode available. `/validate` now reports `unknown_keys`
  beside `valid`, and `/yaml` names dropped keys in a comment header so the
  warning rides on the artefact that gets committed and mailed around.
  Warnings, not errors: the CLI loads such a file too, so failing it here
  would make the composer stricter than the thing it configures. The walk
  sees through `X | None` unions but deliberately not into free-form
  `dict[str, str]` fields — recursing into `normalise.substitutions` would
  report an operator's own letterhead replacements as typos. Probing also
  confirmed no config validator touches the filesystem, so this
  unauthenticated endpoint (plan §6) is not a file-existence oracle — now a
  test rather than an assumption.

  Backend only — the Pipeline Composer screen is still a `ScreenStub`, as
  the other inspector screens have been since merges 5–6 and 8.
- **Console dashboard read API (`docs/ui-plan.md` merge 8).**
  `GET /api/dashboard` serves queue state and per-stage progress from two
  sources the pipeline already writes, with no new schema or instrumentation.
  `JobQueue` grows four read-only views beside `stats()` — `list_jobs()`,
  `workers()` (the fleet, from `locked_by` on running rows), `stale_jobs()`
  and `throughput()` (completions per minute, derived from `updated_at`).
  `stale_jobs()` is the read-only twin of `requeue_stale`: same predicate, no
  recovery, so the console names a stalled batch and a worker fixes it.

  The queue-less half needed no configuration at all. Every shard stage
  already checkpoints to a dot-directory *inside the run*
  (`<run>/.chunk-checkpoint/` and friends) which `STAGE_CONTRACTS` names via
  `checkpoint_dirname` — so `store/checkpoint.py`'s new `read_checkpoints()`
  reads stage progress out of the run the console is already pointed at, in
  both deployments, deriving the map from the contracts rather than re-typing
  it. Throughput there is the checkpoint's own `started_at` → `updated_at`
  span, labelled batch-granular because that is what a per-batch write can
  honestly support (plan §4).

  A queue is optional and orthogonal to the run source: `--dsn` /
  `$WOMBLEX_DB_DSN` when there is one, and an unreachable queue reports
  `queue_error` rather than 500ing, so checkpoint progress still renders next
  to it. `JobQueue` gained an optional `connect_timeout` for that reason —
  workers may block until the OS gives up, but a polled console facing a
  routable-but-dead host would pin a request thread per poll (measured: 5s
  and a `queue_error`, against an unbounded wait before).

  `run_id` reaches this endpoint as a *query* parameter, not a path segment,
  so no routing normalisation precedes it: the local checkpoint join
  contains itself with the same `is_safe_run_id` guard the feedback writer
  uses, matching the containment the remote branch already got from its
  `list_dirs` check. `read_checkpoints` skips well-formed-but-unusable JSON
  (a top-level list) and keeps the counters of a checkpoint whose timestamps
  are null, withholding only the rate they cannot support.

  Backend only — the Dashboard screen is still a `ScreenStub`, as the
  inspector screens have been since merges 5–6.
- **Console report action (`docs/ui-plan.md` merge 7).** The console's first
  and only write path: `POST /api/runs/{run_id}/feedback` files a reviewer's
  note about a record as **one JSON file per report** — never an append, so
  there is no read-modify-write and no lost update when two reviewers click at
  once (measured: 40 concurrent posts, 40 distinct files). Reports land under a
  `feedback/<run_id>/` root that is always a *sibling* of the runs, never
  inside one, so re-running a stage or purging a run neither disturbs
  accumulated feedback nor is disturbed by it — locally beneath `output_root`
  (retention only purges `run-*`), remotely at the store's own `feedback/`
  prefix. `--feedback-dir` / `$WOMBLEX_UI_FEEDBACK_DIR` covers deployments that
  mount `output_root` read-only. `reported_by` resolves from a trusted header
  or env var, never the client body, and stays advisory — there is no auth to
  verify it against (plan §6).

  `store/feedback_output.py` owns the root/run_id join and so enforces its
  containment (`is_safe_run_id`). Found by probing, not by reading the diff: a
  `..` run_id passes an `is_dir()` check and lands a report at `runs/runs/<id>/`,
  escaping the feedback root. HTTP routing never allowed it (a path param cannot
  match `/`), but `readers.write_feedback` is library API and the
  sibling-of-runs invariant is the module's guarantee, not the router's.

  Backend only: the `ReportIssue` control waits on the Corpus and Chunk
  Inspector screens, still `ScreenStub` placeholders after merges 5–6 landed
  each inspector's read endpoints ahead of its screen.
- **Console frontend shell (`docs/ui-plan.md` merge 4).** `ui/` — a Svelte 5 +
  SvelteKit workspace, built to static files by a Node stage in
  `Dockerfile.ui` and served by `womblex.ui.app.create_app` alongside the
  existing read API. No JavaScript runtime ships in the image: the final
  stage `COPY --from=frontend-builder`s only the compiled `ui/build`, and
  `create_app`'s new `spa_dir` mounts it if present — a bare `womblex[ui]`
  install with no SvelteKit build alongside it still serves `/api/*` alone,
  same as before this merge.

  Ships the chrome the plan calls for and nothing past it: a top bar (logo,
  global search input, run selector, density toggle, theme toggle) and a
  collapsible side nav routing between the five domains named in §3, each
  currently a stub page naming its data source and the merge that fills it
  in. Theme and density are `localStorage`-persisted preferences applied via
  `data-theme` / `data-density` on `<html>`, matching `DESIGN.md`'s "shell
  attribute, not a per-table prop" rule; dark is the default, per the design
  system's dark-first principle.

  Tokens follow `DESIGN.md` §"Colour" for every value it states explicitly
  (surfaces, status fills, `--font-mono`); the base tokens it names but
  doesn't restate the hex for (DeepCivic's own `DESIGN.md` lives in a
  separate repository, not available here) are a best-effort fill picked to
  satisfy what *is* stated — the measured-contrast table, lime/purple never
  doubling as page background or body text — and flagged as such in
  `docs/decisions.md` rather than asserted as a pixel-exact match. Barlow
  Condensed and Inclusive Sans are self-hosted via `@fontsource`, so the
  console makes no Google Fonts request, per the design system's
  air-gapped-friendly rule.

  The SPA catch-all serves `index.html` for client routes but **404s the
  `/api/` namespace explicitly**. Without that guard a wrong or future
  endpoint answered `200` with an HTML body, which a JSON client surfaces as
  a parse error rather than the 404 it is; `/apiary` and friends still reach
  the shell, so the guard is the namespace, not the prefix. The fallback
  also resolves and containment-checks the requested path before touching
  the filesystem, so `..` traversal returns the shell instead of escaping
  `ui/build`.

  Accessibility was verified by measurement, not inspection, and two of
  `DESIGN.md`'s rules turned out not to survive its own light theme — the
  system is dark-first and its figures were computed against the dark page.
  Lime as the active-nav *label* colour measures 11.86:1 on the dark nav but
  1.32:1 on the light one (`--surface-raised` is `#ffffff` there), so lime
  stays as the 2px active rule — a fill, always legible — and the label
  carries the state in weight and `--foreground` (13.72:1 / 17.46:1). The
  purple `--accent` is likewise not used as body copy (4.42:1 dark, 3.64:1
  light), matching the doc's own "structural / large text only". Every one
  of the shell's nine tab stops was confirmed to render the `--ring`
  indicator. The related finding that `DESIGN.md`'s status-pill rule holds
  only in dark mode is recorded in `docs/decisions.md` for merge 5's
  `StatusPill`, which is the first component it would bite. The density
  control drives the side-nav row height (48/40/32px) rather than only
  taking effect once the grids land, so the shipped control does something
  and the mechanism is proven before merge 5 depends on it.

  A Node job in `ci.yml`, independent of the Python matrix, lints
  (`eslint`), type-checks (`svelte-check`) and builds the SPA on every push —
  a broken frontend now fails its own job instead of hiding inside a Python
  test run. `tests/test_ui.py` gained the same kind of drift guard merge 3's
  image tests did: the Dockerfile's builder-stage script and output
  directory are checked against `ui/package.json` and `svelte.config.js`
  rather than asserted separately, and the SPA-mount behaviour (client-route
  fallback, `/api/` 404s, traversal containment) is exercised directly.
- **Console sidecar image (`docs/ui-plan.md` merge 3).** `Dockerfile.ui` and a
  `ui` service in `docker-compose.yml`, so the console deploys the way the
  plan's §2 describes it — its own container beside the workers, same image
  shape and same env vars, reading shared state rather than running in-process
  with the pipeline. `docker compose up -d ui` serves the read API at :8080
  over the same `s3://womblex` bucket the workers publish to.

  Installs `womblex[ui,cloud]`: fastapi/uvicorn plus the object-storage reads
  a distributed run needs, and still no boto3 in either extra. It carries the
  same `libglib`/`libGL` system libraries as the pipeline image, because the
  console's `store/` readers reach `ingest.extract` for `ExtractionResult` and
  so load PyMuPDF — verified by import trace, which also confirms ultralytics,
  torch, sentence-transformers and cv2 stay out of the console's import path.

  The container is hardened `read_only: true`, which is the container-level
  statement of the plan's rule that no screen writes to a stage output, with a
  `tmpfs` `/tmp` for the temp dir a store-backed read stages the manifest
  through (`ui/readers.py`). Confirmed the console needs no other writable
  path, `$HOME` included, by serving both endpoints with `HOME` pointed at a
  directory that does not exist.

  A separate Dockerfile rather than a build-arg variant of the existing one
  because the two diverge once the SPA lands. The SvelteKit build stage is
  **not** here, though the plan listed it under this merge: a stage that
  `COPY`s a directory which does not exist yet cannot build, and one defined
  but never referenced is dead weight — so it arrives in merge 4 with `ui/`
  itself, where it can be exercised. What this merge ships builds and runs.

  `tests/test_ui.py` gains the drift guard the deployment files otherwise
  lack: the image's `ENTRYPOINT` is parsed against the real CLI parser (a
  renamed or removed `womblex ui` flag fails the test rather than the
  container), the bind address is asserted non-loopback, and the compose
  service is checked for exactly one run source — two would make
  `resolve_settings` raise and the container exit 1.
- **Console read API skeleton (`womblex ui`).** The first code merge of the
  optional console (`docs/ui-plan.md` merge 2): a `[ui]` extra
  (`fastapi` + `uvicorn`, no boto3), a `womblex ui [--output-root DIR |
  --store URI] [--port 8080]` command, and two read-only endpoints —
  `/api/runs` (the run selector, from `describe_run()`) and
  `/api/runs/{run_id}/manifest` (the documents table, `MANIFEST_SCHEMA`).

  Remote reads are in scope from this merge rather than deferred, because a
  cloud sidecar has no filesystem to read: a store-backed request stages the
  (small) manifest into a temp dir and hands it to the *same* local reader,
  which is the shape `womblex finalize` already uses — so the parquet logic
  has one implementation and `RemoteStore` only ever moves bytes. Both paths
  prefer a consolidated `manifest.parquet` and fall back to the per-batch
  shard manifests, so a distributed run reads correctly before and after
  `finalize`.

  No pipeline logic lives in `ui/` — every read goes through the existing
  `store/` readers, per the plan's governing rule. The app binds one run
  source at construction, so no endpoint can be talked into reading a
  directory the operator did not mount, and it binds to loopback by default
  (the console has no auth by design; `--host` is the explicit opt-out).
- **Run index: `describe_run()` and `RemoteStore.list_dirs()`.** `list_runs()`
  returns paths; a run selector wants what is *in* a run. `describe_run()`
  summarises a run root as run_id, document count, the stages present, and
  created/updated timestamps — reading only artefacts the pipeline already
  writes (the consolidated `manifest.parquet` when a run has been finalised,
  else the per-batch shard manifests). "Stages present" is defined by the new
  `STAGE_SUFFIXES` table: stage name → the sidecar suffix that stage writes,
  so presence of a matching file *is* the lifecycle checkpoint. Redaction is
  absent from it deliberately — it rewrites element text in memory and leaves
  no sidecar to detect.

  `RemoteStore` gains `list_dirs()`, which surfaces fsspec's common-prefix
  pseudo-directories so a caller can enumerate `runs/<run_id>/` in object
  storage without knowing the run ids ahead of time. `store/output.py` exports
  `ELEMENTS_SUFFIX` alongside the other sidecar-suffix constants rather than
  leaving the elements one spelled out inline.

  Groundwork for the console's run selector (`docs/ui-plan.md` §4 names this
  as the one genuinely missing read), but the CLI benefits from it too.
- **Isaacus on Amazon SageMaker (private, air-gapped deployment).** Every
  Isaacus call — AI chunking, `enrich`, `embed` — now routes to SageMaker
  endpoints in the user's own AWS account when `ISAACUS_SAGEMAKER_ENDPOINTS` is
  set, instead of the hosted API. No API key is involved (the integration signs
  requests with AWS credentials); `isaacus_available()` accepts either
  deployment, so the chunk gate no longer demands a key that an air-gapped
  install cannot have.

  Subscriptions are per model plus a universal one, and Womblex assumes no
  shape: the variable declares comma-separated `name[@region][=model|model|...]`
  entries, an entry *without* `=models` serving everything. The two mix freely
  — a dedicated embedder endpoint alongside a catch-all resolves the way the
  integration's own router does. Region falls back to
  `ISAACUS_SAGEMAKER_REGION` then the AWS SDK's; `ISAACUS_SAGEMAKER_PROFILE`
  selects a profile.

  Stages name the model they call, so an undeployed subscription fails at
  client construction — naming the model, listing what is served — rather than
  as `No SageMaker endpoints registered for model` mid-batch, or a bare
  `AssertionError` when no region resolves. AI chunking now passes an
  explicitly built client to semchunk instead of letting it construct one from
  `ISAACUS_API_KEY`. Token counting is unchanged: the tokeniser stays local.

  `[isaacus]` gains `isaacus-sagemaker` (imported lazily, only when the
  variable is set). It pulls boto3 in, which is why the boto3-free rule stays
  scoped to `[cloud]`.
- **`elem_order` document-order anchor on table chunks.** `CHUNKS_SCHEMA` gains
  a nullable `elem_order` column, populated **only** for `content_type='table'`
  chunks with the `elem_order` of the table element the chunk came from.
  Consumers can now recover narrative ↔ table document order — sort narrative
  chunks by `start_char`, table chunks by `elem_order` — which the two disjoint
  chunk projections (one narrative string, one markdown string per table)
  otherwise lost. Null for narrative chunks (they straddle elements, so no
  single anchor exists) and for spreadsheet sheets (a sheet aggregates many
  `sheet_cell`s and has no narrative to be ordered against).

  Deliberately *not* a coordinate-space change: narrative chunks, the
  reassembled enrichment input, and the Kanon-2 mention↔chunk offset mapping
  are all untouched, so no existing shard's offsets shift and no re-enrichment
  is needed. `read_chunks` back-fills the column with nulls for shards written
  before it, so pre-existing chunk shards stay readable; genuinely missing
  columns still raise as before.
- **Deployment-shaped install extras.** `[local]` (deliberately empty — the
  base install *is* the local CPU deployment) and `[cloud-ocr]` (an alias of
  `[bedrock]`, named for what it buys rather than the vendor) join the
  existing `[cloud]` / `[isaacus]` / `[bedrock]`, so an install can state
  which deployment it is. No package versions changed; `uv.lock` moves only
  its extras metadata.
- **README installation matrix** mapping deployment → extra → what it adds,
  plus a "Selecting the backend" table for the local/cloud runtime split.

### Changed
- `[cloud]` documented as explicitly *not* implying `[cloud-ocr]`: s3fs reaches
  S3 through aiobotocore → botocore, so object-storage staging needs no boto3,
  and a scalable cloud deployment keeps the bundled CPU OCR engine unless it
  opts into the hosted VLM engine. boto3 stays confined to its single import
  site, `ingest/llm_ocr.py`.

## [0.4.0] - 2026-08-06

Minor under 0.x: additive in surface (the `run-stage` command and its stage
contracts; no new schema, no new config key), with one change a consumer can
observe — **the `money` op's narrative output changes for text it already
read**. Amounts written with space-grouped thousands were stored wrong by 10³
(`$10 000` as ten dollars) and are now correct; `$US`-marked amounts, worded
amounts and parenthesised restatements now resolve where they previously
yielded nothing or the wrong sign. Re-run `womblex money` over any shard
directory whose spans were produced by 0.3.0: the sidecar is regenerated in
place and no other stage depends on it.

**No parquet schema changed**: `ELEMENT_SCHEMA`, `TABLE_CELLS_SCHEMA`,
`FORM_FIELDS_SCHEMA`, `CHUNKS_SCHEMA`, `EMBEDDINGS_SCHEMA`,
`MONEY_SPANS_SCHEMA` and `MONEY_COLUMNS_SCHEMA` are byte-identical to `0.3.0`.
`money_spans.evidence` carries one new value, `p11`, in its existing string
column.

### Added
- **Money: financial values expressed in narrative structure.** The detector
  read amounts written as digits beside a symbol; the forms prose actually uses
  around them were missed, and two of them were stored *wrong* rather than
  skipped. See [docs/money-extraction.md](docs/money-extraction.md).

  - **Worded amounts (pattern 11)** — `two million dollars`, `five hundred
    thousand dollars`, `fifty cents`, `half a million dollars`, `one and a half
    million dollars`, parsed to exact `Decimal`s by `process/money_words.py`.
    A currency word is required, so a worded number on its own is not money:
    the corpus's only worded-number phrase (`more than one million
    Australians`) is a headcount and is declined. Measured: zero worded false
    positives across 1.6 MB of real government text.

    The parser declines what English does not write as one number, because
    each of these parses arithmetically into money the document never wrote:
    `between ten and twenty dollars` and `ten–twenty dollars` (a range —
    reading 30), `nineteen fifty dollars` (a year), `in million dollars` (a
    table's unit declaration, where only a number or an article makes it an
    amount), `one thousand million`, and `zero`/`nil dollars`.
  - **Space-grouped thousands** — `$10 000`, `$1 500 000` (AGPS convention,
    plus the no-break and thin spaces a PDF text layer emits). Previously
    `Penalty: $10 000` was stored as **ten dollars**; a group is exactly three
    digits and may not be followed by another, so `$5 2020` no longer binds an
    amount to the year beside it.
  - **`$US`/`$A` symbol order** — `$US655.5m`, `$A250,000`, `$AUD1.2m`. Also
    fixes the blocker that lost them: the metre pattern matches at `5m` inside
    `$US655.5m`, and the currency check behind it now steps back over the
    number's own digits (this silently affected `US$655.5m` too).
  - **Restatement** — `one million dollars ($1,000,000)` read as an accounting
    negative (−1,000,000) and, once worded amounts existed, counted the same
    money twice. The restating half is now dropped, in either order. Two
    bracketed *digit* amounts of equal value are left alone: `$5,000 (5,000)`
    is this year and last, not one amount written twice.
  - **Signs and brackets** — a leading true minus (`−$5.2 million`) is a
    negative, while the en dash stays the range separator; `$(1,234.50)` is an
    accounting negative like `($1,234.50)`; `50¢` is a sub-unit.
  - **Declines a second dotted group** — `$3.219.3m` (a real ANAO typo for
    `$3,219.3m`) yielded `$3.219`, three dollars for a $3.2 billion budget.
    Repairing the typo would be a guess; the amount is now declined.
  - Cell scanning no longer skips prose cells outright — the pre-filter admits
    a currency word, so a worded amount in a contract table is reachable.
  - Qualifier vocabulary gains the drafting forms (`not exceeding`, `a maximum
    of`, `in the order of`), stored separately from the value as before.

  Measured over the benchmark fixtures, comparing spans by offset: two values
  corrected, three amounts recovered, **no span lost**. `find_money` over the
  1.3 MB ANAO report costs 1.42s → 1.83s.

  `process/money.py` passed the 750-line cap, so number reading, currency
  resolution and false-positive blocking moved to `process/money_numbers.py`;
  every name is re-exported from `money.py` and no import site changed.

- **`womblex run-stage` — remote per-batch shard-stage runner.** Runs a
  downstream `*_shards()` stage directly against object storage, so a
  distributed run no longer has to be synced down before it can be chunked,
  enriched, embedded or masked.

  ```bash
  womblex run-stage --stage chunk --store s3://womblex --run-id <run_id> \
      --config configs/example.yaml
  womblex run-stage --stage chunk --shards out/<run_id>/documents   # local
  ```

  It generalises `finalize`, which already downloads one sidecar class,
  calls an unchanged library function against a temp `Path`, and uploads the
  result. **No `*_shards()` signature changed.** Covers `normalise`, `spellfix`,
  `chunk`, `money`, `enrich`, `embed`, `link`, `pii`, `graph-refresh` and
  `quality`; `manifest` is deliberately absent because `finalize` is it.

  - Stage contracts are declarative (`cloud/stage_contracts.py`), with
    conditional inputs and produced outputs resolved from **config**, not from
    the stage name: `chunk` pulls `*.enrichment_doc.parquet` only when
    `chunking_model` is set; `chunk`/`enrich`/`money` pull the overlay sidecar
    named by `processing.text_source` (`money.text_source` outranks it); `pii`
    declares `*.clean_text.parquet` only when `write_clean_text`; `enrich`
    declares `*.enrichment_doc.parquet` only when persisting.
  - Bases are discovered from **extraction-role siblings only** — a
    `*.chunks.parquet` with no extraction sibling is not a batch, and
    `*.form_fields.parquet` is discovery-only so it is never downloaded.
  - **Every declared output is verified before any is uploaded**, so a stage
    cannot leave a half-written set behind. Skip fires only when *all* declared
    outputs are present, so even a transport failure mid-publish never reads as
    complete — the next run redoes the base. Idempotent: re-run as more batches
    land and only the new ones are processed. Finding nothing to do exits 1, so
    a typo'd `--run-id` is not mistaken for success.
  - `graph-refresh` is modelled explicitly as an **in-place mutator** (outputs ⊆
    inputs): never skipped, both sidecars re-uploaded unconditionally, resting on
    its existing idempotency.
  - `quality` is **run-scoped**, staging every batch's chunks in one pass —
    per-batch execution would miss cross-batch duplicates *and* emit colliding
    cluster ids, since `_cluster_ids` numbers clusters per pass.
  - Stages needing the Isaacus API now **fail non-zero** rather than
    publishing nothing (`chunk_shards` otherwise warns and returns empty);
    `link` preflights that its worker-local reference register resolves.
  - `--stage-checkpoints` optionally stages the stage's checkpoint *directory*
    in and out; the default remains output-exists skip, which is race-free
    across concurrent runners.

  Stage *ordering* is the caller's: `embed` needs `chunk`, `pii` is terminal
  after `enrich` and `embed`, `spellfix` chains off `normalise`. A base whose
  required inputs are absent is reported as not-ready (the fleet may still be
  draining); if *every* base is, that is a stage-ordering error and exits 1.

## [0.3.0] - 2026-07-29

Minor under 0.x: additive on the whole (the `money` op, its CLI command and its
two parquet sidecars; the shared table-grid algorithm), with two changes a
consumer can observe — a scanned page carrying a clean table now emits a
`kind='table'` element where it previously emitted a `[TABLE]` placeholder text
block, and the unreachable `ImageExtractor` was removed. **No extraction schema
changed**: `ELEMENT_SCHEMA`, `TABLE_CELLS_SCHEMA`, `FORM_FIELDS_SCHEMA`,
`CHUNKS_SCHEMA` and `EMBEDDINGS_SCHEMA` are byte-identical to `0.2.0`.

### Added
- **`money` annotation op** (`womblex money --shards`). Recovers monetary
  amounts from the extraction parquet and writes two siblings per batch:
  `*.money_spans.parquet` (one row per amount) and `*.money_columns.parquet`
  (the column-classification audit). Offline, API-free, no ordering dependency
  on enrich, and it never rewrites element or chunk text. Implements the design
  in [docs/money-extraction.md](docs/money-extraction.md).

  Two evidence paths, because most of this corpus's amounts carry no currency
  marker at all:

  - **Self-evidencing** (`process/money.py`) — a symbol, ISO 4217 code or
    currency word sits with the number. The pattern set is applied in priority
    order with overlap resolution, magnitude expansion (`$4.2bn`, 97% of marked
    narrative amounts carry a scale suffix), range endpoints linked rather than
    collapsed, qualifiers (`up to`, `~`) stored separately from the value, and
    accounting negatives gated — an unanchored bracketed-number scan is the
    corpus's worst false-positive source (`s167(1)`, `(02) 6203 7300`).
    Candidates embedded in Australian false-positive classes (dates, times,
    phone numbers, ABNs/ACNs, legislative references, measurements,
    percentages) are rejected.
  - **Column-evidenced** (`process/money_columns.py`) — a bare number whose
    money-ness comes from its column: number format carrying a currency symbol
    (definitive), else money-vocabulary header plus predominantly numeric
    cells. Numeric cells never promote a column alone; whole-word vetoes
    suppress one (`age` vetoes `Age`, `Average Cost` survives); null markers
    are absent values excluded from the numeric fraction; the header supplies
    the column's scale (`$m`, `$'000`) and currency. A column with no
    recoverable header is left un-extracted.

  Values are exact `decimal128(38, 4)`, not floats — aggregating a 48,997-row
  register accumulates float error. Three loci are anchored in two coordinate
  spaces and never mixed: `narrative` spans are character offsets into the
  reassembled narrative in the `processing.text_source` layer (stamped on every
  row, so they join enrichment mentions and map to chunks), `table_cell` is
  `(parent_elem_order, row, col)`, `sheet_cell` is `(sheet, row, col)`.

  Header continuation rows are folded into the header: PDF financial tables
  wrap `Approved` / `Budget $m` across two rows and declare only the first,
  which previously left the column looking like a nameless run of bare
  numbers. One leading non-numeric row is absorbed when the rest of the column
  is numeric, so a genuine text data row is never eaten.

  No new dependencies. Pattern 10 (bare numbers near financial vocabulary in
  narrative) and continental number formats ship off by default.

  A number in a header no longer declares a thousands scale: the `'000`
  pattern matched the `000` inside any number, so a `Grants over $10,000`
  header multiplied every cell beneath it by 1,000.

  Tier-3 ISO codes are gated on surrounding context rather than merely
  scored lower, because several are ordinary English words in capitals:
  ungated, `TOP 10 projects` reads as ten Tongan paʻanga and
  `ALL OTHER COMPENSATION ($)` resolves to Albanian lek. A tier-3 code needs a
  currency symbol or financial trigger word nearby; in a header it must be
  parenthesised (`Value (PGK)`). Tier 1/2 codes stand alone.

  Count columns are no longer read as money. A financial table marks a count
  column `(#)` exactly as it marks a money column `($)` — the same page can
  carry `Threshold ($)` and `Threshold (#)` — but the header tokeniser dropped
  `#` entirely and promoted the count column on the vocabulary term alone.

  A veto term no longer suppresses a column whose header declares its own
  currency: `Grant Date Fair Value of Stock and Option Awards ($)` is a money
  column that happens to contain the word "date", and vetoing it lost all five
  amounts beneath it. Count columns on the same page carry `(#)` rather than
  `($)` and stay vetoed. The overridden term is still recorded in the column
  audit.

  First real-document run (four benchmark fixtures through the real pipeline,
  every span hand-checked): all 42 marked narrative amounts recovered from the
  ANAO Major Projects Report, and its `Approved Budget $m` column reconciles
  three ways — 25 project amounts summing to the table's own total row and to
  the narrative's independently written "$78.7 billion". Details and the
  measured limits in [docs/money-extraction.md](docs/money-extraction.md).

- **Table-cell reconstruction on OCR'd pages (#17), step A0 — scope and
  plumbing.** The layout pass (`_layout_blocks_and_tables`) now receives the
  per-detection OCR regions and the OCR render's pixel dimensions from the
  orchestrator, the raw material for reconstructing cells inside a detected
  table rect. Both arguments are optional, so callers that don't supply them
  keep their exact previous behaviour, and **no tables are produced yet** —
  `tables` is still returned empty on every path.

  The scope this fixes is which engines can ever reach reconstruction:
  region-based ones only. LLM/VLM engines (`mistral-ocr`, `ollama`) resolve
  reading order natively, return markdown with no regions, and are dispatched
  to `_markdown_page_block` — there are no quads to bin, so a markdown
  pipe-table parser is their separate, deferred feeder. The accuracy suite's
  extraction calls now pin `engine="paddleocr"` accordingly; under a config
  default of an LLM engine its numbers would describe a different pipeline.

  Two pieces of the reconstructor's foundation come with the seam:
  `_regions_in_rect()`, the OCR-quad → table-rect intersection by centroid
  containment, and a coordinate-space guard — the OCR render and the layout
  render are the same page at the same dpi, so unless the OCR render's
  dimensions are supplied and match, the coordinates are not known to be
  comparable and the regions are dropped with a warning rather than binned.
  Losing inputs is the correct failure; a mis-binned grid would be confidently
  wrong downstream. Deskewed pages are a separate hazard this check does not
  catch (dims survive `warpAffine`), handled by a later page-level refusal.
  Per-table debug logging records how many OCR regions fall inside each
  detected table, so the size of the gap is traceable per page before the
  reconstructor lands.
  Plan and sequencing (now folded into the standard docs) in
  [docs/decisions.md](docs/decisions.md) “Table-cell reconstruction on OCR
  pages” and [docs/evaluation.md](docs/evaluation.md) §2b.

- **Table-cell reconstruction on OCR'd pages (#17), step A1 — one shared
  grid algorithm, two feeders.** The row/column inference that
  `spreadsheet_print` already used (y-band binning, data-anchored column
  clustering, x-left cell assignment) is lifted into `ingest/table_grid.py`
  and consumed unchanged by `spreadsheet_print`; the point-space tolerances
  became parameters so pixel-space callers scale them by `dpi/72` instead of
  running ~2.8× too tight at 200 dpi. The OCR-side row-clustering preamble
  that `_spatial_sort_regions` and `_table_aware_text` each carried nearly
  line-for-line is now one shared helper (`rows_from_spans`), closing the
  repo's third table-ish duplication.

  `ingest/ocr_tables.py` is the new second feeder:
  `reconstruct_table(regions, table_rect, dpi, conf)` reduces the OCR quads
  inside a layout-detected table rect to spans and reconstructs the grid as
  a `TableData` — or returns `None`, never a partial, below its precision
  gates (minimum columns/rows, a left-edge column-fit ratio, header text
  actually recovering, and a row-fill density floor added in B2; each refusal
  debug-logged). Refusal on a hard shape is a correct round-1 outcome. The
  header band and body bands bin separately, so a first body row with a blank
  leading cell — an indented or grouped row — is no longer folded into the
  header and lost by the wrapped-cell continuation rule. Element lineage is
  deliberate: confidence comes
  from the constituent region confidences capped by the detector's, and
  `context["producer"] = "table_grid"` distinguishes reconstructed tables
  from PyMuPDF-fallback ones in the parquet. Nothing is wired into the
  layout pass yet — `tables` is still returned empty on every extraction
  path until A3.

- **Table reconstruction benchmark (#17, steps B0 + B3 + B1.2).** The
  table metric is fixed before anything is measured against it (B0): the
  DocLayNet GT aggregation no longer charges stray sub-3-span
  Table-labelled runs (footnote lines mislabelled Table) as false
  negatives — table recall on the vendored fixtures was 25% largely by
  annotation artefact, 50% after the fix — and steering's stale
  "0 predictions" layout claim is corrected.

  `tests/test_table_benchmark.py` (B3 + B1.2, `benchmark`-marked) measures
  reconstruction *conditioned on a correct table rect*, no layout detector
  in the loop: deterministic table pages rendered from the two vendored
  spreadsheet sources (3 pages × 30 rows of the Approved-providers CSV;
  one page per MSO fuel sheet), rasterised at 200 dpi, OCR'd with
  paddleocr, reconstructed with the rect known by construction, scored
  positionally against the drawn strings under a declared normalisation
  (NFKC + dash folding + whitespace collapse). Round-1 baseline: all six
  rendered-clean fixtures reconstruct with exact structure and full header
  recovery; cell accuracy 84–99%, every mismatch glyph-level OCR
  recognition rather than grid binning. The hard scan fixture
  `dense_text_548` tracks without a gate and **refuses** (the row-fill
  density gate B2 added rejects its sparse ~0.45-fill grid; pre-B2 it
  yielded a 12×12 partial against the 39×11 GT). The off-spec
  `sparse_text_344` CSV/meta GT is removed (declared non-GT).

- **Table-cell reconstruction on OCR'd pages (#17), steps A3 + A2 — the
  OCR-PDF path now produces cells.** A layout-detected table region on an
  OCR'd PDF page is handed to `reconstruct_table`, and where the grid clears
  its precision gates the page gains a `kind="table"` element with cells
  instead of the table's text being swallowed into page narrative. This is
  the first path on which `_layout_blocks_and_tables` returns a non-empty
  `tables` list. Nothing downstream changed to accommodate it: the element
  goes through the same `_table_to_element` → writer → `table_cells.parquet`
  route as native and spreadsheet-print tables, so the chunker's markdown
  projection and the money stage's `table_cell` locus pick it up as-is.

  The double-count this had to avoid is the **narrative fallback**, not the
  `[TABLE]` placeholder. Layout-derived blocks carry no text, so the "no
  block has text" fallback fires on essentially every layout-successful page
  and emits one block holding the whole page's OCR text — table content
  included. Where a table reconstructs, that narrative is now rebuilt from
  the OCR regions *outside* its rect, so the chunker sees the table once
  (as markdown) rather than twice. A page that is only a table emits no
  narrative block at all. The same absorbed regions are withheld from
  form-pair extraction, so a colon-bearing cell can't land in both a form
  element and the table. On refusal — the precision gates, or A2 below —
  the page keeps its previous behaviour exactly, byte for byte.

  `PageResult.text` stays the verbatim full-page OCR text. The subtraction
  is an element-stream concern; page text feeds text-coverage and the CER
  metrics, which compare against a transcript of the whole page.

  **A2 — deskewed pages refuse rather than mis-bin.** `preprocess_for_ocr`
  deskews via `warpAffine` before OCR when |angle| > 0.5°, so the region
  coordinates are in rotated space while the layout pass renders the raw
  page. `warpAffine` preserves the frame, so A0's dimension guard cannot
  catch this. The orchestrator now reads `"deskew" ∈ steps` off `_ocr_page`
  and the layout pass drops its cell source on such pages — a page-level
  refusal consistent with precision-first. Mapping the layout rect into
  deskewed space is deferred to the round that targets real scans. Flat
  contemporary documents, round 1's target, almost never trip deskew.

  The image path is untouched by A3 — that was A4's scope, which turned
  out to be a no-op (below).

- **Table-cell reconstruction on OCR'd pages (#17), step A4 — images were
  never a separate path; the dead extractor is gone.** A4 was scoped to
  route `ImageExtractor` through the layout pass, on the premise that
  standalone images bypassed table reconstruction. The premise was wrong:
  `extract_text` gates the legacy path-based dispatch on
  `(SPREADSHEET, DOCX, TEXT)`, and `IMAGE` is not in it — it falls through
  to `fitz.open()` + `extract_pdf_with_plan`, because PyMuPDF opens an
  image as a one-page document. Images have always reached
  `_apply_ocr_page`, so **A3 already gave them table reconstruction**;
  verified by driving a real `.png` through `extract_text` and observing a
  cellified `table` element carrying `context_producer=table_grid` beside a
  narrative paragraph with the table text subtracted.

  `ImageExtractor` was therefore unreachable from every production and
  measurement path (the accuracy suites call `extract_text` or
  `get_paddle_reader` directly). It is deleted, along with `get_extractor`'s
  unreachable `DocumentType.IMAGE` case and the `strategies.py` re-export.
  `get_extractor`'s `dpi` / `lang` / `engine` / `engine_options` parameters
  go with it — they existed only to construct `ImageExtractor`, and a
  function that silently ignores an `engine=` argument is a trap; the
  signature is now `get_extractor(profile)`, returning
  `PathExtractionStrategy`.

  This is a **breaking change for direct importers** of
  `womblex.ingest.strategies.ImageExtractor`,
  `womblex.ingest.strategies_scanned.ImageExtractor`, or `get_extractor`'s
  removed keyword arguments. Nothing inside womblex used any of them. Route
  images through `extract_text` instead — it is what the pipeline does.

  `TestImageDocumentsRouteThroughTheOrchestrator` pins the routing from the
  `extract_text` entry point, so a future change reintroducing an image
  bypass fails there rather than silently losing table reconstruction on
  every image input. `table_to_element` moved from the orchestrator to
  `ingest/views.py`, joining the reverse projections so the whole
  view↔element mapping is in one file; its body is unchanged.

  Stale claims corrected in the same pass, all of which predated this work:
  steering's "every image input … is still unchanged"; `money-extraction.md`'s
  note
  that `dense_text_548` is out of reach because it is a PNG (it is reached
  — what limits it is grid quality on a stacked-header table, which #17 B2
  owns); `get_extractor`'s docstring; CLAUDE.md's and dataflow's
  "non-PDFs via `get_extractor`"; and the generated EXTRACTION.md
  strategy-matrix row `| IMAGE | ImageExtractor (legacy) | Direct PaddleOCR |`.

- **Table-cell reconstruction on OCR'd pages (#17), step B2 — precision
  gates calibrated.** The reconstructor's precision gates were provisional
  structural constants; B2 calibrated them against the rendered-clean
  cohort (must reconstruct) and a false-table cohort (must refuse). A new
  `MIN_ROW_FILL_RATIO = 0.75` gate in `ingest/ocr_tables.py` — mean cell
  occupancy across the reconstructed body — is the load-bearing signal:
  measured, the clean fixtures fill 0.98–1.00 and the hard/false shapes
  0.375–0.49, so 0.75 sits in the empty gap. The over-segmented,
  over-merged grid a hierarchical or form shape produces is structurally
  large but mostly empty, which the existing `MIN_*` count gates and the
  left-edge `MIN_ASSIGNED_RATIO` could not see; density does.

  Effect: all six rendered-clean fixtures still reconstruct, the eight
  false-table probes (3 non-table DocLayNet pages + 5 FUNSD forms) all
  refuse — closing three false positives the provisional gates had let
  through (`diverse_layout_49` 32×3, `funsd/82200067_0069` 15×8,
  `funsd/87528321` 21×6) — and `dense_text_548` refuses rather than
  emitting its pre-B2 12×12 partial. The plan's right-edge overflow
  question is resolved by measurement: the overflow signal is 0 on every
  fixture (`column_for_x` absorbs right-of-last-column content), so density,
  not assigned-ratio symmetry, is the guardrail.

  Benchmark additions in `tests/test_table_benchmark.py`: an alignment
  projection (`cells → DataFrame`, header row → uniquified column names)
  feeding `utils/tabular_metrics.py` (`structural_fidelity` +
  `data_integrity`), so a reconstructed OCR grid is scored by the same
  metrics the spreadsheet ingest uses; the false-table cohort
  (`TestFalseTableCohort`, false-positive count gates the build); and the
  Appendix A.6 GT acceptance checker (`TestGroundTruthAcceptance`,
  parametrised over every `*_table.csv` beside a DocLayNet fixture). Wiring
  these into the generated `EXTRACTION.md` and the CI-level regression gate
  remains B4/B5.

- **Table-cell reconstruction on OCR'd pages (#17), step B4 — report + docs
  wiring.** The benchmark's table results now surface in the generated
  `docs/accuracy/EXTRACTION.md`. `generate_extraction_report` gains a
  `## Table Reconstruction` section (`tests/accuracy_reports.py →
  _table_reconstruction_section`) placed directly under the DocLayNet
  per-class layout section it decomposes: detection is stage 1 there,
  reconstruction (conditioned on a correct rect, B1.2) is this section. It
  renders three cohorts — rendered-clean (gated), the `dense_text_548`
  tracking row, and a **separate false-table table** headed by the live
  false-positive count so the precision guardrail is visible rather than
  buried. `test_fixture_accuracy`'s `_results` accumulator gains a `"tables"`
  key, and `tests/test_table_benchmark.py` *aliases* its module `_results`
  list to that key (private-list fallback if imported in isolation), so its
  existing entries flow into the shared accumulator with no duplicate
  plumbing and the session-scoped `write_report` finaliser renders them.

  Two deliberate calls beyond the plan text. **Money recall is not a
  column:** the benchmark has no labelled money ground truth
  (`docs/money-extraction.md`),
  so no honest recall can be quoted — the section notes the omission and the
  `table_cell` locus that makes reconstructed tables column-classifiable
  regardless, rather than fabricating a figure. **CHUNKING.md was annotated,  not regenerated:** its numbers predate tables landing on OCR pages and its
  generator is still unwritten, so the table-reconstruction knock-on is
  recorded under its Known Limitations rather than hand-edited with invented
  counts (it shifts on the next full regeneration).

  `docs/evaluation.md` gains §2b (Document-Table Reconstruction Accuracy),
  kept distinct from §2 (spreadsheet-file → parquet, where the source is
  already a grid): §2b's grid is *inferred* from OCR quads and can be wrong.
  It records the two-stage decomposition and the full metric set (structural
  fidelity, cell match, data integrity, false-table rate, A.6 GT acceptance).
  Only B5 (turning the structural/false-table asserts into build-failing CI
  gates) remains.

- **Table-cell reconstruction on OCR'd pages (#17), step B5 — regression
  guard.** The last open item on #17: the benchmark's round-1 *sanity*
  asserts become build-failing gates, so a future regression that collapses
  a clean grid or reintroduces a false table fails the build rather than only
  shifting a reported number. In `tests/test_table_benchmark.py`:

  - **Rendered-clean cohort — gated.** `TestRenderedCleanTables` now asserts
    per fixture that the grid reconstructs (a refusal on a clean rendered
    table is a reconstructor regression), the row *and* column counts match
    the drawn GT exactly, and cell agreement clears `MIN_CELL_MATCH = 0.75`.
    The exact row/column counts are the load-bearing structural gate (what a
    mis-binned grid breaks); the content floor — set below the measured
    minimum (0.844 on a fuel sheet, 9 pt glyph confusions) with headroom for
    OCR non-determinism — catches a structurally-correct-but-garbage grid the
    counts alone would pass, without flaking on glyph noise. `structural_fidelity`
    (which additionally compares the exact column-*name* set) stays a reported
    field, not a gate: gating on exact header identity would flake on a
    single-glyph header misread — the same OCR noise the cell floor tolerates
    — for no precision the counts + floor don't already give.
  - **False-table cohort — gated.** `TestFalseTableCohort`'s per-fixture
    `assert table is None` *is* the "false-table count == 0" gate: a single
    false positive on any of the 3 non-table DocLayNet pages or 5 FUNSD forms
    fails the build. Enforced fixture by fixture, so it holds under `-k`
    selection and `pytest-xdist` (no probe depends on another having run) and
    adds no OCR cost over the B2 cohort it tightens.
  - **`dense_text_548` — tracking, ungated.** `TestDenseTextTracking` keeps
    the sole invariant that it never emits a full false 39×11 grid; refusal
    (the post-B2 outcome) or a partial are both valid round-1 results.

  The engine is pinned to paddleocr by construction — `get_paddle_reader`
  builds the reader directly with no config-engine indirection (A0) — so a
  config default flipping to an LLM engine cannot silently turn these gates
  into no-ops. With B5 landed, every stage of #17 round 1 is complete.

### Fixed
- **A declined continental number no longer leaks its decimal tail as an
  amount.** In Australian (default) mode `find_money` correctly refuses to read
  `1.234,56` — the reading is ambiguous and `international_numbers` is the
  deliberate opt-in — but declining the candidate that *starts* at the run left
  the rest of it exposed, and `,56 EUR` is itself a complete match for the
  suffix patterns. `1.234,56 EUR` came back as `56 EUR`, a value wrong by 10³,
  which is precisely the failure the guard exists to prevent. Ambiguous numeric
  runs (continental decimals, and malformed thousands groups like `$1,23`) are
  now blocked whole, so the amount is missed rather than misread.

  Only the ISO-suffix, currency-word and symbol-suffix patterns leaked;
  prefix-marker forms were always safe, because the tail has no leading marker
  to match. That asymmetry is why the existing locale test (`€1.000,50`, a
  prefix form) passed throughout. International mode is unaffected — there the
  continental reading is the correct one.

- **CI runs the type-check and test steps again.** `ruff` was declared
  unpinned, so CI resolved 0.16.0, whose expanded *default* rule set reported
  297 errors across a tree that had been green — 233 of them pre-existing and
  unrelated, surfaced by the release rather than by any change. Lint runs
  before mypy and pytest, so both were being skipped entirely and the `money`
  op merged without CI ever executing its tests. `ruff` is now bounded
  (`>=0.16,<0.17`) so an upstream release can no longer turn CI red on its own;
  raising that ceiling is a deliberate commit that also clears whatever the new
  defaults flag.

  The tree is now clean under 0.16.0's defaults. `--fix` resolved 174 findings
  mechanically; `BLE001`, `S110` and `S112` are suppressed in
  `[tool.ruff.lint]` because the codebase deliberately does what they flag —
  every site is a batch- or per-document isolation boundary, and narrowing
  those handlers to named exception types would let one malformed document
  abort a 1500-document run. The remaining 55 were resolved individually.
  Two are worth noting beyond the mechanical: the readability smoke-tests in
  `store/output.py` / `store/shard_audit.py` (`pq.ParquetFile(p).metadata`,
  whose whole purpose is to raise on a corrupt footer) now bind their result
  rather than being deleted as useless expressions, and `analyse/graph.py`
  carried a crossreference edge whose `source` was the same value on both
  arms of its conditional — collapsed to the value it already produced, so
  behaviour is unchanged, but the condition looks like it was meant to
  distinguish something.

- **The Isaacus test suites run in CI.** CI installed `.[dev,cloud]`, omitting
  the `isaacus` extra, so the enrich / graph / query / embed modules hit their
  module-level `importorskip("isaacus")` and skipped wholesale — 66 tests that
  need no API key never ran. CI now installs the extra; only the 10 tests
  requiring a live endpoint still skip on the missing key. Installing the SDK
  also unmasked a real typing error in `process/chunker.py`, where
  `isaacus_client` is deliberately `object | None` to keep the module SDK-free;
  the narrowing to semchunk's concrete client type now happens at the call
  boundary.

- **`mypy` passes with `openpyxl` installed.** The new read-only openpyxl pass
  below was the codebase's first import of it and had no entry in the
  `ignore_missing_imports` override list, so the type-check leg failed on a
  missing stub package.

- **Spreadsheet extraction preserves `number_format` and a numeric
  `value_type`.** `ingest/spreadsheet.py` read cells with pandas
  (`dtype=str`), which discards both, so every `sheet_cell` element landed with
  `value_type="text"` and `number_format=None` despite `ELEMENT_SCHEMA` having
  columns for each. A second read-only openpyxl pass now supplies them.
  Values are untouched — the pandas read stays authoritative, so the verbatim
  contract ("1,234" stays "1,234") is unchanged.

  This matters because a register's money column is frequently identifiable
  *only* from its format: a GrantConnect award export carries `$#,##0.00` on
  48,997 cells whose text is a bare `50000`, and no cell, header or value in
  that workbook contains a currency symbol. The format was the sole
  unambiguous currency marker in the file and was being dropped at the
  extraction boundary, where no downstream stage could recover it. Only
  non-`General` formats are retained, keeping the lookup small. CSV sheets have
  no cell formats and are unaffected; a failed openpyxl pass logs a warning and
  leaves the fields unset rather than failing extraction.

## [0.2.0] - 2026-07-19

### Added
- **Pre-extracted records ingest (`ingest/records.py`).** Turns already-clean
  text records (a JSONL corpus; the Open Australian Legal Corpus) straight into
  the standard element-shard layout (`*.elements.parquet` + sidecars +
  `*._manifest.parquet`) so the `enrich → chunk → embed → graph-refresh`
  pipeline runs over a pre-extracted corpus unchanged — unlike the register
  ingests (`gnaf`/`abn`/`geo`) which *bypass* the NLP pipeline, this one *feeds*
  it. `source_hash = sha256(record_id + text)` is content-addressed (unchanged
  records are cache hits on re-ingest); text is split into paragraph blocks so
  the reassembled narrative round-trips. Corpus-agnostic — a
  `RecordFieldMapping` (declared by a thin `stories/<corpus>` config) names the
  id / text / provenance fields. Record metadata lands in a
  `*.provenance.parquet` sidecar (`store/provenance_output.py`) consolidated
  into a run-root `manifest.parquet` (source_hash → provenance).
- **Token-budget request packer (`utils/token_packer.py`).** Isaacus rate
  limits bind on *tokens per request/window*, not request count, so requests
  are packed by exact local token counts from the kanon-2 tokenizer:
  `pack_by_tokens` groups items to `min(max_items, token_budget)`; an
  over-budget item is sent solo; `split_on_boundaries` splits an over-ceiling
  document on blank-line boundaries into offset-tagged segments. `TokenCounter`
  is a cached, offline wrapper over the tokenizer.
- **Enrichment — token-aware batching + long-doc split (`enrich_stage.py`).**
  Replaces the one-doc-per-call loop with packer-driven requests of
  `min(max_texts_per_request=8, token_budget)` (8× fewer requests for small
  docs; token-aware so a batch of long judgments never overpacks a
  429-triggering request). A doc over `split_ceiling` is split and its
  per-segment results offset-merged (`analyse/enrich_merge.py`). `enrich.py`
  honours a `Retry-After` header on 429. New `EnrichmentConfig` knobs:
  `tokenizer`, `max_texts_per_request`, `token_budget`, `split_ceiling`.
- **Graph-edge refresh stage (`analyse/graph_refresh.py`, `womblex
  graph-refresh`).** Offline, deterministic rebuild of mention→chunk edges from
  the entity + chunk sidecars (both carry char offsets) — needed because AI
  chunking runs *after* enrichment, so the enrich-time graph has no chunk edges
  yet. Populates `enrichment_entities.chunk_index` and refreshes
  `*.graph_edges.parquet`, preserving hierarchy/citation edges. Idempotent.
- **Offline kanon-2 tokenizer.** The tokenizer is vendored under
  `_models/kanon-2-tokenizer` and resolved locally by both the token packer and
  `create_chunker` — no Hugging Face round-trip per run, offline-safe.
- **Distributed / cloud execution (`womblex[cloud]`).** Optional scale-out for
  long batch runs without changing the local CPU-first default. Three pieces:
  (1) `store/remote.py` — an fsspec stage-in/stage-out object-storage adapter
  (S3/MinIO/GCS/local) that confines all remote-storage knowledge to one place
  so the `Path`-based stages stay untouched; (2) `cloud/queue.py` — a Postgres
  `FOR UPDATE SKIP LOCKED` job queue over one `womblex_jobs` table where the row
  `status` *is* the distributed checkpoint (idempotent re-enqueue on
  `(run_id, batch_num)`, per-job retry, crashed-worker requeue); (3)
  `cloud/worker.py` — a worker that claims a batch, stages its inputs, runs the
  shared `batch.process_batch` body, and publishes `batch-NNNN.*.parquet` shards
  back. New CLI: `womblex enqueue` / `worker` / `jobs` / `finalize` (the last
  consolidates a distributed run's shard manifests into
  `<run>/manifest.parquet` in the store — the explicit end-step `womblex run`
  performs locally). Outputs are the ordinary shard layout, so `manifest` /
  `chunk --shards` consume a distributed run exactly like a local one.
  `process_batch` is also now the single shared body behind `womblex run`, so
  local and distributed modes cannot diverge.
- **Container image + compose stack.** `Dockerfile` (extraction + `[cloud]`)
  and `docker-compose.yml` bundling Postgres (queue), MinIO (object store), and
  horizontally scalable workers (`docker compose up --scale worker=N`).
- **CI security job.** `ci.yml` gains a `security` job: Semgrep SAST over `src/`
  with the Python + OWASP Top Ten rulesets (blocking) and `pip-audit`
  dependency scanning (informational — the ML dep tree carries advisories we
  can't action directly). The test job now also installs the `cloud` extra so
  the object-storage tests run in CI.
- **ABN Lookup bulk extract ingest.** New `ingest/abn_bulk.py` stream-parses
  the ABR bulk extract XML files (`yyyymmddPublicNN.xml`, ~6 GB uncompressed
  across 20 files) with constant memory and writes two Parquet files per
  input: `<stem>.parquet` (one row per ABR record — ABN/status, entity type,
  main entity name or legal-entity name parts with given names as separate
  `given_name_1` / `given_name_2` columns since a single given name may
  itself contain a space, state/postcode, ACN, GST) and
  `<stem>_names.parquet` (one row per registered name — main/legal, business,
  trading and DGR fund names keyed by ABN, ready for `link/` register
  consumption). Values are verbatim strings, absent optionals are `""`, and
  provenance (schema version, source file, MD5, row counts) rides as parquet
  metadata — the `ingest/gnaf.py` pattern. Failures are isolated per file:
  any error (malformed XML, read/write failure) logs with the source name,
  removes partial output, and lets the directory ingest continue. New
  `womblex ingest-abn <file|dir>` CLI command; bypasses the NLP pipeline.
  (`ingest/abn_bulk.py`, `cli/ingest.py`, `tests/test_abn_bulk.py` —
  all-synthetic fixtures.) The shared MD5 helper moved to
  `utils/checksum.py`, replacing the per-module copies in `ingest/gnaf.py`
  and `ingest/geospatial.py`.
- **Spreadsheet preamble/header detection.** Export products that open with
  title rows, generated-date lines or `key: value` metadata blocks above the
  real header (e.g. AusTender contract-notice exports, agency stats
  workbooks) previously had the first row parsed as the header, with pandas
  fabricating `Unnamed: N` column names that landed verbatim-violating cell
  values on the element stream — and ragged CSVs (one-field title row above
  a wide header) failed outright. Sheets are now read with `header=None`
  (CSVs via a new field-count-sniffing `read_csv_raw`, capped at `nrows`
  when sampling) and split via `split_preamble`: the header is the
  candidate row (≥2 non-empty cells in a 10-row window) that starts the
  longest run of table-consistent rows below it — a blank separator or the
  wider table below breaks a title/metadata row's run, ties prefer the
  wider candidate, single-cell section rows are neutral, and a width-ratio
  rule plus row-0 fallback covers header-only and single-column sheets.
  Preamble rows land verbatim on the sheet_meta element
  (`meta["preamble"]`) and the row-0-is-header contract of the cell grid is
  preserved for downstream table views. Header-first, single-column and
  uniformly narrow (key/value, glossary) sheets are unaffected. Detection
  (`_detect_spreadsheet`) shares the same reader and split with headroom so
  the 500-row classification sample is unchanged, and `SheetInfo.key_column`
  resolves against the real header. (`ingest/spreadsheet.py`,
  `ingest/detect.py`.)
- **Run-level document manifest.** `womblex run` now consolidates the per-batch
  `batch-NNNN._manifest.parquet` sidecars into a single
  `<run_root>/manifest.parquet` at the end of the run — the published
  documents table mapping `source_hash` (the join key on every chunk/sidecar
  row) back to `doc_id`, `filename`, extraction method, counts and status, so
  shipped chunks are attributable to their source documents. A new
  `womblex manifest --shards <dir> [-o PATH]` command regenerates it for
  existing runs. (`store/run_manifest.py`, `cli/pipeline.py`.)
- **Shippable enrichment graph.** `enrich_shards` now writes a
  `*.graph_edges.parquet` sibling per batch alongside the entities/meta
  sidecars — the Kanon-2 document graph (containment, segment hierarchy,
  person/location hierarchy, citations, cross-references, contact-info and
  date relations) flattened to the existing `GRAPH_EDGE_SCHEMA`, with
  `document_id` carrying the `source_hash` so it joins the other sidecars.
  When the batch already has a `*.chunks.parquet` sibling, narrative chunks
  are mapped in so the graph includes mention→chunk edges. On resume, a batch
  missing its graph sidecar is re-enriched so prior runs gain it (the graph is
  only buildable from the live enrichment result). New
  `write_graph_edges_shard` / `read_graph_edges` / `graph_edges_path_for`.
  (`analyse/enrich_stage.py`, `store/enrichment_output.py`.)
- **CLI fix — `womblex chunk --shards` + `--config` combinable.** The two
  flags were in a `required=True` mutually exclusive argparse group, which made
  the `--shards` branch's config handling (chunking settings, `chunking_model`
  for AI chunking, `processing.text_source`) unreachable from the CLI —
  per-stage AI chunking was dead-ended. They now combine: `--shards` with
  `--config` sources chunking settings from the YAML; `--config` alone remains
  the E2E composition mode. (`cli/pipeline.py`.)
- **Single-enrichment reuse for AI chunking.** When AI chunking
  (`chunking.chunking_model`) and the `enrich` stage are both on, the enrich
  stage now persists the raw ILGS Document per doc to a new
  `*.enrichment_doc.parquet` sidecar (opt-in `enrichment.persist_document`,
  auto-enabled by `WomblexConfig`), and the chunk stage reuses it for semchunk's
  AI path instead of re-enriching — eliminating the double Kanon-2 call. Reuse is
  gated by a **byte-identity guard**: a persisted `Document.text` is used only
  when it equals the chunk stage's freshly reassembled narrative for that
  `source_hash`; on any mismatch (different `text_source`, stale/corrupt blob,
  absent sidecar) the doc falls back to self-enrich, so offsets can never desync
  the PII mention↔chunk mapping. Requires running `enrich` before `chunk`; the
  `WomblexConfig` validator now warns about that ordering rather than about
  double-enrichment. New self-contained `store/enrichment_doc.py`;
  `enrich_documents_raw` / `enrich_document_raw` expose the raw SDK Document;
  `chunk_batch` gains `narrative_overrides`. Verified live against
  `kanon-2-enricher` (gates in `docs/decisions.md`). (`store/enrichment_doc.py`,
  `analyse/enrich.py`, `analyse/enrich_stage.py`, `process/chunker.py`,
  `process/chunk_stage.py`, `config.py`, `cli/link.py`.)
- **AI chunking pass-through (semchunk 4).** `ChunkingConfig.chunking_model`
  (default `null`) enables semchunk 4's AI chunking — chunk boundaries follow
  the Isaacus enricher's (`kanon-2-enricher`) structure spans instead of the
  offline token/recursive split. Opt-in and off by default, so callers using a
  non-Kanon tokeniser keep purely offline chunking (composable). `create_chunker`
  now forwards `chunking_model`, `isaacus_client`, and `tokenizer_kwargs`
  straight to `semchunk.chunkerify` (thin-adapter doctrine — semchunk's params
  are the feature surface); threaded through both the E2E `run_chunking` path and
  the per-stage `chunk_shards`. Graph-reuse across the chunk + enrich stages (so
  the narrative is enriched once, not twice) is now implemented — see
  "Single-enrichment reuse for AI chunking" above. Bumps `semchunk>=3.0` →
  `>=4.0`. (`process/chunker.py`, `config.py`, `process/chunk_stage.py`,
  `operations/chunk.py`.)
- **`spellfix` stage — dictionary-gated OCR character-confusion repair
  (`womblex spellfix`).** A separate, opt-in cleaning op (distinct from the
  fidelity-neutral `normalise`) that fixes digit/letter glyph confusions
  (`chi1d`→`child`). Validates candidates against the bundled en_AU Hunspell
  dictionary (`spylls`, harvested from the Australian Writing MCP; MIT/SCOWL) and
  rewrites a token only on three gates: out-of-dictionary trigger,
  single-character in-dictionary candidate, and a *unique* such candidate.
  Default Tier A swaps only OCR digit→letter homoglyphs (length-preserving);
  Tier B general edit-distance-1 is opt-in (`--general` / `general_edits`,
  carries proper-noun risk). Repairs at the **element layer** (reads
  `*.elements.parquet`, chaining off the normalise overlay when present) and
  writes a `*.spellfix_text.parquet` element-text overlay + a
  `*.spellfix_corrections.parquet` audit — raw elements untouched. New deps:
  `spylls`; bundled dict under `_models/en_AU`. (`process/spellfix.py`,
  `process/spellfix_stage.py`, `store/spellfix_output.py`, `cli/spellfix.py`,
  `SpellfixConfig`.)
- **Composable element-text overlays via one `processing.text_source`.** New
  `process/text_overlay.py` resolves the normalise / spellfix element-text layer
  selected by a single pipeline setting (`elements` | `normalised` | `spellfix`)
  and applies it before reassembly at **both** the chunk and enrich sites, so
  chunking, embeddings, Kanon-2 enrichment and PII all consume the same repaired
  text in one offset coordinate space. Deliberately one knob (not per-stage):
  enrichment runs on the whole document and PII maps mention offsets onto chunks
  via `chunk.start_char`, so the enricher input and chunk source must match.
- **Enricher `overflow_strategy` (default `auto`).** `enrich_documents` /
  `EnrichmentConfig` now pass `overflow_strategy` to `enrichments.create`,
  defaulting to `auto` (vs upstream `null`, which errors on >16k-token inputs).
  Kanon-2 chunks long documents internally and stitches the ILGS graph back into
  a single prediction; returned span offsets still index the full source, so the
  PII offset mapping is unaffected. Fixes long FOI bundles erroring on enrichment.
- **`score --text-source` — CER of extraction vs normalisation.** `womblex
  score` (and `score_labels`) now accept `text_source={elements,normalised}`:
  `normalised` reassembles the labelled page from the `*.normalised_text.parquet`
  sidecar instead of the verbatim element stream, so a caller can measure how
  the cleanup/normalisation stage changes CER against the same GT.
- **Benchmark: ACT-ECI labelled-pages raw-vs-normalised CER.** New
  `TestActEciLabelledPages` (`-m benchmark`) extracts each labelled page, scores
  raw extraction and normalise-stage output against the per-page GT, and reports
  a per-strategy `Raw CER / Norm CER / Δ` table in `docs/accuracy/EXTRACTION.md`.
  Degenerate GT (<20 chars) excluded; a regression guard asserts normalisation
  never worsens CER. (Fixtures cohort expanded 7→19 labelled pages.)
- **`quality` stage — chunk-quality annotation sidecar (`womblex quality`).**
  Reads `*.chunks.parquet` and writes a `*.chunk_quality.parquet` sibling
  (joined on `(source_hash, chunk_index)`) with ML-readiness flags
  (`char_len`, `alpha_frac`, `is_short`, `boilerplate_flag`) and cross-batch
  duplicate cluster ids (`exact_dup_id`, `near_dup_id`). Duplicate clustering
  is self-contained (no datasketch dep): `exact_dup_id` over
  whitespace/case/punctuation-normalised text, `near_dup_id` via a fixed-seed
  MinHash+LSH (default 64 perms / 4 bands ≈ Jaccard 0.92). Annotation only —
  chunk text is never mutated; runs as a single global pass since dedup is
  corpus-wide. `boilerplate_patterns` are corpus-driven config, never
  hardcoded. New `QualityConfig`; 5 unit tests.
- **`normalise` stage — `unicode_hygiene` transform.** Folds unicode
  whitespace (NBSP, en/em spaces, ideographic space, U+2028/9 separators) to
  ASCII space/newline and strips zero-width marks, BOM and stray control
  chars; smart quotes and em/en dashes are preserved. New `unicode_hygiene`
  toggle on `NormaliseConfig` (default on), composed ahead of the existing
  transforms. 4 new unit tests.

### Fixed
- **OpenCV 5 compatibility in skew detection.** `detect_skew_angle`
  (`ingest/heuristics_cv2.py`) unpacked `HoughLinesP` segments as `line[0]`,
  which assumes OpenCV 4's `(N, 1, 4)` layout; OpenCV 5 flattens to `(N, 4)`,
  turning `line[0]` into a scalar and crashing every OCR-path extraction
  (`TypeError: cannot unpack non-iterable numpy.int32`). The segments are now
  reshaped to `(-1, 4)` before unpacking, which accepts both layouts. New
  direct unit tests pin both shapes so the regression no longer needs the
  full OCR fixture suite to surface.
- **mypy no longer pins `python_version = "3.11"`.** The pin forced the CI
  3.12 matrix leg to re-check under 3.11 grammar — redundant with the 3.11
  leg, and broken once numpy ≥ 2.5 (3.12-only) began shipping PEP 695 `type`
  statements in its stubs, which mypy rejects under a 3.11 target. Each leg
  now checks at its own interpreter version.
- **Register manifest now covers `ingest-geo` and derives roles from footer
  metadata only.** `cmd_ingest_geo` never called `write_register_manifest`
  despite the documented `abn`/`gnaf`/`geo` coverage, and the module's
  namespace whitelist said `geo` while `ingest/geospatial.py` writes
  `geospatial.*` footer keys — so geo outputs could not be indexed at all.
  The namespace is now taken from whichever `<ns>.source_file` footer key is
  present (no per-register registry to keep in sync), geo ingest writes the
  manifest like ABN/G-NAF, and the ABN ingest tags each output with an
  `abn.role` footer key (`records`/`names`) so the manifest's role column
  comes from metadata rather than the `_names` filename suffix — the exact
  glob-style fragility the manifest exists to remove. ABN outputs written
  before this change lack the role key and re-index as `records`; re-run
  `ingest-abn` to restore the distinction. Also renames the module's
  `RUN_MANIFEST_FILENAME` constant to `REGISTER_MANIFEST_FILENAME` — it had
  borrowed the *run* manifest's constant name from `store/run_manifest.py`
  while naming a different artefact.
- **`RemoteStore` no longer leaks s3fs-shaped options into non-S3 backends.**
  `storage_options_from_env()` built AWS-style kwargs (`key`, `secret`,
  `client_kwargs.endpoint_url`) and `from_uri` applied them to *any* remote
  protocol, so a `gs://`/`az://` store with AWS env vars set (common in mixed
  environments) got misconfigured. The helper now takes the target URI and
  returns options only for `s3://`; other backends authenticate via their own
  native mechanisms. Also: `womblex enqueue`'s batch-size fallback now reads
  `ProcessingConfig()` instead of restating the default, and the worker
  derives its shard-upload glob from the `BatchOutcome.shard_path` the batch
  reported, so the `batch-NNNN` naming scheme lives only in `womblex.batch`.
- **K9-fig — full-page scans no longer dropped from chunking as `figure`.**
  The dominant-region fallback in `_layout_blocks_and_tables` collapsed a
  whole page's OCR onto one block tagged with the largest layout region's
  kind; when that was DocLayNet `Picture` → `figure`, a full-page scanned
  document became a single `figure` element, which is excluded from the
  chunk narrative (`figure` ∉ `TEXT_KINDS`) — silently losing the document.
  New shared helper `_ocr_region_block_type(text, layout_kind)` promotes a
  non-text fallback kind to `paragraph` when the OCR yields ≥5 words; sparse
  output (page-number stamps, bare logos) keeps `figure`. (The original E4
  audit mis-attributed this to `_ocr_image_regions`; that path now routes
  through the same helper.) On the ACT-ECI corpus: `figure` 1,200→154,
  `paragraph` +1,046, and all 16 previously zero-chunk complaint documents
  now produce chunks (docs-with-chunks 2,610→2,626). 4 new unit tests.

### Added
- **I7 — entity-link sidecar: `womblex enrich` + `womblex link` per-stage CLIs.**
  Two new per-stage stages mirroring `womblex chunk --shards`, each with an
  independent `CheckpointManager` and per-batch sibling parquets.
  `womblex enrich --shards <dir>` reassembles each doc's narrative
  (`reassemble_narrative`), calls the Kanon-2 enricher one doc at a time
  (per-doc failure isolation), and writes `*.enrichment_entities.parquet` +
  `*.enrichment_meta.parquet` (reusing `store/enrichment_output.py` schemas,
  keyed on `source_hash`). `womblex link --shards <dir> --config <yaml>`
  resolves enrichment candidates (corporate persons + address locations) to a
  reference register and writes `*.entity_links.parquet`. **Generic by design:**
  the schema uses an `entity_type` discriminator (no domain columns), the
  matcher (`link/matcher.py`) is generic record-linkage (alias → address-exact
  → token-set name-fuzzy, stdlib `difflib`, no new dependency), and the corpus
  declares register column-roles via the new `linking`/`reference` config — the
  library knows nothing about specific registers. Reference loading
  (`link/reference.py`) is bundle-aware by interface (CSV implemented; the
  multi-file/geospatial seam is reserved, not built). Doc-grain attribution is
  a derived read view over the persisted mention-grain rows, not a second file.
  New `isaacus` is the optional extra (`uv sync --extra isaacus`). Live smoke
  over the 17-doc Artemis set attributed 15/17 to the correct canonical service
  (`SE-40002132`); the 2 misses are an enrichment-recall gap and an
  OCR-typo+no-address doc, not matcher faults. New `tests/test_link.py` (23) +
  `tests/test_enrich_stage.py`; full fast suite green.
  - **Matcher** uses stdlib `difflib` only (no rapidfuzz dependency): alias →
    address-exact → name-fuzzy, where name-fuzzy combines a token-set ratio
    (suburb-suffix recall, cross-brand precision) with OCR-tolerant per-token
    char similarity (folds "Earty"→"Early" while still rejecting a different
    brand). With OCR tolerance the Artemis smoke reaches **16/17**.
  - **`enrich`** isolates per-doc failures and, critically, does **not**
    checkpoint a doc whose enrichment errored — a transient/connection failure
    stays unprocessed so a resume retries it (regression-tested).
- **`womblex embed --shards` — chunk embeddings stage (Kanon-2 embedder).**
  `analyse/embed.py` (thin `embeddings.create` wrapper: 128-text batching, 429
  retry, order-preserving, `retrieval/document`/`query` task-aware) +
  `analyse/embed_stage.py` (`embed_shards` over `*.chunks.parquet` →
  `*.embeddings.parquet`, one vector per chunk, per-stage `CheckpointManager`,
  batch-level failure isolation) + `cli/embed.py` + `EmbeddingConfig` +
  `EMBEDDINGS_SCHEMA`/IO. The substrate for downstream search/clustering and a
  doc→entity attribution backstop for no-extraction docs. `tests/test_embed_stage.py`.
- **I8–I10 — `womblex pii --shards` per-stage CLI: graph-driven detection +
  `<PERSON_n>` masking.** Reads `*.chunks.parquet` + `*.enrichment_entities.parquet`
  and writes `*.pii_spans.parquet` (audit; one row per span with `entity_id` +
  `replacement`) plus a masked `*.clean_text.parquet` (publishable text layer,
  drop-in for chunks). The Kanon-2 graph is the primary entity source
  (`natural`→PERSON, `address`→ADDRESS); graph spans map onto narrative chunks
  via `chunk.start_char`. Masking is **terminal** — applied after enrich + embed,
  never rewriting the raw chunks that feed Isaacus. Tags are typed + numbered per
  document off the graph entity (`<PERSON_1>`…). `pii/cleaner.py` refactored to a
  span-returning `detect_spans()` + shared `_anonymize()`; the regex/cosine
  backstop is now opt-in (`PIIConfig.use_regex_backstop`, default off — low
  precision). New `store/pii_output.py`, `pii/pii_stage.py`, `cli/pii.py`,
  `PIIConfig.write_clean_text`. `tests/test_pii_stage.py`.
- **I3 — `womblex redact --shards` per-stage CLI.** `womblex redact` is
  now dual-mode, mirroring `womblex chunk`: `--shards <dir> --pdfs <dir>`
  runs per-stage redaction detection over an existing extraction shard
  directory and writes `*.redactions.parquet` siblings; `--config <yaml>`
  runs the E2E extract+redact path unchanged. The `--shards`/`--config`
  group is mutually exclusive and required. `--pdfs` is mandatory in
  `--shards` mode because detection rasterises the source pages (unlike
  chunking, which works purely from the element stream). The per-stage
  path calls the existing `redact.batch.annotate_redactions_for_shards`
  engine via a shared `_run_redact_shards` helper. 8 new CLI tests.

### Changed
- **`operations.py` split into an `operations/` package.** The 902-line module
  (over the 750-line cap) became one module per independent operation
  (`models`/`extract`/`redact`/`chunk`/`pii`/`enrich`/`persist`), each ≤90
  lines. The flat import surface (`from womblex.operations import run_extraction`,
  …) is preserved by `operations/__init__` re-exports — no caller changes.
  Behaviour-neutral; `test_integration.py` patch targets for `create_chunker`
  moved to `womblex.operations.chunk`.
- **Resume-integrity self-heal generalised across stages.** `store/shard_audit.py`
  gains `reconcile_stage_checkpoint_with_shards(mgr, dir, *, suffix)`; the chunk
  reconcile now delegates to it, and `enrich`/`link`/`embed` wire it (+
  `--no-verify-resume`) — so every `CheckpointManager`-backed stage drops +
  re-does batches with corrupt sidecars on resume, identically.
- **I5 — SemChunk wrapper audit (P2).** Audited `process/chunker.py`
  against semchunk 3.2.5: `create_chunker` exposes every `chunkerify`
  creation parameter and `chunk_batch` passes every relevant
  `Chunker.__call__` parameter through (`offsets=True` pinned because
  Womblex needs char offsets for page mapping). No semchunk-native
  surface is reimplemented or shadowed. **Removed the dead
  `ChunkingConfig.batch` flag** — it mapped to no semchunk parameter,
  was consumed by no code path, and its description referred to the
  pre-I2 per-document-vs-batch behaviour that I2 deleted (chunk_batch
  always batches the whole input list). **Widened `chunk_size` to
  `int | None`** so semchunk's auto-derive path (`None` → size from the
  tokeniser's `model_max_length`) passes through faithfully; the default
  stays `480` (the Kanon-2 window), so behaviour is unchanged unless a
  config explicitly sets `chunk_size: null`. Documented the adapter
  boundary explicitly in the `chunker.py` module docstring, the
  `ChunkingConfig` docstring, and `docs/extraction.md`; the three
  default divergences from upstream (`tokenizer`, `chunk_size=480`,
  `processes=1`) are each annotated with their corpus reason. Pure
  thin-adapter cleanup — chunk output is byte-identical to I2 by
  construction (the removed field was never read; the new `None` path is
  opt-in). 98 chunker/config/pipeline/output tests pass; 79 integration
  tests pass.
- **`annotate-redactions` is now a deprecated back-compat alias** for
  `redact --shards <dir> --pdfs <dir>`. Its positional-argument surface
  (`annotate-redactions <shards> <pdfs>`) is preserved verbatim and routes
  through the same `_run_redact_shards` helper, so existing scripts keep
  working with byte-identical output. New callers should prefer
  `redact --shards`. The redact stage retains the engine's JSON
  `--checkpoint` rather than the `CheckpointManager` used by `chunk`;
  unifying the two is a deferred P1 follow-up.

### Added
- **I2 — `womblex chunk --shards` per-stage CLI + `chunks.parquet`
  sidecar.** New `CHUNKS_SCHEMA` (source_hash, chunk_index, text,
  start_char/end_char, content_type, has_redaction, page_start /
  page_end) in `store/output.py` with `write_chunks` / `read_chunks` /
  `verify_chunks_persistence`. New `process/chunk_stage.py` walks a
  shard directory, reassembles narrative + tables from the element
  stream per source_hash, calls the single `chunk_batch` engine, and
  writes a `*.chunks.parquet` sibling per batch. Per-stage
  `CheckpointManager` keyed `<dataset>_chunk_checkpoint.json`;
  chunks-side resume integrity in `shard_audit.scan_chunks_directory`
  / `reconcile_chunk_checkpoint_with_shards` archives corrupt
  `*.chunks.parquet` independently of the element-stream files. The
  shared `chunk_batch` powers both per-stage `chunk_shards` and E2E
  `operations.run_chunking`, so `--shards` and `--config` modes feed
  semchunk identical inputs.

### Changed
- **`process/chunker.py` collapsed against semchunk v3+ surface.**
  Deleted per-doc wrappers `chunk_text`, `chunk_texts_batch`,
  `chunk_document` (+ `_chunk_document_sequential` /
  `_chunk_document_batch` dispatchers) — semchunk already batches
  across a list of texts and parallelises over `processes` workers
  when handed one. The new single entry point `chunk_batch(inputs,
  chunker, ...)` flattens every doc's narrative into one semchunk
  call (with `overlap`) and every doc's table markdowns into another
  (no overlap), so `processes` and the progress bar parallelise
  across the entire batch instead of being thrown away per-document.
  `TextChunk` gained `page_start` / `page_end` (nullable); the
  redaction-split repair pass propagates page spans across a merge.
  New helpers `reassemble_narrative`, `collect_tables_from_elements`,
  `build_chunk_input` formalise the "element stream → ChunkInput"
  projection shared by both invocation paths.
- **`operations.run_chunking` rewired through `chunk_batch`.** Builds
  one `ChunkInput` per completed result from
  `dr.extraction.elements` (canonical), not `dr.extraction.full_text`
  (which is derived from `pages` and reflects in-memory mutations).
  Behaviour change: in-memory PII / redact-blackout mutations to
  `pages[i].text` no longer flow to chunks under `womblex run`.
  Aligns the E2E path with the per-stage one (both consume the
  element stream); future PII / redact stages will reattach via
  their sidecars per P1.

### Added
- **Shard integrity scan on `--resume` (E1).** New
  `womblex.store.shard_audit.reconcile_checkpoint_with_shards` runs at
  the top of `cmd_run` when `--resume` is given. Walks every batch's
  four sibling parquet files: confirms presence + non-empty + parquet-
  readable, and that manifest `elements_count` / `table_cells_count` /
  `form_fields_count` sums match the actual sidecar row counts. Any
  batch failing a check has its `doc_id`s dropped from the checkpoint
  and its files renamed with a `.corrupt` suffix so reader globs
  (`*.elements.parquet` etc.) skip them; the dropped docs get re-
  extracted into new batches past the high-water mark. Batches whose
  manifest is itself unreadable can't be reconciled automatically (no
  way to enumerate `doc_id`s) — they're logged loudly and the operator
  is told to re-run without `--resume`. Defaults on; opt out with
  `--no-verify-resume`. Closes the silent-failure class of post-write
  filesystem corruption (drive glitch, partial sync, manual deletion)
  that motivated the i1b batch-0087 0-byte incident.

- **`womblex verify-shards` CLI (E2).** Audits a run / shard directory
  for corruption + cross-batch consistency; takes a shard dir or a run
  root (auto-detects `documents/`). Reports per-batch integrity, total
  elements / methods / kind counts, dupe and empty hashes. With
  `--compare-to <other>` produces a side-by-side diff against another
  run (useful for K-cluster-style "what changed between two
  extractions" investigations — promotes the ad-hoc `i1b_audit.py`
  pattern to first-class library + CLI). Optional `--input-dir <pdfs>`
  surfaces source-vs-manifest count drift. Exits 2 when corruption is
  detected so CI / cron pipelines can fail loudly. New module:
  `womblex.store.shard_audit`. New tests in `tests/test_shard_audit.py`
  (19).

### Changed
- **Manifest schema gains `doc_id` column.** `MANIFEST_SCHEMA` now
  carries the extraction's `doc_id` directly, removing the implicit
  `Path(filename).stem == checkpoint.doc_id` coincidence that previously
  bound the resume reconcile join. The reader (`read_manifest`) is
  backward-compatible: manifests written before the bump derive
  `doc_id` from `Path(filename).stem` on read so existing runs reconcile
  without re-extraction. Parser version bump is intentionally deferred
  — the schema is additive and reads gracefully.

### Added
- **K7(b) — Document-layout YOLO model (DocLayNet).** New default layout
  checkpoint `yolo11n_doc_layout.pt` (5.37 MB,
  [Armaggheddon/yolo11-document-layout](https://huggingface.co/Armaggheddon/yolo11-document-layout),
  MIT) replaces the COCO-trained `yolov8n.pt` as the primary layout
  backend. `YOLOLayoutAnalyzer` detects the loaded model's taxonomy from
  its class names: DocLayNet's 11 document classes (Caption, Footnote,
  Formula, List-item, Page-footer, Page-header, Picture, Section-header,
  Table, Text, Title) map directly into womblex `ElementKind` values
  via the new `_YOLO_DOCLAYNET_LABEL_MAP`. COCO weights remain as a
  best-effort fallback when the DocLayNet checkpoint isn't resolvable.
  Inference imgsz follows a per-taxonomy default (DocLayNet: 832, COCO:
  640) — empirically equivalent on this corpus to the model card's 1280
  recommendation while matching COCO speed; override to 1280 when
  small-class (Caption / Footnote) recall matters. Closes the
  1,587-element `kind='figure'` mis-classification on scanned pages
  tracked in docs/decisions.md; unlocks Caption / Footnote producers
  (K6 closes as a side effect).

- **`footnote` ElementKind.** New text-bearing kind added to
  `ElementKind`, `TEXT_KINDS`, and `_BLOCK_TYPE_TO_KIND`. Primary
  producer is the DocLayNet `Footnote` class via the new label map.
  Downstream stages (PII / redact / chunk) operate on text kinds and
  pick up the new kind automatically through `TEXT_KINDS`. Future
  iterations may refine signatory / footnote separation now that the
  distinction is preserved.

- **K2′ — OCR form-pair bboxes.** New `_extract_form_pairs_from_regions`
  in `ingest/forms.py` walks per-region OCR detections (PaddleOCR /
  RapidOCR per-line bboxes) and produces `FormField` entries with real
  positions. `_apply_ocr_page` prefers this path; the legacy
  `_extract_form_pairs_from_lines` survives as a fallback for LLM-OCR
  engines that resolve reading order natively and don't emit per-region
  bboxes. Closes the K2′ silent-zero-bbox issue on 4,184 of 5,183 OCR
  form elements (80.7%). Same plumbing unblocks inline-per-span
  redaction markers on raster pages (P6 option (c); see docs/decisions.md).

### Changed
- **`@pytest.mark.slow` tests now run by default.** Removed the
  `-m 'not slow'` default from `[tool.pytest.ini_options].addopts`. The
  24 OCR-fixture tests in `tests/test_fixtures.py` were originally marked
  slow because they invoked EasyOCR (30+ seconds each). The backend has
  since moved to rapidocr-onnxruntime and the whole cohort completes in
  ~7 seconds, so excluding them was costing coverage without saving
  meaningful time. The `slow` marker is retained (description updated)
  so users can still pass `-m 'not slow'` or `-m slow` for ad-hoc
  filtering.

### Added
- **`run_id` + retention plumbing (I1 of publishable-corpus track).**
  Pipeline runs now write outputs to `<output_root>/<run_id>/documents/`
  rather than `<output_root>/documents/`. Run id resolution order:
  `--run-id` CLI flag → `dataset.run_id` in config → auto-generated
  `run-YYYYMMDDTHHMMSSZ` timestamp. `--resume` without a run id picks
  the most-recent existing run dir. Checkpoints follow under
  `<checkpoint_dir>/<run_id>/`.

  New `processing.retention` config block: `policy: rolling | keep_all`
