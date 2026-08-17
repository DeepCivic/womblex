# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries are terse by design; rationale lives in the PR/commit history.

## [Unreleased]

### Changed
- **Console: execution is on by default; `--audit-only` is the opt-out.** Inverts the merge-11 switch — the console can dispatch runs into the queue without a flag, and `womblex ui --audit-only` gives a pure read/inspect console (the old `--allow-execute` is removed). Still queue-only, so dispatch needs both a `--store` and a `--dsn`; the `/api/execute/status` payload renames `allow_execute` → `audit_only` and `ExecutionCapability.can_execute` is now `not audit_only and has_store and has_queue`.

### Added
- **Console Dashboard — screen (`docs/ui-plan.md` merge 8).** Run-scoped, self-refreshing screen over `/api/dashboard`; completes merge 8, leaving only the `ReportIssue` control (7).
  - Queue half (needs a DSN): KPI tiles over exact status counts, total and throughput; `locked_by` worker fleet; stale-job detection naming what `--stale-timeout` recovers; the `womblex_jobs` list with stale rows flagged inline.
  - Checkpoint half (always): per-stage progress from inside the selected run, batch-granular bar + lifetime-average rate.
  - Renders in both deployments — with no queue configured the checkpoint half still shows; read-only (names a stalled job, never requeues it).

## [0.5.1] - 2026-08-17

Minor, additive. Headline: the optional console (`womblex ui`) — a read-only sidecar over pipeline artefacts. Also: Isaacus on Amazon SageMaker (`ISAACUS_SAGEMAKER_ENDPOINTS`), a nullable `elem_order` on `CHUNKS_SCHEMA` (back-filled on read), and deployment extras (`[local]`, `[cloud-ocr]`). No breaking/schema break: `CHUNKS_SCHEMA` gains a nullable column, every other schema byte-identical to `0.4.0`.

### Added
- **Console Pipeline Composer — presets (`DEFAULT-Isaacus`).** Named pre-configured pipelines from a dropdown; a preset is a *partial* `WomblexConfig` (never `dataset`/`paths`) deep-merged onto the form, served as data (`ui/presets.py`, `/api/composer/presets`) and validated at import. `configs/default-isaacus.yaml` ships the same shape for the CLI, pinned against the preset.
- **Console Execution Controls — screen (merge 11).** `ui/src/routes/execute` over `/api/execute/status` + `/enqueue`: a configure-and-run form, and a banner naming the one missing piece (audit-only / no store / no queue) when dispatch is unavailable. `EnqueueRefused` carries the HTTP status (403/409/400) so the client distinguishes the three failure shapes.
- **Console Execution Controls — backend (merge 11).** `GET /api/execute/status` + `POST /api/execute/enqueue` (`ui/execute.py`), the console's first writable-to-a-run surface. Dispatch is always the queue (thin wrapper over `cmd_enqueue`'s batching); `--allow-execute` is the switch, enforced in one `_guard`; needs both a store and a DSN or refuses 403/409.
- **Console Pipeline Composer — config form (merge 9).** `SchemaForm.svelte` renders `WomblexConfig`'s JSON Schema recursively (no hand-typed mirror); Validate and YAML download go to the endpoints (server-rendered, byte-identical to `run --config`). Node `enabled` toggles write the config section's `enabled` via a served `config_section` map.
- **Console Pipeline Composer — stage graph (merge 9).** The pipeline DAG from `/api/composer/graph` plus a per-stage detail panel; nodes laid out by longest path from `extract`, ordering derived from `required_inputs` edges (not hand-coded). Also fixed the frontend CI job (svelte-check `$state(null)` narrowing; a `Map`-in-`$derived` eslint flag).
- **Console Resources Console (merge 10).** `GET /api/resources` (three connection cards) + `POST /test/store` and `/test/queue` live checks. No new detection logic — reuses `is_remote_uri`/`storage_options_from_env`, `unserved_models()`, `dashboard.queue_section`. The `GET` makes no network call; each card's test pays its own timeout.
  - **Credentials do not leave the process:** store keys reported as configured/not, DSN masked. Fixed a full DSN leak — the masker missed libpq's keyword form (`host=… password=…`), now masked along with `?password=` and short secrets.
- **Console Pipeline Composer read API (merge 9).** `/api/composer/graph`, `/schema`, `POST /validate`, `/yaml`. Graph renders `STAGE_CONTRACTS` (edges from `required_inputs` via `PRODUCER_OF`, acyclic, all reachable from `extract`); `/schema` is `model_json_schema()` verbatim; `/validate` + `/yaml` build `WomblexConfig(**raw)` like `load_config`.
  - **Fixed a typo'd-key silent-drop:** Pydantic ignores unknown keys, so `chunkng:` validated clean and vanished on render. `/validate` now reports `unknown_keys`; `/yaml` names dropped keys in a header comment. Warnings not errors (the CLI loads such a file too); walk skips free-form `dict` fields.
- **Console dashboard read API (merge 8).** `GET /api/dashboard` serves queue state + per-stage progress, no new schema. `JobQueue` gains `list_jobs()`, `workers()`, `stale_jobs()` (read-only twin of `requeue_stale`), `throughput()`; `store/checkpoint.py` gains `read_checkpoints()` reading dot-dir checkpoints inside the run (map from `STAGE_CONTRACTS`). Queue optional and orthogonal to run source; an unreachable queue reports `queue_error` not 500 (`JobQueue` gained `connect_timeout`). `run_id` join contained via `is_safe_run_id`.
- **Console report action (merge 7).** `POST /api/runs/{run_id}/feedback` — the console's only write path — files one JSON file per report (no append, no lost update) under a `feedback/<run_id>/` root that is always a *sibling* of runs. `reported_by` from a trusted header/env, advisory. `store/feedback_output.py` owns and contains the join (`is_safe_run_id`, fixing a `..` escape).
- **Console frontend shell (merge 4).** `ui/` SvelteKit workspace, built by a Node stage in `Dockerfile.ui` and served by `create_app` (no JS runtime in the image; `/api/*` still served without the SPA build). Top bar + collapsible side nav over the five domains; theme/density persisted to `localStorage`; tokens per `DESIGN.md`; self-hosted fonts. SPA catch-all serves `index.html` but 404s `/api/`, with traversal containment. Independent Node CI job (lint/check/build).
  - Accessibility measured, not inspected: two `DESIGN.md` rules don't survive its own light theme (lime as active-nav label 1.32:1 there), so lime stays a fill and the label carries state in weight/`--foreground`; recorded in `docs/decisions.md`.
- **Console sidecar image (merge 3).** `Dockerfile.ui` + a `ui` compose service — its own container beside the workers, `womblex[ui,cloud]` (no boto3), hardened `read_only` with a `tmpfs /tmp`. SvelteKit build stage deferred to merge 4 (a stage copying a not-yet-existing dir can't build). Drift-guard tests parse the `ENTRYPOINT` against the real CLI.
- **Console read API skeleton (`womblex ui`).** `[ui]` extra (fastapi+uvicorn), a `womblex ui` command, and `/api/runs` + `/api/runs/{id}/manifest`. Remote reads in scope from the start (a store-backed request stages the manifest to a temp dir and reuses the local reader). Binds one run source; loopback by default.
- **Run index: `describe_run()` + `RemoteStore.list_dirs()`.** Summarises a run as run_id/document-count/stages-present/timestamps from existing artefacts (`STAGE_SUFFIXES` maps stage→sidecar suffix); `list_dirs()` enumerates `runs/<id>/` in object storage. Groundwork for the run selector; the CLI benefits too.
- **Isaacus on Amazon SageMaker (private, air-gapped).** Every Isaacus call routes to SageMaker endpoints in the user's AWS account when `ISAACUS_SAGEMAKER_ENDPOINTS` is set (AWS-signed, no API key). Comma-separated `name[@region][=model|…]` subscriptions, per-model plus universal; undeployed model fails at client construction naming what's served. `[isaacus]` gains `isaacus-sagemaker` (pulls boto3 — why the boto3-free rule stays scoped to `[cloud]`).
- **`elem_order` document-order anchor on table chunks.** `CHUNKS_SCHEMA` gains a nullable `elem_order`, set only for `content_type='table'` chunks, so consumers recover narrative↔table order. Not a coordinate-space change (offsets untouched, no re-enrichment); `read_chunks` back-fills nulls for older shards.
- **Deployment-shaped install extras.** `[local]` (empty — base install *is* local CPU) and `[cloud-ocr]` (alias of `[bedrock]`) join `[cloud]`/`[isaacus]`/`[bedrock]`. No package versions changed.
- **README installation matrix** (deployment → extra → what it adds) plus a backend-selection table.

### Changed
- `[cloud]` documented as explicitly *not* implying `[cloud-ocr]`: s3fs reaches S3 via aiobotocore→botocore, so object-storage staging needs no boto3; boto3 stays confined to `ingest/llm_ocr.py`.

## [0.4.0] - 2026-08-06

Minor, additive (the `run-stage` command + stage contracts). One observable change: the `money` op's narrative output changes for text it already read — space-grouped thousands (`$10 000`) were stored wrong by 10³ and are now correct; `$US`-marked/worded/restated amounts now resolve. Re-run `womblex money` over any 0.3.0 shard dir (regenerated in place, nothing depends on it). No parquet schema changed; `money_spans.evidence` carries one new value (`p11`).

### Added
- **Money: financial values in narrative structure** (`docs/money-extraction.md`). Two values corrected, three recovered, no span lost. `process/money.py` split (number reading → `money_numbers.py`); re-exported, no import site changed.
  - Worded amounts (`two million dollars`, `fifty cents`; `money_words.py`), currency word required; declines ranges/years/unit-declarations.
  - Space-grouped thousands (`$10 000`, incl. NBSP/thin space); a group is exactly three digits.
  - `$US`/`$A` symbol order (`$US655.5m`); fixes the metre-pattern blocker that also hit `US$655.5m`.
  - Restatement `one million dollars ($1,000,000)` no longer double-counted or read negative; equal bracketed digit amounts left alone.
  - Signs/brackets (true minus vs en-dash range, accounting `$(1,234.50)`, `50¢`); declines a second dotted group (`$3.219.3m`).
- **`womblex run-stage` — remote per-batch shard-stage runner.** Runs a downstream `*_shards()` stage directly against object storage (generalises `finalize`; no `*_shards()` signature changed). Covers normalise/spellfix/chunk/money/enrich/embed/link/pii/graph-refresh/quality.
  - Declarative stage contracts (`cloud/stage_contracts.py`); conditional inputs/outputs resolved from config, not stage name.
  - Every declared output verified before any is uploaded; skip only when all present; idempotent; exits 1 on nothing-to-do.
  - `graph-refresh` modelled as in-place mutator (never skipped); `quality` run-scoped (single pass for cross-batch dedup); Isaacus-needing stages fail non-zero.
  - Stage *ordering* is the caller's; a base with absent required inputs is not-ready, all-absent is an ordering error (exit 1).

## [0.3.0] - 2026-07-29

Minor, additive (the `money` op + two sidecars; the shared table-grid algorithm). Two observable changes: a scanned page with a clean table now emits a `kind='table'` element (was a `[TABLE]` placeholder), and the unreachable `ImageExtractor` was removed. No extraction schema changed.

### Added
- **`money` annotation op** (`womblex money --shards`). Recovers amounts to `*.money_spans.parquet` + `*.money_columns.parquet`; offline, no ordering dep, never rewrites text. Exact `decimal128(38,4)`; three loci in two coordinate spaces (`narrative`/`table_cell`/`sheet_cell`), never mixed. First real run: all 42 marked ANAO narrative amounts recovered, `Approved Budget $m` reconciles three ways.
  - Self-evidencing (`money.py`): symbol/ISO/word beside the number; magnitude expansion, range linking, gated accounting negatives; AU false-positive classes rejected.
  - Column-evidenced (`money_columns.py`): number format or money-header + numeric cells; whole-word vetoes; header supplies scale/currency; continuation-row header folding.
  - Fixes: `'000` no longer matched inside any number; tier-3 ISO codes context-gated (`TOP 10`≠paʻanga); count columns `(#)` not read as money; own-currency headers survive a veto term.
- **Table-cell reconstruction on OCR'd pages (#17).** Cells reconstructed inside a layout-detected table rect on OCR pages; one shared grid algorithm, precision-gated (refuse over partial).
  - A0 — plumbing: OCR regions + render dims passed to the layout pass; region↔rect intersection + a coordinate-space guard; region-based engines only (LLM/VLM deferred).
  - A1 — shared `ingest/table_grid.py` (point-space tolerances parameterised for pixel callers) + `ingest/ocr_tables.py` `reconstruct_table` (returns `None` below gates).
  - A2/A3 — OCR-PDF path emits `kind="table"` with cells; narrative rebuilt from regions *outside* the rect (no double-count); deskewed pages refuse rather than mis-bin.
  - A4 — images already reached reconstruction via `extract_text`; the dead `ImageExtractor` deleted (breaking for direct importers; nothing internal used it). `get_extractor` signature now `(profile)`.
  - B0/B2 — GT aggregation fix (recall 25%→50% by artefact); `MIN_ROW_FILL_RATIO=0.75` density gate closes three false positives.
  - B3/B4/B5 — rendered-table benchmark (`tests/test_table_benchmark.py`) + `EXTRACTION.md`/`evaluation.md §2b` wiring; sanity asserts become build-failing gates (exact row/col counts + `MIN_CELL_MATCH`, false-table count == 0). #17 round 1 complete.

### Fixed
- **Declined continental number no longer leaks its decimal tail.** `1.234,56 EUR` came back as `56 EUR` (wrong by 10³); ambiguous numeric runs are now blocked whole. Only suffix-pattern forms leaked; international mode unaffected.
- **CI runs type-check + tests again.** Unpinned `ruff` resolved 0.16.0 whose new defaults reported 297 errors, skipping mypy/pytest (the `money` op merged untested). `ruff` bounded `>=0.16,<0.17`; tree clean under its defaults (`BLE001`/`S110`/`S112` suppressed at isolation boundaries).
- **Isaacus test suites run in CI.** CI omitted the `isaacus` extra, skipping 66 no-key tests; now installed. Unmasked and fixed a real typing error in `process/chunker.py`.
- **`mypy` passes with `openpyxl` installed** (missing `ignore_missing_imports` entry).
- **Spreadsheet extraction preserves `number_format` + numeric `value_type`.** A read-only openpyxl pass supplies both (pandas discarded them); values untouched. Matters because a register's money column is often identifiable *only* from its format (`$#,##0.00` on bare `50000`).

## [0.2.0] - 2026-07-19

### Added
- **Pre-extracted records ingest (`ingest/records.py`).** Turns clean text records (JSONL; Open Australian Legal Corpus) into the standard shard layout so the enrich→chunk→embed→graph pipeline runs over them; content-addressed `source_hash`, corpus-agnostic `RecordFieldMapping`, provenance sidecar.
- **Token-budget request packer (`utils/token_packer.py`).** Packs Isaacus requests by exact local kanon-2 token counts (limits bind on tokens, not request count); over-budget item solo, over-ceiling doc split on blank lines. Cached offline `TokenCounter`.
- **Enrichment — token-aware batching + long-doc split (`enrich_stage.py`).** Packer-driven requests (8× fewer for small docs); over-`split_ceiling` docs split and offset-merged; honours `Retry-After`. New `EnrichmentConfig` knobs (`tokenizer`, `max_texts_per_request`, `token_budget`, `split_ceiling`).
- **Graph-edge refresh stage (`analyse/graph_refresh.py`, `womblex graph-refresh`).** Offline rebuild of mention→chunk edges from entity+chunk sidecars (AI chunking runs after enrichment, so the enrich-time graph lacks them); idempotent.
- **Offline kanon-2 tokenizer** vendored under `_models/kanon-2-tokenizer`, resolved locally (no HF round-trip).
- **Distributed / cloud execution (`womblex[cloud]`).** Optional scale-out; local CPU default unchanged.
  - `store/remote.py` — fsspec stage-in/out object-storage adapter, confining remote knowledge (Path-based stages untouched).
  - `cloud/queue.py` — Postgres `FOR UPDATE SKIP LOCKED` job queue; row `status` *is* the distributed checkpoint (idempotent re-enqueue, retry, stale requeue).
  - `cloud/worker.py` + CLI (`enqueue`/`worker`/`jobs`/`finalize`); ordinary shard layout, so downstream consumes a distributed run like a local one. `process_batch` is the single shared body behind `womblex run`.
- **Container image + compose stack.** `Dockerfile` + `docker-compose.yml` bundling Postgres, MinIO and scalable workers (`--scale worker=N`).
- **CI security job.** Semgrep SAST (Python + OWASP, blocking) + `pip-audit` (informational); test job installs the `cloud` extra.
- **ABN Lookup bulk extract ingest (`ingest/abn_bulk.py`).** Stream-parses the ABR XML (~6 GB) at constant memory to `<stem>.parquet` (records) + `<stem>_names.parquet`; verbatim strings, provenance in parquet metadata, per-file failure isolation. `womblex ingest-abn`; shared MD5 helper → `utils/checksum.py`.
- **Spreadsheet preamble/header detection.** Reads `header=None` and `split_preamble` finds the header as the row starting the longest run of table-consistent rows below it; preamble kept verbatim on `sheet_meta`. Fixes fabricated `Unnamed: N` columns and ragged-CSV failures; header-first/narrow sheets unaffected.
- **Run-level document manifest.** `womblex run` consolidates per-batch manifests into `<run>/manifest.parquet` (source_hash → doc_id/filename/method/counts/status); `womblex manifest --shards` regenerates it.
- **Shippable enrichment graph.** `enrich_shards` writes `*.graph_edges.parquet` (Kanon-2 document graph flattened to `GRAPH_EDGE_SCHEMA`, `document_id`=source_hash), mapping in chunk edges when chunks exist; resume re-enriches a batch missing it.
- **`womblex chunk --shards` + `--config` combinable** (were mutually exclusive, dead-ending per-stage AI chunking); `--shards`+`--config` sources chunking settings from YAML.
- **Single-enrichment reuse for AI chunking.** With AI chunking + enrich both on, enrich persists the raw ILGS Document to `*.enrichment_doc.parquet` and chunk reuses it (no double Kanon-2 call), gated by a byte-identity guard (mismatch → self-enrich, offsets never desync). Requires enrich before chunk.
- **AI chunking pass-through (semchunk 4).** `ChunkingConfig.chunking_model` (default null) follows the enricher's structure spans; opt-in, forwards params straight to `chunkerify`. Bumps `semchunk>=4.0`.
- **`spellfix` stage — dictionary-gated OCR glyph repair (`womblex spellfix`).** Opt-in; rewrites a token only on three gates (out-of-dict trigger, single-char in-dict candidate, unique) against bundled en_AU Hunspell (`spylls`). Element-layer overlay + audit; raw untouched. Tier A digit→letter default, Tier B opt-in.
- **Composable element-text overlays via one `processing.text_source`** (`elements`|`normalised`|`spellfix`), applied before reassembly at both chunk and enrich sites (one knob — enricher input and chunk source must match).
- **Enricher `overflow_strategy` (default `auto`)** passed to `enrichments.create` (vs upstream null, which errored >16k tokens); offsets still index the full source. Fixes long FOI bundles.
- **`score --text-source={elements,normalised}`** — measure how normalisation changes CER against the same GT.
- **Benchmark: ACT-ECI labelled-pages raw-vs-normalised CER** (`-m benchmark`); regression guard asserts normalisation never worsens CER. Cohort 7→19 pages.
- **`quality` stage — chunk-quality sidecar (`womblex quality`).** `*.chunk_quality.parquet` with ML-readiness flags + duplicate cluster ids (self-contained MinHash+LSH, no datasketch); annotation only, single global pass; config-driven `boilerplate_patterns`.
- **`normalise` — `unicode_hygiene` transform** (default on): folds unicode whitespace to ASCII, strips zero-width/BOM/control; smart quotes and dashes preserved.
- **Entity-link sidecars: `womblex enrich` + `womblex link` per-stage CLIs.** `enrich` reassembles narrative and writes entity/meta sidecars (per-doc failure isolation, and does *not* checkpoint an errored doc so resume retries); `link` resolves candidates to a reference register (`*.entity_links.parquet`). Generic by design (`entity_type` discriminator, stdlib-`difflib` matcher, config-declared register roles). New `isaacus` extra. Artemis smoke 16/17.
- **`womblex embed --shards` — chunk embeddings (Kanon-2 embedder).** `*.embeddings.parquet` (one vector/chunk, 128-text batching, 429 retry, task-aware); substrate for search/clustering + a no-extraction attribution backstop.
- **`womblex pii --shards` — graph-driven detection + `<PERSON_n>` masking.** `*.pii_spans.parquet` (audit) + masked `*.clean_text.parquet`; Kanon-2 graph is the primary source, spans mapped onto chunks via `start_char`. Masking terminal (after enrich+embed, never rewriting raw chunks); regex/cosine backstop opt-in.
- **`womblex redact --shards` per-stage CLI** (dual-mode like `chunk`): `--shards --pdfs` writes `*.redactions.parquet` (detection rasterises pages, so `--pdfs` required); `--config` runs the E2E path.
- **Shard integrity scan on `--resume` (E1).** `reconcile_checkpoint_with_shards` walks each batch's four sidecars (present/non-empty/readable + manifest count sums), drops+`.corrupt`-renames failures for re-extraction; unreadable manifests logged loudly. Default on (`--no-verify-resume`).
- **`womblex verify-shards` CLI (E2).** Audits a run/shard dir for corruption + cross-batch consistency; `--compare-to` diffs two runs, `--input-dir` surfaces source-vs-manifest drift; exits 2 on corruption.
- **`run_id` + retention plumbing (I1).** Runs write to `<output_root>/<run_id>/documents/`; id resolves `--run-id` → `dataset.run_id` → auto `run-<ts>`. New `processing.retention` block (`rolling`|`keep_all`).
- **Document-layout YOLO model (DocLayNet), K7(b).** New default `yolo11n_doc_layout.pt` (MIT) replaces COCO `yolov8n.pt`; taxonomy auto-detected from class names, 11 classes mapped to `ElementKind`. Closes the 1,587-element `figure` mis-classification; per-taxonomy imgsz default.
- **`footnote` ElementKind** (added to `TEXT_KINDS`; DocLayNet `Footnote` producer; downstream stages pick it up automatically).
- **OCR form-pair bboxes, K2′.** `_extract_form_pairs_from_regions` produces `FormField`s with real positions from per-region OCR; legacy line-based path is the LLM-OCR fallback. Closes silent-zero-bbox on 4,184/5,183 form elements.

### Changed
- **`operations.py` split into an `operations/` package** (over the 750-line cap); flat import surface preserved by re-exports, behaviour-neutral.
- **Resume-integrity self-heal generalised** — `reconcile_stage_checkpoint_with_shards` now backs chunk/enrich/link/embed identically (+ `--no-verify-resume`).
- **SemChunk wrapper audit (I5).** Exposed every `chunkerify`/`__call__` param; removed the dead `ChunkingConfig.batch` flag; widened `chunk_size` to `int | None` (auto-derive), default 480 unchanged. Byte-identical output.
- **`process/chunker.py` collapsed against semchunk v3+.** Single `chunk_batch` entry point (semchunk already batches/parallelises); `TextChunk` gains nullable `page_start`/`page_end`; new reassembly/collection helpers shared by both invocation paths.
- **`operations.run_chunking` rewired through `chunk_batch`**, building `ChunkInput` from `extraction.elements` (canonical). Behaviour change: in-memory PII/redact mutations to `pages[i].text` no longer flow to chunks under `womblex run`.
- **`annotate-redactions` is now a deprecated alias** for `redact --shards --pdfs` (positional surface preserved, byte-identical output).
- **`@pytest.mark.slow` tests run by default** (backend moved to rapidocr; cohort now ~7s). Marker retained for `-m 'not slow'`.
- **Manifest schema gains `doc_id` column** (removes the implicit `stem==doc_id` coincidence); `read_manifest` back-compat derives it from `filename` for older manifests. Parser-version bump deferred (additive).

### Fixed
- **OpenCV 5 compatibility in skew detection.** `HoughLinesP` segments unpacked as `line[0]` crashed every OCR extraction under OpenCV 5's `(N,4)` layout; reshaped to `(-1,4)`, accepting both. Direct unit tests pin both shapes.
- **`mypy` no longer pins `python_version="3.11"`** (broke the 3.12 leg on numpy PEP 695 stubs); each leg checks at its own version.
- **Register manifest covers `ingest-geo`, roles from footer metadata.** `cmd_ingest_geo` never wrote a manifest and the namespace said `geo` vs `geospatial.*`; namespace now taken from the `<ns>.source_file` key, ABN roles from an `abn.role` footer key (not filename suffix). Re-run `ingest-abn` to restore the role distinction. Constant renamed `REGISTER_MANIFEST_FILENAME`.
- **`RemoteStore` no longer leaks s3fs options into non-S3 backends** — `storage_options_from_env()` now takes the URI and returns options only for `s3://`. Also: `enqueue` batch-size fallback reads `ProcessingConfig()`; worker derives its upload glob from `BatchOutcome.shard_path`.
- **Full-page scans no longer dropped from chunking as `figure` (K9-fig).** The dominant-region fallback tagged a whole page `figure` (∉ `TEXT_KINDS`), silently losing it; `_ocr_region_block_type` promotes to `paragraph` at ≥5 words. ACT-ECI: `figure` 1,200→154, all 16 zero-chunk docs now chunk.
