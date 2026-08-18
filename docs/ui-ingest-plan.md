# Console — Ingest & Output Locations, Composer-Only Dispatch, On-Screen Failures

A delivery plan for the next block of console work, following on from
[`ui-plan.md`](ui-plan.md) (merges 1–11, all landed). Visual system:
[`../DESIGN.md`](../DESIGN.md).

Status: **proposed**. No code written; the merge sequence below is the unit of review.

## 1. Why

The console today treats the object store as **one bucket for everything**. `UISettings`
holds a single `store_uri`; `docker-compose.yml` says outright *"Inputs and outputs share
the s3://womblex bucket"*; and `cloud/worker.py` uses one `RemoteStore` for both
`download_to_dir(job.input_keys, …)` and `upload_glob(…, job.shard_prefix)`. Because there
is no configured *ingest* location, the operator retypes a store-relative `input_prefix`
every time they dispatch — once on the Execution Controls screen and again in the Pipeline
Composer's enqueue section, which duplicates the same form.

Three consequences:

- **Ingest is not configuration.** There is nowhere to say "documents arrive here", so the
  location is retyped per run, in two places, with no validation. Pasting
  `s3://bucket/inbox` into either field silently produces a nonsense glob under the store
  root and a misleading `400 no supported documents under s3://bucket/inbox`.
  (`PathsConfig.input_root` is a `pathlib.Path`, so the composer's own field mangles
  `s3://…` to `s3:/…` too.)
- **Two screens do one job.** `/execute` and the composer's "Enqueue this run" section both
  call the same `enqueueExtraction()`. The composer is the right home; the panel is
  redundant.
- **Failures are hard to read on screen.** The one durable failure reason is
  `womblex_jobs.error`, written as a bare `repr(e)` and shown truncated in a Dashboard
  cell. Worse, `operations/persist.py:write_batch_parquet` only writes manifest rows for
  `status == "completed"` documents — so a document that failed extraction has **no
  manifest row at all**, its error survives only in a log line, and the Corpus Inspector's
  already-built failed-only filter can never show it.

Intended outcome: ingest and output locations are named once per deployment (local folder
or object-store URI, enforced to be different buckets) and editable from the Resources
Console; the composer is the only dispatch surface and re-inputs nothing; and when a run
fails, the screen says why.

## 2. Decisions

| Decision | Resolution |
|---|---|
| Local-folder ingest dispatch | Stays queue-backed. `RemoteStore.from_uri` is fsspec-based and already opens local paths, so `--ingest /data/inbox --store /data/outbox` works with no new runner. The queue-less in-process runner stays deferred ([`ui-plan.md`](ui-plan.md) §4) |
| Ingest scope per run | The whole configured ingest root is the run's input. No prefix field on any screen. The API keeps an *optional* `input_prefix` so `womblex enqueue --input-prefix` retains sub-folder parity |
| Ingest vs output overlap | Hard fail at start-up, on save, and at enqueue. Compose gains a second bucket |
| Editing locations | Env / compose values are **defaults**, not the only source. The Resources Console can *add* a location where none is set and *update* one that is, persisted to a small writable settings file and applied without a restart. Provenance (`from environment` vs `set here`) is shown, with a reset-to-default action |
| Scope of editing | **Buckets and folders only.** The DSN, AWS credentials and Isaacus keys stay read-only and env-provided ([`ui-plan.md`](ui-plan.md) §4: "accepting a credential means storing one") |
| Batching controls | Stay on the composer. Batch size becomes a 10 / 50 / 100 select; the server stays permissive (`ge=1`) so the CLI keeps full range |

## 3. Approach

Ingest becomes a **location setting alongside the output store**, resolved the way
everything else already is — CLI flag, then env var, never in the YAML config (there is no
storage section in `config.py` by design, and there should not be one) — and then, for the
console only, overlaid by an operator-saved value from the Resources Console.

Resolution order for both locations, lowest to highest:

```
CLI flag  →  env var ($WOMBLEX_INGEST_URI / $WOMBLEX_STORE_URI)  →  saved override
```

The saved override is one small JSON file in a writable settings dir. It cannot live in the
output store, because it is the file that *names* the output store — so it needs its own
mount, which is the "one writable config volume" [`ui-plan.md`](ui-plan.md) §4 already
calls for and never built. Absent a settings dir the cards are read-only and say so,
exactly as preset saving already degrades (409 + an on-screen explanation of the flag).

One field per location on screen, two behind it: `UISettings` keeps its existing
`output_root` **XOR** `store_uri` shape, and the scheme of what the operator types decides
which is set (`s3://…`/`gs://…` → `store_uri`; a filesystem path → also `store_uri`, since
`RemoteStore` opens local paths through fsspec and the queue-backed local deployment needs
the same `runs/<run_id>/` layout a bucket has). `output_root` stays what it is today — the
legacy read-only mode over a locally-run `womblex run` output tree — and saving an output
location switches the console out of it. No change to `readers.py`'s local/remote fork.

New surface, total:

- `--ingest` / `$WOMBLEX_INGEST_URI` on `womblex ui`, `womblex enqueue`, `womblex worker`
- `--settings-dir` / `$WOMBLEX_UI_SETTINGS_DIR` on `womblex ui`
- `GET /api/execute/ingest` — one preflight endpoint serving both the composer's
  "N documents ready" line and the Resources Console's ingest card
- `PUT /api/resources/locations` — save / update / clear the two locations

Everything else is rewiring or deletion.

## 4. Delivery sequence

Sized to the 500-changed-line merge cap. Each merge stands alone and leaves the tree green.

| # | Merge | Nature |
|---|---|---|
| 1 | Ingest as a distinct store (library + CLI) | Back-compatible |
| 2 | Console reads the configured ingest (backend) | Additive |
| 3a | Editable ingest / output locations (backend) | Endpoint the screen does not yet call |
| 3b | Editable location cards (frontend) | Frontend |
| 4 | Composer owns dispatch; Execution Controls removed | Frontend, net negative |
| 5 | Run failure reasons on screen | Independent |

3a/3b are split at that seam specifically because the combined diff would run over the cap.

### Merge 1 — Ingest as a distinct store (library + CLI)

**`store/remote.py`** — two helpers beside `is_remote_uri` / `storage_options_from_env`:

- `store_root(uri) -> tuple[str, str]` — normalised `(bucket_or_mount, prefix)`, parsed
  consistently with `fsspec.core.url_to_fs`.
- `assert_distinct_stores(ingest, output)` — raises `ValueError` naming both URIs when they
  resolve to the same bucket, or when either prefix contains the other. The single
  enforcement point; the CLI and `UISettings` both call it, neither re-implements it.

**`cloud/worker.py`**

- `run_worker(dsn, store_uri, config, *, ingest_uri: str | None = None, …)`; `None` keeps
  today's single-store behaviour, so existing invocations are unchanged.
- `_process_job(job, config, store, ingest)` — `ingest.download_to_dir(...)`,
  `store.upload_glob(...)`. Two `RemoteStore`s, nothing else moves.
- Replace `queue.fail(job.id, repr(e))` with a message naming the exception type, its text,
  and the ingest root the worker was reading from. A bucket mismatch then reads as
  `FileNotFoundError: inputs/a.pdf — worker ingest root s3://wrong-bucket` on screen
  instead of a bare repr.

**`cloud/queue.py` + `sql/womblex_jobs.sql`** — one additive, idempotent column so a
mismatch fails fast rather than per-batch: `ALTER TABLE womblex_jobs ADD COLUMN IF NOT
EXISTS ingest_root TEXT` inside `ensure_schema()`; `JobSpec`/`Job` carry it; the worker
refuses a job whose `ingest_root` is set and differs from its own, with both roots in the
error. `NULL` means legacy — use the worker's own root. No data migration, and
`ensure_schema()` already runs before every console enqueue.

**`cli/cloud.py`** — `_resolve_ingest(args)` mirroring `_resolve_store` / `_resolve_dsn`;
`--ingest` on `enqueue` and `worker`; `--input-prefix` on `enqueue` becomes **optional**,
defaulting to the ingest root; `cmd_enqueue` lists from the ingest store and calls
`assert_distinct_stores` before touching the queue.

**`docker-compose.yml` + README** — `WOMBLEX_INGEST_URI: ${WOMBLEX_INGEST_URI:-s3://womblex-inbox}`
in the `x-cloud-env` anchor; `createbuckets` makes `womblex-inbox` as well as `womblex`;
the header comment stops saying inputs and outputs share a bucket. `seed-demo` is
unaffected — it seeds `runs/` in the output bucket.

**Tests** (`tests/test_cloud.py`) — overlap rejection, worker downloading from a second
store, the `ingest_root` mismatch refusal.

### Merge 2 — Console reads the configured ingest (backend, additive)

**`ui/deps.py`** — `UISettings.ingest_uri: str | None`; `resolve_settings` reads
`--ingest` then `$WOMBLEX_INGEST_URI`, and calls `assert_distinct_stores` when both are
present. Threaded through `ui/app.py:create_app` and `cli/ui.py`.

**`ui/execute.py`**

- `ExecutionCapability` gains `has_ingest`, `ingest_uri`, `output_uri`; `can_execute`
  requires ingest. `_guard` gains a `no_ingest` reason (→ 409), ordered after `no_store`.
- `has_store` stops meaning "`is_remote`" and starts meaning "an output location a
  `RemoteStore` can open" — so the queue-backed *local-path* deployment can dispatch
  instead of being told it needs a bucket. `output_root`-only mode still cannot, which is
  correct: that tree has no `runs/` prefix for the queue to publish into.
- `enqueue_extraction(settings, *, input_prefix: str | None = None, …)` — lists from
  `RemoteStore.from_uri(settings.ingest_uri)` at `input_prefix or ""`, publishes to
  `runs/<run_id>/documents` in the output store, and stamps `ingest_root` on each `JobSpec`.
- New `ingest_preflight(settings) -> dict` — `{uri, kind, reachable, document_count,
  sample, error}`, reusing the `list_files` + `SUPPORTED_EXTENSIONS` filter the enqueue
  already performs and the `test_store` try/except shape from `ui/resources.py`.

**`ui/routes/execute.py`** — `EnqueueRequest.input_prefix` becomes optional;
`_REASON_STATUS["no_ingest"] = 409`; add `GET /api/execute/ingest`. Keeping `input_prefix`
optional rather than deleting it is what lets this merge land without touching the frontend.

**`ui/resources.py` / `ui/routes/resources.py`** — a fourth card, `get_ingest_card` +
`test_ingest`, following `get_store_card` / `test_store` exactly
(`POST /api/resources/test/ingest`). Read-only at this stage; merge 3 makes both location
cards editable.

**`ui/composer.py`** — no re-input, at the source: `get_config_schema()` strips the `paths`
section (precedent: presets already strip `dataset`/`paths` as run identity, not shape),
and `validate_config` / `render_yaml` inject the deployment's locations into `paths` before
constructing `WomblexConfig`. When those locations are object-store URIs they cannot be
expressed as `pathlib.Path`, so `render_yaml` emits a header comment naming
`$WOMBLEX_INGEST_URI` / `$WOMBLEX_STORE_URI` instead of writing a mangled `s3:/…` — the
same "storage is env, not YAML" rule the CLI already follows. The composer's form stops
showing ingest/output with no frontend change, since `SchemaForm` renders whatever the
schema says.

**Tests** (`tests/test_ui.py`, `tests/test_ui_resources.py`) — ingest resolution, overlap
rejection at construction, the `no_ingest` guard, enqueue defaulting to the whole ingest
root, preflight shape, schema no longer carrying `paths`, YAML round-trip.

### Merge 3a — Editable ingest / output locations (backend)

The piece that makes env and compose values *defaults* rather than the only source.

**`ui/settings_store.py`** (new, ~90 lines) — read/write one `locations.json`
(`{"ingest_uri": …, "store_uri": …}`, either key absent = no override) in the settings dir.
Same self-contained shape as the preset file helpers in `ui/presets.py`
(`parse_saved_preset` / `serialise_preset_record`): validate on parse, refuse anything but
the two known keys, tolerate a missing file.

**`ui/deps.py`**

- `UISettings.settings_dir: Path | None` and a `settings_writable` property, mirroring the
  existing `presets_dir` / `presets_writable` pair exactly.
- `resolve_settings` gains an overlay step: build the base settings from flags + env, then
  apply the saved override. Saving an output location sets `store_uri` and clears
  `output_root`, so the XOR invariant `__post_init__` enforces still holds.
- `get_settings` re-resolves per request instead of returning a value frozen at start-up,
  mtime-gated so the common case is a `stat()`. This is what makes an edit take effect
  without a restart; `app.state` keeps the base settings, so "reset to default" is just
  deleting the override.

**`ui/resources.py`**

- Each location card reports `value`, `source` (`flag` | `env` | `saved`), and `editable`.
- `save_locations(settings, *, ingest_uri, store_uri)` — validates shape, calls
  `assert_distinct_stores` (overlap enforcement now lives on the save path as well as at
  start-up), writes the file, and returns the refreshed cards plus the reachability
  verdicts from the existing `test_store` / `test_ingest`. Reachability is **reported, not
  required**: a bucket that does not exist yet is a normal state to save.
- Guarded by the same reasons an enqueue is: `--audit-only` refuses (403), no settings dir
  refuses (409). The DSN, AWS credentials and Isaacus key are not accepted by this endpoint
  at all — they stay masked and env-provided.

**`ui/routes/resources.py`** — `PUT /api/resources/locations`; 400 on overlap or a
malformed URI, 403 audit-only, 409 no writable settings dir.

**`cli/ui.py`** — `--settings-dir`, documented like `--presets-dir` (absent ⇒ locations are
read-only, env/flag values still serve).

**`docker-compose.yml`** — a small named volume mounted rw at the settings dir on the `ui`
service. `read_only: true` stays; an explicitly-mounted rw volume is compatible with it.

**Tests** (`tests/test_ui_resources.py`) — overlay precedence (flag < env < saved), the XOR
invariant surviving a saved output location, overlap rejected on save, audit-only 403,
no-settings-dir 409, reset clearing the override, and an edit being visible to the very
next request without an app rebuild.

### Merge 3b — Editable location cards (frontend)

`ui/src/routes/resources/+page.svelte` — the ingest and output cards gain a text field,
Save, Test and Reset-to-default, each showing its provenance chip (`from environment` /
`set here`) and, on the ingest card, the document count from the merge 2 preflight. A 409
disables editing permanently with the `--settings-dir` explanation, reusing the pattern the
composer already has for preset saving. Credential rows stay read-only and masked, with the
existing env-provided label. `ui/src/lib/api.ts` gains `saveLocations()` / `testIngest()`
and a `LocationsRefused` error class alongside `SavePresetRefused`.

### Merge 4 — Composer owns dispatch; Execution Controls removed (frontend)

- **Delete** `ui/src/routes/execute/+page.svelte` and its `NAV_ITEMS` entry in
  `ui/src/lib/nav.ts` (drop the now-unused `Play` icon import).
- **`ui/src/routes/composer/+page.svelte`** — the "Enqueue this run" section loses
  `enqueuePrefix`, the `configString('paths', …)` helper and the "Use composed paths"
  button. In their place: a read-only **Deployment locations** strip (ingest URI, output
  URI, queue) fed by `GET /api/execute/ingest` + `/status`, showing "N documents ready" or
  the preflight's error, and linking to the Resources Console to change them — the composer
  displays the locations, it never edits them.
- **Batching controls stay, and move here.** `batchSize` and `maxAttempts` currently exist
  only on the deleted panel, so the composer gains both. Batch size becomes a **select of
  10 / 50 / 100** (default 50) rather than the free number input it is today — a free
  integer invites a 1 or a 5000 with no feedback, and these three cover the real span from
  "watch a small set drain" to "bulk". `maxAttempts` carries over as the existing small
  number input (min 1, default 3). The server side is unchanged and stays permissive —
  `EnqueueRequest.batch_size` remains `Field(50, ge=1)` so `womblex enqueue --batch-size`
  keeps full range; the increments are a screen affordance, not a new library constraint.
- Run id stays optional (blank mints a fresh timestamped id). The `blocker` banner from the
  deleted page moves across, extended with the `no_ingest` case.
- **`ui/src/lib/api.ts`** — `getIngestPreflight()`; `ExecutionStatus` and `EnqueueRequest`
  types updated.

Mostly deletion; net line count should fall.

### Merge 5 — Run failure reasons on screen

- **`store/output.py` + `operations/persist.py`** — the one library fix behind this.
  `write_results` takes `failures: list[tuple[str, str, str]]` (doc_id, source_path, error)
  and appends manifest-only rows with `status="error"`; `write_batch_parquet` passes the
  `status == "error"` results it currently discards, and writes the shard when there are
  failures even if no document completed. The Corpus Inspector's failed-only filter and
  `DocumentGrid`'s `doc.error` cell then work as built, with no UI change.
- **`ui/src/lib/api.ts`** — route every fetch through the existing `errorDetail()` helper.
  It is already written and already prefers the server's `detail` over a bare status, but
  only `listRuns` and `listPresets` use it; the other ten helpers throw
  `` `GET …: ${resp.status}` `` and discard the reason the server sent.
- **`ui/src/routes/composer/+page.svelte`** — after a successful enqueue, poll the existing
  `GET /api/dashboard?run_id=…` and render a compact run-health strip: failed/total
  batches, the first `job.error` in full, failed-document count, and links to the Dashboard
  and Corpus Inspector. No new endpoint.
- **`ui/src/routes/dashboard/+page.svelte`** — make the truncated error cell expandable so
  the full stored text (up to the queue's 2000-char cap) is readable without a tooltip.
- **Tests** — a failed document produces a manifest row with `status="error"` and its
  message.

## 5. Reuse

Reused rather than rebuilt: `RemoteStore.from_uri` / `list_files` / `download_to_dir` (the
gap was only ever a *second* store handle — pulling from S3 is already implemented),
`SUPPORTED_EXTENSIONS`, `JobQueue.enqueue`'s `(run_id, batch_num)` idempotency,
`is_safe_run_id`, `generate_run_id`, `resources.test_store`'s try/except shape, the
`presets_dir` / `presets_writable` / 409-explains-the-flag pattern (copied wholesale for
settings), `ui/presets.py`'s save-parse-validate file shape, `dashboard.queue_section`,
`api.ts:errorDetail`, `DocumentGrid`'s failed-only filter.

## 6. Verification

Per merge:

```bash
uv run python -m pytest tests/test_cloud.py tests/test_ui.py tests/test_ui_resources.py -v
uv run python -m pytest tests/ -m "not slow and not benchmark"
cd ui && npm run lint && npm run check && npm run build
git diff --stat $(git merge-base HEAD origin/main)..HEAD   # 500-line cap
```

End to end, against the compose stack:

```bash
docker compose --profile local up -d postgres minio createbuckets init
mc cp sample.pdf local/womblex-inbox/     # the NEW inbox bucket
docker compose up -d ui                   # console at :8080
```

Then on screen:

1. **Resources Console, read** — ingest card shows `s3://womblex-inbox` tagged *from
   environment*, Test reports reachable; store card shows `s3://womblex`. Start the console
   with both pointed at the same bucket and confirm it refuses at start-up, naming both
   URIs.
2. **Resources Console, edit** — the core of this change:
   - Change the ingest value to a second bucket, Save. The card re-tags *set here*; the
     next enqueue reads the new bucket with no restart.
   - Reset to default; the card returns to the env value.
   - Try to save an ingest equal to (or nested under) the output location — rejected with
     both URIs named, nothing written.
   - Save a **local folder** as the output location on a console launched with an `s3://`
     store, confirm the run list re-reads from the folder, then reset.
   - Restart the console and confirm the saved override survives.
   - Restart with `--audit-only` (403) and without `--settings-dir` (409); in both cases
     the fields disable and the card explains the flag rather than failing on click.
3. **Pipeline Composer** — no input-prefix field anywhere; the deployment strip shows both
   locations and "N documents ready"; the config form has no `paths` section; Generate YAML
   still validates and the header comment names the env vars. Batch size offers 10/50/100
   and max attempts is present. Enqueue with a blank run id and batch size 10; confirm the
   Dashboard shows the expected batch count for the document count.
4. **Nav** — no Execution Controls entry; `/execute` falls through to the SPA index.
5. **Dashboard** — `docker compose up --scale worker=2 worker`; batches drain.
6. **Failure path** — put a corrupt/zero-byte `.pdf` in the inbox and re-run. Confirm the
   composer's run-health strip names the failure, the Dashboard's error cell expands to the
   full message, and the Corpus Inspector lists the document with `status=error` and its
   error text (the last of these is what merge 5's manifest change unlocks).
7. **Mismatch guard** — start a worker with a different `--ingest` and confirm it refuses
   the job immediately with both roots named, rather than failing per file.
