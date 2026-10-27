# Service API — Womblex

Womblex serves other software in two modes:

- **Data contract.** A consumer reads the versioned Parquet a run writes, or
  imports the declared Python API. Both are specified in
  [contract.md](contract.md); the bundle that carries a run to another system is
  in [egress.md](egress.md).
- **Callable service.** A consumer submits work to `womblex serve` and reads
  results back over the authenticated `/v1` HTTP API described here.

## Deployment

`womblex serve` needs the `[api]` extra (fastapi, uvicorn, python-multipart)
and names it when missing. It binds to `127.0.0.1:8081` by default.

| Flag | Env | Purpose |
|---|---|---|
| `--store` | `WOMBLEX_STORE_URI` | Object store holding run output |
| `--dsn` | `WOMBLEX_DB_DSN` / `DATABASE_URL` | DBOS system database (Postgres; a local SQLite file when omitted) |
| `--ingest` | `WOMBLEX_INGEST_URI` | Where uploads land and runs are enqueued from; without it, uploads and run submission answer 503 |
| `--max-upload-mb` | — | Largest upload request, default 256 |
| `--host`, `--port` | — | Bind address |
| `--insecure-no-auth` | — | Every request is an admin. Development only |

In compose, the `api` service (profile `api`, port 8081) runs on the worker
image with the client registry mounted read-only. See
[deployment-images.md](deployment-images.md).

## Authentication and scopes

Callers authenticate with static service tokens (`Authorization: Bearer …`).
This is private-network authentication for a trusted subsystem, not a public
identity system.

- The registry is a YAML file named by `WOMBLEX_API_CLIENTS`. Each entry holds
  `client_id`, `token_sha256` and `scopes`. Only hashes are stored.
- `womblex api-token --client X` mints a token (shown once) and prints its
  registry entry.
- Tokens are compared in constant time (`hmac.compare_digest`).
- `serve` refuses to start with an empty registry unless given
  `--insecure-no-auth`.

| Scope | Grants |
|---|---|
| `submit` | Upload documents and submit runs |
| `read` | Run listings, status, manifest, files, metrics, masked text |
| `read_raw` | Text layers whose contract sensitivity is `raw` |
| `admin` | Every scope, and every run regardless of owner |

## Run ownership

Runs are known to the run board (the `owner` attribute on each DBOS workflow) and scoped to the client
that submitted them.

- A run owned by another client answers 404, as an unknown run does.
- `admin` sees every run. An admin's submitted run carries no owner.
- Runs dispatched from the CLI or a workflow carry no owner, so through the API
  they are visible to `admin` only.
- The routes that read a run's shards from the store (`/stages`, `/audit`, `/detail`, `/logs`, `/progress`) also
  answer an `admin` for a run only the store holds: one `womblex run` wrote, or one seeded.
- A run never has mixed ownership: the queue refuses an enqueue that names a
  different owner (`RunOwnedError`).

## Endpoints

| Method | Path | Scope | Behaviour |
|---|---|---|---|
| GET | `/v1/health` | none | Liveness |
| GET | `/v1/ready` | none | Store and queue reachability; 503 when either fails |
| GET | `/v1/ingest?input_prefix=` | submit | What a submission naming the prefix would run: document count and a sample, or a nested layout's refusal with each subdirectory's prefix and count; confined to the caller's tree like a submission |
| POST | `/v1/uploads` | submit | Multipart `files` to `<ingest>/<client_id>/<upload_id>/`; returns that folder as `input_prefix` |
| POST | `/v1/runs` | submit | Enqueue extraction plus the downstream stages the config enables; returns `run_id`, `document_count`, `batch_count` and `stages` |
| GET | `/v1/runs` | read | The caller's runs with a status rollup, latest activity first, at most 1000 (`RUN_LIMIT`) |
| GET | `/v1/runs/{id}` | read | One run's state, per-status counts and status sentence (`summary`) |
| GET | `/v1/runs/{id}/manifest` | read | The consolidated run manifest |
| GET | `/v1/runs/{id}/files` | read | Object keys, row counts and footer `contract_version` / `sensitivity` |
| GET | `/v1/runs/{id}/documents/{hash}/text` | read / read_raw | One document's text in document order, by `layer` |
| GET | `/v1/runs/{id}/metrics` | read | Queue stats, workers and throughput as JSON |
| GET | `/v1/store/runs` | admin | Every run the store holds, with document counts and the stages present, whether or not the queue knows it |
| GET | `/v1/runs/{id}/stages/{stage}` | read | The documents with a row in that stage's sidecar; 400 for a stage with none of its own |
| GET | `/v1/runs/{id}/documents/{hash}/detail` | read_raw | The document's chunks with its entities, PII spans (with offsets in their chunk), narrative money and quality rows |
| GET | `/v1/runs/{id}/logs` | read | The run's batch logs (name, size, modified), newest first; empty for a run older than published logs |
| GET | `/v1/runs/{id}/logs/{name}` | read | One log as `text/plain` (`?download=true` as a file); a name that is not a log's and an absent log give one 404 listing the logs there are |
| GET | `/v1/runs/{id}/progress` | read | Each stage's checkpoint, read from the run itself, in pipeline order |
| GET | `/v1/queue` | read | The caller's runs on the queue in one read: counts, jobs (`job_limit`), workers, jobs stale past `stale_after`, throughput over `window_seconds`; `run_id` narrows it |
| GET | `/v1/runs/{id}/audit` | read | `verify-shards`' structural audit of the run's extraction shards |
| GET | `/v1/resources` | admin | The store, ingest location, queue and Isaacus deployment, credentials masked; no network call |
| POST | `/v1/resources/test/{store,ingest,queue}` | admin | A live check of one: `{target, reachable, error}` |
| GET | `/v1/stages` | read | The pipeline graph: a node per stage (and `extract`), an edge per required input |
| GET | `/v1/config/schema` | read | `WomblexConfig`'s JSON Schema, without the deployment's `paths` |
| POST | `/v1/config/validate` | read | `{valid, errors}` for a posted config under this deployment's `paths` |
| GET | `/v1/presets` | read | Built-in presets, then saved ones (a saved name shadows a built-in) |
| GET | `/v1/presets/{name}` | read | One preset; 404 when none |
| POST | `/v1/presets` | admin | Save or replace a preset under the store's `presets/`; 400 for an unsafe name or a config that would not load |
| DELETE | `/v1/presets/{name}` | admin | Delete a saved preset; 404 when none, and for a built-in |
| GET | `/v1/collections` | read | The collections the caller's stored workflows feed (admin: all, and the store's collection runs no stored workflow feeds any more, with no workflows), with the workflows that feed each |
| GET | `/v1/collections/{name}` | read | One collection: its workflows, its document count and the workflows its run's files record; 404 for another owner's |
| POST | `/v1/collections/{name}/add` | submit | Fire the stored workflow now (`womblex add-to-collection`); 202 with the firing's board id. Body `{workflow}` names it when several feed the collection (409 otherwise); 409 `{code, message}` for a firing that would be refused, before any row |
| POST | `/v1/collections/{name}/rerun` | submit | Dispatch the stored workflow's steps over the collection (`womblex rerun-collection`); 202 with the stages and the units newly dispatched; refused as `add` is |
| POST | `/v1/hooks/{workflow}/{hook}` | the hook's HMAC | An inbound event (Requirement 35): a signed JSON body with `event` and `event_id`; 202 `{status: accepted, …}`, or `{status: refused, code, message}` with 401 `bad_signature`, 404 `unknown_hook`, 400 `bad_event`, 403 `event_not_declared`, 413 `body_too_large` (over 1 MiB), or 409 for a workflow refusal or `unknown_delivery`. A resend returns the first outcome; one whose first handling stored none is handled again |
| GET | `/v1/workflows/{id}/inbound-events` | read | A stored workflow's inbound events, newest first, with each outcome (`limit`); another owner's workflow is 404 |
| POST | `/v1/config/yaml` | read | The posted config as YAML `womblex run --config` reads; 422 with the errors when invalid |

`POST /v1/runs`, `POST /v1/uploads` and `POST /v1/presets` answer 201; the collection actions answer 202. Pydantic models generate the OpenAPI spec; a test
pins its operation and model surface so breaking changes show in review.

### Uploads

- Only each file's bare name is kept, so a client path cannot place a file
  elsewhere or nest it.
- An unusable, repeated or unsupported name (judged by `select_supported`) is
  400 and nothing is written.
- A request over `--max-upload-mb` is 413.

### Run submission

- Body: `input_prefix`, optional `preset` **or** `config` (not both),
  `batch_size` (default 50).
- A non-admin caller's `input_prefix` must be its `<client_id>` folder or under
  it (403 otherwise).
- The config is validated before any queue row is written (400). It governs
  every stage of the run, extraction included: the run carries it, and each
  worker that claims its work runs under it, supplying only its own store,
  paths, credentials and model locations. A config enabling no stage is an
  extraction-only run; with neither `preset` nor `config` the run carries the
  default config.
- A run no live worker serves (its extraction, or a stage it enables, under
  its config) is refused with 409 before any queue row: start a worker whose
  config names the same models.
- The run id is minted server-side: the timestamp id plus a random suffix.
- A prefix with documents in nested folders is refused, as for every run.

### Run state

`failed` if any job failed; `running` while jobs run or some are pending
beside some done; `pending` while all are pending; `cancelled` when a job
was cancelled; otherwise `done`. `orphaned` is true for a run whose firing
queued extraction and failed before its stages were dispatched. `summary` is the
sentence `womblex jobs --run-id` shows, built from the exact
per-stage counts, e.g. `Run r1 is running (extract: 4 done; chunk: 1 done, 2 pending).`

### Document text layers

| `layer` | Source | Sensitivity | Scope |
|---|---|---|---|
| `masked` (default) | `*.clean_text.parquet` | masked | read |
| `chunks` | `*.chunks.parquet` | raw | read_raw |
| `elements` | `*.elements.parquet` | raw | read_raw |

A document with no rows in the layer is 404.

## Out of scope

Registering a callback or webhook through the API (a caller polls; a run a
workflow starts sends its events to the workflow's webhook, [workflows.md](workflows.md)),
run cancellation, per-client quotas, Prometheus metrics.

## Verification

End to end on the local compose stack:

1. `womblex serve`
2. `curl -H "Authorization: Bearer …" -F files=@fixture.pdf …/v1/uploads`
3. `POST /v1/runs` with the returned `input_prefix`
4. a worker drains the queue
5. `GET /v1/runs/{id}` reaches `done`
6. `GET …/documents/{hash}/text` returns masked text
7. `/files` shows `contract_version` and `sensitivity`
