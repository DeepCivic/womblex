# Service API — Womblex

Womblex serves other software in two modes:

- **Data contract.** A consumer reads the versioned Parquet a run writes, or
  imports the declared Python API. Both are specified in
  [contract.md](contract.md); the bundle that carries a run to another system is
  in [egress.md](egress.md).
- **Callable service.** A consumer submits work to `womblex serve` and reads
  results back over the authenticated `/v1` HTTP API described here.

The operator console (`womblex ui`) is an admin and debugging utility, not an
integration surface. It has no authentication and binds to loopback.

## Deployment

`womblex serve` needs the `[api]` extra (fastapi, uvicorn, python-multipart)
and names it when missing. It binds to `127.0.0.1:8081` by default.

| Flag | Env | Purpose |
|---|---|---|
| `--store` | `WOMBLEX_STORE_URI` | Object store holding run output |
| `--dsn` | `WOMBLEX_DB_DSN` / `DATABASE_URL` | Postgres job queue |
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

Runs are known to the job queue (`womblex_jobs.owner`) and scoped to the client
that submitted them.

- A run owned by another client answers 404, as an unknown run does.
- `admin` sees every run. An admin's submitted run carries no owner.
- Runs dispatched from the CLI or console carry no owner, so through the API
  they are visible to `admin` only.
- A run never has mixed ownership: the queue refuses an enqueue that names a
  different owner (`RunOwnedError`).

## Endpoints

| Method | Path | Scope | Behaviour |
|---|---|---|---|
| GET | `/v1/health` | none | Liveness |
| GET | `/v1/ready` | none | Store and queue reachability; 503 when either fails |
| POST | `/v1/uploads` | submit | Multipart `files` to `<ingest>/<client_id>/<upload_id>/`; returns that folder as `input_prefix` |
| POST | `/v1/runs` | submit | Enqueue extraction plus the downstream stages the config enables; returns `run_id`, `document_count`, `batch_count` and `stages` |
| GET | `/v1/runs` | read | The caller's runs with a status rollup, latest activity first, at most 1000 (`RUN_LIMIT`) |
| GET | `/v1/runs/{id}` | read | One run's state and per-status counts |
| GET | `/v1/runs/{id}/manifest` | read | The consolidated run manifest |
| GET | `/v1/runs/{id}/files` | read | Object keys, row counts and footer `contract_version` / `sensitivity` |
| GET | `/v1/runs/{id}/documents/{hash}/text` | read / read_raw | One document's text in document order, by `layer` |
| GET | `/v1/runs/{id}/metrics` | read | Queue stats, workers and throughput as JSON |

Both POST routes answer 201. Pydantic models generate the OpenAPI spec; a test
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
- The config is validated before any queue row is written (400). It selects
  stages only; workers run them under their own config, as with
  `womblex enqueue-stages`. A config enabling no stage is an extraction-only
  run.
- The run id is minted server-side: the timestamp id plus a random suffix.
- A prefix with documents in nested folders is refused, as for every run.

### Run state

`failed` if any job failed; `running` while jobs run or some are pending
beside some done; `pending` while all are pending; otherwise `done`.

### Document text layers

| `layer` | Source | Sensitivity | Scope |
|---|---|---|---|
| `masked` (default) | `*.clean_text.parquet` | masked | read |
| `chunks` | `*.chunks.parquet` | raw | read_raw |
| `elements` | `*.elements.parquet` | raw | read_raw |

A document with no rows in the layer is 404.

## Out of scope

Callbacks and webhooks (consumers poll), run cancellation, per-client quotas,
Prometheus metrics.

## Verification

End to end on the local compose stack:

1. `womblex serve`
2. `curl -H "Authorization: Bearer …" -F files=@fixture.pdf …/v1/uploads`
3. `POST /v1/runs` with the returned `input_prefix`
4. a worker drains the queue
5. `GET /v1/runs/{id}` reaches `done`
6. `GET …/documents/{hash}/text` returns masked text
7. `/files` shows `contract_version` and `sensitivity`
