# Workflows

A workflow is a named recipe in a YAML file: a preset, optional config changes,
the downstream steps to run in order, and optionally cron triggers. The file is
the master copy: every workflow command takes its path, and creating or editing
one needs no code. Only `schedule-workflow` stores a copy, in the database DBOS's
own tables live in (Postgres for a fleet, a local SQLite file otherwise), because
its firings read it. The "why" is in [decisions-operations.md](decisions-operations.md)
("A workflow is a file").

## Writing one

[`configs/workflows/nightly.yaml`](../configs/workflows/nightly.yaml) is an example,
with the commands that use it in its comments.

Every command checks the file first and refuses what could never run: an unknown
key, an unknown stage, a stage listed twice, a bad cron expression or timezone,
or a config that does not validate. A stored workflow's owner is the API caller
that stored it (none, admin, from the CLI); the runs it starts are theirs, and a CLI-stored
workflow's are admin-only, like CLI runs. Its digest (`sha256:…`) is over the parsed spec,
so formatting and comments do not change it, and is stamped on every sidecar
its steps write.

| Field | Meaning |
|---|---|
| `name` | The workflow's id: its schedules, its stored copy and its runs are named for it (a `collection` names the run instead). |
| `preset` | The base config, a built-in preset or one saved under `--presets-dir`. `dataset` and `paths` are never taken from the workflow. |
| `config` | Overlay on the preset's config: mappings merge, anything else replaces, `null` drops a key to its default. |
| `input_binding` | `store` (the input store a firing lists and extracts from; a scheduled workflow must name it, spelt exactly as the workers' `--ingest`, or `--store` where they have none, since firings are routed by that string), `ingest_prefix` (the documents under it a firing extracts) and `batch_size`. Extraction is not a step. |
| `steps` | Downstream stages, each at most once, run in the order listed (`dispatch-workflow` needs pipeline order). |
| `collection` | Optional. The run the workflow's firings append to, in place of the workflow's name: one safe token. Not part of the digest. |
| `triggers` | Cron expressions scheduled by `schedule-workflow`. |
| `destinations` | Where the run is delivered after its last step. `object_store`: `name`, `uri` (a local path or a URI a store can open, checked when the file is read), `include_sources` (default `true`) and optional `credentials` (`key_env` and `secret_env`, the environment variables holding an S3 key and secret, never the values). `postgres`: `name`, `dsn_env` (the environment variable holding the connection string), `table` (default `womblex_chunks`) and `text_layer` (`masked`, the default, or `raw`). `webhook`: `name`, `url` and `secret_env` (the environment variable holding the signing key). |

## Delivery

After the last step, whether or not a step failed, the run is delivered to each
destination: an `object_store` gets the run's egress bundle ([egress.md](egress.md))
under `<uri>/<run id>/`. A `postgres` destination gets one row per embedded chunk
with its embedding (a pgvector column; `pip install womblex[pgvector]`): the masked
`clean_text` by default, the raw chunk text only with `text_layer: raw`, each row
naming its layer and, when masked, its `mask_status`. A chunk the layer holds no
text for is left out and counted, never sent from the other layer. A Postgres
destination needs the `embed` step, and masked text the `pii` step; a workflow
without them is refused. The table is created if absent, and the `vector` extension
with it; pgvector lets only a superuser create the extension, so for any other role
run `CREATE EXTENSION vector` in that database once beforehand, or every delivery
fails with `permission denied`. A delivery after a failed step says so, in the bundle's
`egress_manifest.json` (`delivery.after_failed`) and in its record. Each destination
is tried three times, with backoff; a delivery that still fails is recorded and
never stops the run, and the exit code is non-zero. Every delivery is recorded as
`<run>/deliveries/<name>.json` and listed in the run record. Delivering again
replaces the bundle, upserts the rows (and removes the run's rows it did not write) and
replaces the record.

A `webhook` destination is sent the run's events, not its results: `extraction_settled`,
`stage_finished` and `stage_failed` (per step), `delivered` and `delivery_failed` (per
other destination) and `run_finished`; nothing per document or batch. Each is a JSON
body with `event`, `event_id`, `run_id`, `workflow`, `status` (the API's `running`,
then `done`, `failed` or `cancelled`; `run_finished` is `failed` only when a step
failed, since a failed delivery has its own event) and the event's own fields (`stage`,
`destination`, `error`), signed with HMAC-SHA256 under the key in `secret_env`: the
`X-Womblex-Signature: sha256=<hex>` header is what a receiver recomputes over the raw
body. An event is tried three times under the same `event_id`, so a receiver can drop
a duplicate; one that never lands is listed in the webhook's record (`events`) and
never stops the run. Polling still works. On the fleet the coordinator sends each
event as a recorded step, so a recovered coordinator does not send one twice, and
`run_finished` carries the run's status as `/v1/runs` reports it.

Each destination is checked before the first step: its credentials' or signing key's
environment variables are set and its URI's storage backend is installed; a failure refuses
the run with `destination_unavailable`. `dispatch-workflow` checks on the host it
runs on, a scheduled firing on the worker that fires it; the workers that deliver
need the same variables. A workflow that delivers runs over a run's `documents/`
directory. On the fleet the coordinator delivers after it writes the run's manifest,
and each firing delivers the whole run again.

Nothing redelivers by itself. After a failed stage is retried successfully,
`retry-stage` prints the command that delivers again:
`womblex deliver FILE --run-id R --store URI` (or `--shards DIR` for a local run).
It replaces the bundle and the record.

## Checking, running and dispatching

| Command | What it does |
|---|---|
| `womblex validate FILE [--shards DIR] [--json]` | Every check a run makes before its first step, and nothing else. Exit 1 if it would be refused. |
| `womblex plan FILE --shards DIR [--assume-price enrich=…]` | Which steps would run or skip, which are free (local) and which paid, and the exact tokens every paid call would send (`enrich`, `embed`, and AI chunking in `chunk`), in each provider's own tokens. Calls no paid API; a cost appears only for a price you supply. |
| `womblex run-workflow FILE --shards DIR` | Run the steps over a local shard directory. Refused before the first step, never halted after it: a failed step is recorded and the later steps still run. |
| `womblex dispatch-workflow FILE --run-id R --store URI [--firing T] [--dry-run]` | Queue the steps for the fleet; the run receives the whole workflow (its name, full spec, resolved config and digest), so nothing is stored. Refused before any queue row for anything the local runner refuses, a step the fleet does not dispatch (`layout`, `quality`), an order the coordinator would change, or work no live worker serves under the workflow's config (`no_worker`). A workflow dispatched over a run another workflow worked runs its own stages. `--firing` dispatches the steps of a scheduled firing whose dispatch failed, over that firing's batches alone. |
| `womblex schedule-workflow FILE [--store URI]` | Store the copy the firings read and make the queue's schedules equal its triggers, in one action; a workflow with no triggers is unscheduled and its copy deleted. Refuses a workflow its firing would refuse (no input store, an unsafe input prefix, no documents, a failing dry-run dispatch). |
| `womblex schedule-workflow FILE\|NAME --remove \| --status` | Unschedule the workflow and delete its stored copy, or list its schedules. Takes the name when the file is gone. |

`run-workflow` runs the steps in the order listed. It refuses an order in which a
step reads a required input that is missing, produced only by a later step, or
rewritten by a later step. `dispatch-workflow` also needs pipeline order, since the
coordinator runs stages in that order: `normalise`, `spellfix`, `enrich`, `chunk`,
`graph-refresh`, `embed`, `money`, `link`, `pii`. The fleet dispatches only these;
`layout` and `quality` run through `run-workflow` or `run-stage`. `pii` runs on the
fleet only as a workflow's step: a run dispatched without a workflow never masks.

## A trigger firing

At each cron time a worker on the workflow's input store starts a firing from the
stored copy; a worker on another store never runs it, and credentials are the worker's own.
The firing runs the same checks as scheduling, then
appends to the run named for the workflow: it queues extraction only for documents
the run has not seen (a changed file is a new document; the earlier version stays, though
its source can no longer be resolved or exported; a rewrite of the same bytes is not new),
then dispatches the steps over those batches alone, under the workflow as it stands
at that firing. A firing with nothing new runs nothing and says so. If the dispatch
fails after the batches are queued, the batches run on and the run reads `orphaned`
(`womblex jobs`, `/v1/runs`) until you dispatch that firing's steps with
`womblex dispatch-workflow FILE --run-id <name> --firing <UTC time> --store <uri>`.
There is no backfill of missed firings. Deleting a workflow file does not unschedule it: run
`schedule-workflow <name> --remove`.

## When a stage fails

A failed stage does not stop the stages after it; the run finishes and says which
failed. Retry one with `womblex retry-stage --run-id R --stage S --config C`: it
resumes from the first failed unit, and stages that ran after the failure are offered
a retry once it succeeds (D-2.1), which redoes only the units the failure left them
without.
