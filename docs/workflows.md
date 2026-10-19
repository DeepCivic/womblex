# Stored workflows

A stored workflow is a named recipe: a preset, optional config changes, the
downstream steps to run in order, and optionally conditions, destinations and
cron triggers. It is kept in the database DBOS's own tables live in (Postgres
for a fleet, a local SQLite file otherwise). The "why" is in
[decisions-operations.md](decisions-operations.md) ("Trust and recipes").

**A workflow is saved only through code.** There is no `womblex workflow save`,
`list`, `show` or `delete`, and no workflow ships by default (D-2.2). The commands
below read a workflow by id; `WorkflowStore` writes it.

## Saving one

```python
from womblex.cloud.workflow_store import WorkflowStore, ensure_workflow_schema, validate_workflow_spec

spec = validate_workflow_spec({
    "preset": "DEFAULT-Isaacus",                      # a built-in preset, or one saved under --presets-dir
    "config": {"chunking": {"chunk_size": 512}},      # laid over the preset's config, key by key
    "input_binding": {"ingest_prefix": "alice/inbox", "batch_size": 50},
    "steps": [{"stage": "normalise"}, {"stage": "enrich"}, {"stage": "chunk"}, {"stage": "embed"}],
    "triggers": [{"cron": "0 2 * * *", "timezone": "Australia/Sydney"}],
})
ensure_workflow_schema(None)                          # None: the local SQLite file; else a DSN
with WorkflowStore(None) as store:
    store.save("nightly", spec, owner="alice")        # owner None is an unscoped (CLI/console) workflow
```

`validate_workflow_spec` refuses what could never run: an unknown stage, a stage
listed twice, a bad cron expression or timezone, a config that does not validate,
a condition on a column outside the manifest. `WorkflowStore.save` runs the same
checks, and the id must be a single safe token (letters, digits, `.`, `_`, `-`).
The workflow's digest (`sha256:…`) is over the spec, and is stamped on every
sidecar its steps write.

| Field | Meaning |
|---|---|
| `preset` | The base config. `dataset` and `paths` are never taken from the workflow. |
| `config` | Overlay on the preset's config: mappings merge, anything else replaces, `null` drops a key to its default. |
| `input_binding` | `ingest_prefix` (the documents a firing extracts) and `batch_size`. Extraction is not a step. |
| `steps` | Downstream stages, each at most once, run in the order listed (`dispatch-workflow` needs pipeline order). |
| `conditions` | `column op value` over manifest columns; they label each document `holds` / `fails` / `unevaluable` and never refuse, halt or narrow a run. Labelled by `run-workflow` only; the fleet does not label. |
| `destinations` | Stored, not acted on yet. |
| `triggers` | Cron expressions scheduled by `schedule-workflow`. |

## Checking, running and dispatching

| Command | What it does |
|---|---|
| `womblex validate <id> [--shards DIR] [--json]` | Every check a run makes before its first step, and nothing else. Exit 1 if it would be refused. |
| `womblex plan <id> --shards DIR [--assume-price enrich=…]` | Which steps would run or skip, and the exact tokens `enrich` and `embed` would send. Calls no paid API; a cost appears only for a price you supply. |
| `womblex run-workflow <id> --shards DIR` | Run the steps over a local shard directory. Refused before the first step, never halted after it: a failed step is recorded and the later steps still run. |
| `womblex dispatch-workflow <id> --run-id R --store URI [--dry-run]` | Queue the steps for the fleet. Refused before any queue row for anything the local runner refuses, a step the fleet does not dispatch (`layout`, `pii`, `quality`), an order the coordinator would change, or a run already worked under another workflow. |
| `womblex schedule-workflow <id> [--store URI --ingest URI] [--remove \| --status]` | Make the queue's schedules equal the workflow's triggers. Refuses a workflow its firing would refuse (an input prefix outside the owner's tree, no documents, a failing dry-run dispatch). |

`run-workflow` runs the steps in the order listed. It refuses an order in which a
step reads a required input that is missing, produced only by a later step, or
rewritten by a later step. `dispatch-workflow` also needs pipeline order, since the
coordinator runs stages in that order: `normalise`, `spellfix`, `enrich`, `chunk`,
`graph-refresh`, `embed`, `money`, `link`. The fleet dispatches only these;
`layout`, `pii` and `quality` run through `run-workflow` or `run-stage`.

## A trigger firing

At each cron time a worker starts a firing: it runs the same checks as scheduling,
queues the extraction batches for run `<id>-<UTC time>`, then dispatches the steps
under the workflow's recipe. If the dispatch fails after the batches are queued, the
batches run on and the run reads `orphaned` (`womblex jobs`, `/v1/runs`) until you
dispatch its steps with `womblex dispatch-workflow <id> --run-id <run> --store <uri>`.
A firing refuses a run that already holds cancelled batches.
It re-extracts the whole prefix every time; there is no "only new documents" and no
backfill of missed firings. Deleting a workflow does not unschedule it: run
`schedule-workflow <id> --remove`.

## When a stage fails

A failed stage does not stop the stages after it; the run finishes and says which
failed. Retry one with `womblex retry-stage --run-id R --stage S --config C`: it
resumes from the first failed unit, and stages that ran after the failure are offered
a retry once it succeeds (D-2.1).
