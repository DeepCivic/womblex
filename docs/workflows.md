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
or a config that does not validate. A workflow has no owner; the runs it starts
are admin-only, like CLI runs. Its digest (`sha256:…`) is over the parsed spec,
so formatting and comments do not change it, and is stamped on every sidecar
its steps write.

| Field | Meaning |
|---|---|
| `name` | The workflow's id: its schedules, its stored copy and its runs are named for it. |
| `preset` | The base config, a built-in preset or one saved under `--presets-dir`. `dataset` and `paths` are never taken from the workflow. |
| `config` | Overlay on the preset's config: mappings merge, anything else replaces, `null` drops a key to its default. |
| `input_binding` | `store` (the input store a firing lists and extracts from; a scheduled workflow must name it), `ingest_prefix` (the documents under it a firing extracts) and `batch_size`. Extraction is not a step. |
| `steps` | Downstream stages, each at most once, run in the order listed (`dispatch-workflow` needs pipeline order). |
| `triggers` | Cron expressions scheduled by `schedule-workflow`. |

## Checking, running and dispatching

| Command | What it does |
|---|---|
| `womblex validate FILE [--shards DIR] [--json]` | Every check a run makes before its first step, and nothing else. Exit 1 if it would be refused. |
| `womblex plan FILE --shards DIR [--assume-price enrich=…]` | Which steps would run or skip, and the exact tokens `enrich` and `embed` would send. Calls no paid API; a cost appears only for a price you supply. |
| `womblex run-workflow FILE --shards DIR` | Run the steps over a local shard directory. Refused before the first step, never halted after it: a failed step is recorded and the later steps still run. |
| `womblex dispatch-workflow FILE --run-id R --store URI [--dry-run]` | Queue the steps for the fleet; the run receives the workflow's recipe (its resolved config and digest), so nothing is stored. Refused before any queue row for anything the local runner refuses, a step the fleet does not dispatch (`layout`, `pii`, `quality`), an order the coordinator would change, or a run already worked under another workflow. |
| `womblex schedule-workflow FILE [--store URI]` | Store the copy the firings read and make the queue's schedules equal its triggers, in one action; a workflow with no triggers is unscheduled and its copy deleted. Refuses a workflow its firing would refuse (no input store, an unsafe input prefix, no documents, a failing dry-run dispatch). |
| `womblex schedule-workflow FILE\|NAME --remove \| --status` | Unschedule the workflow and delete its stored copy, or list its schedules. Takes the name when the file is gone. |

`run-workflow` runs the steps in the order listed. It refuses an order in which a
step reads a required input that is missing, produced only by a later step, or
rewritten by a later step. `dispatch-workflow` also needs pipeline order, since the
coordinator runs stages in that order: `normalise`, `spellfix`, `enrich`, `chunk`,
`graph-refresh`, `embed`, `money`, `link`. The fleet dispatches only these;
`layout`, `pii` and `quality` run through `run-workflow` or `run-stage`.

## A trigger firing

At each cron time a worker on the workflow's input store starts a firing from the
stored copy; a worker on another store never runs it, and credentials are the worker's own.
The firing it runs the same checks as scheduling,
queues the extraction batches for run `<name>-<UTC time>`, then dispatches the steps
under the workflow's recipe. If the dispatch fails after the batches are queued, the
batches run on and the run reads `orphaned` (`womblex jobs`, `/v1/runs`) until you
dispatch its steps with `womblex dispatch-workflow FILE --run-id <run> --store <uri>`.
A firing refuses a run that already holds cancelled batches.
It re-extracts the whole prefix every time; there is no "only new documents" and no
backfill of missed firings. Deleting a workflow file does not unschedule it: run
`schedule-workflow <name> --remove`.

## When a stage fails

A failed stage does not stop the stages after it; the run finishes and says which
failed. Retry one with `womblex retry-stage --run-id R --stage S --config C`: it
resumes from the first failed unit, and stages that ran after the failure are offered
a retry once it succeeds (D-2.1).
