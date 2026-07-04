# OALC corpus asset (`stories/oalc`)

Configuration + invocation for building a persistent enrichment / AI-chunking /
embedding asset over the **Open Australian Legal Corpus** (OALC). Per Womblex's
corpus policy, this directory holds only config + a thin runner; all pipeline
machinery is library code (`ingest/records.py`, `analyse/*`, `process/*`). The
asset data + `RUNLOG.md` live outside the repo, in the corpus tree
(`.../open-australian-legal-corpus/derived/v<version>/`).

## Pipeline

```
records-ingest → enrich → AI-chunk → embed → graph-refresh
```

Enrichment persists the ILGS Document; AI chunking (`chunking_model:
kanon-2-enricher`) reuses it at zero extra API cost; graph-refresh rebuilds
mention→chunk edges offline afterwards. Every stage is resumable via its
per-stage checkpoint (re-run the command to resume). Sets accumulate in one
`derived/` tree — new selections append after existing shards (`next_batch_num`)
and checkpoints skip already-done batches, so nothing is re-spent.

## Files

- `field_mapping.yaml` — OALC record → element-shard field mapping (id, text,
  provenance columns), consumed by `run_tier.py`.
- `emergency_planning_set.yaml` — a curated **instrument** set (citation regex
  rules): NSW/Federal emergency, disaster, planning & environment legislation
  incl. all NSW LEPs and the EPBC Act + subordinate instruments.
- `instrument_references.yaml` — text-reference patterns selecting **judgments**
  that cite those instruments.
- `run_tier.py` — the runner (selection + ingest + staged run + measurements).

## Selection modes (`run_tier.py`)

| Flag | Selects | Streaming |
|------|---------|-----------|
| `--tier {t0,t1,t2,t3}` | a jurisdiction/type slice; `--sample N` stratifies by length | yes (materialised only for `--sample`) |
| `--select-file <yaml>` | legislation/instruments by **citation** regex rules | no (small curated sets) |
| `--reference-file <yaml>` | decisions whose **text** references any pattern, excluding already-ingested | yes (large sets) |

## Usage

Requires the Womblex venv, `ISAACUS_API_KEY` exported (stages), and the vendored
kanon-2 tokenizer (bundled — no Hugging Face access needed). Run ingest as its
own invocation from the long stage pass so a resume never re-appends:

```bash
ASSET=/path/to/open-australian-legal-corpus
# 1) select + ingest (offline; appends new batches)
python stories/oalc/run_tier.py --select-file stories/oalc/emergency_planning_set.yaml \
    --corpus $ASSET/corpus.jsonl --derived $ASSET/derived/v7.1.0 --stages ingest
# 2) the API + offline stages (resumable)
python stories/oalc/run_tier.py --select-file stories/oalc/emergency_planning_set.yaml \
    --corpus $ASSET/corpus.jsonl --derived $ASSET/derived/v7.1.0 \
    --stages enrich,chunk,embed,graph-refresh
```

Per-run measurements land in `derived/v7.1.0/<label>_measurements.json`; the
narrative record (throughput, per-request ceiling, cost, forecasts) is
`derived/v7.1.0/RUNLOG.md`.

## Not in OALC

OALC carries legislation + caselaw only. Policy frameworks and short-lived
instruments (e.g. AGCMF, DRFA, NSW Temporary Local Planning Instruments) and
some repealed Acts are absent — source them externally and ingest via
`ingest/records.py` (it takes any pre-extracted text) if needed.
