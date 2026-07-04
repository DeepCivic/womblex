"""OALC corpus-asset tier runner (step 0).

Invocation + measurement script (corpus-specific; the pipeline machinery is
all in the womblex library). Selects a tier's records from the pristine
``corpus.jsonl``, ingests them into the derived shard tree, and runs the
asset pipeline stages (enrich → chunk → embed → graph-refresh), recording
per-endpoint throughput / token / 429 measurements for the RUNLOG.

T0 pilot: ``--tier t0 --sample 500`` selects 500 NSW decisions stratified by
length (char-length proxy, tail deliberately included), then measures. Later
tiers select the whole slice (``--tier t1`` = all NSW decisions, no sample).

Usage (from the Womblex repo root, its venv active)::

    python stories/oalc/run_tier.py --tier t0 --sample 500 \
        --corpus /path/to/corpus.jsonl \
        --derived /path/to/open-australian-legal-corpus/derived/v7.1.0 \
        --stages ingest,enrich,chunk,embed,graph-refresh

Spend is real (Isaacus API). ``--stages ingest`` alone does the offline
selection + ingest + exact token stats without any API spend.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from womblex.ingest.records import RecordFieldMapping, ingest_records
from womblex.store.provenance_output import write_corpus_manifest
from womblex.utils.token_packer import TokenCounter

logger = logging.getLogger("oalc.run_tier")

_STORY_DIR = Path(__file__).resolve().parent

# Tier predicates over (type, jurisdiction). T0 shares T1's slice (NSW
# decisions) but samples it; T2/T3 are defined for later tiers.
_TIER_PREDICATES = {
    "t0": lambda t, j: j == "new_south_wales" and t == "decision",
    "t1": lambda t, j: j == "new_south_wales" and t == "decision",
    "t2": lambda t, j: t == "decision" and j != "new_south_wales",
    "t3": lambda t, j: t in ("primary_legislation", "secondary_legislation", "bill"),
}


def load_mapping() -> RecordFieldMapping:
    raw = yaml.safe_load((_STORY_DIR / "field_mapping.yaml").read_text())
    return RecordFieldMapping(
        id_field=raw["id_field"],
        text_field=raw["text_field"],
        provenance_fields=raw.get("provenance_fields", []),
        collection_id=raw.get("collection_id", ""),
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def scan_slice(corpus_path: Path, tier: str) -> list[tuple[str, int]]:
    """Stream the corpus, return (version_id, char_len) for the tier's records."""
    predicate = _TIER_PREDICATES[tier]
    out: list[tuple[str, int]] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if predicate(r.get("type"), r.get("jurisdiction")):
                out.append((r["version_id"], len(r.get("text", "") or "")))
    logger.info("scan: %d records match tier %s", len(out), tier)
    return out


def stratified_sample(items: list[tuple[str, int]], n: int, seed: int) -> set[str]:
    """Pick ``n`` version_ids stratified by char length, tail included.

    Sorts by length, splits into ``n`` equal-count bins and takes one random
    id per bin — so the sample spans the whole length range (each decile of
    length equally represented, over-weighting the long tail relative to a
    proportional sample). The single longest record is force-included so the
    extreme tail — where rate limits bite — is always probed.
    """
    if len(items) <= n:
        return {vid for vid, _ in items}
    rng = random.Random(seed)
    ordered = sorted(items, key=lambda t: t[1])
    picked: set[str] = set()
    bin_size = len(ordered) / n
    for i in range(n):
        lo = int(i * bin_size)
        hi = max(lo + 1, int((i + 1) * bin_size))
        vid, _ = ordered[rng.randrange(lo, min(hi, len(ordered)))]
        picked.add(vid)
    picked.add(ordered[-1][0])  # force the longest doc
    # Top up if bin collisions left us short of n.
    idx = len(ordered) - 1
    while len(picked) < n and idx >= 0:
        picked.add(ordered[idx][0])
        idx -= 1
    return picked


def extract_records(corpus_path: Path, wanted: set[str]):
    """Second pass: stream the full records for ``wanted`` version_ids.

    A generator — never materialises the whole slice, so a full tier (tens of
    thousands of ~30 KB docs) ingests in constant memory. ``ingest_records``
    consumes it streaming (batches of ``batch_size``).
    """
    remaining = set(wanted)
    pulled = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = r.get("version_id")
            if vid in remaining:
                yield r
                pulled += 1
                remaining.discard(vid)
                if not remaining:
                    break
    logger.info("extract: pulled %d/%d records", pulled, len(wanted))


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass
class StageMeasurement:
    stage: str
    wall_seconds: float = 0.0
    units: int = 0                 # docs or chunks processed
    input_tokens: int = 0          # exact local kanon-2 token count of inputs
    requests: int = 0
    rate_limit_hits: int = 0
    max_request_tokens: int = 0    # largest single request's token count (ceiling probe)
    max_ok_request_tokens: int = 0  # largest request that succeeded
    notes: list[str] = field(default_factory=list)

    def tokens_per_min(self) -> float:
        return self.input_tokens / (self.wall_seconds / 60) if self.wall_seconds else 0.0


class _RateLimitCounter(logging.Handler):
    """Counts Kanon 429 backoff warnings emitted during a stage."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.hits = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "Rate limited" in record.getMessage():
            self.hits += 1


def token_stats(counter: TokenCounter, texts: list[str]) -> dict:
    """Exact per-doc token distribution for the RUNLOG."""
    counts = sorted(counter.count_batch(texts))
    if not counts:
        return {}
    n = len(counts)

    def pct(p: float) -> int:
        return counts[min(n - 1, int(p * n))]

    return {
        "docs": n, "total_tokens": sum(counts),
        "min": counts[0], "p50": pct(0.50), "p90": pct(0.90),
        "p99": pct(0.99), "max": counts[-1],
        "over_16k": sum(1 for c in counts if c > 16384),
        "over_32k": sum(1 for c in counts if c > 32768),
        "over_100k": sum(1 for c in counts if c > 100_000),
    }


# ---------------------------------------------------------------------------
# Recording client (per-request token size + usage, for the enrich/embed probes)
# ---------------------------------------------------------------------------


class _RecordingEndpoint:
    """Wraps an SDK endpoint's ``create`` to tally request tokens + usage."""

    def __init__(self, inner: object, counter: TokenCounter, m: StageMeasurement):
        self._inner = inner
        self._counter = counter
        self._m = m

    def create(self, **kwargs):  # noqa: ANN003
        texts = list(kwargs.get("texts", []))
        toks = sum(self._counter.count_batch(texts)) if texts else 0
        self._m.requests += 1
        self._m.max_request_tokens = max(self._m.max_request_tokens, toks)
        resp = self._inner.create(**kwargs)  # type: ignore[attr-defined]
        self._m.max_ok_request_tokens = max(self._m.max_ok_request_tokens, toks)
        usage = getattr(resp, "usage", None)
        self._m.input_tokens += getattr(usage, "input_tokens", 0) or toks
        return resp


class _RecordingClient:
    """Delegates to a real Isaacus client, recording enrich/embed requests."""

    def __init__(self, inner: object, counter: TokenCounter, m: StageMeasurement, endpoint: str):
        self._inner = inner
        setattr(self, endpoint, _RecordingEndpoint(getattr(inner, endpoint), counter, m))

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def _timed(m: StageMeasurement, logger_name: str = "womblex.analyse.enrich"):
    """Context manager: time a stage + count 429s from ``logger_name``."""

    class _Ctx:
        def __enter__(self_):
            self_.handler = _RateLimitCounter()
            logging.getLogger(logger_name).addHandler(self_.handler)
            self_.start = time.time()
            return self_

        def __exit__(self_, *exc):
            m.wall_seconds = time.time() - self_.start
            m.rate_limit_hits = self_.handler.hits
            logging.getLogger(logger_name).removeHandler(self_.handler)
            return False

    return _Ctx()


# ---------------------------------------------------------------------------
# Stage orchestration
# ---------------------------------------------------------------------------


def run_enrich(shard_dir: Path, counter: TokenCounter, cfg) -> StageMeasurement:  # noqa: ANN001
    from womblex.analyse.enrich_stage import enrich_shards
    from womblex.cli._shared import make_isaacus_client
    from womblex.store.checkpoint import CheckpointManager

    m = StageMeasurement(stage="enrich")
    client = _RecordingClient(make_isaacus_client(), counter, m, "enrichments")
    ckpt = CheckpointManager(shard_dir.parent / ".enrich-checkpoint", "oalc_enrich")
    ckpt.load()
    with _timed(m):
        result = enrich_shards(
            shard_dir, cfg.enrichment, client=client,
            persist_document=True, checkpoint_mgr=ckpt, token_counter=counter,
        )
    m.units = result.docs_enriched
    m.notes.append(f"entities={result.total_entities}")
    return m


def run_embed(shard_dir: Path, counter: TokenCounter, cfg) -> StageMeasurement:  # noqa: ANN001
    from womblex.analyse.embed_stage import embed_shards
    from womblex.cli._shared import make_isaacus_client
    from womblex.store.checkpoint import CheckpointManager

    m = StageMeasurement(stage="embed")
    client = _RecordingClient(make_isaacus_client(), counter, m, "embeddings")
    ckpt = CheckpointManager(shard_dir.parent / ".embed-checkpoint", "oalc_embed")
    ckpt.load()
    with _timed(m, "womblex.analyse.embed"):
        result = embed_shards(shard_dir, cfg.embedding, client=client, checkpoint_mgr=ckpt)
    m.units = result.chunks_embedded
    return m


def run_chunk(shard_dir: Path, counter: TokenCounter, cfg) -> StageMeasurement:  # noqa: ANN001
    from womblex.process.chunk_stage import chunk_shards
    from womblex.store.checkpoint import CheckpointManager

    m = StageMeasurement(stage="chunk")
    # AI chunking reuses the persisted enrichment Document (zero API tokens when
    # byte-identity holds); input token throughput is the reassembled narrative.
    ckpt = CheckpointManager(shard_dir.parent / ".chunk-checkpoint", "oalc_chunk")
    ckpt.load()
    with _timed(m, "womblex.process.chunk_stage"):
        result = chunk_shards(shard_dir, cfg.chunking, checkpoint_mgr=ckpt)
    m.units = result.total_chunks
    m.notes.append(f"docs_chunked={result.docs_chunked}")
    return m


def run_graph_refresh(shard_dir: Path) -> StageMeasurement:
    from womblex.analyse.graph_refresh import refresh_graph_edges
    from womblex.store.checkpoint import CheckpointManager

    m = StageMeasurement(stage="graph-refresh")
    ckpt = CheckpointManager(shard_dir.parent / ".graph-refresh-checkpoint", "oalc_graph")
    ckpt.load()
    with _timed(m, "womblex.analyse.graph_refresh"):
        result = refresh_graph_edges(shard_dir, checkpoint_mgr=ckpt)
    m.units = result.edges_added
    m.notes.append(f"docs_refreshed={result.docs_refreshed}")
    return m


def _asset_config(cfg_path: Path | None):  # noqa: ANN202
    """Load the OALC WomblexConfig, or synthesise the stage sub-configs."""
    from womblex.config import (
        ChunkingConfig,
        DatasetConfig,
        EmbeddingConfig,
        EnrichmentConfig,
        PathsConfig,
        WomblexConfig,
    )

    if cfg_path and cfg_path.exists():
        from womblex.config import load_config

        return load_config(cfg_path)
    return WomblexConfig(
        dataset=DatasetConfig(name="oalc-v7.1.0"),
        paths=PathsConfig(input_root=Path("."), output_root=Path("."), checkpoint_dir=Path(".")),
        enrichment=EnrichmentConfig(enabled=True, persist_document=True),
        chunking=ChunkingConfig(enabled=True, chunking_model="kanon-2-enricher"),
        embedding=EmbeddingConfig(enabled=True, task="retrieval/document"),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OALC corpus-asset tier runner (step 0)")
    ap.add_argument("--tier", required=True, choices=sorted(_TIER_PREDICATES))
    ap.add_argument("--corpus", type=Path, required=True, help="Path to pristine corpus.jsonl")
    ap.add_argument("--derived", type=Path, required=True, help="derived/v<version>/ output root")
    ap.add_argument("--sample", type=int, default=None, help="Stratified sample size (T0). Omit = whole slice.")
    ap.add_argument("--seed", type=int, default=20260704)
    ap.add_argument("--config", type=Path, default=None, help="Optional WomblexConfig YAML for stage settings")
    ap.add_argument("--stages", default="ingest,enrich,chunk,embed,graph-refresh",
                    help="Comma list from: ingest,enrich,chunk,embed,graph-refresh")
    ap.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    shard_dir = args.derived / "shards"
    counter = TokenCounter()
    cfg = _asset_config(args.config)
    mapping = load_mapping()
    measurements: dict[str, object] = {"tier": args.tier, "sample": args.sample}

    if "ingest" in stages:
        items = scan_slice(args.corpus, args.tier)
        wanted = stratified_sample(items, args.sample, args.seed) if args.sample else {v for v, _ in items}
        if args.sample:
            # Small sampled slice (T0): materialise so we can emit exact token
            # stats for the RUNLOG.
            records = list(extract_records(args.corpus, wanted))
            ingest_records(records, shard_dir, mapping, batch_size=args.batch_size)
            stats = token_stats(counter, [r.get(mapping.text_field, "") or "" for r in records])
            measurements["token_stats"] = stats
            logger.info("ingest: token stats %s", json.dumps(stats))
        else:
            # Full tier: stream extract → ingest in constant memory (tens of
            # thousands of docs). Length distribution is the char lengths from
            # the scan; exact token stats are skipped (measured live per stage).
            ingest_records(
                extract_records(args.corpus, wanted), shard_dir, mapping,
                batch_size=args.batch_size,
            )
            char_lens = sorted(c for _, c in items)
            measurements["char_len_stats"] = {
                "docs": len(char_lens),
                "p50_chars": char_lens[len(char_lens) // 2] if char_lens else 0,
                "max_chars": char_lens[-1] if char_lens else 0,
            }
        write_corpus_manifest(shard_dir)

    stage_runners = {
        "enrich": lambda: run_enrich(shard_dir, counter, cfg),
        "chunk": lambda: run_chunk(shard_dir, counter, cfg),
        "embed": lambda: run_embed(shard_dir, counter, cfg),
        "graph-refresh": lambda: run_graph_refresh(shard_dir),
    }
    for name in ("enrich", "chunk", "embed", "graph-refresh"):
        if name not in stages:
            continue
        m = stage_runners[name]()
        measurements[name] = {
            "wall_seconds": round(m.wall_seconds, 1),
            "units": m.units, "input_tokens": m.input_tokens,
            "tokens_per_min": round(m.tokens_per_min()),
            "requests": m.requests, "rate_limit_hits": m.rate_limit_hits,
            "max_request_tokens": m.max_request_tokens,
            "max_ok_request_tokens": m.max_ok_request_tokens,
            "notes": m.notes,
        }
        logger.info("stage %s: %s", name, json.dumps(measurements[name]))

    out = args.derived / "t0_measurements.json"
    out.write_text(json.dumps(measurements, indent=2))
    logger.info("wrote measurements → %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
