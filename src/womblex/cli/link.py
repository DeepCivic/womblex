"""CLI: graph-building stages — ``enrich`` (Kanon-2) and ``link`` (register match).

Both are per-stage (``--shards``) commands over an existing extraction shard
directory, mirroring ``womblex chunk --shards``: each has an independent
``CheckpointManager`` and writes sibling parquets in place. ``enrich``
requires the ``isaacus`` extra + ``ISAACUS_API_KEY``; ``link`` is offline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command, make_isaacus_client

logger = logging.getLogger("womblex")


# --- enrich ------------------------------------------------------------------


def _register_enrich(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Shard dir (`<run_id>/documents/`) to enrich; writes "
             "`*.enrichment_entities.parquet` + `*.enrichment_meta.parquet` in place.",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML to source enrichment settings (model, retries, "
             "skip_short_documents). Defaults to EnrichmentConfig() if omitted.",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Per-stage checkpoint root (default: `<shard_dir>/../.enrich-checkpoint/`).")
    p.add_argument("--dataset", type=str, default="enrich",
                   help="Checkpoint dataset name. Default: 'enrich'.")
    p.add_argument("--no-resume", action="store_true",
                   help="Clear enrich-stage checkpoint before running (re-enrich every batch).")
    p.add_argument("--no-verify-resume", action="store_true",
                   help="Skip the resume-time `*.enrichment_entities.parquet` integrity scan.")


def cmd_enrich(args: argparse.Namespace) -> int:
    """Enrich a shard directory via the Kanon-2 enrichment API (per-stage)."""
    from womblex.analyse.enrich_stage import enrich_shards
    from womblex.config import EnrichmentConfig, load_config
    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.enrichment_output import ENRICHMENT_ENTITIES_SUFFIX
    from womblex.store.shard_audit import reconcile_stage_checkpoint_with_shards

    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if not any(shard_dir.glob("*._manifest.parquet")):
        logger.error("--shards directory has no `*._manifest.parquet`: %s", shard_dir)
        return 1

    if args.config:
        _cfg = load_config(args.config)
        enrichment_config = _cfg.enrichment
        text_source = _cfg.processing.text_source
        # Persist the raw Document for chunk-stage reuse whenever this config
        # also enables AI chunking — even if enrichment.enabled is false, since
        # the user is running `enrich` explicitly here. WomblexConfig already
        # auto-enables persist_document for the both-on case.
        persist_document = (
            enrichment_config.persist_document
            or bool(_cfg.chunking.chunking_model)
        )
    else:
        enrichment_config = EnrichmentConfig()
        text_source = "elements"
        persist_document = enrichment_config.persist_document

    try:
        import isaacus
    except ImportError:
        logger.error("isaacus SDK not installed. Install with: uv sync --extra isaacus")
        return 1
    try:
        client = make_isaacus_client()  # reads + strips ISAACUS_API_KEY
    except Exception as e:
        logger.error("Could not construct Isaacus client (is ISAACUS_API_KEY set?): %s", e)
        return 1

    checkpoint_root = args.checkpoint_dir or shard_dir.parent / ".enrich-checkpoint"
    ckpt = CheckpointManager(checkpoint_root, f"{args.dataset}_enrich")
    if args.no_resume:
        ckpt.clear()
    else:
        ckpt.load()
        if not args.no_verify_resume:
            dropped = reconcile_stage_checkpoint_with_shards(
                ckpt, shard_dir, suffix=ENRICHMENT_ENTITIES_SUFFIX)
            if dropped:
                logger.warning("Resume integrity scan: dropped %d doc(s) with corrupted "
                               "enrichment shards; they will be re-enriched.", len(dropped))

    logger.info("enrich --shards: dir=%s model=%s text_source=%s overflow=%s persist_doc=%s",
                shard_dir, enrichment_config.model, text_source,
                enrichment_config.overflow_strategy, persist_document)
    result = enrich_shards(shard_dir, enrichment_config, client=client,
                           text_source=text_source, persist_document=persist_document,
                           checkpoint_mgr=ckpt)
    logger.info(
        "Done: %d batches written, %d docs enriched, %d entities",
        result.batches_written, result.docs_enriched, result.total_entities,
    )
    return 0


# --- link --------------------------------------------------------------------


def _register_link(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Shard dir with `*.enrichment_entities.parquet`; writes "
             "`*.entity_links.parquet` in place.",
    )
    p.add_argument(
        "--config", type=Path, required=True,
        help="YAML carrying the `linking:` section (reference register mapping).",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Per-stage checkpoint root (default: `<shard_dir>/../.link-checkpoint/`).")
    p.add_argument("--dataset", type=str, default="link",
                   help="Checkpoint dataset name. Default: 'link'.")
    p.add_argument("--no-resume", action="store_true",
                   help="Clear link-stage checkpoint before running (re-link every batch).")
    p.add_argument("--no-verify-resume", action="store_true",
                   help="Skip the resume-time `*.entity_links.parquet` integrity scan.")


def cmd_link(args: argparse.Namespace) -> int:
    """Match enrichment candidates to a reference register (per-stage, offline)."""
    from womblex.config import load_config
    from womblex.link.stage import link_shards
    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.output import ENTITY_LINKS_SUFFIX
    from womblex.store.shard_audit import reconcile_stage_checkpoint_with_shards

    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if not any(shard_dir.glob("*.enrichment_entities.parquet")):
        logger.error(
            "--shards directory has no `*.enrichment_entities.parquet` — run "
            "`womblex enrich --shards` first: %s", shard_dir,
        )
        return 1

    linking_config = load_config(args.config).linking
    if linking_config.reference is None:
        logger.error("config `linking.reference` is required to run the link stage")
        return 1

    checkpoint_root = args.checkpoint_dir or shard_dir.parent / ".link-checkpoint"
    ckpt = CheckpointManager(checkpoint_root, f"{args.dataset}_link")
    if args.no_resume:
        ckpt.clear()
    else:
        ckpt.load()
        if not args.no_verify_resume:
            dropped = reconcile_stage_checkpoint_with_shards(
                ckpt, shard_dir, suffix=ENTITY_LINKS_SUFFIX)
            if dropped:
                logger.warning("Resume integrity scan: dropped %d doc(s) with corrupted "
                               "entity_links shards; they will be re-linked.", len(dropped))

    logger.info("link --shards: dir=%s reference=%s", shard_dir, linking_config.reference.path)
    result = link_shards(shard_dir, linking_config, checkpoint_mgr=ckpt)
    logger.info(
        "Done: %d batches written, %d docs linked, %d/%d link rows matched",
        result.batches_written, result.docs_linked, result.matched_links, result.total_links,
    )
    return 0


COMMANDS = [
    Command("enrich", "Enrich a shard directory via Kanon-2 (per-stage)", _register_enrich, cmd_enrich),
    Command("link", "Match enrichment candidates to a reference register", _register_link, cmd_link),
]
