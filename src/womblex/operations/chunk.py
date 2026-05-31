"""run_chunking — split extracted text into token-bounded chunks."""

from __future__ import annotations

import logging

from womblex.config import WomblexConfig
from womblex.operations.models import DocumentResult
from womblex.process.chunker import build_chunk_input, chunk_batch, create_chunker
from womblex.redact.stage import annotate_chunks

logger = logging.getLogger(__name__)


def run_chunking(
    results: list[DocumentResult], config: WomblexConfig,
) -> list[DocumentResult]:
    """Split extracted text into token-bounded chunks.

    Reassembles narrative + tables from each result's element stream
    (the canonical source of truth) and feeds every result into a
    single :func:`chunk_batch` call so semchunk's ``processes`` and
    progress arguments parallelise across the whole batch. Applies
    ``flag``-mode redaction annotation to chunks when a redaction
    report exists on the extraction.
    """
    if not config.chunking.enabled:
        return results

    chunker = create_chunker(
        tokenizer=config.chunking.tokenizer,
        chunk_size=config.chunking.chunk_size,
        memoize=config.chunking.memoize,
        cache_maxsize=config.chunking.cache_maxsize,
        max_token_chars=config.chunking.max_token_chars,
    )

    chunk_cfg = config.chunking
    eligible: list[DocumentResult] = [
        dr for dr in results
        if dr.status == "completed" and dr.extraction is not None
    ]
    if not eligible:
        return results

    inputs = [
        build_chunk_input(
            source_hash=dr.doc_id,
            elements=dr.extraction.elements,
            include_tables=chunk_cfg.chunk_tables,
        )
        for dr in eligible
    ]

    chunks_by_doc = chunk_batch(
        inputs,
        chunker,
        overlap=chunk_cfg.overlap,
        processes=chunk_cfg.processes,
        progress=chunk_cfg.progress,
    )

    for dr in eligible:
        dr.chunks = chunks_by_doc.get(dr.doc_id, [])
        redaction_report = dr.extraction.redaction_report
        if (
            config.redaction.enabled
            and config.redaction.mode == "flag"
            and redaction_report
        ):
            dr.chunks = annotate_chunks(dr.chunks, redaction_report)
        logger.debug(
            "chunked: doc=%s chunks=%d narrative=%d table=%d",
            dr.doc_id,
            len(dr.chunks),
            sum(1 for c in dr.chunks if c.content_type == "narrative"),
            sum(1 for c in dr.chunks if c.content_type == "table"),
        )

    return results
