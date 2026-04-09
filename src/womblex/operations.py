"""Independent operations for document processing.


Each function is standalone with clear input/output contracts.

There is no orchestrator — callers compose operations directly.


Operations:

- ``run_extraction``    — detect document type and extract text.

- ``run_redaction``     — detect black-box regions, apply mode to page text.

- ``run_chunking``      — split extracted text into token-bounded chunks.

- ``run_pii_cleaning``  — replace PII spans with ``<ENTITY_TYPE>`` tags.

- ``run_enrichment``    — call Isaacus (requires chunks + external client).


Standalone ingest paths (G-NAF PSV, SHP) have their own modules

under ``ingest/`` and do not use these operations.

"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field

from pathlib import Path

from typing import Any


from womblex.config import WomblexConfig
from womblex.ingest.detect import DocumentProfile, detect_file_type

from womblex.ingest.extract import ExtractionResult, extract_text

from womblex.process.chunker import TextChunk, chunk_document, create_chunker

from womblex.redact.stage import (

    annotate_chunks,

    annotate_extraction,

    build_detector,

    detect_redactions,

    apply_text_redaction,

)



logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------

# Result models

# ---------------------------------------------------------------------------



@dataclass

class DocumentResult:

    """Processing result for a single document."""


    path: Path

    doc_id: str

    profile: DocumentProfile | None = None

    extraction: ExtractionResult | None = None

    chunks: list[TextChunk] = field(default_factory=list)

    enrichment: Any = None  # EnrichmentResult when isaacus extra installed

    graph: Any = None  # DocumentGraph when isaacus extra installed

    error: str | None = None

    status: str = "pending"



@dataclass

class BatchResult:

    """Processing result for a batch of documents."""


    results: list[DocumentResult] = field(default_factory=list)


    @property

    def succeeded(self) -> int:

        return sum(1 for r in self.results if r.status == "completed")


    @property

    def failed(self) -> int:

        return sum(1 for r in self.results if r.status == "error")


    @property

    def enriched(self) -> int:

        return sum(1 for r in self.results if r.enrichment is not None)



# ---------------------------------------------------------------------------

# Extraction

# ---------------------------------------------------------------------------



def run_extraction(paths: list[Path], config: WomblexConfig) -> list[DocumentResult]:

    """Detect document types and extract text.


    One ``DocumentResult`` per logical extraction unit (PDFs and DOCX

    produce one; spreadsheets produce one per row or sheet).  No chunking
    or redaction is applied here.


    Args:

        paths: Document file paths to process.

        config: Pipeline configuration.


    Returns:

        List of DocumentResult with extraction populated.

    """

    results: list[DocumentResult] = []


    for path in paths:

        try:

            profile = detect_file_type(path, config.detection)

        except Exception as e:

            logger.error("Detection failed: doc=%s error=%s", path.stem, e)

            results.append(

                DocumentResult(path=path, doc_id=path.stem, error=str(e), status="error")

            )
            continue


        try:

            extractions = extract_text(

                path,

                profile,

                dpi=config.extraction.ocr.dpi,

                lang=config.extraction.ocr.lang,

            )

        except Exception as e:

            logger.error("Extraction failed: doc=%s error=%s", path.stem, e)

            results.append(

                DocumentResult(

                    path=path,

                    doc_id=path.stem,

                    profile=profile,

                    error=str(e),

                    status="error",

                )

            )
            continue


        for extraction in extractions:

            doc_id = extraction.document_id or path.stem

            dr = DocumentResult(

                path=path, doc_id=doc_id, profile=profile, extraction=extraction

            )

            if extraction.error:

                logger.warning("extraction error: doc=%s error=%s", doc_id, extraction.error)

                dr.error = extraction.error

                dr.status = "error"

            else:

                dr.status = "completed"

            results.append(dr)


        logger.info(

            "Extracted %s: units=%d",

            path.stem,

            len(extractions),

        )

    return results



# ---------------------------------------------------------------------------

# Redaction (independent post-extraction stage)

# ---------------------------------------------------------------------------



def run_redaction(

    results: list[DocumentResult], config: WomblexConfig

) -> list[DocumentResult]:

    """Detect and handle redacted regions in extracted documents.


    Renders each PDF page as an image, runs the black-box detector, and

    applies the configured mode:


    - ``flag``:     Annotate chunks/records (no text change).

    - ``blackout``: Prepend ``[REDACTED]`` to affected page text.

    - ``delete``:   Clear affected page text entirely.


    Non-PDF documents (spreadsheets, DOCX) are skipped — redaction

    detection requires a rasterisable page source.


    Args:

        results: DocumentResults from ``run_extraction``.

        config: Pipeline configuration (uses ``config.redaction``).


    Returns:

        The same list with redaction applied.

    """

    if not config.redaction.enabled:
        return results


    detector = build_detector(config.redaction)

    mode = config.redaction.mode


    for dr in results:

        if dr.status != "completed" or not dr.extraction:
            continue

        if dr.path.suffix.lower() not in {".pdf"}:
            continue


        report = detect_redactions(

            dr.path,

            dr.extraction.page_count,

            detector,

            dpi=config.redaction.dpi,

        )


        if not report.total:
            continue


        dr.extraction.redaction_report = report

        annotate_extraction(dr.extraction, report)


        if mode != "flag":

            apply_text_redaction(dr.extraction.pages, report, mode)


        logger.info(

            "Redaction [%s]: doc=%s pages_affected=%d regions=%d",

            mode,

            dr.doc_id,

            len(report.affected_pages),

            report.total,

        )

    return results



# ---------------------------------------------------------------------------

# Chunking (independent post-extraction stage)

# ---------------------------------------------------------------------------



def run_chunking(

    results: list[DocumentResult], config: WomblexConfig

) -> list[DocumentResult]:

    """Split extracted text into token-bounded chunks.


    Chunks narrative text and (optionally) tables separately. Applies

    ``flag``-mode redaction annotation to chunks when a redaction report

    exists on the extraction.


    Args:

        results: DocumentResults from extraction (and optionally redaction).

        config: Pipeline configuration (uses ``config.chunking``).


    Returns:

        The same list with ``chunks`` populated on each result.

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

    for dr in results:

        if dr.status != "completed" or not dr.extraction:
            continue


        tables = dr.extraction.tables if chunk_cfg.chunk_tables else None

        dr.chunks = chunk_document(

            dr.extraction.full_text,

            chunker,

            tables=tables,

            overlap=chunk_cfg.overlap,

            processes=chunk_cfg.processes,

        )  # type: ignore[arg-type]


        # Flag-mode redaction annotation on chunks

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



# ---------------------------------------------------------------------------

# PII Cleaning (independent post-extraction stage)

# ---------------------------------------------------------------------------



def run_pii_cleaning(

    results: list[DocumentResult], config: WomblexConfig

) -> list[DocumentResult]:

    """Replace PII spans with ``<ENTITY_TYPE>`` tags.


    Uses regex pattern recognisers (Presidio-style) for candidate detection

    and a Sentence Transformers context model (all-MiniLM-L6-v2) for

    low-confidence validation.  Requires ``pip install womblex[pii]``.


    The pipeline point is configurable via ``config.pii.pipeline_point``:


    - ``post_extraction``: Cleans page texts before chunking.

    - ``post_chunk``:      Cleans individual chunk texts after chunking.


    Args:

        results: DocumentResults from extraction (and optionally chunking).

        config: Pipeline configuration (uses ``config.pii``).


    Returns:

        The same list with PII replaced in-place.

    """

    if not config.pii.enabled:
        return results


    try:

        from womblex.pii.cleaner import PIICleaner

        from womblex.pii.stage import clean_chunks, clean_enriched_chunks, clean_extraction

    except ImportError as exc:

        raise ImportError(

            "PII cleaning requires the 'pii' extra. "

            "Install with: pip install womblex[pii]"

        ) from exc


    cleaner = PIICleaner(

        entities=config.pii.entities,

        model=config.pii.model,

        context_similarity_threshold=config.pii.context_similarity_threshold,

    )

    point = config.pii.pipeline_point


    for dr in results:

        if dr.status != "completed" or not dr.extraction:
            continue

        try:

            if point == "post_extraction":

                count = clean_extraction(dr.extraction, cleaner)

            elif point == "post_chunk":

                count = clean_chunks(dr.chunks, cleaner) if dr.chunks else 0

            elif point == "post_enrichment":

                if not dr.enrichment:

                    logger.debug(

                        "PII [post_enrichment]: doc=%s has no enrichment — falling back to regex",

                        dr.doc_id,

                    )

                    count = clean_chunks(dr.chunks, cleaner) if dr.chunks else 0

                elif not dr.chunks:

                    logger.warning(

                        "PII [post_enrichment]: doc=%s has no chunks — skipping", dr.doc_id,

                    )
                    continue

                else:

                    count = clean_enriched_chunks(

                        dr.chunks,

                        dr.enrichment,

                        cleaner,

                        entities=set(config.pii.entities),

                        person_types=set(config.pii.person_types),

                    )

            else:

                logger.warning("Unknown PII pipeline_point %r for %s — skipping", point, dr.doc_id)
                continue


            if count:

                logger.info(

                    "PII [%s]: doc=%s replacements=%d",

                    point,

                    dr.doc_id,

                    count,

                )

        except Exception as exc:

            logger.error("PII cleaning failed: doc=%s error=%s", dr.doc_id, exc)

    return results



# ---------------------------------------------------------------------------

# Enrichment (requires chunks — depends on chunking stage)

# ---------------------------------------------------------------------------



def run_enrichment(

    results: list[DocumentResult],

    config: WomblexConfig,

    client: object,

) -> None:

    """Enrich completed documents via the Isaacus API.


    Requires ``pip install womblex[isaacus]``.  Failures are logged

    per-document but do not halt the batch.  Results are stored on

    ``DocumentResult.enrichment`` and ``.graph``.


    Args:

        results: DocumentResults with chunks populated.

        config: Pipeline configuration (uses ``config.enrichment``).

        client: An ``isaacus.Isaacus`` client instance.

    """

    try:

        from womblex.analyse.enrich import enrich_document

        from womblex.analyse.graph import build_document_graph

    except ImportError as e:

        raise ImportError(

            "Isaacus enrichment requires the 'isaacus' extra. "

            "Install with: pip install womblex[isaacus]"

        ) from e


    enrich_cfg = config.enrichment

    if not enrich_cfg.enabled:
        return


    for dr in results:

        if dr.status != "completed" or not dr.extraction:
            continue


        full_text = dr.extraction.full_text

        if enrich_cfg.skip_short_documents > 0 and len(full_text) < enrich_cfg.skip_short_documents:

            logger.debug(

                "Skipping enrichment for %s (too short: %d chars)", dr.doc_id, len(full_text)

            )
            continue


        try:

            enrichment = enrich_document(

                full_text,

                client,

                model=enrich_cfg.model,

                max_retries=enrich_cfg.max_retries,

                retry_base_delay=enrich_cfg.retry_base_delay,

            )

            dr.enrichment = enrichment

            dr.graph = build_document_graph(

                document_id=dr.doc_id,

                enrichment=enrichment,

                chunks=dr.chunks or None,

            )

            logger.info(

                "Enriched %s: %d segments, %d persons, %d locations, %d nodes, %d edges",

                dr.doc_id,

                len(enrichment.segments),

                len(enrichment.persons),

                len(enrichment.locations),

                len(dr.graph.nodes),

                len(dr.graph.edges),

            )

        except Exception as e:

            logger.error("Enrichment failed for %s: %s", dr.doc_id, e)



# ---------------------------------------------------------------------------

# Enrichment batch wrapper

# ---------------------------------------------------------------------------



def enrich_batch(

    batch: BatchResult,

    config: WomblexConfig,

    client: object,

) -> None:

    """Enrich a batch via the Isaacus API.


    Thin wrapper around ``run_enrichment`` for backward compatibility.

    """

    run_enrichment(batch.results, config, client)



# ---------------------------------------------------------------------------

# Parquet output helpers

# ---------------------------------------------------------------------------



def write_batch_parquet(batch: BatchResult, output_path: Path) -> Path | None:

    """Write completed batch results to Parquet."""

    from womblex.store.output import write_results


    rows = []

    for r in batch.results:

        if r.status == "completed" and r.extraction:

            rows.append((r.doc_id, str(r.path), r.extraction))


    if not rows:

        logger.warning("No completed results to write to Parquet")

        return None


    return write_results(rows, output_path)



def write_batch_enrichment(batch: BatchResult, output_dir: Path) -> dict[str, Path | None]:

    """Write enrichment results and graph data to Parquet."""

    try:

        from womblex.store.enrichment_output import (

            write_enrichment_metadata,

            write_entity_mentions,

            write_graph_edges,

        )

    except ImportError as e:

        raise ImportError(

            "Enrichment output requires the 'isaacus' extra. "

            "Install with: pip install womblex[isaacus]"

        ) from e


    entity_rows: list[tuple[str, Any, list[object] | None]] = []

    graph_rows: list[tuple[str, Any]] = []

    meta_rows: list[tuple[str, Any]] = []


    for r in batch.results:

        if r.enrichment is None:
            continue

        entity_rows.append((r.doc_id, r.enrichment, r.chunks or None))  # type: ignore[arg-type]

        meta_rows.append((r.doc_id, r.enrichment))

        if r.graph is not None:

            graph_rows.append((r.doc_id, r.graph))


    paths: dict[str, Path | None] = {

        "entities": None,

        "graph_edges": None,

        "enrichment_meta": None,

    }


    if entity_rows:

        paths["entities"] = write_entity_mentions(entity_rows, output_dir / "entities.parquet")

    if graph_rows:

        paths["graph_edges"] = write_graph_edges(graph_rows, output_dir / "graph_edges.parquet")

    if meta_rows:

        paths["enrichment_meta"] = write_enrichment_metadata(

            meta_rows, output_dir / "enrichment_meta.parquet"

        )

    return paths

