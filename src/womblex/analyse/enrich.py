"""Isaacus enrichment API wrapper.

Calls the kanon-2-enricher model to transform extracted document text
into structured ILGS Documents containing segments, entities, and
relationships.  Operates on full document text (not chunks).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from womblex.analyse.models import (
    CrossReference,
    DateInfo,
    Email,
    EnrichmentResult,
    ExternalDocument,
    IDNumber,
    Location,
    Person,
    PhoneNumber,
    Quote,
    Segment,
    Span,
    Term,
    Website,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "kanon-2-enricher"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 2.0


# ---------------------------------------------------------------------------
# Response conversion helpers
# ---------------------------------------------------------------------------


def _to_span(raw: object) -> Span | None:
    """Convert an SDK span object to our Span dataclass."""
    if raw is None:
        return None
    return Span(start=raw.start, end=raw.end)  # type: ignore[union-attr]


def _to_span_list(raw_list: list[object] | None) -> list[Span]:
    """Convert a list of SDK span objects to our Span dataclasses."""
    if not raw_list:
        return []
    return [Span(start=s.start, end=s.end) for s in raw_list]  # type: ignore[union-attr]


def _convert_segments(raw_segments: list[object]) -> list[Segment]:
    result: list[Segment] = []
    for seg in raw_segments:
        result.append(Segment(
            id=seg.id,  # type: ignore[union-attr]
            kind=seg.kind,  # type: ignore[union-attr]
            type=seg.type,  # type: ignore[union-attr]
            category=seg.category,  # type: ignore[union-attr]
            span=_to_span(seg.span),  # type: ignore[union-attr,arg-type]
            parent=seg.parent,  # type: ignore[union-attr]
            children=list(seg.children) if seg.children else [],  # type: ignore[union-attr]
            level=seg.level,  # type: ignore[union-attr]
            type_name=_to_span(seg.type_name),  # type: ignore[union-attr]
            code=_to_span(seg.code),  # type: ignore[union-attr]
            title=_to_span(seg.title),  # type: ignore[union-attr]
        ))
    return result


def _convert_locations(raw: list[object]) -> list[Location]:
    result: list[Location] = []
    for loc in raw:
        result.append(Location(
            id=loc.id,  # type: ignore[union-attr]
            name=_to_span(loc.name),  # type: ignore[union-attr,arg-type]
            type=loc.type,  # type: ignore[union-attr]
            mentions=_to_span_list(loc.mentions),  # type: ignore[union-attr]
            parent=loc.parent,  # type: ignore[union-attr]
            children=list(loc.children) if loc.children else [],  # type: ignore[union-attr]
        ))
    return result


def _convert_persons(raw: list[object]) -> list[Person]:
    result: list[Person] = []
    for per in raw:
        result.append(Person(
            id=per.id,  # type: ignore[union-attr]
            name=_to_span(per.name),  # type: ignore[union-attr,arg-type]
            type=per.type,  # type: ignore[union-attr]
            role=per.role,  # type: ignore[union-attr]
            mentions=_to_span_list(per.mentions),  # type: ignore[union-attr]
            parent=per.parent,  # type: ignore[union-attr]
            children=list(per.children) if per.children else [],  # type: ignore[union-attr]
            residence=per.residence,  # type: ignore[union-attr]
        ))
    return result


def _convert_crossreferences(raw: list[object]) -> list[CrossReference]:
    return [
        CrossReference(
            start=xref.start,  # type: ignore[union-attr]
            end=xref.end,  # type: ignore[union-attr]
            span=_to_span(xref.span),  # type: ignore[union-attr,arg-type]
        )
        for xref in raw
    ]


def _convert_terms(raw: list[object]) -> list[Term]:
    return [
        Term(
            id=t.id,  # type: ignore[union-attr]
            name=_to_span(t.name),  # type: ignore[union-attr,arg-type]
            meaning=_to_span(t.meaning),  # type: ignore[union-attr,arg-type]
            mentions=_to_span_list(t.mentions),  # type: ignore[union-attr]
        )
        for t in raw
    ]


def _convert_external_documents(raw: list[object]) -> list[ExternalDocument]:
    return [
        ExternalDocument(
            id=exd.id,  # type: ignore[union-attr]
            name=_to_span(exd.name),  # type: ignore[union-attr,arg-type]
            type=exd.type,  # type: ignore[union-attr]
            reception=exd.reception,  # type: ignore[union-attr]
            mentions=_to_span_list(exd.mentions),  # type: ignore[union-attr]
            pinpoints=_to_span_list(exd.pinpoints),  # type: ignore[union-attr]
            jurisdiction=exd.jurisdiction,  # type: ignore[union-attr]
        )
        for exd in raw
    ]


def _convert_quotes(raw: list[object]) -> list[Quote]:
    return [
        Quote(
            span=_to_span(q.span),  # type: ignore[union-attr,arg-type]
            amending=q.amending,  # type: ignore[union-attr]
            source_segment=q.source_segment,  # type: ignore[union-attr]
            source_document=q.source_document,  # type: ignore[union-attr]
            source_person=q.source_person,  # type: ignore[union-attr]
        )
        for q in raw
    ]


def _convert_dates(raw: list[object]) -> list[DateInfo]:
    return [
        DateInfo(
            value=d.value,  # type: ignore[union-attr]
            type=d.type,  # type: ignore[union-attr]
            mentions=_to_span_list(d.mentions),  # type: ignore[union-attr]
            person=d.person,  # type: ignore[union-attr]
        )
        for d in raw
    ]


def _convert_contact_info(
    raw_emails: list[object],
    raw_websites: list[object],
    raw_phones: list[object],
    raw_ids: list[object],
) -> tuple[list[Email], list[Website], list[PhoneNumber], list[IDNumber]]:
    emails = [
        Email(
            address=e.address, person=e.person,  # type: ignore[union-attr]
            mentions=_to_span_list(e.mentions),  # type: ignore[union-attr]
        )
        for e in raw_emails
    ]
    websites = [
        Website(
            url=w.url, person=w.person,  # type: ignore[union-attr]
            mentions=_to_span_list(w.mentions),  # type: ignore[union-attr]
        )
        for w in raw_websites
    ]
    phones = [
        PhoneNumber(
            number=p.number, person=p.person,  # type: ignore[union-attr]
            mentions=_to_span_list(p.mentions),  # type: ignore[union-attr]
        )
        for p in raw_phones
    ]
    ids = [
        IDNumber(
            number=i.number, person=i.person,  # type: ignore[union-attr]
            mentions=_to_span_list(i.mentions),  # type: ignore[union-attr]
        )
        for i in raw_ids
    ]
    return emails, websites, phones, ids


# ---------------------------------------------------------------------------
# SDK response → EnrichmentResult
# ---------------------------------------------------------------------------


def _convert_document(doc: object) -> EnrichmentResult:
    """Convert an Isaacus SDK ILGS Document to our EnrichmentResult."""
    emails, websites, phones, ids = _convert_contact_info(
        doc.emails, doc.websites, doc.phone_numbers, doc.id_numbers,  # type: ignore[union-attr]
    )
    return EnrichmentResult(
        text=doc.text,  # type: ignore[union-attr]
        type=doc.type,  # type: ignore[union-attr]
        jurisdiction=doc.jurisdiction,  # type: ignore[union-attr]
        title=_to_span(doc.title),  # type: ignore[union-attr]
        subtitle=_to_span(doc.subtitle),  # type: ignore[union-attr]
        segments=_convert_segments(doc.segments),  # type: ignore[union-attr]
        crossreferences=_convert_crossreferences(doc.crossreferences),  # type: ignore[union-attr]
        locations=_convert_locations(doc.locations),  # type: ignore[union-attr]
        persons=_convert_persons(doc.persons),  # type: ignore[union-attr]
        emails=emails,
        websites=websites,
        phone_numbers=phones,
        id_numbers=ids,
        terms=_convert_terms(doc.terms),  # type: ignore[union-attr]
        external_documents=_convert_external_documents(doc.external_documents),  # type: ignore[union-attr]
        quotes=_convert_quotes(doc.quotes),  # type: ignore[union-attr]
        dates=_convert_dates(doc.dates),  # type: ignore[union-attr]
        headings=_to_span_list(doc.headings),  # type: ignore[union-attr]
        junk=_to_span_list(doc.junk),  # type: ignore[union-attr]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class EnrichmentUsage:
    """Token usage from an enrichment API call."""

    input_tokens: int


@dataclass
class EnrichmentResponse:
    """Response from enriching one or more documents."""

    results: list[EnrichmentResult]
    usage: EnrichmentUsage


def enrich_document(
    text: str,
    client: object,
    *,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> EnrichmentResult:
    """Enrich a single document via the Isaacus enrichment API.

    Args:
        text: Full extracted document text.
        client: An ``isaacus.Isaacus`` client instance.
        model: Enrichment model identifier.
        max_retries: Maximum retry attempts for rate-limit (429) errors.
        retry_base_delay: Base delay in seconds for exponential backoff.

    Returns:
        EnrichmentResult containing segments, entities, and relationships.

    Raises:
        RuntimeError: If enrichment fails after all retries.
    """
    return enrich_documents([text], client, model=model,
                            max_retries=max_retries,
                            retry_base_delay=retry_base_delay)[0]


def enrich_documents(
    texts: list[str],
    client: object,
    *,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> list[EnrichmentResult]:
    """Enrich one or more documents via the Isaacus enrichment API.

    Handles 429 rate-limit errors with exponential backoff.

    Args:
        texts: List of full document texts.
        client: An ``isaacus.Isaacus`` client instance.
        model: Enrichment model identifier.
        max_retries: Maximum retry attempts for rate-limit errors.
        retry_base_delay: Base delay in seconds for exponential backoff.

    Returns:
        List of EnrichmentResult objects, one per input text.

    Raises:
        RuntimeError: If enrichment fails after all retries.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.enrichments.create(  # type: ignore[union-attr]
                model=model,
                texts=texts,
            )
            # Convert SDK response to our models
            results: list[EnrichmentResult] = []
            for r in response.results:
                results.append(_convert_document(r.document))

            logger.info(
                "Enriched %d document(s), usage: %d input tokens",
                len(results),
                response.usage.input_tokens,
            )
            return results

        except Exception as e:
            last_error = e
            error_str = str(e)

            # Retry on rate limits (429)
            if "429" in error_str or "rate" in error_str.lower():
                delay = retry_base_delay * (2 ** attempt)
                logger.warning(
                    "Rate limited on enrichment (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, max_retries + 1, delay,
                )
                time.sleep(delay)
                continue

            # Non-retryable error
            logger.error("Enrichment failed: %s", e)
            raise RuntimeError(f"Enrichment failed: {e}") from e

    raise RuntimeError(
        f"Enrichment failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error
