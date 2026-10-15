"""Isaacus enrichment API wrapper, and the enrichment provider seam.

Calls the kanon-2-enricher model to transform extracted document text
into structured ILGS Documents containing segments, entities, and
relationships.  Operates on full document text (not chunks).

An `EnrichmentProvider` is what the enrich stage calls: text in, one
`EnrichmentResult` per text, offsets into the text sent, results independent of
batch composition. It declares the limits the stage packs and splits by
(`ProviderLimits`). Providers are the `enrichment` registry slot
(`enrichment.model`; plugins through `womblex.models.enrichment`), built by
`make_enrichment_provider`; `kanon-2-enricher` (`IsaacusEnrichment`) is the default.
A factory's `isaacus` and `returns_document` traits answer the pre-launch gates
(Isaacus availability, persisted Documents) before anything is built. AI chunking
follows the enrichment it reuses: with a provider that returns no Document it is
off (`ai_chunking_model`), and chunking is semchunk's local token splitting rather
than a paid call of its own.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

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
from womblex.utils.model_registry import SLOT_ENRICHMENT, record_use, register, resolve

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.config import ChunkingConfig, EnrichmentConfig, WomblexConfig
    from womblex.utils.token_packer import TokenCounter

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "kanon-2-enricher"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 2.0


def _retry_after_seconds(exc: object) -> float | None:
    """Extract a ``Retry-After`` delay (seconds) from an SDK exception, if any.

    The Isaacus SDK's ``RateLimitError`` carries the HTTP ``response``; its
    ``Retry-After`` header (delta-seconds) is the server's own backoff advice.
    Returns ``None`` when unavailable so the caller falls back to exponential.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
    except Exception:
        return None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Response conversion helpers
# ---------------------------------------------------------------------------


def _to_span(raw: object) -> Span | None:
    """Convert an SDK span object to our Span dataclass."""
    if raw is None:
        return None
    return Span(start=raw.start, end=raw.end)  # type: ignore[attr-defined]


def _to_span_list(raw_list: list[object] | None) -> list[Span]:
    """Convert a list of SDK span objects to our Span dataclasses."""
    if not raw_list:
        return []
    return [Span(start=s.start, end=s.end) for s in raw_list]  # type: ignore[attr-defined]


def _convert_segments(raw_segments: list[object]) -> list[Segment]:
    result: list[Segment] = []
    for seg in raw_segments:
        result.append(Segment(
            id=seg.id,  # type: ignore[attr-defined]
            kind=seg.kind,  # type: ignore[attr-defined]
            type=seg.type,  # type: ignore[attr-defined]
            category=seg.category,  # type: ignore[attr-defined]
            span=_to_span(seg.span),  # type: ignore[attr-defined, arg-type]
            parent=seg.parent,  # type: ignore[attr-defined]
            children=list(seg.children) if seg.children else [],  # type: ignore[attr-defined]
            level=seg.level,  # type: ignore[attr-defined]
            type_name=_to_span(seg.type_name),  # type: ignore[attr-defined]
            code=_to_span(seg.code),  # type: ignore[attr-defined]
            title=_to_span(seg.title),  # type: ignore[attr-defined]
        ))
    return result


def _convert_locations(raw: list[object]) -> list[Location]:
    result: list[Location] = []
    for loc in raw:
        result.append(Location(
            id=loc.id,  # type: ignore[attr-defined]
            name=_to_span(loc.name),  # type: ignore[attr-defined, arg-type]
            type=loc.type,  # type: ignore[attr-defined]
            mentions=_to_span_list(loc.mentions),  # type: ignore[attr-defined]
            parent=loc.parent,  # type: ignore[attr-defined]
            children=list(loc.children) if loc.children else [],  # type: ignore[attr-defined]
        ))
    return result


def _convert_persons(raw: list[object]) -> list[Person]:
    result: list[Person] = []
    for per in raw:
        result.append(Person(
            id=per.id,  # type: ignore[attr-defined]
            name=_to_span(per.name),  # type: ignore[attr-defined, arg-type]
            type=per.type,  # type: ignore[attr-defined]
            role=per.role,  # type: ignore[attr-defined]
            mentions=_to_span_list(per.mentions),  # type: ignore[attr-defined]
            parent=per.parent,  # type: ignore[attr-defined]
            children=list(per.children) if per.children else [],  # type: ignore[attr-defined]
            residence=per.residence,  # type: ignore[attr-defined]
        ))
    return result


def _convert_crossreferences(raw: list[object]) -> list[CrossReference]:
    return [
        CrossReference(
            start=xref.start,  # type: ignore[attr-defined]
            end=xref.end,  # type: ignore[attr-defined]
            span=_to_span(xref.span),  # type: ignore[attr-defined, arg-type]
        )
        for xref in raw
    ]


def _convert_terms(raw: list[object]) -> list[Term]:
    return [
        Term(
            id=t.id,  # type: ignore[attr-defined]
            name=_to_span(t.name),  # type: ignore[attr-defined, arg-type]
            meaning=_to_span(t.meaning),  # type: ignore[attr-defined, arg-type]
            mentions=_to_span_list(t.mentions),  # type: ignore[attr-defined]
        )
        for t in raw
    ]


def _convert_external_documents(raw: list[object]) -> list[ExternalDocument]:
    return [
        ExternalDocument(
            id=exd.id,  # type: ignore[attr-defined]
            name=_to_span(exd.name),  # type: ignore[attr-defined, arg-type]
            type=exd.type,  # type: ignore[attr-defined]
            reception=exd.reception,  # type: ignore[attr-defined]
            mentions=_to_span_list(exd.mentions),  # type: ignore[attr-defined]
            pinpoints=_to_span_list(exd.pinpoints),  # type: ignore[attr-defined]
            jurisdiction=exd.jurisdiction,  # type: ignore[attr-defined]
        )
        for exd in raw
    ]


def _convert_quotes(raw: list[object]) -> list[Quote]:
    return [
        Quote(
            span=_to_span(q.span),  # type: ignore[attr-defined, arg-type]
            amending=q.amending,  # type: ignore[attr-defined]
            source_segment=q.source_segment,  # type: ignore[attr-defined]
            source_document=q.source_document,  # type: ignore[attr-defined]
            source_person=q.source_person,  # type: ignore[attr-defined]
        )
        for q in raw
    ]


def _convert_dates(raw: list[object]) -> list[DateInfo]:
    return [
        DateInfo(
            value=d.value,  # type: ignore[attr-defined]
            type=d.type,  # type: ignore[attr-defined]
            mentions=_to_span_list(d.mentions),  # type: ignore[attr-defined]
            person=d.person,  # type: ignore[attr-defined]
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
            address=e.address, person=e.person,  # type: ignore[attr-defined]
            mentions=_to_span_list(e.mentions),  # type: ignore[attr-defined]
        )
        for e in raw_emails
    ]
    websites = [
        Website(
            url=w.url, person=w.person,  # type: ignore[attr-defined]
            mentions=_to_span_list(w.mentions),  # type: ignore[attr-defined]
        )
        for w in raw_websites
    ]
    phones = [
        PhoneNumber(
            number=p.number, person=p.person,  # type: ignore[attr-defined]
            mentions=_to_span_list(p.mentions),  # type: ignore[attr-defined]
        )
        for p in raw_phones
    ]
    ids = [
        IDNumber(
            number=i.number, person=i.person,  # type: ignore[attr-defined]
            mentions=_to_span_list(i.mentions),  # type: ignore[attr-defined]
        )
        for i in raw_ids
    ]
    return emails, websites, phones, ids


# ---------------------------------------------------------------------------
# SDK response â†’ EnrichmentResult
# ---------------------------------------------------------------------------


def _convert_document(doc: object) -> EnrichmentResult:
    """Convert an Isaacus SDK ILGS Document to our EnrichmentResult."""
    emails, websites, phones, ids = _convert_contact_info(
        doc.emails, doc.websites, doc.phone_numbers, doc.id_numbers,  # type: ignore[attr-defined]
    )
    return EnrichmentResult(
        text=doc.text,  # type: ignore[attr-defined]
        type=doc.type,  # type: ignore[attr-defined]
        jurisdiction=doc.jurisdiction,  # type: ignore[attr-defined]
        title=_to_span(doc.title),  # type: ignore[attr-defined]
        subtitle=_to_span(doc.subtitle),  # type: ignore[attr-defined]
        segments=_convert_segments(doc.segments),  # type: ignore[attr-defined]
        crossreferences=_convert_crossreferences(doc.crossreferences),  # type: ignore[attr-defined]
        locations=_convert_locations(doc.locations),  # type: ignore[attr-defined]
        persons=_convert_persons(doc.persons),  # type: ignore[attr-defined]
        emails=emails,
        websites=websites,
        phone_numbers=phones,
        id_numbers=ids,
        terms=_convert_terms(doc.terms),  # type: ignore[attr-defined]
        external_documents=_convert_external_documents(doc.external_documents),  # type: ignore[attr-defined]
        quotes=_convert_quotes(doc.quotes),  # type: ignore[attr-defined]
        dates=_convert_dates(doc.dates),  # type: ignore[attr-defined]
        headings=_to_span_list(doc.headings),  # type: ignore[attr-defined]
        junk=_to_span_list(doc.junk),  # type: ignore[attr-defined]
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
    overflow_strategy: str = "auto",
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
                            overflow_strategy=overflow_strategy,
                            max_retries=max_retries,
                            retry_base_delay=retry_base_delay)[0]


def enrich_document_raw(
    text: str,
    client: object,
    *,
    model: str = DEFAULT_MODEL,
    overflow_strategy: str = "auto",
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> tuple[EnrichmentResult, object]:
    """Like :func:`enrich_document` but also return the *raw* SDK Document.

    The raw ``isaacus.types.ilgs.v1.document.Document`` is the object semchunk
    4 accepts for AI chunking without re-enriching (``docs/decisions.md``). It
    is already in hand inside :func:`enrich_documents_raw` and normally
    discarded after conversion; this entry point hands it back so the enrich
    stage can persist it for chunk-stage reuse.
    """
    return enrich_documents_raw([text], client, model=model,
                                overflow_strategy=overflow_strategy,
                                max_retries=max_retries,
                                retry_base_delay=retry_base_delay)[0]


def enrich_documents(
    texts: list[str],
    client: object,
    *,
    model: str = DEFAULT_MODEL,
    overflow_strategy: str = "auto",
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> list[EnrichmentResult]:
    """Enrich one or more documents via the Isaacus enrichment API.

    Thin wrapper over :func:`enrich_documents_raw` that drops the raw Document.

    Args:
        texts: List of full document texts.
        client: An ``isaacus.Isaacus`` client instance.
        model: Enrichment model identifier.
        overflow_strategy: How Kanon-2 handles input beyond its context window
            ("auto"/"chunk"/"drop_end"/"null"). "auto" chunks + stitches into one
            prediction; returned span offsets still index the full source text.
        max_retries: Maximum retry attempts for rate-limit errors.
        retry_base_delay: Base delay in seconds for exponential backoff.

    Returns:
        List of EnrichmentResult objects, one per input text.

    Raises:
        RuntimeError: If enrichment fails after all retries.
    """
    return [
        result for result, _raw in enrich_documents_raw(
            texts, client, model=model, overflow_strategy=overflow_strategy,
            max_retries=max_retries, retry_base_delay=retry_base_delay,
        )
    ]


def enrich_documents_raw(
    texts: list[str],
    client: object,
    *,
    model: str = DEFAULT_MODEL,
    overflow_strategy: str = "auto",
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> list[tuple[EnrichmentResult, object]]:
    """Enrich documents, returning ``(EnrichmentResult, raw SDK Document)`` pairs.

    The single source of the retry/backoff logic; :func:`enrich_documents`
    and :func:`enrich_document` are thin views that drop the raw Document.
    Handles 429 rate-limit errors with exponential backoff.

    Raises:
        RuntimeError: If enrichment fails after all retries.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.enrichments.create(  # type: ignore[attr-defined]
                model=model,
                texts=texts,
                overflow_strategy=overflow_strategy,
            )
            # Convert SDK response to our models, keeping the raw Document.
            results: list[tuple[EnrichmentResult, object]] = []
            for r in response.results:
                results.append((_convert_document(r.document), r.document))

            logger.info(
                "Enriched %d document(s), usage: %d input tokens",
                len(results),
                response.usage.input_tokens,
            )
            return results

        except Exception as e:
            last_error = e
            error_str = str(e)

            # Retry on rate limits (429), honouring a Retry-After header when the
            # server sends one (adaptive throttle: back off exactly as told, else
            # exponential). Whole ~150-200K-token requests trigger these.
            if "429" in error_str or "rate" in error_str.lower():
                retry_after = _retry_after_seconds(e)
                delay = retry_after if retry_after is not None else retry_base_delay * (2 ** attempt)
                logger.warning(
                    "Rate limited on enrichment (attempt %d/%d), retrying in %.1fs%s",
                    attempt + 1, max_retries + 1, delay,
                    " (Retry-After)" if retry_after is not None else "",
                )
                time.sleep(delay)
                continue

            # Non-retryable error
            logger.error("Enrichment failed: %s", e)
            raise RuntimeError(f"Enrichment failed: {e}") from e

    raise RuntimeError(
        f"Enrichment failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Provider seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderLimits:
    """What a provider declares, so the stage packs and splits requests by it."""

    max_texts_per_request: int
    token_ceiling: int
    """A text over this many tokens is split on boundaries and offset-merged."""
    token_counter: TokenCounter
    needs_network: bool
    returns_document: bool
    """Whether `enrich` hands back a Document AI chunking can reuse."""


class EnrichmentProvider(Protocol):
    name: str
    limits: ProviderLimits

    def enrich(self, texts: list[str]) -> list[tuple[EnrichmentResult, object | None]]:
        """One `(result, document)` pair per text; `document` is `None` unless `returns_document`."""
        ...


@dataclass
class IsaacusEnrichment:
    """Kanon-2 enrichment through the Isaacus SDK; its limits come from `EnrichmentConfig`."""

    client: object
    config: EnrichmentConfig
    limits: ProviderLimits
    name: str = "isaacus"

    def enrich(self, texts: list[str]) -> list[tuple[EnrichmentResult, object | None]]:
        return list(enrich_documents_raw(
            texts, self.client,
            model=self.config.model,
            overflow_strategy=self.config.overflow_strategy,
            max_retries=self.config.max_retries,
            retry_base_delay=self.config.retry_base_delay,
        ))


def isaacus_provider(
    config: EnrichmentConfig, client: object, token_counter: TokenCounter | None = None,
) -> IsaacusEnrichment:
    """The Isaacus provider; *client* may be `None` where only the limits are read (`plan`)."""
    from womblex.utils.token_packer import TokenCounter

    return IsaacusEnrichment(client, config, ProviderLimits(
        max_texts_per_request=config.max_texts_per_request,
        token_ceiling=config.split_ceiling,
        token_counter=token_counter or TokenCounter(config.tokenizer),
        needs_network=True,
        returns_document=True,
    ))


def _make_isaacus(
    *, config: EnrichmentConfig, client: object = None, token_counter: TokenCounter | None = None,
    **_: object,
) -> IsaacusEnrichment:
    return isaacus_provider(config, client, token_counter)


register(SLOT_ENRICHMENT, DEFAULT_MODEL, _make_isaacus,
         traits={"isaacus": True, "returns_document": True})


def provider_traits(model: str) -> dict[str, object]:
    """*model*'s registered traits; empty for an unknown name, which the model check reports."""
    try:
        return resolve(SLOT_ENRICHMENT, model).traits
    except ValueError:
        return {}


def calls_isaacus(model: str) -> bool:
    """Whether the provider registered as *model* is served by an Isaacus deployment."""
    return bool(provider_traits(model).get("isaacus"))


def returns_document(model: str) -> bool:
    """Whether *model* hands back a Document that AI chunking can reuse."""
    return bool(provider_traits(model).get("returns_document"))


def make_enrichment_provider(
    config: EnrichmentConfig, client: object = None, token_counter: TokenCounter | None = None,
) -> EnrichmentProvider:
    """Build the provider `config.model` names; *client* is used only by an Isaacus provider."""
    entry = resolve(SLOT_ENRICHMENT, config.model)
    provider: EnrichmentProvider = entry.factory(
        config=config, client=client, token_counter=token_counter,
    )
    record_use(entry)
    return provider


def ai_chunking_model(config: WomblexConfig) -> str | None:
    """`chunking.chunking_model` when the enrichment provider returns a Document, else `None`.

    AI chunking reuses that Document; without one the chunker would call Isaacus itself,
    a paid call the configured provider did not ask for, so semchunk's token splitting runs.
    """
    model = config.chunking.chunking_model
    return model if model and returns_document(config.enrichment.model) else None


def effective_chunking(config: WomblexConfig) -> ChunkingConfig:
    """`config.chunking` as the chunk stage runs it, warning when AI chunking is set aside."""
    if config.chunking.chunking_model and ai_chunking_model(config) is None:
        logger.warning(
            "chunking.chunking_model=%r is set aside: enrichment provider %r returns no Document "
            "to reuse, so chunks come from semchunk's local token splitting",
            config.chunking.chunking_model, config.enrichment.model,
        )
        return config.chunking.model_copy(update={"chunking_model": None})
    return config.chunking
