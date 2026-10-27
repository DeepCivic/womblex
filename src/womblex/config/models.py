"""Model-slot config: the OCR engine, enrichment and embedding settings."""

from typing import Any

from pydantic import Field, field_validator, model_validator

from womblex.config.process import StrictModel


def _reject_removed(data: Any, section: str) -> Any:
    """Refuse the per-consumer layout keys that moved to the top-level ``layout:``.

    Ignoring them would run the default model in place of the one the config names.
    """
    if isinstance(data, dict):
        for key in ("layout_model", "layout_options"):
            if key in data:
                raise ValueError(
                    f"{section}.{key} was removed; use layout.{key.removeprefix('layout_')}"
                )
    return data


class OCRConfig(StrictModel):

    """OCR engine settings.

    Supported engines:

    - ``paddleocr`` (default): local rapidocr-onnxruntime, returns regions.
    - ``mistral-ocr``: Mistral's Pixtral Large VLM inferenced via AWS
      Bedrock (Converse API). Returns markdown with native reading order.
    - ``ollama``: local multimodal LLM via an OpenAI-compatible endpoint
      (Ollama at ``OLLAMA_BASE_URL``). Returns markdown with native
      reading order.

    ``engine_options`` forwards engine-specific kwargs:

    - PaddleOCR: ``use_cls`` (default ``True``, as RapidOCR), RapidOCR's
      per-line 180-degree text classifier.
    - Mistral OCR: ``model`` (default ``mistral.pixtral-large-2502-v1:0``,
      or ``MISTRAL_OCR_MODEL_ID`` env), ``region`` (default from
      ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` env or ``us-east-1``). AWS
      credentials resolve via the standard boto3 chain.
    - Ollama: ``model`` (default ``llama3.2-vision``), ``base_url``
      (default from ``OLLAMA_BASE_URL`` env or
      ``http://localhost:11435/v1``), ``prompt``.
    """

    engine: str = "paddleocr"

    dpi: int = Field(default=200, ge=72, le=600)
    lang: str = "eng"
    engine_options: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _no_layout_keys(cls, data: Any) -> Any:
        return _reject_removed(data, "extraction.ocr")

    num_threads: int = Field(
        default=4, ge=1,
        description="Cap on OCR (onnxruntime) + layout (torch) inference threads. "
                    "Prevents the two engines each grabbing every core "
                    "(oversubscription); keep low for Chromebook-class targets, "
                    "raise on many-core servers. Also settable via "
                    "WOMBLEX_INFERENCE_THREADS.",
    )


class EnrichmentConfig(StrictModel):

    """Isaacus enrichment settings."""


    enabled: bool = Field(default=False, description="Run enrichment stage")

    model: str = Field(default="kanon-2-enricher", description="Isaacus enrichment model")

    overflow_strategy: str = Field(
        default="auto",
        description="How Kanon-2 handles documents exceeding its 16k-token context: "
                    "'auto'/'chunk' chunk internally and stitch back into one prediction "
                    "(offsets still index the full source); 'drop_end' truncates; 'null' "
                    "errors. Pass-through to enrichments.create. Defaults to 'auto' (vs "
                    "upstream 'null') because FOI bundles routinely exceed 16k tokens.",
    )

    @field_validator("overflow_strategy")
    @classmethod
    def _check_overflow(cls, v: str) -> str:
        if v not in ("auto", "chunk", "drop_end", "null"):
            raise ValueError(f"overflow_strategy must be auto|chunk|drop_end|null, got {v!r}")
        return v

    max_retries: int | None = Field(
        default=None, ge=0,
        description="The Isaacus SDK's own retry count (429, 5xx and timeouts, honouring "
                    "Retry-After); null keeps the SDK default. Womblex adds no retry of its own.",
    )

    batch_size: int = Field(default=10, ge=1, description="Documents per enrichment batch")

    tokenizer: str = Field(
        default="isaacus/kanon-2-tokenizer",
        description="HuggingFace tokeniser id for exact local token counting when "
                    "packing token-budgeted requests (free on Hugging Face).",
    )

    max_texts_per_request: int = Field(
        default=8, ge=1,
        description="API doc-count ceiling per enrichment request (Isaacus max is 8). "
                    "Requests pack to min(max_texts_per_request, token_budget).",
    )

    token_budget: int = Field(
        default=32768, ge=1,
        description="Per-request token budget (B). Docs are packed so a request's "
                    "combined tokens stay within this; a doc over it is sent solo. "
                    "Rate limits bind on tokens/request — start ~32K and probe at T0.",
    )

    split_ceiling: int | None = Field(
        default=None, ge=1,
        description="Split a solo document above this token count client-side on "
                    "structural (blank-line) boundaries into <= split_ceiling segments, "
                    "enriched separately and offset-merged. Null (the default) splits "
                    "nothing: the API takes long input whole under overflow_strategy.",
    )

    include_tables: bool = Field(
        default=True,
        description="Also send each table's markdown to the enricher, separately from "
                    "the narrative, so table chunks have PII candidates of their own. "
                    "Adds Kanon-2 token spend in proportion to table text; tables ignore "
                    "skip_short_documents.",
    )

    skip_short_documents: int = Field(

        default=0, ge=0,

        description="Skip enrichment for documents shorter than this many characters (0 = enrich all)",

    )

    persist_document: bool = Field(
        default=False,
        description=(
            "Persist the raw ILGS Document per doc to *.enrichment_doc.parquet so "
            "the chunk stage reuses it for semchunk-4 AI chunking without "
            "re-enriching (docs/decisions-dead-ends.md). Off by default (large blob); "
            "auto-enabled by WomblexConfig when chunking.chunking_model is set."
        ),
    )


class EmbeddingConfig(StrictModel):
    """Isaacus embedding settings (kanon-2-embedder). Thin pass-through."""

    enabled: bool = Field(default=False, description="Run the embed stage")
    model: str = Field(default="kanon-2-embedder", description="Isaacus embedding model")
    task: str | None = Field(
        default="retrieval/document",
        description="Embedding task: retrieval/document (index) | retrieval/query | null.",
    )
    dimensions: int | None = Field(
        default=None, description="Optional output dimensionality (model default if null).",
    )
    max_retries: int | None = Field(
        default=None, ge=0, description="The Isaacus SDK's own retry count; null keeps the SDK default.",
    )
