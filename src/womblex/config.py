"""Configuration loading and validation for womblex pipelines."""


from pathlib import Path

from typing import Any


import yaml

from pydantic import BaseModel, Field



class PathsConfig(BaseModel):

    """Filesystem paths for input, output, and checkpoints."""


    input_root: Path

    output_root: Path

    checkpoint_dir: Path



class DetectionConfig(BaseModel):

    """Thresholds for document type detection."""


    min_text_coverage: float = Field(default=0.3, ge=0.0, le=1.0)

    form_signal_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    table_signal_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    max_sample_pages: int = Field(default=5, ge=1, description="Max pages to sample for classification")



class OCRConfig(BaseModel):

    """OCR engine settings."""

    engine: str = "paddleocr"

    dpi: int = Field(default=200, ge=72, le=600)
    lang: str = "eng"



class RedactionConfig(BaseModel):

    """Redaction pipeline settings.


    Redaction runs as a separate pipeline stage after extraction.

    It renders PDF pages as images, detects black-box regions, and

    applies the configured mode to affected page text.


    Modes:

    - ``flag``:    Mark records/chunks that overlap redacted regions (no text change).

    - ``blackout``: Replace affected page text with ``[REDACTED]`` markers.

    - ``delete``:   Remove affected page text entirely.
    """


    enabled: bool = True

    mode: str = Field(

        default="flag",

        description="Redaction mode: flag | blackout | delete",

    )

    threshold: int = Field(default=50, ge=0, le=255, description="Pixel darkness threshold for detection")

    min_area_ratio: float = Field(default=0.001, ge=0.0, le=1.0)

    max_area_ratio: float = Field(default=0.9, ge=0.0, le=1.0)

    dpi: int = Field(default=150, ge=72, le=600, description="DPI for rendering pages during detection")



class PIIConfig(BaseModel):

    """PII cleaning pipeline settings.


    PII cleaning runs as a separate pipeline stage using regex pattern

    recognisers (Presidio-style) validated by a Sentence Transformers

    context model (all-MiniLM-L6-v2).


    Pipeline points:

    - ``post_extraction``: Clean page texts before chunking.

    - ``post_chunk``:      Clean individual chunk texts after chunking.

    - ``post_enrichment``: Clean chunk texts using Isaacus graph entities

      as high-confidence candidates, supplemented by regex detection.

      Requires enrichment to have run first (chunks and enrichment must

      exist on the DocumentResult).


    Requires ``pip install womblex[pii]``.
    """


    enabled: bool = Field(default=False, description="Run PII cleaning stage")

    entities: list[str] = Field(

        default=["PERSON"],

        description="Entity types to detect and replace",

    )

    person_types: list[str] = Field(

        default=["natural"],

        description=(

            "Enrichment person types to treat as PII. "

            "Values: natural, corporate, politic. "

            "Only applies to post_enrichment pipeline point."

        ),

    )

    pipeline_point: str = Field(

        default="post_chunk",

        description="When to run: post_extraction | post_chunk | post_enrichment",

    )

    context_similarity_threshold: float = Field(

        default=0.35, ge=0.0, le=1.0,

        description="Cosine similarity cutoff for low-confidence candidate validation",

    )

    model: str = Field(

        default="all-MiniLM-L6-v2",

        description="Sentence Transformers model for context validation",

    )



class NativeExtractionConfig(BaseModel):

    """Native text extraction settings."""


    include_tables: bool = True



class ExtractionConfig(BaseModel):

    """Top-level extraction settings."""


    native: NativeExtractionConfig = NativeExtractionConfig()

    ocr: OCRConfig = OCRConfig()



class ChunkingConfig(BaseModel):

    """Chunking configuration for semchunk.


    All semchunk v3+ parameters are exposed. Creation-time parameters

    (``memoize``, ``max_token_chars``, ``cache_maxsize``)
    are set when building the chunker.

    Per-call parameters (``overlap``, ``processes``, ``batch``, ``progress``)
    are passed at chunk time.
    """


    tokenizer: str = "isaacus/kanon-2-tokenizer"

    chunk_size: int = Field(default=480, ge=1)

    enabled: bool = Field(default=True, description="Run chunking stage")

    chunk_tables: bool = Field(default=True, description="Convert tables to markdown and chunk separately")

    overlap: int | float | None = Field(

        default=None,

        description="Boundary context sharing. <1 = proportion of chunk_size, >=1 = absolute tokens. None = no overlap.",

    )

    memoize: bool = Field(default=True, description="Cache token counts for repeated substrings")

    cache_maxsize: int | None = Field(
        default=None,
        description="Upper bound on memoization cache entries. None = unbounded.",
    )

    max_token_chars: int | None = Field(

        default=None,

        description="Max chars per token estimate — optimises token counting for long inputs",

    )

    processes: int = Field(

        default=1, ge=1,

        description="Parallel chunking workers. Default 1 (single-threaded, suitable for Chromebook deployment).",

    )

    batch: bool = Field(
        default=False,
        description=(
            "Collect all texts (narrative + tables) and chunk in one call. "
            "Enables real multiprocessing when processes > 1."
        ),
    )

    progress: bool = Field(
        default=False,
        description="Show a tqdm progress bar during chunking.",
    )



class EnrichmentConfig(BaseModel):

    """Isaacus enrichment settings."""


    enabled: bool = Field(default=False, description="Run enrichment stage")

    model: str = Field(default="kanon-2-enricher", description="Isaacus enrichment model")

    max_retries: int = Field(default=3, ge=0, description="Max retries for rate-limit errors")

    retry_base_delay: float = Field(default=2.0, ge=0.0, description="Base delay for exponential backoff")

    batch_size: int = Field(default=10, ge=1, description="Documents per enrichment batch")

    skip_short_documents: int = Field(

        default=0, ge=0,

        description="Skip enrichment for documents shorter than this many characters (0 = enrich all)",

    )



class DatasetConfig(BaseModel):

    """Dataset metadata."""

    name: str



class ProcessingConfig(BaseModel):

    """Batch processing settings."""


    batch_size: int = Field(default=100, ge=1)

    checkpoint_every: int = Field(default=100, ge=1)


class WomblexConfig(BaseModel):
    """Complete configuration for Womblex operations."""

    dataset: DatasetConfig
    paths: PathsConfig
    detection: DetectionConfig = DetectionConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    redaction: RedactionConfig = RedactionConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    enrichment: EnrichmentConfig = EnrichmentConfig()
    pii: PIIConfig = PIIConfig()
    processing: ProcessingConfig = ProcessingConfig()


def load_config(path: Path) -> WomblexConfig:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated WomblexConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        pydantic.ValidationError: If the config does not match the schema.
    """
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return WomblexConfig(**raw)

