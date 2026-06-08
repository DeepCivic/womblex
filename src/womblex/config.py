"""Configuration loading and validation for womblex pipelines."""


from pathlib import Path

from typing import Any


import yaml

from pydantic import BaseModel, Field, model_validator



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

    """OCR engine settings.

    Supported engines:

    - ``paddleocr`` (default): local rapidocr-onnxruntime, returns regions.
    - ``deepseek-ocr``: local DeepSeek-OCR via an OpenAI-compatible
      endpoint (e.g. Ollama at ``OLLAMA_BASE_URL``). Returns markdown
      with native reading order.

    ``engine_options`` forwards engine-specific kwargs:

    - DeepSeek-OCR: ``model`` (default ``deepseek-ocr:3b``),
      ``base_url`` (default from ``OLLAMA_BASE_URL`` env or
      ``http://localhost:11435/v1``), ``prompt`` (default
      ``"<image>\\nFree OCR."``).
    """

    engine: str = "paddleocr"

    dpi: int = Field(default=200, ge=72, le=600)
    lang: str = "eng"
    engine_options: dict = Field(default_factory=dict)
    num_threads: int = Field(
        default=4, ge=1,
        description="Cap on OCR (onnxruntime) + layout (torch) inference threads. "
                    "Prevents the two engines each grabbing every core "
                    "(oversubscription); keep low for Chromebook-class targets, "
                    "raise on many-core servers. Also settable via "
                    "WOMBLEX_INFERENCE_THREADS.",
    )



class RedactionConfig(BaseModel):

    """Redaction pipeline settings.


    Redaction runs as a separate pipeline stage after extraction.

    It renders PDF pages as images, detects black-box regions, and

    applies the configured mode to affected page text.


    Modes:

    - ``flag``:    Mark records/chunks that overlap redacted regions (no text change).

    - ``blackout``: Replace affected page text with ``<REDACTED>`` markers.

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

    use_layout_filter: bool = Field(
        default=True,
        description=(
            "On raster-fallback pages, run YOLO layout analysis and drop "
            "contour hits inside figure / chart / form-background regions. "
            "Suppresses 02737-class scanned_mixed false positives. "
            "Best-effort: no-op if ultralytics is unavailable."
        ),
    )



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

    use_regex_backstop: bool = Field(

        default=False,

        description=(

            "Run the local regex+context detector alongside the enrichment "
            "graph spans. Default False: the Kanon-2 graph is the high-precision "
            "entity source; the regex/context backstop is noisy on this corpus "
            "(~15% precision — orgs/headings tagged PERSON), so it is opt-in for "
            "recall experiments only."

        ),

    )

    write_clean_text: bool = Field(

        default=True,

        description=(

            "Also write the masked `*.clean_text.parquet` sidecar (the "
            "publishable text layer) alongside `*.pii_spans.parquet`. Spans are "
            "replaced with typed+numbered tags (`<PERSON_1>`, …) keyed to the "
            "graph entity. Set False for a spans-only (measurement) run."

        ),

    )



class SpreadsheetPrintConfig(BaseModel):
    """Spreadsheet-printed-to-PDF extractor settings.

    Triggered when a doc has a native text layer + table signal + either a
    filename matching one of `filename_hints` or table signal on ≥50 % of
    pages. Captures a single multi-page TableData with row-by-row data and
    a metadata block (the label-value fields above the first data row).
    """

    metadata_location: str = "both"  # "both" | "table" | "document"
    filename_hints: list[str] = [
        "schedule", "index", "manifest", "register",
        "list-of", "table-of", "appendix",
    ]


class NativeExtractionConfig(BaseModel):

    """Native text extraction settings."""


    include_tables: bool = True
    spreadsheet_print: SpreadsheetPrintConfig = SpreadsheetPrintConfig()



class ExtractionConfig(BaseModel):

    """Top-level extraction settings."""


    native: NativeExtractionConfig = NativeExtractionConfig()

    ocr: OCRConfig = OCRConfig()



class ChunkingConfig(BaseModel):

    """Chunking configuration for semchunk.

    Thin pass-through to semchunk 3.x — every field below either maps
    directly to a semchunk parameter or is a Womblex-only integration
    concern semchunk can't own. There are no Womblex toggles that
    re-expose a semchunk feature under a different name.

    Maps to ``semchunk.chunkerify`` (creation-time): ``tokenizer``
    (→ ``tokenizer_or_token_counter``), ``chunk_size``, ``memoize``,
    ``cache_maxsize``, ``max_token_chars``.

    Maps to ``semchunk.Chunker.__call__`` (per-call): ``overlap``,
    ``processes``, ``progress``. (``offsets`` is pinned ``True`` in the
    adapter — Womblex always needs char offsets for page mapping.)

    Womblex-only (no semchunk equivalent): ``enabled`` (stage gate),
    ``chunk_tables`` (element-stream → markdown projection).

    Default divergences from semchunk upstream, each with a corpus
    reason: ``tokenizer="isaacus/kanon-2-tokenizer"`` matches the
    analysis side; ``chunk_size=480`` is the Kanon-2 window (upstream
    defaults to ``None`` = auto-derive from the tokeniser's
    ``model_max_length``, which this field still accepts as a
    pass-through); ``processes=1`` keeps single-thread Chromebook
    portability.

    The Kanon-2 tokeniser is obtainable only through the Isaacus API, so
    the chunk stage is gated on it: when the Isaacus SDK + ``ISAACUS_API_KEY``
    aren't available the stage logs and skips (see
    ``womblex.utils.availability.isaacus_available``). Chunking therefore
    runs alongside the rest of the Isaacus-dependent analysis, not offline.
    """


    tokenizer: str = "isaacus/kanon-2-tokenizer"

    chunk_size: int | None = Field(
        default=480,
        ge=1,
        description=(
            "Maximum tokens per chunk. None passes through to semchunk, "
            "which derives the size from the tokeniser's model_max_length. "
            "Defaults to 480 (the Kanon-2 window) rather than upstream's "
            "None auto-derive — see class docstring."
        ),
    )

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

    progress: bool = Field(
        default=False,
        description="Show a tqdm progress bar during chunking.",
    )



class NormaliseConfig(BaseModel):
    """Downstream text-cleaning op (``womblex normalise``).

    Applies verbatim-policy-respecting cleanup *after* extraction and writes
    a ``*.normalised_text.parquet`` text layer over the narrative elements.
    Each toggle maps to a pure transform in :mod:`womblex.process.normalise`.
    """

    unicode_hygiene: bool = Field(
        default=True,
        description="Fold unicode whitespace (NBSP, en/em spaces, ideographic "
                    "space, U+2028/9 separators) to ASCII space/newline and strip "
                    "zero-width marks, BOM and stray control chars. Smart quotes "
                    "and em/en dashes are preserved.",
    )
    collapse_whitespace: bool = Field(
        default=True,
        description="Collapse inline space/tab runs to one and strip per-line "
                    "trailing whitespace (newlines preserved).",
    )
    despace_page_marker: bool = Field(
        default=True,
        description="Heal sub-glyph-kerning '3|P age' footers back to '3|Page' "
                    "(footer/header kinds only).",
    )
    substitutions: dict[str, str] = Field(
        default_factory=dict,
        description="Literal {find: replace} fixes for known letterhead / font-map "
                    "typos. Empty by default — corpus-driven, never hardcoded in core.",
    )


class SpellfixConfig(BaseModel):
    """Dictionary-gated OCR character-confusion repair (``womblex spellfix``).

    A separate, opt-in cleaning op (distinct from ``normalise``, which is
    fidelity-neutral formatting only). Reads ``*.chunks.parquet`` and writes a
    repaired ``*.chunks_repaired.parquet`` layer plus a ``*.spellfix_corrections.
    parquet`` audit trail — the raw chunks are never modified. Only out-of-
    dictionary tokens with a single unambiguous in-dictionary candidate are
    rewritten. See ``docs/decisions.md`` "Dictionary-gated OCR repair".
    """

    enabled: bool = Field(default=False, description="Run the spellfix stage.")
    general_edits: bool = Field(
        default=False,
        description="Tier B: enable general edit-distance-1 candidates "
                    "(insert/delete/substitute/transpose) in addition to the default "
                    "Tier A digit→letter homoglyph swaps. Higher recall but carries a "
                    "proper-noun corruption risk — opt-in.",
    )
    dict_name: str = Field(
        default="en_AU",
        description="Hunspell dictionary name resolved via utils.models "
                    "(bundled under `_models/en_AU`).",
    )


class QualityConfig(BaseModel):
    """Chunk-quality annotation op (``womblex quality``).

    Reads ``*.chunks.parquet`` and writes a ``*.chunk_quality.parquet`` sidecar
    (joined on ``source_hash``/``chunk_index``) with ML-readiness flags and
    cross-batch duplicate cluster ids. Annotation only — never mutates chunks.
    """

    enabled: bool = Field(default=True, description="Run the quality stage.")
    short_chars: int = Field(
        default=50, ge=1,
        description="char_len below this marks `is_short` (footer/page-number noise).",
    )
    boilerplate_patterns: list[str] = Field(
        default_factory=list,
        description="Regexes flagging boilerplate (letterhead footer, scope text). "
                    "Empty by default — corpus-driven, never hardcoded in core.",
    )
    dedup: bool = Field(default=True, description="Compute exact_dup_id / near_dup_id.")
    minhash_permutations: int = Field(default=64, ge=8)
    minhash_bands: int = Field(
        default=4, ge=1,
        description="LSH bands; with N permutations the near-dup Jaccard threshold "
                    "is ~ (1/bands)**(bands/N). 4 bands / 64 perms ≈ 0.92.",
    )
    shingle_words: int = Field(default=5, ge=1, description="Word-shingle size for MinHash.")

    @model_validator(mode="after")
    def _bands_divide_permutations(self) -> "QualityConfig":
        if self.minhash_permutations % self.minhash_bands != 0:
            raise ValueError(
                f"minhash_bands ({self.minhash_bands}) must divide "
                f"minhash_permutations ({self.minhash_permutations}); otherwise "
                "trailing permutations are silently unused."
            )
        return self


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

    run_id: str | None = Field(
        default=None,
        description=(
            "Identifier for this run instance. Multiple runs co-exist under "
            "<output_root>/<run_id>/documents/. If None, an ISO timestamp "
            "(run-YYYYMMDDTHHMMSSZ) is generated when the run starts."
        ),
    )



class RetentionConfig(BaseModel):
    """Run-output retention policy.

    Controls whether older run directories under ``<output_root>/`` are
    auto-purged when a new run starts. The current run is always preserved.
    """

    policy: str = Field(
        default="rolling",
        description=(
            "rolling = keep `keep` most-recent runs (including current), "
            "purge older. keep_all = no auto-purge; user manages purges manually."
        ),
    )
    keep: int = Field(
        default=2, ge=1,
        description="Number of runs to retain under `rolling`. Ignored when policy=keep_all.",
    )


class ProcessingConfig(BaseModel):

    """Batch processing settings."""


    batch_size: int = Field(default=100, ge=1)

    checkpoint_every: int = Field(default=100, ge=1)

    retention: RetentionConfig = RetentionConfig()


class ReferenceConfig(BaseModel):
    """Declares how a corpus reference register maps onto the generic matcher.

    The library knows nothing about specific registers; the corpus declares
    which columns play which role. The matcher resolves a document mention
    to a canonical entity via:

    - ``match_exact_cols`` — normalised equality = definitive match
      (confidence 1.0). For ACT childcare this is the service address
      columns; concatenated + normalised, it survives OCR noise on names.
    - ``match_fuzzy_cols`` — difflib similarity; best ``>= name_threshold``
      matches. Typically the legal/trading/service name columns.
    """

    path: Path = Field(description="Reference table file (CSV for v1).")
    format: str = Field(default="csv", description="Reference format. Only 'csv' implemented.")
    id_col: str = Field(description="Column holding the canonical entity id (e.g. SE-/PR-).")
    name_col: str = Field(description="Column holding the canonical display name.")
    entity_type: str = Field(
        default="entity",
        description="Constant entity_type tag for matches (e.g. 'service').",
    )
    parent_id_col: str | None = Field(
        default=None, description="Optional hierarchy FK column (e.g. provider id of a service).",
    )
    match_exact_cols: list[str] = Field(
        default_factory=list,
        description="Columns concatenated+normalised for definitive equality matching.",
    )
    match_fuzzy_cols: list[str] = Field(
        default_factory=list, description="Columns scored by normalised fuzzy similarity.",
    )
    alias_table: Path | None = Field(
        default=None,
        description=(
            "Optional CSV of corpus-curated alias -> entity_id overrides for "
            "entities the register doesn't carry (e.g. prior trustees). "
            "Columns: alias, entity_id."
        ),
    )


class LinkingConfig(BaseModel):
    """Entity-link stage settings. Generic; corpus supplies the reference."""

    enabled: bool = Field(default=False, description="Run the entity-link stage")
    reference: ReferenceConfig | None = Field(
        default=None, description="Reference register mapping (required when enabled).",
    )
    candidate_kinds: list[str] = Field(
        default_factory=lambda: ["corporate", "address"],
        description=(
            "Enrichment entity_type values treated as link candidates. "
            "Pass-through to the Kanon-2 taxonomy — corporate persons + "
            "address locations by default."
        ),
    )
    name_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Minimum normalised fuzzy similarity for a name match.",
    )


class EmbeddingConfig(BaseModel):
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
    max_retries: int = Field(default=3, ge=0)
    retry_base_delay: float = Field(default=2.0, ge=0.0)


class WomblexConfig(BaseModel):
    """Complete configuration for Womblex operations."""

    dataset: DatasetConfig
    paths: PathsConfig
    detection: DetectionConfig = DetectionConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    redaction: RedactionConfig = RedactionConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    normalise: NormaliseConfig = NormaliseConfig()
    spellfix: SpellfixConfig = SpellfixConfig()
    quality: QualityConfig = QualityConfig()
    enrichment: EnrichmentConfig = EnrichmentConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    linking: LinkingConfig = LinkingConfig()
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

