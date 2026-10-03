"""Process-stage config models: normalise, spellfix, segmentation, quality, money."""

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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
    fidelity-neutral formatting only). Reads ``*.elements.parquet`` (chaining on
    top of the normalise layer when present) and writes a repaired
    ``*.spellfix_text.parquet`` element-text overlay plus a
    ``*.spellfix_corrections.parquet`` audit trail — the raw elements are never
    modified. Consumers opt in by setting ``processing.text_source='spellfix'``.
    Only out-of-dictionary tokens with a single unambiguous in-dictionary
    candidate are rewritten. See ``docs/decisions.md`` "Dictionary-gated OCR repair".
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


class SegmentationConfig(BaseModel):
    """Ground-truth segmentation of an element stream into reviewable units.

    Cuts a document's elements into contiguous ranges a human can correct in
    one sitting. Boundaries are a function of the element stream and these
    values alone — never of a stage's output — so a reviewed unit survives
    any change to chunking, cleaning or enrichment. See
    :mod:`womblex.process.segmenter`.
    """

    token_budget: int = Field(
        default=2000, ge=1,
        description="Maximum tokens per segment, counted by the caller's tokeniser. "
                    "A single element over this is emitted alone and flagged oversize — "
                    "there is no boundary inside an element to split at.",
    )
    page_ceiling: int = Field(
        default=5, ge=1,
        description="Maximum pages one segment may span. Applied before the token "
                    "budget, so a segment is bounded by whichever binds first. A source "
                    "with no page concept (DOCX, spreadsheet) is bounded by the budget alone.",
    )
    oversize: Literal["flag", "error"] = Field(
        default="flag",
        description="What to do with an element that exceeds token_budget on its own: "
                    "'flag' emits it as a solo segment marked oversize; 'error' refuses "
                    "to segment the document.",
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


class MoneyColumnsConfig(BaseModel):
    """Column-evidenced half of the money op — bare cells in a money column."""

    enabled: bool = Field(default=True, description="Classify table/sheet columns.")
    numeric_fraction_min: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum fraction of non-null cells that must parse as numbers "
                    "before a header can promote a column. Null markers (—, n/a, nil) "
                    "are absent values and are excluded from the denominator, not "
                    "counted against it.",
    )
    min_cells: int = Field(
        default=3, ge=1,
        description="Minimum non-null cells before header evidence is trusted.",
    )
    extra_header_terms: list[str] = Field(
        default_factory=list,
        description="Corpus-specific money header vocabulary, added to the built-in set.",
    )
    extra_veto_terms: list[str] = Field(
        default_factory=list,
        description="Corpus-specific header terms that suppress a column (whole-word).",
    )


class MoneyConfig(BaseModel):
    """Monetary amount annotation op (``womblex money``).

    Reads ``*.elements.parquet`` + ``*.table_cells.parquet`` and writes
    ``*.money_spans.parquet`` + ``*.money_columns.parquet`` sidecars. Offline,
    API-free, annotation only — element and chunk text are never rewritten.
    Amounts are recovered along two paths: self-evidencing (a symbol, ISO code
    or currency word sits with the number) and column-evidenced (a bare number
    whose money-ness comes from its column's header or number format). See
    ``docs/money-extraction.md``.
    """

    enabled: bool = Field(default=True, description="Run the money stage.")
    narrative: bool = Field(
        default=True, description="Scan reassembled narrative text (self-evidencing path).",
    )
    default_currency: str = Field(
        default="AUD",
        description="Currency assumed where a document states none. Australian "
                    "government publications use `$` to mean AUD unless another "
                    "currency is explicitly established.",
    )
    international_numbers: bool = Field(
        default=False,
        description="Accept continental formats (1.000,50). Off by default: "
                    "Australia does not use comma decimals, and inferring locale "
                    "adds false positives for no benefit on this corpus.",
    )
    implicit_context: bool = Field(
        default=False,
        description="Pattern 10 — bare numbers near financial trigger vocabulary in "
                    "narrative text. Low precision on this corpus; opt in for recall "
                    "experiments only. Header vocabulary (the high-value use of the "
                    "same terms) is unaffected by this flag.",
    )
    min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Drop narrative candidates scoring below this.",
    )
    context_chars: int = Field(
        default=160, ge=0,
        description="Characters of surrounding text stored with each narrative span.",
    )
    text_source: str | None = Field(
        default=None,
        description="Element-text layer the narrative offsets index. Null inherits "
                    "processing.text_source, which is what keeps money spans in the "
                    "same coordinate space as enrichment mentions and chunks.",
    )
    columns: MoneyColumnsConfig = MoneyColumnsConfig()

    @field_validator("default_currency")
    @classmethod
    def _check_currency(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Z]{3}", v):
            raise ValueError(f"default_currency must be a 3-letter ISO 4217 code, got {v!r}")
        return v

    @field_validator("text_source")
    @classmethod
    def _check_text_source(cls, v: str | None) -> str | None:
        if v is not None and v not in ("elements", "normalised", "spellfix"):
            raise ValueError(f"text_source must be elements|normalised|spellfix, got {v!r}")
        return v
