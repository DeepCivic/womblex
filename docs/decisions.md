# Design decisions, dead-ends & known limitations

The durable "why" behind the library: decisions and their rejected
alternatives, approaches that were tried and abandoned (so they aren't
re-attempted), library-general limitations, and the deferred backlog. Current
*state* (what's built, what shipped) lives in [CHANGELOG.md](../CHANGELOG.md);
conventions live in [CLAUDE.md](../CLAUDE.md); the canonical schema reference is
[extraction.md](extraction.md). Corpus-specific run history and numbers are a
corpus concern, not a library one.

> This file is intentionally corpus-agnostic. It records mechanisms and
> rationale, not dataset counts or run identifiers.

## Pipeline shape — per-stage sidecars

Each stage reads the prior stage's persisted sidecar and writes its own, all
joinable on `source_hash` (+ `chunk_index` at chunk grain). A stage invoked on
its own must not depend on which stages ran before — only on what is on disk.
The extraction shard (`*.elements.parquet` + typed sidecars) is **verbatim and
never rewritten**; every downstream mutation (chunks, enrichment, links,
embeddings, PII spans, masked text) is a separate sibling parquet. Each
`*-stage` CLI (`chunk`/`redact`/`enrich`/`link`/`embed`/`pii`) has an
independent `CheckpointManager` and a resume-time integrity scan. Per-stage and
E2E (`womblex run`) modes feed the same engines by construction.

## Key design decisions

### Entity linking — Kanon-2 candidates + register matching
Kanon-2 enrichment is a strong candidate *generator* but does not canonicalise
(one real entity surfaces under many surface forms — legal/trading variants,
OCR-corrupted forms, and occasionally a different legal entity). So raw
enrichment output cannot be the entity graph on its own. The chosen approach is
**(b)**: collect Kanon-2 `corporate` persons + `address` locations as
candidates and resolve them against a corpus-declared reference register via a
generic record-linkage matcher (`link/matcher.py`): alias → address-exact →
OCR-tolerant token-set name-fuzzy (stdlib `difflib`, no rapidfuzz). The
**structured address is the most robust key** (immune to OCR noise on names);
normalised name is the fallback. Rejected: (a) raw Kanon-2 as the graph — no
canonical id, no variant unification; (c) rebuild NER from scratch — discards
Kanon-2's candidate/address signal.

The matcher and reference consumption are **generic** (a link is a resolved
attribution of a document mention to a reference entity; `entity_type` is a free
value, no domain columns). The corpus declares which register columns play
which role (`linking`/`reference` config).

### PII — graph-driven detection, masking after Isaacus
PII detection **is** the enrichment graph: select PII-typed entities (`natural`
→ PERSON, `address` → ADDRESS), map their mention offsets onto chunks, mask.
There is no separate detector and **no second enrichment pass**.

- **Detection runs on chunks; the graph is the entity source.** Graph mentions
  (full-narrative offsets) map into `narrative` chunks via `chunk.start_char`.
  `table` chunks carry a different offset space, so narrative graph spans are
  never applied to them.
- **Recall is flexed by enrichment *duration*, not by a second detector.**
  Higher recall comes from finer enrichment granularity (e.g. per-chunk) and
  `overflow_strategy` for long documents — both trade compute for recall while
  holding the model's precision. A local regex/context backstop exists but is
  **opt-in** (`pii.use_regex_backstop`, default off): it is low precision
  (title-case headings, organisation names, sign-offs caught as PERSON) and
  buys recall by sacrificing precision. The graph is the high-precision floor.
- **Masking is terminal — applied *after* enrich + embed.** The enricher strips
  `<…>` tags as OCR noise, so masking before enrichment is self-defeating (it
  deletes the entity the graph must find, and the tag wouldn't survive). The
  `pii` stage writes a masked `*.clean_text.parquet` sidecar; it never rewrites
  the raw chunks that feed Isaacus. Embeddings are computed on raw text and are
  treated as an internal substrate; if embeddings are ever published, re-embed
  the masked `clean_text` instead.

### PII / redaction marker convention
House style is **Presidio-style typed angle-bracket tags**. PII person masks are
`<PERSON_n>` — typed **and** numbered, keyed to the graph entity (within-doc
coreference), so distinct people stay distinct (preserves retrieval/clustering
utility) while the same person is consistent. This beats a flat `<PERSON>`
(collapses distinct people) and beats realistic surrogate names (re-identification
/ optics risk); full removal is for human disclosure, not an analysis corpus —
it breaks downstream utility. Source redactions use `<REDACTED>`. Standards
reviewed: OAIC de-identification framework + Privacy Act, NIST SP 800-188 / IR
8053, HIPAA Safe Harbor, Microsoft Presidio operators, de-identification
format-consistency / surrogation utility studies, the Text Anonymization
Benchmark.

**Entity scope:** PERSON is the default. ADDRESS is opt-in — a service/business
address is not personal information (and is often a record-linkage key), so a
blanket address mask is usually wrong; distinguish residential from business
addresses before enabling it.

### Redaction detection — vector-first, raster fallback
`redact/stage.py` enumerates `page.get_drawings()` for near-black filled
rectangles first (native PDFs; no area threshold) and falls back to raster
contour detection on scanned pages. Filters surfaced during validation:
near-black RGB/CMYK fill; `min_width ≥ 3pt` (excludes narrow column
separators); `min_height ≥ 8pt` (excludes glyph-rendering small filled rects on
PDFs that draw text as filled-path glyphs). Raster-path layout exclusion (YOLO
regions as exclusion zones) is best-effort and gated by
`RedactionConfig.use_layout_filter`.

### Element-kind classification (the "K-cluster")
The element stream classifies each block into an `ElementKind`. Key
classification decisions:

- `signature` is the signatory **block**, not the closing phrase. The old
  `_SIGNATURE_RE` (matching "Yours sincerely") was removed — closing phrases
  fall to `paragraph` until a proper signatory-block detector exists (`caption`
  / `signature` are reserved kinds).
- `list_item` matches `(a)`/`(i)`/`(1)`/bullets; bare `1.` is excluded as
  ambiguous with numbered paragraphs.
- `header` round-trips into a real kind (was silently demoted to `paragraph`).
- Form-pair label denylist (`Penalty`, `OFFICIAL`, `Note`, `Caution`) stops
  regulation-citation / banner text being matched as form fields.
- Layout backend is the **DocLayNet `yolo11n_doc_layout.pt`** checkpoint (11
  document classes), not COCO `yolov8n.pt` (0 document semantics). The COCO
  weights remain a fallback; `YOLOLayoutAnalyzer` detects the taxonomy from the
  loaded class names and picks the label map + per-taxonomy `imgsz`.
- **Full-page-scan figure trap:** the OCR dominant-region fallback in
  `_layout_blocks_and_tables` collapses a whole page's OCR onto one block using
  the largest region's kind; when that is a `Picture`, a text-bearing full-page
  scan becomes a single `figure` and is excluded from chunking. Fixed by
  `_ocr_region_block_type()`, which promotes a non-text fallback kind to
  `paragraph` when OCR yields ≥5 words; sparse output (page numbers, bare logos)
  stays `figure`.
- OCR form-pair bboxes: a region-walking extractor
  (`_extract_form_pairs_from_regions`) assigns real per-field bboxes from
  PaddleOCR/RapidOCR line detections; the legacy line-only path is retained for
  LLM-OCR engines that resolve reading order natively.

## Rejected approaches / dead-ends

- **OCR-side table-detection relaxation — do not retry without a new
  discriminator.** Four variants of relaxing the OCR `_table_aware_text` rule
  were tried; all hit the same trade-off cliff — the relaxed rule helps real
  tables and hurts forms by the same amount under per-region OCR input.
  Re-attempting needs a genuinely new signal (layout-from-image, paragraph-gap
  vs cell-gap, column-spread-vs-page-width), not another threshold tweak.
- **PDF-annotation read for redactions — not viable for flattened releases.**
  `page.annots()` returns nothing when redaction tooling flattens bars into the
  page content stream (a common publication step). No annotation-based path is
  available there; detection must work from drawings/raster.
- **"All native-text element bboxes are zero" — retracted.** This was a probe
  formatter artefact (`:.0f` rounding 0–1 normalised floats to "0"); native-text
  kinds are bbox-populated. The narrower real issue was OCR-form bbox loss,
  since fixed.

## Known limitations (library-general)

- **Residual low-confidence native-text table fabrication.** PyMuPDF's
  text-strategy `find_tables` can synthesise a spurious `kind='table'`
  (`confidence=0.60`, `text_len=0`) when large redaction blocks carve prose into
  whitespace-aligned columns. Text-bearing elements still capture the prose
  verbatim, so it is additive noise, not corruption. Closing it needs a
  cross-validation step (reject native low-conf tables where the layout model
  predicts no Table at the bbox).
- **Handwriting is an OCR-engine ceiling.** The PaddleOCR ONNX backend cannot
  read handwriting; cross-cell handwritten forms and photographed/creased forms
  reach high CER. Out of scope without an HTR backend + dewarping.
- **Redaction-induced paragraph breaks (native pages).** PyMuPDF returns text
  either side of a mid-paragraph redaction as separate blocks; joined with
  `\n\n` they emerge as two paragraphs. Belongs to a downstream cleaning op.
- **Native-text footer whitespace artefacts.** Sub-glyph kerning yields
  `3|P age`-style footers on the native path; verbatim-text policy means
  extraction won't normalise — a downstream cleaning op's job.
- **Verbatim policy.** Extraction emits producer bytes; OCR/font-map errors
  (letterhead typos etc.) are preserved. Systematic cleanup belongs to a
  downstream `clean_text`-style op, not extraction.

## Deferred / backlog

- **Downstream text-cleaning op (#B/#D)** — *v1 shipped* as `womblex normalise
  --shards` (`process/normalise.py` transforms + `process/normalise_stage.py`
  driver). Writes a `*.normalised_text.parquet` text layer over the narrative
  elements (distinct sidecar, resolving the `*.clean_text.parquet` collision —
  this is the *cleaning* layer, PII's is the *masking* layer). v1 covers
  intra-element transforms: inline-whitespace collapse, `3|P age` footer-glyph
  despacing, and config-driven letterhead/font-map substitutions. **Still
  deferred:**
  - *Re-joining redaction-induced paragraph breaks* — a cross-element op; needs
    a reassembly join-hint (`reassemble_narrative` decides the `\n\n` boundary),
    not an intra-element text edit. Not yet wired.
  - *Consumption* — the sidecar is written but no downstream stage reads it yet
    (chunking still consumes raw `elements`). Same write-first / consume-later
    shape the PII stage used (`pii_spans` then `clean_text`); wiring chunking to
    prefer normalised text behind a flag is the next step.

  **Scope — fidelity-neutral, not OCR-error correction (measured 2026-06).** The
  normalise op cleans *formatting* (whitespace, footer glyphs, known typos); it
  is **not** an OCR-error corrector and does not move CER/WER. Two reasons,
  both measured: (1) `cer()`/`wer()` already collapse whitespace internally, so
  the dominant transform is invisible to those metrics (Throsby ΔCER = +0.0000);
  (2) real OCR errors are *non-systematic* — across 18 labelled production pages
  the only recurring char confusion is `c`↔`C`; the rest are one-off,
  context-dependent misreads that no substitution rule can target. Rule-based
  OCR correction is therefore a **dead end** — do not re-attempt. OCR-error work
  belongs at the *engine/resolution* level (see the OCR-quality plan: recognition
  input resolution, preprocessing gate, confidence-gated VLM escalation), not in
  a post-extraction text rewrite.
- **Inline-per-span source redactions (#C).** Page-prefix `<REDACTED>` is in
  place; inline-per-span placement needs bbox-to-text character mapping (raster
  path now has per-word bboxes; native path needs a text-to-bbox map).
- **Redact-stage checkpoint unification.** `redact` keeps its own JSON
  checkpoint rather than the `CheckpointManager` surface used by the other
  per-stage CLIs.
- **Enrichment graph-edges sibling.** Enrichment writes entities + metadata
  only; relationship edges are not yet persisted.
- **E2E composition of enrich/link/pii under `womblex run`.** Per-stage CLIs are
  the primary path; full E2E composition of the graph/PII stages is deferred.
- **`overflow_strategy` pass-through on `EnrichmentConfig`.** Oversized
  documents currently 400 instead of auto-chunking. Low priority — oversized
  inputs are typically large tabular/reference data that should not be routed to
  the enricher at all (reference data belongs on the graph/reference path).
