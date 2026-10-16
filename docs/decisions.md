# Design decisions

The durable "why" behind the library: decisions and their rejected
alternatives. Operating decisions (distributed execution, workflows,
deployment, the service) and the dead-ends, limitations and backlog each have
their own document. Current
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
per-stage CLI has an independent `CheckpointManager` and a resume-time integrity
scan (`redact` keeps its own JSON `--checkpoint`). `womblex run` is
extraction-only since 0.5.10; downstream runs via `run-stage`/`enqueue-stages`.

## Key design decisions

### Extraction routes per page; detection holds no extraction logic
A government bundle mixes cover letter, columnar table, form and signed
declaration in one file, so document-level routing collapses it to one strategy
and loses the per-region structure. PDFs and standalone images are profiled per
page (`page_profile.py`) and the orchestrator dispatches native or OCR per page;
`DocumentType` is a summary attribute there and selects the extractor only for
the path-based formats (SPREADSHEET, DOCX, TEXT, MARKDOWN). Files stay under the
750-line cap by splitting along these seams (`orchestrator.py` /
`page_profile.py` / `extract.py`, `strategies_*` behind a re-export shim, a
per-topic `cli/` package).

### Pre-extracted corpora — an ingest that *feeds* the pipeline
`ingest/records.py` (2026-07) exists because Womblex's stages consume
element-shard directories, but a pre-extracted corpus (the Open Australian
Legal Corpus; any JSONL of already-clean text) needs no extraction. It is the
odd one out among the standalone ingests: `gnaf`/`abn`/`geo` produce flat
register tables that *bypass* the NLP pipeline, whereas records ingest produces
the element stream that *feeds* it. Deliberate choices:

- **`source_hash = sha256(record_id + text)`** (not file bytes — there is no
  file). Content-addressed, so the asset refresh procedure is a cache hit by
  construction: re-ingesting an unchanged record yields the same hash, and its
  existing enrichment/chunk/embedding sidecars still join.
- **Text → one `paragraph` element per blank-line block.** The canonical
  document text for the asset is therefore the reassembled narrative
  (`\n\n`-joined), byte-identical to the source for the `\n\n`-delimited
  majority. Offsets are internally consistent by construction — enrich and
  chunk both reassemble the same elements — so byte-fidelity to the *original*
  JSONL is not required (no consumer maps enrichment offsets back to it).
- **Corpus-agnostic.** A `RecordFieldMapping` (declared by `stories/<corpus>`)
  names the id/text/provenance fields; the library knows nothing about OALC.
  Provenance columns go in a `*.provenance.parquet` sidecar
  (`store/provenance_output.py`), not `MANIFEST_SCHEMA` — keeping the shared
  manifest schema stable while carrying arbitrary per-corpus metadata,
  consolidated into a run-root `manifest.parquet` (source_hash → provenance).

### Rate limits bind on tokens — token-budget request packing
Observed (and confirmed across a ~280M-token programme): Isaacus rate limits
bind on **tokens per request/window**, not request count. So request packing is
never naive doc-count batching. `utils/token_packer.py` packs to
`min(max_items, token_budget)` using **exact local counts** from the kanon-2
tokenizer (free on Hugging Face, vendored under `_models/` — counting is
offline and exact, not estimated). `enrich_stage` uses it to send
`min(8 docs, token_budget)` per request — an 8× request-count cut for small
docs, while staying token-safe for a batch of long judgments. Measured: this
sustained ~0.7–1.0 M tok/min single-process with **zero 429s** (throughput is
workload-shaped — decisions pack denser than bulky legislation).

### Long documents — client-side split + offset-merge; split docs not persisted
A document past `split_ceiling` (default 100K tokens) is split on structural
(blank-line) boundaries into sub-documents enriched separately, and the
per-segment `EnrichmentResult`s are stitched back by `analyse/enrich_merge.py`:
every span shifted by the segment's `start_char`, every entity/segment id
namespaced by segment index. This is the same offset stitch the enricher does
internally for >16K inputs, applied one level up — needed only because a single
very large *request* is the rate-limit risk. **Split docs are deliberately not
persisted** for AI-chunking reuse (no single ILGS Document spans the full
narrative), so those few long-tail docs self-enrich once at chunk time.
*Measured tradeoff:* negligible for caselaw (~0.4% of docs >100K) but material
for legislation (~4%, +~15% on enrich cost); the lever is to raise
`split_ceiling` — a single 377K-token request was observed to succeed with no
429, so the ceiling is far above the earlier ~150–200K assumption and the
32K/100K defaults are conservative.

### Graph-edge refresh — enrichment precedes AI chunking, so edges lag
Semchunk-4 AI chunking reuses the persisted enrichment Document, so the order
is `enrich → chunk`. Consequence: the `*.graph_edges.parquet` written at enrich
time cannot contain mention→chunk edges (no chunks exist yet), and
`enrichment_entities.chunk_index` is `-1`. `analyse/graph_refresh.py` closes the
gap **offline** after chunking: entity mention spans and chunk offsets share the
same narrative coordinate space, so overlap is deterministic and API-free. It
rewrites `chunk_index` and replaces the `mentioned_in` edges while preserving
hierarchy/citation/cross-reference edges; idempotent by construction. Only
entity mentions are relinked — segment/cross-reference chunk edges need the full
`EnrichmentResult` and are out of scope.

### Reference registers — dedicated ingests; document formats — generic
Two pathways, chosen deliberately (2026-06). Widely-used reference registers
and formats with novel quirks (G-NAF PSV, ABN bulk extract XML, ESRI
Shapefiles with geometry + CRS) get **dedicated standalone ingest modules**,
their heavy dependencies (`pyogrio`, `geopandas`, `shapely`) lazy-imported — the schema projection is irreducibly
source-specific, and a config-driven xpath/column DSL would be bigger than the
~100 lines of parsing it replaces. The dedicated modules share one skeleton
(discover → per-file ingest with isolation → Parquet + provenance metadata;
`utils/checksum.md5_file` is the shared piece) but the skeleton itself is not
yet abstracted — three copies with one shared helper; consolidate into a
register-ingest layer if/when a fourth register lands. **Document and export
formats (Excel/CSV) stay generic**: no filename knowledge, no per-dataset
toggles; corpus wrinkles (AusTender preamble rows) are solved as
general heuristics in the shared spreadsheet path. Dataset-specific paths,
thresholds and hypotheses live in `configs/*.yaml`; core modules know no dataset.

### Spreadsheet header detection — run-scoring over width-ratio
Export products open with title rows, generated-date lines, or `key: value`
metadata blocks above the real header; pandas' default `header=0` fabricates
`Unnamed: N` column names from them. Sheets are read `header=None` and split by
`split_preamble`. The header is the candidate row (≥2 non-empty cells in a
10-row window) that **starts the longest run of table-consistent rows below
it** — blank rows break a run, single-cell section rows are neutral, ties
prefer the wider candidate. Chosen over the simpler first-wide-row /
width-ratio rules because both misfire on multi-cell metadata lines and on
titles wider than a narrow table; width-ratio survives only as the fallback
when no candidate has a table body below it (header-only sheets). Rejected:
content heuristics (uniqueness, non-numeric-ness of header values) — false
positives on real headers with duplicate/numeric labels outweighed the gain;
per-corpus threshold config — these are format heuristics, not dataset
settings. Known limitation: a preamble row spilling across most of the table's
width directly above the header (no blank separator) can still win.

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

### Models are local, named, and checked before a run
- **ONNX, no ML framework.** OCR is PaddleOCR's ONNX models on
  `rapidocr-onnxruntime` with no PaddlePaddle (bundled v5 mobile preferred, the
  wheel's v4 the fallback); layout is PP-DocLayout-M on `onnxruntime`. Metrics
  carry no C-extension either: `utils/metrics.py` is a numpy-vectorised
  Levenshtein (pure-Python DP at 500 chars or fewer), and CER-s spatially sorts
  words to separate recognition errors from reading-order errors.
- **Resolved locally, recorded by digest.** `resolve_local_model_path` searches
  `WOMBLEX_MODELS_DIR`, `_models/`, then `models/` (HF snapshot layout and bare
  files) before any download, loads lazily, and is the load record: digests are
  over the bytes, so they recompute from the model files alone; a presence probe
  passes `record=False`.
- **Registered names only.** An unknown name raises listing the known ones,
  import paths are refused, a plugin cannot shadow a built-in, and options pass
  through to the factory unchanged. A slot is recorded as used only when its
  factory is called — which is what makes an API-backed OCR engine with no
  artefact to digest establishable from the footer.
- **Checked before the first document** (`processing.models_check`: `off` /
  `load` / `smoke`). A service-backed model has nothing to load, so each one an
  enabled stage calls gets one minimal request (Isaacus models; `ping()` for
  Bedrock / Ollama), so a paid later stage fails before the stages ahead of it
  are paid for. The check is a probe, not use: it runs under
  `suppress_use_recording` + `suppress_recording`, and a caching wrapper must
  test `recording_suppressed()` to bypass its cache, or the run's real build
  would hit the probe's instance and never record. Failures are collected per
  scope and each caller applies its own: `run` stops before any output, a
  worker releases (never fails) jobs of a failed scope, and the service API
  checks registration only, since its host may not carry the models.

### PII — graph-driven detection, masking after Isaacus
PII detection **is** the enrichment graph: select PII-typed entities (`natural`
→ PERSON, `address` → ADDRESS), map their mention offsets onto chunks, mask.
There is no separate detector and **no second enrichment pass**.

- **Detection runs on chunks; the graph is the entity source.** Graph mentions
  (full-narrative offsets) map into `narrative` chunks via `chunk.start_char`.
  `table` chunks carry a different offset space, so narrative graph spans are
  never applied to them; a table's own enrichment supplies theirs (below).
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
- **`clean_text` says whether masking could have found anything** (2026-10;
  D1 of the trust-and-recipes plan). Each row
  carries `mask_status`: `masked`, `no_entity` (a candidate source covered the
  chunk and found nothing) or `not_masked` (none did, so the text is verbatim).
  *Rejected:* refusing the stage with a typed error when a chunk has no
  candidate source (a table chunk has none when `enrichment.include_tables` is
  off or its document was not enriched, so such runs would refuse), and omitting
  `clean_text` rows for uncovered chunks (readers lose the chunk silently). The
  column is additive, so existing readers keep working.
- **Each table's markdown is enriched as its own text** (2026-10; D2 of the
  plan). The string is the one `chunker.table_texts` hands the chunker, sent as
  a separate request text beside the narrative in the same requests
  (`enrichment.include_tables`, default on), so narrative offsets do not move
  and a table mention's offsets share a coordinate space with the table's
  chunks. Its mentions carry `text_layer='table_markdown'`, the table's
  `elem_order` (or `sheet`) and `t<n>:`-namespaced entity ids (each result
  restarts at `p1`). The PII stage maps them onto the table's chunks by
  `chunk.start_char`, as for the narrative, when the chunk's text is the
  markdown's slice at its offsets; a chunk that does not line up (repaired or
  stale) or whose table cannot be found takes no table spans and reads
  `not_masked`, because offset masking there would be unsafe. A table chunk
  counts as covered when its document's meta row records `table_count > 0`.
  *Grounds*, from the Isaacus enrichment docs: batch
  composition does not change any document's results; the returned `doc.text`
  is identical to the input, with spans as code-point offsets into it; entity
  ids are unique within a document, so a whole table keeps one id per person
  across its chunks; tables are a native `figure`/`table` segment; billing is
  input tokens plus overhead per text, at US$3.50 per million;
  `overflow_strategy: auto` handles text over 16,384 tokens; and per-chunk
  enrichment would need a second enrichment pass, which the PII decision above
  rules out. *Limits:* a person in both the narrative and a table gets two
  different mask numbers; the model sees a table without its caption or heading
  context; a sheet, which has no element position, is matched to its chunks by
  text; a table over the 100K split ceiling is sent whole (one blank-line
  block, so nothing to cut on). *Not done:* tables folded into the narrative
  (moves every narrative offset), and graph edges for table results (their ids
  and offsets are not in the narrative's space). With `include_tables` on, a
  batch whose meta footer lacks `womblex.tables_enriched` (enriched with the
  pass off, or by an older release) is enriched again on resume; the
  narrative is re-sent and re-billed along with its tables. *Measured spend* (kanon-2
  tokeniser, synthetic set, 28 documents, 6 with tables): 16,584 narrative
  tokens and 22,832 table tokens, so the table pass adds ~138% — the set is
  register-heavy (the sightings register alone is 12,716 table tokens), and a
  corpus of prose reports will add far less. `include_tables: false` switches
  it off, and table chunks are then `not_masked`.
- **Coverage is PERSON and ADDRESS.** ORGANISATION, URL, phone and email are
  not detected.

### The evidence reference
One anchor shape replaces the per-sidecar anchors of `money_spans`,
`entity_links` and `pii_spans` (2026-10; D6, contract 2.0): eleven columns, ending in
`anchor_level` and `evidence_check` (`docs/contract.md`).

- **No null receipts.** A row carries the most precise location that is reliable
  and a label for it (`span`, `element`, `chunk`, `document`); each level is
  checked at its own precision and labelled `holds`, `fails` or `unchecked` in
  `evidence_check`; the last two are warned about with a count. *Rejected:* null evidence for a span that cannot be placed (it
  reads as "no evidence", and for PII a refusal would mean dropping the mask).
- **A failed check never refuses the batch.** It is written and labelled `fails`.
  *Rejected:* refusing the batch's publish (the first cut of this item). A refusal
  stops the stage part way, after paid enrichment, and blocks the run from
  completing over one receipt; trust comes from the output saying what it could
  not verify, not from withholding the rest.
- **A position is never searched for.** Offsets are carried from where they were
  computed or the row is given a coarser level; a search would hand back a
  plausible position for the wrong occurrence.
- **Offsets are in the space of the text `text_layer` names, not element-relative.**
  A narrative span crossing two elements is still one row, and the join to chunks
  (`start_char`) and to enrichment mentions stays an overlap test.
- **Masking never depends on anchoring.** A table chunk that does not line up with
  its table takes no table spans and reads `not_masked` (offset masking there
  would be unsafe), and its span rows are located to the document or the chunk.
- **A link is checked against the enricher's text for the mention** (`mention_text`),
  not its `candidate_text`: `name` is the entity's, the same for every mention, so
  it differs from a mention's own text ("the Minister" against "Jane Doe").
- **Replaced, not duplicated.** The old columns are gone and older files read through
  a shim. *Rejected:* keeping the old columns beside the new (two anchors that can
  disagree). The womblex-benchmark suites that read these sidecars need the paired
  change.
- **`money_columns` is not a span sidecar** and keeps its own column keys.
- **A finished run can be re-checked.** `womblex verify-evidence <run>` recomputes
  every row's check against the source (the write-time label is a record, not a
  proof: sources change, and older files have none), beside `verify-shards`.

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
PDFs that draw text as filled-path glyphs). Raster-path layout exclusion
(the layout step's `figure` / `table` regions, as exclusion zones) is best-effort and gated by
`RedactionConfig.use_layout_filter`.

Detection runs *after* extraction, never inside it: masking within `_ocr_page()`
misfired on form fields, chart regions and diagram fills, suppressing text it
should keep, so extractors never call `pre_ocr_mask`. The `RedactionReport` is a
typed field on `ExtractionResult`.

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
- Layout backend is **PP-DocLayout-M on `onnxruntime`** (`ingest/layout_onnx.py`,
  registered default `pp-doclayout-m`, #130). It replaced the previous
  layout analyser, which was removed in L2.
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

### Table-cell reconstruction on OCR pages — region-based engines only
Reconstructing cells inside a layout-detected table rect needs per-detection
OCR quads to bin into rows and columns. Only region-based engines (paddleocr)
produce them: LLM/VLM engines (`mistral-ocr`, `ollama`) set
`reading_order_native` and return markdown with an **empty `regions` list**, so
the orchestrator sends them to `_markdown_page_block` and they never reach the
layout pass at all. Table reconstruction is therefore scoped to the
region-based path on both the OCR-PDF and image routes, and any benchmark
measuring it pins `engine="paddleocr"` — under a config default of an LLM
engine the numbers would describe a different pipeline. A
markdown-pipe-table → `TableData` parser is the LLM path's separate feeder,
deferred.

`_layout_blocks_and_tables` takes the OCR regions plus the OCR render's pixel
dimensions, together or not at all. The layout step's boxes are normalised, so
they are mapped onto the OCR render's own dimensions; regions supplied without
those dimensions cannot be placed and are dropped with a warning. Losing
reconstruction inputs is the correct failure; a mis-binned grid would be
confidently wrong downstream. Deskewed pages are a distinct hazard (OCR runs on
the warped image, the layout model on the raw one) that normalising does **not**
remove — `warpAffine` preserves the frame's dimensions — so they get their own
page-level refusal: the orchestrator reads `"deskew" ∈ steps` off `_ocr_page`
and the layout pass drops its cell source, keeping the page's
pre-reconstruction behaviour exactly.
Mapping the layout rect into deskewed space is deferred to the round targeting
real scans.

### The OCR table double-count is the narrative fallback, not the placeholder
When a table reconstructs on an OCR'd page, the page narrative must be rebuilt
from the OCR regions *outside* the table rect. The reason is easy to get wrong:
layout-derived non-table blocks always carry `text=""`, so
`_layout_blocks_and_tables`'s "no block has text" fallback fires on essentially
every layout-successful page and returns **one block holding the whole page's
OCR text, table content included** — already discarding the `[TABLE]`
placeholder. Subtracting only the placeholder would change nothing; the
chunker would still see the table twice, once as narrative prose and once as
markdown. The complement is re-emitted with `_table_aware_text` rather than
`_spatial_sort_regions`, because the reconstructor supersedes it only *inside*
the rects the layout model found — outside them it still covers table runs the
model missed. The same absorbed regions are withheld from
`_extract_form_pairs_from_regions`, so a colon-bearing cell cannot land in both
a form element and the table.

`PageResult.text` stays the verbatim full-page OCR text through all of this.
The subtraction is an element-stream concern; page text feeds `_text_coverage`
and the accuracy suite's CER, which compare against a transcript of the whole
page, and chunking reads `elements` rather than `pages[i].text`.

### The ground-truth preset name rides the run stamp, not a helper argument

A ground-truth unit's sidecar records both `preset_digest` (which configuration,
exactly) and `preset` (its readable name). The digest was already the run's
`config_digest`, read by the producer from the extraction shard's footer along
with `run_id` and `parser_version` — every derivation-recipe field inherited
from the artefacts, none asserted by a caller. `preset` had no footer source, so
it stayed a sentinel.

It is filled the same way rather than by a new argument to `build_ground_truth`:
`store/run_stamp.py` stamps the configuration's `dataset.name` as a sixth
`womblex.*` footer key, written whenever a config names one and carried through
stage inheritance like `config_digest`, and the producer reads it beside its
`preset_digest` sibling. A helper argument would have been the one recipe field
taken on a caller's word rather than checked against the file, and would have
made the producer — handed a shard directory, not a config — reach back for the
config to supply it.

The key is general provenance, "which configuration produced this file", not a
ground-truth concern: no stage branches on it, so the pipeline keeps no
ground-truth mode. It falls back to the sentinel for a shard stamped before the
key existed or by a config that declared no name.

### Money recognition — hand-written patterns, exact `Decimal`s
- **No parsing library.** Evaluated `price-parser` / `money-parser`,
  `quantulum3`, `pint`, spaCy `MONEY`, Presidio and LayoutLMv3. The price
  parsers drop magnitude (`$2m` → 2, `$8.7 billion` → 8.7) and accounting-bracket
  signs — e-commerce assumptions of small, unscaled, always-marked prices.
  `quantulum3`, the closest call, is a physical-units library: it surfaces page
  counts, percentages, years and section numbers that must be filtered away
  (where precision is actually decided), and returns floats, from which exact
  values cannot be recovered — it fails the thin-adapter test. `pint` models
  physical dimensions, not currency, and is float-first. spaCy `MONEY`
  (unvalidated) and Presidio return spans with no value, and Presidio would
  merge money into the PII stack. LayoutLMv3 needs page images and labelled
  data; the op's input is parquet. None addresses the real risk — deciding
  whether a bare column of numbers is money. The hand-written surface (a
  15-entry scale table plus the Australian pattern set) is small and stable;
  worded amounts (`process/money_words.py`) were written in-house on the same
  argument, as a capability decision rather than a measured recall win, with a
  currency-word gate that declines worded headcounts.
- **Header continuation rows fold into the header.** PDF financial tables wrap
  a header over two rows (`Approved` / `Budget $m`) and the extractor declares
  only the first, losing the unit and money vocabulary. `fold_header_continuation`
  absorbs one leading body row, only when it is non-numeric text and the rest of
  the column is numeric — a header-reading fix, not a relaxation of the no-header
  rule.
- **Rejected: cross-validation by re-reading sources** — a second extraction
  path. Any reconciliation works on extracted output, compares multisets of
  values (counts hide compensating errors), and reports pages with no text layer
  `unverifiable`, never `mismatch`.
- **Deferred:** bare numeric columns with no recoverable header (promoting them
  leaks percentage columns); implicit narrative financial context (off pending a
  precision measurement); penalty units (revisit with a legislation/contract
  corpus).

### Permissive-dependencies Phase 0 — pypdfium2 over pdfminer for the text engine
The permissive-dependencies Phase 0 spike (scratch only, measured against
the womblex-collection fixture PDFs) settled the text-engine choice for P6 and
the plan's three unverified mechanism claims — two as stated, one the other way
round. The spike's scripts were not retained, so each figure below carries the
method that produced it and the page set it ran over: the numbers are meant to
be re-derivable, and the first pass's were not.

- **pdfminer's native segmentation misses the speed gate by ~67x, which is
  the axis that rejects it. On fidelity to fitz it is the *better* of the
  two.** The plan's working assumption was that pdfminer's own `LAParams`
  layout analysis is the default choice because the library does it natively,
  and hand-rolling segmentation from pypdfium2 characters would only be
  justified if pdfminer missed the speed gate. It misses it decisively.
  Measured over 76 native-text pages (≥200 chars of text layer) from 19
  womblex-collection PDFs, text extraction only, fitz
  `get_text("text", TEXT_DEHYPHENATE)` as the reference:

  | engine | ms/page | vs fitz | CER vs fitz: median / p90 / max | pages > 0.2 |
  |---|---|---|---|---|
  | fitz | 1.4 | — | — | — |
  | pdfminer (`pdfplumber.open(path, laparams={})`) | 93.9 | 67x | 0.010 / 0.237 / 0.720 | 8 / 76 |
  | pypdfium2 characters, rebuilt | 11.7 | 8.4x | 0.122 / 0.698 / 0.891 | 17 / 76 |

  So the ordering is not "pypdfium2 wins on both axes" — an earlier draft of
  this entry said that, and re-measurement does not support it. pdfminer's
  *output* tracks fitz closely (median CER 0.010, i.e. near-identical on a
  typical page); it is 67x the cost that buys it. The rebuild is 8.4x fitz
  and much further from fitz's text, and the rebuild measured here is the
  naive one — lines binned on a 3pt baseline tolerance, no dehyphenation, no
  block grouping, no column handling — so its CER is an upper bound on what
  P6 would ship, but the fidelity work behind closing that gap is real and
  not yet done. pdfminer is rejected as the P6 candidate on speed; pypdfium2
  characters, rebuilt in womblex, is what Phase 0 carries forward, with P6
  owning the segmentation quality the numbers above say is still missing.
- **The divergence tail is multi-column reading order, and it is not
  pdfminer-specific.** Both candidates' worst pages are the same pages: the
  five measured pages of the ASX annual-report excerpt account for every
  pdfminer page above CER 0.49, and the rebuild is no better there (0.61–0.74
  on the same pages). What the tail measures is reading order on
  multi-column, table-adjacent text, where each engine serialises a page
  differently from fitz rather than reading it wrongly — which is why CER
  against fitz is a *similarity* measure here, not an accuracy one. Scoring
  either engine as right or wrong on those pages needs the transcript CER
  half of H-B, i.e. the ground-truth revision.
- **pdfplumber's table-finding is not better calibrated than fitz's, just
  faster.** Over 65 native-text pages from the same fixture set: pdfplumber's
  "text" strategy fires on 63 of them, which is the same over-firing
  `extract.py` already guards against for fitz, not a signal that 63 pages
  hold tables. On the "lines" strategy fitz fires on 15 pages and pdfplumber
  on 25; they agree on *whether* a page has a table on 51 of 65 and disagree
  on 14, while of the 13 pages where both found one they agree on rows and
  columns on 10. So the disagreement is about existence, not shape. Timing,
  same pages: fitz `find_tables` 79.9 ms/page, pdfplumber "lines" 53.8,
  pdfplumber "text" 28.5 — faster, which is a genuine win independent of
  calibration. P7's pdfplumber adapter needs the same kind of post-filtering
  `extract.py` already carries for fitz (the cross-check against
  `get_text("dict")` block density), not a fresh assumption that pdfplumber
  tables arrive clean.
- **MuPDF's page rect for a standalone raster image honours the file's own
  DPI tag.** It is not a fixed 72dpi assumption: the page rect in points is
  `pixels / (dpi / 72)`, read from the image's own metadata, falling back to
  a 96dpi assumption when the file carries none (Pillow's own default when a
  caller saves a PNG without specifying one). Measured on a 200x100px PNG
  saved at each of 72 / 96 / 150 / 300 dpi and with no tag: the rect comes
  back 200x100, 150x75, 96x48, 48x24 and — untagged — 150x75 pt, i.e. exactly
  `px / (dpi / 72)` with 96 assumed. P5's Pillow-based `_image.py` must read
  `image.info.get("dpi", (96, 96))` to reproduce the same page rect fitz
  would have reported, not hardcode 72.
- **`get_images()` lists one entry per *draw*, not one per resource — and
  `_extract_images_from_page` does not account for that.** An earlier draft of
  this entry had the mechanism backwards. Measured directly: one image XObject
  drawn twice on a page (both by two `insert_image(stream=…)` calls and by
  re-using the first draw's xref) reports as **two** `get_images(full=True)`
  entries carrying the **same** xref, and `get_image_rects(xref)` returns both
  rects. So the two calls overlap rather than decompose cleanly into resources
  and draws. `_extract_images_from_page` loops `get_images` outer and
  `get_image_rects` inner with no xref de-duplication, so an image drawn N
  times on one page emits N² image elements. It is latent, not active: no page
  in the fixture corpus repeats an xref (0 of the first 20 pages of all 30
  fixture PDFs), which is why it has never shown up. Fixing it is its own
  change — image elements feed `content_digest`, so it re-pins
  `tests/test_default_digest.py` — and P5 must not port the quadratic loop
  across to pypdfium2's differently-shaped image-object API.
- **pypdfium2 needs disciplined object closing; thread safety is the
  caller's job, not the library's.** A 300-odd-document open/close sweep in
  one process threw a spurious "Data format error" re-opening a file that
  had opened cleanly earlier in the same run — caused by leaving
  `PdfTextPage` objects unclosed across their parent documents' lifetimes,
  not by the file. Closing every child object before its parent resolved
  it. Separately, pypdfium2's own packaged documentation ("Incompatibility
  with Threading") settles the question the first spike left open: PDFium
  is not thread-safe, full stop — no two pdfium calls may run concurrently
  on different threads of one process, *even against different documents*,
  unless every call is serialised behind one mutex; the 8-thread smoke test
  that produced no crash earlier is consistent with this (small, fast calls
  rarely overlap in practice) but was never a safety guarantee. There is no
  internal lock to rely on; a caller who skips the mutex "would crash or
  corrupt the process," in the library's own words. This turns out not to
  bind today regardless, and the check is one grep: neither server imports
  fitz at all (`api/` for `womblex serve`, `ui/` for `womblex ui` — both only
  enqueue and read parquet), which is the thing protecting them: their route
  handlers are sync `def`, so uvicorn runs them in a thread pool, and a
  single fitz/pdfium call added to a route would be concurrent by default.
  Beyond that, no module under `src/` constructs a thread,
  a thread pool or a process pool. The one place extraction actually runs,
  `cloud/worker.py`, claims and processes one job at a time in a single
  thread per process, scaling concurrency by running more worker
  *processes*, never threads inside one. P1/P5 just need to keep it that
  way — no pdfium call site becomes reachable from more than one thread of
  the same process. If a future change ever wants in-process thread
  concurrency over pdfium (a thread pool inside one worker, say), the fix is
  one process-wide `threading.Lock` around every pdfium call, not per-object
  isolation — the restriction is process-wide, not per-document.
