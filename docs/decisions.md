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

### Distributed execution — stage-in/stage-out, not a filesystem abstraction
Cloud scale-out (2026-06) reuses the existing batch shape rather than rewriting
it. Womblex already shards, checkpoints per batch, and isolates per-doc
failures; the only gaps for horizontal scale were *shared state* and a *safe
claim*. Two deliberate choices:

- **Object storage by stage-in/stage-out, not by threading an abstraction
  through every stage.** A worker pulls a batch's inputs to a local scratch dir,
  runs the ordinary `Path`-based pipeline, and pushes the shard sidecars back
  (`store/remote.py`, fsspec). *Rejected:* making `shard_dir` a filesystem
  abstraction wired through all ~10 `*_shards()` stages + `output.py` +
  `shard_audit.py` — a large, risky refactor that fights the `Path` idiom for no
  gain over copying a handful of files at the job boundary. fsspec's local
  backend means the air-gapped/CPU default exercises the *same* code path with
  no S3 dependency touched.
- **A Postgres `FOR UPDATE SKIP LOCKED` queue is the distributed checkpoint, not
  a second mechanism alongside the JSON `CheckpointManager`.** Concurrent
  workers writing one checkpoint file would race; distinct rows under
  `SKIP LOCKED` do not. So the worker path does **not** use `CheckpointManager`
  — the job `status` carries resumability, and re-running `enqueue` (idempotent
  on `(run_id, batch_num)`) is the resume. *Rejected:* Redis/Celery (new
  infra, against the single-datastore goal) and an in-app file lock (no
  multi-host safety). One Postgres table, no broker.

The load-bearing invariant: `cmd_run` and the worker call **one** shared
`batch.process_batch`, so local and distributed runs produce byte-identical
shards by construction — the same reasoning as "per-stage and E2E feed the same
engines".

### Reference registers — dedicated ingests; document formats — generic
Two pathways, chosen deliberately (2026-06). Widely-used reference registers
with novel format quirks (G-NAF PSV, ABN bulk extract XML) get **dedicated
standalone ingest modules** — the schema projection is irreducibly
source-specific, and a config-driven xpath/column DSL would be bigger than the
~100 lines of parsing it replaces. The dedicated modules share one skeleton
(discover → per-file ingest with isolation → Parquet + provenance metadata;
`utils/checksum.md5_file` is the shared piece) but the skeleton itself is not
yet abstracted — three copies with one shared helper; consolidate into a
register-ingest layer if/when a fourth register lands. **Document and export
formats (Excel/CSV) stay generic**: no filename knowledge, no per-dataset
toggles; corpus wrinkles (AusTender preamble rows) are solved as
general heuristics in the shared spreadsheet path.

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

- **AI chunking (semchunk 4) — single-enrichment graph reuse.** *Shipped
  2026-06, off-by-default.* The `chunking.chunking_model` pass-through lets
  semchunk pick chunk boundaries from the Kanon-2 enricher's structure spans; it
  is **opt-in and off by default** so non-Kanon tokeniser users keep the offline
  token/recursive split (composability). The *cost* concern — AI chunking
  enriches each document at chunk time while the separate `enrich` stage
  enriches the *same* reassembled narrative again, paying Kanon-2 twice — is now
  solved: the enrich stage persists the graph and the chunk stage reuses it,
  enriching once. The design below records how and why; the implementation notes
  follow each step.

  **Mechanism (verified against installed semchunk 4.0.0 source).**
  `semchunk.chunk` / the chunker accept `str | ILGSDocument`, where
  `ILGSDocument` is `isaacus.types.ilgs.v1.document.Document` — the *exact* SDK
  object `analyse/enrich.py` already receives as
  `response.results[i].document` and discards after `_convert_document`. When a
  Document is passed, semchunk sets `text = ilgs_doc.text` and builds its span
  tree **from the doc, with no API call** (`semchunk.py:227,236,265`); chunk
  offsets then index `ilgs_doc.text`. Both `enrich_shards` and `chunk_shards`
  reassemble the narrative through the *same* `reassemble_narrative` + same
  `text_source` overlay (identical by construction), and `enrich`/`chunk` are
  sibling stages depending only on `elements`, so running enrich first is
  dependency-safe (embed reads chunks; link/pii read enrichment; nothing feeds
  enrich from chunks).

  **Why a new sidecar is unavoidable.** Stages are separate CLI commands with
  no shared memory, and the existing `*.enrichment_entities.parquet` /
  `*.enrichment_meta.parquet` are flattened and lossy (entity mentions + meta
  only — no segment tree, headings, crossreferences, or the full span set
  semchunk consumes). Reuse requires persisting the *raw* Document. It is a
  Stainless-generated pydantic model, so `model_dump_json()` ↔
  `model_validate_json()` round-trips losslessly.

  **Chosen design — Option 2 (persist raw Document, reorder enrich→chunk,
  chunk reuses).** Rejected alternatives: Option 1 (share the in-memory
  Document — impossible across separate CLI stages); Option 3 (merge enrich +
  chunk into one stage — breaks the one-stage-one-sidecar composability the
  pipeline is built on). Option 2 steps:
  1. New self-contained store `store/enrichment_doc.py` →
     `*.enrichment_doc.parquet`, one row `{source_hash, text_source,
     document_json}`; mirrors `enrichment_output.py`. The `text_source` column
     records which cleaning overlay (`elements`/`normalised`/`spellfix`) the
     persisted `document.text` was reassembled under — it is the cheap key for
     the reuse guard in step 4.
  2. `analyse/enrich.py` exposes the raw Document alongside the converted
     `EnrichmentResult` (it is already in hand inside `enrich_documents`).
  3. `enrich_stage.py` writes the doc sidecar **off by default** — opt-in via
     an `enrichment.persist_document` flag, auto-enabled when
     `chunking.chunking_model` is set; users who never reuse pay no storage.
     Stamps the run's `text_source` into the sidecar. Checkpoint/skip parity
     with the entities sidecar.
  4. `chunk_stage.py` + `process/chunker.chunk_batch`: when `chunking_model`
     is set, load the doc sidecar and, per `source_hash`, pass the rehydrated
     Document into semchunk for the *narrative* path (tables stay token-mode).
     Offsets index `document.text == narrative`, so `page_breaks` mapping is
     unchanged.
     **Reuse guard (the coordinate-space invariant made runtime).** The
     "same string, one coordinate space" rule (see the Kanon-2 repair decision
     below) holds *within* one invocation because `chunk_stage` and
     `enrich_stage` apply the identical `text_source` overlay. Reuse spans two
     separate CLI invocations, so that is no longer guaranteed: a user could
     `enrich --text-source spellfix` then `chunk --text-source elements`, and the
     persisted `document.text` would index a *different* narrative than chunk's
     own reassembly — silently desyncing the PII mention↔chunk offset mapping
     this rule exists to protect. Therefore the reuse is gated: the
     **authoritative check is byte-identity** — accept the persisted Document
     only when `document.text` equals chunk's freshly reassembled narrative for
     that `source_hash`. The stamped `text_source` is persisted for audit and as
     a cheap pre-filter, but byte-identity subsumes it (and also permits safe
     reuse when overlays are relabelled yet produce identical text). On any
     mismatch (or a missing sidecar),
     **fall back to self-enrich** (`chunking_model` only) — i.e. promote
     verification gate 1 from a one-time check to a per-document runtime guard,
     mirroring the project's existing "missing overlay falls back to verbatim"
     idiom. The fallback double-enriches that document only.
  5. Ordering contract: with AI chunking on, run `enrich` before `chunk`.
     Repoint the existing `WomblexConfig` validator from "warns about
     double-enrich" to "warns only when reuse isn't wired (sidecar absent /
     enrich after chunk)".

  **Verification gates — require a live Isaacus key. Checked 2026-06 against a
  live `kanon-2-enricher` key with semchunk 4.0.0 / isaacus 0.20.0:**
  1. ✅ `document.text` is byte-identical to the input narrative (the offset
     basis the whole reuse rests on).
  2. ✅ A rehydrated `Document.model_validate_json()` satisfies semchunk's
     `isinstance(text, ILGSDocument_Runtime)` runtime check (same SDK class
     identity, `semchunk.semchunk:227`), and semchunk chunks the rehydrated
     Document down the AI path (no API call).
  3. ⚠️ *Partially.* On a normal-sized doc, `overflow_strategy="auto"` returns
     `document.text` == full source and all mention spans index within bounds
     (verified: 12 real spans, correct slices). **Residual:** the true
     multi-prechunk stitch (a doc exceeding the enricher context window) was
     not exercised — close this with a large real fixture before declaring
     production-ready.

  **Status:** shipped 2026-06 (offline tests + live round-trip on the vendored
  Throsby fixture). The `chunk_batch` Document-acceptance change was the main
  risk and is covered by the byte-identity guard above. Remaining caveat: the
  gate-3 large-document residual — until a doc exceeding the enricher context
  window is exercised, treat very large inputs under `chunking_model` + reuse as
  unverified.

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

### Dictionary-gated OCR repair (`womblex spellfix`) — *v1 shipped (2026-06)*
A **separate** op from `normalise` (which stays fidelity-neutral). It targets a
narrow, observed failure the dead-end above does *not* cover: digit/letter glyph
confusions surviving into chunks (`chi1d`→`child`, `p1an`→`plan`). Crucially it
is **not** the rejected substitution-table approach — it does not enumerate
errors as fixed rules. It validates candidates against the bundled en_AU Hunspell
dictionary (`spylls`, harvested from the Australian Writing MCP; MIT/SCOWL) and
rewrites a token only when **three gates** pass: out-of-dictionary trigger,
single-character in-dictionary candidate, and a *unique* such candidate
(unambiguity gate). Default Tier A swaps only OCR digit→letter homoglyphs
(length-preserving, near-zero false positives); Tier B (general edit-distance-1)
is opt-in and carries a proper-noun corruption risk, so it is flag-gated.

What it does **not** catch: valid-word misreads (`com`→`corn`) — the wrong word
is in the dictionary, so nothing fires; those still belong to the engine/
resolution level. Implemented as `process/spellfix.py` (corrector) +
`process/spellfix_stage.py` (driver) + `store/spellfix_output.py`.

**Repair lives at the element layer, not the chunk layer — dictated by how
Kanon-2 is designed to be used.** The enricher (`kanon-2-enricher`) ingests the
*whole document* as one string and does its own internal chunking/segmentation
(`overflow_strategy='auto'` stitches long docs back into a single prediction);
its returned ILGS spans are Unicode code-point offsets into *that* source string.
PII then maps those mention offsets onto our chunks via `chunk.start_char`. So
the enricher input and the chunk source must be the **same string** in **one
coordinate space**. Repairing at the chunk layer could never feed enrichment (it
reassembles from elements, not chunks) and would split that coordinate space.
Therefore spellfix writes an **element-text overlay** (`*.spellfix_text.parquet`,
same shape as `*.normalised_text.parquet`) plus a `*.spellfix_corrections.parquet`
audit; the raw `*.elements.parquet` is never modified.

**Composition is a linear cleaning chain, selected by one setting.** Cleaning ops
chain — `elements → normalised_text → spellfix_text` (spellfix overlays the
normalise layer when present) — each a full passthrough layer; the last one
produced is the canonical text. Consumers select it via a **single** pipeline
setting, `processing.text_source` (`elements` | `normalised` | `spellfix`),
resolved by `process/text_overlay.py` and applied before reassembly at *both*
sites (`chunk_stage`, `enrich_stage`). It is deliberately one knob, not per-stage:
divergent layers would desync the Kanon-2 mention↔chunk offset mapping. Embeddings
and PII then inherit the repaired text for free (chunks derive from the same
overlaid elements). A missing overlay falls back to verbatim, so stage *ordering*
is the only requirement, not a hard dependency.
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
