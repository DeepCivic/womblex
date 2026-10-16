# Dead-ends, limitations & backlog

Approaches tried and abandoned, library-general limitations, and the deferred backlog.

## Rejected approaches / dead-ends

- **pdfminer.six's own layout analysis as the permissive text engine — lost on
  speed.** Rejected in the permissive-dependencies Phase 0 spike on the speed
  gate (~67x fitz per page; on fidelity to fitz it is the better of the two
  candidates) in favour of a character-level rebuild over pypdfium2; the
  measurement is in "Permissive-dependencies Phase 0" in decisions.md. Re-attempting needs a reason to
  expect different numbers (a pdfminer release that changes `LAParams`, or a
  corpus whose reading order its segmentation handles and the rebuild does
  not), not a preference for native segmentation.
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
- **A generic document-verification engine — deleted rather than wired in.** A
  `verify/` package held `run_verifications`: a two-pass structural check plus a
  weak-signal scan classifying a run `passed` / `warning` / `failed` on a
  garbled-character ratio, a redaction-garbling ratio, a confidence floor and a
  page-count ceiling. It was never called by any pipeline path, and its own
  tests exercised the helpers rather than the entry point. Wiring it in was not
  a matter of renaming columns: it expects one row per document carrying
  `document_id`, `source_path` and `text`, where the manifest names the first
  `doc_id`, expresses provenance as `ingest_root` plus `source_relpath`, and
  carries no text at all — text is per-*element*, as is `confidence`, and
  `page_count` exists in no schema. Adapting it therefore meant a grain change
  and narrative reassembly, which is a rewrite. Its remit is also taken:
  `process/quality.py` and its stage annotate chunk quality properly, with a
  sidecar and a checkpoint. And its central idea is one this repository
  declines on purpose — threshold-based quality scoring, which the contributor
  guidance rules out while the corpus is still poorly understood. Deleted with
  its exports and tests. Integrity checking that *is* wired lives in
  `verify_shard_persistence()` and `store/shard_audit.py`, neither of which
  scores anything.
- **"All native-text element bboxes are zero" — retracted.** This was a probe
  formatter artefact (`:.0f` rounding 0–1 normalised floats to "0"); native-text
  kinds are bbox-populated. The narrower real issue was OCR-form bbox loss,
  since fixed.

- **Dependencies assessed for the trust-and-recipes work (2026-10).** Against the repository's rules (thin adapters, delete Womblex code a library takes over, no heavyweight ML in core, a core install that runs without a database). DBOS was adopted (see "Distributed execution" in decisions-operations.md).

  | Candidate | Licence | Verdict | Reason |
  |---|---|---|---|
  | OpenLineage (`openlineage-python`) | Apache-2.0 | No dependency | Emit spec-conformant JSON from the run record and test it against the published schema; an optional extra only if pushing to a lineage server becomes a requirement |
  | Docling (full converter) | MIT | Reject for core | Brings torch and its own layout models; its layout model could be a `womblex.models.layout` plugin outside core, judged by the benchmark ([`layout.md`](layout.md)) |
  | docling-core | MIT | Optional egress format, later | Light, pydantic-based; only as an output format for consumers that want Docling's schema |
  | docling-parse | MIT | Hold | A fallback if pypdfium2 + pdfplumber miss their parity gates |
  | Docling Graph | MIT | Reject; borrow the idea | Its deterministic provenance ledger informed the evidence reference; the package brings full Docling plus LLM clients |
  | Pydantic AI | MIT | Reject | Field extraction is out of scope; a topic annotation would go through Isaacus |
  | Unstructured | Apache-2.0 | Reject | Heavy, torch-based inference; duplicates the existing extractors |
  | MarkItDown | MIT | Reject | Drops page geometry, which the evidence reference needs |
  | Temporal, Prefect, Haystack | MIT / Apache-2.0 | Reject | A separate service, or a pipeline model competing with `STAGE_CONTRACTS` |
  | lakeFS | BUSL-1.1 | Reject | Not a permissive licence; a separate service |
  | OpenInference | Apache-2.0 | Reject for now | Model calls are Isaacus and Bedrock; JSON logs with run context already exist |

## Known limitations (library-general)

- **Residual low-confidence native-text table fabrication.** The
  text-strategy `find_tables` (measured under PyMuPDF; not re-measured under
  pdfplumber) can synthesise a spurious `kind='table'`
  (`confidence=0.60`, `text_len=0`) when large redaction blocks carve prose into
  whitespace-aligned columns. Text-bearing elements still capture the prose
  verbatim, so it is additive noise, not corruption. Closing it needs a
  cross-validation step (reject native low-conf tables where the layout model
  predicts no Table at the bbox).
- **Block grouping on the pdfium backend follows MuPDF's measured rules.**
  Measured with probe PDFs at 7pt and 10pt, in the unrotated page frame the
  characters are already in (`/Rotate` does not enter). Horizontal text, either
  way up: a line on the previous line's row, overprints included, continues its
  block; otherwise it continues within 1.5 font sizes of the baseline (above or
  below) unless its origin lies more than 0.5pt right of the previous line's,
  whichever way the text runs. Vertical text has no start condition, only a
  baseline window that favours opposite sides: up-running text takes -1.5 to
  +0.5 sizes, downward text -0.5 to +1.5 (so the rotated index's `Subsection` /
  `code` and `Out of Scope` / `Exemption` split while the next header cell
  rejoins). Oblique text continues only as the next cell of a row. A change of
  size or weight alone never breaks a block. `_same_block` in `_text.py` holds
  it; `emu-spacing-variants.pdf` and `test_pdf_block_parity.py` pin it, with
  header, overprint and line-above shapes at 0, 90, 180 and 270 degrees.
  Block line counts and element kinds equal MuPDF's on every synthetic PDF
  (measured on PyMuPDF 1.27.2.2 before its removal and pinned in the tests).
  pdfium sorts vertical text by position where MuPDF keeps content order, so
  `read_chars` restores it for vertical characters from the page's text objects
  (matched by text and box); and it injects a stray space between overlapping
  objects, which `_lines` drops when it lies off the row and the row carries on
  across it. Not matched: oblique overprints and lines above (the row test uses
  bounding boxes, crude at small angles), and characters pdfium reports
  outside every text object's box.
- **Handwriting is an OCR-engine ceiling.** The PaddleOCR ONNX backend cannot
  read handwriting; cross-cell handwritten forms and photographed/creased forms
  reach high CER. Out of scope without an HTR backend + dewarping.
- **Redaction-induced paragraph breaks (native pages).** The PDF text layer returns text
  either side of a mid-paragraph redaction as separate blocks; joined with
  `\n\n` they emerge as two paragraphs. Belongs to a downstream cleaning op.
- **Native-text footer whitespace artefacts.** Sub-glyph kerning yields
  `3|P age`-style footers on the native path; verbatim-text policy means
  extraction won't normalise — a downstream cleaning op's job.
- **Verbatim policy.** Extraction emits producer bytes; OCR/font-map errors
  (letterhead typos etc.) are preserved. Systematic cleanup belongs to a
  downstream `clean_text`-style op, not extraction.
- **Console base tokens are a best-effort fill, not a verified match
  (`ui/src/app.css`, merge 4).** `DESIGN.md` states the console-specific
  tokens explicitly (surfaces, status fills, `--font-mono`, both themes) but
  names several base tokens — `--background`/`--foreground`/`--border` in
  light mode, `--muted-foreground` in light mode, `--ring` — as
  DeepCivic-inherited without restating their hex; that table lives in
  DeepCivic's own `DESIGN.md`, in a separate repository this one has no
  access to. The values shipped satisfy what `DESIGN.md` *does* state (the
  measured-contrast table's dark-mode figures, lime/purple never doubling as
  page background or body text, `#18182a` as the pill near-black) but are
  not verified against DeepCivic's actual values. Reconcile once that
  repository is reachable from a Womblex session.
- **`DESIGN.md`'s light theme is underspecified, and two of its rules do not
  hold there.** The system is dark-first, and its stated rules were checked
  against the dark page; measured against the light one they break. Recorded
  because merge 5's `StatusPill` is the first component that would inherit
  the contradiction. All figures below are WCAG 2.1 ratios computed from the
  tokens `DESIGN.md` itself lists.
  - **The status-pill rule holds only in dark mode.** "A status pill or dot
    is always a solid `--status-*` fill with near-black `#18182a` text …
    identical in light and dark mode and needs no per-theme override." Over
    the *dark* fills that is exact — 4.64:1 (failed) to 13.18:1 (running),
    matching the doc's stated "4.6:1 through 13.1:1". Over the *light* fills
    the same near-black gives pending 1.47:1, warning 3.48:1, done 3.48:1,
    failed 3.53:1 — only running passes. Those light values are tuned for
    **white** text (4.94:1 to 11.88:1), so the pill does need a per-theme
    text colour, contrary to the rule. *Resolved 2026-10:* the text colour
    flips per theme (white on the light fills, near-black on the dark ones),
    with lime running near-black in both; `DESIGN.md` states the revised rule.
  - **Lime cannot be an active-nav *text* colour in light mode.** "Active
    item takes a lime left rule + lime icon" works on the dark nav
    (`#c8ef35` on `#22222e` = 11.86:1) but not the light one, where
    `--surface-raised` is `#ffffff` and the same lime computes to 1.32:1.
    The shell therefore uses lime for the 2px rule (a fill, always legible)
    and carries the active state in the label's weight and `--foreground`
    (13.72:1 dark / 17.46:1 light).
- **Only native PDF tables have cell boxes.** DOCX, OCR-reconstructed and spreadsheet-print table cells carry a null `bbox` (`extraction.md`). OCR tables could take one from the `table_grid` bands and columns if a consumer needs it.

## Deferred / backlog

- **Model plugins: swappable models without a schema change.** *Shipped
  2026-10.* Requirement: 24 in
  [functional_requirements.md](functional_requirements.md). Design calls:
  - **Entry-point registry.** Shipped as `utils/model_registry.py`, not the
    originally-planned `womblex/plugins.py`: it resolves each slot from one
    `womblex.models.<slot>` entry-point group per slot (not the per-slot
    group names first planned) via `importlib.metadata`. The built-ins
    register through the same registry, so nothing has a special path. The
    first merge records here why a registry is justified despite the "no
    strategy patterns" rule.
  - **One baseline table.** Shipped as `model_registry.DEFAULT_MODELS` (O3,
    not part of the original four merges) rather than a module of its own:
    changing the default group edits only that table, and `is_default_group`
    lets a caller — the benchmark — ask whether a set of selections matches
    it.
  - **Models by name, never path.** Plugins resolve through
    `utils/models.resolve_local_model_path`; a `womblex.model_roots`
    entry-point group adds search roots.
  - **Readers declare their output shape** (`regions` or `markdown`),
    replacing `LLM_OCR_ENGINES`; `is_llm_engine` now reads the `markdown` trait. The reader cache is keyed
    by engine plus a frozen copy of its options.
  - **Layout slot (merge 2).** `layout.model` / `options` (formerly
    `extraction.ocr.layout_model` / `layout_options`, removed in L3b, and
    `redaction.layout_model` / `layout_options`, removed in L3c) name a
    registered analyser (default `pp-doclayout-m`); options pass to its
    factory unchanged. The two selections are independent because the
    standalone `redact` commands hold only `RedactionConfig`. An unknown name
    raises before the layout pass instead of reading as a missing model.
    `check_layout_regions` (in `ingest/interfaces/protocols.py`) is the
    conformance check against `LAYOUT_BLOCK_TYPES`; a non-conforming model is
    logged and the page falls back to full-page text.
  - **Tokeniser and spellfix-dictionary slots (merge 4).**
    `chunking.tokenizer` / `tokenizer_options` and `spellfix.dict_name` /
    `dict_options` name registered models. A tokeniser factory returns a
    Hugging Face id or a `(str) -> int` counter; a dictionary is any object
    with `lookup(word)`. Because config names registered models only, a bare
    Hugging Face id in `chunking.tokenizer` is now an error listing the known
    names; the built-in `huggingface` model names any id by option
    (`tokenizer_options: {name: org/tok}`), which must still resolve locally
    (`tokenizer_available`) or chunking is skipped, and `hunspell` does the
    same for another dictionary directory. The dictionary is built before the
    first document, so bad options or a missing directory fail up front. The enrichment token-budget tokeniser stays
    out of scope.
  - **Profiling confidence sampling stays on PaddleOCR** (`morphology.py`),
    because its thresholds were calibrated there.
  - **Prerequisite — shipped.** `config.py` (920 lines) was split
    mechanically into the `config/` package (#132).

- **Trust baseline and recipes.** *In progress 2026-10; the trust baseline and
  stored workflows shipped.* Scope and the dependency assessment are recorded
  in decisions-operations.md ("Trust and recipes") and above ("Dependencies assessed"). Outstanding work:
  [plan-trust-and-recipes.md](plan-trust-and-recipes.md).
- **Permissive dependencies — Apache-2.0 compatibility.** *Shipped 2026-10
  (layout swap #130, layout stage L3, pdfium default F1, PyMuPDF removed F2).*
  The previous layout model's dependency and PyMuPDF were not licence-compatible
  with Apache-2.0. Layout is an Apache-2.0 PP-DocLayout ONNX model on
  `onnxruntime`; PDFs and images open through a womblex-owned `ingest/pdf/` seam
  onto pypdfium2 + pdfplumber. The between-backend parity gates cleared on the
  synthetic set at F1, and the MuPDF values they measured are pinned in
  `tests/test_pdf_block_parity.py` and `tests/test_pdf_text.py`. Left open:
  multi-column reading order, Phase 0's divergence tail, which no synthetic page
  reproduces (the ASX-page investigation is in
  [plan-post-gt-baseline.md](plan-post-gt-baseline.md)).
  - **Behaviour changes from the swap.** Calling `extract_text` directly can no
    longer open XPS, EPUB, MOBI, CBZ or SVG, nor an image format MuPDF decodes
    and Pillow does not; PDF and the common image formats remain, and WebP and
    AVIF are new. None of the dropped formats was reachable through the CLI or
    API. `content_digest` moved at F1 (text stays verbatim but comes from a
    different producer). Paragraph segmentation, dehyphenation and bold-based
    heading detection may shift slightly; `image_count` may count drawn images
    only, which moves the sub-page OCR gate; rendering anti-aliasing differs, so
    OCR output differs slightly; the table pass may be slower.
  - **The PDF library version is not in the run stamp.** `content_digest`
    depends on it, but the womblex version and `uv.lock` pin the library.
  - **Out of scope, kept for parity:** the unrotated-text versus rotated
    `page.rect` mismatch outside `spreadsheet_print`.
  - **Layout measurements at the swap (uncontrolled, no like-for-like run).**
    PP-DocLayout-M is 23 MB and runs at about 65 ms per CPU page. Scored with
    the benchmark's DocLayNet scorer: layout F1 0.297; the one real
    ground-truth table (`dense_text_548`) found; table regions on 5 FUNSD
    forms: 3. Table-class recall fell from 50% to 25%, and `dense_text_548`
    gave three table regions where the ground truth has one, with a `chart`
    box (mapped to `figure`) almost identical to the `table` box. The
    DocLayNet fixtures hold one table and one picture, so they
    cannot separate models; the end-to-end suites are the real gate.
  - **Layout feeds three things only.** Table rects (to `reconstruct_table`),
    figure and table rects (redaction exclusion zones) and the dominant
    region's kind: on a successful page the non-table regions collapse into
    one OCR text block.
  - **No PDF-backend registry slot.** The registry is for swappable models;
    a backend slot would be a toggle. `open_document` takes a path and nothing
    else.
  - **Layout is its own stage (L3, shipped 2026-10)**, so a layout model can
    be tuned and measured on its own ([layout.md](layout.md)). One model
    (`layout:`) runs once per selected page in the extraction batch, after
    profiling and before OCR; OCR's table rects and block types and
    redaction's exclusion zones read its regions, persisted in
    `*.layout_regions.parquet`. Before it, layout ran twice and unpersisted
    (inside OCR at the OCR dpi, inside redaction at 150 dpi), so a page could
    get two models at two resolutions and neither result could be inspected
    or rerun. The per-consumer `layout_model` / `layout_options` keys were
    removed outright and are refused at config load.
    - *Rejected: persist OCR regions and reconstruct tables downstream.*
      OCR-page table elements would then depend on a downstream stage and
      need an element-structure overlay.
    - *Page scope `consumers` by default:* OCR-routed pages (a `markdown`
      engine bypasses layout) plus, with the redaction filter on, pages
      without vector redactions. Both are known before layout runs, so there
      is no circularity. `all` is for measuring a model. Which consumers are
      on is part of the fingerprint, since it changes the pages analysed.
    - *Normalised boxes, rendered at the OCR dpi.* Each consumer maps them
      onto its own render, replacing the old OCR-versus-layout
      pixel-dimension check. The A2 deskew refusal stays with OCR, which
      alone knows it deskewed.
    - *A rerun replaces only the sidecar.* Nothing reads layout after
      extraction, so no stale-output tracking is added: the elements
      footer's fingerprint against the sidecar's is the out-of-date signal,
      and applying a new model to tables or redaction is a re-extract.
      `layout` stays out of `DOWNSTREAM_STAGES`, as a rerun is deliberate.
    - *A failed page is a status row, not a failed document.* OCR falls back
      to whole-page text; redaction runs the page unfiltered and records it
      (`unfiltered_redaction_pages`), over-reporting rather than missing one.

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
  tree **from the doc, with no API call** (`semchunk.py`); chunk
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
  live `kanon-2-enricher` key with semchunk 4.0.0 / isaacus 0.20.0; the
  end-to-end path is now a standing regression test
  (`tests/test_pipeline.py::TestAiChunkingLive`, live like the enrich/embed
  tests — no mocks, skips cleanly without `ISAACUS_API_KEY`):**
  1. ✅ `document.text` is byte-identical to the input narrative (the offset
     basis the whole reuse rests on).
  2. ✅ A rehydrated `Document.model_validate_json()` satisfies semchunk's
     `isinstance(text, ILGSDocument_Runtime)` runtime check (same SDK class
     identity, `semchunk.semchunk`), and semchunk chunks the rehydrated
     Document down the AI path (no API call).
  3. ⚠️ *Partially.* On a normal-sized doc, `overflow_strategy="auto"` returns
     `document.text` == full source and all mention spans index within bounds
     (verified: 12 real spans, correct slices). **Residual:** the true
     multi-prechunk stitch (a doc exceeding the enricher context window) was
     not exercised — close this with a large real fixture before declaring
     production-ready.

  **Status:** shipped 2026-06 (offline seam tests + a live AI-chunking test over
  the vendored Throsby fixture, `tests/test_pipeline.py::TestAiChunkingLive`).
  The `chunk_batch` Document-acceptance change was the main risk and is covered
  by the byte-identity guard above. Remaining caveat: the gate-3 large-document
  residual — until a doc exceeding the enricher context window is exercised,
  treat very large inputs under `chunking_model` + reuse as unverified.
  *Superseded:* a homegrown boundary-hints layer (enrichment spans as preferred
  splits and no-split zones fed to semchunk) — semchunk 4 absorbed the concern.

- **Local enrichment fallback.** Without Isaacus there are no graph spans, so
  PII has only the opt-in backstop and AI chunking falls back to token splitting.
  A config-selected local provider (spaCy / fine-tuned NER) populating
  `EnrichmentResult` would serve both offline, at lower quality. *Wanted
  (2026-10)*: a GLiNER-based provider is in development by the maintainer, so
  no interim provider is added. *Shipped (2026-10, trust plan items 1.2–1.3):*
  an `enrichment` slot in `utils/model_registry.py`, default `kanon-2-enricher`,
  with providers installed through `womblex.models.enrichment` entry points like
  the other slots. A factory's `isaacus` and `returns_document` traits answer the
  pre-launch gates before anything is built; the provider's own limits drive the
  run. AI chunking follows the enrichment it reuses: with a provider that
  returns no Document, `chunking.chunking_model` is set aside (warned) and
  chunks come from semchunk's local token splitting. Isaacus's open-source
  parts (semchunk, the vendored Kanon-2 tokeniser) run locally; a paid call is
  made only where the configured provider asks for one, never as a fallback. The `embedding` slot
  followed (item 1.4): default `kanon-2-embedder`, a provider declaring texts per
  request and network need. *The provider contract:* text in, one `EnrichmentResult` per
  text, offsets into the text sent, results independent of batch composition, and
  labels mapped to the existing taxonomy inside the adapter. *Each provider
  declares its limits:* texts per request, token ceiling, token counter, whether
  it needs the network, and whether it returns a Document that AI chunking can
  reuse. Womblex keeps packing, splitting, the table pass, the sidecars, evidence
  and chunk mapping, so a provider changes who recognises entities and nothing
  downstream. The Isaacus-only features (AI-chunking reuse, `persist_document`)
  stay as they are, with token splitting as the fallback for a provider that
  returns no Document. *Rationale:* a pure Isaacus pipeline is the Isaacus SDK
  used directly; the slot keeps Womblex's own pipeline from hard-wiring one
  provider. Embedding follows the same pattern, and the GLiNER provider is the
  first alternative once the slot exists.

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

  **Consumption — shipped.** `process/text_overlay.py` (`load_overlay` /
  `apply_overlay`) resolves the normalise / spellfix element-text layer
  selected by one pipeline setting (`processing.text_source`:
  `elements` | `normalised` | `spellfix`), applied before reassembly at both
  the `chunk` and `enrich` sites so they share one coordinate space
  (`money.text_source` outranks the pipeline default for the `money` stage;
  `spellfix` chains off `normalise` when both run). `score --text-source`
  reports CER of extraction vs normalisation over the same ground truth.

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
overlaid elements). A declared non-`elements` overlay that is missing fails
loudly at these transform sites and at the render path (`build_ground_truth`)
alike, rather than producing output under a layer it did not apply.
- **Inline-per-span source redactions (#C).** Page-prefix `<REDACTED>` is in
  place; inline-per-span placement needs bbox-to-text character mapping (raster
  path now has per-word bboxes; native path needs a text-to-bbox map).
- **Redact-stage checkpoint unification.** `redact` keeps its own JSON
  checkpoint rather than the `CheckpointManager` surface used by the other
  per-stage CLIs.
- **Enrichment graph-edges sibling.** *Shipped 0.2.0.* `enrich_shards` writes
  `*.graph_edges.parquet`; `analyse/graph_refresh.py` rebuilds its chunk edges.
- **E2E composition of enrich/link/pii under `womblex run`.** *Superseded
  0.5.10:* `womblex run` is extraction-only by design; downstream stages run via
  `run-stage` / `enqueue-stages`.
- **`overflow_strategy` pass-through on `EnrichmentConfig`.** *Shipped 0.2.0.*
- **No projection carries narrative ↔ table document order into chunks.**
  *Option (b) complete 2026-08 — anchor plus the offset map that makes it
  usable; the interleaved-projection question (option a) stays open.*
  `build_chunk_input`
  (`chunker.py`) splits the element stream into two disjoint projections:
  `reassemble_narrative` takes TEXT_KINDS only (tables skipped) and
  `collect_tables_from_elements` takes `kind='table'` only.
  `chunk_batch` then runs two separate semchunk calls and appends all table
  chunks after all narrative chunks. A table chunk's `start_char`/`end_char`
  index *its own markdown string*, not the narrative, so its only
  positional anchor is `page_start`/`page_end` — page granularity on PDFs, and
  nothing at all on DOCX/spreadsheets, where `Element.page` is `None` by design
  (see the nullable-page note in `store/output.py`). No `elem_order` link
  recovers it: chunks join to elements by offset-range overlap, "not via
  `elem_order`" (`store/output.py`), and table-chunk offsets are in a
  different coordinate space.

  Worth reviewing because `elem_order` was introduced precisely to preserve
  this: the pre-element schema (one parquet per batch, nested struct lists for
  tables / forms / images / text_blocks) lost the on-disk order between a table
  and the paragraph before it, making the budget-statement class of documents
  unrepresentable, and forced spreadsheets to emit one ExtractionResult per row.
  `elem_order` and first-class spreadsheet cells replaced it, and the DOCX fixture in the suite is a portfolio budget
  statement (`test_output.py`). Extraction preserves the interleaving; the
  projection that feeds Isaacus discards it. Nothing is lost on disk —
  `*.elements.parquet` retains `elem_order` — so this is a projection gap, not
  data loss.

  Not obviously a defect: separate table chunking is deliberate (table markdown
  chunks in token mode with no overlap, because semantically chunking a
  markdown grid is meaningless), so folding tables into the narrative stream is
  a design change with real trade-offs. Two shapes were weighed: (a) a third
  document-order projection alongside the existing two, or (b) an `elem_order`
  anchor on table chunks, letting consumers reconstruct order without
  disturbing the chunking strategy.

  **(b) shipped 2026-08.** `CHUNKS_SCHEMA.elem_order` is populated for table
  chunks only. Chosen over (a) because it changes no coordinate space —
  narrative chunks, the enrichment input and the mention↔chunk offset mapping
  are untouched, so no existing offsets shift and no re-enrichment is needed.
  `read_chunks` back-fills nulls for older shards.

  **The anchor alone did not recover the order, though** — as first shipped it
  had no consumer, and could not have had one. The recipe recorded in three
  places ("sort narrative chunks by `start_char` and table chunks by
  `elem_order`") yields two orderings in two *incomparable* key spaces:
  interleaving them needs the narrative offset each `elem_order` sits at, and
  nothing exposed it (`reassemble_narrative` returns only `page_breaks`, and
  chunks otherwise join to elements by offset overlap, never by `elem_order`).
  `page_start` is a page-granular workaround on PDFs and no workaround at all
  on DOCX/spreadsheets, where `Element.page` is `None` by design — precisely
  the interleaved budget-statement case the anchor exists for.

  **Completed 2026-08 by exposing the offset map.** `chunker.element_spans`
  returns `(elem_order, start, end)` per narrative-bearing element, and
  `chunker.chunks_in_document_order(rows, spans)` does the interleave — a table
  anchors at the narrative offset where the element after it begins, sorting
  before a narrative chunk that starts there (the table precedes that text),
  with sheets last (no anchor, no narrative to be ordered against).
  `element_spans` and `reassemble_narrative` are both views over one internal
  `_narrative_pieces` generator, so the coordinate space cannot drift between
  them. The ordering lives in the library rather than in each consumer, per the
  corpus/library split: the benchmark must not derive an ordering of its own.
  Still a projection, not a chunking change — no schema change, no offset
  shifts, and the spans must be read under the same `text_source` overlay the
  chunks were written under.

  **Remaining consumer work.** The Chunk Inspector still renders in
  `chunk_index` order (`ui/readers.py`), i.e. all narrative then all tables, so
  every table shows detached at the end of the document. Wiring it up needs the
  reader to pull `*.elements.parquet` per `source_hash` (local *and* remote
  paths) and to know the run's `text_source` — a separate change.

  **(a) is not pursued (decided 2026-10).** Search relevance is to be raised
  by other roadmap items, not by folding tables into the enrichment input. The
  cost analysis below is kept as the record of why it was weighed.
  Folding tables into `reassemble_narrative` would change the coordinate space
  for all four of its consumers at once (`chunk_stage`, `enrich_stage`,
  `money_stage`, plus the byte-identity claim in `records.py`), which means:
  every existing shard's offsets are invalidated (they live *inside* the
  enrichment/money/chunk sidecars, so this cannot be a sidecar migration);
  `money_stage`'s `narrative` locus would double-count every amount its
  `table_cell` locus already finds, needing an explicit suppression rule; and
  table markdown is token-dense, so the enrich token budget and over-ceiling
  split path get hit far harder — an unmeasured Kanon-2 cost increase. Only
  worth paying if the goal is specifically that the *enricher and embedder* see
  tables in surrounding context, which is a retrieval-quality hypothesis and is
  cheap to test offline (build the interleaved string for a sample, compare
  retrieval) before committing to any schema change.

  **The consumer it blocked.** The benchmark's tabular and financial review
  surface is a document-order spine with tables anchored in position, and the
  ANAO and DFAT ground truth is built through it. `chunks_in_document_order`
  is that spine — the benchmark deliberately does not derive an ordering of its
  own, since a review file it ordered itself would have a reviewer correcting
  the benchmark's guess instead of the pipeline's output. Option (a) is not
  what it was waiting on: it needs tables *positioned*, not tables folded into
  the enrichment input.

### Modality routing — prose / tabular / register as the primary fork — *proposed (2026-06), long-range; out of scope of the trust-and-recipes plan (2026-10)*
Extraction currently makes its coarsest cut by *format* (`extract_text` splits
non-PDF SPREADSHEET/DOCX/TEXT vs PDF, `extract.py`), and the prose-vs-tabular
decision for a **spreadsheet trapped in a PDF** is made *late and inside* the
per-page orchestrator (`extract_with_plan`, `orchestrator.py`) — gated by
`qualify_for_spreadsheet_print`, after per-page profiling, interleaved with prose
dispatch, and threaded through element assembly as `is_spreadsheet_print` /
`include_tables=not is_spreadsheet_print`. So the PDF orchestrator is *partly a
spreadsheet engine*, and tabular content is graded by a prose metric (CER/WER)
and reconstructed by prose-shaped layout heuristics (the layout stage measures
~30 % F1 with heavy over-segmentation). The proposal: make **modality**
(`PROSE` / `TABULAR` / `REGISTER`) an explicit fork decided **once, before** the
orchestrator, orthogonal to the 12 `DocumentType`s (not a replacement — `doc_type`
stays the summary attribute it already is).

**Options reviewed.**
- **A — status quo (do nothing).** Leave the tabular decision buried in the
  orchestrator. *Rejected:* the coupling is the observed smell — the orchestrator
  carries spreadsheet state, and tabular is scored/reconstructed as if it were
  prose. It works but doesn't scale as more tabular shapes land.
- **B — flatten every input to PDF, then one extraction path.** Normalise DOCX /
  XLSX / images to PDF up front so there is a single pipeline. *Rejected — this is
  the harmful reading of "narrow the state".* It narrows on *format*, not
  representation: it converts near-zero-CER native text (DocxExtractor reads
  OOXML in document order; SpreadsheetExtractor reads exact cells) into a
  layout-recovery + OCR problem (0.2–0.5 CER, ~30 % layout F1), paginates
  unbounded grids arbitrarily (the very problem `spreadsheet_print.py` exists to
  undo), and violates the verbatim-boundary policy by round-tripping bytes
  through a renderer. It buys a single code path at the cost of a measurable
  regression on exactly the documents that currently extract cleanly.
- **C — modality router, document-level (chosen direction).** Narrow on
  *representation*, not format. A single `classify_modality(profile, profiles,
  filename) -> Modality` at the `extract_text` boundary routes the whole doc:
  `PROSE` → orchestrator / Docx/Text extractors (scored by CER/WER); `TABULAR` →
  one grid-first path unifying native spreadsheets, CSVs, *and* spreadsheet-print
  PDFs (scored by `utils/tabular_metrics.py`, not CER); `REGISTER` → bypass the
  NLP pipeline (already the case for G-NAF / ABN / geospatial — the existing
  proof the pattern works). The element stream stays the narrow waist: tabular
  still emits `sheet_cell` / table elements, so **nothing downstream changes**.
- **D — per-page modality from the outset.** Classify modality per page so a
  mixed FOI bundle sends its manifest pages tabular and its cover-letter pages
  prose. *Deferred, not rejected:* it is the honest end-state given the existing
  per-page philosophy, but it is the hard ML part and should land last, on a
  clean structure, only once fixtures show doc-level is insufficient.

**Why C over B/D.** C moves the fork *earlier* (before orchestration) without
collapsing modalities into one lossy format (B's mistake) and without paying for
per-page classification before it is shown to matter (D). It confines the change
behind one classifier + the element-stream contract, so it is measurable against
the accuracy suite and reversible.

**Phasing (impact-to-effort).** (0) Lock the current accuracy numbers as the
regression baseline and add ≥1 labelled spreadsheet-print PDF + its matching
XLSX/CSV to the womblex-collection fixtures — fixtures are the bottleneck (the
same GT gap that leaves the Throsby/Isaacus report sections unpopulated).
(1) Lift the fork: extract `classify_modality`, reproduce today's behaviour
byte-for-byte, route `TABULAR` to a thin `extract_tabular()` that calls existing
code — pure seam-creation, near-zero risk, the highest-leverage step. (2) Drain
the orchestrator: delete `is_spreadsheet_print` / `spreadsheet_tables` /
`include_tables=...` now that tabular routes pre-orchestration; the orchestrator
goes back to per-page prose/OCR only. (3) Unify tabular extraction under one
grid-first path scored by `tabular_metrics`. (4) *(deferred)* per-page modality
(Option D) if measured to matter. The load-bearing invariant throughout: the
element stream is the contract, so downstream stages are untouched by
construction.
