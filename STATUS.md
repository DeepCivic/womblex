# STATUS — Womblex extraction quality, 2026-05-20

Codebase changes shipped against the ACT Early Childhood Incidents
corpus. Pairs with [stories/STATUS.md](../../../stories/STATUS.md) for
the corpus-side extraction-quality view.

> **Note (2026-05-16 refactor).** The extraction output shape was
> refactored to an element-stream + typed sidecars layout
> (`*.elements.parquet`, `*.table_cells.parquet`,
> `*.form_fields.parquet`, `*._manifest.parquet`). See
> [CHANGELOG.md](CHANGELOG.md) "Changed — BREAKING" and
> [docs/extraction.md](docs/extraction.md) for the canonical reference.
> The `tables[]` / `text_blocks[]` language in the Phase 5 section
> below describes the pre-refactor schema; findings still hold, but
> on-disk shape now lives in the four sibling parquet files above.

## Snapshot

| metric | value |
|---|---|
| unit tests | All unit tests pass. Fast-suite snapshot (excludes long-running accuracy benchmarks `test_fixture_accuracy.py`, `test_integration.py`, `test_womblex_collection_accuracy.py`): 484 passed, 6 skipped, 24 deselected in ~3 min. Headline per-file counts post K-cluster: `test_extract` 37 (was 19; +18 from `TestClassifyNativeBlock` / `TestFormLabelDenylist` / `TestYoloLabelMapDefault` / `TestPageBreakEmission`), `test_chunker` 60, `test_pii` 37, `test_pii_enrichment` 33, `test_redaction` 58 (was 55; +3 from `exclude_rects` coverage). |
| labels CER vs production parquet (`womblex score`) | hybrid 0.520 · native_with_structured 0.044 · scanned_machinewritten 0.051 · scanned_mixed 0.217 — element-stream reassembly preserves text fidelity vs pre-refactor pipeline output. **Hybrid 0.520 is misleading (n=2 average dragged by `00729-Papilio-Barton p4`, an 8-char near-blank page where GT uses full-width Unicode `ｐａge` and predicted produces ASCII `Page` — measurement artefact, not quality gap). The substantive hybrid sample (`00979-SCN-REDACTEDP01 p0`) sits at CER 0.040.** |
| corpus re-extraction (2026-05-17) | 2,626 source files · 3h 5m · 0 failed · 424 sibling-parquet files (106 batches × 4) |
| `kind='table'` contamination | 2,791 elements pre-fix → 172 post-fix (94% reduction); 3 residual conf=0.60 fabrications, ~10% borderline at conf=0.80, real manifests intact |
| `meta` map carries doc/table context | Verified on all 3 spreadsheet-print manifests (`context_213A reference`, `context_Element #`, `context_Text from motion`, `context_motion`) |
| raster-path layout filter | YOLO COCO regions consumed as exclusion zones via `RedactionConfig.use_layout_filter` (default true). Vector path unchanged. See [CHANGELOG.md](CHANGELOG.md) "Historical engineering notes → Detector — raster-path layout filter (2026-05-22)" for design + cost. |
| accuracy-doc generators | Refreshed 2026-05-22 to describe orchestrator + element-stream + vector-first detector + YOLO exclusion zones. `EXTRACTION.md` / `REDACTION_HANDLING.md` / `PII_CLEANING.md` regenerated cleanly. |
| non-`table` element kind audit | Surfaced 2026-05-22 (90,843 elements audited). Eight concrete fixes K1-K8 tracked in "Open follow-ups". Headline: `kind='signature'` 100% mis-classified (442 closing-phrase matches, no signatory blocks); `kind='figure'` 65% mis-classified on scanned pages (1,044 of 1,600); `list_item` / `caption` / `page_break` declared but never produced. (K2's "all-zero bbox" claim turned out to be a probe-formatter artefact — bboxes are populated for all native-text kinds; real narrower issue is OCR-derived forms with no bbox, tracked as K2′.) Full audit in `stories/STATUS.md`. |
| element-kind audit fix cluster | K1 / K3 / K4 / K5 / K7(a) / K8 landed 2026-05-22. K2 retracted 2026-05-23 as a false finding (probe formatter artefact). **K2′ and K7(b) landed 2026-05-25** — OCR per-region bboxes wired through to the form-pair extractor; DocLayNet YOLO11 nano swap (`yolo11n_doc_layout.pt`, MIT) as primary layout backend with COCO fallback. New `footnote` ElementKind. K6 (caption detection) closes as a side effect of K7(b). |
| post-K-cluster corpus re-extraction | Completed 2026-05-23 (2h 35m · 2,626 succeeded · 0 failed). All K-cluster effects landed cleanly: `signature` 442→0, `list_item` 0→4,015, `page_break` 0→6,561, `header` 0→335, `form` 5,183→4,391 (−792 spurious), `figure` 1,600→1,587 (small — K7(a) was bounded by COCO's explicit screen-class mappings, not just the unknown default). Strategy distribution unchanged, all known limitations preserved (no regressions). Text fidelity preserved: per-strategy mean CER identical to pre-re-extraction (hybrid 0.520, native_with_structured 0.044, scanned_machinewritten 0.051, scanned_mixed 0.217). |
| i1b corpus re-extraction (DocLayNet, 2026-05-25) | Run `run-20260525T214943Z`, 2,626 succeeded. Element-kind deltas vs post-K-cluster (i1a) baseline: `figure` 1,587→1,200 (−387), `list_item` 4,015→5,403 (+1,388), `caption` 0→10 (new producer), `form` 4,391→4,201 (−190), `footnote` 0→net-new kind. Per-strategy CER unchanged. Single post-i1b code change is `paddle_ocr.py` imgsz tuning (1280→832) — measured quality-neutral, redaction regions + CER identical between resolutions. No further corpus re-extraction required at the extraction stage. |
| shard integrity + verify-shards | New `store/shard_audit.py` + `womblex verify-shards` CLI (E1/E2). Resume-time integrity scan default-on; corrupted batches archived to `.corrupt`, affected `doc_id`s re-extracted. `MANIFEST_SCHEMA` gained `doc_id` (backward-compatible read). Closes the post-write filesystem-corruption class that prompted the i1b batch-0087 incident. 26 new tests. |
| i1b element-kind audit (E4, 2026-05-26) | 1,200 `figure` elements: **88% (1,056) are full-page OCR mis-classifications.** _(Root-cause site corrected 2026-05-30: the producer is the dominant-region fallback in `_layout_blocks_and_tables`, **not** `_ocr_image_regions` as first written here — see K9-fig.)_ Cluster candidates surfaced as K9-fig (1,056 figure FPs; the "10 caption FPs" were a non-issue — caption ∈ TEXT_KINDS), K10-list (1,134 of 5,403 list_items lack a leading marker — DocLayNet over-fires on letterhead/address blocks). **Was deferred pending downstream signal:** wins are material only if I2 (chunking) filters by kind. I2 confirmed it does — K9-fig landed 2026-05-30 (figure 1,200→154, 16 lost docs recovered); K10 stays cosmetic (list_item ∈ TEXT_KINDS, reaches chunks). |
| residual conf=0.60 table fabrications (E5, 2026-05-26) | Re-checked under DocLayNet: 01093 p2 / 01094 p2 / 01349 p3 still carry one `kind='table'` with `confidence=0.60`, `extractor=native_text`, `text_len=0`. Source is PyMuPDF's `table_finder` heuristic on the native PDF text layer — DocLayNet doesn't touch this code path. Closing these requires a new cross-validation step (reject native_text low-conf tables when DocLayNet predicts no Table region at the bbox). **3 docs out of 2,626 — accepted limitation, not worth pursuing.** |
| scanned_mixed redaction cohort (E6, 2026-05-26) | Region counts across the 11-doc scanned_mixed cohort under DocLayNet exclusion zones (`max_area_ratio=0.05`): 02737 = 3 regions (vs COCO baseline 10 — confirms K7(b) precision win); 6 of 11 docs detect 0 regions across 2-25 pages each (likely correct: `R-*-Direct-Complaint.pdf` CRM exports without narrative redactions); 01538 p3 = 29 regions stands out (dense or FP — unverified). Raw data in [/tmp/e6_scanned_mixed_redaction.json](/tmp/e6_scanned_mixed_redaction.json). |
| extraction stage closure (2026-05-26) | `02737 p0` was reviewed against its labels-packet PNG. Diagnosis: photo of a printed form filled out by hand, taken on an angle, paper creased. The 988-char extraction is the printed labels; the missing ~960 chars are handwritten responses RapidOCR's PaddleOCR ONNX backend cannot read. **Architecturally bounded — chasing this would require HTR (e.g. TrOCR) + dewarping, which is a major capability for a corpus that is overwhelmingly typed. The extraction stage is done; ship I2 (chunking) next per the per-stage execution model.** |
| I2 build landed (2026-05-27) | `chunks.parquet` schema + writer in [`store/output.py`](src/womblex/store/output.py); single `chunk_batch` entry point in [`process/chunker.py`](src/womblex/process/chunker.py) (deleted per-doc `chunk_text` / `chunk_texts_batch` / `chunk_document` wrappers — semchunk handles cross-doc batching natively when handed a list); per-stage `chunk_shards()` over an existing shard dir in new [`process/chunk_stage.py`](src/womblex/process/chunk_stage.py); `womblex chunk --shards <dir>` CLI with independent per-stage `CheckpointManager` + default-on resume integrity scan (`--no-verify-resume` to skip); chunks-side reconcile (`reconcile_chunk_checkpoint_with_shards`) in [`store/shard_audit.py`](src/womblex/store/shard_audit.py) — wired into the CLI resume path so a corrupt `*.chunks.parquet` drops its docs from the chunk-stage checkpoint and re-chunks them, mirroring the elements-side E1 pattern (without touching the elements shards). E2E `run_chunking` rewired to the same `chunk_batch` — both paths reassemble narrative from `elements` via shared `build_chunk_input(...)`. Behaviour change: chunks no longer reflect in-memory `pages[i].text` mutations from PII/redact-blackout stages (matches the to-disk policy that elements is canonical; PII/redact-blackout sidecars in P3/P6 will reattach via their own joins). 30 new test_chunker tests, 9 new test_output tests, 5 new TestCmdChunkShards tests (incl. corrupt-chunks resume recovery), 9 new chunks-audit tests. |
| Pre-I2 cleanup (2026-05-27) | Purged corpus-side legacy artefacts: `output/documents/` (pre-I1 layout), `output/run-20260523T132457Z/` (i1a — superseded by i1b modulo K7(b)+K2′ which i1b has), `output-pre-kcluster-2026-05-17/` + matching `checkpoints-pre-kcluster-2026-05-17/`, top-level `checkpoints/act_eci_extract_checkpoint.json` (pre-I1). Corpus now has a single live run dir (`output/run-20260525T214943Z/` = i1b) — retention bookkeeping is uniform across all surviving outputs. |
| I2 corpus chunk run (2026-05-28) | `womblex chunk --shards run-20260525T214943Z/documents/` over the i1b extraction. 11,797 chunks across 2,610 of 2,626 completed docs (16 zero-chunk). _(The "16 zero-chunk = correct behaviour, no text-bearing elements" read here was **wrong** — overturned by K9-fig 2026-05-30: those 16 are full-page scans whose text was mis-tagged `figure` and excluded from chunking. After the K9-fig relabel + re-chunk all 2,626 docs chunk; total 12,506.)_ 9,933 narrative + 1,864 table chunks; median 3 chunks/doc, mean 4.5, max 948 (FOI master spreadsheet manifest, single source). Median chunk len 1,540 chars; 3 chunks corpus-wide are bare `"2"` page-number footers stranded between full-capacity neighbours (cosmetic). `has_redaction=False` everywhere as expected under per-stage execution (in-memory PII/redact-blackout no longer flows into chunks; P3/P6 sidecars will reattach via their own joins). |
| Doc-metadata sidecar scope decision (2026-05-28) | **P5/I4 dropped from library scope.** Filename-component parsing is corpus-specific (the ACT FOI release uses two distinct schemes plus a long tail of variants); per CLAUDE.md's corpus/library boundary it belongs in `stories/<corpus>/`. The library already preserves the raw `filename` on `_manifest.parquet`, and a corpus-side `*.doc_metadata.parquet` will compose with downstream Womblex stages via `source_hash` without further library changes. See "P5 — Document-metadata sidecar (dropped from library scope)" below for the full rationale + promotion criteria. |
| I5 landed — SemChunk wrapper audit (2026-05-30) | Audited `process/chunker.py` against semchunk 3.2.5. Pass-through 100% (`chunkerify` ×5 creation params via `create_chunker`; `Chunker.__call__` `overlap`/`processes`/`progress` via `_chunker_batch`, `offsets=True` pinned for page mapping). No semchunk surface reimplemented/shadowed. Sole finding: dead `ChunkingConfig.batch` toggle (no semchunk param, no consumer, stale pre-I2 description) — removed. Adapter boundary documented in `chunker.py`, `ChunkingConfig`, and `docs/extraction.md`; 3 upstream-default divergences annotated. Output byte-identical to I2 by construction (removed field read by no path). 98 tests pass. |
| I7 built + smoked; full run blocked on net (2026-05-31) | `womblex enrich --shards` + `womblex link --shards` landed (generic/library-first: `entity_type` schema, record-linkage matcher with alias → address-exact → OCR-tolerant token-set name-fuzzy, stdlib difflib; bundle-aware reference consumption; corpus-declared column-roles). New **`womblex embed --shards`** stage (Kanon-2 chunk embeddings → `*.embeddings.parquet`). Live 17-doc Artemis smoke: **16/17 → correct `SE-40002132`** (only R-03247, no org extracted, unmatched). 110 relevant tests green. **Full 2,626-doc enrich could NOT run — Isaacus endpoint unreachable (network outage ~14:00); corpus shards clean + run staged/resumable.** Env note: background Bash has no network here; only short foreground calls reach the API. |
| I3 landed — `womblex redact --shards` (2026-05-28) | `womblex redact` is now dual-mode (`--shards <dir> --pdfs <dir>` per-stage \| `--config` E2E), mirroring `womblex chunk`. Per-stage path writes `*.redactions.parquet` over an existing shard dir via the unchanged `redact.batch.annotate_redactions_for_shards` engine; `--pdfs` is required (detection rasterises pages). Legacy `annotate-redactions` retained as a back-compat alias routing through the same shared helper. Pure CLI-surface refactor, no behaviour change. 8 new CLI tests; `test_redaction.py` 66 passed. Checkpoint stays the engine's JSON `--checkpoint` (not yet unified with chunk's `CheckpointManager` — deferred P1 follow-up). |

## Open follow-ups

Tracked here so they don't drift back into "Deferred". Update or strike-through when resolved.

### Publishable-corpus track (P1-P7, planning 2026-05-23)

Captured from a corpus-publishability review against the ACT_EarlyChildhoodIncidents working dir. Drives the next end-to-end run of `stories/ACT_EarlyChildhoodIncidents/Documents/` and any future corpus published from Womblex. The corpus is the test case; every change here is library-first per CLAUDE.md "Corpus relationship to library".

Working notes for the run itself live in [`stories/STATUS.md`](../../stories/STATUS.md); this section tracks the Womblex-side code work required, sequenced so earlier items unblock later ones.

#### Execution model — per-stage, not end-to-end

**No end-to-end publishable run is planned for some time.** The corpus is being built one stage at a time. The workflow per iteration is:

1. Build the feature for the next stage (e.g. extraction, then chunking, then PII, …).
2. Run **that stage only** over the corpus.
3. Review the stage's output (compare against prior stage's sidecar where applicable).
4. If satisfied, talk through what the next stage should do given the output.

This means:

- **Per-stage CLI invocation is the primary workflow.** Each stage must have a first-class CLI entry point (e.g. `womblex chunk --shards <dir>`, `womblex pii --shards <dir>`) that consumes the prior stage's persisted sidecar and writes its own.
- **`womblex run` (the monolithic E2E command) is retained and remains first-class** for future Womblex users with simpler corpora who *do* want to run end-to-end. We are not deleting it. But it is **not** the path this corpus takes.
- **Stage outputs must be byte-identical** whether produced by per-stage invocation or as part of E2E `run`. Tests cover both paths.

Practically: the development items P1-P7 each ship two CLI surfaces — the per-stage entry point (for our use) and a guarantee they compose under `run` (for future users). A stage's "done" criterion is **both** modes pass, not just one.

#### Design principles for the publishable run

- **Stage independence.** Each downstream stage reads the prior stage's persisted sidecar and writes its own. No in-memory mutation handoff between stages — also no cross-stage configuration assumptions: a stage invoked on its own must not require knowledge of which stages ran before, only what's on disk.
- **Reviewability.** A human comparing a piece of content across stages must be able to join sidecars on `(source_hash, elem_order)` and see how each stage transformed it. Per-stage assessment is the explicit gate between iterations.
- **Verbatim invariant.** `elements.parquet` is never rewritten post-extraction. Mutations (redaction markers, PII rewrites, downstream cleaning) live exclusively as sidecars.

| ID | Status | Item | Notes |
|---|---|---|---|
| **P1** | partial | **Sidecar persistence + stage-aware checkpointing.** | I1 slice (run_id + retention plumbing) landed 2026-05-23. Per-stage sidecar writers (chunks, pii_spans, clean_text, redactions integration) remain. |
| ~~**P2**~~ | ✓ landed 2026-05-30 (I5) | **SemChunk wrapper audit (`isaacus-dev/semchunk`).** | Audited against semchunk 3.2.5: 100% pass-through, no reimplementation. Removed dead `ChunkingConfig.batch` flag; documented adapter boundary in code + `docs/extraction.md`. See I5 entry. |
| **P3** | open | **Configurable PII (entity types) + GT expansion.** | Depends on P1 *and* P4 (PII detection consumes the entity graph for consistent masking across name variants). |
| **P4** | built, QA gate (I7) | **Entity-link sidecar — Kanon-2 first.** | I6 (2026-05-30) settled **(b) Kanon-2 + register-matching**. I7 (2026-05-31) built `womblex enrich`/`link` per-stage CLIs + generic matcher; 17-doc Artemis smoke = 15/17 correct. **Paused at QA gate before the full 2,626-doc enrich run** (user request). Depends on P1; **upstream of P3**. |
| ~~**P5**~~ | dropped from library | ~~**Document-metadata sidecar (filename + FOI-master cross-ref).**~~ | Moved to corpus scope 2026-05-28 — filename parsing is per-corpus, not library-general. See I4 entry below and `stories/STATUS.md`. |
| **P6** | open question | **Redaction marker strategy.** | Decision deferred pending industry-standards review. |
| **P7** | partial | **Quality fixes — K7(b), K2′, downstream `clean_text` op.** | K7(b) and K2′ landed 2026-05-25 (DocLayNet swap + OCR-form bboxes). Downstream `clean_text` op (#B/#D) remains open — belongs to a later iteration. |

#### Iteration sequence (I1-I10+)

Concrete sequencing for the per-stage workflow above. Each iteration is one build slice + one **per-stage** corpus run + one assessment gate. Stop after any iteration; resume when ready. Per-stage CLI is the primary invocation path; `womblex run` (E2E) remains supported but isn't used to advance this corpus.

Rhythm per iteration: build → run **that stage only** against corpus → diff against prior stage's sidecar → decide whether to proceed.

- **I1 ✓ landed 2026-05-23 + hardened to 2026-05-26 — Sidecar pattern + run_id + resume-time integrity.**
  - Initial slice (2026-05-23): `dataset.run_id`, `processing.retention.policy/keep`, output layout `<output_root>/<run_id>/documents/`, checkpoint layout `<checkpoint_dir>/<run_id>/`, retention purge on fresh runs.
  - **B1 (2026-05-24)** — resume-time batch-numbering fix: `cmd_run` offsets `batch_num` from `checkpoint.last_batch` so resumed extraction never overwrites prior shards.
  - **E1 (2026-05-26)** — resume-time shard integrity scan (`reconcile_checkpoint_with_shards` in `store/shard_audit.py`). Corrupt batches archived with `.corrupt` suffix; their docs dropped from the checkpoint and re-extracted on the next pass. Default-on in `cmd_run`; opt-out via `--no-verify-resume`. Closes the post-write filesystem-corruption class that prompted the i1b batch-0087 incident.
  - **E2 (2026-05-26)** — `womblex verify-shards <run-or-shard-dir> [--compare-to <other>]` CLI for offline integrity inspection. `MANIFEST_SCHEMA` gained `doc_id` (backward-compatible read derives it from `Path(filename).stem` for shards written before the bump).
  - Process: re-run extraction stage over the corpus (i1a 2026-05-23, then i1b 2026-05-25 against DocLayNet — see snapshot row).
  - Assess: sidecar layout correct; extraction output byte-identical to current; retention purges as expected; corrupted shards self-heal on resume.

- **I2 ✓ landed 2026-05-27 (build) + corpus run completed 2026-05-28 — `womblex chunk` per-stage CLI + chunks sidecar writer.**
  - **Pre-I2 cleanup** (landed 2026-05-27): purged the pre-I1 `output/documents/` layout, the i1a `run-20260523T132457Z/`, and the `output-pre-kcluster-2026-05-17/` snapshot. Corpus now has a single live run dir (i1b `run-20260525T214943Z/`).
  - Build: `chunks.parquet` writer; per-stage `womblex chunk --shards <dir>` CLI; per-stage `CheckpointManager`; resume-time chunks-shard integrity scan (`reconcile_chunk_checkpoint_with_shards` in `store/shard_audit.py`); E2E `womblex run` composition preserved (both modes feed `chunk_batch` identical inputs via shared `build_chunk_input`).
  - Process (2026-05-28): chunked all 106 batches of the i1b extraction in one pass via `womblex chunk --shards run-20260525T214943Z/documents/`.
  - Audit: 11,797 chunks across 2,610 of 2,626 completed docs (16 zero-chunk). _(Superseded by K9-fig 2026-05-30 — the 16 were full-page scans mis-tagged `figure`, not a genuine no-text shape; post-fix all 2,626 docs chunk, total 12,506.)_ 9,933 narrative + 1,864 table chunks; median 3 chunks/doc, mean 4.5, max 948 (the FOI master spreadsheet manifest). Median chunk len 1,540 chars. `has_redaction=False` everywhere by construction (in-memory PII/redact-blackout mutations no longer flow through to chunks under per-stage execution; tracked in P3/P6 sidecars).
  - Known cosmetic edge: 3 chunks corpus-wide are bare `"2"` — standalone page-number footers stranded when both neighbouring narrative chunks were already at semchunk's size budget. Documented in `stories/STATUS.md`; not pursued.

- **I3 ✓ landed 2026-05-28 (CLI surface) — `womblex redact` per-stage CLI (consolidate `annotate-redactions`).**
  - Build: `womblex redact` is now dual-mode, mirroring `womblex chunk` — `--shards <dir> --pdfs <dir>` runs the per-stage redaction detection over an existing shard directory (writing `*.redactions.parquet` siblings); `--config <yaml>` runs the E2E extract+redact path unchanged. The `--shards`/`--config` mutually-exclusive group is `required=True`. `--pdfs` is mandatory in `--shards` mode because detection rasterises the source pages (unlike chunking, which works purely off the element stream). Legacy `annotate-redactions <shards> <pdfs>` retained as a back-compat alias — both routes call a shared `_run_redact_shards` helper, which calls the unchanged `redact.batch.annotate_redactions_for_shards` engine. **No behaviour change**: byte-identical sidecars, same JSON `--checkpoint`, same detector. 8 new CLI tests (`TestCmdRedactShards` ×6, `TestAnnotateRedactionsAlias` ×2); full `test_redaction.py` 66 passed.
  - **Note — checkpoint mechanism divergence (deliberate).** The redact stage keeps the engine's existing ad-hoc JSON checkpoint (`--checkpoint PATH`) rather than the `CheckpointManager` + `--checkpoint-dir`/`--dataset`/`--no-resume` surface used by `chunk --shards`. Unifying the two is a separate refactor (new behaviour) deferred out of this pure-plumbing slice; tracked as a P1 follow-up.
  - Process: run **redact stage only** over the corpus — not yet run (the existing Phase 2 `*.redactions.parquet` annotation pass from 2026-05-20 used the same engine; re-running via `redact --shards` would reproduce it).
  - Assess: `*.redactions.parquet` matches what the standalone `annotate-redactions` CLI produced today (same engine); spot-check the 11 scanned_mixed docs and the 3 conf=0.60 residual table pages.
  - Proceed when sidecar matches and known limits are preserved.

- ~~**I4 — Doc-metadata sidecar (P5).**~~ **Dropped from the library scope 2026-05-28.** Filename-component parsing (R-prefix vs numeric-prefix, FOI ref, case number, direction marker, etc.) is highly corpus-specific — the schema in `R-<doc_id>-<foi_ref>-<service>-CAS-<n>-<direction>-<doc_type>.pdf` is unique to this ACT FOI release; other corpora will carry their own filename conventions. Per the corpus/library boundary in CLAUDE.md, parsing logic of this shape belongs in `stories/<corpus>/`, not the library. The raw filename is already preserved on the manifest's `filename` column, so downstream stages have everything they need; any derivation is corpus work.
  - The corpus may still elect to write its own `*.doc_metadata.parquet` sidecar via a corpus-side script (and downstream Womblex stages will happily join against it on `source_hash` if it exists) — but no `womblex doc-metadata` CLI ships in the library. See `stories/STATUS.md` for the corpus-side plan.

- **I5 ✓ landed 2026-05-30 — SemChunk wrapper audit (P2).**
  - Build: audited `process/chunker.py` against semchunk **3.2.5**. **Pass-through coverage is complete** — `create_chunker` exposes all 5 `chunkerify` creation params (`tokenizer`→`tokenizer_or_token_counter`, `chunk_size`, `memoize`, `cache_maxsize`, `max_token_chars`); `_chunker_batch` passes all relevant `Chunker.__call__` params (`overlap`, `processes`, `progress`) with `offsets=True` pinned (Womblex needs char offsets for page mapping). No semchunk-native surface is reimplemented or shadowed (`table_to_markdown`, `_repair_redaction_splits`, `_page_for_offset`, `reassemble_narrative`/`collect_tables_from_elements`/`build_chunk_input` are all element-stream/marker concerns semchunk can't own). **One finding: `ChunkingConfig.batch` was a dead, Womblex-invented toggle** — mapped to no semchunk parameter, consumed by no code path, and its description referred to the pre-I2 per-doc-vs-batch behaviour I2 deleted. **Removed** (config field + docstring + commented `example.yaml` line). Also **widened `chunk_size` to `int | None`** — upstream `chunkerify` defaults to `None` (auto-derive from the tokeniser's `model_max_length`), and Womblex previously forced an `int`; the `None` path now passes through. Default stays `480` (Kanon-2 window), so the divergence is now a *default value* choice, not a missing capability. Verified `model_validate`: `None` accepted, `0` still rejected (ge=1 holds on the int branch), default 480. Full semchunk public surface re-checked (`Chunker`/`chunkerify`/`chunk`/`semchunk` — the low-level module `chunk()` is wrapped by the high-level path Womblex already uses; nothing missed). Adapter boundary now documented explicitly in the `chunker.py` module docstring, the `ChunkingConfig` docstring, and a new `docs/extraction.md` "Chunking adapter boundary" subsection. The 3 default divergences from upstream (`tokenizer="isaacus/kanon-2-tokenizer"`, `chunk_size=480`, `processes=1`) are each annotated with their corpus reason and retained.
  - Process: **no corpus re-run performed** — the only behavioural surfaces touched were a dead config field (read by no code path) and an opt-in `None` chunk_size (default unchanged), so chunk output is byte-identical to I2 by construction. The spec's "re-run chunk stage to confirm byte-identical" was **accepted-on-risk as noted, not measured** (per user, 2026-05-30); the empirical chunk-diff over the i1c shard dir remains available if confirmation is ever wanted. 98 chunker/config/pipeline/output tests pass; 79 integration tests pass (30 min).
  - Assess: wrapper is honestly thin; pass-through 100% (incl. `chunk_size=None` auto-derive); sole real divergence (dead `batch` flag) removed; boundary documented in code + docs.

- **I6 ✓ spike run 2026-05-30 — Entity-linking Kanon-2 spike (P4 part 1). Decision: (b) Kanon-2 + register-matching wrapper.**
  - Build: `isaacus` extra installed into `.venv` (SDK 0.20.0); live auth confirmed against the `.env` key via a minimal `kanon-2-enricher` call. Spike script `/tmp/i6_artemis_spike.py` maps the **Artemis document set** (17 PDFs — 14 R-prefix + 3 numeric-prefix, all one provider, doc types: direct complaint / notification / decision letter / show-cause) to their i1c-shard narrative via `_load_elements`+`reassemble_narrative`, then runs `analyse.enrich.enrich_documents` per doc. Provider/service entities surface as `persons` with `type='corporate'` (authorities come back as `type='politic'`); not as `locations`.
  - Process: 17-doc sample only (no corpus run), ~96k chars total.
  - Assess — **Kanon-2 is a strong candidate generator but does not canonicalise:**
    - **Recall 16/17** docs yielded ≥1 Artemis-family corporate person (only `R-03247` returned none); 3 docs independently surfaced the service street address as a `location[address]`.
    - **~13 distinct surface forms for one entity** — legal-name forms (`ARTEMIS EDUCATION PTY LTD`, `ARTEMIS EDUCATION PTYLTD`), trading/service forms (`Artemis Early Learning`, `…Centre`, `…Fyshwick`, `…– Fyshwick`, newline-split, lowercase), OCR-corrupted forms (`Artemis Earty Learning`, `Artemis Early Leaming`), and a **different legal entity** (`Canberra Childcare Pty Ltd ATF The Fyshwick Child Care Trust`, a prior trustee, not in the current register). Raw Kanon-2 output therefore can't be the entity graph on its own — that would break the P3 PII-consistency goal (P4 is upstream of P3 precisely to unify variants).
    - **The ACT register closes the gap cleanly.** "Artemis" is a real register entry, not a pseudonym: Approved-providers → `PR-40030037` (Legal "ARTEMIS EDUCATION PTY LTD" / Trading "Artemis Early Learning"); Education-services → `SE-40002132` at `11 Cessnock St, Fyshwick 2609`. Every Kanon-2 variant fuzzy-matches to these via Legal Name, Trading Name, or — most robustly, immune to OCR noise — the **structured street address** Kanon-2 also extracts. (a) is insufficient (no canonical ID, no variant unification); (c) discards Kanon-2's address/candidate signal and rebuilds NER from scratch.
  - **P4 v1 shape (→ I7): `womblex link --shards <dir>` =** Kanon-2 enrich → collect `corporate` persons + `address` locations as candidates → resolve against the ACT registers (name fuzzy-match + address exact-match) → emit `*.entity_links.parquet` `(source_hash, provider_id, service_id, confidence, evidence)`. Address is the primary key, normalised name the fallback. **Hand-back gate — awaiting user go-ahead on I7 scope before any full-corpus link run.**

- **I7 ✓ built + tested + smoked 2026-05-31; full-corpus run BLOCKED on an Isaacus network outage — Entity-link sidecar (P4 part 2).**
  - Build (landed): two per-stage stages mirroring `womblex chunk --shards`. **`womblex enrich --shards`** (`analyse/enrich_stage.py`) reassembles narrative via `reassemble_narrative`, calls Kanon-2 per doc (per-doc failure isolation; transient/connection failures are NOT checkpointed so resume retries them), writes `*.enrichment_entities.parquet` + `*.enrichment_meta.parquet` (reuses `store/enrichment_output.py` schemas, keyed on `source_hash`). **`womblex link --shards --config`** (`link/` package: `normalise`/`reference`/`matcher`/`stage`) resolves corporate-person + address candidates to a reference register, writes `*.entity_links.parquet` at mention grain (doc grain = derived read view). **Library-first/generic split honoured:** generic `entity_type` schema (no provider/service columns), generic record-linkage matcher, bundle-aware reference consumption (CSV built; geospatial/multi-file seam reserved), corpus declares register column-roles via new `linking`/`reference` config. `isaacus` optional extra. Tests: `test_link.py` + `test_enrich_stage.py` (26).
  - **Matcher landed in two steps (both via smoke):** (1) token-set ratio fixed suburb-suffix recall + cross-brand precision (7→15/17); (2) **OCR-tolerant per-token char similarity** (`_TOKEN_SIM_FLOOR=0.72`) folds OCR typos like "Earty"→"Early" into the match while still rejecting a different brand (15→**16/17**). stdlib `difflib` only, no new dependency.
  - Process — **smoke validated, full corpus BLOCKED.** Live 17-doc Artemis smoke (real `enrich_shards` → `link_shards`, real ACT register): **16/17 → correct `SE-40002132`** (parent `PR-40030037`); only `R-03247` unmatched (Kanon-2 extracted no org — the embed-stage doc→entity backstop's job). The **full 2,626-doc enrich could not complete: the Isaacus endpoint became unreachable** (~14:00 onwards 2026-05-31). Confirmed not code/key (a 2-hr run + a fresh single-doc foreground probe both fail with httpx "Connection error", a connection failure not a 401). **Also learnt: background / long-auto-backgrounded Bash has no network in this dev env — only short genuinely-foreground calls reach the API**, so even when the endpoint returns, a 2-hr networked job can't run as one background command here (must chunk in foreground windows, or run from a normal shell). Corpus shards are clean (enrich artefacts purged; elements/chunks/manifests = 106 each intact); the run is staged + resumable for when connectivity returns.
  - Open QA items for the corpus-phase run (when network returns): (a) accept 16/17 v1 recall or add the embed backstop for R-03247; (b) corpus alias table for the prior-trustee mention; (c) corpus-side threshold tuning. All corpus-side.
  - Resume-integrity reconcile **generalised + wired** (2026-05-31): `reconcile_stage_checkpoint_with_shards(mgr, dir, *, suffix)` in `store/shard_audit.py` is the shared self-heal engine; `reconcile_chunk_checkpoint_with_shards` now delegates to it, and enrich/embed/link CLIs all wire it + `--no-verify-resume`. So every `CheckpointManager`-backed stage (chunk/enrich/embed/link) self-heals corrupted sidecars identically on resume. (`redact` stays on its own JSON checkpoint — I3 divergence.)
  - Deferred follow-ups: E2E `womblex run` composition of enrich/link; graph-edges sibling (only entities+meta in v1).

- **Embed stage ✓ built + tested 2026-05-31 (new Womblex capability, per user) — `womblex embed --shards`.**
  - `analyse/embed.py` (thin Kanon-2 `embeddings.create` wrapper — 128-text batching, 429 retry, order-preserving, task-aware) + `analyse/embed_stage.py` (`embed_shards` over `*.chunks.parquet` → `*.embeddings.parquet`, vector per chunk, per-stage `CheckpointManager`, batch-level failure isolation) + `cli/embed.py` + `EmbeddingConfig` + `EMBEDDINGS_SCHEMA`/IO. `test_embed_stage.py` (mocked embedder). Rationale (from the embeddings discussion): NOT for register name-matching (embeddings blur the distinctive brand token precision relies on), but YES for (i) a doc→entity semantic-attribution backstop that catches no-extraction docs like R-03247, and (ii) the downstream semantic-analysis substrate (search/cluster/classify). Live corpus embed not yet run (same network outage).

- **I8 — PII plumbing + GT expansion (P3, measurement gate).**
  - Build: `pii.entity_types: ["PERSON"]` config; `pii_spans` sidecar writer; per-stage `womblex pii --shards <dir>` CLI; author PERSON GT against the 18-page labels packet. PII detection consumes the entity graph produced by I7 (`pii/cleaner.py` merges enrichment-derived spans) to mask name variants ("Janine Fairburn" / "J. Fairburn" / "Asst. Director") as one entity.
  - Process: run **PII stage only** on the labels packet (not the corpus yet).
  - Assess: PERSON precision/recall vs new GT. Decision: (a) sidecar-only on corpus, (b) tune + re-measure, (c) apply rewrites to clean_text.
  - Proceed when PII quality is *characterised*, not necessarily fixed. Measurement gate, not quality gate.

- **I9 — PII corpus run (sidecar-only by default).**
  - Build: whatever tuning came out of I8.
  - Process: full-corpus **PII stage only**. Write rewrites to `clean_text.parquet` only if I8 said acceptable.
  - Assess: sidecar size; spot-check 20 false-positive candidates.
  - Proceed when the PII layer is what you want to publish.

- **I10 — Redaction marker decision + implementation (P6).**
  - Build: make the (a)/(b)/(c)/(d) decision based on industry-standards review. Implement chosen strategy as a `clean_text` op.
  - Process: re-run **clean_text stage only** with markers.
  - Assess: human review of marker placement on 10 docs across strategies.
  - Proceed when markers are correctly placed (or strategy (a) "no markers" is chosen).

- **I11+ — Quality fixes (P7).**
  K7(b) (DocLayNet YOLO swap) and K2′ (OCR-form bboxes) landed 2026-05-25 ahead of the I-sequence as a focused extraction-quality iteration. **Remaining**: `clean_text` op for #B/#D (footer whitespace artefacts, redaction-induced paragraph breaks) as its own iteration. Re-run **extraction stage only** → measure deltas.

**Properties of this rhythm.**
- Stop anywhere — every iteration ends in a coherent corpus state.
- Diffability — retention default keeps the previous run; sidecar-vs-sidecar diff across any two iterations is easy.
- Cost-asymmetric — I1-I5 are plumbing (fast iterations); I6-I9 hit Kanon-2 / API surface (I6/I7 entity-linking, I8/I9 PII) and each take a full corpus pass after the spike or GT-author step.
- Three explicit hand-back-to-user gates: I6 (Kanon-2 entity spike), I8 (PII characterisation), I10 (markers).

#### P1 — Sidecar persistence + stage-aware checkpointing

**Problem.** `run` today computes redact-blackout, PII rewrites, and chunks but the parquet writer only persists `elements.parquet` (extraction-verbatim by policy). All downstream stage outputs are silently dropped — even with `pii.enabled: true` / `chunking.enabled: true` / `redaction.mode: blackout`, nothing reaches disk except the elements shard and (via the separate `annotate-redactions` CLI) a sparse `*.redactions.parquet`. PII / chunking / blackout markers are effectively unreachable on disk under the current writer.

**Scope.**

- New sidecars per stage, all joinable on `(source_hash, elem_order)`:
  - `*.clean_text.parquet` — post-rewrite element text (output of `clean_text` op + redact-blackout + PII applied, when those stages run)
  - `*.pii_spans.parquet` — `(elem_order, start, end, entity_type, replacement)` so the rewrite is auditable and reversible
  - `*.redactions.parquet` — already lands via `annotate-redactions`; fold into `run` so a single pass produces the full set
  - `*.chunks.parquet` — schema already drafted in `dataflow.md` "chunks.parquet (planned)"; complete the writer
- Per-stage `CheckpointManager` — one checkpoint file per stage. Rollback = delete the stage's sidecar(s) and re-run the stage; downstream stages re-run from their own checkpoint.
- Complete any half-implemented stage functions before next E2E run (chunk persistence writer, PII writer integration, redact-blackout-to-sidecar wiring).
- CLI: each stage gains an independent entry point that consumes the prior sidecar (e.g. `womblex pii --shards <dir>` reads `*.clean_text.parquet` if present else `*.elements.parquet`; writes `*.pii_spans.parquet` + an updated `*.clean_text.parquet`). Stage independence is the explicit design goal — each module becomes less dependent on prior modules over time.

**Config additions.**

- ✓ **`dataset.run_id: str | None`** — landed 2026-05-23 ([config.py](src/womblex/config.py)). Resolved at run time: `--run-id` CLI > `dataset.run_id` in config > auto-generated `run-YYYYMMDDTHHMMSSZ`. Outputs land at `<output_root>/<run_id>/documents/`; checkpoints at `<checkpoint_dir>/<run_id>/`. `--resume` without an explicit id picks the most-recent existing run dir.
- ✓ **`processing.retention.policy: "rolling" | "keep_all" = "rolling"`** — landed 2026-05-23 ([store/retention.py](src/womblex/store/retention.py)). `rolling` purges runs beyond the keep window on each fresh run (not on `--resume`); `keep_all` is no-op. Current run always preserved. **Only subdirectories whose name starts with `run-` are candidates for retention** — legacy / hand-named dirs (e.g. `output/documents/` from a pre-run_id layout, or `output/baseline-snapshot/`) are preserved unconditionally. To bring a hand-named run under the policy, name it with a `run-` prefix.
- ✓ **`processing.retention.keep: int = 2`** — landed 2026-05-23.
- `pii.entity_types: list[str] = ["PERSON"]` — see P3.

**I1 slice (landed 2026-05-23).** Just the plumbing — run_id resolution, output/checkpoint layout under `<run_id>/`, retention policy. No new stage output content; extraction shards are byte-identical to pre-I1 layout (just nested one level deeper). New module `womblex.store.retention`; 18 tests in `tests/test_retention.py`; 8 tests added to `tests/test_config.py`. CHANGELOG entry under [Unreleased] / Added.

**Remaining in P1** (subsequent slices I2-I3):

- `*.chunks.parquet` writer — schema in `dataflow.md`, not yet written
- `*.pii_spans.parquet` writer — PII results currently mutate `pages[i].text` in memory, never persist
- `*.clean_text.parquet` writer — surface post-stage element text rewrites
- Fold `annotate-redactions` behaviour into the per-stage `womblex redact` command (it currently writes `*.redactions.parquet` via a separate CLI; the goal is one stage = one CLI verb)
- Per-stage `CheckpointManager` instances — independent rollback per stage
- **Per-stage CLI entry points** for every operation (`womblex chunk --shards`, `womblex pii --shards`, etc.) consuming the prior stage's sidecar. **This is the primary invocation path for this corpus** (see Execution model above); the existing `womblex run` E2E command remains supported for future users with simpler corpora.

**Why this is P1.** Per-stage execution is the workflow this corpus uses; sidecar persistence is what makes a per-stage invocation possible at all. Today no stage past extraction writes its output to disk, so a per-stage chunking / PII / redact-blackout invocation has nothing to consume and nothing to produce. P3-P6 each ship per-stage CLI entry points, all of which depend on P1's sidecar plumbing. P7 quality fixes are mechanically independent but consumers can only see their value through the sidecars P1 enables.

#### P2 — SemChunk wrapper audit (`isaacus-dev/semchunk`)

**Scope.** semchunk is mature and widely used; Womblex's role around it is a **thin adapter** that handles only the integration concerns the library can't know about — parquet I/O, element-stream → `ChunkInput` projection, source-hash plumbing, `<REDACTED>` cross-boundary repair. Everything else is semchunk's. Audit `process/chunker.py` for any gating, shadowing, or reimplementation of semchunk-native surface; verify `ChunkingConfig` pass-through coverage matches the current upstream; track upstream defaults except where the corpus has a measured reason to diverge; document the adapter boundary explicitly so future contributors don't drift back into wrapper bloat.

**Constraints.**

- **No-API-key pathways must remain the default and supported path** — any new paid-API tokeniser stays opt-in; HF tokeniser identifier and callable token counter remain primary.
- **Pass-through, not gatekeeping.** semchunk's own parameters *are* the feature surface; do not re-expose them as separate Womblex toggles. A future corpus that doesn't need feature X simply doesn't set the corresponding `chunking` config field — there is no Womblex-side flag to flip.
- **Track upstream defaults.** Where a Womblex default diverges from semchunk's, the divergence must have a measured corpus-specific reason (`tokenizer="isaacus/kanon-2-tokenizer"` matches the analysis side; `processes=1` is Chromebook-portability; etc.). Audit each one; revert undocumented divergences.
- **Anti-shadow rule.** When semchunk absorbs a concern Womblex previously handled, delete the Womblex code rather than running a parallel implementation. Precedent: I2 deleted `chunk_text` / `chunk_texts_batch` / `chunk_document` once semchunk batched a list of texts natively.
- **Non-blocking for P1.** Wrapper audit can land after P1 if needed; P1's `chunks.parquet` writer treats SemChunk as the chunker without depending on a specific feature set.

**Why independent of P1.** Chunker output shape is unchanged — defaults-off chunks remain byte-identical to I2; the audit's success criterion is *the wrapper is thinner*, not *the output is different*.

#### P3 — Configurable PII + GT expansion

**Scope.**

- `pii.entity_types: list[str]` config — which entity classes to detect + mask. For the ACT_EarlyChildhoodIncidents corpus the only required class is `PERSON`; other corpora may want `EMAIL` / `PHONE` / `ADDRESS` etc.
- PII detection precision is unmeasured on this corpus surface. The only existing benchmark is Throsby (PERSON precision 16.7%). Author additional PII GT against the existing 18-page labels packet (or a fresh subset) before running PII on the corpus.
- PII output remains sidecar-only in v1 (`*.pii_spans.parquet`) — no auto-rewrite of `clean_text` until precision is measured and acceptable. Consumers can apply spans themselves.

**Depends on P1** for `pii_spans.parquet` sidecar plumbing and **on P4** for the entity graph. `pii/cleaner.py` merges enrichment-derived spans before emitting `<ENTITY_TYPE>` tags, so canonical-entity coverage from P4 gates PERSON masking consistency across name variants ("Janine Fairburn" / "J. Fairburn" / "Asst. Director, OECEY" → one entity). Without P4, PII sees variants as independent strings and masks them inconsistently.

#### P4 — Entity-link sidecar (Kanon-2 first)

**Approach.** Before building register-matching heuristics, try the existing Kanon-2 enrichment path (`analyse/enrich.py`) for entity extraction + canonicalisation. If Kanon-2's `entities` output reliably attributes documents to providers/services, the entity-link sidecar is a thin wrapper that resolves enrichment entities against the ACT provider/service registers (`Approved-providers-act-export.csv`, `Education-services-act-export.csv`).

**Sidecar.** `*.entity_links.parquet` with `(source_hash, provider_id, service_id, confidence, evidence)`.

**Inputs already on disk.**
- 1,667 of 2,615 PDFs have site-name embedded in filename (R-prefix files)
- FOI master manifest (31,668 cells, `kind='table'`) carries authoritative provider name + doc-type for the 948 numeric-prefix outbound letters
- ACT registers (PR-* / SE-* canonical identifiers)

**Pragmatic ordering.** Spike Kanon-2 on a 20-doc sample first; decide between Kanon-2-driven vs register-matching-driven attribution based on observed quality. Don't pre-build heuristics.

**Depends on P1** for sidecar plumbing; **upstream of P3** (PII detection consumes the entity graph this sidecar produces). Independent of P6 otherwise.

#### ~~P5 — Document-metadata sidecar~~ (dropped from library scope 2026-05-28)

**Decision.** Filename-component parsing is not a library-general capability. The ACT FOI release uses two schemas (`R-<doc_id>-<foi_ref>-<service>-CAS-<n>-<direction>-<doc_type>.pdf` for inbound/notification documents; `<doc_id>-<foi_ref>-<release_date>-<doc_type>.pdf` for outbound letters) plus a long tail of variants (`<doc_id>A`, `<doc_id>_`, no-date forms). A second corpus would carry a different scheme entirely. Per CLAUDE.md's corpus/library boundary — "the codebase doesn't know about specific datasets" — this work belongs in `stories/<corpus>/`, not in Womblex.

**What the library already provides for this work.**

- The raw `filename` is preserved verbatim on `_manifest.parquet`'s `filename` column.
- `source_hash` is the cross-stage join key; any corpus-side `*.doc_metadata.parquet` written next to the shard files will compose with downstream Womblex stages without library changes.
- The FOI-master manifest tables (3 of them, 34,261 lookup rows total) extract cleanly under the I1+I2 layout as `kind='table'` elements with per-cell rows in `*.table_cells.parquet` — corpus scripts can read them with `read_table_cells()`.

**Promotion criteria.** Once a second corpus needs a similar derivation, identify the *general* shape (e.g. "given a header-keyed embedded lookup table, join its rows to other docs by ID") and promote that helper to the library — the corpus-specific parser stays corpus-side. Precedent: `score.py` and `redact/batch.py` were promoted from corpus scripts after their generality became evident.

**Date-of-event** (the `date` column in the Artemis timeline CSV) requires text-level extraction. Same boundary applies — corpus concern, not library.

#### P6 — Redaction marker strategy (open question)

**Decision pending** industry-standards review. Options under consideration:

- **(a) No markers in clean_text.** Consumers join `*.redactions.parquet` to locate redactions; clean_text reads as natural prose. Avoids confusing encoders.
- **(b) Page-prefix marker** (current bracket-only design). Easy to implement; preserves redaction signal at cost of slight encoder distraction.
- **(c) Inline-per-span marker** (e.g. `<REDACTED>` at the precise span). Highest fidelity but depends on K2′ (raster path) and a new text-to-bbox character mapping (native path).
- **(d) Two corpus variants** — publish both a marker-free `clean_text` and a marker-tagged `clean_text_with_markers` sidecar. Doubles storage of the rewritten-text shard but lets downstream consumers pick.

**Action.** Defer the decision; user wants more information on industry standards (FOI release publication conventions, regulatory dataset conventions, encoder-distraction studies) before choosing. Track here; revisit before P1 ships.

#### P7 — Quality fixes (already tracked individually)

These three fixes already live in this Open-follow-ups section as separate items. Reaffirmed as in-scope for the publishable run:

- ~~**K7(b)**~~ — Landed 2026-05-25 (DocLayNet YOLO11 nano swap).
- ~~**K2′**~~ — Landed 2026-05-25 (per-region OCR form-pair bboxes).
- **#B** — Native-text footer whitespace artefacts (`3|P age`) → expand into a general `clean_text` downstream op (rewrites `pages[i].text`, writes `*.clean_text.parquet`). Naturally houses the redaction-induced para break fix (#D) and letterhead typo fixes (`Govemment`/`AcT`) under the same op.

**Honest sequencing note.** P7 doesn't gate P1-P5. If P1 sidecar plumbing is on the critical path, P7 can land in parallel or after the first publishable run as iterative quality improvements.

#### Out of scope for the publishable run

- Corpus README publication — defer until P1-P5 are in place; working notes live in `stories/STATUS.md` as we proceed.
- Enrichment / Isaacus integration beyond P4's Kanon-2 entity-extraction spike — analyst-output, not corpus-input.
- PDF bundling (the Artemis chronological-bundle artefact) — analyst-output; trivial library work if it later becomes useful.
- Date-of-event extraction from document body text — defer to follow-up.

### Element-kind audit fix cluster (K1-K9 landed; K10 deferred-cosmetic; K11 non-issue)

Surfaced by the corpus-wide audit completed 2026-05-22 — see `stories/STATUS.md` "Non-`table` element kind audit" for the full data set. **K1, K3, K4, K5, K7(a), K8 landed 2026-05-22 as a single change set; K2′, K6, K7(b) landed 2026-05-25 as the second change set** (DocLayNet swap + OCR-form bboxes). All audited fixes resolved; awaiting full-corpus re-extraction to refresh the per-kind counts in the "Corpus-scale measured" table below.

**K9-fig landed 2026-05-30** once I2 (chunking) confirmed it filters the narrative by `kind` (TEXT_KINDS) — the 16 total-loss complaint docs were proof the win was material content recovery, not cosmetic. **K10-list and K11-cap remain low-priority cosmetic**: both involve kinds that *are* in `TEXT_KINDS` (`list_item`, `caption`), so their text already reaches chunks — the mis-kinding is a wrong label, not lost content. See the K9-K11 rows below.

| ID | Status | Fix | Code site |
|---|---|---|---|
| **K1** ✓ | landed | `_SIGNATURE_RE` removed from `_classify_native_block`. Closing phrases ("Yours sincerely") no longer emit `kind='signature'` — they fall to `paragraph` until a proper signatory-block detector lands. | [extract.py:325-346](src/womblex/ingest/extract.py#L325-L346) |
| ~~K2~~ | retracted | **Originally "every native-text element bbox is zero" — that was a probe-script formatter artefact (`:.0f` rounding normalised 0-1 floats to "0"). Corpus measurement confirms bbox population at 100% for paragraph / heading / footer / signature / figure / image / table / native_with_structured forms.** The narrower real issue is K2′ below. | (verified 2026-05-23) |
| **K2′** ✓ | landed 2026-05-25 | New `_extract_form_pairs_from_regions` walks per-region OCR detections (PaddleOCR / RapidOCR line-level bboxes) instead of assembled text. Each `FormField` carries the source region's normalised bbox; `_apply_ocr_page` prefers this path. The legacy line-only `_extract_form_pairs_from_lines` is retained for LLM-OCR engines that resolve reading order natively (no per-region bboxes). `_ocr_page` now returns regions + pix dimensions alongside text so the orchestrator can wire them through. Unblocks P6 inline-per-span on raster pages. | [forms.py:148](src/womblex/ingest/forms.py#L148); [strategies_scanned.py:208](src/womblex/ingest/strategies_scanned.py#L208) |
| **K3** ✓ | landed | Label denylist added to `_looks_like_form_label`: `Penalty`, `OFFICIAL`, `Note`, `Caution` — captures the regulation-citation / document-banner patterns that drove ~250-500 spurious forms in hybrid + structured. | [forms.py:38-46](src/womblex/ingest/forms.py#L38-L46) |
| **K4** ✓ | landed | `header` added to `ElementKind` literal, `TEXT_KINDS` frozenset, and `_BLOCK_TYPE_TO_KIND` mapping. `_classify_native_block` was already returning `"header"`; now it round-trips into `kind='header'` instead of silently demoting to `paragraph`. | [elements.py:26](src/womblex/ingest/elements.py#L26); [orchestrator.py:212](src/womblex/ingest/orchestrator.py#L212) |
| **K5** ✓ | landed | `_LIST_ITEM_RE` added to `_classify_native_block`: matches `(a)` / `(b)` / `(i)` / `(1)` / bullets `•·-*`. Bare `1. `-prefix excluded (ambiguous with numbered paragraphs in this corpus). | [extract.py:320](src/womblex/ingest/extract.py#L320) |
| K6 ✓ | landed 2026-05-25 via K7(b) | DocLayNet's `Caption` class produces `kind='caption'` directly. Closes the deferred caption-via-adjacency track. | [paddle_ocr.py:149](src/womblex/ingest/paddle_ocr.py#L149) |
| **K7(a)** ✓ | landed | `_YOLO_COCO_LABEL_MAP` default changed from `figure` to `paragraph`. Unknown COCO classes (the dominant case on scanned pages, since COCO doesn't have document classes) now bucket to text. Explicit screen/keyboard/etc. mappings preserved. | [paddle_ocr.py:232](src/womblex/ingest/paddle_ocr.py#L232) |
| **K7(b)** ✓ | landed 2026-05-25 (imgsz tuned 2026-05-26) | Primary layout backend swapped from COCO `yolov8n.pt` (80 classes, 0 doc semantics) to DocLayNet `yolo11n_doc_layout.pt` (11 doc classes, [Armaggheddon/yolo11-document-layout](https://huggingface.co/Armaggheddon/yolo11-document-layout), MIT, 5.37 MB, SHA-256 `3629fc7a…81f`). New `_YOLO_DOCLAYNET_LABEL_MAP`; `YOLOLayoutAnalyzer` detects taxonomy from loaded class names and selects map + per-taxonomy imgsz (DocLayNet 832, COCO 640). New `footnote` ElementKind. `redact/stage._LAYOUT_EXCLUSION_BLOCK_TYPES` now taxonomy-agnostic (matches on block_type `figure`/`table`, works for both DocLayNet `Picture`/`Table` and COCO `tv`/`laptop`/etc). COCO weights retained as fallback only. | [paddle_ocr.py:149](src/womblex/ingest/paddle_ocr.py#L149); [redact/stage.py:66](src/womblex/redact/stage.py#L66) |
| **K8** ✓ | landed | Orchestrator emits `kind='page_break'` between consecutive pages in `extract_with_plan`. N-1 breaks for N pages. | [orchestrator.py:401-406](src/womblex/ingest/orchestrator.py#L401-L406) |
| **K9-fig** ✓ | landed 2026-05-30 | 1,056 of 1,200 i1b `kind='figure'` elements are `extractor=ocr_paddle`, `bbox=(0,0,1,1)`, paragraph-text content. **Root cause was NOT `_ocr_image_regions` (the original E4 diagnosis named the wrong site).** It is the dominant-region fallback in `_layout_blocks_and_tables`: YOLO layout regions carry no segmented text, so the fallback always fires and collapses the whole page's OCR onto one block using the *largest* region's kind. When that is DocLayNet `Picture`→`figure`, a full-page scan becomes one `figure` excluded from chunking. Fix: shared helper `_ocr_region_block_type(text, layout_kind)` promotes a non-text fallback kind to `paragraph` when OCR yields ≥5 words; sparse output (page numbers, bare logos) keeps `figure`. Threshold calibrated on the 1,056 affected elements. Applied to the existing i1c corpus by **in-place parquet relabel** (pure `kind` transform — no re-extraction; relabel verified ≡ live re-extraction on the affected docs). **Deltas i1b→i1c: figure 1,200→154, paragraph 35,931→36,977, docs-with-chunks 2,610→2,626 (all 16 total-loss docs recovered), chunks 11,797→12,506.** | [strategies_scanned.py:440](src/womblex/ingest/strategies_scanned.py#L440); helper [strategies_scanned.py:288](src/womblex/ingest/strategies_scanned.py#L288) |
| **K10-list** | deferred (cosmetic) | 1,134 of 5,403 i1b `kind='list_item'` elements (21%) lack any leading list marker — samples are letterhead / address blocks DocLayNet over-fires its `List-item` class on for OCR'd pages. The 79% with `(a)` / `(1)` / `(i)` / `•` / `1.` markers look correct. Proposed fix: post-filter DocLayNet list_item predictions to require a leading marker, or demote no-marker list_item to paragraph at the orchestrator boundary. **No content loss — `list_item` ∈ `TEXT_KINDS`, so these reach chunks; only the kind label is wrong. Cosmetic, low priority.** | [paddle_ocr.py](src/womblex/ingest/paddle_ocr.py) (DocLayNet label map consumer) |
| **K11-cap** | resolved — not content loss | All 10 i1b `kind='caption'` elements are `ocr_paddle` full-page `(0,0,1,1)` with paragraph-text content. **But `caption` ∈ `TEXT_KINDS`, so these always reached chunking** — the mis-kinding (caption vs paragraph) is cosmetic, never lost content. The K9-fig fix deliberately passes text kinds (incl. caption) through unchanged. No action taken. | (n/a) |

**Validation.** Six landed fixes verified via:
- Unit tests: `tests/test_extract.py` adds `TestClassifyNativeBlock` (10 cases for K1/K4/K5), `TestFormLabelDenylist` (4 cases for K3), `TestYoloLabelMapDefault` (2 cases for K7(a)), `TestPageBreakEmission` (2 cases for K8). All 37 test_extract pass; broader 484-test fast-suite pass with no regressions.
- Sample re-extraction on 12 docs (3 per stratum) shows expected per-kind deltas vs the pre-K-cluster production parquet:
  | kind | sample delta | direction |
  |---|---:|---|
  | `page_break` | **+25** | K8 emitting per page transition |
  | `list_item` | **+20** | K5 picking up regulation sub-paragraph markers |
  | `paragraph` | **−17** | net reduction from list_item reclassification + signature drops |
  | `signature` | **−4** | K1 — all 4 sampled signatures were "Yours sincerely" closings, now dropped |
  | `form` | **−3** | K3 denylist filtering Penalty/OFFICIAL on sampled docs |
  | `header` | **+1** | K4 round-tripping top-of-page short text into the canonical kind |

**Corpus-scale measured** (post-K-cluster re-extraction 2026-05-23, 2h 35m, 2,626 succeeded, 0 failed):

| kind | pre (2026-05-17) | post (2026-05-23) | Δ | note |
|---|---:|---:|---:|---|
| paragraph | 40,980 | 37,107 | −3,873 | net of list_item / header reclassification, partial offset from K1 |
| sheet_cell | 36,780 | 36,780 | 0 | unchanged |
| form | 5,183 | 4,391 | **−792** | K3 denylist removed more than projected (~250-500 estimated) |
| page_break | 0 | **6,561** | +6,561 | K8 — averaging ~2.5/doc (corpus skews to 1-3 page notification forms) |
| footer | 2,695 | 2,695 | 0 | unchanged |
| heading | 2,656 | 2,634 | −22 | small reclassification to list_item / header |
| list_item | 0 | **4,015** | +4,015 | K5 |
| figure | 1,600 | 1,587 | **−13** | K7(a) impact was small — figure count is driven by explicit COCO screen-class mappings (`tv` / `laptop` / `monitor`), not the unknown-class default. **K7(b) remains the right lever for closing this.** |
| image | 331 | 331 | 0 | unchanged |
| signature | **442** | **0** | **−442** | K1 — every closing-phrase mis-classification gone |
| table | 172 | 172 | 0 | §1 limitations preserved (3 residual conf=0.60 fabrications still present, as documented) |
| header | 0 | **335** | +335 | K4 |
| sheet_meta | 4 | 4 | 0 | unchanged |
| **total** | **90,843** | **96,612** | **+5,769** | net of new kinds + reclassifications |

**Strategy distribution identical** (detection logic unchanged): 1,777 scanned_machinewritten · 628 hybrid · 165 structured · 40 native_with_structured · 11 scanned_mixed · 4 spreadsheet · 1 scanned_handwritten.

**Text fidelity preserved exactly** — per-strategy mean CER on the 18-page labels packet matches 2026-05-21 numbers to 3 decimal places. K-cluster reclassified element kinds without touching text content.

**Spot-check on known limitations**: §1 residual fabrications (01093, 01094, 01349) still carry their 1 junk table each — behaviour preserved, no regressions. §11 02737 unchanged except +1 page_break. FOI master manifest (`Schedule-of-documents-Part-2b`) intact.

**K7(a) impact was smaller than originally projected** (−13 vs expected near-zero on scanned_machinewritten figure count). Root cause: COCO YOLO's explicit screen-class mappings (`tv` / `laptop` / `monitor` / `keyboard` / `mouse` / `scissors` / `clock` → `figure`) catch most scanned-page hits, not the unknown-class default. Closing the remaining 1,587 figure mis-classifications requires K7(b) (document-layout YOLO swap) — confirms the dependency framing.

Previous on-disk parquet preserved at `stories/.../womblex-extract/output-pre-kcluster-2026-05-17/` for historical comparison.

---

**K2′ + K7(b) validation (2026-05-25, pre-corpus-rerun).** Numbers below are from local validation, not the full-corpus re-extraction (which has not yet been scheduled — see STATUS-side todo).

*DocLayNet fixture F1* (5 labelled pages from `fixtures/fixtures/doclaynet/`). All per-class F1 jumped from 0% (COCO had no document semantics) to real numbers:

| class | precision | recall | F1 |
|---|---:|---:|---:|
| header | 75.0% | 100.0% | **85.7%** |
| table | 100.0% | 50.0% | 66.7% |
| paragraph | 23.3% | 58.3% | 33.3% |
| heading | 25.0% | 25.0% | 25.0% |
| list_item | 9.1% | 50.0% | 15.4% |
| footer / caption / figure / footnote | 0% | 0% | 0% (FP-only — DocLayNet predicts more regions than GT merges) |

Generated by `tests/test_fixture_accuracy.py::TestDocLayNet`; full results in [docs/accuracy/EXTRACTION.md](docs/accuracy/EXTRACTION.md).

*10-doc corpus sample* (well-distributed: 3 `scanned_machinewritten`, 3 `hybrid`, 2 `native_with_structured`, 2 `scanned_mixed`, picked from the labels packet — covers the 02737-class `scanned_mixed` cohort that motivated K7(b)).

| measure | result |
|---|---|
| total elements | 478 (legacy COCO) → 477 (new DocLayNet) — essentially identical |
| `kind='list_item'` | **+28 across 7 docs** (DocLayNet now detects on raster pages; COCO/K5 only fired on native text) |
| `kind='paragraph'` | **−41** (correct demotion to list_item / footer / heading on raster pages) |
| `kind='figure'` on 02737 | **0** (was a known false-positive source; DocLayNet produces no spurious figures on this doc) |
| K2′: form bbox quality | **13 of 13 previously-zero form bboxes are now populated** with real positions; remaining form-count drift (-3 forms across 3 docs) is region-based extraction being more conservative than line-based |
| extraction failures | 0 of 10 |
| OCR engine behaviour | unchanged — text fidelity preserved by construction (layout swap touches classification only, not the OCR pass) |

Validation script archived as `/tmp/validate_10docs.py` (not committed — point-in-time probe).

*Labels-packet CER (2026-05-25)*. Full re-extraction of all 21 unique labels-packet PDFs, then scored via `womblex score`. Per-strategy mean CER **identical to the 2026-05-23 baseline** to 3 decimal places: hybrid 0.520, native_with_structured 0.044, scanned_machinewritten 0.051, scanned_mixed 0.217. Confirms K2′/K7(b) introduced zero text-fidelity drift.

*Redaction-precision cohort (2026-05-25)*. Re-ran [validate_redaction_detection.py](../../stories/ACT_EarlyChildhoodIncidents/womblex-extract/validate_redaction_detection.py) against the labels packet under DocLayNet vs the May 19 COCO baseline:

| cohort | COCO regions | DocLayNet regions | direction |
|---|---:|---:|---|
| 02737 (canonical scanned_mixed FP case) | 10 | 4 | precision ↑↑ (−60%) |
| scanned_mixed total (3 docs) | 29 | 10 | precision ↑↑ (−65%) |
| 01093 p2 (GT: 18 bars) | 6 detected | 14 detected | **recall 33% → 78%** |
| 01094 p2 (GT: 17 bars) | 7 detected | 13 detected | **recall 41% → 76%** |
| 01349 p3 (GT: 33+ bars) | 3 detected | 68 detected | **recall ~9% → ~100%** (stacked PDF rects per visible bar) |

Both precision (on false-positive-prone scanned_mixed) and recall (on heavily-redacted hybrid/structured) improved on the docs with per-region GT. K7(b)'s redaction-precision claim from STATUS.md:251 is validated.

**Full-corpus re-extraction is the next gate** — refresh "Corpus-scale measured" table above and rerun per-strategy CER. Not blocking this iteration; planned as its own session.

---

**K7(b) speed diagnosis (2026-05-26).** Earlier conjecture of a 5× corpus slowdown was wrong. Investigation:

- Per-page YOLO inference (5 runs, ms): COCO@640 ~18ms ≡ DocLayNet@640 ~17ms; DocLayNet@832 ~27ms; DocLayNet@1024 ~40ms; DocLayNet@1280 ~76ms. **At equal imgsz, DocLayNet matches COCO speed exactly** — the model swap itself is free.
- Per-strategy F1 on the 5 DocLayNet fixtures shows imgsz=832 matches or beats imgsz=1280 on text-dominant fixtures; the 1280 advantage is concentrated in small-class recall (Caption / Footnote) which the model misses on this corpus at any resolution.
- Repeated 21-doc re-extraction (380s @ imgsz=1280, 444s @ imgsz=832) showed **wall-time is dominated by OCR, not layout** — run-to-run variance exceeds the per-call savings (~50 ms × ~50 calls = 2.5s out of ~400s). Redaction-detection regions and per-strategy CER are identical between imgsz=832 and 1280.
- The Early-childhood manifest doc (37 rotated table pages, 205s in both runs) routes through `spreadsheet_print` correctly and is not a K7(b) artefact — it's the orchestrator running both spreadsheet_print AND per-page processing on every page. Pre-existing cost.

**Outcome.** Default `imgsz` set to **832** in `_TAXONOMY_IMGSZ["doclaynet"]` ([paddle_ocr.py](src/womblex/ingest/paddle_ocr.py)). Lower than the model card's 1280 recommendation but empirically equivalent on this corpus. The full-corpus re-extraction is expected to land in roughly the same wall-time as the May 23 baseline (2h 35m), not 13 hours.

### Larger tracks (separate work)

- ~~**#A — Document-layout YOLO swap (K7(b) above).**~~ Landed 2026-05-25. DocLayNet checkpoint integrated; redaction-precision and figure-mis-classification both improved on measured cohorts. See K7(b) row in the fix-cluster table and the K2′ + K7(b) validation section above.
- **#B — Native-text footer whitespace artefacts** (stories §8). `3|P age` / `3| Page` from sub-glyph kerning. Belongs to a downstream cleaning op that rewrites `pages[i].text` — verbatim-text policy means extraction won't normalise.
- **#C — Inline-per-span source redactions.** Bracket-only unification (2026-05-21) is page-prefix only. Inline-per-span needs bbox-to-text-position mapping. Two paths with two different requirements:
  - **Raster pages**: depends on K2′ (wire PaddleOCR per-word bboxes through extraction).
  - **Native pages**: needs a separate text-to-bbox character-position mapping over `page.get_text("dict")` spans.
- **#D — Redaction-induced paragraph breaks on native pages** (stories §9). PyMuPDF `blocks` join with `\n\n` either side of a redaction. Fix paths: (a) refine `_render_blocks_with_breaks` adjacent-baseline detection or (b) downstream orphan-line re-join.
- **#E — CHUNKING.md generator.** Hand-maintained today (numbers from 2026-03-22; framings refreshed 2026-05-22). The other three accuracy docs have generators; this one doesn't. Adding a `generate_chunking_report` closes a quiet drift vector.
- **#F — Full corpus re-extraction.** Refresh the on-disk production parquet at `stories/.../womblex-extract/output/` with current K-cluster code behaviour. Not blocking anything (nothing downstream consumes the parquet yet), and the 12-doc sample already proved the K-cluster works. Defer until downstream pipeline work starts or until multiple code changes have stacked.

### Closed (this session)

- ~~K2 investigation~~ (2026-05-23; **retracted as a false finding** — bboxes are populated correctly at 100% for all native-text kinds; the audit's all-zero observation came from a `:.0f` formatter rounding 0-1 normalised floats. Narrower real issue surfaced as K2′ above.)
- ~~K-cluster QA pass~~ (2026-05-23; 33 modified files audited; 6 minor doc/STATUS fixes landed; 484 unit tests pass, 0 regressions)
- ~~Element-kind audit fix cluster~~ (2026-05-22; K1 / K3 / K4 / K5 / K7(a) / K8 landed — see fix cluster table above)
- ~~Measure #6 precision gain~~ (2026-05-22; −8 / −5.0% on the 11-doc cohort, 02737 unchanged — see [CHANGELOG.md](CHANGELOG.md) "Detector — raster-path layout filter" cohort measurement)
- ~~Audit non-`table` element kinds at corpus scale~~ (2026-05-22; surfaced K1-K8 above)
- ~~Decide `use_layout_filter` default~~ (2026-05-22; keep `True` — no doc regressed, modest per-doc cost; full-benchmark 7× slowdown acceptable)
- ~~Marker convention unification~~ (2026-05-21; see [CHANGELOG.md](CHANGELOG.md) "Redaction & PII marker conventions")
- ~~PDF annotation read probe~~ (2026-05-21; not viable for this corpus — see `stories/STATUS.md` Outstanding §4(b))
- ~~Doc-drift audit~~ (2026-05-21; 23 stale claims fixed across CLAUDE.md / README.md / architecture / dataflow / steering / accuracy generators)

## Don't

The conventions in `CLAUDE.md` continue to apply. One reminder worth
pinning here:

- **Don't retry an OCR-side table-detection relaxation** without
  introducing a new discriminator (layout-from-image, paragraph-gap
  vs cell-gap, column-spread-vs-page-width threshold). Four
  variants tried; all hit the same trade-off cliff. The relaxed
  rule helps real tables and hurts forms by exactly the same
  amount under per-region OCR input. Full write-up in
  [CHANGELOG.md](CHANGELOG.md) "Historical engineering notes →
  What changed → 5. (reverted) OCR-side `_table_aware_text` relaxation".
