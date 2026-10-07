# Permissive dependencies — plan

*Status: in progress (2026-10). L1 has shipped (#130), the README layout step is done, L2 is done, and P1 to P3b, D1 (both in #151), P4a and P5 are done: every extractor reads through the seam, `fitz` is imported only in `_fitz.py`, the permissive dependencies are in the lock, and test PDFs that are written to a file are built on reportlab; the pdfium backend opens PDFs and images with geometry, rendering, images, drawings and widgets (text and tables to come). The Phase 0 spike is done (scratch only; outcome in `decisions.md`). H-B's between-backend half has landed in womblex-benchmark, publishing `docs/accuracy/BACKEND_PARITY.md`; its CER-against-transcripts half still waits on the ground-truth revision, as does every other step gated on a GT-scored report — L1-B and F1-B's regeneration, and those only. Everything else is deliverable now: the P2/P3 identity gates read H-B, which runs today, so the critical path is now P4 to P7, then P8/F1/F2, with L3 independent of all of them. Each merge updates this document's merge list as it lands, and the document is retired into `decisions.md` once F2 ships.*

## Context
Womblex is Apache-2.0. Two of its core dependencies are licensed AGPL-3.0, with a paid commercial licence as the only alternative:

- **`ultralytics`.** It runs the layout detector, `YOLOLayoutAnalyzer` in `ingest/paddle_ocr.py`. The fallback `yolov8n.pt` weights are Ultralytics', and the primary `yolo11n_doc_layout.pt` weights were trained with Ultralytics.
- **`pymupdf` (`import fitz`).** It opens every PDF and every standalone image.

The plan removes both and declares any capability that is lost.

**Decisions taken:**
- Both dependencies go.
- YOLO is removed outright, not kept as an optional extra.
- The replacement layout model is committed under `src/womblex/_models/`.
- YOLO is retired as part of adapting womblex for cloud deployment, where any layout model can be plugged into the registry's layout slot and benchmarked on consumption pricing. No controlled YOLO-versus-PP-DocLayout run is made; the last YOLO numbers are the 2026-08 reports.
- Layout becomes its own stage (L3), so a layout model can be tuned and measured on its own.
- Until L3 ships, the README states that layout detection is not supported for local deployment.
- L1-B waits on the ground-truth revision. L2 does not wait for it.

**What layout actually feeds.** On every page where layout succeeds, the non-table regions collapse into one OCR text block. The model therefore contributes only three things:
- table rects, passed to `reconstruct_table` in `ingest/strategies_scanned.py`;
- figure and table rects, used as redaction exclusion zones by `redact/stage.py`;
- the dominant region's kind.

**Layout measurements so far.** PP-DocLayout-M was converted to ONNX with `paddle2onnx`. It is Apache-2.0, 23 MB, and runs on `onnxruntime` at about 65 ms per CPU page. Scored with the benchmark's own DocLayNet scorer:

| Measure | PP-DocLayout-M | YOLO |
|---|---|---|
| Layout F1 | 0.297 | 0.303 |
| The one real ground-truth table (`dense_text_548`) | found | found |
| Table regions on 5 FUNSD forms | 3 | 5 |

The DocLayNet fixtures hold one table and one picture, so they cannot separate the two models. The end-to-end suites below are the real gate.

**PyMuPDF surface.** A document is opened in five places, and `fitz.Document`, `Page` and `Rect` objects cross twelve modules. The calls in use are:
- `get_text` in its plain, `dict`, `words` and `blocks` forms, with `TEXT_DEHYPHENATE` / `TEXT_PRESERVE_WHITESPACE`;
- `find_tables`, with the `lines` and `text` strategies;
- `get_pixmap`, including with `clip`;
- `get_images` and `get_image_rects`;
- `get_drawings`;
- `widgets`;
- `rotation` and `rotation_matrix` (used in `spreadsheet_print` only);
- opening a PNG or JPEG as a one-page document.

Nothing in `src/` writes a PDF. The test builders in both repositories do.

## Approach
1. **Layout first.** It is small and self-contained.
2. **Then PyMuPDF:**
   1. Put a womblex-owned `ingest/pdf/` seam in front of fitz.
   2. Prove the seam is byte-identical on the fitz backend.
   3. Build the permissive backend behind it.
   4. Measure the two backends against each other.
   5. Flip the default.
   6. Delete fitz.

The merges are listed below. W is womblex, B is womblex-benchmark (a paired merge), and an approval tag means the merge edits `pyproject.toml` and needs human sign-off.

### Phase 0 — spike. Done (scratch only, nothing merged).
Checked against the vendored fixtures and the womblex-collection PDFs before the seam's shape is fixed. Outcome recorded in [`decisions.md`](decisions.md#permissive-dependencies-phase-0--pypdfium2-over-pdfminer-for-the-text-engine). Reversed the plan's working assumption: pypdfium2's character-rebuild is the P6 text-engine candidate, not pdfminer's native `LAParams` segmentation, which misses the speed gate by ~67x — on fidelity to fitz pdfminer is the better of the two, so P6 owns real segmentation work. pdfplumber tables are faster than fitz's but no better calibrated. Two of the three mechanism claims are settled as stated; the third came back the other way, and `get_images` turns out to list draws rather than resources, which leaves a latent quadratic duplication in `_extract_images_from_page` for P5 not to port. On threading: PDFium is not thread-safe (confirmed from pypdfium2's own docs, not just measurement) and needs a mutex around every call if ever shared across threads in one process — moot for now, since `womblex serve` never calls fitz/pdfium directly and `cloud/worker.py` is already single-threaded per process. P1/P5 just need to keep it that way.

- **Text engine, library-native first.** Compared against fitz `words` and `blocks`:
  - pdfminer.six's own layout analysis (`LAParams`: `LTTextLine` and `LTTextBox`, reached through pdfplumber);
  - lines and blocks rebuilt from pypdfium2 characters (`FPDFText_*` boxes, size, weight).

  Measured between-backend CER and wall time per page over 76 native-text pages. pdfminer's segmentation was expected to be the default since the library does it natively; it tracks fitz closely (median CER 0.010) but costs 67x fitz per page, so the speed gate rejected it (see `decisions.md` for the table).
- **Tables.** pdfplumber's `TableFinder` (lines and text strategies) against `fitz.find_tables`: table counts and shapes on table pages, and time per page. `page_profile` calls `find_tables` on every page, so timing matters.
- **Unverified mechanism claims, settled by measurement:**
  - MuPDF's page rect for a PNG or JPEG, with and without a dpi tag;
  - whether `get_images` lists image resources rather than drawn images — **settled: draws**, the opposite of the assumption, and `_extract_images_from_page` emits N² elements for an image drawn N times on a page (latent; no fixture triggers it). See `decisions.md`.
  - whether PDFium is safe to call from `womblex serve`'s threads — **settled: no**, a process-wide mutex is required for any in-process thread concurrency over pdfium (per pypdfium2's own docs); not a live risk today since extraction only ever runs single-threaded in `cloud/worker.py`. See `decisions.md`.

### Layout
- **L1 (W). Shipped (#130).** `PPDocLayoutAnalyzer` in `ingest/layout_onnx.py`, the model under `_models/pp-doclayout-m/`, registered as the default of the registry's layout slot (`pp-doclayout-m`). Two follow-ups, which ride with L2:
  - add the `layout_onnx.py` row to the `CLAUDE.md` module table;
  - `onnxruntime` is imported with `type: ignore[import-untyped]`; L2's direct dependency replaces it with a mypy override.

  Re-exporting the model: `PaddlePaddle/PP-DocLayout-M` at the revision in `docs/models.md`, then `paddle2onnx --model_dir . --model_filename inference.json --params_filename inference.pdiparams --save_file inference.onnx --opset_version 14` in a venv with `paddle2onnx`, `paddlepaddle`, `onnx`, `onnxruntime` and `setuptools` (`paddle2onnx` 2.1.0 needs `setuptools` and pulls in neither). The export takes `image` (N,3,640,640) and `scale_factor` (N,2), batch size 1. A different `paddle2onnx` version may change the ONNX digest.
- **README (W), with L2. Done, landing with the README edit.** The README states that layout detection is not supported for local deployment until L3 ships, and its stale "layout regions via `ultralytics` YOLO" line is replaced.
- **L2 (W, approval). Done.** One dependency-scoped removal. YOLO is no longer registered, so this is dead code and the registry, model check and `DEFAULT_MODELS` do not change:
  - delete `YOLOLayoutAnalyzer`, the `_YOLO_*` maps, `_select_label_map`, `_TAXONOMY_IMGSZ`, both `.pt` files (in `_models/` and `models/`) and the `TestYolo*` classes;
  - drop `ultralytics` and its mypy override, and make `onnxruntime` a direct dependency;
  - fix the YOLO mentions in `.semgrep/rules/deserialisation.yaml`, the CI disk-space comment, `CLAUDE.md`, `docs/models.md` (both `.pt` rows and the "without `ultralytics`" fallback note), `redact/stage.py`, `redact/detector.py`, `ingest/llm_ocr.py`, `utils/models.py`, `configs/example.yaml`, `tests/test_extract.py` and `tests/test_models_resolution.py`.
- **L3 (W). Layout as its own stage.** A design note comes first and settles:
  - what the stage persists: layout regions per page, keyed by `(source_hash, page)`, in its own `store/<stage>_output.py` sidecar;
  - how OCR table reconstruction gets its table rects. Today it runs inside the OCR page operation on the same render. Either the layout stage runs before OCR within the batch, or OCR regions are persisted so reconstruction can run downstream;
  - whether redaction reads the same regions instead of building its own analyser from `redaction.layout_model`;
  - that the model is still chosen through the registry's layout slot. A tuned model is a plugin, never a new toggle. The stage adds what the slot cannot: its own inputs, outputs and checkpoint, so a model can be rerun and measured without re-running OCR.

  When L3 ships, the README note from L2 is removed.
- **L1-B (B). Waits on the ground-truth revision.** When it runs, it scores PP-DocLayout-M against the revised ground truth; there is no YOLO comparison.
  - Make the DocLayNet layout-F1 test honour `--model` (it calls `get_layout_analyzer()` with no arguments today).
  - Bring the `DOCLAYNET_TO_WOMBLEX` comment and the YOLO wording in `accuracy_reports.py` in line with `LABEL_MAP`.
  - The table benchmark and the false-table cohort feed ground-truth or whole-page rects to `reconstruct_table` and never run layout, so they are not layout gates. The layout gate on tables is end-to-end: tables emitted by `extract_text` on the FUNSD and DocLayNet scanned fixtures.
  - Check the findings from L1's uncontrolled run: table-class recall fell from 50% to 25%, and `dense_text_548` gave three table regions where the ground truth has one, with a `chart` box (mapped to `figure`) almost identical to the `table` box. `LABEL_MAP` and the 0.3 threshold are the knobs.
  - Regenerate `EXTRACTION.md` and `REDACTION_HANDLING.md`, and spot-check exclusion area on the 02737-class scanned forms.

### PyMuPDF
- **H-B (B), first. Between-backend half landed** (`accuracy/backend_parity.py` + `accuracy/test_backend_parity.py` in womblex-benchmark, publishing `docs/accuracy/BACKEND_PARITY.md`). A backend-diff harness. It extracts the womblex fixtures and the womblex-collection with two womblex builds or two backends, and compares:
  - open failures, page count, doc type, every `PageProfile` field (the page rect included) and per-page plan operation;
  - element kinds and counts, `content_digest`, table shapes, form fields and the full per-page geometry of every detected redaction;
  - CER between the two, and against transcripts;
  - timing, split by phase.

  It writes `BACKEND_PARITY.md`. It lands first because it is also the identity gate for P2 and P3, which keeps any scoring against the collection inside the benchmark. The between-backend comparisons need no ground truth and can run now; CER against transcripts waits on the ground-truth revision. **Landed now:** every axis above except CER (no second backend exists yet to disagree with the first — see Phase 0's note in [`decisions.md`](decisions.md) — so today's run is both sides on the current fitz build, i.e. a repeatability baseline). 45 fixtures, identical. Three bounds the gate's wording should be read with: PDFs are capped to their first 20 pages (uncapped, the 406-page Auditor-General fixture pushed one run past an hour on OCR-dispatched pages alone), the harness opens with `fitz` so the DOCX/XLSX/CSV/XML/TXT fixtures reach `tests/test_default_digest.py` instead, and the comparison is reflexive while there is one backend — so the suite asserts a floor (every fixture opened and produced elements) and the report publishes element, table and redaction totals, or uniform breakage would read as agreement. CER (between backends and against transcripts) is deferred until a real second backend exists.
- **P1 (W). Split in two: the vocabulary, then the entry point and first backend.** Written out as one merge it is ~580 changed lines, so it ships as P1a then P1b; each passes the suite alone.
  - **P1a. Done.** `ingest/pdf/types.py`, no third-party imports at runtime (numpy under `TYPE_CHECKING` for `render`'s annotation): `Rect` (normalising, with `transform` for the rotated-page path); `Span`, `Line`, `Block`; `Word`, a NamedTuple shaped like fitz's 8-tuple so `grid_projection` slicing is unchanged; `FoundTable`, `Drawing`, `Widget`, `PageImage`; and the `Page` and `Document` protocols. Page methods are typed rather than fitz-shaped: `plain_text(dehyphenate)`, `text_dict`, `words(dehyphenate)`, `text_blocks(dehyphenate)`, `find_tables(strategy)`, `render(dpi, clip) -> ndarray`, `images`, `drawings`, `widgets`, `rect`, `rotation`, `rotation_matrix`, `number`, `doc_name`. With `tests/test_pdf_types.py` and the three doc updates. 356 lines.
  - **P1b. Done.** `ingest/pdf/__init__.open_document(path, backend=…)` resolving its backend through `_BACKENDS` at call time, and `ingest/pdf/_fitz.py` wrapping fitz. `test_public_api` now asserts that neither `import womblex` nor importing the seam loads *any* PDF backend, not just fitz. With `tests/test_pdf_fitz.py`. ~430 lines.
  - Two shape decisions worth carrying forward. `render` returns an `(h, w, 3)` uint8 RGB array, absorbing `_pixmap_to_array` and dropping the alpha branch its callers never used. `images()` returns one `PageImage` per *draw*, de-duplicating xrefs first — which is the fix for the latent N² duplication in `_extract_images_from_page` (see `decisions.md`); it changes no digest in the corpus, since no fixture page repeats an xref, so P2's identity gate is unaffected.
- **P2 (W). Done.** Port `extract.py`, `orchestrator.py`, `strategies_file.py`, `grid_projection.py` and `forms.py` to the seam. Gate: H-B reports identical `content_digest` against `main`, and `tests/test_default_digest.py` passes unchanged. That test pins the digests of two native PDFs and four non-PDF fixtures, so it is a CI-speed identity check beside H-B's full one.
  - Landed with a transitional `ingest.pdf.native(obj)` that unwraps the backend page or document for the not-yet-ported callees (`_ocr_page` and friends, `extract_spreadsheet_print`); it went with P3b. `FitzDocument.wrap` (the tests' bridge, since they still build documents with fitz) went with P4b. `_fitz` now omits `flags` instead of passing 0 when not dehyphenating.
- **P3a (W) + P3a-B (B). Done.** Port `detect.py`, `page_profile.py` and `morphology.py`. Gate: identical digests and `PageProfile`s. `backend_parity.py` called `fitz.open` then `profile_pages(doc)`, which this port breaks, so the harness opens through `open_document` in the paired merge. Checked beyond H-B with an old-versus-new comparison of `PageProfile` and `DocumentProfile` over every fixture PDF and image: no differences. `ingest.pdf.native` no longer serves `page_profile`; it goes with P3b.
- **P3b (W). Done.** Port `strategies_scanned.py`, `spreadsheet_print.py`, `redact/stage.py` and `redact/batch.py`. Gate: identical digests and `RedactionReport`s, and `fitz` appears in `src/` only in `_fitz.py`. `ingest.pdf.native`, the `native` properties on the fitz adapter and `extract._pixmap_to_array` are deleted. Checked with an old-versus-new run over all 13 fixture PDFs and images (first 6 pages): `content_digest`, `RedactionReport` geometry with and without the layout filter, and `extract_spreadsheet_print` output are identical, as are unrotated synthetic pages. One difference, on rotated pages only: `Rect.transform` is double precision where fitz's `Rect * Matrix` rounds to float32, so rotated spreadsheet-print spans move by up to 1.5e-5 pt. No vendored fixture is rotated; the womblex-collection run through H-B is still owed.
- **D1 (W, approval). Done, in the P3b PR as its own commit.** `pypdfium2>=4.30`, `pdfplumber==0.11.10` and `pillow>=10.0` in core, `reportlab>=4.0` in `[dev]`; the lockfile resolves pypdfium2 5.14.0, pdfplumber 0.11.10 (with pdfminer.six), pillow 12.3.0 (pdfplumber 0.11.10 requires `>=12.2`, so pillow moves from 12.1.1) and reportlab 5.0.1. Nothing imports them yet.
- **P4 (W) + P4-B (B). Split in two**, since as one merge it sits on the 500-line cap. Afterwards fitz stays in tests only where it is the subject: `test_pdf_fitz.py`, the import checks in `test_pdf_types.py` and `test_public_api.py`, and `test_default_digest.py`'s `fitz.VersionBind` guard, which F1 moves.
  - **P4a. Done.** `tests/_pdf_builders.PdfBuilder`: a stateful builder over reportlab's `Canvas` — pages, text at a baseline point, filled rects (fill and 1pt stroke in one colour), images kept to aspect and centred — in top-left points with fitz's defaults (A4, 11pt Helvetica), saved with `invariant=1`, read back through `open_document` with no backend named. No textbox helper: the one `insert_textbox` caller never wraps. Ported the sites that already wrote a file: `test_redaction`, `test_model_registry`, both `_image_to_pdf` copies and `test_extract`'s page-break tests. Checked against the fitz builders: identical redaction boxes on all seven vector cases, identical `PageProfile` and word boxes on text pages, pixel-identical renders of image pages at three page sizes.
  - **P4b. Done.** The live-page fixtures — `letter_page` in `test_grid_projection` and `blank_page` in `test_table_reconstruction` (with its `fitz.Page` annotations, `get_pixmap` calls and `_png`) — and `test_extract`'s block-count and prose-gate tests; the `fitz.open` readers in `test_spreadsheet_print`; then delete `FitzDocument.wrap`. Check that the prose page still gives about one block, or the prose gate test passes without the gate firing.
  - **P4-B**, paired with P4b: in the benchmark, port the table-fixture builder in `test_table_benchmark.py` and the `page_count` call in `test_fixture_accuracy.py`.
  - Lines, text fields and page rotation joined the builder with P5's pdfium tests.
- **P5 (W). Done, split in two**, since as one merge it is ~800 lines: the PDF document and page, then the image path.
  - **P5a. Done.** `ingest/pdf/_pdfium_doc.py`: document and page, registered as `backend="pdfium"`. User space is flipped against the crop box into fitz's top-left space; form XObjects are descended and composed through their matrices; drawing rects come from path points (pdfium's `get_bounds` adds stroke width, MuPDF's rect does not); `render` draws through `FPDF_RenderPageBitmap` into a bitmap sized by `types.render_box`, MuPDF's round-out rule. Text methods and `find_tables` raise `NotImplementedError` until P6/P7. `PageImage.xref` is 0, since pdfium exposes no object numbers.
  - **P5b. Done.** `ingest/pdf/_image.py`: MuPDF's image formats that Pillow decodes (PNG, JPEG, TIFF, BMP, GIF, JPEG 2000, PNM, PSD) plus WebP and AVIF, through Pillow, one page per frame — an animated GIF, PNG or WebP or an MPO JPEG gives every frame where MuPDF gave the first; a PSD is its composite — any other format a `ValueError`. An undeclared JPEG 2000 is 72dpi, as MuPDF measured it; a JPEG 2000 or PSD that declares a resolution is not yet checked against MuPDF; 16-bit grey is scaled to 8 bits, where Pillow would clip it. Phase 0's rule re-measured and widened: the horizontal resolution on both axes, rounded; 96 when undeclared; 72 outside 72..4800; EXIF orientation applied. It reads each format's own fields, because Pillow reports 72 for a JPEG whose EXIF lacks a resolution and 1 for an untagged TIFF. Renders resample bilinearly, the Pillow filter closest to MuPDF on the vendored PNGs (worst mean absolute difference 1.63 of 255 over the 15 vendored PNGs at 72, 150 and 300 dpi). `_pdfium_doc.open_document` sniffs the PDF header in the first kilobyte, as MuPDF did, and hands anything else to it. Checked against fitz on every vendored PNG and on synthetic PNG/JPEG/TIFF files: page rects agree, including multi-frame TIFF and EXIF orientation. `render_box` keeps `fz_round_rect`'s 0.001px tolerance, so 595.2pt at 150dpi is 1240px as under MuPDF.
  - P5a checked against fitz over every vendored fixture PDF (first 20 pages), plus synthetic pages (rotated, offset crop box, nested forms, widgets): page rects, rotation matrices, image rects, drawing kinds and rects, widgets and render shapes agree. Differences: drawing rects by under 0.01pt (single versus double precision), fill colours quantised to 1/255, and anti-aliasing.
  - For P8: MuPDF's `get_drawings` also reports annotation and widget appearance streams, which pdfium's page objects do not hold (`FPDFAnnot_GetObject` reaches them, in appearance space). No vendored fixture is affected.

  Images stay on the orchestrator path; there is still no separate image extractor.
- **P6 (W).** `ingest/pdf/_text.py`, the text engine Phase 0 chose. It covers dehyphenation, and bold detection from font weight or name.
- **P7 (W).** `ingest/pdf/_tables.py`, an adapter onto pdfplumber's `TableFinder`.
- **P8… (W).** Fidelity fixes driven by `BACKEND_PARITY.md`, repeated until the gates below hold.
- **F1 (W) + F1-B (B).**
  - Flip `open_document`'s default to the permissive backend.
  - Regenerate `EXTRACTION`, `REDACTION_HANDLING`, `PII_CLEANING`, `READING_ORDER` and `CHUNKING`, plus the table and false-table suites.
  - Re-pin `tests/test_default_digest.py` deliberately, and move its version guard from `fitz.VersionBind` to the pypdfium2 version.
- **F2 (W, approval).**
  - Delete `_fitz.py` and the backend argument, and drop `pymupdf`, its mypy override and the `pymupdf_layout` warning filter.
  - Simplify the CI fitz-notice workaround and the Dockerfile comments.
  - Rewrite `CLAUDE.md`: the PyMuPDF-import and dehyphenation pitfalls, and the rule that "everything fitz can open routes through `extract_text`".
  - Update `dataflow`, `heuristics_disambiguation` and the README.
  - Move this plan's calls into `decisions.md`, and retire this document.

**Reuse:**
- `utils/models.resolve_local_model_path`, which digests the model into the run stamp;
- `store/content_digest.content_digest`, the identity handle;
- the benchmark's `_match_layout_regions` / `_aggregate_doclaynet_blocks`;
- `extract._pixmap_to_array`'s RGB and alpha handling, which moves into `render()`.

**Out of scope, kept as-is for parity:**
- the unrotated-text versus rotated-`page.rect` mismatch outside `spreadsheet_print`.

Adding images to the CLI's `SUPPORTED_EXTENSIONS`, kept out of this plan, landed as its own merge after P5: the formats the default fitz backend opens, without WebP or AVIF.

## Open questions
- **Record the PDF library version in the run stamp?** `content_digest` depends on the PDF library's version, but no footer records it. Options: add it alongside D1, when pypdfium2 arrives; raise it as a separate requirement; or leave it, since the womblex version and `uv.lock` already pin the library.
- **No PDF-backend registry slot.** Settled, recorded here because the registry makes it tempting: the registry is for swappable models, and a backend slot would be the toggle this plan rules out.

## Conventions this plan holds to
- **Merge size.** At most 500 changed lines per merge, green on ruff, mypy and pytest. `uv.lock` doesn't count. P3 is pre-split; any merge that grows splits again before review.
- **File size.** At most 750 lines per file. `ingest/pdf/` is split by concern, so no backend file approaches the cap.
- **Thin adapters.** Library-native behaviour is preferred (pdfminer layout analysis, pdfplumber `TableFinder`). Womblex code covers only coordinates, typing and the fitz-compatible shapes its callers need.
- **No toggles.** Backend selection is a private `open_document` argument used by H-B. It is not config or an environment variable, and it is deleted at F2. The `Page` / `Document` protocols stay afterwards because PDF and image pages remain two implementations.
- **Docs move with the code.** Every merge that adds, moves or retires a module updates `architecture.md`, `project-structure.md` and the `CLAUDE.md` module table in the same PR. Every merge adds a CHANGELOG entry and ticks itself off here.
- **Verbatim text.** Dehyphenation and segmentation are extractor behaviour, so they live in the backend, never as a post-pass.
- **Unusual input warns and continues.** A format the new backend cannot open (fitz also opened XPS, EPUB, MOBI, CBZ and SVG) becomes a per-document error status with the document ID logged. It never aborts the batch.
- **Benchmark boundary.** All scoring against womblex-collection ground truth, and every `docs/accuracy/` report, is produced in womblex-benchmark.

## Declared behaviour changes
- **Fewer formats.** Calling `extract_text` directly can no longer open XPS, EPUB, MOBI, CBZ or SVG, nor an image format MuPDF decodes and Pillow does not. PDF and the common image formats remain, and WebP and AVIF are new. None of the dropped formats is reachable through the CLI or API today.
- **New content digests.** `content_digest` changes at F1, and `tests/test_default_digest.py` is re-pinned in that merge. Text stays verbatim, but it comes from a different producer.
- **Small text differences.** Paragraph segmentation, dehyphenation and bold-based heading detection may shift slightly.
- **Image counts.** `image_count` may count drawn images only, which moves the sub-page OCR gate. Phase 0 confirms whether it does.
- **Rendering.** Anti-aliasing differs, so OCR output differs slightly.
- **Speed.** The table pass may be slower. Phase 0 measures it, and the gate bounds it.

## Gates
These are migration gates, not quality scores. They retire with this plan.

| Merge | Gate |
|---|---|
| Every merge | `uv run ruff check src/ tests/`, `uv run mypy src/` and `uv run python -m pytest tests/ -v` pass; `uv lock --check` passes on approval merges; touched files are under 750 lines (`wc -l`); `git diff --stat $(git merge-base HEAD origin/main)..HEAD` is within the cap |
| L1-B | Waits on the ground-truth revision. DocLayNet F1 ≥ 0.29 with the `dense_text_548` table found; end-to-end tables emitted on the scanned fixtures checked by hand; no regression in `REDACTION_HANDLING` |
| P2, P3a, P3b | Identical `content_digest`, `PageProfile` (page rect included) and per-page redaction geometry on every fitz-openable fixture, as reported by H-B — PDFs to their first 20 pages, images whole; `tests/test_default_digest.py` unchanged, which is what gates the non-PDF formats H-B cannot open |

F1 flips the default only when all of these hold:

| Measure | Required |
|---|---|
| New open failures | Zero |
| Doc type | ≥ 98% agreement |
| Per-page plan operation | ≥ 99% agreement |
| `has_text_layer` | 100% agreement |
| Native-text CER between backends | Median ≤ 0.01, p95 ≤ 0.03 |
| Auditor-General transcript CER | ≤ 0.216 (currently 0.211) |
| ACT-ECI CER | No strategy worse by more than 0.01 |
| Table count | Equal on ≥ 95% of table pages |
| Table-benchmark F1 | Drops by at most 0.01 |
| False-table cohort | Does not grow |
| Vector-redaction counts | Identical |
| AcroForm and FUNSD field counts | Equal |
| `CHUNKING`, `READING_ORDER`, `PII_CLEANING`, `REDACTION_HANDLING` | No regression |
| Native extraction | ≤ 1.5× the fitz wall time |
