# Permissive dependencies — plan

*Status: proposed (2026-10). No merge has shipped. Each merge updates this document's merge list as it lands, and the document is retired into `decisions.md` once F2 ships.*

## Context
Womblex is Apache-2.0. Two of its core dependencies are licensed AGPL-3.0, with a paid commercial licence as the only alternative:

- **`ultralytics`.** It runs the layout detector, `YOLOLayoutAnalyzer` in `ingest/paddle_ocr.py`. The fallback `yolov8n.pt` weights are Ultralytics', and the primary `yolo11n_doc_layout.pt` weights were trained with Ultralytics.
- **`pymupdf` (`import fitz`).** It opens every PDF and every standalone image.

The plan removes both and declares any capability that is lost.

**Decisions taken:**
- Both dependencies go.
- YOLO is removed outright, not kept as an optional extra.
- The replacement layout model is committed under `src/womblex/_models/`.

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

### Phase 0 — spike (scratch only, nothing merged)
Check these against the vendored fixtures and the womblex-collection PDFs before the seam's shape is fixed. Record the outcome in `decisions.md`, under Rejected approaches if a path fails.

- **Text engine, library-native first.** Compare against fitz `words` and `blocks`:
  - pdfminer.six's own layout analysis (`LAParams`: `LTTextLine` and `LTTextBox`, reached through pdfplumber);
  - lines and blocks rebuilt from pypdfium2 characters (`FPDFText_*` boxes, size, weight).

  Measure between-backend CER, paragraph-block counts and wall time per page. pdfminer's segmentation is the default because the library does it natively. Rebuilding segmentation in womblex is only justified if pdfminer misses the speed gate.
- **Tables.** pdfplumber's `TableFinder` (lines and text strategies) against `fitz.find_tables`: table counts and shapes on table pages, and time per page. `page_profile` calls `find_tables` on every page, so timing matters.
- **Unverified mechanism claims, to settle by measurement:**
  - MuPDF's page rect for a PNG or JPEG, with and without a dpi tag;
  - whether `get_images` lists image resources rather than drawn images;
  - whether PDFium is safe to call from `womblex serve`'s threads.

### Layout
- **L1 (W).**
  - Add `PPDocLayoutAnalyzer`, satisfying the `LayoutAnalyzer` protocol (`ingest/interfaces/protocols.py`). It goes in `ingest/paddle_ocr.py`, or in a new `ingest/layout_onnx.py` if that file would pass the cap.
  - Read `label_list` and the preprocessing from the model's own `inference.yml`.
  - Map labels to `block_type`: table to `table`; image, chart and seal to `figure`; the rest to text kinds. Keep the 0.3 threshold.
  - Commit `_models/pp-doclayout-m/inference.{onnx,yml}`. Leave it out of package-data, as the YOLO weights are, and resolve it with `utils/models.resolve_local_model_path`.
  - Repoint `get_layout_analyzer()`. `strategies_scanned.py` and `redact/stage.py` key off `block_type` and do not change.
  - Docs: `docs/models.md` (source revision, `paddle2onnx` version, SHA-256), `_models/README.md`, `architecture.md`, `project-structure.md`, the `CLAUDE.md` row and CHANGELOG.
- **L1-B (B).**
  - Bring `DOCLAYNET_TO_WOMBLEX` and the report wording in line with the new labels.
  - Regenerate `EXTRACTION.md`, the table benchmark, the false-table cohort and `REDACTION_HANDLING.md`.
  - Spot-check exclusion area on the 02737-class scanned forms.
- **L2 (W, approval).** One dependency-scoped removal:
  - delete `YOLOLayoutAnalyzer`, the `_YOLO_*` maps, `_select_label_map`, `_TAXONOMY_IMGSZ`, both `.pt` files (in `_models/` and `models/`) and the `TestYolo*` classes;
  - drop `ultralytics` and its mypy override, and make `onnxruntime` a direct dependency;
  - fix the YOLO notes in `.semgrep/rules/deserialisation.yaml`, the CI disk-space comment and `CLAUDE.md`.

### PyMuPDF
- **H-B (B), first.** A backend-diff harness. It extracts the womblex fixtures and the womblex-collection with two womblex builds or two backends, and compares:
  - open failures, doc type, `PageProfile` fields and per-page plan operation;
  - element kinds and counts, `content_digest`, tables, form fields and vector redactions;
  - CER between the two, and against transcripts;
  - timing.

  It writes `BACKEND_PARITY.md`. It lands first because it is also the identity gate for P2 and P3, which keeps any scoring against the collection inside the benchmark.
- **P1 (W).**
  - `ingest/pdf/types.py`, with no third-party imports: `Rect`; `Span`, `Line` and `Block`; `Word`, a NamedTuple shaped like fitz's 8-tuple so `grid_projection` slicing is unchanged; `FoundTable`, `Drawing`, `Widget`; and the `Page` and `Document` protocols.
  - The page methods are typed rather than fitz-shaped: `plain_text`, `text_dict`, `words(dehyphenate)`, `text_blocks(dehyphenate)`, `find_tables(strategy)`, `render(dpi, clip) -> ndarray`, `image_rects`, `drawings`, `widgets`, `rect`, `rotation`, `rotation_matrix`, `number`, `doc_name`.
  - `ingest/pdf/__init__.open_document(path)` imports its backend lazily. `ingest/pdf/_fitz.py` wraps fitz.
  - Extend `test_public_api` so `import womblex` loads no PDF backend.
  - Docs: `architecture.md`, `project-structure.md`, the `CLAUDE.md` module table.
- **P2 (W).** Port `extract.py`, `orchestrator.py`, `strategies_file.py`, `grid_projection.py` and `forms.py` to the seam. Gate: H-B reports identical `content_digest` against `main`.
- **P3a (W).** Port `detect.py`, `page_profile.py` and `morphology.py`. Gate: identical digests and `PageProfile`s.
- **P3b (W).** Port `strategies_scanned.py`, `spreadsheet_print.py`, `redact/stage.py` and `redact/batch.py`. Gate: identical digests and `RedactionReport`s, and `fitz` appears in `src/` only in `_fitz.py`.
- **D1 (W, approval).** Add `pypdfium2`, `pdfplumber` (pinned) and `pillow` to core, and `reportlab` to `[dev]`. Dependencies only; the lockfile lands as its own change.
- **P4 (W) + P4-B (B).**
  - Add `tests/_pdf_builders.py` on reportlab (filled rects, text at a point or in a box, embedded PNG). Rendering goes through the seam.
  - Port the builders in `test_redaction`, `test_extract`, `test_grid_projection`, `test_table_reconstruction`, `test_fixtures` and `test_bench_ocr_accuracy`.
  - In the benchmark, port the table-fixture builder in `test_table_benchmark.py` and the `page_count` call in `test_fixture_accuracy.py`.
- **P5 (W).**
  - `ingest/pdf/_pdfium_doc.py`: document and page, coordinates converted to fitz's top-left convention, rendering with crop, image objects, filled paths with fill colour, and widgets.
  - `ingest/pdf/_image.py`: PNG, JPEG and TIFF through Pillow, using the page-rect rule found in Phase 0.

  Images stay on the orchestrator path; there is still no separate image extractor.
- **P6 (W).** `ingest/pdf/_text.py`, the text engine Phase 0 chose. It covers dehyphenation, and bold detection from font weight or name.
- **P7 (W).** `ingest/pdf/_tables.py`, an adapter onto pdfplumber's `TableFinder`.
- **P8… (W).** Fidelity fixes driven by `BACKEND_PARITY.md`, repeated until the gates below hold.
- **F1 (W) + F1-B (B).**
  - Flip `open_document`'s default to the permissive backend.
  - Regenerate `EXTRACTION`, `REDACTION_HANDLING`, `PII_CLEANING`, `READING_ORDER` and `CHUNKING`, plus the table and false-table suites.
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
- the unrotated-text versus rotated-`page.rect` mismatch outside `spreadsheet_print`;
- adding images to the CLI's `SUPPORTED_EXTENSIONS`.

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
- **Fewer formats.** Calling `extract_text` directly can no longer open XPS, EPUB, MOBI, CBZ or SVG. PDF, PNG, JPEG and TIFF remain. None of the dropped formats is reachable through the CLI or API today.
- **New content digests.** `content_digest` changes at F1. Text stays verbatim, but it comes from a different producer.
- **Small text differences.** Paragraph segmentation, dehyphenation and bold-based heading detection may shift slightly.
- **Image counts.** `image_count` may count drawn images only, which moves the sub-page OCR gate. Phase 0 confirms whether it does.
- **Rendering.** Anti-aliasing differs, so OCR output differs slightly.
- **Speed.** The table pass may be slower. Phase 0 measures it, and the gate bounds it.

## Gates
These are migration gates, not quality scores. They retire with this plan.

| Merge | Gate |
|---|---|
| Every merge | `uv run ruff check src/ tests/`, `uv run mypy src/` and `uv run python -m pytest tests/ -v` pass; `uv lock --check` passes on approval merges; touched files are under 750 lines (`wc -l`); `git diff --stat $(git merge-base HEAD origin/main)..HEAD` is within the cap |
| L1-B | DocLayNet F1 ≥ 0.29 with the `dense_text_548` table found; no regression in table reconstruction, the false-table cohort or `REDACTION_HANDLING` |
| P2, P3a, P3b | Identical `content_digest`, `PageProfile` and `RedactionReport` on every fixture, as reported by H-B |

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
