# Permissive dependencies — outstanding work

*Status: in progress (2026-10). Outstanding: P8, F1, F2 (PyMuPDF). Everything shipped is recorded in [`CHANGELOG.md`](../CHANGELOG.md), [`architecture.md`](architecture.md), [`models.md`](models.md) and [`decisions.md`](decisions.md), not here. Every benchmark-side action (L3c-B, L1-B, H-B, F1-B, the womblex-collection runs and every gate scored in womblex-benchmark) lives in [`plan-post-gt-baseline.md`](plan-post-gt-baseline.md), and nothing here waits on it. Everything left here is deliverable now, gated on what this repository can measure. Each merge updates this list as it lands, and the document is deleted once F2 ships.*

## Context
Womblex is Apache-2.0, so its dependencies must be licence-compatible. The remaining incompatible one is `pymupdf` (`import fitz`; opens every PDF and standalone image) and it is being replaced: every extractor already reads through the `ingest/pdf/` seam, `fitz` is imported only in `ingest/pdf/_fitz.py`, and the pdfium backend (`backend="pdfium"`) opens PDFs and images with geometry, rendering, images, drawings, widgets and text. Its `find_tables` is `_tables.py` (P7). The default backend is still fitz.

A merge is W (womblex) or B (womblex-benchmark, paired). An approval tag means the merge edits `pyproject.toml` and needs human sign-off.

## Layout
The layout stage (L3) has shipped; [`layout.md`](layout.md) documents it and [`decisions.md`](decisions.md) records why it is shaped as it is. The remaining layout work (L3c-B, L1-B) is benchmark-side and now lives in [`plan-post-gt-baseline.md`](plan-post-gt-baseline.md).

## PyMuPDF
Benchmark-side merges (H-B, F1-B) live in [`plan-post-gt-baseline.md`](plan-post-gt-baseline.md). They do not gate anything here: a divergence they find comes back as a P8 fix.

- **P8… (W).** Fidelity fixes, measured by comparing the two backends over `fixtures/synthetic/` in this repository, repeated until the gates below hold. A shape the synthetic set lacks is added to its generator. Known inputs:
  - Block grouping, found by running `extract_text` over the synthetic PDFs under each backend: `_same_block` in `_text.py` merges any lines within 0.7 line heights, so a heading and every paragraph after it become one block. `koala-habitat-audit` gives 24 paragraphs under fitz and 6 paragraphs, 10 headings and 18 footers under pdfium, and every native PDF's digest differs. On the synthetic PDFs a size or bold change between lines always starts a fitz block and a forward line pitch of 1.36 font sizes or less never does (1.56 or more always does). All of it comes from reportlab at one leading, so the generator gains pages at other leadings and paragraph spacings before the rule is fixed.
  - Multi-column reading order: Phase 0's divergence tail (see `decisions.md`). Two synthetic two-column pages, one with the columns in content order and one with their lines interleaved, give `plain_text` identical to fitz's, so there is no reproducer here; the ASX-page investigation is in the baseline plan.
- **F1 (W).** Lands when the gates below hold; F1-B in the baseline plan runs after it.
  - Flip `open_document`'s default to the permissive backend.
  - Re-pin `tests/test_default_digest.py` deliberately, and move its version guard from `fitz.VersionBind` to the pypdfium2 version.
- **F2 (W, approval).**
  - Delete `_fitz.py` and the backend argument, and drop `pymupdf`, its mypy override and the `pymupdf_layout` warning filter.
  - Simplify the CI fitz-notice workaround and the Dockerfile comments.
  - Rewrite `CLAUDE.md`: the PyMuPDF-import and dehyphenation pitfalls, and the rule that "everything fitz can open routes through `extract_text`".
  - Update `dataflow`, `heuristics_disambiguation` and the README.
  - Move the declared behaviour changes below into `decisions.md`, and delete this document.

**Out of scope, kept as-is for parity:** the unrotated-text versus rotated-`page.rect` mismatch outside `spreadsheet_print`.

## Open question
- **Record the PDF library version in the run stamp?** `content_digest` depends on the PDF library's version, but no footer records it. Options: add it to the stamp; raise it as a separate requirement; or leave it, since the womblex version and `uv.lock` already pin the library.

## Conventions this plan holds to
- **Merge size.** At most 500 changed lines per merge, green on ruff, mypy and pytest. `uv.lock` doesn't count. Any merge that grows splits again before review.
- **File size.** At most 750 lines per file. `ingest/pdf/` is split by concern.
- **Thin adapters.** Library-native behaviour is preferred (pdfplumber `TableFinder`). Womblex code covers only coordinates, typing and the fitz-compatible shapes its callers need.
- **No toggles.** Backend selection is a private `open_document` argument used by H-B, deleted at F2. The `Page` / `Document` protocols stay afterwards because PDF and image pages remain two implementations.
- **Docs move with the code.** Every merge that adds, moves or retires a module updates `architecture.md`, `project-structure.md` and the `CLAUDE.md` module table in the same PR, adds a CHANGELOG entry and ticks itself off here.
- **Verbatim text.** Dehyphenation and segmentation are extractor behaviour, so they live in the backend, never as a post-pass.
- **Unusual input warns and continues.** A format the new backend cannot open (fitz also opened XPS, EPUB, MOBI, CBZ and SVG) becomes a per-document error status with the document ID logged. It never aborts the batch.
- **Benchmark boundary.** All scoring against womblex-collection ground truth, and every `docs/accuracy/` report, is produced in womblex-benchmark.

## Declared behaviour changes (at F1)
- **Fewer formats.** Calling `extract_text` directly can no longer open XPS, EPUB, MOBI, CBZ or SVG, nor an image format MuPDF decodes and Pillow does not. PDF and the common image formats remain, and WebP and AVIF are new. None of the dropped formats is reachable through the CLI or API today.
- **New content digests.** `content_digest` changes at F1, and `tests/test_default_digest.py` is re-pinned in that merge. Text stays verbatim, but it comes from a different producer.
- **Small text differences.** Paragraph segmentation, dehyphenation and bold-based heading detection may shift slightly.
- **Image counts.** `image_count` may count drawn images only, which moves the sub-page OCR gate.
- **Rendering.** Anti-aliasing differs, so OCR output differs slightly.
- **Speed.** The table pass may be slower; the gate bounds it.

## Gates
Migration gates, not quality scores. They retire with this plan.

| Merge | Gate |
|---|---|
| Every merge | `uv run ruff check src/ tests/`, `uv run mypy src/` and `uv run python -m pytest tests/ -v` pass; `uv lock --check` passes on approval merges; touched files are under 750 lines (`wc -l`); `git diff --stat $(git merge-base HEAD origin/main)..HEAD` is within the cap |

F1 flips the default only when all of these hold, each measured between the two backends over `fixtures/synthetic/` (no ground truth):

| Measure | Required |
|---|---|
| New open failures | Zero |
| Doc type | ≥ 98% agreement |
| Per-page plan operation | ≥ 99% agreement |
| `has_text_layer` | 100% agreement |
| Native-text CER between backends | Median ≤ 0.01, p95 ≤ 0.03 |
| Table count | Equal on ≥ 95% of table pages |
| Vector-redaction counts | Identical |
| AcroForm field counts | Equal |
| Native extraction | ≤ 1.5× the fitz wall time |

The gates scored against womblex-collection ground truth (transcript CER, table benchmark, false-table cohort, FUNSD fields, the `CHUNKING`, `READING_ORDER`, `PII_CLEANING` and `REDACTION_HANDLING` reports) are F1-B checks in the baseline plan, run after F1.
