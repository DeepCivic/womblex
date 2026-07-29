"""Table reconstruction measured against rendered-clean ground truth.

Benchmark for OCR table reconstruction (#17; see docs/evaluation.md §2b and
docs/decisions.md “Table-cell reconstruction on OCR pages”): feed a
*known-correct* table rect straight to ``reconstruct_table`` and measure
the grid, without the layout detector in the loop. The primary
fixtures are rendered from the two vendored spreadsheet sources — render →
rasterise → OCR (paddleocr) → reconstruct → compare against the
rendered GT.

The precision half of the picture:

- **The metric set.** A thin alignment projection turns a reconstructed
  ``TableData`` into a DataFrame (header row → column names) so the
  ``utils.tabular_metrics`` scorers (``structural_fidelity``,
  ``data_integrity``, ``key_column_preservation``) apply directly — the
  same metrics the spreadsheet ingest uses. Cell-content agreement is
  scored on ``(row, col, text)`` after alignment, which catches a
  column-shift a raw cell count misses.
- **The false-table cohort** (the precision guardrail): fixtures with *no*
  GT table — the non-table DocLayNet pages and the FUNSD forms — fed to the
  reconstructor whole. Any table emitted is a false positive. This is what
  makes "precision over coverage" falsifiable, and it calibrated
  ``MIN_ROW_FILL_RATIO`` in ``ocr_tables``.
- **The GT acceptance checker** for the DocLayNet table GT (see the
  [table ground-truth authoring spec](../docs/evaluation.md) in
  docs/evaluation.md §2b).

``dense_text_548`` (hard scan shape) tracks without a gate. It
*refuses* — the density gate the false-table cohort calibrated also rejects
its stacked-header hierarchical shape — which is the precision-first
outcome intended.

The GT is exactly what was drawn; the scorer declares its normalisation
(NFKC + dash folding + whitespace collapse) — GT stays verbatim.

The recorded outcomes are turned into **build-failing gates**:

- **Rendered-clean cohort (gated):** each fixture must reconstruct, the grid
  must be structurally exact (row **and** column counts match the drawn GT),
  and cell agreement must clear ``MIN_CELL_MATCH`` — a floor set below the
  measured minimum (0.844 on the fuel sheets) with headroom for OCR
  non-determinism, so it catches a binning/recognition catastrophe, not
  glyph-level noise. ``structural_fidelity`` (which also compares the exact
  column-*name* set) is **reported, not gated**: a single-glyph header
  misread — the same OCR noise the cell floor tolerates in body cells —
  would flip it and flake the build, so the load-bearing structural gate is
  the row/column *counts*, which are what a mis-binned grid actually breaks.
- **False-table cohort (gated):** every probe must refuse — the aggregate
  false-positive count is **zero**, the precision guardrail, enforced
  fixture by fixture (each probe asserts refusal, so a single FP anywhere
  fails the build).
- **``dense_text_548`` (tracking, never gated):** reported without a gate
  until the scan round sets one; the only invariant is that it never emits a
  full false 39x11 grid.

The engine is pinned to paddleocr by construction — ``get_paddle_reader``
constructs the reader directly, with no config-engine indirection, so a
config default flipping to an LLM engine cannot turn these gates into
no-ops.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np
import pytest

pytest.importorskip("rapidocr_onnxruntime")

from womblex.ingest.extract import TableData
from womblex.ingest.ocr_tables import reconstruct_table
from womblex.ingest.paddle_ocr import get_paddle_reader
from womblex.utils.tabular_metrics import data_integrity, structural_fidelity

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures"
SPREADSHEETS_DIR = FIXTURES_DIR / "womblex-collection" / "_spreadsheets"
DOCLAYNET_DIR = FIXTURES_DIR / "doclaynet"

CSV_SOURCE = SPREADSHEETS_DIR / "Approved-providers-au-export_20260204.csv"
XLSX_SOURCE = SPREADSHEETS_DIR / "mso-statistics-sept-qtr-2025.xlsx"

FUNSD_IMAGES_DIR = FIXTURES_DIR / "funsd" / "images"

# The false-table cohort: pages with no GT table. Feeding the whole page
# rect to the reconstructor must yield a refusal — any table is a false
# positive (the precision guardrail). The DocLayNet trio are the
# non-table pages from the layout GT (table_0 despite its name holds no
# Table-labelled span); the FUNSD forms are dense label/value
# pairs, the likeliest false positive.
FALSE_TABLE_DOCLAYNET = ["diverse_layout_49", "formula_29", "table_0"]
FALSE_TABLE_FUNSD = [
    "85540866", "82200067_0069", "87594142_87594144", "87528321", "87528380",
]

# Discovered at import so the GT-acceptance checker can parametrize per GT file.
_GT_CSV_PATHS = sorted(DOCLAYNET_DIR.glob("*_table.csv"))

RENDER_DPI = 200
FONT = "helv"
FONT_SIZE = 9.0
ROW_PITCH = 16.0
MARGIN = 36.0
COLUMN_GAP = 24.0

# Columns rendered per source — chosen for bounded width so a page stays a
# realistic print size; the rendered selection *is* the GT.
CSV_COLUMNS = ["Provider Approval Number", "Address", "Suburb", "State", "Postcode"]
CSV_ROWS_PER_PAGE = 30
CSV_PAGES = 3
# Positional pick from the 8 fuel-sheet columns (the last column's header
# differs by a footnote marker between sheets).
XLSX_COLUMN_IDX = [0, 1, 2, 3, 7]
XLSX_SHEETS = ["Diesel", "Gasoline", "Kerosene"]

# B5 gate floor for the rendered-clean cohort. Measured baseline
# (2026-07-28): cell agreement ran 0.844–0.987 across the six fixtures, the
# minimum on a fuel sheet (9 pt glyph confusions: 6<->9, 0->o, a lost space).
# 0.75 sits comfortably below that with headroom for OCR non-determinism, so
# the gate fails on a binning/recognition collapse rather than flaking on
# glyph noise. The exact row/column *counts* are the load-bearing structural
# gate (what a mis-binned grid breaks); this floor guards content.
# ``structural_fidelity`` (exact column-name set) is reported, not gated —
# see the note on the gate asserts below.
MIN_CELL_MATCH = 0.75

# B4: publish into the shared extraction-report accumulator so the
# ``write_report`` session finaliser in test_fixture_accuracy renders these
# rows into EXTRACTION.md's Table Reconstruction section. Aliasing the list
# (not copying) keeps a single source of truth — every append here lands in
# ``_results["tables"]`` there. Falls back to a private list when that module
# can't import (e.g. the benchmark OCR stack — cv2 — is absent, or this file
# is exercised in isolation), so the asserts below still run stand-alone.
try:
    from tests.test_fixture_accuracy import _results as _extraction_results

    _results: list[dict] = _extraction_results["tables"]
except ImportError:  # pragma: no cover - isolated-run fallback
    _results = []

_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―"})


def _norm(s: str) -> str:
    """The scorer's declared normalisation: NFKC + dash fold + ws collapse."""
    return " ".join(unicodedata.normalize("NFKC", s).translate(_DASHES).split())


def _table_to_frame(headers: list[str], rows: list[list[str]]):
    """B2 alignment projection: a reconstructed grid → a normalised DataFrame.

    Header row becomes the column names, body rows the data, everything
    passed through ``_norm`` so ``tabular_metrics`` compares like with like
    (the scorer declares its normalisation; the GT frame is built the same
    way). Duplicate/blank header names are uniquified positionally so the
    frame is well-formed — a spanning-header collision is a reconstruction
    signal the structural metric already reflects via the column count.
    """
    import pandas as pd

    cols: list[str] = []
    seen: dict[str, int] = {}
    for i, h in enumerate(headers):
        name = _norm(h) or f"col{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    data = [[_norm(c) for c in row[:len(cols)]] + [""] * (len(cols) - len(row))
            for row in rows]
    return pd.DataFrame(data, columns=cols)


def _fmt_cell(v: object) -> str:
    """Deterministic printed form for a source cell."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if hasattr(v, "strftime"):  # Timestamp / datetime
        return v.strftime("%b-%Y")
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


@dataclass
class RenderedTable:
    """One rendered page: the drawn GT plus the known table rect (points)."""

    name: str
    headers: list[str]
    rows: list[list[str]]
    rect_pt: tuple[float, float, float, float]
    page_index: int


def _text_len(s: str) -> float:
    return fitz.get_text_length(s, fontname=FONT, fontsize=FONT_SIZE)


def _render_table_page(
    doc: fitz.Document, name: str, headers: list[str], rows: list[list[str]],
) -> RenderedTable:
    """Draw one left-aligned grid on a new page sized to its content."""
    widths = [
        max(_text_len(headers[i]), *(_text_len(r[i]) for r in rows))
        for i in range(len(headers))
    ]
    page_w = 2 * MARGIN + sum(widths) + COLUMN_GAP * (len(headers) - 1)
    page_h = 2 * MARGIN + ROW_PITCH * (len(rows) + 2)
    page = doc.new_page(width=page_w, height=page_h)

    xs = [MARGIN]
    for w in widths[:-1]:
        xs.append(xs[-1] + w + COLUMN_GAP)

    y = MARGIN + ROW_PITCH
    for i, h in enumerate(headers):
        page.insert_text((xs[i], y), h, fontname=FONT, fontsize=FONT_SIZE)
    for row in rows:
        y += ROW_PITCH
        for i, cell in enumerate(row):
            if cell:
                page.insert_text((xs[i], y), cell, fontname=FONT, fontsize=FONT_SIZE)

    rect_pt = (
        MARGIN - 6.0,
        MARGIN + ROW_PITCH - FONT_SIZE - 4.0,
        xs[-1] + widths[-1] + 6.0,
        y + 4.0,
    )
    return RenderedTable(
        name=name, headers=headers, rows=rows, rect_pt=rect_pt,
        page_index=page.number,
    )


def _csv_tables() -> tuple[fitz.Document, list[RenderedTable]]:
    import pandas as pd

    df = pd.read_csv(CSV_SOURCE, dtype=str).fillna("")
    doc = fitz.open()
    tables = []
    for p in range(CSV_PAGES):
        chunk = df.iloc[p * CSV_ROWS_PER_PAGE:(p + 1) * CSV_ROWS_PER_PAGE]
        rows = [[_fmt_cell(r[c]) for c in CSV_COLUMNS] for _, r in chunk.iterrows()]
        tables.append(
            _render_table_page(doc, f"approved_providers_p{p + 1}", CSV_COLUMNS, rows)
        )
    return doc, tables


def _xlsx_tables() -> tuple[fitz.Document, list[RenderedTable]]:
    import pandas as pd

    doc = fitz.open()
    tables = []
    for sheet in XLSX_SHEETS:
        df = pd.read_excel(XLSX_SOURCE, sheet_name=sheet)
        headers = [str(df.columns[i]) for i in XLSX_COLUMN_IDX]
        # Row 0 is the sheet's repeated fuel-name subheader — not data.
        body = df.iloc[1:]
        rows = [[_fmt_cell(r.iloc[i]) for i in XLSX_COLUMN_IDX] for _, r in body.iterrows()]
        tables.append(_render_table_page(doc, f"mso_{sheet.lower()}", headers, rows))
    return doc, tables


def _reconstruct_rendered(
    reader, doc: fitz.Document, rt: RenderedTable,
) -> TableData | None:
    page = doc[rt.page_index]
    pix = page.get_pixmap(dpi=RENDER_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    result = reader.read_page(img[:, :, :3] if pix.n == 4 else img)
    scale = RENDER_DPI / 72.0
    rect_px = tuple(v * scale for v in rt.rect_pt)
    # conf=1.0: B1.2 conditions on a correct region — no detector in the loop.
    return reconstruct_table(
        result.regions, rect_px, RENDER_DPI, 1.0,
        pix_dims=(int(pix.width), int(pix.height)),
    )


def _reconstruct_whole_page(reader, img_path: Path) -> TableData | None:
    """Feed a real page image's whole rect to the reconstructor.

    Used by the false-table cohort: there is no GT table rect, so the
    reconstructor sees the entire page. DocLayNet renders US-letter pages,
    so the tolerance dpi follows from the image width (8.5 in).
    """
    import cv2

    img = cv2.imread(str(img_path))
    result = reader.read_page(img)
    dpi = round(img.shape[1] / 8.5)
    return reconstruct_table(
        result.regions,
        (0.0, 0.0, float(img.shape[1]), float(img.shape[0])),
        dpi, 1.0, pix_dims=(img.shape[1], img.shape[0]),
    )


def _score(rt: RenderedTable, table: TableData | None) -> dict:
    entry: dict = {
        "fixture": rt.name,
        "gt_rows": len(rt.rows), "gt_cols": len(rt.headers),
        "outcome": "reconstructed" if table is not None else "refused",
    }
    if table is None:
        return entry
    entry["got_rows"] = len(table.rows)
    entry["got_cols"] = len(table.headers)
    entry["header_hits"] = sum(
        _norm(g) == _norm(h)
        for g, h in zip(rt.headers, table.headers)
    )
    total = hits = 0
    mismatches: list[tuple[int, int, str, str]] = []
    for ri in range(min(len(rt.rows), len(table.rows))):
        for ci in range(min(len(rt.headers), len(table.headers))):
            total += 1
            gt, got = _norm(rt.rows[ri][ci]), _norm(table.rows[ri][ci])
            if gt == got:
                hits += 1
            elif len(mismatches) < 10:
                mismatches.append((ri, ci, gt, got))
    entry["cell_total"] = total
    entry["cell_hits"] = hits
    entry["cell_match"] = hits / total if total else 0.0
    entry["mismatches"] = mismatches

    # B2 metric set: run the tabular_metrics scorers over the alignment
    # projection so a reconstructed OCR grid is measured the same way the
    # spreadsheet ingest is. structural_fidelity checks rows+cols+names
    # (reported, not gated — see TestRenderedCleanTables); data_integrity is
    # exact cell match on the shared columns.
    gt_frame = _table_to_frame(rt.headers, rt.rows)
    got_frame = _table_to_frame(table.headers, table.rows)
    struct = structural_fidelity(gt_frame, got_frame)
    integ = data_integrity(gt_frame, got_frame)
    entry["structural_ok"] = struct.passed
    entry["data_integrity"] = integ.score
    return entry


def _log_entry(entry: dict) -> None:
    if entry["outcome"] == "refused":
        logger.info("table bench %s: REFUSED (gt %dx%d)",
                    entry["fixture"], entry["gt_rows"], entry["gt_cols"])
        return
    logger.info(
        "table bench %s: gt %dx%d got %dx%d headers %d/%d cells %d/%d (%.1f%%)",
        entry["fixture"], entry["gt_rows"], entry["gt_cols"],
        entry["got_rows"], entry["got_cols"],
        entry["header_hits"], entry["gt_cols"],
        entry["cell_hits"], entry["cell_total"], entry["cell_match"] * 100,
    )
    for ri, ci, gt, got in entry["mismatches"]:
        logger.info("  mismatch r%d c%d: gt=%r got=%r", ri, ci, gt, got)


@pytest.fixture(scope="module")
def reader():
    return get_paddle_reader(lang="eng")


def _rendered_specs() -> list[tuple[str, str]]:
    return (
        [("csv", f"approved_providers_p{p + 1}") for p in range(CSV_PAGES)]
        + [("xlsx", f"mso_{s.lower()}") for s in XLSX_SHEETS]
    )


@pytest.fixture(scope="module")
def rendered() -> dict[str, tuple[fitz.Document, RenderedTable]]:
    docs: dict[str, tuple[fitz.Document, RenderedTable]] = {}
    if CSV_SOURCE.exists():
        doc, tables = _csv_tables()
        for rt in tables:
            docs[rt.name] = (doc, rt)
    if XLSX_SOURCE.exists():
        doc, tables = _xlsx_tables()
        for rt in tables:
            docs[rt.name] = (doc, rt)
    return docs


@pytest.mark.benchmark
class TestRenderedCleanTables:
    """B3 primary fixtures — the flat contemporary shape round 1 ships for."""

    @pytest.mark.parametrize(
        "name", [n for _, n in _rendered_specs()],
    )
    def test_rendered_reconstruction(self, name: str, reader, rendered) -> None:
        if name not in rendered:
            pytest.skip(f"Source fixture missing for {name}")
        doc, rt = rendered[name]
        table = _reconstruct_rendered(reader, doc, rt)
        entry = _score(rt, table)
        _results.append(entry)
        _log_entry(entry)
        # B5 gates: a clean rendered grid is exactly the round-1 target
        # shape, so anything short of an exact grid with content above the
        # floor is a build-failing regression, not a tracked outcome. The
        # gate is the row/column *counts* (what a mis-binned grid breaks) plus
        # the cell-content floor — NOT ``structural_ok``, which also demands
        # an exact column-name-set match and so would flake on a single-glyph
        # header misread, the same OCR noise the cell floor deliberately
        # tolerates below. ``structural_ok`` stays a reported field (B4).
        assert entry["outcome"] == "reconstructed", (
            f"{name}: clean rendered grid refused — reconstructor regression"
        )
        assert entry["got_cols"] == entry["gt_cols"], (
            f"{name}: got {entry['got_cols']} cols, expected {entry['gt_cols']}"
        )
        assert entry["got_rows"] == entry["gt_rows"], (
            f"{name}: got {entry['got_rows']} rows, expected {entry['gt_rows']}"
        )
        assert entry["cell_match"] >= MIN_CELL_MATCH, (
            f"{name}: cell match {entry['cell_match']:.3f} below floor "
            f"{MIN_CELL_MATCH} — binning/recognition regression"
        )


@pytest.mark.benchmark
class TestDenseTextTracking:
    """B1.2 on the hard scan fixture — reported, never gated in round 1."""

    def test_gt_rect_reconstruction(self, reader) -> None:
        import json

        import cv2

        from tests.test_fixture_accuracy import (
            DOCLAYNET_LABELS,
            _aggregate_doclaynet_blocks,
        )

        img_path = DOCLAYNET_DIR / "dense_text_548.png"
        ann_path = DOCLAYNET_DIR / "dense_text_548.json"
        if not img_path.exists():
            pytest.skip(f"Fixture missing: {img_path}")

        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        labels = [DOCLAYNET_LABELS.get(lbl, "Unknown") for lbl in ann["labels"]]
        table_blocks = [
            box for box, lbl in _aggregate_doclaynet_blocks(ann["bboxes"], labels)
            if lbl == "Table"
        ]
        assert len(table_blocks) == 1  # post-B0: strays filtered

        img = cv2.imread(str(img_path))
        result = reader.read_page(img)
        # DocLayNet renders US-letter pages; the tolerance dpi follows from
        # the image width.
        dpi = round(img.shape[1] / 8.5)
        table = reconstruct_table(
            result.regions, tuple(table_blocks[0]), dpi, 1.0,
            pix_dims=(img.shape[1], img.shape[0]),
        )
        outcome = "refused" if table is None else (
            f"partial {len(table.rows)}x{len(table.headers)}"
        )
        _results.append({
            "fixture": "dense_text_548", "gt_rows": 39, "gt_cols": 11,
            "outcome": outcome, "tracking": True,
        })
        logger.info("table bench dense_text_548 (tracking): %s", outcome)
        # Post-B2 this shape refuses: the density gate (MIN_ROW_FILL_RATIO)
        # the false-table cohort calibrated also rejects the 7-line stacked
        # header + hierarchical rows here (it binned to a sparse ~0.45-fill
        # grid). Refusal is the precision-first outcome round 1 wanted.
        # Tracking, not gated: if a future recogniser change flips this to a
        # partial, that is still a valid round-1 outcome — assert the
        # invariant that matters (never a *full* false 39x11 grid).
        assert table is None or len(table.rows) < 39


@pytest.mark.benchmark
class TestFalseTableCohort:
    """B2 precision guardrail, B5 gate — pages with no GT table must refuse.

    Feeds the whole-page rect to the reconstructor. A returned table is a
    false positive; the cohort must be clean (zero FPs). Each probe asserts
    refusal, so the per-fixture asserts *are* B5's "false-table count == 0"
    gate — a single false positive anywhere fails the build, and holds under
    ``-k`` selection and ``pytest-xdist`` because no probe depends on another
    having run. This is what makes "precision over coverage" falsifiable, and
    it calibrated ``MIN_ROW_FILL_RATIO``.
    """

    @pytest.mark.parametrize("stem", FALSE_TABLE_DOCLAYNET)
    def test_doclaynet_non_table_refuses(self, stem: str, reader) -> None:
        img_path = DOCLAYNET_DIR / f"{stem}.png"
        if not img_path.exists():
            pytest.skip(f"Fixture missing: {img_path}")
        table = _reconstruct_whole_page(reader, img_path)
        self._record(f"doclaynet/{stem}", table)
        assert table is None, (
            f"false table on non-table page {stem}: "
            f"{None if table is None else (len(table.rows), len(table.headers))}"
        )

    @pytest.mark.parametrize("stem", FALSE_TABLE_FUNSD)
    def test_funsd_form_refuses(self, stem: str, reader) -> None:
        img_path = FUNSD_IMAGES_DIR / f"{stem}.png"
        if not img_path.exists():
            pytest.skip(f"Fixture missing: {img_path}")
        table = _reconstruct_whole_page(reader, img_path)
        self._record(f"funsd/{stem}", table)
        assert table is None, (
            f"false table on FUNSD form {stem}: "
            f"{None if table is None else (len(table.rows), len(table.headers))}"
        )

    @staticmethod
    def _record(name: str, table: TableData | None) -> None:
        outcome = "refused" if table is None else (
            f"FALSE TABLE {len(table.rows)}x{len(table.headers)}"
        )
        _results.append({
            "fixture": name, "false_table_probe": True,
            "outcome": outcome,
            "false_positive": table is not None,
        })
        logger.info("table bench false-table %s: %s", name, outcome)


@pytest.mark.benchmark
class TestGroundTruthAcceptance:
    """GT acceptance — a table GT that fails these is a bug in the GT.

    Runs over every ``*_table.csv`` beside a DocLayNet fixture: rectangular,
    unique non-empty column names, UTF-8 no BOM, no trailing whitespace in
    cells, ``n_header_rows`` consistent with the meta, and the cell count
    within a sane band of the fixture json's Table-labelled word count.
    """

    def _gt_csvs(self) -> list[Path]:
        return sorted(DOCLAYNET_DIR.glob("*_table.csv"))

    def test_at_least_one_gt_present(self) -> None:
        if not self._gt_csvs():
            pytest.skip("No DocLayNet table GT vendored")

    @pytest.mark.parametrize(
        "csv_path",
        _GT_CSV_PATHS or [pytest.param(None, marks=pytest.mark.skip(reason="no GT"))],
        ids=lambda p: p.stem if p is not None else "none",
    )
    def test_gt_conforms(self, csv_path: Path) -> None:
        import csv
        import json

        raw = csv_path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), "GT has a UTF-8 BOM"
        text = raw.decode("utf-8")  # asserts valid UTF-8

        rows = list(csv.reader(text.splitlines()))
        assert rows, "empty GT"
        header, *body = rows
        width = len(header)

        # Rectangular.
        for i, r in enumerate(rows):
            assert len(r) == width, f"row {i} has {len(r)} cells, expected {width}"
        # Unique, non-empty column names.
        assert all(h.strip() for h in header), "blank column name"
        assert len(set(header)) == width, "duplicate column names"
        # No trailing whitespace in any cell.
        for i, r in enumerate(rows):
            for c in r:
                assert c == c.strip() or not c.strip(), (
                    f"row {i} cell has trailing/leading whitespace: {c!r}"
                )

        # n_header_rows consistent with the meta and the file.
        meta_path = csv_path.with_suffix(".meta.json")
        assert meta_path.exists(), f"missing {meta_path.name}"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        n_header = meta["n_header_rows"]
        assert 1 <= n_header <= len(rows), f"n_header_rows {n_header} out of range"

        # Cell count within a sane band of the fixture's Table-labelled words.
        ann_path = DOCLAYNET_DIR / f"{csv_path.stem.replace('_table', '')}.json"
        if ann_path.exists():
            from tests.test_fixture_accuracy import DOCLAYNET_LABELS
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            labels = [DOCLAYNET_LABELS.get(lbl, "Unknown") for lbl in ann["labels"]]
            table_words = sum(
                len(str(w).split())
                for w, lbl in zip(ann.get("words", []), labels)
                if lbl == "Table"
            )
            gt_cells = sum(1 for r in body for c in r if c.strip())
            if table_words:
                # Loose band: transcription splits/merges words vs the
                # annotation, so allow a wide 0.3x-3x window — this catches a
                # GT off by an order of magnitude, not fine differences.
                assert 0.3 * table_words <= gt_cells <= 3.0 * table_words, (
                    f"GT cell count {gt_cells} implausible vs "
                    f"{table_words} Table-labelled words"
                )
