"""B3 + B1.2 — table reconstruction measured against rendered-clean GT.

Round-1 benchmark for #17 (docs/table-cell-reconstruction-plan.md): feed a
*known-correct* table rect straight to ``reconstruct_table`` and measure
the grid, without the layout detector in the loop (B1.2). The primary
fixtures are rendered from the two vendored spreadsheet sources — render →
rasterise → OCR (paddleocr, per A0) → reconstruct → compare against the
rendered GT (B3). ``dense_text_548`` (hard scan shape) tracks without a
gate; refusal is a correct round-1 outcome there.

The GT is exactly what was drawn; the scorer declares its normalisation
(NFKC + dash folding + whitespace collapse) — GT stays verbatim.
Hard gates are B5's; this module records outcomes so B2 can calibrate the
provisional precision gates in ``ocr_tables``.
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

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures"
SPREADSHEETS_DIR = FIXTURES_DIR / "womblex-collection" / "_spreadsheets"
DOCLAYNET_DIR = FIXTURES_DIR / "doclaynet"

CSV_SOURCE = SPREADSHEETS_DIR / "Approved-providers-au-export_20260204.csv"
XLSX_SOURCE = SPREADSHEETS_DIR / "mso-statistics-sept-qtr-2025.xlsx"

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

_results: list[dict] = []

_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―"})


def _norm(s: str) -> str:
    """The scorer's declared normalisation: NFKC + dash fold + ws collapse."""
    return " ".join(unicodedata.normalize("NFKC", s).translate(_DASHES).split())


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
        # Round-1 sanity (B5 formalises the gates): a clean rendered grid is
        # exactly the target shape — refusal here is a reconstructor defect.
        assert entry["outcome"] == "reconstructed"
        assert entry["got_cols"] == entry["gt_cols"]


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
        # Tracking only: either refusal or a partial grid is a valid round-1
        # outcome for this shape (7-line stacked header, hierarchical rows).
        assert True
