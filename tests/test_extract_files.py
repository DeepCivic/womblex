"""Tests for the path-based extractors — SpreadsheetExtractor and DocxExtractor."""

from pathlib import Path

import pytest

from womblex.ingest.spreadsheet import SpreadsheetExtractor
from womblex.ingest.strategies import DocxExtractor

# ---------------------------------------------------------------------------
# SpreadsheetExtractor (uses real fixture spreadsheets)
# ---------------------------------------------------------------------------


class TestSpreadsheetExtractor:
    """One ExtractionResult per workbook with cell-grained elements.

    The previous one-result-per-row shape was removed; a workbook now
    yields a single result whose ``elements`` are sheet_meta + sheet_cell
    in workbook order.
    """

    def test_extracts_real_csv(self, spreadsheet_dir: Path) -> None:
        csv_path = spreadsheet_dir / "platypus-sightings-register.csv"

        ext = SpreadsheetExtractor()
        result = ext.extract_path(csv_path)

        assert result.method == "spreadsheet"
        assert result.error is None
        assert result.metadata is not None
        kinds = [e.kind for e in result.elements]
        assert kinds.count("sheet_meta") >= 1
        assert kinds.count("sheet_cell") >= 1

    def test_extracts_real_xlsx(self, spreadsheet_dir: Path) -> None:
        xlsx_path = spreadsheet_dir / "echidna-population-statistics.xlsx"

        ext = SpreadsheetExtractor()
        result = ext.extract_path(xlsx_path)

        assert result.method == "spreadsheet"
        assert result.metadata is not None
        # Multi-sheet workbook: one sheet_meta per sheet.
        n_sheet_meta = sum(1 for e in result.elements if e.kind == "sheet_meta")
        assert n_sheet_meta >= 1

    def test_handles_missing_file(self, tmp_path: Path) -> None:
        ext = SpreadsheetExtractor()
        result = ext.extract_path(tmp_path / "missing.csv")
        assert result.error is not None
        assert result.elements == []

    def test_xlsx_title_preamble_split_from_header(self, tmp_path: Path) -> None:
        """Export-product shape (e.g. AusTender): title + blank row above the header.

        The real header row must be the one marked is_header — pandas'
        fabricated ``Unnamed: N`` names must never appear as cell values.
        """
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Contract Notice Export"
        ws.append(["Contract Notice Export"])
        ws.append([])
        headers = ["Agency", "CN ID", "Publish Date", "Value", "Description", "Supplier Name"]
        ws.append(headers)
        ws.append(["Example Agency", "CN0000001", "2026-01-05 09:00:00", "12345.67",
                   "Office fitout", "Example Supplier Pty Ltd"])
        ws.append(["Example Agency", "CN0000002", "2026-01-06 10:00:00", "99000",
                   "ICT services", "Another Supplier Pty Ltd"])
        path = tmp_path / "AusTenderContractNoticeExport_20260101_000000.xlsx"
        wb.save(path)

        result = SpreadsheetExtractor().extract_path(path)
        assert result.error is None

        header_cells = [
            e.value for e in result.elements
            if e.kind == "sheet_cell" and (e.meta or {}).get("is_header")
        ]
        assert header_cells == headers
        assert not any("Unnamed" in (e.value or "") for e in result.elements)

        sheet_meta = next(e for e in result.elements if e.kind == "sheet_meta")
        assert sheet_meta.meta is not None
        assert sheet_meta.meta["preamble"] == "Contract Notice Export"

        data_cells = [
            e.value for e in result.elements
            if e.kind == "sheet_cell" and e.row == 1
        ]
        assert data_cells[0] == "Example Agency"
        assert "Example Supplier Pty Ltd" in data_cells

    def test_xlsx_multicell_metadata_preamble(self, tmp_path: Path) -> None:
        """Metadata blocks above the header span 2 cells but are not the header.

        A naive "first row with >=2 non-empty cells" rule would pick the
        ``Agency: | ...`` row; the width-ratio rule keeps scanning until a
        row matching the table's real width.
        """
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Quarterly Statistics Report"])
        ws.append(["Agency:", "Example Commission"])
        ws.append(["Generated:", "2026-06-11"])
        ws.append([])
        headers = ["Region", "Quarter", "Applications", "Approvals", "Refusals", "Pending"]
        ws.append(headers)
        ws.append(["NSW", "Q3", "120", "100", "15", "5"])
        path = tmp_path / "stats.xlsx"
        wb.save(path)

        result = SpreadsheetExtractor().extract_path(path)
        assert result.error is None

        header_cells = [
            e.value for e in result.elements
            if e.kind == "sheet_cell" and (e.meta or {}).get("is_header")
        ]
        assert header_cells == headers
        sheet_meta = next(e for e in result.elements if e.kind == "sheet_meta")
        assert sheet_meta.meta is not None
        preamble = sheet_meta.meta["preamble"]
        assert "Quarterly Statistics Report" in preamble
        assert "Example Commission" in preamble
        assert "2026-06-11" in preamble

    def test_ragged_csv_with_title_row(self, tmp_path: Path) -> None:
        """A one-field title row above a wide CSV header must not fail the read."""
        path = tmp_path / "export.csv"
        path.write_text(
            "Contract Notice Export\n"
            "\n"
            "Agency,CN ID,Value,Description\n"
            "Example Agency,CN0000001,12345.67,Office fitout\n",
            encoding="utf-8",
        )

        result = SpreadsheetExtractor().extract_path(path)
        assert result.error is None

        header_cells = [
            e.value for e in result.elements
            if e.kind == "sheet_cell" and (e.meta or {}).get("is_header")
        ]
        assert header_cells == ["Agency", "CN ID", "Value", "Description"]
        sheet_meta = next(e for e in result.elements if e.kind == "sheet_meta")
        assert (sheet_meta.meta or {}).get("preamble") == "Contract Notice Export"

    def test_windows_1252_csv_keeps_its_accents(self, tmp_path: Path) -> None:
        path = tmp_path / "cp1252.csv"
        path.write_bytes("Name,Place\nO’Brien,Café\n".encode("cp1252"))

        result = SpreadsheetExtractor().extract_path(path)
        assert result.error is None
        values = [e.value for e in result.elements if e.kind == "sheet_cell"]
        assert "O’Brien" in values and "Café" in values

    def test_two_column_key_value_sheet_keeps_row0_header(self, tmp_path: Path) -> None:
        """Uniformly narrow sheets are not mistaken for preamble + header."""
        path = tmp_path / "kv.csv"
        path.write_text(
            "Field,Value\nLicence number,L1234\nStatus,Active\n",
            encoding="utf-8",
        )

        result = SpreadsheetExtractor().extract_path(path)
        assert result.error is None

        header_cells = [
            e.value for e in result.elements
            if e.kind == "sheet_cell" and (e.meta or {}).get("is_header")
        ]
        assert header_cells == ["Field", "Value"]
        sheet_meta = next(e for e in result.elements if e.kind == "sheet_meta")
        assert "preamble" not in (sheet_meta.meta or {})

    def test_title_wider_than_table_not_header(self, tmp_path: Path) -> None:
        """A title row spanning more cells than the table itself stays preamble.

        Run-scoring catches what a pure width rule cannot: the title's run
        is broken by the blank row below it, while the real header is
        followed by its data body.
        """
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Licence", "Register", "Extract"])  # 3-cell title
        ws.append([])
        ws.append(["Field", "Value"])  # real 2-col header
        ws.append(["Licence number", "L1234"])
        ws.append(["Status", "Active"])
        path = tmp_path / "register.xlsx"
        wb.save(path)

        result = SpreadsheetExtractor().extract_path(path)
        assert result.error is None

        header_cells = [
            e.value for e in result.elements
            if e.kind == "sheet_cell" and (e.meta or {}).get("is_header")
        ]
        assert header_cells == ["Field", "Value"]
        sheet_meta = next(e for e in result.elements if e.kind == "sheet_meta")
        assert sheet_meta.meta is not None
        assert "Licence" in sheet_meta.meta["preamble"]

    def test_sub_header_row_does_not_shift_header(self, tmp_path: Path) -> None:
        """A single-cell section row directly under the header is neutral."""
        path = tmp_path / "sectioned.csv"
        path.write_text(
            "Name,Code,Status\n"
            "Section A,,\n"
            "Alpha,A1,Active\n"
            "Beta,B2,Active\n",
            encoding="utf-8",
        )

        result = SpreadsheetExtractor().extract_path(path)
        assert result.error is None

        header_cells = [
            e.value for e in result.elements
            if e.kind == "sheet_cell" and (e.meta or {}).get("is_header")
        ]
        assert header_cells == ["Name", "Code", "Status"]

    def test_xlsx_long_keyvalue_block_above_wide_header(self, tmp_path: Path) -> None:
        """GrantConnect shape: a long 2-column key/value block above a wide header.

        The *Grant Award Published* export opens with a ~20-row two-column
        "Criteria Summary" block before the real 32-column header. A
        run-length-only rule picks the key/value block (its consistent
        2-wide run out-scores a later header); scoring breadth-times-depth
        and widening the scan window pick the real, much wider header.
        """
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Grant Award Published"
        ws.append(["Criteria Summary"])
        # 20-row two-column key/value block.
        criteria = [
            ("Portfolio/Agency", "All Active Agencies"),
            ("Date Range", "01-Jan-2026 to 31-Mar-2026"),
            ("Grant Program", "All Programs"),
            ("Grant Activity", "All Activities"),
            ("Category", "All Categories"),
        ]
        for _ in range(4):  # repeat to reach ~20 rows of metadata
            for key, value in criteria:
                ws.append([key, value])
        ws.append([])  # blank separator before the real header

        headers = [f"Col{i}" for i in range(32)]
        headers[0:3] = ["Agency", "GA ID", "Internal Reference ID"]
        ws.append(headers)
        for r in range(25):  # a deep data body so the header out-scores the block
            ws.append([f"Agency {r}", f"GA{r:06d}", f"REF{r}"] + [""] * 29)
        path = tmp_path / "GrantConnect-Grant-Award-Published_20260401.xlsx"
        wb.save(path)

        result = SpreadsheetExtractor().extract_path(path)
        assert result.error is None

        header_cells = [
            e.value for e in result.elements
            if e.kind == "sheet_cell" and (e.meta or {}).get("is_header")
        ]
        assert header_cells[:3] == ["Agency", "GA ID", "Internal Reference ID"]
        assert "Portfolio/Agency" not in header_cells

        sheet_meta = next(e for e in result.elements if e.kind == "sheet_meta")
        assert sheet_meta.meta is not None
        preamble = sheet_meta.meta["preamble"]
        assert "Criteria Summary" in preamble
        assert "All Active Agencies" in preamble

    def test_csv_without_preamble_unchanged(self, tmp_path: Path) -> None:
        """A plain header-first CSV keeps its row-0 header behaviour."""
        path = tmp_path / "plain.csv"
        path.write_text("Name,Code\nAlpha,A1\nBeta,B2\n", encoding="utf-8")

        result = SpreadsheetExtractor().extract_path(path)
        assert result.error is None

        header_cells = [
            e.value for e in result.elements
            if e.kind == "sheet_cell" and (e.meta or {}).get("is_header")
        ]
        assert header_cells == ["Name", "Code"]
        sheet_meta = next(e for e in result.elements if e.kind == "sheet_meta")
        assert "preamble" not in (sheet_meta.meta or {})
        rows = {
            (e.row, e.col): e.value
            for e in result.elements if e.kind == "sheet_cell"
        }
        assert rows[(1, 0)] == "Alpha"
        assert rows[(2, 1)] == "B2"


class TestSheetCellMergeRange:
    """Requirement 6 (a2): every covered cell is emitted; the address rides on the anchor.

    a1 annotated ``merge_range`` onto the merge's top-left cell but left the
    covered cells absent. a2 emits those covered cells — blank included, since
    Excel leaves a merged region's covered cells empty — so the merge's
    footprint survives downstream and a blank-anchored merge has an emitted
    anchor to carry it. Only the anchor carries the address; covered blanks
    carry an empty value and no ``merge_range``.
    """

    def test_merged_header_annotated_on_top_left(self, tmp_path: Path) -> None:
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Report"
        ws.append(["Region", "Quarter", "Value"])
        # A merged 3-wide banner sitting in the data body (raw row 1 ->
        # emitted row 1), so it lands on the emitted grid rather than the
        # dropped preamble.
        ws.append(["Merged Banner", None, None])
        ws.merge_cells("A2:C2")
        ws.append(["NSW", "Q3", "1000"])
        path = tmp_path / "merged.xlsx"
        wb.save(path)

        result = SpreadsheetExtractor().extract_path(path)
        assert result.error is None

        by_pos = {
            (e.row, e.col): e
            for e in result.elements if e.kind == "sheet_cell"
        }
        # Top-left of the merge (emitted row 1, col 0) carries the address.
        assert by_pos[(1, 0)].merge_range == "A2:C2"
        assert by_pos[(1, 0)].value == "Merged Banner"
        # The covered cells are now emitted (a2) — blank, part of the merge's
        # footprint — but the address rides only on the anchor.
        assert (1, 1) in by_pos
        assert (1, 2) in by_pos
        assert by_pos[(1, 1)].value == ""
        assert by_pos[(1, 2)].value == ""
        assert by_pos[(1, 1)].merge_range is None
        assert by_pos[(1, 2)].merge_range is None
        # Non-merged cells carry no merge_range.
        assert by_pos[(0, 0)].merge_range is None
        assert by_pos[(2, 2)].merge_range is None

    def test_no_merges_leaves_merge_range_unset(self, tmp_path: Path) -> None:
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Name", "Code"])
        ws.append(["Alpha", "A1"])
        path = tmp_path / "plain.xlsx"
        wb.save(path)

        result = SpreadsheetExtractor().extract_path(path)
        assert result.error is None
        assert all(
            e.merge_range is None
            for e in result.elements if e.kind == "sheet_cell"
        )

    def test_merge_below_preamble_aligns_to_grid(self, tmp_path: Path) -> None:
        """row_offset must map the raw merge coordinate onto the emitted grid."""
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Export"
        ws.append(["Contract Notice Export"])  # title preamble (raw row 0)
        ws.append([])                            # blank (raw row 1)
        ws.append(["Agency", "Detail", "Value"])  # real header (raw row 2)
        # Merge a data cell pair two rows into the data body.
        ws.append(["Example Agency", "Combined detail", "1000"])
        ws.merge_cells("B4:C4")  # raw row 3 -> emitted row 1
        path = tmp_path / "export.xlsx"
        wb.save(path)

        result = SpreadsheetExtractor().extract_path(path)
        assert result.error is None

        by_pos = {
            (e.row, e.col): e
            for e in result.elements if e.kind == "sheet_cell"
        }
        # Header row is emitted as row 0; the merged data cell is emitted row 1.
        assert by_pos[(0, 0)].value == "Agency"
        assert by_pos[(1, 1)].merge_range == "B4:C4"

    def test_blank_anchored_merge_emits_anchor_to_carry_address(
        self, tmp_path: Path,
    ) -> None:
        """a2's unblock: a merge whose top-left is blank still gets annotated.

        a1 could not carry such a merge — nothing was emitted at the anchor.
        a2 emits the (blank) anchor, so the ``merge_range`` has a cell to ride.
        """
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Report"
        ws.append(["Region", "Quarter", "Value"])
        # A data row whose first two cells are merged and blank (the label
        # was left empty in the source), value in the third.
        ws.append([None, None, "1000"])
        ws.merge_cells("A2:B2")  # blank anchor at raw (1, 0) -> emitted row 1
        path = tmp_path / "blank_anchor.xlsx"
        wb.save(path)

        result = SpreadsheetExtractor().extract_path(path)
        assert result.error is None

        by_pos = {
            (e.row, e.col): e
            for e in result.elements if e.kind == "sheet_cell"
        }
        # The blank anchor is emitted and carries the address a1 could not place.
        assert by_pos[(1, 0)].value == ""
        assert by_pos[(1, 0)].merge_range == "A2:B2"
        # Its covered blank neighbour is emitted too, without the address.
        assert by_pos[(1, 1)].value == ""
        assert by_pos[(1, 1)].merge_range is None
        # The real value beside the merge is unaffected.
        assert by_pos[(1, 2)].value == "1000"

    def test_covered_blanks_do_not_leak_beyond_the_merge(
        self, tmp_path: Path,
    ) -> None:
        """Only merge-covered blanks are emitted; ordinary blanks stay absent."""
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Report"
        ws.append(["A", "B", "C"])
        ws.append(["Merged", None, "x"])  # A2:B2 merged; C2 populated
        ws.merge_cells("A2:B2")
        ws.append(["y", None, "z"])       # B3 is an ordinary (non-merged) blank
        path = tmp_path / "mixed.xlsx"
        wb.save(path)

        result = SpreadsheetExtractor().extract_path(path)
        assert result.error is None

        by_pos = {
            (e.row, e.col): e
            for e in result.elements if e.kind == "sheet_cell"
        }
        # Covered blank (row 1, col 1) is emitted.
        assert (1, 1) in by_pos
        assert by_pos[(1, 1)].value == ""
        # Ordinary blank (row 2, col 1) is not part of any merge — stays absent.
        assert (2, 1) not in by_pos


# ---------------------------------------------------------------------------
# DocxExtractor
# ---------------------------------------------------------------------------


class TestDocxExtractor:
    def test_handles_missing_docx_library(self, tmp_path: Path) -> None:
        ext = DocxExtractor()
        result = ext.extract_path(tmp_path / "test.docx")
        # python-docx not installed in test env
        assert result.error is not None
        assert "docx" in result.error.lower()

    def test_multi_row_header_band_from_tbl_header(self, tmp_path: Path) -> None:
        """A two-row header declared via ``w:tblHeader`` populates ``header_rows``.

        Word marks each row of a hierarchical header band with the "repeat as
        header row" property; both rows must land in ``header_rows`` so the
        unit-bearing second row (``$'000`` / ``%``) is not read as data.
        """
        docx = pytest.importorskip("docx")
        from docx.oxml.ns import qn

        d = docx.Document()
        d.add_paragraph("Intro prose.")
        tbl = d.add_table(rows=4, cols=2)
        for r in range(2):  # mark the first two rows as a header band
            trpr = tbl.rows[r]._tr.get_or_add_trPr()
            trpr.append(trpr.makeelement(qn("w:tblHeader"), {}))
        values = [["Budget", "Budget"], ["$'000", "%"], ["100", "5"], ["200", "10"]]
        for r in range(4):
            for c in range(2):
                tbl.rows[r].cells[c].text = values[r][c]
        path = tmp_path / "multi_header.docx"
        d.save(path)

        result = DocxExtractor().extract_path(path)
        assert result.error is None
        table = next(e for e in result.elements if e.kind == "table")
        assert table.header_rows == [0, 1]

    def test_only_leading_run_of_header_rows_counts(self, tmp_path: Path) -> None:
        """A ``tblHeader`` deeper in the body does not fold data rows into the band.

        Word occasionally sets the property on a mid-table repeat row; taking
        only the leading contiguous run keeps the intervening data rows out of
        ``header_rows``.
        """
        docx = pytest.importorskip("docx")
        from docx.oxml.ns import qn

        d = docx.Document()
        tbl = d.add_table(rows=4, cols=2)
        for r in (0, 2):  # row 0 is the header; row 2 is a stray repeat marker
            trpr = tbl.rows[r]._tr.get_or_add_trPr()
            trpr.append(trpr.makeelement(qn("w:tblHeader"), {}))
        for r in range(4):
            for c in range(2):
                tbl.rows[r].cells[c].text = f"r{r}c{c}"
        path = tmp_path / "stray_header.docx"
        d.save(path)

        result = DocxExtractor().extract_path(path)
        assert result.error is None
        table = next(e for e in result.elements if e.kind == "table")
        assert table.header_rows == [0]

    def test_no_tbl_header_falls_back_to_row_zero(self, tmp_path: Path) -> None:
        """A plain table with no declared header keeps the single-row-header default."""
        docx = pytest.importorskip("docx")

        d = docx.Document()
        tbl = d.add_table(rows=2, cols=2)
        tbl.rows[0].cells[0].text = "H1"
        tbl.rows[0].cells[1].text = "H2"
        tbl.rows[1].cells[0].text = "1"
        tbl.rows[1].cells[1].text = "2"
        path = tmp_path / "plain.docx"
        d.save(path)

        result = DocxExtractor().extract_path(path)
        assert result.error is None
        table = next(e for e in result.elements if e.kind == "table")
        assert table.header_rows == [0]
