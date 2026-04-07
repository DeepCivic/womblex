"""Spreadsheet extraction — one ExtractionResult per logical row or sheet.

For data and glossary sheets each non-empty row produces a separate ExtractionResult.
For narrative and key-value sheets the whole sheet becomes one ExtractionResult.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from womblex.ingest.detect import SheetInfo

if TYPE_CHECKING:
    import pandas as pd
    from womblex.ingest.detect import DocumentProfile
from womblex.ingest.extract import (
    ExtractionMetadata,
    ExtractionResult,
    PageResult,
    Position,
    TableData,
)

logger = logging.getLogger(__name__)

_POS_FULL = Position(x=0.0, y=0.0, width=1.0, height=1.0)


def _classify_sheet(name: str, df: pd.DataFrame) -> SheetInfo:
    """Classify a sheet's structure to guide extraction routing."""
    rows, cols = len(df), len(df.columns)
    if cols == 0:
        return SheetInfo(
            name=name, sheet_type="data", row_count=rows, col_count=0,
            key_column=None, has_sub_headers=False,
        )

    try:
        avg_len = float(df.apply(lambda c: c.astype(str).str.len()).stack().mean())
    except Exception:
        avg_len = 0.0

    # Classify by shape and content density.
    # Single column or very long cells → narrative prose.
    # Exactly two columns in a glossary-sized range → glossary.
    # Two columns, small → key-value form.
    # Everything else defaults to tabular data.
    if cols == 1 or (cols <= 3 and avg_len > 150):
        sheet_type = "narrative"
    elif cols == 2 and 50 <= rows <= 500:
        sheet_type = "glossary"
    elif cols == 2 and rows < 50:
        sheet_type = "key_value"
    else:
        sheet_type = "data"

    # Prefer a column whose header signals identity; fall back to first column.
    headers_lower = [str(c).lower() for c in df.columns]
    key_column: str | None = None
    for kw in ("id", "name", "code", "key"):
        key_column = next(
            (str(df.columns[i]) for i, h in enumerate(headers_lower) if kw in h),
            None,
        )
        if key_column:
            break
    if key_column is None:
        key_column = str(df.columns[0])

    # Sub-headers: rows with only the first column populated, occurring 3+ times.
    has_sub_headers = False
    if cols > 1:
        first_nonempty = df.iloc[:, 0].astype(str).str.strip().str.len() > 0
        others_empty = ~df.iloc[:, 1:].astype(str).apply(
            lambda c: c.str.strip().str.len() > 0
        ).any(axis=1)
        has_sub_headers = int((first_nonempty & others_empty).sum()) > 2

    return SheetInfo(
        name=name, sheet_type=sheet_type, row_count=rows, col_count=cols,
        key_column=key_column, has_sub_headers=has_sub_headers,
    )


class SpreadsheetExtractor:
    """Extract from CSV/Excel files, yielding one result per logical unit.

    Data and glossary sheets produce one ExtractionResult per non-empty row.
    Narrative and key-value sheets produce one result for the whole sheet.
    """

    def __init__(self, profile: DocumentProfile | None = None) -> None:
        self.profile = profile

    def extract_path(self, path: Path) -> list[ExtractionResult]:
        import pandas as pd

        suffix = path.suffix.lower()
        sheet_infos = (
            self.profile.sheet_meta if (self.profile and self.profile.sheet_meta) else None
        )

        try:
            if suffix == ".csv":
                df = pd.read_csv(path, dtype=str, keep_default_na=False)
                info = sheet_infos[0] if sheet_infos else _classify_sheet("default", df)
                info.row_count = len(df)  # correct sample count from detection
                return self._extract_sheet(path.stem, "default", df, info)
            else:
                xl = pd.ExcelFile(str(path))
                results: list[ExtractionResult] = []
                for i, name in enumerate(xl.sheet_names):
                    df = xl.parse(name, dtype=str, keep_default_na=False)
                    info = (
                        sheet_infos[i]
                        if sheet_infos and i < len(sheet_infos)
                        else _classify_sheet(str(name), df)
                    )
                    info.row_count = len(df)  # correct sample count from detection
                    results.extend(self._extract_sheet(path.stem, str(name), df, info))
                if not results:
                    return [self._error_result("No sheets found or all sheets empty")]
                return results
        except Exception as e:
            return [self._error_result(f"Failed to read spreadsheet: {e}")]

    def _extract_sheet(
        self, stem: str, sheet_name: str, df: pd.DataFrame, info: SheetInfo
    ) -> list[ExtractionResult]:
        if info.sheet_type in ("narrative", "key_value"):
            return [self._whole_sheet(stem, sheet_name, df, info)]
        return self._rows(stem, sheet_name, df, info)

    def _whole_sheet(
        self, stem: str, sheet_name: str, df: pd.DataFrame, info: SheetInfo
    ) -> ExtractionResult:
        text = df.to_string(index=False)
        doc_id = f"{stem}:{sheet_name}" if sheet_name != "default" else stem
        return ExtractionResult(
            pages=[PageResult(page_number=0, text=text, method="spreadsheet")],
            method="spreadsheet",
            document_id=doc_id,
            tables=[TableData(
                headers=list(df.columns),
                rows=[list(r) for r in df.values],
                position=_POS_FULL,
                confidence=0.95,
            )],
            metadata=ExtractionMetadata(
                extraction_strategy="spreadsheet",
                confidence=0.95,
                processing_time=0.0,
                page_count=1,
                text_coverage=1.0 if text.strip() else 0.0,
            ),
        )

    def _rows(
        self, stem: str, sheet_name: str, df: pd.DataFrame, info: SheetInfo
    ) -> list[ExtractionResult]:
        import numpy as np

        headers = list(df.columns)
        key_col = info.key_column
        results: list[ExtractionResult] = []

        # Build a positional boolean mask of sub-header rows to skip.
        sub_header_mask: np.ndarray | None = None
        if info.has_sub_headers and len(df.columns) > 1:
            first_nonempty = df.iloc[:, 0].astype(str).str.strip().str.len() > 0
            others_empty = ~df.iloc[:, 1:].astype(str).apply(
                lambda c: c.str.strip().str.len() > 0
            ).any(axis=1)
            sub_header_mask = (first_nonempty & others_empty).values

        for row_idx, (_, row) in enumerate(df.iterrows()):
            if sub_header_mask is not None and sub_header_mask[row_idx]:
                continue

            pairs = [(h, str(v).strip()) for h, v in zip(headers, row) if str(v).strip()]
            if not pairs:
                continue
            text = "\n".join(f"{h}: {v}" for h, v in pairs)

            if key_col and key_col in df.columns:
                key_val = str(row[key_col]).strip().replace("/", "-")[:60]
                doc_id = (
                    f"{stem}:{key_val}"
                    if sheet_name == "default"
                    else f"{stem}:{sheet_name}:{key_val}"
                )
            else:
                doc_id = (
                    f"{stem}:row:{row_idx}"
                    if sheet_name == "default"
                    else f"{stem}:{sheet_name}:row:{row_idx}"
                )

            results.append(ExtractionResult(
                pages=[PageResult(page_number=row_idx, text=text, method="spreadsheet")],
                method="spreadsheet",
                document_id=doc_id,
                tables=[TableData(
                    headers=headers,
                    rows=[[str(v) for v in row.values]],
                    position=_POS_FULL,
                    confidence=0.95,
                )],
                metadata=ExtractionMetadata(
                    extraction_strategy="spreadsheet",
                    confidence=0.95,
                    processing_time=0.0,
                    page_count=1,
                    text_coverage=1.0 if text.strip() else 0.0,
                ),
            ))

        if not results:
            logger.warning(
                "spreadsheet produced no data rows: stem=%s sheet=%s", stem, sheet_name
            )
        return results

    @staticmethod
    def _error_result(msg: str) -> ExtractionResult:
        return ExtractionResult(
            pages=[],
            method="spreadsheet",
            error=msg,
            metadata=ExtractionMetadata(
                extraction_strategy="spreadsheet",
                confidence=0.0,
                processing_time=0.0,
                page_count=0,
                text_coverage=0.0,
            ),
        )
