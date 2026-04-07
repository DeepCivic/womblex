"""Document type detection for extraction strategy routing.

Profiles a PDF to determine which extraction strategy to use.
Detection is based on text-layer presence, text quality, image presence,
table structure signals, and handwriting indicators.
"""

import logging
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import fitz

from womblex.config import DetectionConfig

# Suppress pymupdf_layout suggestion from find_tables()
warnings.filterwarnings("ignore", message=".*pymupdf_layout.*")

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document types that drive extraction strategy selection."""

    # PDF types
    SCANNED_HANDWRITTEN = "scanned_handwritten"      # OCR with handwriting config
    SCANNED_MACHINEWRITTEN = "scanned_machinewritten"  # standard OCR
    SCANNED_MIXED = "scanned_mixed"                  # both handwritten and typed
    NATIVE_NARRATIVE = "native_narrative"            # text layer, no structure
    NATIVE_WITH_STRUCTURED = "native_with_structured"  # text layer + tables/images
    STRUCTURED = "structured"                        # pure tabular content
    IMAGE = "image"                                  # photo format (out of scope)
    HYBRID = "hybrid"                                # multiple types in one file
    
    # Non-PDF types
    DOCX = "docx"                                    # Word document (may contain images)
    SPREADSHEET = "spreadsheet"                      # CSV/Excel (may have narrative rows)
    TEXT = "text"                                     # Plain text file (passthrough)
    
    UNKNOWN = "unknown"                              # failed detection


@dataclass
class SheetInfo:
    name: str           # sheet name, or "default" for single-sheet CSV
    sheet_type: str     # "data" | "narrative" | "glossary" | "key_value"
    row_count: int
    col_count: int
    key_column: str | None  # column whose values become part of document_id
    has_sub_headers: bool   # rows that act as section dividers


@dataclass
class DocumentProfile:
    """Result of document type detection."""

    doc_type: DocumentType
    page_count: int
    has_text_layer: bool
    text_coverage: float
    has_images: bool
    has_tables: bool
    has_handwriting_signals: bool
    ocr_confidence: float | None  # None if not sampled
    glyph_regularity: float | None  # 0-1, high = typed (None if not sampled)
    stroke_consistency: float | None  # 0-1, high = typed (None if not sampled)
    confidence: float
    ocr_region_confidences: list[float] | None = None  # per-region scores (0-1) from OCR
    sheet_meta: list[SheetInfo] | None = None  # populated for SPREADSHEET type


# Minimum characters per page to count as having meaningful text.
_MIN_TEXT_LENGTH = 100

# Minimum vector drawing operations to treat a page as "vector-rendered text".
# Pages with many drawings but no text and no images are likely text rendered
# as vector paths (Form XObjects), requiring OCR via pixmap.
_MIN_VECTOR_DRAWINGS = 30

# Pattern for table-like structures: rows with repeated delimiters or whitespace alignment.
_TABLE_PATTERN = re.compile(
    r"(?:"
    r"(?:.*\|.*\|.*\n){2,}"  # pipe-delimited rows
    r"|(?:.*\t.*\t.*\n){2,}"  # tab-delimited rows
    r"|(?:\s{2,}\S+\s{2,}\S+.*\n){3,}"  # whitespace-aligned columns
    r")",
    re.MULTILINE,
)


def _has_table_structure(text: str) -> bool:
    """Detect table-like patterns in extracted text."""
    return bool(_TABLE_PATTERN.search(text))


def _has_structural_tables(page: fitz.Page, min_cells: int = 4) -> bool:
    """Detect tables using PyMuPDF's structural table finder."""
    import sys, io  # noqa: E401
    try:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            tables = page.find_tables()
        finally:
            sys.stdout = old_stdout
        for table in tables.tables:
            if table.row_count * table.col_count >= min_cells:
                return True
        return False
    except Exception:
        return False


def _has_form_structure(page: fitz.Page) -> bool:
    """Detect form field structures on a page.

    Looks for widget annotations (interactive form fields) or
    a high density of short text fragments that suggest labels.
    """
    # Check for interactive form widgets
    widgets = list(page.widgets())
    if len(widgets) >= 2:
        return True

    # Check for label-like text blocks: many short text spans
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    short_text_count = 0
    for block in blocks:
        if block.get("type") == 0:  # text block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if 1 <= len(text) <= 30:
                        short_text_count += 1
    # A page with many short labels is likely a form
    return short_text_count >= 10


def _page_to_grayscale(page: fitz.Page, dpi: int = 72) -> tuple:
    """Convert a PDF page to grayscale numpy array.
    
    Returns (gray_image, width, height) tuple.
    """
    import cv2
    import numpy as np
    
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    
    if pix.n >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    return gray, pix.width, pix.height


def _has_ruled_lines(page: fitz.Page, dpi: int = 72) -> bool:
    """Detect ruled/lined paper patterns (notebook paper).
    
    Looks for evenly-spaced horizontal lines spanning most of page width.
    Requires multiple lines with consistent spacing to distinguish from
    email separators or table borders.
    """
    import cv2
    import numpy as np
    
    gray, width, height = _page_to_grayscale(page, dpi)
    
    # Threshold to binary (invert so lines are white)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Detect long horizontal lines (ruled paper lines span most of page width)
    min_line_width = int(width * 0.4)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_width, 1))
    ruled_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Find rows with significant horizontal line content
    row_sums = np.sum(ruled_lines > 0, axis=1)
    threshold = min_line_width * 0.5
    line_rows = np.where(row_sums > threshold)[0]
    
    if len(line_rows) < 10:
        return False
    
    # Check for even spacing (notebook paper has consistent line spacing)
    spacings = np.diff(line_rows)
    # Filter out tiny gaps (noise) - real line spacing is 20+ pixels at 72dpi
    spacings = spacings[spacings > 15]
    if len(spacings) < 5:
        return False
    # Check if spacing is consistent (std dev < 30% of mean)
    mean_spacing = np.mean(spacings)
    std_spacing = np.std(spacings)
    return bool(mean_spacing > 0 and (std_spacing / mean_spacing) < 0.3)


def _analyze_glyph_regularity(page: fitz.Page, dpi: int = 150) -> float | None:
    """Analyze bounding box regularity of text glyphs using connected components.
    
    Typed text has uniform glyph heights and consistent horizontal spacing.
    Handwritten text has high variance in both dimensions.
    
    Returns regularity score 0-1 (high = typed, low = handwritten) or None if
    insufficient glyphs detected.
    """
    import cv2
    import numpy as np
    
    gray, width, height = _page_to_grayscale(page, dpi)
    
    # Adaptive threshold to handle varying backgrounds
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    
    # Find connected components (individual glyphs/characters)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    # Filter components by size to get likely text glyphs
    # Exclude background (label 0) and very small/large components
    min_area = 20  # Minimum pixels for a glyph
    max_area = (width * height) * 0.01  # Max 1% of page
    min_height = 5
    max_height = height * 0.1  # Max 10% of page height
    
    glyph_heights = []
    
    for i in range(1, num_labels):  # Skip background
        x, y, w, h, area = stats[i]
        if min_area <= area <= max_area and min_height <= h <= max_height:
            # Aspect ratio filter: glyphs are roughly square-ish to tall
            aspect = w / h if h > 0 else 0
            if 0.1 <= aspect <= 3.0:
                glyph_heights.append(h)
    
    if len(glyph_heights) < 50:
        # Not enough glyphs to make a determination
        return None
    
    heights = np.array(glyph_heights)
    
    # Use mode-based analysis: typed text clusters tightly around dominant font size
    # Find the mode (most common height)
    from collections import Counter
    counter = Counter(heights)
    mode_height = counter.most_common(1)[0][0]
    
    # Count glyphs within 25% of mode height
    tolerance = max(mode_height * 0.25, 3)  # At least 3 pixels tolerance
    near_mode = np.sum(np.abs(heights - mode_height) <= tolerance)
    mode_ratio = near_mode / len(heights)
    
    # Map mode_ratio [0.3, 0.7] to regularity [0.0, 1.0]
    # Typed text: 60-80% near mode
    # Handwritten: 30-50% near mode
    regularity = max(0.0, min(1.0, (mode_ratio - 0.3) / 0.4))
    
    return float(regularity)


def _analyze_stroke_width_variance(page: fitz.Page, dpi: int = 150) -> float | None:
    """Analyze stroke width consistency using morphological operations.
    
    Typed text has consistent stroke widths (from uniform font rendering).
    Handwritten text has variable stroke widths (pen pressure, angle changes).
    
    Returns consistency score 0-1 (high = typed, low = handwritten) or None if
    insufficient strokes detected.
    """
    import cv2
    import numpy as np
    
    gray, width, height = _page_to_grayscale(page, dpi)
    
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    
    # Skeletonize to get stroke centerlines
    # Use morphological thinning approximation
    skeleton = binary.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    # Iterative thinning (simplified skeletonization)
    for _ in range(10):
        eroded = cv2.erode(skeleton, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(skeleton, temp)
        skeleton = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    
    # Calculate distance transform on original binary image
    # Distance at skeleton points gives approximate stroke width
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Sample stroke widths at skeleton points
    skeleton_points = skeleton > 0
    stroke_widths = dist_transform[skeleton_points]
    
    # Filter out noise (very small values)
    stroke_widths = stroke_widths[stroke_widths > 1.0]
    
    if len(stroke_widths) < 100:
        # Not enough stroke data
        return None
    
    # Calculate coefficient of variation
    # Typed text: CV typically 0.1-0.25
    # Handwritten: CV typically 0.3-0.6+
    mean_width = np.mean(stroke_widths)
    std_width = np.std(stroke_widths)
    cv = std_width / mean_width if mean_width > 0 else 1.0
    
    # Map CV range [0.1, 0.5] to consistency [1.0, 0.0]
    consistency = max(0.0, min(1.0, 1.0 - (cv - 0.1) / 0.4))
    
    return consistency


def _has_handwriting_signals(page: fitz.Page, dpi: int = 150) -> bool:
    """Detect handwriting indicators on a scanned page.
    
    Uses multiple signals:
    1. Ruled line detection (notebook paper)
    2. Glyph regularity analysis (typed = regular, handwritten = irregular)
    3. Stroke width variance (typed = consistent, handwritten = variable)
    
    Returns True if handwriting is likely present.
    """
    # Check for ruled paper first (strong handwriting signal)
    if _has_ruled_lines(page, dpi=72):
        return True
    
    # Analyze glyph regularity
    glyph_regularity = _analyze_glyph_regularity(page, dpi)
    
    # Analyze stroke width consistency
    stroke_consistency = _analyze_stroke_width_variance(page, dpi)
    
    # If we have both signals, combine them
    if glyph_regularity is not None and stroke_consistency is not None:
        # Both scores low (< 0.4) suggests handwriting
        if glyph_regularity < 0.4 and stroke_consistency < 0.4:
            return True
        # One very low (< 0.25) is also a strong signal
        if glyph_regularity < 0.25 or stroke_consistency < 0.25:
            return True
    
    # If only one signal available, use stricter threshold
    elif glyph_regularity is not None and glyph_regularity < 0.3:
        return True
    elif stroke_consistency is not None and stroke_consistency < 0.3:
        return True
    
    return False


def _sample_ocr_confidence(
    page: fitz.Page, dpi: int = 150
) -> tuple[float | None, list[float] | None]:
    """Sample OCR confidence on a page using PaddleOCR.

    Returns (avg_confidence_0_100, region_confidences_0_1).
    avg_confidence is scaled to 0-100 to match classify thresholds.
    region_confidences are per-text-region scores on the 0-1 scale.
    Typed text typically scores avg >= 85, handwriting scores 40-70.
    """
    try:
        import numpy as np
        from womblex.ingest.paddle_ocr import get_paddle_reader

        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        reader = get_paddle_reader("eng")
        results = reader.readtext(img)  # [(bbox, text, confidence), ...]

        region_confidences = [
            float(conf) for _bbox, text, conf in results if text.strip()
        ]

        if not region_confidences:
            return None, None

        avg_100 = (sum(region_confidences) / len(region_confidences)) * 100
        return avg_100, region_confidences
    except Exception:
        return None, None


def _classify(
    text_pages: int,
    image_pages: int,
    table_signals: int,
    handwriting_signals: int,
    ocr_confidence: float | None,
    glyph_regularity: float | None,
    stroke_consistency: float | None,
    total_pages: int,
    config: DetectionConfig,
) -> DocumentProfile:
    """Classify a document based on aggregated page-level signals."""
    if total_pages == 0:
        return DocumentProfile(
            doc_type=DocumentType.UNKNOWN,
            page_count=0,
            has_text_layer=False,
            text_coverage=0.0,
            has_images=False,
            has_tables=False,
            has_handwriting_signals=False,
            ocr_confidence=None,
            glyph_regularity=None,
            stroke_consistency=None,
            confidence=0.0,
        )

    text_coverage = text_pages / total_pages
    has_text = text_pages > 0
    has_images = image_pages > 0
    has_tables = table_signals > 0
    has_handwriting = handwriting_signals > 0
    table_ratio = table_signals / total_pages
    handwriting_ratio = handwriting_signals / total_pages if total_pages > 0 else 0.0

    # Native documents (have text layer on most pages)
    if text_coverage >= config.min_text_coverage:
        if table_ratio >= 0.8:
            # Tables on nearly every page — pure tabular content
            doc_type = DocumentType.STRUCTURED
            confidence = 0.85
        elif table_ratio >= config.table_signal_threshold or has_images:
            doc_type = DocumentType.NATIVE_WITH_STRUCTURED
            confidence = 0.85
        else:
            doc_type = DocumentType.NATIVE_NARRATIVE
            confidence = min(0.7 + text_coverage * 0.3, 0.95)
    
    # Hybrid: some pages have text layer, some don't
    elif has_text and has_images and 0.1 < text_coverage < config.min_text_coverage:
        doc_type = DocumentType.HYBRID
        confidence = 0.65
    
    # Scanned documents (no/minimal text layer, need OCR)
    elif has_images:
        # Calculate combined morphology score if available
        morphology_score: float | None = None
        if glyph_regularity is not None and stroke_consistency is not None:
            morphology_score = (glyph_regularity + stroke_consistency) / 2
        elif glyph_regularity is not None:
            morphology_score = glyph_regularity
        elif stroke_consistency is not None:
            morphology_score = stroke_consistency
        
        if handwriting_ratio >= 0.8:
            # Strong handwriting signals (ruled paper detected)
            doc_type = DocumentType.SCANNED_HANDWRITTEN
            confidence = 0.75
        elif has_handwriting and handwriting_ratio < 0.8:
            # Mix of handwritten and typed
            doc_type = DocumentType.SCANNED_MIXED
            confidence = 0.70
        elif morphology_score is not None and morphology_score >= 0.6:
            # High regularity → likely typed/printed
            doc_type = DocumentType.SCANNED_MACHINEWRITTEN
            confidence = min(0.5 + morphology_score * 0.4, 0.85)
        elif morphology_score is not None and morphology_score < 0.35:
            # Low regularity → likely handwritten
            doc_type = DocumentType.SCANNED_HANDWRITTEN
            confidence = 0.6
        elif ocr_confidence is not None and ocr_confidence >= 70:
            # Fall back to OCR confidence if morphology inconclusive
            doc_type = DocumentType.SCANNED_MACHINEWRITTEN
            confidence = min(0.5 + ocr_confidence / 200, 0.85)
        elif ocr_confidence is not None and ocr_confidence < 70:
            # Low OCR confidence → route to UNKNOWN
            doc_type = DocumentType.UNKNOWN
            confidence = 0.4
        else:
            # No morphology/OCR signals available — default to machine-written OCR.
            # If we know there are images, OCR is the right path regardless.
            doc_type = DocumentType.SCANNED_MACHINEWRITTEN
            confidence = 0.5
    
    # Unknown: can't determine
    else:
        doc_type = DocumentType.UNKNOWN
        confidence = 0.3

    return DocumentProfile(
        doc_type=doc_type,
        page_count=total_pages,
        has_text_layer=has_text,
        text_coverage=text_coverage,
        has_images=has_images,
        has_tables=has_tables,
        has_handwriting_signals=has_handwriting,
        ocr_confidence=ocr_confidence,
        glyph_regularity=glyph_regularity,
        stroke_consistency=stroke_consistency,
        confidence=confidence,
    )


def detect_document_type(
    path: Path,
    config: DetectionConfig | None = None,
) -> DocumentProfile:
    """Classify a PDF document for extraction strategy selection.

    Args:
        path: Path to the PDF file.
        config: Detection thresholds. Uses defaults if not provided.

    Returns:
        DocumentProfile with detected type and metadata.
    """
    if config is None:
        config = DetectionConfig()

    doc = fitz.open(str(path))
    try:
        text_pages = 0
        image_pages = 0
        table_signals = 0
        handwriting_signals = 0
        scanned_page_for_analysis: fitz.Page | None = None
        
        total_pages = len(doc)
        max_pages = config.max_sample_pages
        
        # Sample evenly distributed pages up to max_sample_pages
        if total_pages <= max_pages:
            page_indices = list(range(total_pages))
        else:
            step = total_pages / max_pages
            page_indices = [int(i * step) for i in range(max_pages)]

        for idx in page_indices:
            page = doc[idx]
            text = page.get_text().strip()
            images = page.get_images()

            has_meaningful_text = len(text) > _MIN_TEXT_LENGTH
            
            if has_meaningful_text:
                text_pages += 1
                if _has_table_structure(text) or _has_structural_tables(page):
                    table_signals += 1

            # Only count as "image page" (needing OCR) if it has images but lacks text.
            # Native PDFs with logos/graphics still have text layers.
            # Also count pages with heavy vector drawings but no text — these are
            # text rendered as vector paths (Form XObjects) that need OCR via pixmap.
            has_ocr_content = bool(images)
            if not has_meaningful_text and not images:
                drawings = page.get_drawings()
                if len(drawings) >= _MIN_VECTOR_DRAWINGS:
                    has_ocr_content = True

            if has_ocr_content and not has_meaningful_text:
                image_pages += 1
                # Track a scanned page for morphology analysis
                if scanned_page_for_analysis is None:
                    scanned_page_for_analysis = page
                if _has_handwriting_signals(page):
                    handwriting_signals += 1
        
        # Scale counts back to full document estimate
        sample_count = len(page_indices)
        if total_pages > sample_count:
            scale = total_pages / sample_count
            text_pages = int(text_pages * scale)
            image_pages = int(image_pages * scale)
            table_signals = int(table_signals * scale)

        # Sample morphology scores if we have scanned pages
        ocr_confidence: float | None = None
        ocr_region_confidences: list[float] | None = None
        glyph_regularity: float | None = None
        stroke_consistency: float | None = None

        if scanned_page_for_analysis is not None and handwriting_signals == 0:
            # Sample morphology scores (no external binary required)
            glyph_regularity = _analyze_glyph_regularity(scanned_page_for_analysis)
            stroke_consistency = _analyze_stroke_width_variance(scanned_page_for_analysis)

            # Only sample OCR if morphology is inconclusive
            if glyph_regularity is None and stroke_consistency is None:
                ocr_confidence, ocr_region_confidences = _sample_ocr_confidence(
                    scanned_page_for_analysis
                )

        profile = _classify(
            text_pages=text_pages,
            image_pages=image_pages,
            table_signals=table_signals,
            handwriting_signals=handwriting_signals,
            ocr_confidence=ocr_confidence,
            glyph_regularity=glyph_regularity,
            stroke_consistency=stroke_consistency,
            total_pages=total_pages,
            config=config,
        )
        profile.ocr_region_confidences = ocr_region_confidences
        return profile
    finally:
        doc.close()


def _detect_spreadsheet(path: Path) -> DocumentProfile:
    """Inspect a CSV or Excel file and classify each sheet's structure."""
    import pandas as pd
    # Local import: spreadsheet.py → extract.py → detect.py would be circular at module level.
    from womblex.ingest.spreadsheet import _classify_sheet  # noqa: PLC0415

    suffix = path.suffix.lower()
    sheet_infos: list[SheetInfo] = []
    try:
        if suffix == ".csv":
            df = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=500)
            sheet_infos.append(_classify_sheet("default", df))
        else:
            xl = pd.ExcelFile(str(path))
            for name in xl.sheet_names:
                df = xl.parse(name, dtype=str, keep_default_na=False, nrows=500)
                sheet_infos.append(_classify_sheet(str(name), df))
    except Exception as e:
        logger.warning("spreadsheet detection failed: path=%s error=%s", path, e)

    return DocumentProfile(
        doc_type=DocumentType.SPREADSHEET,
        page_count=len(sheet_infos) or 1,
        has_text_layer=True,
        text_coverage=1.0,
        has_images=False,
        has_tables=True,
        has_handwriting_signals=False,
        ocr_confidence=None,
        glyph_regularity=None,
        stroke_consistency=None,
        confidence=0.9,
        sheet_meta=sheet_infos or None,
    )


def _detect_docx(path: Path) -> DocumentProfile:
    """Detect Word document characteristics.

    Note: DOCX files may contain embedded images that need OCR.
    This basic detection doesn't analyze image content.
    """
    has_images = False
    has_tables = False
    text_length = 0
    try:
        from docx import Document
        doc = Document(str(path))
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                has_images = True
                break
        has_tables = len(doc.tables) > 0
        for para in doc.paragraphs:
            text_length += len(para.text)
    except ImportError:
        # python-docx not installed, return basic profile
        pass
    except Exception:
        # Failed to parse, return unknown
        return DocumentProfile(
            doc_type=DocumentType.UNKNOWN,
            page_count=0,
            has_text_layer=False,
            text_coverage=0.0,
            has_images=False,
            has_tables=False,
            has_handwriting_signals=False,
            ocr_confidence=None,
            glyph_regularity=None,
            stroke_consistency=None,
            confidence=0.3,
        )
    
    return DocumentProfile(
        doc_type=DocumentType.DOCX,
        page_count=1,  # DOCX doesn't expose page count easily
        has_text_layer=True,
        text_coverage=1.0 if text_length > 0 else 0.0,
        has_images=has_images,
        has_tables=has_tables,
        has_handwriting_signals=False,
        ocr_confidence=None,
        glyph_regularity=None,
        stroke_consistency=None,
        confidence=0.85,
    )


def detect_file_type(
    path: Path,
    config: DetectionConfig | None = None,
) -> DocumentProfile:
    """Classify any supported file for extraction strategy selection.

    Handles PDFs, Word documents, spreadsheets, and plain text files.

    Args:
        path: Path to the file.
        config: Detection thresholds. Uses defaults if not provided.

    Returns:
        DocumentProfile with detected type and metadata.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    
    if suffix == ".pdf":
        return detect_document_type(path, config)
    elif suffix == ".docx":
        return _detect_docx(path)
    elif suffix in (".csv", ".xlsx", ".xls"):
        return _detect_spreadsheet(path)
    elif suffix == ".txt":
        return DocumentProfile(
            doc_type=DocumentType.TEXT,
            page_count=1,
            has_text_layer=True,
            text_coverage=1.0,
            has_images=False,
            has_tables=False,
            has_handwriting_signals=False,
            ocr_confidence=None,
            glyph_regularity=None,
            stroke_consistency=None,
            confidence=1.0,
        )
    else:
        return DocumentProfile(
            doc_type=DocumentType.UNKNOWN,
            page_count=0,
            has_text_layer=False,
            text_coverage=0.0,
            has_images=False,
            has_tables=False,
            has_handwriting_signals=False,
            ocr_confidence=None,
            glyph_regularity=None,
            stroke_consistency=None,
            confidence=0.0,
        )
