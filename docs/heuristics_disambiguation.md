# Heuristics Disambiguation

Reference for CV2 and NumPy-based heuristics used in document classification and processing.

---

**CV2** — spatial, structural, morphological operations on image geometry

| Heuristic | Signal | Technical Reference | Status |
|-----------|--------|---------------------|--------|
| Skew angle (`minAreaRect`) | Scan indicator | `heuristics_cv2.detect_skew_angle` | ✓ Implemented |
| Horizontal ruled line detection (morphology) | Handwriting indicator | `detect._has_ruled_lines` | ✓ Implemented |
| Solid rectangle detection (`findContours` + fill) | Redaction indicator | `redact.detector.RedactionDetector.detect` | ✓ Implemented |
| Large dark connected components (`connectedComponentsWithStats`) | Redaction indicator | `redact.detector.RedactionDetector._is_redaction_candidate` | ✓ Implemented |
| Connected component bounding box regularity | Typed vs handwritten | `detect._analyze_glyph_regularity` | ✓ Implemented |
| Stroke width variance (erosion/dilation) | Typed vs handwritten | `detect._analyze_stroke_width_variance` | ✓ Implemented |
| Table/grid detection (morphological line extraction) | Structured content | `heuristics_cv2.detect_table_grid` | ✓ Implemented |
| Contour complexity regularity | Typed vs handwritten | `heuristics_cv2.analyze_contour_complexity` | ✓ Implemented |
| Laplacian variance (blur detection) | Pre-OCR validation | `heuristics_cv2.calculate_blur_score` | ✓ Implemented |
| Photo/text region segmentation (edge density) | Region classification | `heuristics_cv2.segment_text_photo_regions` | ✓ Implemented |
| Column alignment (vertical projection) | Typed vs handwritten | — | Not implemented |
| Scanner border/shadow at margins | Scan indicator | — | Not implemented |
| White-on-black glyph remnants in dark regions | Imperfect redaction | — | Not implemented |
| Colour uniformity inside dark bounding boxes | Redaction confidence | — | Not implemented |
| Cardinal rotation (90°/180°/270°) | Orientation correction | — | Not implemented |

---

**NumPy** — statistical, frequency domain, density operations on pixel arrays

| Heuristic | Signal | Technical Reference | Status |
|-----------|--------|---------------------|--------|
| Intensity histogram analysis | Noise floor, scan vs native | `heuristics_numpy.analyze_histogram` | ✓ Implemented |
| OTSU threshold value distribution | Bimodal separation, scan vs native | `heuristics_numpy.analyze_otsu_threshold` | ✓ Implemented |
| Row/column pixel sums (`np.sum`) | Line regularity | `detect._has_ruled_lines` (inline) | ✓ Implemented |
| FFT periodicity (`np.fft.fft2`) | Typed regularity, moiré detection | — | Not implemented |
| Per-region mean/stddev | Ink density evenness | — | Not implemented |
| Pixel density per horizontal band | Handwriting unevenness | — | Not implemented |
| Dark pixel ratio across margin zones | Scanner shadow gradient | — | Not implemented |
| Percentile spread of pixel intensities | Image quality / compression artefacts | — | Not implemented |
| Watermark detection (FFT high-freq) | Watermark presence | — | Not implemented |
| DPI estimation (margin analysis) | Resolution detection | — | Not implemented |

---

**Both / combined signal**

| Heuristic | CV2 role | NumPy role | Technical Reference | Status |
|-----------|----------|------------|---------------------|--------|
| Stroke consistency | Morphological sizing | Variance across components | `detect._analyze_stroke_width_variance` | ✓ Implemented |
| Binarisation decision | — | Histogram bimodality | `heuristics_numpy.analyze_histogram` + `analyze_otsu_threshold` | ✓ Implemented |

---

**Key Classes and Functions**

| Module | Identifier | Purpose |
|--------|------------|---------|
| `womblex.ingest.detect` | `_page_to_grayscale` | Convert PDF page to grayscale numpy array |
| `womblex.ingest.detect` | `_has_ruled_lines` | Detect notebook paper (evenly-spaced horizontal lines) |
| `womblex.ingest.detect` | `_analyze_glyph_regularity` | Measure bounding box variance of connected components |
| `womblex.ingest.detect` | `_analyze_stroke_width_variance` | Measure stroke width consistency via distance transform |
| `womblex.ingest.detect` | `_has_handwriting_signals` | Composite handwriting detection (combines above) |
| `womblex.ingest.detect` | `_has_table_structure` | Regex-based table detection in extracted text |
| `womblex.ingest.detect` | `_has_structural_tables` | PyMuPDF table finder (min cell count) |
| `womblex.ingest.detect` | `_has_form_structure` | Detect form widgets and label-like text blocks |
| `womblex.ingest.detect` | `_sample_ocr_confidence` | PaddleOCR confidence sampling (fallback) |
| `womblex.redact.detector` | `RedactionDetector` | CV2-based redaction detection and masking |
| `womblex.redact.detector` | `RedactionDetector.detect` | Find dark rectangular regions |
| `womblex.redact.detector` | `RedactionDetector.mask` | White-out redacted regions before OCR |
| `womblex.ingest.heuristics_cv2` | `detect_skew_angle` | Detect document rotation via Hough lines |
| `womblex.ingest.heuristics_cv2` | `detect_table_grid` | Find table/grid structures via morphology |
| `womblex.ingest.heuristics_cv2` | `analyze_contour_complexity` | Measure contour regularity for typed vs handwritten |
| `womblex.ingest.heuristics_cv2` | `calculate_blur_score` | Laplacian variance for blur/sharpness detection |
| `womblex.ingest.heuristics_cv2` | `segment_text_photo_regions` | Segment text vs photo regions via edge density |
| `womblex.ingest.heuristics_numpy` | `analyze_histogram` | Intensity distribution for scan detection |
| `womblex.ingest.heuristics_numpy` | `analyze_otsu_threshold` | Bimodal threshold analysis |
