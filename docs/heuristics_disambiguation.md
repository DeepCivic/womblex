# Heuristics Disambiguation

Reference for CV2 and NumPy-based heuristics used in document classification and processing.

---

**CV2** — spatial, structural, morphological operations on image geometry

| Heuristic | Signal | Technical Reference | Status |
|-----------|--------|---------------------|--------|
| Skew angle (`minAreaRect`) | Scan indicator | `heuristics_cv2.detect_skew_angle` | ✓ Implemented |
| Horizontal ruled line detection (morphology) | Handwriting indicator | `detect._has_ruled_lines` | ✓ Implemented |
| Solid rectangle detection (`findContours` + fill) | Redaction indicator | `redaction.RedactionDetector.detect` | ✓ Implemented |
| Large dark connected components (`connectedComponentsWithStats`) | Redaction indicator | `redaction.RedactionDetector._is_redaction_candidate` | ✓ Implemented |
| Connected component bounding box regularity | Typed vs handwritten | `detect._analyze_glyph_regularity` | ✓ Implemented |
| Stroke width variance (erosion/dilation) | Typed vs handwritten | `detect._analyze_stroke_width_variance` | ✓ Implemented |
| Table/grid detection (Hough transform) | Structured content | `heuristics_cv2.detect_table_grid` | ✓ Implemented |
| Column alignment (vertical projection) | Typed vs handwritten | `heuristics_cv2.detect_column_alignment` | ✓ Implemented |
| Scanner border/shadow at margins | Scan indicator | `heuristics_cv2.detect_scanner_margins` | ✓ Implemented |
| White-on-black glyph remnants in dark regions | Imperfect redaction | `heuristics_cv2.analyze_redaction_quality` | ✓ Implemented |
| Contour complexity regularity | Typed vs handwritten | `heuristics_cv2.analyze_contour_complexity` | ✓ Implemented |
| Colour uniformity inside dark bounding boxes (`mean`/`stddev`) | Redaction confidence | `heuristics_cv2.analyze_redaction_quality` | ✓ Implemented |
| Laplacian variance (blur detection) | Pre-OCR validation | `heuristics_cv2.calculate_blur_score` | ✓ Implemented |
| Cardinal rotation (90°/180°/270°) | Orientation correction | `heuristics_cv2.detect_cardinal_rotation` | ✓ Implemented |
| Photo/text region segmentation | Region classification | `heuristics_cv2.segment_text_photo_regions` | ✓ Implemented |

---

**NumPy** — statistical, frequency domain, density operations on pixel arrays

| Heuristic | Signal | Technical Reference | Status |
|-----------|--------|---------------------|--------|
| Row/column pixel sums (`np.sum`) | Column alignment, line regularity | `detect._has_ruled_lines`, `heuristics_numpy.analyze_horizontal_bands` | ✓ Implemented |
| Intensity histogram (`np.histogram`) | Noise floor, scan vs native | `heuristics_numpy.analyze_histogram` | ✓ Implemented |
| OTSU threshold value distribution | Scan vs native | `heuristics_numpy.analyze_otsu_threshold` | ✓ Implemented |
| FFT periodicity (`np.fft.fft2`) | Typed regularity, moiré detection | `heuristics_numpy.analyze_fft_periodicity` | ✓ Implemented |
| Per-region mean/stddev | Ink density evenness, typed vs handwritten | `heuristics_numpy.analyze_ink_density` | ✓ Implemented |
| Pixel density per horizontal band | Handwriting unevenness | `heuristics_numpy.analyze_horizontal_bands` | ✓ Implemented |
| Dark pixel ratio across margin zones | Scanner shadow gradient | `heuristics_numpy.analyze_margin_darkness` | ✓ Implemented |
| Percentile spread of pixel intensities | Image quality / compression artefacts | `heuristics_numpy.analyze_percentile_spread` | ✓ Implemented |
| Inter-line spacing variance (from row sums) | Typed vs handwritten | `heuristics_numpy.analyze_horizontal_bands` | ✓ Implemented |
| Watermark detection (FFT high-freq) | Watermark presence | `heuristics_numpy.detect_watermark_confidence` | ✓ Implemented |
| DPI estimation (margin analysis) | Resolution detection | `heuristics_numpy.estimate_dpi` | ✓ Implemented |

---

**Both / combined signal**

| Heuristic | CV2 role | NumPy role | Technical Reference | Status |
|-----------|----------|------------|---------------------|--------|
| Ink density | Binarise + contours | Per-band density stats | `heuristics_numpy.analyze_ink_density` | ✓ Implemented |
| Noise characterisation | Salt/pepper morphology | Histogram tail analysis | `heuristics_numpy.analyze_histogram` | ✓ Implemented |
| Redaction confidence | Shape detection | Uniformity stats inside shape | `heuristics_cv2.analyze_redaction_quality` | ✓ Implemented |
| Stroke consistency | Morphological sizing | Variance across components | `detect._analyze_stroke_width_variance` | ✓ Implemented |

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
| `womblex.ingest.detect` | `_has_form_structure` | Detect form widgets and label-like text blocks |
| `womblex.ingest.detect` | `_sample_ocr_confidence` | PaddleOCR confidence sampling (fallback) |
| `womblex.redact.detector` | `RedactionDetector` | CV2-based redaction detection and masking |
| `womblex.redact.detector` | `RedactionDetector.detect` | Find dark rectangular regions |
| `womblex.redact.detector` | `RedactionDetector.mask` | White-out redacted regions before OCR |
| `womblex.ingest.heuristics_cv2` | `detect_skew_angle` | Detect document rotation via Hough lines |
| `womblex.ingest.heuristics_cv2` | `detect_table_grid` | Find table/grid structures via morphology |
| `womblex.ingest.heuristics_cv2` | `detect_column_alignment` | Measure vertical projection regularity |
| `womblex.ingest.heuristics_cv2` | `detect_scanner_margins` | Detect scanner shadow at page edges |
| `womblex.ingest.heuristics_cv2` | `analyze_redaction_quality` | Check redaction solidity and remnants |
| `womblex.ingest.heuristics_cv2` | `analyze_contour_complexity` | Measure contour regularity for typed vs handwritten |
| `womblex.ingest.heuristics_cv2` | `calculate_blur_score` | Laplacian variance for blur/sharpness detection |
| `womblex.ingest.heuristics_cv2` | `detect_cardinal_rotation` | Detect 90°/180°/270° rotation |
| `womblex.ingest.heuristics_cv2` | `segment_text_photo_regions` | Segment text vs photo regions via edge density |
| `womblex.ingest.heuristics_numpy` | `analyze_histogram` | Intensity distribution for scan detection |
| `womblex.ingest.heuristics_numpy` | `analyze_otsu_threshold` | Bimodal threshold analysis |
| `womblex.ingest.heuristics_numpy` | `analyze_fft_periodicity` | Frequency domain regularity and moiré detection |
| `womblex.ingest.heuristics_numpy` | `analyze_ink_density` | Ink coverage distribution across regions |
| `womblex.ingest.heuristics_numpy` | `analyze_horizontal_bands` | Line spacing regularity analysis |
| `womblex.ingest.heuristics_numpy` | `analyze_image_quality` | Compression artifacts, sharpness, noise |
| `womblex.ingest.heuristics_numpy` | `analyze_margin_darkness` | Scanner shadow gradient detection |
| `womblex.ingest.heuristics_numpy` | `analyze_percentile_spread` | Dynamic range / quality assessment |
| `womblex.ingest.heuristics_numpy` | `detect_watermark_confidence` | FFT-based watermark presence detection |
| `womblex.ingest.heuristics_numpy` | `estimate_dpi` | Estimate scanning resolution from margins |
