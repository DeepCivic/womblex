# Permissive deps — L1 working notes

*Scratch notes from the L1 session (2026-10-02). Not a decision record; fold the outcome into `permissive-deps-plan.md` / `decisions.md` and delete this file when L1-B lands.*

## State
- Branch `permissive-deps-l1-pp-doclayout` (womblex), one commit, pushed. No PR opened.
- L1 (W) is done: `ingest/layout_onnx.py` (`PPDocLayoutAnalyzer`), model under `_models/pp-doclayout-m/`, `get_layout_analyzer()` repointed, `tests/test_layout_onnx.py`, docs and CHANGELOG. YOLO code is untouched (removed in L2).
- Not done: L1-B (benchmark pair), the `CLAUDE.md` module-table row for `layout_onnx.py`, Phase 0 spike.
- ruff and mypy clean. Full womblex suite passes except one error that also occurs on main: `test_ai_chunking_reuse.py::TestReuseGuard::test_mismatched_text_falls_back_to_string`.

## Reproducing the model export
Source: `PaddlePaddle/PP-DocLayout-M` on Hugging Face, revision `7dbfcce3154a55776dc71ca026a4a2a8388dad8d` (files: `inference.json`, `inference.pdiparams`, `inference.yml`).
```bash
uv venv cv && uv pip install -p cv paddle2onnx onnx onnxruntime paddlepaddle setuptools
cv/bin/paddle2onnx --model_dir . --model_filename inference.json \
  --params_filename inference.pdiparams --save_file inference.onnx --opset_version 14
```
- `paddle2onnx` 2.1.0 imports `paddle`, which needs `setuptools`; neither is pulled in by `paddle2onnx` itself.
- Constant folding is skipped (needs `onnx_graphsurgeon`, which wants pip in the venv). The export is valid without it.
- Exported graph takes `image` (N,3,640,640) and `scale_factor` (N,2) and returns `[cls, score, x0, y0, x1, y1]` rows in original-image pixels; batch size 1 only (NMS op).
- Expected SHA-256: `inference.onnx` `a5a92ea2…4250`, `inference.yml` `76aeb103…d70d` (full values in `docs/models.md`). A different `paddle2onnx` version may change the ONNX digest.

## Benchmark run (womblex-benchmark against this branch)
Setup:
```bash
git clone https://github.com/DeepCivic/womblex-benchmark ../womblex-benchmark
cd ../womblex-benchmark && git lfs pull        # fixtures are LFS pointers until pulled (~200 MB)
<womblex>/.venv/bin/python -m pytest accuracy/test_fixture_accuracy.py \
  accuracy/test_table_benchmark.py accuracy/test_womblex_collection_accuracy.py -q
```
- The benchmark finds womblex by sibling directory (`womblex` or `Womblex`), or set `WOMBLEX_DOCS_DIR`. Suites write reports into `<womblex>/docs/accuracy/`.
- Took about 18 minutes. 71 passed, 2 skipped, 1 failed.
- Failure: `test_table_benchmark.py::TestDenseTextTracking::test_gt_rect_reconstruction` expects 1 ground-truth table on `dense_text_548`, finds 3 (`_aggregate_doclaynet_blocks` output). It reads annotations only and never calls the layout model, so likely not L1-related; unconfirmed against main.
- I restored the regenerated `docs/accuracy/` files rather than committing them: the committed reports date from 2026-08-18, so the diff mixed unrelated drift with the layout change.

## Layout results (DocLayNet, 5 fixtures)
| Measure | Before (committed report) | PP-DocLayout-M |
|---|---|---|
| Average block P / R / F1 | 29.2 / 40.8 / 31.9% | 29.5 / 40.3 / 29.7% |
| `table` class (P / R / F1) | 100 / 50 / 66.7% | 100 / 25 / 40.0% (1 true, 3 false regions) |
| `dense_text_548` F1 | 66.7% | 37.5% |
| `diverse_layout_49` F1 | 27.8% | 14.3% |
| `sparse_text_344` / `formula_29` F1 | 33.3 / 11.8% | 40.0 / 37.5% |
| False-table cohort | — | 0 of 8 false positives |

The "before" column is the stale committed report, not a fresh YOLO run, so it is not a controlled comparison. Plan gate (F1 at least 0.29, `dense_text_548` table found) holds on the average.

## Open items
1. Run the same three suites on `main` to get a controlled YOLO baseline, then diff.
2. Check whether the `dense_text_548` ground-truth failure also fails on main.
3. Investigate the extra table regions (3 vs 1) and the rise in figure and header false positives; the label map (`LABEL_MAP`) and the 0.3 threshold are the knobs. `number` maps to `footer` and `chart` to `figure`, which overlaps `image` on the same box (seen on `dense_text_548`: a `chart` box almost identical to the `table` box).
4. L1-B work in the benchmark: update `DOCLAYNET_TO_WOMBLEX` and the report wording in `accuracy_reports.py` (still names YOLO), regenerate reports, spot-check redaction exclusion area on the 02737-class scanned forms.
5. Add the `layout_onnx.py` row to the `CLAUDE.md` module table (listed in the plan for L1, not yet done).
6. `onnxruntime` is imported with `type: ignore[import-untyped]`; L2 makes it a direct dependency and can replace that with a mypy override.
7. Environment: `_models/pp-doclayout-m/` is not in package-data (as the YOLO weights), so an installed wheel resolves it only via `WOMBLEX_MODELS_DIR` or an editable install.
