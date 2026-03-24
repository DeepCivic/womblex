# Chunking Accuracy

Chunking behaviour across all fixtures. The chunker wraps `semchunk` with configurable
tokeniser and chunk size. No chunking ground truth exists yet — this report documents
observed behaviour to establish a baseline for future evaluation.

**Date:** 2026-03-22
**Chunker:** semchunk 3.x
**Config:** word-count tokeniser, chunk_size=200 words (test default)

## Fixture Overview

| Dataset | Fixture | Format | Chunking Ground Truth | Chunkable |
|---------|---------|--------|-----------------------|-----------|
| FUNSD | `85540866` | PNG | None | No — 25 words (below chunk size) |
| FUNSD | `82200067_0069` | PNG | None | No — 181 words (below chunk size) |
| FUNSD | `87594142_87594144` | PNG | None | Marginally — 434 words |
| FUNSD | `87528321` | PNG | None | Marginally — 211 words |
| FUNSD | `87528380` | PNG | None | Marginally — 428 words |
| IAM | `short_1602` | PNG | None | No — 1 word |
| IAM | `median_15` | PNG | None | No — 9 words |
| IAM | `long_4` | PNG | None | No — 22 words |
| IAM | `wide_1739` | PNG | None | No — 12 words |
| IAM | `narrow_1163` | PNG | None | No — 2 words |
| DocLayNet | `dense_text_548` | PNG | None | Marginally — 413 words |
| DocLayNet | `diverse_layout_49` | PNG | None | Marginally |
| DocLayNet | `sparse_text_344` | PNG | None | No — 13 words |
| DocLayNet | `formula_29` | PNG | None | Marginally |
| DocLayNet | `table_0` | PNG | None | Marginally |
| womblex | `Throsby...Redacted` | PDF | None | Yes — 730 words |
| womblex | `Auditor-General_Report_2020-21_19` | PDF | None | Yes — 193,240 words |
| womblex | `dfat-corporate-plan-2025-26` | DOCX | None | Yes — 5,836 words |
| womblex | `Approved-providers-au-export` | CSV | None | Yes — 10,859 rows |
| womblex | `mso-statistics-sept-qtr-2025` | XLSX | None | Yes — 34 sheet results |

> The OCR fixtures (FUNSD, IAM, DocLayNet) are image crops and single pages — they
> produce text below or near the chunk size threshold. Chunking evaluation is
> meaningful only for the womblex-collection fixtures which produce document-length text.

## Womblex Collection — Chunking Results

Chunked with a word-count tokeniser at 200-word chunk size (test configuration).
Production uses a HuggingFace tokeniser (e.g. `isaacus/kanon-2-tokenizer`) at 480 tokens.

| Fixture | Extraction Results | Total Chars | Chunks | Avg Chunk (chars) |
|---------|-------------------|-------------|--------|-------------------|
| `Throsby...Redacted` | 1 | 4,898 | 5 | 979 |
| `Auditor-General_Report_2020-21_19` | 1 | 1,326,218 | 1,182 | 1,122 |
| `dfat-corporate-plan-2025-26` | 1 | 41,451 | 35 | 1,184 |
| `Approved-providers-au-export` | 10,859 | 2,422,497 | 10,867 | 222 |
| `mso-statistics-sept-qtr-2025` | 34 | 34,786 | 36 | 966 |

### Observations

**Throsby (5 chunks from 3 pages):**
Produces well-sized chunks from a short narrative document. With 730 words and a
200-word chunk limit, the 5-chunk split is expected. Chunk boundaries should fall at
paragraph breaks (numbered paragraphs 1-16).

**Auditor-General (1,182 chunks from 406 pages):**
Large report chunks cleanly. Average chunk size of 1,122 chars is consistent across
the document. Note: benchmark tests now evaluate only the first 30 pages; the full
406-page numbers here are from an earlier run.

**DFAT Corporate Plan (35 chunks from 1 DOCX):**
DOCX extraction produces a single text block; semchunk splits it into 35 chunks.
Average chunk size of 1,184 chars is close to the Auditor-General, suggesting
consistent chunking behaviour across document types.

**Approved Providers CSV (10,867 chunks from 10,859 rows):**
Near 1:1 mapping of rows to chunks. The 8 extra chunks are rows whose text exceeded
the chunk size (provider records with long address or trading name fields). Average
chunk size of 222 chars reflects the short per-row text from key-value extraction.

**MSO Statistics XLSX (36 chunks from 34 sheet results):**
Two sheets produced text exceeding the chunk limit, resulting in 2 extra chunks.
Average chunk size of 966 chars reflects mixed narrative and tabular content.

### Chunk Size Distribution Characteristics

| Document type | Chunk size pattern | Notes |
|---------------|-------------------|-------|
| Narrative PDF | ~1,000-1,200 chars | Consistent; boundary at paragraph breaks |
| DOCX | ~1,000-1,200 chars | Same as narrative PDF — expected |
| CSV (per-row) | ~200-300 chars | Short; one row per chunk typical |
| XLSX (per-sheet) | ~800-1,200 chars | Variable; depends on sheet type |

## Chunking and Redaction Interaction

When `redaction.mode = "flag"`, chunks on pages with detected redactions are annotated
with `has_redaction = True`. The Throsby document tests this:

| Fixture | Redacted pages | Chunks affected |
|---------|---------------|-----------------|
| `Throsby...Redacted` | Page 0 | Chunks sourced from page 0 |
| `Auditor-General_Report_2020-21_19` | 283 pages (false positives) | Would incorrectly flag ~70% of chunks |

The Auditor-General false-positive problem (see `REDACTION_HANDLING.md`) propagates
into chunking — disabling redaction detection or raising the threshold for non-redacted
datasets prevents this.

## Known Limitations

1. **No chunking ground truth** — there is no human-annotated "correct" chunking for any
   fixture. Evaluation is limited to structural metrics (count, size distribution).
2. **Test tokeniser differs from production** — tests use word-count; production uses a
   HuggingFace subword tokeniser. Chunk boundaries will differ.
3. **No semantic coherence metric** — chunks are evaluated by size, not by whether they
   contain semantically complete units. Future work could measure topic coherence or
   downstream retrieval performance.
4. **Table chunking not evaluated** — `chunk_tables=true` converts tables to markdown
   before chunking, but no fixtures have table-specific chunking ground truth.
5. **Spreadsheet row-level chunking is trivial** — CSV/XLSX rows are short enough that
   most become single chunks. The chunker adds little value here; it's the extraction
   strategy (one result per row) that determines granularity.
