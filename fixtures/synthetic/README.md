# Synthetic fixtures

Invented documents about Australian animals, in the shapes real government
releases take. They let every test that needs a real file run on a bare
checkout, with no benchmark data and no paid API. They are not ground truth:
scoring extraction against real documents happens in womblex-benchmark.

The content is invented and names no real person or organisation. It is
released under the repository's licence (Apache-2.0).

| File | Shape | Stands in for |
|---|---|---|
| `documents/wombat-portfolio-budget-statements.docx` | DOCX: headings, prose, nine tables (one header-only) | A portfolio budget statement |
| `documents/quokka-care-decision-notice_redacted.pdf` | Three-page native PDF, a logo image, six vector redactions on page one | A redacted FOI decision notice |
| `documents/koala-habitat-audit.pdf` | Six-page native PDF, chapter headings, page footers, one ruled table | An audit report |
| `documents/koala-habitat-audit_transcript.txt` | Plain text of the audit report | A report transcript |
| `documents/bilby-foi-documents-index.pdf` | A spreadsheet printed to PDF: four rotated pages (portrait MediaBox, `/Rotate 90`), a metadata block, two-line headers, 144 rows | An FOI manifest |
| `documents/bilby-schedule-of-documents.pdf` | A one-page schedule printed from a spreadsheet, 44 rows | An FOI schedule of documents |
| `documents/numbat-scanned-survey-page.pdf` | Image-only PDF of `scans/page-dense.png`: OCR and the layout step run on it | A scanned report page |
| `spreadsheets/platypus-sightings-register.csv` | 300-row register export | A register CSV |
| `spreadsheets/echidna-population-statistics.xlsx` | Three sheets, title rows above each table | A statistics workbook |

`scans/` holds images with a `.gt.txt` of the text drawn on each, in three
groups that stand in for the public datasets the benchmark scores against:

| Files | Shape | Stands in for |
|---|---|---|
| `form-*.png` | A titled form, labelled boxes holding typed values, a tick box | FUNSD forms |
| `line-field-note-*.png` | One line of text | IAM handwriting lines |
| `page-{dense,table,sparse}.png` | A report page: prose and a ruled table, a larger table, or a heading and one line | DocLayNet pages |

Each is drawn clean, then given grey paper, sparse speckle and a 0.4 degree
skew so it reads as a scan.

Test code reaches these through `tests/_synthetic.py`.

## Regenerating

```bash
uv run python fixtures/synthetic/generate.py
```

Generation is deterministic (seeded content, reportlab's invariant mode, fixed
zip timestamps and document dates), so an unchanged generator reproduces the
committed bytes. A change to the generator changes source hashes and content
digests; re-pin the tests that pin them in the same merge.
