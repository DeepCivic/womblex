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
| `spreadsheets/platypus-sightings-register.csv` | 300-row register export | A register CSV |
| `spreadsheets/echidna-population-statistics.xlsx` | Three sheets, title rows above each table | A statistics workbook |

Test code reaches these through `tests/_synthetic.py`.

## Regenerating

```bash
uv run python fixtures/synthetic/generate.py
```

Generation is deterministic (seeded content, reportlab's invariant mode, fixed
zip timestamps and document dates), so an unchanged generator reproduces the
committed bytes. A change to the generator changes source hashes and content
digests; re-pin the tests that pin them in the same merge.
