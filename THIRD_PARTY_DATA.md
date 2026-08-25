# Third-Party Data

Test fixtures used by this project are maintained in a separate repository:

- **Repository:** [DeepCivic/womblex-benchmark](https://github.com/DeepCivic/womblex-benchmark) (formerly `womblex-development-fixtures`)
- **Contents:** FUNSD form images, IAM handwriting lines, DocLayNet layout pages, womblex-collection documents
- **Purpose:** Real-document test data for extraction accuracy benchmarks

## Vendored minimum vs. full set

A **minimal, redistributable** fixture set is vendored in this repo under
`fixtures/fixtures/` so a bare clone runs the bulk of the suite with no extra
setup: FUNSD / IAM / DocLayNet samples, the register CSV/XLSX, the public
ANAO Auditor-General and DFAT budget documents, and the Throsby 213A notice PDF
(the single ACT FOI document vendored — see below). The tests resolve fixtures
at `fixtures/fixtures/`.

The **full** set (every strategy document, the benchmark cohorts, the ACT-ECI
labelled pages, etc.) lives in the external repo and is used by the accuracy
benchmarks:

```bash
git clone https://github.com/DeepCivic/womblex-benchmark.git fixtures
```

**ACT FOI 213A documents.** The **Throsby notice PDF** is vendored under
`fixtures/fixtures/womblex-collection/_documents/` so the per-stage
enrich/persist-document tests run on a bare checkout. The remaining 213A
documents (the returned-documents Index, the Part-2b Schedule) are research-use
only and are **not vendored** — they live only in the external fixtures repo.
Tests that need those (spreadsheet-print, parts of integration) skip cleanly
without them ("FOI fixture not available"); clone the full fixtures repo to run
those locally. Per-fixture vendoring is a deliberate, human-approved decision.

## Licence

Each fixture dataset retains its original licence. See the fixtures repository README for per-dataset attribution and licence details.
