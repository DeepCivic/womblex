# Third-Party Data

Test fixtures used by this project are maintained in a separate repository:

- **Repository:** [DeepCivic/womblex-development-fixtures](https://github.com/DeepCivic/womblex-development-fixtures)
- **Contents:** FUNSD form images, IAM handwriting lines, DocLayNet layout pages, womblex-collection documents
- **Purpose:** Real-document test data for extraction accuracy benchmarks

## Setup for development

Clone the fixtures repository into the project root:

```bash
git clone https://github.com/DeepCivic/womblex-development-fixtures.git fixtures
```

Tests expect the fixtures at `fixtures/fixtures/` (the inner `fixtures/` is the data root within that repository).

## Licence

Each fixture dataset retains its original licence. See the fixtures repository README for per-dataset attribution and licence details.
