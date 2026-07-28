# Monetary amount recognition

Design for the `money` annotation op: recovering monetary amounts from
Womblex's extraction output, normalising them to exact values, and recording
them as a joinable sidecar.

Status: **shipped.** The op is built — `womblex money --shards <dir>` writes
`*.money_spans.parquet` + `*.money_columns.parquet` per batch
(`process/money*.py`, `store/money_output.py`). The extractor change under
[Shipped prerequisite](#shipped-prerequisite) is merged. What remains open is
the measurement: there is still no labelled ground truth, so no precision or
recall figure is quoted here — see
[Open gap: no ground truth](#open-gap-no-ground-truth).

## Scope and naming

"Money amounts", not the legal sense of *currency* (point-in-time / in-force
dates). The two were conflated early and are unrelated problems. The legal
sense is a separate and cheaper win — `DateInfo.effective/expiry` and
`Quote.amending` are already returned by Kanon-2 and then discarded, because
`ENTITY_SCHEMA` (`store/enrichment_output.py`) persists only
`person|location|term|external_document`. Dates survive as a bare `date_count`
int; the same applies to emails, websites, phones, id_numbers and quotes.
That work is tracked separately and is not in scope here.

This file is named `money.md` rather than `currency.md` to keep the distinction
visible.

## The problem shape, as measured

Measured across the benchmark corpus (29 PDFs, two register spreadsheets, one
DOCX). These counts are **detector output, not accuracy** — there is no
labelled money ground truth in the benchmark, so no precision or recall figure
can honestly be quoted for any approach yet. See
[Open gap: no ground truth](#open-gap-no-ground-truth).

| Locus | Amounts | Carries a currency marker |
|---|---|---|
| PDF running text (29 PDFs) | 658 | yes |
| AusTender contract register, `Value` column | 1,777 | **none** |
| GrantConnect award register, `Value (AUD)` column | 48,997 | **none** |
| Auditor-General financial tables (pp. 120–260 sample) | ~1,271 | **none** |

The finding that shapes the design: **the overwhelming majority of monetary
amounts in this corpus carry no currency marker at all.** In a 48,997-row grant
register recording $22.7bn of awards, exactly one `$` survives extraction — an
aggregate in the sheet preamble. Symbol-keyed detection alone reaches roughly
1.3% of the corpus's amounts.

A second measured fact: of the amounts that *are* marked, **97% carry a scale
suffix** (`$33.1 million`, `$78.7bn`, `($684.2m)`) across at least six
spellings. Scale handling is the dominant narrative form, not an edge case.

### What this implies

The useful axis is not *running text vs tables*. It is **whether the amount
carries its own evidence, or inherits it from a column**:

- **Self-evidencing** — a currency symbol, ISO code or currency word sits with
  the number. Recognised by pattern matching over text.
- **Column-evidenced** — a bare number whose money-ness comes from its column:
  the header (`Value`, `Value (AUD)`, `Approved Budget $m`) and, for
  spreadsheets, the cell's number format (`$#,##0.00`).

Both paths are required. The first is the smaller share of volume; the second
is where the corpus's money actually lives.

## Design principles

The extractor is optimised for **precision over recall**. Australia is not a
multilingual financial corpus: the overwhelming majority of genuine monetary
references are AUD, expressed as `$`, `A$`, `AU$`, `AUD`, `Australian
dollar(s)` or `cents`, with occasional USD, NZD, GBP and EUR. Supporting every
ISO currency remains worthwhile, but ranking and confidence should reflect
Australian document reality rather than international completeness.

The extractor is:

- locale-aware
- context-aware
- structure-aware
- confidence scored
- deterministic
- easily extensible

**It is not intended to identify arbitrary numbers. Every extraction must have
positive evidence that the number represents money.** For self-evidencing
amounts that evidence is inline; for column-evidenced amounts it is the column
header and number format. A bare number with neither is not an extraction.

## Currency model

Currencies are classified into confidence tiers rather than treated equally.

### Tier 1 — Australian (highest confidence)

```
$   AUD   A$   AU$
Australian dollar   Australian dollars
dollar   dollars   cent   cents
```

Australian government publications almost always use `$` to mean AUD unless
another currency has been explicitly established earlier in the document.

### Tier 2 — Common international

```
USD  NZD  GBP  EUR  JPY  CAD  SGD  CHF  HKD  CNY  RMB
```

These occur regularly in procurement, defence, treasury, trade and economic
reporting.

### Tier 3 — Full ISO 4217

Every ISO currency code is supported but assigned lower confidence unless
reinforced by surrounding context. **Three uppercase letters are never treated
as a currency unless they are members of the ISO 4217 list** — `ABC` and `XYZ`
are not currencies.

"Unless reinforced by surrounding context" is a **gate**, not just a
confidence penalty, because a number of ISO codes are ordinary English words
in capitals: `ALL` (Albanian lek), `TOP` (Tongan paʻanga), `TRY`, `PEN`, `CUP`,
`MAD`, `BOB`, `CAD`. Ungated, `TOP 10 projects were funded` is ten paʻanga and
`ALL 25 recipients` is Albanian lek — both shapes are common in government
reporting. A tier-3 code is admitted only when a currency symbol or financial
trigger word sits within ~48 characters; tier 1 and 2 codes stand alone. The
same asymmetry applies to column headers: a *parenthesised* code names the
column's currency (`Value (PGK)`), while a bare one is trusted only at tier
1/2 — so `ALL OTHER COMPENSATION ($)`, a standard heading in this document
class, resolves through its `$` rather than to Albanian lek.

## Number recognition

### Australian number format (default)

```
1        10        100
1,000    10,000    100,000
1.50     100.00    1,000.50
-100     -$100     AUD -50
.50      0.50
```

Australia does not use comma decimals. `1.000,50` is therefore **not**
interpreted as an Australian amount. Inferring locale automatically introduces
false positives for no benefit on this corpus.

### Optional international mode

Configurable, off by default. When enabled, accepts `1.000,50` and
`10.000.000,00`, normalising after locale detection.

## Currency indicators

**Symbols** — `$`, `A$`, `AU$`, `US$`, `NZ$`, `€`, `£`, `¥`, `₹`, `₩`, `₽`, `₿`,
including Unicode variants.

**ISO codes** — recognised only if in the ISO 4217 list.

**Currency words** —

- Australian: `dollar`, `dollars`, `cent`, `cents`, `Australian dollar(s)`
- International: `euro(s)`, `pound(s)`, `sterling`, `yen`, `yuan`, `renminbi`,
  `rupee(s)`, `peso(s)`, `franc(s)`, `won`, `ruble`/`rouble`, `dirham`

## Extraction patterns

Applied in strict priority order. Priority matters: it drives overlap
resolution.

| # | Pattern | Examples | Confidence |
|---|---|---|---|
| 1 | Symbol prefix | `$100`, `-$250`, `A$500`, `AU$5 million` | Very high |
| 2 | ISO prefix | `AUD 100`, `USD 50`, `EUR 1000` | Very high |
| 3 | ISO suffix | `100 AUD`, `500 USD` | High |
| 4 | Currency word | `100 dollars`, `250 Australian dollars`, `50 cents` | High |
| 5 | Symbol suffix | `100$`, `50€` | Medium |
| 6 | Magnitude expression | `$5 million`, `AUD 12 billion`, `$4.2bn`, `$500k` | Very high |
| 7 | Range | `$10–20 million`, `$100-$150`, `between $5 and $10` | Inherits |
| 8 | Approximate value | `about $100`, `~$50`, `up to $50,000` | Inherits |
| 9 | Accounting negative | `($100)`, `AUD (500)` | Context-gated |
| 10 | Implicit financial context | `The estimated cost is 250.` | Low |

### Magnitude suffixes

Supported: `k`, `m`, `b`, `bn`, `tn`, `million`, `billion`, `trillion`.

The bare single letters `m`, `b`, `k` are interpreted as multipliers **only**
when one of the following holds:

- preceded by a currency indicator
- followed by a currency indicator
- inside a recognised financial context

This gate exists to reject `100m road`, `50m radius`, `20m hose`.

### Ranges

Australian documents use ranges frequently. Both endpoints are extracted and
the relationship between them preserved, rather than collapsing to one value.

### Approximate values

`about`, `approximately`, `around`, `~`, `>`, `<`, `at least`, `up to`,
`no more than`. The qualifier is stored **separately** from the value — never
folded into it.

### Accounting negatives

Bracketed amounts are read as negative **only when accounting context is
detected**. Ungated, this pattern is the single worst source of false positives
in the corpus: an unanchored bracketed-number scan fired 656 times and was
almost entirely `s167(1)`, `(02) 6203 7300` and `(2018)`. Within a classified
money column, brackets *are* accounting negatives and the gate is satisfied by
the column itself.

### Implicit financial context

Attempted only after every explicit pattern fails, and scored lowest.
Trigger vocabulary is Australian-focused:

```
cost  price  fee  charge  payment  salary  income  wage  expense
budget  appropriation  grant  funding  allocation  revenue  profit
loss  compensation  claim  benefit  rebate  levy  fine  penalty
premium  excess  deductible  invoice  quote  estimate
contract value  replacement value  sum insured
```

**Measured calibration:** in narrative text this path is low precision on this
corpus and should default to off, enabled deliberately for recall experiments.
The same trigger vocabulary is high value when applied to *column headers*,
where it is the primary signal — see below.

## Australian false positives

This is where production systems fail. Candidates are rejected when embedded
in:

| Class | Examples | Note |
|---|---|---|
| Dates | `01/07/2025`, `2025-07-01`, `1 July 2025` | |
| Times | `10:30`, `14:45`, `0930 hrs` | |
| Phone numbers | `02 6123 4567`, `0412 345 678`, `1800 123 456` | |
| ABNs | `12 345 678 901` | High value on Australian datasets |
| ACNs | `123 456 789` | |
| Postcodes | `2600`, `3000` | Reject only where address context exists |
| Parcel / land identifiers | `Lot 5`, `DP12345`, `SP4567`, PID, LGA IDs | Common in government datasets |
| Legislative references | `Section 10`, `Clause 12`, `Schedule 3`, `Division 2` | |
| Incident numbers | `INC123456`, `F2024/12345`, `IR000456` | Common in emergency datasets |
| Measurements | `50m`, `100 km`, `20 kg`, `5 ha`, `10 MW`, `250 ML`, `40°C` | Metric suffixes are never monetary multipliers |
| Percentages | `10%`, `15.5%`, `100 percent` | |

## Column-evidenced amounts

The structural path, and the one carrying ~98.7% of the corpus's amounts. A
column is classified once; every cell beneath inherits the verdict.

**Evidence, strongest first:**

1. **Number format carrying a currency symbol** — `$#,##0.00`. Definitive.
   Available for spreadsheets as of the shipped prerequisite below.
2. **Money-vocabulary header** — the trigger list above applied to the header
   text (`Value`, `Value (AUD)`, `Amount`, `Approved Budget $m`), combined with
   the cells being predominantly numeric.
3. **Predominantly numeric cells** — supporting evidence only. It never
   promotes a column on its own: identifiers, counts and postcodes are
   numerically indistinguishable from money.

**Vetoes.** A header matching non-money vocabulary suppresses a column that
would otherwise be promoted on vocabulary alone, even when its cells are
numeric and thousands-separated: `postcode`, `abn`, `acn`,
`id`, `count`, `number`, `phone`, `year`, `date`, `percent`, `%`, `rate`,
`ratio`, `index`, `quantity`, `fte`, `headcount`, `latitude`, `longitude`.
Term matching must be **whole-word** — `age` is a veto term and `Average Cost`
must survive it. A veto does **not** override a header that declares its own
currency: `Grant Date Fair Value of Stock and Option Awards ($)` is a money
column containing the incidental word `date`, and vetoing it loses all five
amounts beneath it. The `($)` is the header describing itself, in the same
string as the veto term, so it wins; the overridden term is still recorded in
the column audit. A count column on the same page carries `(#)`, not `($)`,
and stays vetoed.

**Null markers.** Financial tables are sparse. `—`, `–`, `-`, `n/a`, `nil`,
`none` and similar are absent values and must be excluded from the numeric
fraction rather than counted against it. Counting them as non-numeric
suppresses genuine money columns: on the DocLayNet compensation-table fixture a
`Threshold ($)` column scores 50% numeric purely from em-dashes.

**Column scale.** Financial tables put the unit in the header (`$m`, `$'000`)
and leave the cells bare, so the header supplies the multiplier for every cell
beneath it. The `'000` form must not match the `000` inside a number already in
the header — `Grants over $10,000` declares no scale, and reading one there
multiplies every cell beneath it by 1,000. Where no header is recoverable — the common case for PDF financial
tables — bare cells are **left alone rather than guessed at**. Under-counting
is the correct failure mode here.

**Currency from header.** `Value (AUD)` states its currency; `Value` does not
and takes the document default (AUD).

## Confidence

Context influences confidence rather than gating extraction outright:

| Evidence | Confidence |
|---|---|
| `Funding of $10 million` | Very high |
| `$#,##0.00` number format on the column | Very high |
| Money header + numeric column | High |
| `Funding of 10 million` | High |
| `Funding of 10` | Medium |
| Bare `10` | Very low — not extracted |

## Normalisation

Both the original text and a canonical representation are stored. **The
original is never lost.**

```
Original:   $5.2 million
Canonical:  currency=AUD  value=5200000
Display:    $5.2 million
```

Values are exact decimals, not floats. Reconciliation and aggregation compare
values for equality, and float would make that comparison unreliable at scale
— summing 48,997 amounts accumulates error. The parquet column type is
`decimal128(38, 4)`.

## Output schema

Every extraction produces structured metadata:

```json
{
  "text": "$5.2 million",
  "value": 5200000,
  "currency": "AUD",
  "currency_source": "symbol",
  "modifier": null,
  "multiplier": "million",
  "negative": false,
  "confidence": 0.99,
  "span": [245, 257],
  "context": "Funding allocation was $5.2 million."
}
```

With an approximation qualifier:

```json
{
  "text": "approximately $500",
  "value": 500,
  "currency": "AUD",
  "modifier": "approximately",
  "confidence": 0.95
}
```

### As built

`*.money_spans.parquet` is that record, flattened, with the anchor made
explicit. One row per amount; `locus` discriminates which anchor group is
populated, and **exactly one group is non-null per row**:

| Locus | Non-null anchor columns |
|---|---|
| `narrative` | `text_source`, `start_char`, `end_char`, `page` |
| `table_cell` | `parent_elem_order`, `row`, `col` |
| `sheet_cell` | `sheet`, `row`, `col`, `elem_order` |

Beyond the JSON above the row also carries `evidence` (`p1`–`p10` for the
narrative patterns, `number_format` / `header+numeric` / `header_currency` for
the column path), `range_group` + `range_role` (which link a range's two
endpoints — the JSON record has no way to express the relationship the design
requires be preserved), and `column_id` (the classified column a cell
inherited from; null when the cell was self-evidencing).

`*.money_columns.parquet` is the second sidecar: one row per column
considered, money or not, with the evidence that decided it — header text,
number format, numeric and null fractions, veto term, currency, scale, and how
many cells it yielded. The column path decides ~98.7% of the corpus's amounts
off a single per-column verdict, and with no labelled ground truth yet that
verdict needs to be reviewable rather than implicit in the spans it produced.

Two departures from the pipeline sketch below, both consequences of the
[placement](#placement-in-womblex) decision:

- **Step 1 does no text rewriting.** Unicode and whitespace normalisation are
  already the `normalise` / `spellfix` overlays' job, and re-doing them inside
  this op would put spans in a private coordinate space that no longer joins to
  enrichment mentions or chunks. The op selects an existing element-text layer
  (`processing.text_source`) and records which one on every narrative row.
- **Step 6's "surrounding sentence" is a capped character window**
  (`context_chars`, default 160), not a parsed sentence. The offsets recover
  anything wider.

One invariant falls out of the same decision and is worth stating, because
violating it fabricates data rather than merely missing some. The reassembled
narrative joins elements with `\n\n`, so **no pattern may match across two line
breaks**: whitespace inside a pattern spans at most one newline, and a range's
separator none at all. Without that, `Payment of $100` and `-$200 was made` —
two unrelated paragraphs, possibly two unrelated table rows — bind into a
single `$100–$200` range. This mirrors the newline rule the PII regexes already
follow ([CLAUDE.md](../CLAUDE.md)). A magnitude suffix *may* sit across one
wrap (`$5\nmillion`), because PDF text layers wrap mid-phrase constantly.

## Processing pipeline

1. **Pre-processing** — preserve original text and character offsets; Unicode
   normalisation; standardise whitespace while maintaining span mappings;
   detect document structure (tables, headers, footers, OCR artefacts).
2. **Candidate generation** — apply the extraction patterns in strict priority
   order (1–10 above).
3. **Overlap resolution** — collect all candidate spans, rank by pattern
   priority, confidence and span length, retain the highest-quality
   non-overlapping match.
4. **False-positive filtering** — exclude candidates embedded in the Australian
   false-positive classes above.
5. **Normalisation** — convert numeric strings to canonical values; expand
   multipliers; resolve accounting negatives; infer default currency (AUD) only
   where Australian document convention supports it.
6. **Contextual enrichment** — infer qualifiers; associate ranges and
   comparative operators; record the surrounding sentence or table cell.
7. **Structured output** — return original and normalised representations with
   confidence, provenance, offsets and contextual metadata.

## Placement in Womblex

An **annotation op**, in the mould of `quality` — offline, API-free, no
ordering dependency on enrich, and it **never rewrites element or chunk text**.

**Input is the extraction parquet**, not the source files: `*.elements.parquet`
plus its `*.table_cells.parquet` sibling. The op does not open a PDF or a
workbook. Extraction is already sufficient; a second reader would be a parallel
extraction path, which this design explicitly rejects.

**Three loci, two coordinate spaces.** Per the offset-space rule in
[decisions.md](decisions.md), these are not mixed:

| Locus | Anchor |
|---|---|
| `narrative` | character offset into the reassembled narrative — the same space enrichment mentions use, so they join, and map to chunks as `graph_refresh` does |
| `sheet_cell` | `(sheet, row, col)` |
| `table_cell` | `(parent_elem_order, row, col)` on the `table_cells` sidecar |

The narrative offsets index whichever element-text layer was selected
(`processing.text_source`: `elements` / `normalised` / `spellfix`), so that
choice is recorded alongside the spans and the space stays self-describing.

**Output** is a `*.money_spans.parquet` sidecar per batch (plus the
`*.money_columns.parquet` verdict audit), joinable on `source_hash`, with a
per-stage `CheckpointManager` like every other stage.

**As built:** `womblex money --shards <dir>`, config under `money:`.
`process/money.py` (self-evidencing patterns) and `process/money_columns.py`
(column classification) are pure cores over strings and cell lists;
`process/money_stage.py` walks the shard directory, applies the selected
text-source overlay, and writes both sidecars; `store/money_output.py` owns
the schemas. A classified money column owns its cells — the column supplies
currency, scale and the accounting-negative gate — while cells in every other
column, vetoed ones included, are still scanned for *self-evidencing* amounts:
a `$1,200.50` cell carries its own evidence whatever its header says.

## Decisions

### Hand-written patterns, not a parsing library

Evaluated: `price-parser`, `money-parser`, `quantulum3`, `pint`, spaCy `MONEY`,
Presidio, LayoutLMv3. **Decision: no new dependency.**

- **`price-parser` / `money-parser`** — measured wrong on the dominant form.
  `$2m` → 2, `$8.7 billion` → 8.7, `AUD$21.9 million` → 21.9, and `(6,550.1)`
  loses its sign. Silently wrong by 10⁶–10⁹ on 97% of narrative amounts. They
  encode e-commerce assumptions: prices are small, unscaled and always marked.
- **`quantulum3`** — the strongest candidate, and the closest call. It extracts
  *and* normalises scale in one pass, handles worded magnitudes natively, and
  is externally maintained. Rejected because: it is a **units** library whose
  competence spans an enormous physical-unit space of which currency is one
  corner, so on an audit report it surfaces page counts, percentages, years and
  section numbers that we then filter away — and that filter is where precision
  is actually decided, and is ours to write either way. It returns **floats**,
  and converting back to Decimal cannot recover precision the float never had.
  It fails the "thin adapter" test in [CLAUDE.md](../CLAUDE.md), which applies
  when the library's full surface *is* the feature; here we would use a few
  percent and suppress the rest. And it only addresses the self-evidencing
  ~1.3% — it has nothing to say about a bare `50000` in a column.
  **Revisit if the corpus grows to legislation and contracts**, where worded
  amounts ("a sum not exceeding five hundred thousand dollars") and penalty
  units become common; both scored zero on the current corpus.
- **`pint`** — solves unit *dimensionality and conversion*. Currency is not a
  physical dimension, the conversion analogue is exchange rates (which pint
  does not carry and this corpus does not need), and it is float-first. What we
  need is a scale lookup and exact arithmetic.
- **spaCy `MONEY`** — unvalidated; the model could not be obtained in this
  environment, so no claim is made about its behaviour. Architecturally it
  returns a labelled span, not a value, so normalisation remains ours
  regardless, and it has no purchase on bare cells.
- **Presidio** — the recognizers live in `presidio-analyzer`, which is *not*
  currently a dependency (`presidio-anonymizer` is). Architecturally it is
  regex + context returning spans, with no normalisation layer. Money is also
  not PII, and routing it through that stack merges two concerns the codebase
  deliberately separates.
- **LayoutLMv3** — needs page images and fine-tuning on labelled key-value
  data. The op's input is parquet, by which point layout is gone; and there are
  no money annotations to fine-tune on. The dependency is the cheap part, the
  labels are the expensive part.

The residual hand-written surface is small and stable: a ~12-entry scale table
and the Australian pattern set. Australian government money vocabulary is
closed and slow-moving. **The risk in this feature is not narrative parsing —
it is deciding whether a bare column of numbers is money, and no candidate
library addresses that.**

### Header continuation rows are folded into the header

Measured on the ANAO Major Projects Report: PDF financial tables wrap their
header across two lines — `Approved` on row 0, `Budget $m` on row 1 — and the
extractor declares only the first a header row. The unit and the money
vocabulary both live in the second, so the column read as a nameless run of
bare numbers and was left alone, losing all 27 approved-budget amounts.

`fold_header_continuation` absorbs **one** leading body row into the header,
and only when that row is non-numeric text while the rest of the column is
numeric enough to be a data column — so a genuine text data row is never eaten.
This is a header-*reading* fix, not a relaxation of the deferred "no
recoverable header" case below: the header is present in the table, just not
where the extractor said it was.

### Cross-validation by re-reading sources: rejected

An earlier design recounted amounts by independently re-reading each source
file and comparing multisets of normalised Decimals. Rejected: it constitutes a
second extraction path, duplicating work extraction already performs correctly,
which is contrary to the objective of this feature. Reconciliation, if
reintroduced, must operate on extracted output.

Two constraints recorded for whenever that is revisited: comparison must be on
**multisets of values, not counts** (counts hide compensating errors — one
amount dropped and another duplicated nets to zero); and pages with no text
layer must be reported **`unverifiable`, never `mismatch`** (38% of benchmark
PDFs have no text layer, and reporting those as failures would bury real ones).

### Deferred

- **Bare cells in PDF financial tables with no recoverable header.** The
  measured header-recovery rate is 41 of 273 tables; the rest put the unit in a
  caption or the row above the grid. Promoting a numeric column with no header
  is where percentage columns leak in. Left un-extracted.
- **Implicit financial context in narrative** — specified above, default off,
  pending a precision measurement.
- **Worded amounts** (`two million dollars`) — zero occurrences measured;
  revisit with a legislation/contract corpus.
- **Penalty units** — zero occurrences across all 29 benchmark PDFs. These are
  audit, FOI and budget documents, not legislation.

## First real-document run

Four benchmark fixtures, run through the real pipeline (extract → money) rather
than synthetic shards. Still **not** a precision/recall measurement — there is
no labelled set — but every span was checked by hand against the source.

| Fixture | Amounts found | Checked against |
|---|---|---|
| ANAO Major Projects Report 2020–21 (PDF, 30pp) | 42 narrative + 53 table_cell | every `$` in the transcript |
| ANAO, same report as a text transcript | 42 narrative | 44 `$` in the file |
| DFAT PBS 2025–26 (DOCX) | 47 narrative + 12 table_cell | source `python-docx` table dump |
| DocLayNet `dense_text_548` (scanned page, OCR) | 1 of ~35 | ground-truth transcript |
| FUNSD `82200067_0069` (transcript) | 0 | ground-truth transcript |

Findings that changed the code are in the [Decisions](#decisions) section
below. The rest, as measurements:

- **Recall on marked narrative amounts is complete on the two ANAO runs.** Of
  44 `$` characters in the transcript, 42 are amounts and all 42 are
  extracted; the other two are the `Budget $m` / `Amount $b` column headers,
  which are unit declarations, not amounts.
- **The `Approved Budget $m` column reconciles three ways.** Its 25 project
  amounts sum to $78,699.2m, matching both the table's own total row and the
  narrative's independently written "$78.7 billion". That is the strongest
  correctness signal available without a labelled set, and it exercises the
  whole column path — header scale, cell parsing, exact decimals.
- **FUNSD's zero is correct.** Its `AMOUNT RECEIVED FROM VENDOR` column is
  empty in the source; the numbers on the page are unit counts and rep counts.
- **Two header-marker defects, both found on one scanned page.** That page
  carries `Threshold ($)` and `Threshold (#)` — dollars and unit counts,
  distinguished only by the marker. The money op honoured neither correctly:
  `Grant Date Fair Value ... ($)` was vetoed on the incidental word `date`,
  and `Threshold (#)` was promoted to money because the header tokeniser
  dropped `#` entirely and matched the vocabulary term `threshold`. The first
  lost 5 real amounts; the second invented 5. Both are fixed, and `#` is now a
  token character precisely because a financial table marks a count column the
  same way it marks a money one.

- **Scanned money tables are unreachable today, and that is the largest
  measured gap.** DocLayNet `dense_text_548` is a proxy-statement
  *Grants of Plan-Based Awards* page: four money columns headed
  `Threshold ($)` / `Target ($)` / `Maximum ($)` / `Grant Date Fair Value …
  ($)`, about 35 amounts. The op recovers **one** — the single footnote where
  OCR preserved a `$`. OCR captures nearly every digit correctly, but
  `_layout_blocks_and_tables` detects the table region (YOLO confidence 0.96)
  and emits it as a `[TABLE]` placeholder block with **no cells**: its
  ``tables`` list is never populated on any code path. So no scanned page
  yields a `table` element, the column path has nothing to classify, and every
  amount on it is a bare number the narrative path is right to decline.

  Fed the same page's real structure, the column path recovers **30 of 30**.
  The op is ready; OCR table-cell reconstruction is the missing piece, and it
  is extraction-side work tracked as item #17 in
  [steering.md](steering.md#table-cell-reconstruction-on-ocrd-pages-17) — not
  a change to this op. This
  also answers the benchmark gap noted below — money loss through OCR was
  unmeasured, and on this page it is ~97%, none of it attributable to the
  detector.

  (Two further amounts are lost to OCR reading `$15.37` as `s15.37`. The op is
  right to decline those: `s15` is precisely the legislative-reference shape
  the false-positive table blocks, so accepting `s` as a currency symbol would
  trade two recoveries for a large class of false positives. That fix belongs
  in OCR or a cleaning op, and it is a rounding error next to the 30.)
- **Plain-text records cannot use the column path.** The ANAO transcript
  flattens the same `Approved Budget $m` table into narrative, where the
  amounts are bare numbers with no column to inherit from — 27 amounts
  recovered from the PDF, 0 from the transcript of the same pages. This is the
  designed refusal, and it is a reason to prefer the structural source when a
  corpus offers both.
- **No financial tables in the DFAT DOCX.** All 51 of its tables are
  performance-measure or glossary tables (confirmed against the source with
  `python-docx`); its money is narrative, and 47 amounts were recovered there.
  Zero money columns is correct, not a miss.

## Open gap: no ground truth

**There is no labelled money data in the benchmark.** Every count in this
document is detector output — what a pattern found — not a measurement of
correctness. No precision or recall figure can be honestly quoted for this
design, or for any alternative, until a labelled set exists.

The proportionate next step is a bounded labelled sample drawn from the
*parquet* — a few hundred candidate strings across the three loci, labelled
money / not-money with expected value. That is a small artefact, not a
subsystem, and it is what would let the `quantulum3` decision above be settled
by measurement rather than argument, and serve as a regression baseline.

Fixture coverage is otherwise good for spreadsheets and native PDFs. The
scanned-money gap is now partly closed: DocLayNet `dense_text_548` is a scanned
page of money columns, and the measured loss is ~97% — attributable entirely to
OCR producing no table cells, not to the detector (see
[First real-document run](#first-real-document-run)). Of the 29 PDFs, 11 have
no text layer and none of those contain monetary amounts, so OCR money loss
across the PDF set specifically remains unmeasured.

## Shipped prerequisite

`ingest/spreadsheet.py` read cells with pandas (`dtype=str`), which discards
both the cell's number format and its stored type. Every `sheet_cell` element
landed with `value_type="text"` and `number_format=None`, despite
`ELEMENT_SCHEMA` carrying columns for both. A second read-only openpyxl pass
now populates them; the pandas read stays authoritative for `value`, so the
verbatim contract ("1,234" stays "1,234") is unchanged.

This matters because it restores the strongest available signal for the
column-evidenced path. GrantConnect's award register carries `$#,##0.00` on
48,997 cells whose text is a bare `50000` — the only unambiguous currency
marker in the entire workbook, previously discarded at the extraction boundary
where no downstream stage could recover it. AusTender's `#,##0.00` carries no
symbol, so that register still depends on its `Value` header; the two together
are why both evidence sources are specified rather than either alone.
