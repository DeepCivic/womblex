# Monetary amount recognition

The canonical reference for the `money` annotation op: how Womblex recovers
monetary amounts from its extraction output, normalises them to exact values,
and records them as a joinable sidecar.

`womblex money --shards <dir>` writes `*.money_spans.parquet` +
`*.money_columns.parquet` per batch (`process/money*.py`,
`store/money_output.py`).

## Scope and naming

"Money amounts", not the legal sense of *currency* (point-in-time / in-force
dates). The two were conflated early and are unrelated problems. The legal
sense is a separate and cheaper win — `DateInfo.effective/expiry` and
`Quote.amending` are already returned by Kanon-2 and then discarded, because
`ENTITY_SCHEMA` (`store/enrichment_output.py`) persists only
`person|location|term|external_document`. Dates survive as a bare `date_count`
int; the same applies to emails, websites, phones, id_numbers and quotes.
That work is tracked separately and is not in scope here.

The op is named `money` rather than `currency` to keep the distinction
visible.

## The problem shape

Across the benchmark corpus (29 PDFs, two register spreadsheets, one DOCX),
**the overwhelming majority of monetary amounts carry no currency marker at
all**. Register columns (AusTender `Value`, GrantConnect `Value (AUD)`) and
financial tables print bare numbers: a 48,997-row grant register recording
$22.7bn of awards keeps exactly one `$` through extraction, an aggregate in the
sheet preamble. Symbol-keyed detection alone reaches roughly 1.3% of the
corpus's amounts. Of the amounts that *are* marked, **97% carry a scale
suffix** (`$33.1 million`, `$78.7bn`, `($684.2m)`) across at least six
spellings, so scale handling is the dominant narrative form, not an edge case.

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
USD  NZD  GBP  EUR  JPY  CAD  SGD  CHF  HKD  CNY
```

These occur regularly in procurement, defence, treasury, trade and economic
reporting. `RMB`, not an ISO code, is read as an alias of `CNY`.

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
10 000   1 500 000            (space-grouped)
1.50     100.00    1,000.50
-100     -$100     AUD -50    −$5.2m   (true minus sign)
.50      0.50
```

Australia does not use comma decimals. `1.000,50` is therefore **not**
interpreted as an Australian amount. Inferring locale automatically introduces
false positives for no benefit on this corpus.

**Space-grouped thousands** are the Australian Government Publishing Service
convention and appear verbatim in legislative penalties. They are not a recall
nicety: `Penalty: $10 000` matched as `$10` and stored a value wrong by 10³,
which is the failure this op exists to prevent. A group is exactly three digits
and may not be followed by another — otherwise `$5 2020` binds an amount to the
year beside it.

A leading **true minus sign** (U+2212) is a negative, as PDF text layers emit
it as readily as the ASCII hyphen; reading `−$5.2 million` as positive inverts
the sign rather than missing the amount. The en dash is deliberately excluded:
it is the range separator, and admitting it would turn `$10–20m` negative.

A number carrying a **second dotted group** is declined rather than partly
read. `$3.219.3m` — a real ANAO typo for `$3,219.3m` — yields `$3.219` if the
readable prefix is taken, three dollars for a $3.2 billion project budget.
Repairing the typo would be a guess; declining is the only honest outcome.

### Optional international mode

Configurable (`international_numbers`), off by default. When enabled, also
accepts `1.000,50` and `10.000.000,00`; there is no locale detection — each
number's separators are read from its own shape.

## Currency indicators

**Symbols** — `$`, `A$`, `AU$`, `US$`, `NZ$`, `CA$`, `C$`, `S$`, `HK$`, `NT$`,
`¢`, `€`, `£`, `¥`, `₹`, `₩`, `₽`, `₿`, plus prefixed forms `$AUD`, `$AU`,
`$A`, `$USD`, `$US`, `$NZD`, `$NZ`, including Unicode/full-width variants
(`﹩`, `＄`).

The **symbol-then-letters** order (`$US655.5m`, `$A250,000`, `$AUD1.2m`) is the
Australian reporting convention, not a typo of `US$`: the ANAO Major Projects
Report writes foreign-military-sales case values that way throughout. Without
it the `$` matches alone, the letters read as the start of the next word, and
the amount is lost entirely.

`¢` is a sub-unit: `50¢` is half a dollar, not fifty of them.

**ISO codes** — recognised only if in the ISO 4217 list.

**Currency words** —

- Australian: `dollar`, `dollars`, `cent`, `cents`, `Australian dollar(s)`
- Other dollar-denominated: `US dollar(s)`, `United States dollar(s)`,
  `New Zealand dollar(s)`, `Canadian dollar(s)`, `Singapore dollar(s)`,
  `Hong Kong dollar(s)`
- International: `euro(s)`, `pound(s)`, `sterling`, `yen`, `yuan`, `renminbi`,
  `rupee(s)`, `peso(s)`, `franc(s)`, `won`, `ruble`/`rouble`, `dirham` — `peso(s)`
  and `franc(s)` are ambiguous across countries, so they resolve to a
  money-marked span with **no** currency code (`"10 pesos"` → value 10,
  currency `None`) rather than guessing an ISO code

## Extraction patterns

The `#` column below is a pattern catalogue index (also the internal `pN`
evidence code, except 8), **not** the overlap-resolution order. When two patterns match
overlapping text, `money.py`'s `_PRIORITY` table decides the winner, in this
order (lowest number wins first): accounting-negative (`p9`) and range
(`p7`, pre-claimed before overlap resolution runs, so it never actually
competes) tie for highest priority, then symbol-prefix/magnitude
(`p1`/`p6`), then ISO-prefix (`p2`), then currency-word/worded-amount tied
(`p4`/`p11`), then ISO-suffix (`p3`), then symbol-suffix (`p5`), then
implicit-context (`p10`) last. Concretely: `($100)` resolves to `-100` via
the accounting-negative pattern, not `100` via the symbol-prefix pattern
sitting inside it — accounting negatives must outrank the symbol pattern
they enclose.

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
| 9 | Accounting negative | `($100)`, `$(100)`, `AUD (500)` | Context-gated |
| 10 | Implicit financial context | `The estimated cost is 250.` | Low |
| 11 | Worded amount | `two million dollars`, `fifty cents`, `half a million dollars` | High |

### Worded amounts

Prose and legal drafting write the amount out where a table prints digits, and
the digit-keyed patterns cannot see those at all — there is no number to anchor
on. Pattern 11 parses the spelled-out number (units, teens, tens, `hundred`,
the scale words, and the fraction forms `half a million` / `three quarters of a
million` / `one and a half million`) to an exact `Decimal`.

**The currency word is the gate**, exactly as it is for pattern 4. A worded
number on its own is not money, and in this corpus that is the shape that
actually occurs: `more than one million Australians overseas` is the only
worded-number phrase in the benchmark DOCX, and it is a headcount. Requiring
the currency word costs nothing there and is what keeps the pattern from
counting every spelled-out number in a document.

The articles are read the way English uses them: `a million dollars` is one
million (the article stands in for the number), while a bare `a dollar` is not
an amount at all — `a dollar figure`, `a dollar amount` are the common uses. A
*leading* `of` or `and` belongs to the sentence rather than the amount and is
trimmed from the span; an `and` inside the number (`one hundred and fifty`) is
part of it.

**What the parser declines**, because each of these parses arithmetically into
a number the document never wrote — and a wrong value is worse than a missing
one:

| Phrase | Naive reading | Why it is declined |
|---|---|---|
| `between ten and twenty dollars` | 30 | a range; `and` joins a hundred or a scale (`one hundred and fifty`), never two plain numbers |
| `ten–twenty dollars`, `ten-twenty dollars` | 30 | the same range, hyphenated. The dash forms are *phrase* separators here so the pair is consumed and declined whole, rather than leaving `twenty dollars` behind |
| `nineteen fifty dollars` | 1,950 | a year: a tens word opens its group or follows a hundred |
| `in million dollars`, `thousand dollars` | 1,000,000 / 1,000 | a table's unit declaration. A number or an article is what makes it an amount |
| `one thousand million` | 10⁹ | scale words must strictly decrease |
| `zero dollars`, `nil dollars` | 0 | states an absence, not an amount |

### Restatement

Drafting writes one amount twice, once in words and once in digits:

```
The contract value is one million dollars ($1,000,000).
```

Both readings of that bracket were wrong. It is the accounting-negative shape,
so the sentence yielded **−1,000,000**; and once worded amounts are recognised
it also yielded the same money **twice**. A parenthesised amount that restates
the one immediately before it is neither: the restating half is dropped and the
primary kept, in whichever order the pair is written.

The two halves must differ in form — one worded, one in digits. Two bracketed
*digit* amounts of equal value are the ordinary financial-statement shape
(`$5,000 (5,000)` is this year and last), and collapsing those would discard a
real negative.

### Magnitude suffixes

Supported: `k`, `thousand(s)`, `m`, `mn`, `million(s)`, `b`, `bn`,
`billion(s)`, `t`, `tn`, `trillion(s)`.

The bare single letters `k`, `m`, `b`, `t` are interpreted as multipliers
**only** when preceded or followed by a currency indicator. Implicit financial
context does not license one (`The budget cost is 100m.` yields no span).
This gate exists to reject `100m road`, `50m radius`, `20m hose`.

### Ranges

Australian documents use ranges frequently. Both endpoints are extracted and
the relationship between them preserved, rather than collapsing to one value.

### Approximate values

`about`, `approximately`, `around`, `circa`, `approx.`/`approx`, `~`, `>`,
`<`, `>=`, `<=`, `at least`, `up to`, `no more than`, `not more than`,
`no less than`, `not less than`, `at most`, `more than`, `less than`,
`greater than`, `in excess of`, `over`, `under`, `nearly`, `almost`,
`not exceeding`/`not to exceed`/`exceeding` (the drafting-language family
also seen in "a sum not exceeding …" worded amounts), `up to a maximum of`,
`to a maximum of`, `a maximum of`, `a minimum of`, `in the order of`,
`of the order of`. The qualifier is stored **separately** from the value —
never folded into it.

### Accounting negatives

Bracketed amounts are read as negative **only when accounting context is
detected**. The symbol sits inside or outside the bracket depending on house
style — `($100)` and `$(100)` mark the same thing. Ungated, this pattern is the single worst source of false positives
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
| Incident numbers | `INC123456`, `IR000456` | Two to four capitals before the digits; a one-letter file reference (`F2024/12345`) is not blocked |
| Measurements | `50m`, `100 km`, `20 kg`, `5 ha`, `10 MW`, `250 ML`, `40°C` | Metric suffixes are never monetary multipliers |
| Percentages | `10%`, `15.5%`, `100 percent` | |

## Column-evidenced amounts

The structural path, and the one carrying ~98.7% of the corpus's amounts. A
column is classified once; every cell beneath inherits the verdict.

**Evidence, strongest first:**

1. **Number format carrying a currency symbol** — `$#,##0.00`. Decisive once
   at least 70% of present cells are numeric (`numeric_fraction_min`), unless a
   veto fires first. Spreadsheets only ([number-format prerequisite](#number-format-prerequisite)).
2. **Money-vocabulary header** — the trigger list above applied to the header
   text (`Value`, `Value (AUD)`, `Amount`, `Approved Budget $m`), combined with
   the cells being predominantly numeric.
3. **Predominantly numeric cells** — supporting evidence only. It never
   promotes a column on its own: identifiers, counts and postcodes are
   numerically indistinguishable from money.

**Vetoes.** Checked before any promotion — even a currency number format; a
`%` number format also vetoes (`percent_format`). The header veto terms:
`postcode`, `abn`, `acn`,
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
| `Funding of ten million dollars` | High |
| `Funding of 10 million` (implicit context, no currency marker) | Low (0.35, flat — no distinction by magnitude suffix) |
| `Funding of 10` (implicit context, no currency marker) | Low (0.35) |
| Bare `10` | Very low — not extracted |
| `ten million` with no currency word | Not extracted |

Implicit-context confidence is a flat 0.35 regardless of whether a magnitude
suffix is present, and 0.35 is below the default `min_confidence` (0.5) — so
under default settings the two implicit-context rows above are not extracted
either. `implicit_context=True` alone is not enough; `min_confidence` must
also be lowered below 0.35.

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

Beyond the JSON above the row also carries `evidence` (`p1`–`p7`, `p9`–`p11`
for the narrative patterns — pattern 8 has no code; its qualifier lands in
`modifier`, `number_format` / `header+numeric` / `header_currency` for
the column path), `range_group` + `range_role` (which link a range's two
endpoints — the JSON record has no way to express the relationship the design
requires be preserved), and `column_id` (the classified column a cell
inherited from; null when the cell was self-evidencing).

`*.money_columns.parquet` is the second sidecar: one row per sheet column and
per column of a table with a declared header row (headerless tables get none),
money or not, with the evidence that decided it — header text,
number format, numeric and null fractions, veto term, currency, scale, and how
many cells it yielded. The column path decides ~98.7% of the corpus's amounts
off a single per-column verdict, and table-cell recall is unmeasured, so that
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
2. **Candidate generation** — apply the extraction patterns (1–11 above).
3. **Overlap resolution** — collect all candidate spans, rank by `_PRIORITY`
   (not catalogue order), then span length, then confidence, and retain the
   highest-ranked non-overlapping match.
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
`process/money.py` (self-evidencing patterns), `process/money_words.py` (the
worded-amount parser behind pattern 11), `process/money_numbers.py` (number
reading and false-positive blocking, split out when the pattern set outgrew the
750-line cap; its names are re-exported from `money.py`) and
`process/money_columns.py` (column classification) are pure cores over strings
and cell lists;
`process/money_stage.py` walks the shard directory, applies the selected
text-source overlay, and writes both sidecars; `store/money_output.py` owns
the schemas. A classified money column owns its cells — the column supplies
currency, scale and the accounting-negative gate — while cells in every other
column, vetoed ones included, are still scanned for *self-evidencing* amounts:
a `$1,200.50` cell carries its own evidence whatever its header says.

## Number-format prerequisite

Spreadsheet extraction (`ingest/spreadsheet.py`) records each `sheet_cell`'s
`number_format` and `value_type` from a read-only openpyxl pass; the pandas
read (`dtype=str`) stays authoritative for `value`, so the verbatim contract
("1,234" stays "1,234") holds.

That number format is the strongest available signal for the
column-evidenced path. GrantConnect's award register carries `$#,##0.00` on
48,997 cells whose text is a bare `50000` — the only unambiguous currency
marker in the entire workbook. AusTender's `#,##0.00` carries no symbol, so
that register still depends on its `Value` header; the two together are why
both evidence sources are specified rather than either alone.
