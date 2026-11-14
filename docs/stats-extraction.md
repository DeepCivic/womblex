# Statistical figure recognition

The canonical reference for the `stats` annotation op: how Womblex finds the
figures a document states about named measures, the period each refers to, and
the words that qualify it, and records them as a joinable sidecar.

`womblex stats --shards <dir>` writes `*.stat_spans.parquet` per batch
(`process/stats*.py`, `store/stats_output.py`).

It sits beside [`money`](money-extraction.md) and follows the same rules:
offline, no model and no API, annotation only (element and chunk text are never
rewritten), reading the extraction parquet and never the source files. It
depends on no other stage, so it runs on a corpus that is never enriched.

## Design principles

1. **The library holds no measure.** What counts as a statistic is a corpus's
   decision. A corpus declares its measures, with the phrases that name each, in
   config or in a vocabulary file the config points at. Womblex matches them and
   knows nothing about any one of them.
2. **A number is a figure only beside a measure.** The vocabulary is the positive
   evidence, as a currency marker is for money. A sentence that names no measure
   yields no figure.
3. **Context lives in the headings.** A document usually states the place and
   period its figures refer to in a heading, not in the sentence. Every row
   carries the headings it sits under, and a figure whose sentence names no
   period takes the one its headings state.
4. **Say what was read, not what was guessed.** A period is given calendar
   bounds only where the text fixes them. A number that could be a year or a
   count is left out. Every figure says where its period came from.
5. **Every span is located and checked.** Rows carry the shared evidence
   reference ([contract.md](contract.md)) and are labelled `holds`, `fails` or
   `unchecked` before the file is written. Nothing is refused after launch.

## Vocabulary

```yaml
stats:
  enabled: true
  vocabulary: vocab/regional-profile.yaml   # optional; merged with the keys below
  measures:
    - id: median_income            # written to measure_id
      label: Median income         # written to measure_label (the id where empty)
      terms: [median income, median weekly income]
    - id: population
      terms: [population, estimated resident population]
      unit: persons                # the unit of a bare number beside this measure
  cues:
    source: [regional insights]    # extra phrases per cue type
```

A vocabulary file has the same two keys, `measures` and `cues`. A term matches
whole words, without regard to case, with any run of spaces between its words,
and never across a line break. Where two terms overlap the longer wins
(`estimated resident population` over `population`).

The sidecar footer records `womblex.stats_vocabulary`, a SHA-256 over the merged
measures and cues, so a file names the vocabulary that produced it whichever
file that vocabulary was read from. A vocabulary that is missing or malformed
refuses the stage before any batch is opened.

With no measure declared the stage still writes periods and cues, warns, and
writes no figure.

## Reference periods

| `period_kind` | Written as | `period` |
|---|---|---|
| `financial_year` | `2019–20`, `2019/20`, `FY20`, `FY2019-20`, `2019-20 financial year` | `2019-20` |
| `quarter` | `2025-Q1`, `Q1 2025`, `March quarter 2025`, `March quarter` | `2025-Q1` (`Q1` with no year) |
| `half` | `2026-H1`, `H1 2026`, `first half of 2026` | `2026-H1` |
| `calendar_year` | `CY2025`, `calendar year 2025` | `CY2025` |
| `year` | `in 2021`, `2021 Census` | `2021` |
| `month` | `March 2025`, `2025-03` | `2025-03` |
| `date` | `30 June 2021` | `2021-06-30` |
| `year_range` | `2019–21`, `2019–2020` | `2019-21`, `2019-2020` |

A financial year is the Australian one, 1 July to 30 June, and `FY20` names the
year it ends in. A quarter is a calendar quarter named by its last month, as the
ABS names them; `April quarter` is not a period. `period_start` and `period_end`
are null where the text does not fix them: a quarter written without its year,
and a `year_range`, which may be calendar or financial years (`2019–2020` is
read as a financial year only when the text says `financial year` or `FY`).

A bare four-digit number is a year only beside a cue (`in`, `during`, `since`,
`until`, `by`, `year`, `Census`). In a heading or a table header a lone `2021`
is the period and is read as one. Lowercase `may` is never a month.

## Figures

A figure is a number in the same sentence as a measure. How close it must be
depends on the number:

- A number that states its own unit (`$1,242`, `4.5%`, `2.1 percentage points`)
  is a quantity on its face. It takes the nearest measure in its sentence.
- A unit-less number (`43,516`, `24 million`) must sit next to the measure:
  after it within a few words (`the population was 43,516`, `Population:
  43,516`), or directly before it (`33 people died`, `24 million hectares`). A
  period in between is ignored (`the population in 2021 was 43,516`); any other
  number, or clause punctuation, breaks the link.

Each figure row carries:

| Column | Holds |
|---|---|
| `value` | The number, `decimal128(38, 4)`, with any `multiplier` (`million`) already applied |
| `unit` | The ISO currency, `percent`, `percentage_points`, or the measure's declared unit |
| `rate_per` | `week`, `year` and so on, for `$1,242/week` or `per week` |
| `measure_id`, `measure_label` | The measure it was read beside |
| `period`, `period_kind`, `period_start`, `period_end` | The period it refers to |
| `period_source` | Where the period came from: `sentence`, `heading`, or for a table `column` / `row` |
| `calculation` | The sentence's calculation cues, `;`-joined |
| `source` | The sentence's source cue, as written |
| `status` | `forecast` when the sentence holds a forecast cue, else `stated` |

Money amounts are read by the money recogniser, so every spelling it reads
(`$33.1 million`, `A$5`, `two million dollars`) is a figure here too.

Numbers that are deliberately not figures: a bare `2021` in running text (a year
or a count; the text does not say); a number that opens a line before a
capitalised word (a paragraph number); a number run on to a full stop, colon,
quote or bracket (a footnote marker); a day of the month; a number before
`Figure`, `Table`, `Chapter`, `Appendix` or `Box` (a contents page); and the
false-positive classes the money op blocks, other than measurements (phone
numbers, ABNs, legislative references). A measurement is a figure:
`271,000 hectares` beside a measure whose term is `hectares`.

## Cues

| `cue_type` | Says | Built in |
|---|---|---|
| `calculation` | how the figure was derived | `per 100,000` (`per_100000`), `per capita`, `share of`, `in real terms`, `nominal terms`, `seasonally adjusted`, `trend terms`, `chain volume`, `current prices`, `annualised`, `age-standardised` |
| `source` | where it came from | `Source: …` (to the end of the line), `cat. no. 6202.0`, `Australian Bureau of Statistics` / `ABS`, `according to` |
| `gap` | that a figure is absent | `not available`, `n.a.`, `not published`, `n.p.`, `no data`, `not collected`, `not applicable`, `suppressed` |
| `forecast` | that it is not an observation | `forecast`, `projected`, `projection`, `expected to`, `anticipated` |

Every cue is a row of its own (`kind = cue`, with `cue_type` and the canonical
`cue`), naming the nearest measure in its sentence if there is one. A figure
also carries its sentence's calculation and source cues, and its `status`.
A corpus adds phrases per type under `stats.cues`.

## Headings

Elements record no heading level, so the path is built from order alone:

- the **title run**: the headings that open the document, before any other text;
- the **current run**: the most recent heading, with any headings directly
  after it.

`heading_path` is the title run followed by the current run, outermost first,
and `heading_elem_orders` their elements. A figure with no period in its
sentence takes the first period of the innermost heading that states one.

The place a figure refers to is not resolved here. It is in `heading_path` as
text, and a place name in a heading is a mention the `link` stage resolves
against a gazetteer: join the two sidecars on `source_hash` and
`heading_elem_orders`.

Page headers, footers and signatures are not scanned.

## Tables

A table is read both ways, and a cell is a figure when it is one number and
nothing else:

- a **row** whose label cell (the first column) names a measure: the column
  header states each figure's period;
- a **column** whose header names a measure: the row's label cell states the
  period.

Where neither axis states a period the headings do. A cell that is a gap marker
and nothing else (`n.p.`) is a `gap` cue for the measure. Table rows are located
by `elem_order`, `cell_row` and `cell_col`, with offsets into the cell's value.

Spreadsheet cells (`sheet_cell` elements) are not read.

## Output

`*.stat_spans.parquet`, one row per span, joinable on `source_hash`:

| `kind` | A row is |
|---|---|
| `period` | a reference period, wherever it appears |
| `measure` | a vocabulary phrase |
| `cue` | a calculation, source, gap or forecast cue |
| `figure` | a number stated beside a measure |

`locus` is `narrative` (offsets index the reassembled narrative under
`text_layer`, the space money spans and enrichment mentions use) or `table_cell`.
The narrative layer follows `processing.text_source`; `stats.text_source`
overrides it for this stage alone, and a declared layer missing from any batch
refuses the stage before it writes.

The file's sensitivity is `raw`: each row carries the text around its span
(`context`) and the headings above it.

## Limits

- **No ground truth exists for this stage.** Its rules were shaped against
  synthetic fixtures and inspection of one long public report; no recall or
  precision figure is claimed.
- A unit-less number next to a measure is taken as its figure even where it
  counts something else (`the population of the 3 towns`).
- A footnote marker run on to a number with no separator (`2020` then `88`
  extracted as `202088`) is read as that number. The text is verbatim from
  extraction and is not repaired here.
- Sentences are split on terminal punctuation followed by a capital letter. A
  cue binds to the figures of its own sentence only, so a `Source:` line under a
  paragraph is a cue row but is not carried onto that paragraph's figures.
- A measure named only in a heading does not make the numbers beneath it
  figures.
