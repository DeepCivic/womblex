# SMS Phase E adjustments — plan

*Status: proposed. It plans the builds that follow from the review of
[sms-phase-e-spikes.md](sms-phase-e-spikes.md). Each piece is removed when it is built, and this
document is deleted when the last one lands.*

## Context

The review of the Phase E spikes gave direction on three items:

1. **Money text.** The text on money rows is there to support coordinates, so the context of a
   financial figure can be retrieved. Suggestions are welcome on how to keep that intent.
2. **One fingerprint per figure.** A figure is one element: a paragraph, table, image, form or
   sheet. Row order and the interpretation of what is inside a figure are already handled by
   other stage features: `cell_row` / `cell_col` on the evidence reference, the money column
   classification, and table markdown.
3. **Overlays, the cheap option.** Overlay rows carry the fingerprint of the element they
   rewrote.

Everything else waits for access to womblex-benchmark (see Deferred):
- fingerprints outside the `raw` files
- the statement identity key
- the second spike iteration

The merge rules in [plan-sms-uplift.md](plan-sms-uplift.md) apply to every merge here. Each merge
also records its decision in [decisions.md](decisions.md) and its change in
[CHANGELOG.md](../CHANGELOG.md).

## 1. Money context through coordinates

**Today.**
- `money_spans.context` holds a window of verbatim text around each amount. For narrative it is
  `context_chars` characters (default 160) from `process/money.py` `context_for`. For a cell it
  is the cell text, up to 200 characters.
- `money_spans` also carries the full evidence reference: `elem_order`, `page`, `bbox`, `sheet`,
  `cell_row`, `cell_col`, `char_start`, `char_end` and `text_layer`.
  [money-extraction.md](money-extraction.md) already says "The offsets recover anything wider."
- The context window therefore duplicates what the coordinates can already retrieve. It is also
  where the unmasked names in the spike output came from.

**Suggestion: retrieve the context from the coordinates, and stop storing it.**
- Remove `context` from `MONEY_SPANS_SCHEMA`.
- Add a reader, `money_context(base, rows, width)`, that rebuilds the window from the elements
  shard through the evidence reference:
  - a narrative row slices the reassembled narrative in the row's `text_layer`, through
    `process/text_overlay.py`, so the offsets index the same text they were found in
  - a cell row returns its cell; it can also return the row's label cell and the column
    heading, which give more context than today's truncated cell text
- The reader lives in a new `process/money_context.py`. `store/` does not import `process/`,
  and `process/money_stage.py` stays focused on detection.
- `context_chars` stays in config as the reader's default width, so existing configs still
  validate.
- `store/document_detail.py` can attach the rebuilt window to its money rows. That route already
  requires `read_raw`.
- Reading context then needs the elements, which are `raw`. Access to the text follows the
  label of the file that holds it.
- Removing the column is a contract **major** bump (3.x to 4.0). An older file still reads:
  `store/money_output.py` projects each shard onto the current schema, so its `context` column
  is dropped on read.

**What remains after the change.**
- `text` (the amount as written) stays: `evidence_check` and `verify-evidence` check each row
  against it.
- `money_columns.header_text` (column headings) stays: the column verdict audit is read against
  it.
- Both are document text, so under the contract's current definition the money roles are still
  not `none`. The label needs a decision:

| Option | Costs | Buys |
|---|---|---|
| A. Relabel both money roles `raw` | Money files are no longer shareable as `none` | The label is true under the definition as it stands. Can land first, on its own, before the column removal |
| B. Narrow the definition of `none` to allow matched values and column headings, never running text | Changes a definition every role is read against. A column heading could still hold a name | Money files stay shareable once `context` is gone |

[contract.md](contract.md) has no rule for the bump size of a changed sensitivity label. The
merge that changes one adds the rule.

**Merges.**
- **1a. Label** (option A or B). Change `ROLE_SENSITIVITY` and [contract.md](contract.md), and
  test the footer each role now carries. Small; independent of the fingerprint work.
- **1b. Context through coordinates.**
  - Remove the column and add `process/money_context.py`.
  - Update `store/document_detail.py`, with a test that an older file still reads.
  - Update [money-extraction.md](money-extraction.md) and the module tables.
  - Tests: a narrative row and a cell row round-trip to their window, under the verbatim layer
    and under an overlay.

## 2. One fingerprint per element

**Shape.**
- One nullable fingerprint column on `ELEMENT_SCHEMA`.
- `TABLE_CELLS_SCHEMA` and `FORM_FIELDS_SCHEMA` get none: a table's and a form's fingerprint
  covers its cells and fields, because `_element_payload` already includes them.
- **Sheets.** A spreadsheet sheet is a figure, but extraction writes it as one `sheet_meta`
  element followed by one `sheet_cell` element per cell. Following the direction, the sheet's
  fingerprint sits on its `sheet_meta` row and is taken over the sheet's cells. `sheet_cell` rows
  carry none. **This reading needs confirming in review**: the alternative is a fingerprint on
  every `sheet_cell`, which cost 3.91x against about 1.44x in the spike.
- The fingerprint is computed in a new module, `store/element_fingerprint.py`
  (`store/output.py` is at 693 lines), from the same serialisation as `content_digest`. The
  spike confirmed that sharing it leaves `content_digest` byte-identical.

**Cost, from the spike's second pass** (32 bytes, every element except `sheet_cell`):
- elements: about 1.44x
- table cells and form fields: unchanged
- write time: under 0.1% of a run

The build merge re-measures these over the synthetic set.

**Build (one merge).**
- the column, and the module that computes it
- `conform_shard` null backfill for `elements`
- `ingest/records.py` writes the column, with a test that it is not null
- a test that reads an elements shard written before the column (none exists today)
- contract minor bump
- [extraction.md](extraction.md), [contract.md](contract.md), and the module tables in CLAUDE.md,
  [architecture.md](architecture.md) and [project-structure.md](project-structure.md)

**Still to settle before the build merge.** None of these depends on the benchmark:
- **Basis.** The spike measured an order-free basis (`_element_payload` without `order`).
  With `order` in the basis, inserting one element changes every fingerprint after it.
- **Width.** 32 or 16 bytes.
- **Narrative span columns** (`char_start` / `char_end` on elements). They cost 1.09x in the
  spike. They are separate from the fingerprint and can stay out of this merge.

**Known limits, recorded rather than fixed.**
- An image element's payload is its alt text and meta, not its pixels. Images with neither can
  share a fingerprint, as all 15 page breaks did in the spike.
- A fingerprint marks content, not identity. Repeated headings share one.
- In the spike, a table statement anchored to the table's fingerprint missed whenever any cell
  of that table changed (K5). Under this direction that is accepted: the cell is located by
  `cell_row` / `cell_col`, not by a fingerprint of its own. The statement identity work
  measures how often it happens.

## 3. Overlay rows carry the base element's fingerprint

**Shape.**
- One nullable `base_fingerprint` column on `NORMALISED_TEXT_SCHEMA` and on the spellfix text
  overlay schema.
- Each is copied from the extraction element's fingerprint when the overlay row is written.
- Spellfix, chained over normalise, still copies the extraction element's fingerprint, not
  normalise's.

**Staleness check.**
- An overlay row whose `base_fingerprint` differs from its element's current fingerprint was
  written over different text.
- `process/text_overlay.py` counts and warns about such rows when it applies a layer. It never
  refuses once a stage has started (CLAUDE.md, "Don't").
- A row written before the column reads as unchecked.
- An open question for this merge: should `require_overlays` also refuse a stale layer before
  launch, as it refuses a missing one?

**The trade made.** The fingerprint describes the verbatim text. A `text_source` consumer that
wants a fingerprint of the text it actually read does not get one.

**Build (one merge, after item 2).**
- the columns
- the copy in `process/normalise_stage.py` and `process/spellfix_stage.py`
- the count and warning in `process/text_overlay.py`
- tests for a fresh layer, a stale layer and an older layer
- contract minor bump
- [contract.md](contract.md)

## Order

1. **1a.** Fixes the live labelling defect. It depends on nothing.
2. **1b.** Independent of the fingerprint work.
3. **Item 2.**
4. **Item 3.** Needs item 2's column.

## Decisions needed before building

- the money label (option A or B), and the bump size for a changed label
- the sheet reading in item 2
- the fingerprint basis and width

## Deferred until benchmark access

- **Fingerprints outside the `raw` files**, keyed or not. Phase F's location fingerprints on
  the evidence reference would put them in the money files, so this also depends on item 1's
  label.
- **The statement identity key (E2).** Item 2 rules out cell-level fingerprints, so the keys to
  measure change (see the spike report's second iteration).
- **The second spike iteration**, narrowed to these items and run over benchmark documents as
  well as the synthetic set.
