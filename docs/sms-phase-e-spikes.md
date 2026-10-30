# SMS uplift Phase E — spike results (for review)

This report covers Phase E of [plan-sms-uplift.md](plan-sms-uplift.md): E1 (element evidence
and fingerprint) and E2 (statement identity). It records what was measured, and its last
section proposes a second iteration built on those results. Nothing in it is a decision yet. No entry has been written to [decisions.md](decisions.md) or
[decisions-dead-ends.md](decisions-dead-ends.md), and Phase F is unchanged.

The spikes ran against `60bdfea`. Phases C and D have landed on main since. Each finding below
that depends on the code was re-checked against main as of `2e216bb`, and still holds unless it
says otherwise.

## Summary

- **One defect in the current code.**
  - `money_spans` is labelled sensitivity `none`, but it carries document text (see Defect).
- **Hazards a build of E1 must handle.** None of these is a defect today.
  - `conform_shard` would reject every older extraction shard.
  - The records-ingest writer would write the new columns null, without an error.
  - Nothing in the test suite reads an older elements shard.
- **E1 cost.**
  - As specified: elements 4.9x and table cells 9.3x larger.
  - Reduced variant (order-free binary fingerprint, narrative span as two integers): elements
    3.9x, or 1.44x without fingerprints on spreadsheet cells. Write time rose by under 0.1% of
    a run. `content_digest` is unchanged.
- **E2.**
  - The planned key is not an identity: 43 of 58 statements share it with another statement
    in the same version.
  - A key anchored to the containing element's order-free fingerprint made no false matches.
  - With a fallback to the planned key, it matched 18 of 24 statements.

## Defect: `money_spans` is labelled `none` but carries document text

- `ROLE_SENSITIVITY` in `store/contract.py` labels `money_spans` (and `money_columns`) `none`.
  [contract.md](contract.md) does the same.
- The contract defines `none` as "no document text at all".
- `money_spans.context` holds about 80 characters of verbatim narrative either side of each
  amount, and `text` holds the amount as written.
- In the E2 spike output, `context` held unmasked names. For example, the row for `$12,500`
  read "…innie Wombat, inspected the service on 3 March 2025 and recorded a shortfall of
  $12,500…".
- A consumer that hands `none` files onward, which is what the footer key is for, would pass on
  unmasked text.
- `money_columns.header_text` is document text too: column headings.
- The defect is not fixed here. Correcting it changes a contract label and its documentation,
  so it is a decision for review.

## Method

- **Input.** All 18 synthetic fixtures (`fixtures/synthetic/`) were extracted once with
  `configs/example.yaml` and cached. The baseline and every variant wrote identical
  `ExtractionResult`s:
  - 3,112 elements, 2,706 of them `sheet_cell`
  - 2,263 table cells
  - 5 form fields

  Extraction took 52.6 s.
- **Size.** Parquet bytes per role, at the writer's own settings (zstd level 3).
- **Write time.** The median of 40 to 60 calls to `write_results` over the whole batch, on one
  machine. Absolute times are indicative; compare them within a table only.
- **E2 pair.** Two versions of one synthetic DOCX decision notice, built for the spike: 12
  paragraphs and a 3-column table, with known edits between them.
  - Money ran as built.
  - Entity links and PII ran on a canned `enrichment_entities` sidecar. It holds the exact
    surface forms of 3 people, 2 organisations and 2 addresses, so no paid enrichment call
    was made.
- **Code.** The prototype code lived on two local branches. Neither was pushed, and both have
  been deleted. Only this report is committed.

## E1 — first pass (as specified)

The prototype added two columns to `ELEMENT_SCHEMA`, `TABLE_CELLS_SCHEMA` and
`FORM_FIELDS_SCHEMA`:
- **`fingerprint`.** SHA-256 hex over `content_digest._element_payload`. Table cells were
  hashed over row, col, value, spans and type; form fields over index, name, value and type.
- **`evidence`.** A struct of the eleven `EVIDENCE_FIELDS`. Text elements were located to the
  span, with their narrative offsets from `chunker.element_spans`. Every other kind was
  located to the element.

| role | base | +fingerprint (hex) | +fingerprint (binary 32) | +evidence | both (hex) |
|---|---|---|---|---|---|
| elements | 36.8 KB | 4.37x | 3.84x | 1.55x | 4.92x |
| table_cells | 11.2 KB | 8.87x | 7.66x | 1.44x | 9.31x |
| form_fields | 2.8 KB | 1.25x | 1.19x | 2.35x | 2.60x |

- **Write time.**
  - The batch took 148 ms against 43 ms, about 20 µs per written row (5,380 element, cell and
    field rows).
  - Against the 52.6 s extraction, that is +0.2% of a run.
  - In a profile of 10 writes, `json.dumps` of the fingerprint payload was the largest single
    cost: 0.56 s of 1.82 s. `content_digest` already serialises the same payload.
- **Growth by shape** (both columns, hex):
  - narrative PDF/DOCX elements: 2.19x
  - their table cells: 9.33x
  - spreadsheet elements: 6.16x
  - scanned-image elements: 1.50x

  A SHA-256 adds about 32 incompressible bytes to each row. Growth therefore follows the
  number of rows, not the volume of text.
- **Fingerprint basis.** `_element_payload` includes `order`, so inserting one element changes
  the fingerprint of every element after it. E2 measures the cost: 5 of 24 statements matched
  with `order` in the basis, against 16 of 24 without it.
- **Evidence struct.** Six of its eleven fields repeat columns already on the row (`elem_order`,
  `page`, `bbox`, `sheet`, `row`, `col`). `anchor_level` is fixed by kind. `evidence_check` is
  always `holds` at write time, because the row is its own source. The only new information
  is the narrative span (`char_start` / `char_end`).

## E1 — second pass (reduced variant)

What changed from the first pass:
- **Order-free basis.** The fingerprint is taken over `_element_payload` without `order`. Cells
  are hashed without row and col, and form fields without their index.
- **Binary storage.** `binary(32)`, and `binary(16)` measured beside it.
- **Narrative span.** Stored as two int32 columns, not a struct.
- **One serialisation per element.** The fingerprint and `content_digest` are computed from the
  same JSON.
- **Old shards.** `conform_shard` back-fills the new columns as null.
- **Records writer.** `ingest/records.py` writes the same columns.

| variant | write (batch) | elements | table_cells | form_fields |
|---|---|---|---|---|
| baseline | 45 ms | 36.8 KB | 11.2 KB | 2.8 KB |
| fp 32 B, cells fingerprinted | 71 ms | 143.9 KB (3.91x) | 44.9 KB (4.00x) | 3.3 KB |
| fp 32 B, no cell fingerprint | 59 ms | 143.9 KB (3.91x) | 11.2 KB (1.00x) | 3.3 KB |
| fp 16 B, cells fingerprinted | 70 ms | 94.5 KB (2.56x) | 29.6 KB (2.63x) | 3.2 KB |
| fp 16 B, no cell fingerprint | 62 ms | 94.5 KB (2.56x) | 11.2 KB (1.00x) | 3.2 KB |

Write time rises by 14 to 26 ms for the batch, under 0.1% of the 52.6 s extraction.

Elements, broken down by component and by document shape (32 B fingerprint):
- **Narrative span.** The two span columns alone cost 1.09x.
- **Sheet cells.** Fingerprinting every element except `sheet_cell` costs 1.44x. Fingerprinting
  every element costs 3.91x, because `sheet_cell` makes up 87% of the rows here.
- **Narrative PDF/DOCX.** Elements grow 1.79x (+40 B a row) and table cells 3.99x (+15 B a row).
- **Spreadsheets.** Elements grow 4.85x (+34 B a row).
- **Cell fingerprints.** An order-free cell fingerprint depends only on the cell's content, so
  repeated values repeat the hash, and it compresses. Table cells grew 4.00x here, against
  7.66x for the first pass's binary fingerprint, which included row and col.

Checks on the second pass:
- `content_digest` is byte-identical on all 18 fixtures, and the `test_default_digest` pins pass.
  Sharing the serialisation is safe.
- An elements, table_cells or form_fields shard written without the new columns reads through
  `conform_shard`, with the columns null.
- `mypy src/` is clean. `ruff` reported one fixable issue in the spike code.
- Full suite: 2,838 passed, 108 skipped, 0 failed.

## E1 — hazards a build must handle

These are not defects today: each one appears only once the columns are added.

1. **Older shards stop reading.**
   - On main, `conform_shard` has no `_BACKFILL` entry for `elements` or `form_fields`, and
     only `bbox` for `table_cells`.
   - With the columns added and no backfill, it rejects every older shard of the three roles:
     "missing columns … schema bump without compat shim?".
   - The first-pass branch still passed the full suite (2,838 passed, 0 failed). No test reads
     an elements, table-cells or form-fields shard written before a column; `test_output.py`
     covers only an older `chunks` shard. A build needs that test.
2. **The records writer would go silently wrong.**
   - `ingest/records.py` builds its own element rows (`_element_row`).
   - With the schema changed and the writer left alone, `from_pylist` writes the new columns
     null, without an error, on every records-ingest shard.
3. **`store/output.py` sits near the line cap.**
   - It was 710 lines at `60bdfea` and reached 740 in the spike, against the 750-line cap.
   - D0 has since brought it to 693.
   - The row building still belongs in a new module beside it.
4. **The build crosses a layer.** The narrative span needs `chunker.element_spans`, so
   `store/` would import `process/` lazily. `store/entity_links_output.py` already imports
   `process/evidence` that way.
5. **Contract bump.**
   - The columns are additive, so the bump is minor: 3.2 to 3.3 on current main.
   - It also needs the null backfill for three roles and the change to the records writer.

## E1 — findings

1. **A fingerprint marks content, not identity.**
   - 26 of the 3,112 rows repeat another row's fingerprint:
     - all 15 `page_break`s hash the same: they carry no content, and `page` is outside the
       payload
     - the column headings the FOI index repeats on every page account for 12 rows, in
       three groups of four
     - a heading the spacing PDF repeats accounts for 4 rows
   - An identity needs a fingerprint plus an occurrence ordinal.
2. **Overlays.**
   - The fingerprint is taken over the verbatim elements layer.
   - On the synthetic set, `normalise` changed 0 of 370 text elements and `spellfix` changed 1,
     so the fixtures cannot measure how often the layers diverge.
   - By construction, an overlay rewrite leaves the extraction fingerprint describing text that
     a `text_source` consumer does not read.
   - Two options for review: overlay rows carry the fingerprint of the element they rewrote,
     which doubles as a staleness check; or each layer carries a fingerprint of its own text.
3. **Coverage.**
   - Every one of the 370 text elements is located to the span.
   - Every other kind is located to the element only: table, form, image, page_break,
     sheet_meta, and all 2,706 `sheet_cell`s.
   - 28 of the 2,263 table cells (1.2%) have a bbox of their own. The rest can only point at
     the table's.
   - `FieldEntry` has no geometry, so a form field can only point at its form element.
4. **Sensitivity.**
   - A fingerprint of a short value, such as "NSW" or a person's name, can be recovered by
     hashing candidates.
   - The extraction roles are `raw` already, so a fingerprint there changes nothing.
   - The evidence reference also rides on `money_spans`, which is labelled `none` (see Defect).
     A location fingerprint on the evidence reference (Phase F) would therefore put hashes of
     document text into a `none` file, unless the fingerprint is keyed.

## E2 — statement identity

Edits between v1 and v2:
- a paragraph inserted at the top
- one paragraph deleted
- one paragraph moved
- one amount changed
- one paragraph reworded with the same amounts and names
- one paragraph extended with an amount that already appears elsewhere
- a table row inserted, and one cell changed

Counts: 28 statements in v1 and 30 in v2 (money 12/15, PII 8/7, links 8/8), with 24 true pairs.

- **Truth.** A statement's counterpart has the same stage, the same source paragraph (or table
  row label), the same text, and the same occurrence within that paragraph or row. A changed
  amount has no counterpart.
- **Locus per sidecar.** `money_spans.locus`, `pii_spans.content_type` and
  `entity_links.candidate_kind`.
- **Text per sidecar.** `text` for money and PII, `mention_text` for links.
- **Joining.** Rows were joined on key equality. Rows that shared a key were paired in document
  order, which is what a plain join does.

| key | unique within its version | matched | false match | missed |
|---|---|---|---|---|
| K0 the plan's key: stage + collection + `source_relpath` + locus + text | 15/58 | 15 | 9 | 9 |
| K1 K0 + ordinal in the document | 58/58 | 15 | 9 | 9 |
| K3 K0 + 24 characters of context either side | 58/58 | 12 | 0 | 12 |
| K2b K0 + first-pass fingerprint of the containing element (includes `order`) + ordinal | 58/58 | 5 | 0 | 19 |
| K5 K0 + second-pass fingerprint of the containing element (the table's, for table cells) + ordinal | 58/58 | 14 | 0 | 10 |
| K6 as K5, with the cell's own fingerprint for table cells | 58/58 | 16 | 0 | 8 |
| K7 K6 first, then K0 on the leftover rows where K0 is unique on both sides | 58/58 | 18 | 0 | 6 |

K7 by stage: money 7 of 10, PII 5 of 7, links 6 of 7, with no false matches.

Findings:
1. **K0 is not an identity.**
   - 43 of the 58 rows share their key with another row in the same version: the same amount,
     or the same name.
   - Joining on it paired 9 of the 24 statements wrongly.
   - A document ordinal (K1) makes the key unique but does not fix the pairing, because inserts
     and moves shift the ordinal.
2. **Anchoring to the containing element's order-free fingerprint stops false matches.**
   - No variant built that way made one.
   - The cost is a miss for every statement in a paragraph that was edited at all, even when
     the statement itself did not change.
3. **The fallback recovers two of those misses.**
   - Each of the 6 that remain lies in an edited paragraph whose text also appears in another
     leftover row on one side or the other: the inserted paragraph, another edited paragraph,
     or the old value of the changed amount.
   - K0 is therefore not unique among the leftovers, and the fallback leaves the row unmatched
     rather than guess.
4. **Table cells need the cell's own fingerprint.** The table element's fingerprint (K5) changes
   when any cell or row changes. That lost both table statements that had a counterpart.
5. **E2 depends on E1's basis.** With `order` in the fingerprint, matching fell to 5 of 24.
6. **A renamed or moved file would miss everything.** This follows by construction and was not
   measured: `source_relpath` is in every key, so all of a renamed or moved file's statements
   miss. The version chain follows the same rule (`store/document_versions.py`).
7. **The sample is small: one document pair, 58 rows.** The numbers rank the key shapes against
   each other. They are not rates to rely on.

## Open questions for review

- **Defect.** Relabel `money_spans` (and `money_columns`) `raw` in `ROLE_SENSITIVITY` and
  [contract.md](contract.md), or move the text out of them.
- **E1 scope.**
  - the fingerprint basis (order-free, over the verbatim layer)
  - the width (32 or 16 bytes)
  - whether table cells and `sheet_cell` rows carry a fingerprint
  - whether the narrative span ships as two columns
  - whether to keep the evidence struct, which repeats existing columns, or drop it
- **Overlays.** Overlay rows carry the base element's fingerprint, or each layer carries its own.
- **Sensitivity.** Whether any fingerprint may leave the `raw` files, and if so whether it is
  keyed.
- **E2 key.**
  - The plan's key, or the two-pass key, which depends on E1.
  - Whether to measure over more document pairs, or a scanned or OCR pair, before deciding.

## Second iteration — revised approach

This is a proposal and has not been run. It carries forward what the first iteration measured
and closes the gaps found when the results were reviewed. It answers none of the open questions
above. Each step names the question it feeds.

### What carries forward

- **Baseline variant.** The second-pass reduced variant:
  - order-free basis over the verbatim layer
  - one serialisation shared with `content_digest`
  - the narrative span as two int32 columns
  - null backfill in `conform_shard`
  - the records writer updated

  The first pass's order-inclusive basis and hex storage are not measured again. The figures
  above stand for them.
- **Still measured as variants, because they are still open:**
  - width (32 or 16 bytes)
  - fingerprints on table cells
  - fingerprints on `sheet_cell` rows
  - the evidence struct beside the two span columns
- **E2 key shapes carried forward:**
  - K0, the plan's key, as a control
  - K6 and K7

  K1, K2b and K5 are dropped: on the first sample, K6 matched more statements than each of them
  with no more false matches.

### Method changes

- **Reproducible figures.**
  - The spike branch is pushed and kept until the clean-up step.
  - Each figure names the script and the commit it ran at.
  - The first iteration's figures could not be re-run during review.
- **Basis on every figure.** Each count names the fingerprint it was taken over: basis and width.
- **Duplicates counted two ways.**
  - rows in a duplicate group
  - rows beyond the first in each group

  The first report's figure of 26 is the second count, but its breakdown lists the first count
  (31).
- **Truth written first.** Each E2 pair's truth table is written to a file before any key runs.
  For each statement it records the counterpart (or none) and the kind of edit.
- **Ratios per shape, not only per batch.** `sheet_cell` makes up 87% of the synthetic batch's
  rows, so a whole-batch ratio mostly measures spreadsheets.

### E1 — measurements

1. **Collisions per kind and width.**
   - Duplicate fingerprints per kind, at 32 and 16 bytes, for elements, table cells (order-free
     cell hash) and form fields.
   - The first iteration did not count table-cell duplicates, and K6 relies on table-cell
     fingerprints.
   - Feeds: width, cell fingerprints.
2. **Fingerprint plus occurrence ordinal.**
   - Prototype the ordinal: the nth element in the document with that fingerprint.
   - Measure its uniqueness, and how many ordinals shift under the E2 edits.
   - Feeds: the E2 key.
3. **Overlays.**
   - The synthetic set cannot measure divergence: normalise changed 0 of 370 text elements and
     spellfix changed 1.
   - On the spike branch only, generate text in the shapes those stages rewrite:
     - footer-glyph spacing
     - runs of inline whitespace
     - digit-for-letter OCR confusions
   - Run both options over it:
     - Overlay rows carry the base element's fingerprint. Measure how many rows are stale after
       a base change, and the size cost.
     - Each layer carries its own fingerprint. Measure the size cost, and how often an overlay's
       fingerprint differs from its base.
   - Feeds: overlays.
4. **Keyed fingerprint.**
   - HMAC-SHA-256 beside plain SHA-256: write time and size.
   - What a key needs: where it lives, and what rotating it breaks.
   - Show that hashing candidate values recovers short values from a plain fingerprint but not
     from a keyed one.
   - Feeds: sensitivity.
5. **Older shards and the records writer.**
   - A test that reads an elements, table-cells and form-fields shard written before the new
     columns.
   - A records-ingest shard written by the updated writer, with each new column checked
     non-null wherever it should be.
   - Feeds: contract bump size (hazards 1, 2 and 5).
6. **Sensitivity audit of every role.**
   - Every role in `ROLE_SENSITIVITY` is checked for columns that hold document text.
   - The first iteration checked the money roles only.
   - Feeds: the defect.

### E2 — measurements

1. **More pairs, one per shape:**
   - the first iteration's DOCX pair, unchanged, as a control
   - a native PDF pair
   - a spreadsheet pair, to measure statements at the `sheet_cell` locus (the first iteration
     had none)
   - a scanned pair: the same pages rendered at two resolutions, so the OCR text differs where
     the content does not
   - a renamed pair and a moved pair, with v2 under a different `source_relpath`
   - a pair split across two collections

   A script on the spike branch generates the pairs. Nothing is added to `fixtures/synthetic/`.
2. **Key shapes:**
   - **K0, K6 and K7**, as before.
   - **K8.** K6 first, then K3 (24 characters of context either side), then K0 where K0 is
     unique on both sides. K3 made no false match in the first iteration and may recover
     statements in edited paragraphs.
   - **K9.** K8 without `source_relpath`, matched within the collection. Measures what the
     renamed and moved pairs gain, and what collisions across documents it adds.
   - **K10.** K8 without `collection`. Measures whether `collection` separates anything.
3. **Reporting.**
   - Matched, false match and missed for each key, broken down by stage and by edit kind, as
     counts.
   - The scanned pair is reported on its own, because its unchanged statements may still
     differ through OCR.

### Out of scope for this iteration

- **Fixing the defect.** It changes a contract label, so it is a decision for review. Step 6
  only measures how far it extends.
- **Real documents.** Measuring size on a womblex-benchmark corpus would give ratios outside the
  synthetic mix. It runs against third-party data, so it waits for the reviewer's go-ahead. It
  is an option, not a step.

### Exit criteria

The second report answers each open question above with a figure, or says why it could not.
The clean-up steps in [plan-sms-uplift.md](plan-sms-uplift.md) follow review, as before.
