# Entity linking

The canonical reference for the `link` stage: how Womblex resolves the mentions
in a document to the entities of a reference register, and records each
resolution as a joinable sidecar.

`womblex link --shards <dir> --config <yaml>` writes `*.entity_links.parquet`
per batch (`link/`, `store/entity_links_output.py`). The stage is offline: no
model, no API.

## Registers

A register is a table of canonical entities, read from **CSV or Parquet**. The
library knows no register: a corpus declares which columns play which role.

```yaml
linking:
  enabled: true
  reference:
    path: registers/abn/20260601Public01_records.parquet
    id_col: abn                      # -> entity_id
    name_col: non_individual_name    # -> canonical_name
    entity_type: organisation
    match_fuzzy_cols: [non_individual_name]
    names:                           # every other name the entity goes by
      path: registers/abn/20260601Public01_names.parquet
      id_col: abn
      name_col: name_text
  references:                        # further registers, tried in order
    - path: registers/asgs/localities.parquet
      id_col: SAL_CODE_2021
      name_col: SAL_NAME_2021
      entity_type: place
      match_fuzzy_cols: [SAL_NAME_2021]
      candidate_kinds: [location]
      single_word: cued
```

- `format` is `csv` or `parquet`, and read from the file's suffix when omitted.
- A Parquet register is read column-wise: only the declared columns are loaded,
  so a geometry or payload column costs nothing. The example reads the two
  files `womblex ingest-abn` writes, and a locality table as `ingest-geo`
  writes one.
- A declared column the file does not have refuses the stage before any batch
  is written.

### One entity, many names

An entity has a legal name, trading names and aliases. Names accumulate from
three places, and every one is scored as a name of the one entity:

1. each `match_fuzzy_cols` column of its row;
2. each further row that repeats its id (a long-format register); such a row's
   `match_exact_cols` are kept too, so an entity matches at every address it lists;
3. each row of the `names` table that carries its id.

A `names` row whose id the register does not hold is ignored and counted in a
warning. `alias_table` is separate: a curated `alias,entity_id` override that
always wins, for an entity known under a name no register carries.

### Several registers

`reference` and `references` together are the registers of a run, tried in the
order declared; a mention links to the first that matches it. Each register may
name the enrichment kinds it answers (`candidate_kinds`), so a place gazetteer
sees only `location` mentions and an entity register only `corporate` ones.
`entity_type` on the row says which register a link came from.

## Where mentions come from

`linking.candidate_source` chooses.

### `enrichment` (default)

The mentions the enrich stage wrote (`*.enrichment_entities.parquet`), selected
by kind. Each is matched in this order:

1. **alias** — the normalised mention is in the alias table (confidence 1.0);
2. **address_exact** — for an `address` mention, normalised equality with the
   register's `match_exact_cols` (1.0);
3. **name_fuzzy** — the best `difflib` similarity over every name of every
   entity, at or above `name_threshold`;
4. otherwise **unmatched**, still written, `matched = false`.

**Narrowing.** Scoring a mention against every name is exact, and too slow for
a register of 10⁵ entities. A register with more than `narrow_above` names
(default 2000) is indexed by character trigrams, and a mention is scored only
against the `narrow_to` names (default 64) that share the most trigrams with it.
Trigrams, not words, because mentions are OCR output: `earty leaming` shares no
word with `early learning`. A trigram held by more than 5% of the names (`pty`,
`ltd`) is skipped unless the mention has no rarer one. Narrowing can lose a
match the full scan would have found; it cannot invent one, since every name it
returns is scored by the same function against the same threshold. A register
at or below `narrow_above` is scored in full.

### `text`

The registers' own names, found in the narrative by exact phrase match on
normalised words, longest name first. Nothing is enriched and nothing leaves the
machine, so this is how a gazetteer or a register is applied to a corpus that
must stay offline. The stage reads `*.elements.parquet` and needs no enrichment
sidecar.

A name is a mention only where the text writes it as a proper noun (its first
and last word capitalised). It may wrap a line; it never crosses a paragraph
break. Matching is exact, so a name with OCR damage is missed.

**One-word names.** `Young` and `Grant` are towns, surnames and words. Each
register sets `single_word`:

| Value | A one-word name found in the text is |
|---|---|
| `accept` (default) | linked |
| `skip` | never linked, and not written |
| `cued` | linked only beside a cue, and never where it reads as a person's name |

Under `cued`, `match_detail` records what decided it:

| `match_detail` | Example | Linked |
|---|---|---|
| `cue:<word>` | `in Young`, `Young Shire`, `City of Young` | yes |
| `cue:state` | `Young NSW`, `Young, 2594` | yes |
| `cue:heading` | the name in a heading, with no person cue | yes |
| `person:honorific` | `Mr Young` | no |
| `person:name` | `Peter Young`, `Grant Thornton` | no |
| `uncued` | `Young people were surveyed` | no |

A four-digit number in brackets, or a year after a comma, is read as an
author-year citation (`Young (2019)`, `Young, 2019`) and is not a `cue:state`.

The built-in cue words before a name are `in`, `at`, `near`, `from`, `to`,
`around`, `across`, `between`, `within`, `outside`, `throughout`, and `of` after
a kind of place (`City of`); `cue_words` adds more. A kind of place after the
name (`Shire`, `Council`, `district`, `region`) is a cue too.

A name that two entities share (`Richmond`) is not linked: `match_detail` is
`entities:<n>`.

Every mention that is not linked for one of these reasons is still written,
`matched = false`, `match_method = ambiguous`, with its reason, so what was left
out can be counted and inspected. Only narrative text is searched, not table
cells.

## Output

`*.entity_links.parquet`, one row per mention:

| Column | Holds |
|---|---|
| `candidate_text`, `candidate_kind` | The mention, and the enrichment kind that produced it (`text` for a text candidate) |
| the evidence reference | Where the mention is ([contract.md](contract.md)) |
| `mention_text` | The text at that location |
| `entity_id`, `entity_type`, `canonical_name`, `parent_entity_id` | The register entity; empty when unmatched |
| `confidence`, `match_method`, `matched` | `alias`, `address_exact`, `name_fuzzy`, `name_exact`, `ambiguous` or `unmatched` |
| `match_detail` | For a text candidate: the cue that linked a one-word name, or why it was not linked |

`read_entity_links(path, grain="doc")` derives one row per document and entity
on read; it is never stored.

## Limits

- No ground truth exists for the one-word rules or for narrowing against a
  real register of 10⁵ entities; both are tested on synthetic registers.
- The one-word rules are English and Australian (state abbreviations,
  postcodes, honorifics).
- A text candidate is matched exactly. Fuzzy matching of names found in text
  would need the mention boundaries a model supplies.
