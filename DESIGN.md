# DESIGN.md — Womblex Console Design System

Design tokens, patterns and component conventions for the **optional Womblex UI**
(`womblex[ui]`). Womblex is a library and CLI first; the console is a reader over
the artefacts the pipeline already writes.

> **This document inherits DeepCivic's `DESIGN.md`.** It records only the
> *deltas* and the console-specific additions. Anything not contradicted here —
> the token naming scheme, Tailwind usage rules, shadcn-svelte conventions,
> motion easing, the accessibility checklist — applies verbatim from the
> DeepCivic system. Where this document and DeepCivic's disagree, **this one
> wins inside the Womblex console** and nowhere else.

---

## Why it diverges at all

DeepCivic.com.au is a public marketing and discovery surface: short sessions,
low data density, brand energy doing real work. The Womblex console is the
opposite shape:

| | DeepCivic site | Womblex console |
|---|---|---|
| Audience | Citizens, researchers | Pipeline engineers, corporate auditors |
| Session | Minutes | Hours |
| Density | Low (cards, hero type) | High → extreme (tabular grids, chunk text) |
| Colour's job | Brand signal | **State signal** (pending / running / done / failed) |
| Network | Always online | Often air-gapped or local-only |

Three adaptation principles follow, and every delta below traces to one of them:

1. **Dark-first.** A full-bleed acid-lime page is right for a poster and wrong
   behind a ten-thousand-row grid read for an afternoon. The console *defaults*
   to DeepCivic's dark palette; lime survives as the primary action, active-nav
   and brand colour, exactly as it does in DeepCivic dark mode.
2. **Colour carries state, not decoration.** DeepCivic has no failure states, so
   coral is decorative there and `--destructive` is tuned for a light page. The
   console needs a measured, legible status palette that works on dark.
3. **Nothing is fetched at runtime.** The pipeline already resolves models
   locally with no network access (`utils/models.py`). Fonts and icons follow
   that rule: **self-hosted in the UI bundle, no Google Fonts request.**

---

## Colour

### Inherited unchanged

Every token in DeepCivic's table exists here with the same name and the same
dark-mode value. Components port across the two codebases without edits. The
Tailwind rule is unchanged and absolute: **semantic token classes only, never a
hardcoded hex.**

### Role changes

| Token | DeepCivic | Womblex console |
|---|---|---|
| Default theme | Light (lime page) | **Dark** (`#1a1a24` page); light theme retained and supported |
| `--background` lime `#c8ef35` | Page surface | **Never a page surface.** Primary buttons, active nav rail, running-state fill, logo badge |
| `--accent` purple `#7b6ff0` | Offset shadows, card outlines | Structural only: focus ring, selected-row rule, graph edges in the composer. **Not** the poster offset shadow |
| `--accent-secondary` coral `#e05548` | Decorative (wavy underline, live dot) | **Dropped.** The live dot uses `--status-running`; a decorative-only red next to a semantic red is a legibility hazard |

### Console additions

Six new tokens. Each exists because the console has a concern DeepCivic does not.

| Token | Dark | Light | Why |
|---|---|---|---|
| `--surface-raised` | `#22222e` | `#ffffff` | Console layers panels inside panels (side nav → grid → detail drawer). Page + card is not enough depth |
| `--surface-sunken` | `#141420` | `#c0ff00` | Wells: log output, code blocks, empty grid states |
| `--status-pending` | `#a0a090` | `#38382a` | Queue/checkpoint state |
| `--status-running` | `#c8ef35` | `#c8ef35` | Queue/checkpoint state |
| `--status-done` | `#4ade80` | `#15803d` | Queue/checkpoint state |
| `--status-failed` | `#ef4444` | `#c44536` | Queue/checkpoint state |
| `--status-warning` | `#f59e0b` | `#b45309` | Stale locks, retries, integrity warnings |
| `--font-mono` | `ui-monospace, SFMono-Regular, "Cascadia Mono", Menlo, monospace` | Promoted to a real token. DeepCivic keeps monospace inline in `.markdown-body`; the console needs it for chunk text, hashes, offsets and connection strings |

`--destructive` is **not** reused for failed status. DeepCivic's `#c44536` is
tuned for white text on a light page; as text on the dark console surface it
computes to 3.5:1 and fails AA. `--status-failed` lifts to `#ef4444` (4.6:1) for
that reason. `--destructive` keeps its DeepCivic job: destructive *actions*
(delete run, clear checkpoint).

### Measured contrast

WCAG 2.1 relative-luminance ratios, computed against the dark page surface
`#1a1a24`. AA needs ≥ 4.5:1 for body text, ≥ 3:1 for large text and UI edges.

| Foreground | Ratio | Verdict |
|---|---|---|
| `#f0f0e8` body text | 15.1:1 | Pass |
| `#c8ef35` lime | 13.1:1 | Pass |
| `#4ade80` done | 10.0:1 | Pass |
| `#f59e0b` warning | 8.0:1 | Pass |
| `#a0a090` muted | 6.5:1 | Pass |
| `#ef4444` failed | 4.6:1 | Pass |
| `#7b6ff0` purple | 4.4:1 | **Fails body.** Structural / large text only |
| `#c44536` DeepCivic destructive | 3.5:1 | **Fails body.** Fills only, never text |

### The one rule that makes status legible in both themes

**A status pill or dot is always a solid `--status-*` fill with near-black
`#18182a` text.** Near-black on all five fills computes to 4.6:1 (failed) through
13.1:1 (running) — so the pill is identical in light and dark mode and needs no
per-theme override. Do not render status colours as text on a page surface.

---

## Typography

Inherited: Barlow Condensed Black 900 for display, Inclusive Sans for body and
UI, the 1.6 body line-height. **Self-hosted as woff2 in the UI bundle** — no
Google Fonts request, per adaptation principle 3.

### Deltas

- **No poster shadow in-app.** `.display-xl` / `.display-lg` and their purple
  `text-shadow` offsets are marketing devices. The console uses
  `font-display text-2xl`–`text-3xl`, flat, for page titles and KPI readouts.
  Barlow Condensed Black stays valuable here: it fits big numbers in narrow
  tiles.
- **Figures align.** Any column of numbers — token counts, row counts, byte
  sizes, durations — sets `font-variant-numeric: tabular-nums`. Where the face
  does not supply the feature, fall back to `--font-mono`.
- **Identifiers are always mono.** `source_hash`, `doc_id`, `run_id`, char
  offsets, S3 URIs, DSNs. Truncate hashes to 12 chars with the full value in a
  `title` and a copy button.
- **Table body copy is 13px/1.45**, below DeepCivic's 16px floor, which applies
  to Inclusive Sans only. Barlow Condensed keeps its ≥ 16px floor absolutely —
  a condensed display face at 13px is not readable.

### Logo

The stamp badge is inherited and re-lettered: a 40×40 `bg-foreground
text-background` tile with **"WOMB" stacked over "LEX"** in Barlow Condensed
Black, letter-spaced. Same construction, same review rule.

---

## Density and layout

### Shell

Per the UX requirements, two persistent chrome elements, both `--surface-raised`:

- **Top bar** (56px, inherited height): global search, notifications, execution
  controls, run selector. System context only.
- **Side nav** (rail, 64px collapsed / 224px expanded): the router between the
  five domains. Active item takes a lime left rule + lime icon.

The `max-w-5xl` centred page shell is **dropped**. Console views are full-bleed
to the viewport; a grid that can show four more columns should.

### Density scale

| Level | Row height | Used by |
|---|---|---|
| Comfortable | 48px | Resources Console, Pipeline Composer |
| Default | 40px | Corpus Inspector, job lists |
| Compact | 32px | Corpus Inspector when the user opts in |

Density is a user preference, persisted locally, applied via a `data-density`
attribute on the shell — not a per-table prop.

---

## Component conventions

shadcn-svelte remains the source of primitives; install rather than build, and
customise through `class` props without forking internals. Beyond DeepCivic's
set, the console needs `Tabs`, `Tooltip`, `DropdownMenu`, `Command`, `Progress`,
`Resizable`, `ScrollArea` and `Skeleton`.

**`StatusPill`** — the single most repeated element. Icon + text label + status
fill, per the rule above. Statuses map 1:1 onto the values the system actually
writes: `pending` / `running` / `done` / `failed` from `womblex_jobs.status`,
plus `stale` (a `running` row past its lock timeout) and `skipped`. `skipped`
reuses the pending fill with a distinct icon — which is legal precisely because
the icon, not the colour, is doing the work.

**`StageNode`** — one operation in the Pipeline Composer. Shows the stage name,
its enabled/disabled toggle, and its declared inputs and outputs. Nodes are
wired by purple `--accent` edges. Disabled stages drop to 40% opacity and keep
their edges visible, so a broken chain is legible as a gap.

**`KpiTile`** — dashboard stat. `font-display` numeral, `--muted-foreground`
label, optional sparkline. Sparkline stroke is `--status-running`; no fill.

**`DocumentGrid`** — the Corpus Inspector's table. Real `<table>` semantics,
sticky header on `--surface-raised`, row virtualisation, selected row marked by
a 2px `--accent` left rule (never by background tint alone). Failed rows are
marked by their StatusPill, never by a tinted row.

**`ChunkCard`** — the Semantic Chunk Inspector's unit. `--surface-sunken` well,
mono text, entity mentions underlined in `--accent` with a hover tooltip,
`<PERSON_n>` masks rendered as inline pills. Header carries chunk index, token
count, char range and content type.

**`ConnectionCard`** — Resources Console. Label, mono connection string
(secrets masked to the last 4 characters), StatusPill, and a "Test" action.
**Never render a credential in full**, in the DOM or in a copy buffer. The
fleet variant adds queue depth and the workers currently holding batches;
workers are ephemeral and scale to zero, so an empty fleet is a normal resting
state and must not render as an error.

**`ReportIssue`** — the console never edits a stage output, so this is the only
way a reviewer acts on a bad record. An icon button on any row or chunk opens a
note field; submitting appends the record plus the note to an append-only
feedback log. Confirmation is a toast, not a modal — reporting should cost one
click and no attention. The reported row keeps its normal appearance
afterwards: a report is an observation, not a state change.

**Cards generally**: DeepCivic's `.card-offset` layered purple shadow is a
poster device — **not used in the console**. Panels are separated by
`--border` and surface elevation.

---

## Motion

Inherited: `cubic-bezier(0.16, 1, 0.3, 1)`, 0.3–0.5s, svelte-motion for stateful
animation, `prefers-reduced-motion` respected.

One delta: **live data does not animate.** Rows that update on a poll or stream
— job lists, queue counts, KPI numerals — change value without entrance
transitions. Staggered entrance on a grid that refreshes every two seconds
produces a permanently moving page. Animate view transitions, drawer opens and
the composer's edge drawing; not data.

The `.live-dot` pulse is inherited, re-coloured to `--status-running`, and used
only where something genuinely is running.

---

## Accessibility

DeepCivic's checklist applies in full and is non-negotiable. Three additions the
console's shape forces:

- **Never encode state by colour alone.** Every status carries an icon and a
  text label. This is WCAG 1.4.1, and it is the console's most load-bearing
  rule — state *is* the information.
- **Grids are tables.** Real `<table>`/`<th scope>` markup with a `<caption>`.
  Virtualised rows set `aria-rowcount` / `aria-rowindex` so the total is
  announced, not just the rendered window.
- **Keyboard reaches everything the mouse does.** Grid arrow-key navigation,
  `/` to focus global search, `Esc` to close drawers, and a visible
  `--ring` focus indicator on every focusable element including grid cells.

Test both themes, and test the console at 32px density — the compact row is
where touch-target and focus-ring rules break first. Interactive controls keep
a ≥ 44×44px hit area even where the visual row is 32px.

---

## Adding new UI

1. Check DeepCivic's `DESIGN.md` first — if it has a rule, follow it.
2. Check whether a **shadcn-svelte** primitive covers the need.
3. Style exclusively with token classes. New colour ⇒ new token, justified here.
4. Run DeepCivic's accessibility checklist, plus the three additions above.
5. Test in dark **and** light, at default **and** compact density.
