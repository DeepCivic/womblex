# Stability and deprecation policy

Womblex is alpha and pre-1.0. This is the canonical statement of what can
change between releases and how a change is announced.
[docs/contract.md](docs/contract.md) holds the on-disk schema itself.

## What is stable

| Surface | Stability |
|---|---|
| Python names in `womblex.__all__`, and the signatures they expose | Stable. Changed or removed only after a deprecation window (below). Pinned by `tests/test_public_api.py` |
| The on-disk data contract (`womblex.contract_version` in every Parquet footer) | Stable, and versioned apart from the package (below) |
| Everything else: other modules, CLI options, config keys, stored-workflow commands, the `/v1` service API | Moving. May change in any minor release, with a CHANGELOG entry |

A module or name outside `womblex.__all__` is documented in the API reference
because it can be imported, not because it is stable.

## Versions before 1.0

Versions are `0.MINOR.PATCH`.

- **Minor** (`0.6.0` to `0.7.0`) may carry breaking changes to anything outside
  the stable surfaces, and ends the deprecation window of a stable name
  deprecated in an earlier minor.
- **Patch** (`0.6.0` to `0.6.1`) carries fixes and additive changes only. It
  breaks nothing that worked in the previous patch of the same minor.

From 1.0, standard [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
applies: a breaking change to a stable surface needs a major release.

## The data contract's own version

The contract version moves with the schema, not with the package:

| Change | Contract bump | Older files |
|---|---|---|
| Additive column | minor | Readers back-fill the column |
| Rename or removal | major | A reader shim is kept for the old major |
| Retired file role | major | No longer written; an older file keeps its own footer |

## How a breaking change is announced

- Every breaking change has a [CHANGELOG](CHANGELOG.md) entry, under `Changed`
  or `Removed`, marked **Breaking:** and naming what a caller must do.
- A version's CHANGELOG section is its release notes.
- A deprecation is listed under `Deprecated` in the release that starts it,
  naming the replacement and the release it is removed in.

## Deprecation window

A stable Python name, or a signature it exposes, is removed or changed
incompatibly only after **one minor release** in which using it emits a
`DeprecationWarning` naming its replacement. The removal is recorded in the
CHANGELOG under `Removed`.

A reader shim for an older data-contract major, or a compatibility alias for a
moving surface, states the release it is removed in when it lands (for example
"removed in 0.7.0").
