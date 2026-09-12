# Deployment images: what each compose service runs

Every service in this project's compose files, and for each one a recorded
decision about where its image comes from. The point of the enumeration is that
it is complete: a service with no decision is a service nobody chose to build
from source, and those are the ones that quietly make a stack unidentifiable.

Two weaknesses make a deployment unable to name what it is running, and they are
the same weakness in different places:

- **Building from source at deploy time.** The image is whatever the working
  tree held when someone ran `docker compose up`. Two people running "the same"
  stack can be running different software, and neither can say which.
- **Referencing a moving tag.** `minio/minio` with no tag resolves to whatever
  the registry currently calls latest, so the same compose file yields a
  different container this week than last.

So third-party images are audited here alongside the ones built from source.
An audit that looked only at what this project builds would pass a stack that is
still unidentifiable.

This is an audit, not a conversion. Whether a service *should* become a
published image is a deployment decision and the answers differ per service —
what is required is that each one has an answer.

## Declared totals

The enumeration below is checked against these counts by
`tests/test_deployment_images.py`, so a service added later without a decision
fails a test rather than quietly joining the stack:

| | Count |
|---|---|
| Compose files | 2 |
| Services in `docker-compose.yml` | 8 |
| Services in `docker-compose.local.override.yml` | 5 |
| Of the base file: build from source | 5 |
| Of the base file: reference a third-party image | 3 |
| Of the base file: reference an image of this project | 0 |

The override introduces no service of its own — all five of its entries adjust
dependency and environment settings on a service the base file already declares.

## The enumeration

`Kind` is what compose does with the service as the file stands today: `build`
compiles from the working tree, `image` pulls a reference, `settings` is an
override entry that does neither and inherits the base file's verdict.

Being reached only through a profile is not an exemption — `womblex`,
`seed-demo` and the three bundled backends are all behind profiles and are
enumerated like any other.

| Service | File | Kind | Reference | Profile | Verdict |
|---|---|---|---|---|---|
| `init` | base | build | `Dockerfile` | — | Published — pipeline image |
| `womblex` | base | build | `Dockerfile` | `cli` | Published — pipeline image |
| `worker` | base | build | `Dockerfile` | — | Published — pipeline image |
| `seed-demo` | base | build | `Dockerfile` | `seed` | Published — pipeline image |
| `ui` | base | build | `Dockerfile.ui` | — | Published — console image |
| `postgres` | base | image | `postgres:16` | `local` | Third party — tag pinned to a major series, moving within it |
| `minio` | base | image | `minio/minio` | `local` | Third party — tag unpinned |
| `createbuckets` | base | image | `minio/mc` | `local` | Third party — tag unpinned |
| `init` | override | settings | — | — | Inherits the base verdict |
| `womblex` | override | settings | — | — | Inherits the base verdict |
| `worker` | override | settings | — | — | Inherits the base verdict |
| `seed-demo` | override | settings | — | — | Inherits the base verdict |
| `ui` | override | settings | — | — | Inherits the base verdict |

## Why the five collapse to two images

All five build-from-source services carry the same verdict, and that is not five
decisions that happened to agree. Four of them — `init`, `womblex`, `worker` and
`seed-demo` — build the identical `Dockerfile`, differing only in the command
they run and, for two of them, a `./configs` mount. Nothing that distinguishes
them is part of the image, so they are one artefact with four entry points;
building them separately would produce four copies of the same bytes. The console is the
second image because `Dockerfile.ui` carries a Node build stage the pipeline has
no reason to hold.

The demo seeder is the one that could reasonably have gone the other way: it
exists to publish a sample corpus and a deployment that never seeds does not
need it. It is published anyway because it shares the pipeline image rather than
adding one — declining to publish it would not save a build, it would only make
one of four commands on the same image unavailable.

## The third-party finding, recorded not fixed

Two of the three third-party references carry no tag, so they resolve to
whatever the registry currently serves. `postgres:16` is pinned to a major
series and still moves within it.

This is recorded rather than corrected because it is a different change with a
different risk: pinning a database image is a deployment decision with an
upgrade path attached, and folding it into an audit is how it would get made
without one. The audit's job is that the finding exists in writing and is
re-checked, not that it is resolved in the same breath.

## What re-checks this

`tests/test_deployment_images.py` parses this document's tables and both compose
files and fails when they disagree. It holds the enumeration to the declared
counts, requires a verdict on every service, and checks that each row's `Kind`
and `Reference` still describe what compose actually declares — so the finding
cannot regrow silently through a service added without a decision, or a verdict
that has quietly stopped being true.
