# Dependency security upgrade plan

## Context
Dependabot reports 34 high/critical alerts. The alerts API returned 403 to this session, so the list below comes from local audits:
- `pip-audit` on `uv.lock`: 16 packages
- `npm audit` on `ui/package-lock.json`: 4 high, 1 moderate, 3 low

You asked for two more things:
1. Stop upgrades breaking other things, as they have before.
2. Bring other vulnerability sources into scope.

## Why past upgrades broke things
**The lockfile doesn't control what CI or the images install.** Every install path runs `pip install` and resolves fresh from the open ranges in `pyproject.toml`, ignoring `uv.lock`:
- CI `test` job: `pip install -e ".[dev,cloud,ui,api]"`
- `Dockerfile`: `pip install ".[${EXTRAS}]"`
- `Dockerfile.ui`: the same
- `release.yml`: builds the wheel with no lock

So a green CI run means "today's latest resolve passed", not "the locked set passed". A new upstream release can break CI or an image build with no commit in between. The history records this four times:

| Recorded break | Source | Cause |
|---|---|---|
| ruff 0.16 brought 297 lint errors and skipped mypy and pytest | CHANGELOG, `pyproject.toml` comment | unpinned in CI |
| DBOS 3.3 dropped vendored `croniter` | commit 1f68bea ("CI installs unpinned and picked up 3.3") | unpinned in CI |
| OpenCV 5 changed the `HoughLinesP` layout and crashed every OCR extraction | CHANGELOG | `opencv-python-headless` unbounded |
| the stock pip resolver stalled on the boto stack | CHANGELOG, Dockerfile comment | fresh resolve in image build |

The same gap also undercuts the alerts themselves. Dependabot scans `uv.lock`, but deployed images don't install from it. Bumping the lock alone changes nothing that ships, and the images may already run different versions from the ones the alerts name.

Several direct dependencies have no lower or upper bound: `opencv-python-headless`, `numpy`, `pyarrow`, `openpyxl`, `tqdm`, `pyyaml`, `mypy`, `pytest-cov`. Today the lock is the only thing holding them still.

## Approach: make the lock authoritative first, then upgrade through it
CLAUDE.md rules that apply:
- Each merge stands alone.
- Dependency changes get their own merge and human approval.
- `uv.lock` doesn't count toward the 500-line cap.

### M0 – Install from the lock everywhere (no version changes)
- **CI `test` job**: replace the pip install with `astral-sh/setup-uv`, then `uv sync --frozen --extra dev --extra ui --extra api`, then run ruff, mypy and pytest via `uv run`. This matches the commands CLAUDE.md already documents. Add `uv lock --check` as the first step so a `pyproject.toml` edit without a re-lock fails.
- **Both Dockerfiles**: copy `uv.lock`, then install the exported lock, for example:
  - `uv export --frozen --no-dev --extra <EXTRAS> --no-emit-project -o /tmp/req.txt`
  - `pip install --require-hashes -r /tmp/req.txt`
  - `pip install --no-deps .`

  The images then ship exactly the audited set, and the "upgrade pip first" workaround can go.
- `release.yml`: no change. A published wheel keeps its ranges, which is normal for a library.
- Checks: the `images` job already builds and smoke-tests both images, and the digest pins in `tests/test_default_digest.py` prove extraction output is unchanged.
- **Expected effect:** CI may flip from today's fresh resolve to the older locked set. Any failure that appears here is drift that was already hidden, and it's useful to find it before the upgrade merges.

### M1 – Scheduled drift detection, so upgrades arrive small and tested
- Add `.github/dependabot.yml` with these ecosystems:
  - `uv` (root)
  - `npm` (`/ui`)
  - `docker` (both Dockerfiles)
  - `github-actions`
- Run weekly, grouped as patch+minor in one PR per ecosystem, with majors as individual PRs. Each PR then runs the lock-based CI from M0, so a break shows up on a PR instead of on `main`.
- Optional: a weekly `canary` CI job running `uv lock --upgrade` and then the test suite, non-blocking. It gives early warning of upstream breakage (the croniter or OpenCV kind) without moving the lock.
- Add `pip-audit` (on the exported lock) and `npm audit --audit-level=high` as a CI job so new advisories show on PRs. This needs a pinned `pip-audit` version, which is a dev-dependency change.

### M2 – Python lockfile bumps: transitive, patch and minor
- `uv lock --upgrade-package` for:
  - aiohttp, multidict, yarl
  - urllib3, boto3, botocore, aiobotocore, s3transfer
  - anyio, idna, certifi
  - fsspec, s3fs
  - lxml
  - pygments, pytest, setuptools
  - cryptography (46 → 50, transitive via pdfminer-six and presidio; code doesn't import it)
- No `pyproject.toml` change. A dry run confirms everything resolves.

### M3 – Pillow 12.3 → 13.0 (major)
- Touches `ingest/pdf/_image.py` (opening non-PDF files, frames, page-rect rule), renders and OCR input.
- Gates:
  - Unchanged `test_default_digest` pins and synthetic source hashes
  - The OCR (`slow`) tests
  - A grep of `src/` for APIs that Pillow 13 removed
- A digest shift means investigate, not re-pin.

### M4 – ML stack: torch 2.10 → 2.14, transformers 5.3 → 5.17, sentence-transformers 5.3 → 6.1
- Affects PII context scoring (`pii/cleaner.py`, the 0.35 threshold) and the `all-MiniLM-L6-v2` load through `utils/models.py`.
- torch switches to CUDA 13 wheels, which changes image size. Measure it in the `images` job.
- Two advisories stay open with no fix published: torch PYSEC-2026-139 and transformers PYSEC-2026-4174. Check whether the vulnerable code path is reachable (e.g. pickle `torch.load` versus safetensors) and record the outcome in the PR body.

### M5 – UI high and moderate fixes
The `ui/` frontend and its npm audit left with the console (womblex-console). Nothing here remains to do.

### M6 – Other vulnerability surfaces (newly in scope)
| Surface | Current state | Proposed |
|---|---|---|
| Base images `python:3.12-slim`, `node:22-slim` | floating tags; OS packages unscanned | pin by digest (Dependabot `docker` bumps them); add a Trivy scan of both CI-built images, reporting high/critical |
| `minio/minio`, `minio/mc` in `docker-compose.yml` | no tag (implicit `latest`) | pin to a release tag or digest |
| `postgres:16` (compose and CI service) | major-only tag | pin to a minor tag or digest |
| GitHub Actions | tag refs (`@v4`); `pypa/gh-action-pypi-publish@release/v1` is a branch ref in a job with `id-token: write` | pin to commit SHAs (Dependabot `github-actions` keeps them current) |
| Unbounded direct deps (opencv, numpy, pyarrow, …) | the lock is the only guard | after M0, add upper bounds at the next major only where a break is on record (opencv `<6`). Otherwise leave to the lock |

## Verification (every merge)
- `uv lock --check`
- `uv sync --frozen --extra dev --extra ui --extra api`
- `uv run ruff check src/ tests/`
- `uv run mypy src/`
- `uv run python -m pytest tests/ -v -m "not benchmark"`
- The OCR `slow` tests (M3/M4)
- `images` job green
- Re-run `pip-audit` / `npm audit` and confirm the targeted IDs are gone
- After the last merge, reconcile the Dependabot tab against the 34.

## Decisions for you
1. **M0** changes how CI and both images install. That is the core mitigation. Do you approve it, and should it land before any version bump?
2. Are M3 (Pillow major) and M4 (sentence-transformers major plus CUDA 13 torch) in scope now?
3. Do you accept the two unfixed advisories (torch, transformers) once reachability is assessed?
4. M1 and M6: which items do you want (Dependabot config, canary job, audit job, Trivy, SHA-pinned actions, image pins)?
5. PR shape: one stacked PR per merge on `claude/ecstatic-einstein-mc61q9`, or another arrangement?

