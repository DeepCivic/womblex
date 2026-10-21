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
- In `ui/`, run `npm audit fix` without `--force`. It covers:
  - vite 8.2.x → ≥8.2.3
  - devalue
  - brace-expansion
  - source-map-js
  - postcss-selector-parser
- The existing `frontend` job (lint, svelte-check, build) gates it.
- `@sveltejs/kit` 3 and `adapter-static` 4 are low-severity majors and stay out.

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

---

## Appended

I'll go through the code for each of the four before writing up the changes.
Still mapping the code: now checking how the workflow keys reach a file's footer, to see what extraction would need.
That's the code mapped. Here are the changes each choice needs, checked against `main` as it stands after #268.
Items 1 and 4 are wording only. Items 2 and 3 are code. Each would be its own merge, under the 500-line rule in CLAUDE.md. Item 2 has one question you need to answer before it can be built.

### 1. Keep "DBOS", remove "(Glossary)" (docs only, about 2 lines)

* `docs/functional_requirements.md`, Requirement 2's Paid line: change "A SageMaker deployment of the paid models (Glossary) is billed…" to "A SageMaker deployment of the Kanon-2 Enricher or Embedder is billed…". That names the paid models directly, so nothing points at a missing section.
* CHANGELOG: one line.
* Nothing else refers to a glossary outside the plan file. "DBOS" already stands in Requirements 2, 16 and 27.

### 4. Use the plan's wording for the webhook key (docs only, about 3 lines)

* Requirement 27's webhook bullet: "signed with a key held where the run executes" becomes "signed with a key held on the worker".
* CHANGELOG: one line.
* No code change. `run-workflow` still sends webhook events from the operator's machine (`cloud/workflow_runner.py` builds `RunEvents`). The bullet would just no longer describe that local case, which is what you accepted.
* `docs/workflows.md` and decision D-3.3 already describe the fleet case, so they stay as they are.

### 3. Extraction files name the workflow (code, about 150–200 lines)

When it applies: a workflow only causes extraction in one case, a scheduled firing (`workflow_dispatch.fire_workflow` queues the batches). `run-workflow` and `dispatch-workflow` run over shards that already exist. Extraction from `womblex run` or `enqueue` wasn't done by a workflow, so it would still name none; stamping a workflow on it would be false. "Every output names the workflow" would hold for scheduled runs only.

Code changes:

* `cloud/jobs.py`: `RunBoard.enqueue` gains `workflow=` (id, digest, spec) and passes it as an argument of each `extract_batch`.
* `cloud/workflow_dispatch.py`: `fire_workflow` passes the stored workflow to `board.enqueue`.
* `cloud/workflows.py`:
   * `extract_batch` and `_extract` gain an optional `workflow` argument, last, so batches already queued still run.
   * `_extract` calls `process_batch` inside `workflow_context(...)`, as `_unit` already does for stage steps.
* `store/run_stamp.py`: move the block in `sidecar_footer` that adds the workflow keys into a small helper (e.g. `workflow_footer()`) that both paths call.
* `store/output.py`: `write_results` merges that helper's keys into the footers of the four extraction files.
* No leak into later stages: `RunStamp` doesn't carry the workflow keys, so a stage run later outside the workflow won't inherit them.
* Side effect in the run record: the record's `workflows[].stages` list will now include `extract` for firing runs. That changes what the field contains, so the record version may need bumping from 8 to 9. I'd decide that while building it unless you have a view.

Tests:

* A firing batch's four files carry `womblex.workflow_digest` and `womblex.workflow`.
* A batch queued without a workflow carries neither.
* A downstream sidecar written outside a workflow doesn't inherit them.
* The DBOS tests already run on SQLite.

Docs:

* `docs/contract.md`: the keys are no longer "absent from extraction files"; they're present on a firing's batches.
* Requirement 27's bullet and Given/Then line.
* `docs/decisions-operations.md`: the plan's rules say a reversed decision is deleted, not kept as an alternative.
* CHANGELOG.

### 2. A real `skipped` status for already-published units (code, about 250–350 lines)

Today there's nothing to mark `skipped` on the fleet:

* A finished unit is never run again: DBOS records it as a completed step and attaches to it, so there's no second run to skip.
* A changed workflow or a new firing gets new stage ids, and all its units run again on purpose (`dbos_app.stage_scope`: "a changed workflow runs its stages rather than attaching"). Their existing outputs were made under the old config, so they're stale.
* The only skip-if-already-published rule is in `run-stage --store` (`stage_runner._skippable`). That path writes no queue row, so its skips appear only in a log count.

So this item first needs a new fleet behaviour, and then the status on top of it.

Question for you: what makes an existing output "already published" rather than stale? The `run-stage --store` rule only checks that every output exists. On the fleet that would wrongly skip:

* a unit after a config change;
* a later stage's retry, whose outputs exist but came from partial input.

The candidates:

* (a) Presence plus matching footer: skip only when every output exists and its footer's `config_digest` (and workflow digest, if any) equals the dispatch's. Retries never skip.
* (b) Presence only: the `run-stage --store` rule, with retries and config changes forced to run. This needs a separate flag to tell those cases apart.
* (c) Something else you had in mind.

Code changes:

* `cloud/workflows.py`, `run_stage`:
   * Before each unit's step, apply the skip test you choose. A skipped unit is recorded as a step that publishes nothing.
   * Retries and re-runs of a later stage (`rerun_stage`, the fork in `cloud/retry.py`) always run every unit.
   * `layout` and `graph-refresh` keep their existing exclusions (`skip_if_output_present=False`, in place).
   * The workflow's result gains `skipped: n`.
* `cloud/jobs.py`:
   * DBOS has no "skipped" state, so the status has to come from the result: a successful stage whose every unit was skipped reads `skipped`.
   * `_board_status` and the row listing need to load results.
   * The rollup counts `skipped` as finished for the run's state.
* `cloud/run_status.py`: add `skipped` to the status order and the sentence.
* `api/models.py`: add `skipped` to the status values. That changes the OpenAPI schema, so `test_openapi_surface_is_pinned` needs its pinned values updated as part of the change.
* Console (`ui/src/lib/status.ts`, `routes/dashboard/+page.svelte`): `cancelled` currently uses the `skipped` pill. It needs a pill of its own, or the two statuses look identical.

Docs:

* Requirement 16's status bullet takes the plan's definition.
* `docs/decisions-operations.md`: the "cancelled… shows the skipped pill" entry is replaced, and a new decision records the fleet skip rule.
* `docs/architecture.md`: its line about stages running as recorded steps.
* CHANGELOG.
* Requirement 2 if the rule changes when paid work is redone. Option (a) means a re-dispatch under the same config no longer pays again.

Tests:

* A re-dispatch whose outputs are already published reads `skipped` and calls nothing.
* A config change runs every unit again.
* A later stage's retry never skips.
* A cancelled job shows its own pill.
