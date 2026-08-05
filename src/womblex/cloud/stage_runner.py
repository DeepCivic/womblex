"""Remote per-batch shard-stage runner — the generalisation of ``finalize``.

``womblex finalize`` is already store-aware in exactly the shape a downstream
stage needs: list one sidecar class under the run's shard prefix, download it to
a temp dir, call the unchanged library function against that ``Path``, upload
the result back. This module extends that shape from a single consolidation step
to the per-batch stages declared in :mod:`womblex.cloud.stage_contracts`.

Invariants:

- **Discovery is separate from required inputs.** Bases come from
  extraction-role siblings only; a downstream sidecar can never introduce one.
  Discovery-only files (``*.form_fields.parquet``) are never downloaded.
- **One unit of work at a time.** ``PER_BATCH`` stages stage exactly one base
  into a temp dir. Only ``quality`` is ``WHOLE_RUN``, because its dedup cluster
  ids are corpus-wide and per-batch execution would emit colliding ids.
- **All declared outputs publish, or none do.** A partial sidecar set can never
  read as complete on a later run.
- **Idempotent.** Completed bases skip on their published outputs, so re-running
  as more batches land processes only the new ones — the property ``finalize``
  already has, extended to every stage.

Stage *ordering* is the caller's: this runs one stage, it does not sequence them.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from womblex.cloud.stage_contracts import (
    DISCOVERY_SUFFIXES,
    PRODUCER_OF,
    MutationMode,
    RunContext,
    StageContract,
    StageScope,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.config import WomblexConfig
    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)


@dataclass
class StageRunSummary:
    """Outcome of one runner invocation."""

    stage: str
    processed: int = 0
    skipped: int = 0
    not_ready: int = 0
    failed: int = 0
    published: int = 0
    bases: int = 0
    not_ready_missing: set[str] = field(default_factory=set)

    @property
    def exit_code(self) -> int:
        if self.failed:
            return 1
        # Every base blocked on an absent upstream sidecar is a stage-ordering
        # error, not a still-draining fleet.
        if self.bases and self.not_ready == self.bases:
            return 1
        return 0

    def log(self) -> None:
        logger.info(
            "%s: %d base(s) — %d processed, %d skipped, %d not-ready, %d failed "
            "(%d file(s) published)",
            self.stage, self.bases, self.processed, self.skipped,
            self.not_ready, self.failed, self.published,
        )


class NotReady(Exception):
    """A required input is absent for this base — upstream has not produced it."""

    def __init__(self, suffix: str):
        self.suffix = suffix
        super().__init__(suffix)


class InputContractError(Exception):
    """Config selects an input that is absent, and running anyway would be wrong.

    Distinct from :class:`NotReady`: that one is a fine, expected state on a
    draining fleet. This one means the stage *would* run and *would* publish —
    just from the wrong text layer — so it is a failure, but an actionable one
    rather than a crash, and is logged without a traceback.
    """


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def remote_bases(keys: list[str]) -> list[str]:
    """Batch base stems in *keys*, discovered from extraction-role siblings only.

    Mirrors ``chunk_stage._batch_bases`` over store-relative keys: the same four
    roles, the same ``.corrupt`` exclusion, the same name sort. A store holding
    only ``*.chunks.parquet`` yields nothing — downstream sidecars do not define
    batches.
    """
    seen: set[str] = set()
    for key in keys:
        name = key.rsplit("/", 1)[-1]
        for suffix in DISCOVERY_SUFFIXES:
            if not name.endswith(suffix):
                continue
            stem = name[: -len(suffix)]
            if stem.endswith(".corrupt"):
                break
            seen.add(stem)
            break
    return sorted(seen)


def _key(shard_prefix: str, stem: str, suffix: str) -> str:
    return f"{shard_prefix}/{stem}{suffix}"


# ---------------------------------------------------------------------------
# Per-unit execution
# ---------------------------------------------------------------------------


def _resolve_inputs(
    contract: StageContract, config: WomblexConfig, shard_prefix: str,
    stems: list[str], present: set[str],
) -> list[str]:
    """Input keys to download for *stems*. Raises ``NotReady`` / ``InputContractError``.

    ``NotReady`` means an upstream stage has not written a required sidecar yet.
    ``InputContractError`` means config selected a *strict* conditional input that
    is absent — running anyway would silently produce a sidecar built from the
    wrong text layer, because ``load_overlay`` falls back to verbatim and only
    warns.
    """
    keys: list[str] = []
    for suffix in contract.required_inputs:
        for stem in stems:
            key = _key(shard_prefix, stem, suffix)
            if key not in present:
                raise NotReady(suffix)
            keys.append(key)

    for cond in contract.conditional_inputs(config):
        for stem in stems:
            key = _key(shard_prefix, stem, cond.suffix)
            if key in present:
                keys.append(key)
            elif cond.strict:
                raise InputContractError(
                    f"{cond.reason} selects {cond.suffix} but {key} is absent. "
                    f"Run `womblex run-stage --stage "
                    f"{PRODUCER_OF.get(cond.suffix, '<upstream>')}` first — "
                    "proceeding would silently fall back to verbatim element text."
                )
            else:
                logger.debug("%s: %s absent for %s (%s) — stage falls back",
                             contract.name, cond.suffix, stem, cond.reason)
    return keys


def _outputs_present(
    contract: StageContract, config: WomblexConfig, shard_prefix: str,
    stems: list[str], present: set[str],
) -> bool:
    """True when every declared output already exists for every stem."""
    return all(
        _key(shard_prefix, stem, suffix) in present
        for stem in stems
        for suffix in contract.outputs(config)
    )


def _publish(
    contract: StageContract, config: WomblexConfig, store: RemoteStore,
    shard_prefix: str, stems: list[str], local_dir: Path,
) -> int:
    """Upload every declared output, or none of them.

    Verifying before uploading is what keeps the skip rule honest: a base whose
    outputs are half-written must not look complete on the next run.
    """
    pending: list[tuple[Path, str]] = []
    for stem in stems:
        for suffix in contract.outputs(config):
            local = local_dir / f"{stem}{suffix}"
            if not local.exists():
                raise FileNotFoundError(
                    f"{contract.name} declared {suffix} for {stem} but did not write it; "
                    "publishing nothing for this unit."
                )
            pending.append((local, _key(shard_prefix, stem, suffix)))

    for local, key in pending:
        store.upload_file(local, key)
    return len(pending)


def _run_unit(
    contract: StageContract, config: WomblexConfig, ctx: RunContext,
    store: RemoteStore, shard_prefix: str, stems: list[str], input_keys: list[str],
) -> int:
    """Stage in, run the unchanged ``*_shards()``, publish. Returns files published."""
    with tempfile.TemporaryDirectory(prefix="womblex-stage-") as tmp:
        documents = Path(tmp) / "documents"
        store.download_to_dir(input_keys, documents)
        contract.run(documents, config, ctx)
        return _publish(contract, config, store, shard_prefix, stems, documents)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def checkpoint_prefix_for(contract: StageContract, output_prefix: str) -> str | None:
    """Store-relative dir holding this stage's checkpoint, or ``None`` if it keeps none."""
    if contract.checkpoint_dirname is None:
        return None
    return f"{output_prefix.strip('/')}/{contract.checkpoint_dirname}"


def _stage_checkpoint_in(
    contract: StageContract, store: RemoteStore, prefix: str, dataset: str, local_dir: Path,
) -> CheckpointManager:
    """Pull the stage's checkpoint dir down and open a manager over it.

    Checkpoint state is a *directory*, not a shard suffix, so it cannot ride
    along with the per-base staging. Opt-in only, and single-invocation per run:
    concurrent runners would clobber each other's JSON — which is precisely why
    the job queue, not a checkpoint file, is the distributed checkpoint.
    """
    from womblex.store.checkpoint import CheckpointManager

    local_dir.mkdir(parents=True, exist_ok=True)
    existing = store.list_files(prefix, "*.json")
    if existing:
        store.download_to_dir(existing, local_dir)
    mgr = CheckpointManager(local_dir, f"{dataset}_{contract.name.replace('-', '_')}")
    mgr.load()
    return mgr


def _stage_checkpoint_out(store: RemoteStore, prefix: str, mgr: CheckpointManager) -> None:
    if mgr.checkpoint_file.exists():
        store.upload_file(mgr.checkpoint_file, f"{prefix}/{mgr.checkpoint_file.name}")


def run_stage_remote(
    contract: StageContract,
    store: RemoteStore,
    shard_prefix: str,
    config: WomblexConfig,
    *,
    ctx: RunContext | None = None,
    force: bool = False,
    checkpoint_prefix: str | None = None,
    checkpoint_dataset: str = "runner",
) -> StageRunSummary:
    """Execute *contract* against a store's shard prefix, one unit at a time."""
    ctx = ctx or RunContext()
    summary = StageRunSummary(stage=contract.name)

    present = set(store.list_files(shard_prefix, "*.parquet"))
    stems = remote_bases(sorted(present))
    if not stems:
        logger.error(
            "No batch bases under %s — expected extraction siblings (%s). "
            "Downstream sidecars alone cannot drive discovery.",
            shard_prefix, ", ".join(DISCOVERY_SUFFIXES),
        )
        return summary

    units = [stems] if contract.scope is StageScope.WHOLE_RUN else [[s] for s in stems]
    summary.bases = len(units)
    if contract.scope is StageScope.WHOLE_RUN:
        logger.info(
            "%s is run-scoped: staging %d base(s) in one pass (its clusters are "
            "corpus-wide; per-batch execution would emit colliding ids).",
            contract.name, len(stems),
        )

    with contextlib.ExitStack() as stack:
        if checkpoint_prefix and contract.checkpoint_dirname:
            ckpt_dir = Path(stack.enter_context(
                tempfile.TemporaryDirectory(prefix="womblex-stage-ckpt-")))
            ctx.checkpoint_mgr = _stage_checkpoint_in(
                contract, store, checkpoint_prefix, checkpoint_dataset, ckpt_dir)
            stack.callback(_stage_checkpoint_out, store, checkpoint_prefix, ctx.checkpoint_mgr)

        for unit in units:
            label = unit[0] if len(unit) == 1 else f"{len(unit)} base(s)"
            try:
                # Skip is only sound for sidecar producers: graph-refresh's
                # outputs are a subset of its inputs, so output-exists can
                # never fire.
                if (
                    not force
                    and contract.mutation is MutationMode.SIDECAR
                    and _outputs_present(contract, config, shard_prefix, unit, present)
                ):
                    logger.info("%s: %s already complete — skipping", contract.name, label)
                    summary.skipped += 1
                    continue

                input_keys = _resolve_inputs(contract, config, shard_prefix, unit, present)
                summary.published += _run_unit(
                    contract, config, ctx, store, shard_prefix, unit, input_keys,
                )
                summary.processed += 1
            except NotReady as nr:
                producer = PRODUCER_OF.get(nr.suffix, "the upstream stage")
                logger.warning(
                    "%s: %s missing %s — %s has not produced it yet",
                    contract.name, label, nr.suffix, producer,
                )
                summary.not_ready += 1
                summary.not_ready_missing.add(nr.suffix)
            except InputContractError as e:
                # Actionable, not a crash — no traceback.
                logger.error("%s: %s refused: %s", contract.name, label, e)
                summary.failed += 1
            except Exception as e:  # one bad base must not stop the run
                logger.exception("%s: %s failed: %s", contract.name, label, e)
                summary.failed += 1

    if summary.not_ready:
        if summary.not_ready == summary.bases:
            missing = ", ".join(sorted(summary.not_ready_missing))
            producers = sorted(
                {PRODUCER_OF.get(s, "<upstream>") for s in summary.not_ready_missing}
            )
            logger.error(
                "%s: every base is missing %s. Run `womblex run-stage --stage %s` first.",
                contract.name, missing, "` / `".join(producers),
            )
        else:
            logger.warning(
                "%s: %d base(s) skipped — required inputs absent; the upstream "
                "stage may still be draining. Re-run when it has.",
                contract.name, summary.not_ready,
            )
    return summary


def run_stage_local(
    contract: StageContract,
    shard_dir: Path,
    config: WomblexConfig,
    *,
    ctx: RunContext | None = None,
) -> StageRunSummary:
    """Run *contract* over a local shard dir — one call, whole directory.

    The parity oracle for the remote path: the stages write per-base sidecars
    independently, so processing the directory in one call and processing it a
    base at a time produce the same bytes.
    """
    ctx = ctx or RunContext()
    summary = StageRunSummary(stage=contract.name)
    contract.run(shard_dir, config, ctx)
    summary.processed = 1
    summary.bases = 1
    return summary


__all__ = [
    "InputContractError",
    "NotReady",
    "StageRunSummary",
    "checkpoint_prefix_for",
    "remote_bases",
    "run_stage_local",
    "run_stage_remote",
]
