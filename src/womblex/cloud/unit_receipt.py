"""Whether a stage unit's outputs are already published for the work about to run.

The fleet's skip rule. A unit that runs on a store leaves a receipt at
``<run>/receipts/<stage>/<unit>.json``: the Womblex version, the digest of the config the stage
ran under (a sidecar's `config_digest` is the run's, inherited from extraction), and the
fingerprint of every input it read and output it published. A unit is skipped only when its
receipt matches all of these as they are now, so a changed config, a rewritten input (an upstream
retry, a rechunk), a rewritten output or a new version runs it again, and an output with no receipt
(written before receipts, or by `--shards`) is never skipped. A receipt names the stage workflow that
wrote it, and a unit is never skipped against its own workflow's receipt: a step recovered after
publishing runs again, as before receipts, so its checksums are recorded. The receipt sits beside the run's
`logs/`, outside `documents/`, so the sidecars stay byte-identical to a local run's.
"""

from __future__ import annotations

import io
import json
import logging
from typing import TYPE_CHECKING, Any

from womblex import __version__
from womblex.cloud.stage_contracts import MutationMode, StageContract
from womblex.store.run_stamp import config_digest

if TYPE_CHECKING:
    from womblex.config import WomblexConfig
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)


def receipt_key(shard_prefix: str, stage: str, unit: list[str]) -> str:
    """Where *unit*'s receipt lives: under the run's output dir, beside `logs/`."""
    label = unit[0] if len(unit) == 1 else "run"
    return f"{shard_prefix.rsplit('/', 1)[0]}/receipts/{stage}/{label}.json"


def _fingerprints(store: RemoteStore, shard_prefix: str, keys: list[str]) -> dict[str, str]:
    """Each key's fingerprint (`RemoteStore.fingerprints`), by file name."""
    return {k.rsplit("/", 1)[-1]: v for k, v in store.fingerprints(keys, under=shard_prefix).items()}


def _receipt(config: WomblexConfig, inputs: dict[str, str], outputs: dict[str, str]) -> dict[str, Any]:
    return {"version": __version__, "config_digest": config_digest(config), "inputs": inputs, "outputs": outputs}


def write_receipt(
    contract: StageContract, config: WomblexConfig, store: RemoteStore, shard_prefix: str,
    unit: list[str], inputs: dict[str, str], published: list[str], by: str | None = None,
) -> None:
    """Record what *unit* ran from (*inputs*, fingerprinted before it ran) and published, written *by*
    a stage workflow. Never raises: the outputs are published, and a unit with no receipt only runs again."""
    key = receipt_key(shard_prefix, contract.name, unit)
    try:
        body = {**_receipt(config, inputs, _fingerprints(store, shard_prefix, published)), "by": by}
        store.write_stream(io.BytesIO(json.dumps(body, sort_keys=True).encode()), key)
    except Exception as exc:  # bookkeeping: the unit's outputs stand
        logger.warning("%s: no receipt written for %s (%s); the unit will run again if dispatched again",
                       contract.name, key, exc)


def input_fingerprints(store: RemoteStore, shard_prefix: str, keys: list[str]) -> dict[str, str]:
    """The inputs' fingerprints as a receipt records them, taken before the unit runs."""
    return _fingerprints(store, shard_prefix, keys)


def already_published(
    contract: StageContract, config: WomblexConfig, store: RemoteStore, shard_prefix: str, unit: list[str],
    by: str | None = None,
) -> bool:
    """True when *unit*'s receipt matches this version, config, its inputs and its outputs, and was
    written by a stage workflow other than *by*."""
    from womblex.cloud.stage_runner import InputContractError, NotReady, _key, _outputs_present, _resolve_inputs

    if contract.mutation is not MutationMode.SIDECAR or not contract.skip_if_output_present:
        return False
    key = receipt_key(shard_prefix, contract.name, unit)
    present = set(store.list_files(shard_prefix, "*.parquet"))
    if not store.exists(key) or not _outputs_present(contract, config, shard_prefix, unit, present):
        return False
    try:
        inputs = _resolve_inputs(contract, config, shard_prefix, unit, present)
        recorded = json.loads(store.read_text(key))
    except (NotReady, InputContractError, ValueError):
        return False
    if not isinstance(recorded, dict):
        return False
    writer = recorded.pop("by", None)
    if writer is not None and writer == by:  # its own receipt: a step recovered after publishing runs again
        return False
    outputs = [_key(shard_prefix, stem, suffix) for stem in unit for suffix in contract.outputs(config)]
    current = _receipt(config, _fingerprints(store, shard_prefix, inputs), _fingerprints(store, shard_prefix, outputs))
    return bool(recorded == current)


__all__ = ["already_published", "input_fingerprints", "receipt_key", "write_receipt"]
