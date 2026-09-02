"""Which run produced this file: the run id, version, configuration and stage.

A run id exists today only as a directory name, so a shard copied out of its
run loses it and a file mixed in from another run is indistinguishable from a
native one. This stamps four facts into every pipeline Parquet's footer
key-value metadata at the moment it is written — run id, Womblex version,
configuration digest and the stage that wrote it — so attribution survives the
file being moved.

The keys share the ``womblex.*`` namespace ``store/source_provenance.py``
established (itself the convention ``store/register_manifest.py`` reads back
from the register ingests): one convention, not a second one. Footer metadata
is additive, so a reader that ignores it reads the file unchanged.

Two properties are deliberate:

**The digest is over the validated configuration, not the file bytes.** Pydantic
defaults and the AI-chunking validator both rewrite what the YAML supplied, so
the file is not what ran. Two runs whose YAML differs only in formatting, key
order or omitted defaults therefore produce one digest.

**``paths`` is excluded from the digest, and so is ``dataset.run_id``.** Paths
say where a deployment keeps its files, not what the pipeline does: a worker
extracts from a scratch directory it was handed, and digesting that would make
a distributed run's stamp differ from the local run of the same pipeline over
the same corpus. The run id is stamped in its own right, so folding it into the
digest would only make every run's digest unique and say nothing.

Attribution, not reproduction: outputs are not bit-reproducible (OCR and model
variability mean two runs over one corpus differ), so the stamp says what
produced a file, never that the file can be recomputed from it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace

from womblex import __version__
from womblex.store.source_provenance import NAMESPACE

RUN_ID_KEY = f"{NAMESPACE}.run_id"
VERSION_KEY = f"{NAMESPACE}.version"
CONFIG_DIGEST_KEY = f"{NAMESPACE}.config_digest"
STAGE_KEY = f"{NAMESPACE}.stage"

# Excluded from the digest: see the module docstring. `paths` is deployment
# location and `dataset.run_id` is the run's own identity, already a key.
_DIGEST_EXCLUDE: dict = {"paths": True, "dataset": {"run_id"}}


def config_digest(config: object) -> str:
    """``sha256:…`` over the validated configuration, minus the excluded keys.

    Takes a ``WomblexConfig`` structurally rather than by import: ``config.py``
    already reaches into this package, and the same duck-typing keeps
    ``IngestProvenance.from_config`` import-cycle free.
    """
    payload = config.model_dump(mode="json", exclude=_DIGEST_EXCLUDE)  # type: ignore[attr-defined]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class RunStamp:
    """The four facts a written file carries about the run that produced it.

    One stamp is declared per run and re-pointed at each stage that writes
    (:meth:`for_stage`), so run id, version and digest cannot drift between the
    files of one run.
    """

    run_id: str
    version: str
    config_digest: str
    stage: str

    @classmethod
    def declare(cls, run_id: str, config: object, *, stage: str) -> RunStamp:
        """Stamp for *run_id* under *config*, written by *stage*.

        Both strings are required: an unstamped file is honest, a file stamped
        with an empty run id or an unnamed stage is not.
        """
        if not str(run_id).strip():
            raise ValueError("run_id is empty; a stamp names the run or is not written")
        if not str(stage).strip():
            raise ValueError("stage is empty; a stamp names the writer or is not written")
        return cls(str(run_id).strip(), __version__, config_digest(config), str(stage).strip())

    def for_stage(self, stage: str) -> RunStamp:
        """The same run, stamped for another writer."""
        if not str(stage).strip():
            raise ValueError("stage is empty; a stamp names the writer or is not written")
        return replace(self, stage=str(stage).strip())

    def footer_metadata(self) -> dict[bytes, bytes]:
        """Namespaced footer metadata for a file this stage is writing."""
        return {
            RUN_ID_KEY.encode(): self.run_id.encode(),
            VERSION_KEY.encode(): self.version.encode(),
            CONFIG_DIGEST_KEY.encode(): self.config_digest.encode(),
            STAGE_KEY.encode(): self.stage.encode(),
        }


def read_footer_stamp(metadata: Mapping[bytes, bytes] | None) -> dict[str, str]:
    """Decode the run-stamp keys out of a Parquet footer.

    ``{}`` when the file carries none — written before this landed, or by a
    caller that declared no run. Partial stamps come back partial rather than
    being completed with invented values.
    """
    if not metadata:
        return {}
    decoded = {k.decode(): v.decode() for k, v in metadata.items()}
    names = (
        (RUN_ID_KEY, "run_id"),
        (VERSION_KEY, "version"),
        (CONFIG_DIGEST_KEY, "config_digest"),
        (STAGE_KEY, "stage"),
    )
    return {name: decoded[key] for key, name in names if key in decoded}


__all__ = [
    "CONFIG_DIGEST_KEY",
    "RUN_ID_KEY",
    "STAGE_KEY",
    "VERSION_KEY",
    "RunStamp",
    "config_digest",
    "read_footer_stamp",
]
