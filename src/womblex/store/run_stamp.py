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

**A downstream stage inherits its run rather than declaring one.** Extraction's
caller knows the run; a stage is handed a shard directory, and its own
``dataset.run_id`` is whatever configuration it was launched with — a copied
config would have it stamp a run it did not produce. So a sidecar takes the run
id and digest from the extraction shard it sits beside (:func:`stamp_for_sidecar`)
and supplies its own version and stage. That also makes a worker's stamp match
the local run's by construction rather than by both being handed the same YAML.

Attribution, not reproduction: outputs are not bit-reproducible (OCR and model
variability mean two runs over one corpus differ), so the stamp says what
produced a file, never that the file can be recomputed from it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex import __version__
from womblex.store.source_provenance import NAMESPACE

RUN_ID_KEY = f"{NAMESPACE}.run_id"
VERSION_KEY = f"{NAMESPACE}.version"
CONFIG_DIGEST_KEY = f"{NAMESPACE}.config_digest"
STAGE_KEY = f"{NAMESPACE}.stage"

# Excluded from the digest: see the module docstring. `paths` is deployment
# location and `dataset.run_id` is the run's own identity, already a key.
_DIGEST_EXCLUDE: dict = {"paths": True, "dataset": {"run_id"}}

# The siblings a downstream sidecar prefers to inherit its run from, in order.
# Spelled out rather than imported from `store/output.py`, which imports this
# module — two constants are cheaper than breaking the cycle.
_PREFERRED_SOURCES = (".elements.parquet", "._manifest.parquet")


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

    @classmethod
    def inherit(cls, run_id: str, digest: str, *, stage: str) -> RunStamp:
        """The stamp a downstream sidecar carries: the run it extends, by *stage*.

        A stage runs over a shard directory and has no independent knowledge of
        the run — its own ``dataset.run_id`` may be anything a copied config
        holds — so the run id and configuration digest are taken from the
        extraction shard the sidecar sits beside. Those two are facts about the
        *run*; the version is the Womblex that wrote these bytes, so it is the
        running one rather than the inherited one. A stage run at a later
        version than the extraction therefore says so.
        """
        if not str(run_id).strip():
            raise ValueError("run_id is empty; a stamp names the run or is not written")
        if not str(stage).strip():
            raise ValueError("stage is empty; a stamp names the writer or is not written")
        return cls(str(run_id).strip(), __version__, str(digest), str(stage).strip())

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


def stamp_from_footers(paths: Iterable[Path], stage: str) -> RunStamp | None:
    """The one run *paths* agree on, stamped for *stage*; ``None`` if there is none.

    ``None`` when no readable file carries a run, and also when the files name
    more than one run — a stamp naming one of several runs would be worse than
    no stamp, which is the rule the ingest-root footer already follows.
    """
    seen: set[tuple[str, str]] = set()
    for path in paths:
        try:
            existing = read_footer_stamp(pq.read_schema(str(path)).metadata)
        except (OSError, pa.ArrowInvalid):  # absent, unreadable, or not parquet
            continue
        if run_id := existing.get("run_id", ""):
            seen.add((run_id, existing.get("config_digest", "")))
    if len(seen) != 1:
        return None
    run_id, digest = seen.pop()
    return RunStamp.inherit(run_id, digest, stage=stage)


def stamp_for_sidecar(base_path: Path, stage: str) -> RunStamp | None:
    """The stamp a sidecar of *base_path* inherits, or ``None`` if there is none.

    *base_path* is the batch's base shard path (``batch-0001.parquet``). The
    run is read from a stamped sibling of that base: the extraction shard or
    its manifest by preference, and failing those any sibling of the batch,
    since every file of one run carries the same run id and digest.

    Falling back matters for the distributed path. A stage worker stages in
    only the inputs its contract declares, and five stages (``embed``,
    ``link``, ``pii``, ``quality``, ``graph-refresh``) do not declare the
    elements shard — so insisting on it would leave their sidecars unstamped
    on a worker and stamped locally, which is exactly the local/distributed
    divergence the stamp exists to rule out.

    ``None`` when no sibling carries a run, or when the siblings disagree — a
    sidecar of an unstamped shard is written unstamped, on the same terms
    extraction already writes one for a run it cannot name.
    """
    preferred = [base_path.parent / f"{base_path.stem}{s}" for s in _PREFERRED_SOURCES]
    for path in preferred:
        if (stamp := stamp_from_footers([path], stage)) is not None:
            return stamp
    others = sorted(set(base_path.parent.glob(f"{base_path.stem}.*.parquet")) - set(preferred))
    return stamp_from_footers(others, stage)


def sidecar_footer(base_path: Path, stage: str) -> dict[bytes, bytes] | None:
    """:func:`stamp_for_sidecar` as footer metadata, for a writer to pass on."""
    stamp = stamp_for_sidecar(base_path, stage)
    return stamp.footer_metadata() if stamp is not None else None


__all__ = [
    "CONFIG_DIGEST_KEY",
    "RUN_ID_KEY",
    "STAGE_KEY",
    "VERSION_KEY",
    "RunStamp",
    "config_digest",
    "read_footer_stamp",
    "sidecar_footer",
    "stamp_for_sidecar",
    "stamp_from_footers",
]
