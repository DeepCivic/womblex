"""Where a deployment keeps its run state: the settings dispatch and the service API share.

Fastapi-free, so a worker can import it. A deployment names exactly one run
source (a local ``output_root`` or an object-store ``store_uri``), and may add
a job queue, an ingest location and a directory of operator-saved presets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from womblex.store.remote import assert_disjoint_locations, is_remote_uri

#: `pathlib.Path` mangles an object-store URI (`Path("s3://foo")` collapses
#: to `s3:/foo`), so an object-store field is filled with this placeholder
#: instead and named in a header comment rather than written where it would lie.
_PATHS_PLACEHOLDER = "."


@dataclass(frozen=True)
class DeploymentSettings:
    """Where run state lives. Exactly one of ``output_root`` / ``store_uri`` is set.

    ``db_dsn`` is the optional job queue. It is orthogonal to the run source: a
    queue is present whenever one is configured and absent otherwise.

    ``presets_dir`` is where operator-saved presets sit in *local* mode; remote
    mode keeps them under the store's own ``presets/`` prefix instead.

    ``ingest_uri`` names where source documents arrive from, distinct from
    ``store_uri`` / ``output_root``. ``None`` means nothing can be dispatched.
    When both it and a store are set they must be disjoint of the store's
    effective ``runs/`` prefix, enforced here at construction.

    ``store_options`` and ``ingest_options`` are each location's own fsspec ``storage_options``
    (`store/storage_options.py`) as written, ``{"env": NAME}`` references unresolved; ``None``
    leaves the environment's (`WOMBLEX_STORE_STORAGE_OPTIONS`, `WOMBLEX_INGEST_STORAGE_OPTIONS`) to apply.
    """

    output_root: Path | None
    store_uri: str | None
    db_dsn: str | None = None
    presets_dir: Path | None = None
    ingest_uri: str | None = None
    store_options: dict[str, Any] | None = field(default=None, compare=False)
    ingest_options: dict[str, Any] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if bool(self.output_root) == bool(self.store_uri):
            raise ValueError("DeploymentSettings needs exactly one of output_root or store_uri")
        if self.ingest_uri and self.store_uri:
            assert_disjoint_locations(self.ingest_uri, self.store_uri)

    @property
    def is_remote(self) -> bool:
        return self.store_uri is not None

    @property
    def presets_writable(self) -> bool:
        """Whether presets can be saved: always on a store, locally only with a ``presets_dir``."""
        return self.is_remote or self.presets_dir is not None


def deployment_paths(settings: DeploymentSettings) -> tuple[dict[str, str], list[str]]:
    """This deployment's `paths` section, plus env vars to note instead of a
    mangled object-store URI. A local folder is written verbatim.

    The stage list a dispatch derives has to come from the config *as the CLI
    would load it*, which means this paths injection.
    """
    env_vars: list[str] = []

    if settings.ingest_uri and is_remote_uri(settings.ingest_uri):
        input_root = _PATHS_PLACEHOLDER
        env_vars.append("$WOMBLEX_INGEST_URI")
    elif settings.ingest_uri:
        input_root = settings.ingest_uri
    else:
        input_root = str(settings.output_root) if settings.output_root else _PATHS_PLACEHOLDER

    if settings.store_uri and is_remote_uri(settings.store_uri):
        output_root = _PATHS_PLACEHOLDER
        env_vars.append("$WOMBLEX_STORE_URI")
    elif settings.store_uri:
        output_root = settings.store_uri
    else:
        output_root = str(settings.output_root) if settings.output_root else _PATHS_PLACEHOLDER

    paths = {
        "input_root": input_root,
        "output_root": output_root,
        "checkpoint_dir": _PATHS_PLACEHOLDER,
    }
    return paths, env_vars
