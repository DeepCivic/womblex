"""The workflow object: schema, validation, and the table of copies `schedule-workflow` stores.

A *workflow* is a named recipe whose master copy is a file (`workflow_file`); the copy a
schedule's firings read is kept in the database DBOS's own tables live in
(`dbos_app.system_database_url`), so a local run needs no second database.

Schema and storage only (`docs/decisions-operations.md`, "A workflow is a file"): `triggers`
are scheduled by `workflow_schedules`, and nothing here runs a workflow. The table is created
if absent; there is no versioned migration yet.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, ValidationError

from womblex.cloud import dbos_app
from womblex.cloud.stage_contracts import STAGE_NAMES
from womblex.cloud.webhook import EVENTS
from womblex.config.process import StrictModel
from womblex.store.remote import validate_location_uri
from womblex.store.storage_options import secrets_outside_env

logger = logging.getLogger(__name__)

#: A workflow id becomes a primary key and, later, a DBOS schedule name: constrained like
#: a preset name or run id (`presets.is_safe_preset_name`, `store.retention.is_safe_run_id`).
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def is_safe_workflow_id(workflow_id: str) -> bool:
    """True if *workflow_id* is a single filesystem/key-safe token."""
    return workflow_id not in {"", ".", ".."} and _SAFE_ID.match(workflow_id) is not None


class WorkflowStep(StrictModel):
    """One downstream stage to run, in list order. Extraction is not a step: `input_binding` feeds it."""

    stage: str


class WorkflowInputBinding(StrictModel):
    """Where a run's documents come from: the input store, and the two knobs `enqueue_extraction` takes.

    `store` is the URI a firing lists and extracts from; only workers whose ingest root it is run
    the firing (`dbos_app.firing_queue`). Credentials come from the worker, never the workflow.
    """

    store: str | None = None
    ingest_prefix: str | None = None
    batch_size: int = Field(default=50, ge=1)


class WorkflowTrigger(StrictModel):
    """A standing cron trigger. Stored, not scheduled — see the module docstring."""

    cron: str
    timezone: str | None = None


class StoreCredentials(StrictModel):
    """Where an object store's key and secret are found: the names of two environment variables,
    never the values (a workflow names where a secret lives, never the secret)."""

    key_env: str
    secret_env: str


class ObjectStoreDestination(StrictModel):
    """Deliver the run's egress bundle (`store/egress.py`) to `uri`, under the run id.

    `credentials` names where an S3 key and secret are found; without it the store's own
    environment applies, as for any store. `storage_options` are fsspec's own for this
    location, passed unchanged (`store/storage_options.py`); a secret in them is written
    `{"env": "NAME"}`, never as its value.
    """

    kind: Literal["object_store"]
    name: str
    uri: str
    include_sources: bool = True
    credentials: StoreCredentials | None = None
    storage_options: dict[str, Any] | None = None


class PostgresDestination(StrictModel):
    """Upsert one row per embedded chunk, with its embedding, into `table` (`cloud/destination_postgres.py`).

    `dsn_env` names the environment variable holding the connection string, never the string.
    `text_layer` is `masked` (the `pii` step's clean text; the default) or `raw` (the chunk text).
    """

    kind: Literal["postgres"]
    name: str
    dsn_env: str
    table: str = "womblex_chunks"
    text_layer: Literal["masked", "raw"] = "masked"


class WebhookDestination(StrictModel):
    """Send the run's events to `url` (`cloud/webhook.py`), signed with the key in `secret_env`.

    `events` narrows what is sent, so a consumer can take the coarse events alone; `None` sends them all.
    """

    kind: Literal["webhook"]
    name: str
    url: str
    secret_env: str
    events: tuple[str, ...] | None = None


#: What an inbound hook may be sent (Requirement 35); each is acted on by `api/routes_inbound.py`.
INBOUND_EVENTS = ("documents_added", "run_requested", "rerun_requested", "delivery_received")

#: Set to `1` in the environment of a proof of concept to turn the inbound checks off. Read only from
#: the environment, never from a spec, so a stored workflow cannot turn it on.
INSECURE_INBOUND_ENV = "WOMBLEX_INSECURE_INBOUND"


def insecure_inbound() -> bool:
    return os.environ.get(INSECURE_INBOUND_ENV) == "1"


class InboundHook(StrictModel):
    """An endpoint other systems post to (`POST /v1/hooks/<workflow>/<name>`), signed with the key in `secret_env`.

    `events` are the ones the hook accepts. Like a webhook destination it names the environment
    variable holding its key, never the key; the signature is HMAC-SHA256 over the raw body.
    """

    name: str
    secret_env: str
    events: tuple[str, ...]


#: The destination kinds a workflow can name, each added in the merge that delivers to it.
WorkflowDestination = Annotated[
    ObjectStoreDestination | PostgresDestination | WebhookDestination, Field(discriminator="kind")]

_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


class StoredWorkflowSpec(StrictModel):
    """The validated shape of a workflow's definition — everything but id and timestamps."""

    preset: str
    config: dict[str, Any] = Field(default_factory=dict)
    input_binding: WorkflowInputBinding = Field(default_factory=WorkflowInputBinding)
    steps: tuple[WorkflowStep, ...]
    triggers: tuple[WorkflowTrigger, ...] = ()
    destinations: tuple[WorkflowDestination, ...] = ()
    collection: str | None = None
    inbound: tuple[InboundHook, ...] = ()


def _check_steps(spec: StoredWorkflowSpec) -> None:
    if not spec.steps:
        raise ValueError("a workflow needs at least one step")
    stages = [s.stage for s in spec.steps]
    unknown = set(stages) - set(STAGE_NAMES)
    if unknown:
        raise ValueError(f"unknown stage(s) in steps: {sorted(unknown)} — known: {sorted(STAGE_NAMES)}")
    repeated = sorted({s for s in stages if stages.count(s) > 1})
    if repeated:
        raise ValueError(f"stage(s) listed more than once in steps: {repeated}")


def _check_triggers(spec: StoredWorkflowSpec) -> None:
    # The parser DBOS's own scheduler validates with, so a stored trigger is one it will accept.
    # DBOS 3.3 depends on the `croniter` package; 3.2 vendors it as `dbos._croniter`.
    try:
        from croniter import croniter  # type: ignore[import-untyped, import-not-found, unused-ignore]
    except ImportError:
        from dbos._croniter import croniter  # type: ignore[import-not-found, attr-defined, no-redef, unused-ignore]

    for trigger in spec.triggers:
        if not croniter.is_valid(trigger.cron, second_at_beginning=True):
            raise ValueError(f"not a valid cron expression: {trigger.cron!r}")
        if trigger.timezone is not None:
            try:
                ZoneInfo(trigger.timezone)
            except (ZoneInfoNotFoundError, ValueError) as e:
                raise ValueError(f"unknown timezone: {trigger.timezone!r}") from e


def _check_destinations(spec: StoredWorkflowSpec) -> None:
    names = [d.name for d in spec.destinations]
    for name in names:
        if not is_safe_workflow_id(name):
            raise ValueError(f"destination name must be one safe token, got {name!r}")
    if repeated := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"destination name(s) used more than once: {repeated}")
    stages = {s.stage for s in spec.steps}
    for d in spec.destinations:
        if isinstance(d, ObjectStoreDestination) and "@" in urlsplit(d.uri).netloc:
            raise ValueError(f"destination {d.name!r}: a URI names no credentials; use `credentials`")
        if isinstance(d, ObjectStoreDestination) and (literal := secrets_outside_env(d.storage_options)):
            raise ValueError(
                f"destination {d.name!r}: storage_options.{literal[0]} holds a secret as a value; "
                'write it as {"env": "NAME"} (a workflow names where a secret lives, never the secret)'
            )
        if isinstance(d, ObjectStoreDestination):
            try:  # `s3:/bucket` would otherwise deliver to a local folder named `s3:`
                validate_location_uri(d.uri, require_backend=False)  # the workers carry the backend; check_destinations asks where it runs
            except ValueError as e:
                raise ValueError(f"destination {d.name!r}: {e}") from e
        if isinstance(d, WebhookDestination):
            url = urlsplit(d.url)
            if url.scheme not in {"http", "https"} or not url.netloc or "@" in url.netloc:
                raise ValueError(f"destination {d.name!r}: a webhook URL is http(s) and names no credentials")
            if d.events is not None and (unknown := sorted(set(d.events) - set(EVENTS))):
                raise ValueError(f"destination {d.name!r}: unknown event(s) {unknown}; events are {list(EVENTS)}")
            if d.events == ():
                raise ValueError(f"destination {d.name!r}: `events` is empty; leave it out to send every event")
        if isinstance(d, PostgresDestination):
            if not _TABLE_NAME.match(d.table):
                raise ValueError(f"destination {d.name!r}: table must be `name` or `schema.name`, got {d.table!r}")
            if "embed" not in stages:
                raise ValueError(f"destination {d.name!r}: a Postgres destination needs the `embed` step")
            if d.text_layer == "masked" and "pii" not in stages:
                raise ValueError(f"destination {d.name!r}: masked text needs the `pii` step "
                                 "(or `text_layer: raw` to send the chunk text)")


def _check_collection(spec: StoredWorkflowSpec) -> None:
    from womblex.store.retention import is_safe_run_id

    if spec.collection is not None and not (is_safe_workflow_id(spec.collection) and is_safe_run_id(spec.collection)):
        raise ValueError(f"collection must be one safe token, got {spec.collection!r}")


def _check_inbound(spec: StoredWorkflowSpec) -> None:
    names = [h.name for h in spec.inbound]
    for name in names:
        if not is_safe_workflow_id(name):
            raise ValueError(f"inbound hook name must be one safe token, got {name!r}")
    if repeated := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"inbound hook name(s) used more than once: {repeated}")
    for h in spec.inbound:
        if not h.events or (unknown := sorted(set(h.events) - set(INBOUND_EVENTS))):
            raise ValueError(f"inbound hook {h.name!r}: events must be a non-empty subset of {list(INBOUND_EVENTS)}"
                             + (f", not {unknown}" if h.events else ""))
        if not h.secret_env.strip():
            raise ValueError(f"inbound hook {h.name!r}: secret_env names the variable holding its key")
    keys = [h.secret_env for h in spec.inbound]
    if shared := sorted({k for k in keys if keys.count(k) > 1}):
        _shared(f"inbound hooks of one workflow share a secret_env: {shared}")


def _shared(problem: str) -> None:
    """Refuse a shared inbound secret, or only warn under `WOMBLEX_INSECURE_INBOUND=1`."""
    if not insecure_inbound():
        raise ValueError(problem + " (each hook needs its own; set WOMBLEX_INSECURE_INBOUND=1 for a proof of concept)")
    logger.warning("%s (allowed by %s)", problem, INSECURE_INBOUND_ENV)


def check_shared_secrets(workflow_id: str, spec: StoredWorkflowSpec, others: list[StoredWorkflow]) -> None:
    """Refuse a hook whose `secret_env` another stored workflow's hook names. The signature is over the
    body alone, so a shared key lets one workflow's sender post to another's hook."""
    mine = {h.secret_env for h in spec.inbound}
    for other in others:
        if other.id != workflow_id and (clash := sorted(mine & {h.secret_env for h in other.spec.inbound})):
            _shared(f"inbound secret_env {clash} is also named by stored workflow {other.id!r}")


def inbound_problems(spec: StoredWorkflowSpec) -> list[str]:
    """Hooks whose key is not set in this environment (a pre-launch check, as for a webhook destination)."""
    return [f"inbound hook {h.name!r}: {h.secret_env} not set" for h in spec.inbound if not os.environ.get(h.secret_env)]


def combine_config(preset_config: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """A preset's config with a workflow's own `config` laid over it, merged key by key.

    Mappings merge recursively, so `enrichment: {model: x}` changes the model and keeps what the
    preset set; a list or scalar replaces the preset's. `null` drops the key, so the field takes
    its schema default (not a literal `None`). `dataset` and `paths` are run identity, not recipe,
    and are dropped. The one combination rule: the save-time check and the runner both use it.
    """
    def merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
        out = dict(base)
        for k, v in top.items():
            b = out.get(k)
            if v is None:
                out.pop(k, None)
            else:
                out[k] = merge(b, v) if isinstance(b, dict) and isinstance(v, dict) else v
        return out

    return {k: v for k, v in merge(preset_config, overlay).items() if k not in {"dataset", "paths"}}


def _check_config_overlay(spec: StoredWorkflowSpec) -> None:
    """Check `spec.config` as a preset is checked, over a built-in preset's overlay when one is named.

    An unknown name may be an operator-saved preset (unreadable without a store), so it passes here;
    the runner resolves it and re-validates the full combination.
    """
    from womblex.presets import _validate_overlay, get_preset

    builtin = get_preset(spec.preset)
    _validate_overlay(combine_config(builtin.config if builtin else {}, spec.config))


def check_spec(spec: StoredWorkflowSpec) -> None:
    """The checks beyond the pydantic shape: steps, triggers and the config overlay.

    Run by `validate_workflow_spec`, by `WorkflowStore.save` (a spec built without
    `validate_workflow_spec` is not stored unchecked) and by the runner on load
    (a row written before a stage was renamed or removed is refused, not run).
    """
    _check_steps(spec)
    _check_triggers(spec)
    _check_destinations(spec)
    _check_collection(spec)
    _check_inbound(spec)
    _check_config_overlay(spec)


def validate_workflow_spec(raw: dict[str, Any]) -> StoredWorkflowSpec:
    """Parse and validate *raw* into a `StoredWorkflowSpec`.

    Raises `pydantic.ValidationError` on a structurally bad spec or config
    overlay, and `ValueError` on bad steps, cron expressions or timezones.
    """
    spec = StoredWorkflowSpec(**raw)
    check_spec(spec)
    return spec


def workflow_digest(spec: StoredWorkflowSpec) -> str:
    """``sha256:…`` over *spec* — the same canonical-JSON recipe as `store.run_stamp.config_digest`."""
    payload = spec.model_dump(mode="json")
    if not payload["destinations"]:  # a workflow that names none keeps the digest it had before they existed
        del payload["destinations"]
    del payload["collection"]  # where documents are kept, not what runs on them: the digest is the recipe's
    if not payload["inbound"]:  # a workflow that names no hook keeps the digest it had before they existed
        del payload["inbound"]
    for destination in payload.get("destinations", ()):  # ...and one that gives no options, as before they existed
        if destination.get("storage_options") is None:
            destination.pop("storage_options", None)
        if destination.get("events") is None:  # ...and a webhook that names no events, as before they could be chosen
            destination.pop("events", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class StoredWorkflow:
    """One workflow as read: from its file (`workflow_file`), or a scheduled copy's row."""

    id: str
    spec: StoredWorkflowSpec
    digest: str
    created_at: str = ""
    updated_at: str = ""
    owner: str | None = None  # the API caller that stored it; `None` is admin (the CLI)

    @property
    def run_id(self) -> str:
        """The run a firing appends to: the collection, else the workflow's own id."""
        return self.spec.collection or self.id


_TABLE = "womblex_workflows"


def _ddl() -> Any:
    import sqlalchemy as sa

    metadata = sa.MetaData()
    return sa.Table(
        _TABLE, metadata,
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("spec_json", sa.Text, nullable=False),
        sa.Column("digest", sa.Text, nullable=False),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("updated_at", sa.Text, nullable=False),
        sa.Column("owner", sa.Text, nullable=True),
    ), metadata


def ensure_workflow_schema(dsn: str | None) -> None:
    """Create the workflow table if it is not already there. Idempotent; writers call it, readers do not."""
    import sqlalchemy as sa

    table, metadata = _ddl()
    engine = sa.create_engine(dbos_app.system_database_url(dsn))
    try:
        metadata.create_all(engine, tables=[table])
        if "owner" not in {c["name"] for c in sa.inspect(engine).get_columns(_TABLE)}:  # a table made before owners
            with engine.begin() as conn:
                conn.execute(sa.text(f"ALTER TABLE {_TABLE} ADD COLUMN owner TEXT"))
    finally:
        engine.dispose()


#: Spec fields a row may still hold from before they were removed; dropped on read, warned if set.
_RETIRED_FIELDS = ("conditions",)


def _spec_from_row(workflow_id: str, spec_json: str) -> tuple[StoredWorkflowSpec, bool]:
    """The row's spec, and whether a retired field that was set had to be dropped.

    Destinations stored before Phase R removed them have another shape; they are dropped too.
    """
    raw = json.loads(spec_json)
    dropped = False
    for name in _RETIRED_FIELDS:
        if raw.pop(name, None):
            dropped = True
            logger.warning("workflow %s: stored %s are no longer supported and are ignored", workflow_id, name)
    try:
        return StoredWorkflowSpec(**raw), dropped
    except ValidationError:
        if not raw.pop("destinations", None):
            raise
        logger.warning("workflow %s: stored destinations predate the current kinds and are ignored", workflow_id)
        return StoredWorkflowSpec(**raw), True


def _row_to_stored(row: Any) -> StoredWorkflow:
    spec, dropped = _spec_from_row(row.id, row.spec_json)
    digest = row.digest
    if dropped:  # the stored digest covers the dropped field; stamp the spec that will run
        digest = workflow_digest(spec)
        logger.warning("workflow %s: digest %s recomputed as %s without the retired fields",
                       row.id, row.digest, digest)
    return StoredWorkflow(
        id=row.id,
        spec=spec,
        digest=digest,
        created_at=row.created_at,
        updated_at=row.updated_at,
        owner=row.owner,
    )


class WorkflowStore:
    """Create, read and delete stored workflows through one SQLAlchemy engine."""

    def __init__(self, dsn: str | None, *, connect_timeout: float | None = None):
        import sqlalchemy as sa

        self._table, _ = _ddl()
        self._has_owner: bool | None = None
        url = dbos_app.system_database_url(dsn)
        if connect_timeout is not None and not url.startswith("sqlite"):
            self._engine = sa.create_engine(
                sa.make_url(url).set(drivername="postgresql+psycopg"),
                connect_args={"connect_timeout": int(connect_timeout)},
                pool_pre_ping=True,
            )
        else:
            self._engine = sa.create_engine(url)

    def close(self) -> None:
        self._engine.dispose()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _select(self, conn: Any) -> Any:
        """Every column, with `owner` read as null from a table made before owners: readers do not migrate."""
        import sqlalchemy as sa

        if self._has_owner is None and sa.inspect(conn).has_table(_TABLE):
            self._has_owner = "owner" in {c["name"] for c in sa.inspect(conn).get_columns(_TABLE)}
        if self._has_owner is False:
            return sa.select(*[c for c in self._table.c if c.name != "owner"], sa.null().label("owner"))
        return sa.select(self._table)

    def _get_row(self, conn: Any, workflow_id: str) -> Any:
        return conn.execute(self._select(conn).where(self._table.c.id == workflow_id)).first()

    def save(self, workflow_id: str, spec: StoredWorkflowSpec, *, owner: str | None = None) -> StoredWorkflow:
        """Create or replace *workflow_id*'s definition, stored by *owner* (`None`: admin).

        Raises `ValueError` if *workflow_id* is unsafe or the spec fails `check_spec`
        (`pydantic.ValidationError` for a bad config overlay).
        """
        import sqlalchemy.exc

        if not is_safe_workflow_id(workflow_id):
            raise ValueError(f"unsafe workflow id: {workflow_id!r}")
        check_spec(spec)
        if spec.inbound:
            check_shared_secrets(workflow_id, spec, self.list())
        try:
            self._write(workflow_id, spec, owner)
        except sqlalchemy.exc.IntegrityError:
            # A concurrent save created the row first; the retry replaces it.
            self._write(workflow_id, spec, owner)
        stored = self.get(workflow_id)
        assert stored is not None  # just written and committed
        return stored

    def _write(self, workflow_id: str, spec: StoredWorkflowSpec, owner: str | None) -> None:
        import sqlalchemy as sa

        now = datetime.now(UTC).isoformat()
        with self._engine.begin() as conn:
            existing = self._get_row(conn, workflow_id)
            created_at = existing.created_at if existing is not None else now
            values = {
                "spec_json": json.dumps(spec.model_dump(mode="json")),
                "digest": workflow_digest(spec),
                "created_at": created_at,
                "updated_at": now,
                "owner": owner,
            }
            if existing is not None:
                conn.execute(
                    sa.update(self._table).where(self._table.c.id == workflow_id).values(**values)
                )
            else:
                conn.execute(sa.insert(self._table).values(id=workflow_id, **values))

    def get(self, workflow_id: str) -> StoredWorkflow | None:
        """One workflow by id, or `None` if absent."""
        with self._engine.connect() as conn:
            row = self._get_row(conn, workflow_id)
        return None if row is None else _row_to_stored(row)

    def list(self) -> list[StoredWorkflow]:
        """Every workflow, oldest first."""
        with self._engine.connect() as conn:
            rows = conn.execute(self._select(conn).order_by(self._table.c.created_at)).fetchall()
        return [_row_to_stored(r) for r in rows]

    def delete(self, workflow_id: str) -> None:
        """Delete *workflow_id*. No-op if absent."""
        import sqlalchemy as sa

        with self._engine.begin() as conn:
            conn.execute(sa.delete(self._table).where(self._table.c.id == workflow_id))


__all__ = [
    "INBOUND_EVENTS",
    "INSECURE_INBOUND_ENV",
    "InboundHook",
    "ObjectStoreDestination",
    "PostgresDestination",
    "StoreCredentials",
    "StoredWorkflow",
    "StoredWorkflowSpec",
    "WebhookDestination",
    "WorkflowDestination",
    "WorkflowInputBinding",
    "WorkflowStep",
    "WorkflowStore",
    "WorkflowTrigger",
    "check_shared_secrets",
    "check_spec",
    "combine_config",
    "ensure_workflow_schema",
    "inbound_problems",
    "insecure_inbound",
    "is_safe_workflow_id",
    "validate_workflow_spec",
    "workflow_digest",
]
