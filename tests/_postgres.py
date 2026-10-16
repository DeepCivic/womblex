"""A fresh Postgres database per test, from CI's service (`WOMBLEX_DB_DSN`); skipped without one."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import pytest


def admin_dsn() -> str:
    dsn = os.environ.get("WOMBLEX_DB_DSN", "")
    if not dsn.startswith(("postgres://", "postgresql://")):
        pytest.skip("needs a Postgres system database (WOMBLEX_DB_DSN)")
    return dsn


@contextmanager
def fresh_database() -> Iterator[str]:
    """A new empty database on the service, dropped afterwards; yields its URL."""
    import psycopg
    import sqlalchemy as sa

    admin = admin_dsn()
    name = f"womblex_t_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    try:
        yield sa.make_url(admin).set(database=name).render_as_string(hide_password=False)
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
