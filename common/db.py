"""Database connections (architecture.md §2; issue #11).

``connect()`` is the only place that opens a psycopg connection from
``DATABASE_URL``. Callers (``PostgresQueue``, the repo layer, ...) receive a
connection; per architecture.md §5's "Database connection" acceptance
criterion, none of them create their own.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg_pool import ConnectionPool

from common.config import get_settings


def connect(**kwargs: Any) -> psycopg.Connection[Any]:
    """Open a new psycopg connection to ``get_settings().DATABASE_URL``.

    psycopg accepts the ``postgresql://`` scheme directly - unlike
    SQLAlchemy, which ``migrations/env.py`` must rewrite to
    ``postgresql+psycopg://`` - so the DSN is passed through unchanged.
    ``kwargs`` (e.g. ``connect_timeout``, ``options``) go to
    ``psycopg.connect`` as extra connection parameters (#39's ``/healthz``);
    a keyword overrides the same parameter in the DSN, so ``options=``
    replaces any ``options`` set in ``DATABASE_URL``.
    """
    return psycopg.connect(get_settings().DATABASE_URL.get_secret_value(), **kwargs)


def _reset_connection(conn: psycopg.Connection[Any]) -> None:
    """Undo per-request session state before a connection goes back to the pool.

    ``read_only``, ``isolation_level`` and ``autocommit`` live on the
    connection object, not in the server session, so a read route's
    ``conn.read_only = True`` would otherwise make the next write route fail.
    """
    conn.read_only = None
    conn.isolation_level = None
    conn.deferrable = None
    conn.autocommit = False


def create_pool(*, connect_timeout: int, statement_timeout_ms: int) -> ConnectionPool[Any]:
    """A bounded, *unopened* pool of connections to ``DATABASE_URL`` (#95).

    Only the API uses it; CLI, workers, planner and migrate keep ``connect()``.
    Sizes and the wait timeout come from settings at call time. The caller
    opens it (``open(wait=False)``, so Postgres being down does not stop the
    app starting) and closes it. ``statement_timeout`` is a startup option,
    so it is set once when the pool creates a connection, not per request.
    ``check`` replaces a connection that went bad while idle before it is
    handed out; ``reset`` clears per-request session state on return.
    """
    settings = get_settings()
    return ConnectionPool(
        settings.DATABASE_URL.get_secret_value(),
        kwargs={
            "connect_timeout": connect_timeout,
            "options": f"-c statement_timeout={statement_timeout_ms}",
        },
        min_size=settings.DB_POOL_MIN_SIZE,
        max_size=settings.DB_POOL_MAX_SIZE,
        timeout=settings.DB_POOL_TIMEOUT_SEC,
        check=ConnectionPool.check_connection,
        reset=_reset_connection,
        name="api",
        open=False,
    )
