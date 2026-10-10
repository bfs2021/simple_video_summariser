"""Shared pytest fixtures (architecture.md 13; task #7).

Only integration tests touch Postgres. Pytest fixtures are lazy — a fixture
only runs if a test actually requests it (directly or through another
fixture) — so with `pytest -m "not integration"` no test asks for
`postgres_dsn`, `postgres_container` is never instantiated, and Docker is
never touched. This is what makes the fast path safe to run with Docker
Desktop stopped.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql
from testcontainers.community.postgres import PostgresContainer

# ``pytester`` is a built-in pytest plugin (no dependency); the shuffle plugin
# is registered here so every run is shuffled (issue #72).
pytest_plugins = ["pytester", "tests.shuffle_plugin"]

# Same image tag as compose.yml (architecture.md 11.4, 16.1: >=18.6).
_POSTGRES_IMAGE = "postgres:18-alpine"


def _probe_postgres(dsn: str, connect_timeout_sec: int) -> None:
    with psycopg.connect(dsn, connect_timeout=connect_timeout_sec) as conn:
        conn.execute("SELECT 1").fetchone()


def _wait_for_postgres(
    dsn: str,
    container_id: str,
    *,
    probe: Callable[[str, int], None] = _probe_postgres,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    deadline_sec: float = 60.0,
    connect_timeout_sec: int = 5,
) -> None:
    parts = urlsplit(dsn)
    endpoint = f"{parts.hostname}:{parts.port}"
    started_at = monotonic()
    deadline = started_at + deadline_sec
    attempts = 0
    delay = 0.1
    last_exc: psycopg.OperationalError | None = None

    while monotonic() < deadline:
        attempts += 1
        try:
            probe(dsn, connect_timeout_sec)
            return
        except psycopg.OperationalError as exc:
            last_exc = exc

        now = monotonic()
        if now >= deadline:
            break
        sleep(min(delay, 1.0, deadline - now))
        delay = min(delay * 2, 1.0)

    elapsed = monotonic() - started_at
    raise RuntimeError(
        f"Postgres container {container_id[:12]} was not reachable at {endpoint} "
        f"after {elapsed:.1f}s and {attempts} attempts"
    ) from last_exc


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    """One Postgres container for the whole test session.

    Root cause of #74: Docker Desktop (WSL) publishes the host port a moment
    after the container reports ready, so the first connect can be refused.
    The fixture waits for a real connection and ``SELECT 1`` on the mapped
    port, up to 60 s, then fails explicitly naming the endpoint. One run of
    the suite verifies it; no repeat loops (testing-guidelines.md).
    """
    with PostgresContainer(_POSTGRES_IMAGE, driver=None) as container:
        _wait_for_postgres(container.get_connection_url(), container.get_container_id())
        yield container


@pytest.fixture
def postgres_dsn(postgres_container: PostgresContainer) -> Iterator[str]:
    """A ``postgresql://`` DSN for a fresh, empty database.

    Each test gets its own database on the shared container, created before
    and dropped after, so no test can see another test's data.
    """
    admin_dsn = postgres_container.get_connection_url()
    db_name = f"test_{uuid.uuid4().hex}"

    with psycopg.connect(admin_dsn, autocommit=True) as admin_conn:
        admin_conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
    try:
        yield _with_dbname(admin_dsn, db_name)
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as admin_conn:
            admin_conn.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(db_name))
            )


def _with_dbname(dsn: str, db_name: str) -> str:
    parts = urlsplit(dsn)
    return urlunsplit(parts._replace(path=f"/{db_name}"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect ``@pytest.mark.image`` tests unless ``-m`` names ``image``.

    An image build downloads packages from PyPI and Debian and takes minutes,
    so it is opt-in (issue #55). Same pattern as the ``whisper`` marker in
    ``tests/adapters/transcription/conftest.py``: a hook rather than
    ``addopts``, because ``-m`` is single-valued and a caller's own
    ``-m "not integration"`` would silently replace an ``addopts`` one.
    """
    markexpr = config.getoption("markexpr") or ""
    if "image" in markexpr:
        return

    keep = [item for item in items if "image" not in item.keywords]
    deselected = [item for item in items if "image" in item.keywords]
    if deselected:
        items[:] = keep
        config.hook.pytest_deselected(items=deselected)
