"""Shared fixtures for the API tests (issue #39).

Fast tests never touch Postgres: ``DATABASE_URL`` points at a closed local
port, and any test that needs a database overrides ``deps.get_conn``. Log
output goes to an in-memory stream so tests can assert on the JSON lines.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

import services.api.main as api_main
from common.config import Settings, get_settings
from common.logging import configure_logging
from tests.common.repo.conftest import head_dsn

__all__ = ["head_dsn"]

UNREACHABLE_DSN = "postgresql://user:SECRETPW@127.0.0.1:1/db"


class LogSink:
    def __init__(self) -> None:
        self.stream = io.StringIO()
        self.configure_calls = 0

    def records(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.stream.getvalue().splitlines()]

    def text(self) -> str:
        return self.stream.getvalue()


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DATABASE_URL", UNREACHABLE_DSN)
    monkeypatch.setenv("LOG_FORMAT", "json")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    yield
    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()


class FakePool:
    """Stands in for ``psycopg_pool.ConnectionPool``: records every call.

    ``getconn`` hands out ``conns`` in order, or raises ``getconn_error``.
    """

    def __init__(
        self, conns: list[Any] | None = None, getconn_error: Exception | None = None
    ) -> None:
        self.conns = list(conns or [])
        self.getconn_error = getconn_error
        self.open_calls: list[dict[str, Any]] = []
        self.getconn_calls = 0
        self.returned: list[Any] = []
        self.closed = False

    def open(self, wait: bool = False, **kwargs: Any) -> None:
        self.open_calls.append({"wait": wait, **kwargs})

    def getconn(self, timeout: float | None = None) -> Any:
        self.getconn_calls += 1
        if self.getconn_error is not None:
            raise self.getconn_error
        return self.conns.pop(0)

    def putconn(self, conn: Any) -> None:
        self.returned.append(conn)

    def close(self) -> None:
        self.closed = True


def request_with(pool: Any) -> Any:
    """The part of a ``Request`` that ``deps.get_conn`` reads: ``app.state.pool``."""
    state = SimpleNamespace() if pool is None else SimpleNamespace(pool=pool)
    return SimpleNamespace(app=SimpleNamespace(state=state))


@pytest.fixture
def use_pool(monkeypatch: pytest.MonkeyPatch) -> Any:
    """``use_pool(FakePool(...))`` makes the app's lifespan use that pool."""

    def install(pool: FakePool) -> FakePool:
        monkeypatch.setattr(api_main, "create_pool", lambda **kwargs: pool)
        return pool

    return install


@pytest.fixture
def logs(monkeypatch: pytest.MonkeyPatch) -> LogSink:
    sink = LogSink()

    def capture(settings: Settings, *, stream: Any = None) -> None:
        sink.configure_calls += 1
        configure_logging(settings, stream=sink.stream)

    monkeypatch.setattr(api_main, "configure_logging", capture)
    return sink


@pytest.fixture
def app(logs: LogSink) -> FastAPI:
    return api_main.create_app()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
