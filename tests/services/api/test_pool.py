"""The API's connection pool, without Postgres (issue #95).

A ``FakePool`` replaces ``psycopg_pool.ConnectionPool`` where the test is
about the app's use of the pool; one test points a real, unopened-until-start
pool at a closed port to check what it logs and answers.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from psycopg_pool import PoolClosed, PoolTimeout, TooManyRequests

import services.api.main as api_main
from common import db as common_db
from common.config import get_settings
from services.api import deps
from tests.services.api.conftest import UNREACHABLE_DSN, FakePool, LogSink, request_with

REPO_ROOT = Path(__file__).resolve().parents[3]
SECRET = "SECRETPW"
BODY_503_METRICS = "database unavailable\n"


# --- lifecycle ---------------------------------------------------------------


def test_the_pool_does_not_exist_before_the_app_starts(app: FastAPI) -> None:
    assert not hasattr(app.state, "pool")


def test_startup_opens_the_pool_without_waiting_and_stores_it_on_app_state(
    app: FastAPI, use_pool: Any
) -> None:
    pool = use_pool(FakePool())

    with TestClient(app):
        assert app.state.pool is pool
        assert pool.open_calls == [{"wait": False}]
        assert pool.closed is False


def test_shutdown_closes_the_pool(app: FastAPI, use_pool: Any) -> None:
    pool = use_pool(FakePool())

    with TestClient(app):
        pass

    assert pool.closed is True


def test_the_pool_is_closed_even_when_the_app_body_fails(app: FastAPI, use_pool: Any) -> None:
    pool = use_pool(FakePool())

    with pytest.raises(RuntimeError), TestClient(app):
        raise RuntimeError("test body failed")

    assert pool.closed is True


def test_the_real_pool_is_closed_after_the_client_exits_and_starts_without_postgres(
    app: FastAPI,
) -> None:
    # The autouse env points DATABASE_URL at a closed port: the app must start.
    with TestClient(app):
        pool = app.state.pool
        assert pool.closed is False

    assert pool.closed is True


def test_importing_the_app_module_creates_no_pool_and_no_connection() -> None:
    code = (
        "import psycopg, psycopg_pool\n"
        "def boom(*a, **k):\n"
        "    raise SystemExit('connected at import')\n"
        "psycopg.Connection.connect = classmethod(boom)\n"
        "import services.api.main as m\n"
        "print(hasattr(m.app.state, 'pool'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


# --- create_pool ---------------------------------------------------------------


def test_create_pool_is_unopened_and_reads_sizes_from_settings_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DB_POOL_MIN_SIZE", "2")
    monkeypatch.setenv("DB_POOL_MAX_SIZE", "3")
    monkeypatch.setenv("DB_POOL_TIMEOUT_SEC", "1.5")
    get_settings.cache_clear()

    pool = common_db.create_pool(connect_timeout=2, statement_timeout_ms=1500)

    assert pool.closed
    assert (pool.min_size, pool.max_size, pool.timeout) == (2, 3, 1.5)
    kwargs: dict[str, Any] = cast(dict[str, Any], pool.kwargs)
    assert kwargs["connect_timeout"] == 2
    assert kwargs["options"] == "-c statement_timeout=1500"


def test_the_api_applies_statement_and_connect_timeouts_within_the_two_second_bound(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    real = common_db.create_pool

    def spy(**kwargs: Any) -> Any:
        pool = real(**kwargs)
        seen.update(cast(dict[str, Any], pool.kwargs))
        return pool

    monkeypatch.setattr(api_main, "create_pool", spy)
    with TestClient(app):
        pass

    assert 0 < int(seen["connect_timeout"]) <= 2
    assert "statement_timeout=" in seen["options"]
    assert 0 < int(seen["options"].split("statement_timeout=")[1].split()[0]) <= 2000


def test_the_pool_checks_connections_when_handing_them_out() -> None:
    pool = common_db.create_pool(connect_timeout=2, statement_timeout_ms=2000)

    assert pool._check is not None
    assert pool._reset is common_db._reset_connection


def test_reset_clears_the_per_request_session_state() -> None:
    class Conn:
        read_only: bool | None = True
        isolation_level: Any = "SERIALIZABLE"
        deferrable: bool | None = True
        autocommit = True

    conn = Conn()
    common_db._reset_connection(conn)  # type: ignore[arg-type]

    assert (conn.read_only, conn.isolation_level, conn.deferrable, conn.autocommit) == (
        None,
        None,
        None,
        False,
    )


# --- get_conn ------------------------------------------------------------------


def test_get_conn_hands_out_the_pooled_connection_and_returns_it() -> None:
    conn = object()
    pool = FakePool([conn])
    gen = deps.get_conn(request_with(pool))

    assert next(gen) is conn
    assert pool.returned == []
    gen.close()

    assert pool.returned == [conn]


def test_get_conn_returns_the_connection_when_the_handler_raises() -> None:
    conn = object()
    pool = FakePool([conn])
    gen = deps.get_conn(request_with(pool))
    next(gen)

    with pytest.raises(ValueError, match="handler"):
        gen.throw(ValueError("handler"))

    assert pool.returned == [conn]


def test_a_failed_checkout_returns_nothing_to_the_pool() -> None:
    pool = FakePool(getconn_error=PoolTimeout("couldn't get a connection after 2.00 sec"))

    with pytest.raises(deps.DatabaseUnavailable):
        next(deps.get_conn(request_with(pool)))

    assert pool.returned == []


@pytest.mark.parametrize("error", [PoolTimeout, PoolClosed, TooManyRequests, psycopg.OperationalError])
def test_every_pool_checkout_failure_is_database_unavailable(error: type[Exception]) -> None:
    with pytest.raises(deps.DatabaseUnavailable):
        next(deps.get_conn(request_with(FakePool(getconn_error=error("x")))))


def test_a_non_database_checkout_error_is_not_swallowed() -> None:
    with pytest.raises(RuntimeError):
        next(deps.get_conn(request_with(FakePool(getconn_error=RuntimeError("bug")))))


def test_get_conn_is_the_single_dependency_of_the_database_routes(app: FastAPI) -> None:
    for route in app.routes:
        if isinstance(route, APIRoute) and any(
            d.call is deps.get_conn for d in route.dependant.dependencies
        ):
            assert not inspect.iscoroutinefunction(route.endpoint)


# --- exhausted pool / database down: 503 ---------------------------------------


def _timeout_pool() -> FakePool:
    return FakePool(getconn_error=PoolTimeout("couldn't get a connection after 2.00 sec"))


def test_healthz_keeps_its_own_503_body_on_a_pool_timeout(
    app: FastAPI, use_pool: Any, logs: LogSink
) -> None:
    use_pool(_timeout_pool())

    with TestClient(app) as client:
        response = client.get("/healthz", headers={"X-Request-Id": "req-pool-1"})

    assert response.status_code == 503
    assert response.json() == deps.UNAVAILABLE_BODY
    timeouts = [r for r in logs.records() if r.get("event") == "db.pool_unavailable"]
    assert timeouts and timeouts[0]["request_id"] == "req-pool-1"
    assert timeouts[0]["error"] == "PoolTimeout"


def test_metrics_keeps_its_fixed_503_body_on_a_pool_timeout(app: FastAPI, use_pool: Any) -> None:
    use_pool(_timeout_pool())

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 503
    assert response.text == BODY_503_METRICS


@pytest.mark.parametrize("path", ["/videos", "/ops/jobs"])
def test_other_routes_answer_a_generic_503_on_a_pool_timeout(
    app: FastAPI, use_pool: Any, path: str
) -> None:
    use_pool(_timeout_pool())

    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 503
    assert response.json() in (deps.READ_UNAVAILABLE_BODY, deps.UNAVAILABLE_BODY)
    for leak in ("PoolTimeout", "couldn't", "postgresql://", SECRET, "127.0.0.1"):
        assert leak not in response.text


def test_a_pool_that_recovers_serves_the_next_request(app: FastAPI, use_pool: Any) -> None:
    class Conn:
        read_only = False

        def execute(self, *a: Any, **k: Any) -> Conn:
            return self

        def fetchall(self) -> list[Any]:
            return []

    pool = use_pool(FakePool([Conn()], getconn_error=PoolTimeout("t")))

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 503
        pool.getconn_error = None
        assert client.get("/healthz").status_code == 200
    assert len(pool.returned) == 1


def test_an_unreachable_database_times_out_to_503_and_leaks_neither_dsn_nor_password(
    app: FastAPI, logs: LogSink, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Real pool, closed port (autouse env), a short wait: no sleep in the test.
    monkeypatch.setenv("DB_POOL_TIMEOUT_SEC", "0.3")
    get_settings.cache_clear()
    caplog.set_level("DEBUG")

    with TestClient(app) as client:
        response = client.get("/healthz")
        metrics = client.get("/metrics")

    assert response.status_code == 503
    assert response.json() == deps.UNAVAILABLE_BODY
    assert metrics.text == BODY_503_METRICS
    everything = logs.text() + caplog.text + response.text + metrics.text
    assert SECRET not in everything
    assert UNREACHABLE_DSN not in everything
    assert "postgresql://" not in everything
    assert any(r.get("event") == "db.pool_unavailable" for r in logs.records())
