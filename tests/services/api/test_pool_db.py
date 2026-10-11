"""The API's connection pool against real Postgres (issue #95)."""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator
from contextlib import closing
from typing import Annotated, Any

import psycopg
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from common.config import get_settings
from services.api import deps
from services.api.main import create_app
from tests.services.api.conftest import LogSink, head_dsn

pytestmark = pytest.mark.integration

__all__ = ["head_dsn"]


def _env(monkeypatch: pytest.MonkeyPatch, dsn: str, **extra: str) -> None:
    monkeypatch.setenv("DATABASE_URL", dsn)
    for key, value in extra.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


def _app_with_probes() -> FastAPI:
    app = create_app()

    @app.get("/__pid")
    def pid(conn: Annotated[psycopg.Connection[Any], Depends(deps.get_conn)]) -> dict[str, int]:
        row = conn.execute("SELECT pg_backend_pid()").fetchone()
        assert row is not None
        return {"pid": row[0]}

    @app.get("/__show")
    def show(conn: Annotated[psycopg.Connection[Any], Depends(deps.get_conn)]) -> dict[str, str]:
        timeout = conn.execute("SHOW statement_timeout").fetchone()
        assert timeout is not None
        return {"statement_timeout": timeout[0]}

    @app.post("/__abort")
    def abort(conn: Annotated[psycopg.Connection[Any], Depends(deps.get_conn)]) -> None:
        conn.execute("CREATE TEMP TABLE pool_probe (x int)")
        conn.execute("SELECT 1 / 0")  # leaves the transaction aborted

    @app.post("/__write")
    def write(conn: Annotated[psycopg.Connection[Any], Depends(deps.get_conn)]) -> dict[str, bool]:
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS pool_write (x int)")
        conn.execute("INSERT INTO pool_write VALUES (1)")
        return {"ok": True}

    @app.get("/__uncommitted")
    def uncommitted(
        conn: Annotated[psycopg.Connection[Any], Depends(deps.get_conn)],
    ) -> dict[str, int]:
        row = conn.execute(
            "SELECT count(*) FROM pg_class WHERE relname = 'pool_write' AND relpersistence = 't'"
        ).fetchone()
        assert row is not None
        return {"temp_tables": row[0]}

    @app.get("/__readonly")
    def readonly(
        conn: Annotated[psycopg.Connection[Any], Depends(deps.get_read_conn)],
    ) -> dict[str, int]:
        row = conn.execute("SELECT pg_backend_pid()").fetchone()
        assert row is not None
        return {"pid": row[0]}

    return app


@pytest.fixture
def client_one_conn(
    head_dsn: str, monkeypatch: pytest.MonkeyPatch, logs: LogSink
) -> Iterator[TestClient]:
    _env(monkeypatch, head_dsn, DB_POOL_MIN_SIZE="1", DB_POOL_MAX_SIZE="1", DB_POOL_TIMEOUT_SEC="0.5")
    with TestClient(_app_with_probes(), raise_server_exceptions=False) as client:
        client.app.state.pool.wait(timeout=10)  # type: ignore[attr-defined]
        yield client


def test_two_sequential_requests_reuse_the_same_backend(client_one_conn: TestClient) -> None:
    first = client_one_conn.get("/__pid").json()["pid"]
    second = client_one_conn.get("/__pid").json()["pid"]

    assert first == second


def test_statement_timeout_is_set_on_the_pooled_connection(client_one_conn: TestClient) -> None:
    assert client_one_conn.get("/__show").json() == {"statement_timeout": "2s"}
    # still set on the reused connection, with no per-request SET
    assert client_one_conn.get("/__show").json() == {"statement_timeout": "2s"}


def test_a_handler_that_fails_mid_transaction_does_not_poison_the_next_request(
    client_one_conn: TestClient,
) -> None:
    before = client_one_conn.get("/__pid").json()["pid"]

    assert client_one_conn.post("/__abort").status_code == 500
    after = client_one_conn.get("/__pid")

    assert after.status_code == 200
    assert after.json()["pid"] == before  # the very same backend, now clean


def test_an_uncommitted_write_is_rolled_back_before_reuse(client_one_conn: TestClient) -> None:
    assert client_one_conn.post("/__write").status_code == 200

    # same backend, new transaction: the temp table created above is gone
    assert client_one_conn.get("/__uncommitted").json() == {"temp_tables": 0}


def test_read_only_session_state_does_not_leak_to_the_next_request(
    client_one_conn: TestClient,
) -> None:
    assert client_one_conn.get("/__readonly").status_code == 200

    assert client_one_conn.post("/__write").status_code == 200


def test_a_backend_terminated_while_idle_is_replaced_and_the_next_request_succeeds(
    client_one_conn: TestClient, head_dsn: str
) -> None:
    pid = client_one_conn.get("/__pid").json()["pid"]
    with psycopg.connect(head_dsn, autocommit=True) as admin:
        admin.execute("SELECT pg_terminate_backend(%s)", (pid,))

    response = client_one_conn.get("/__pid")

    assert response.status_code == 200
    assert response.json()["pid"] != pid


def test_an_exhausted_pool_answers_503_without_waiting_for_ever(
    client_one_conn: TestClient, logs: LogSink
) -> None:
    pool = client_one_conn.app.state.pool  # type: ignore[attr-defined]
    held = pool.getconn()
    try:
        health = client_one_conn.get("/healthz", headers={"X-Request-Id": "req-held"})
        other = client_one_conn.get("/ops/jobs")
    finally:
        pool.putconn(held)

    assert health.status_code == 503
    assert health.json() == {"status": "unavailable"}
    assert other.status_code == 503
    assert "psycopg" not in other.text and "postgresql" not in other.text
    assert any(
        r.get("event") == "db.pool_unavailable" and r.get("request_id") == "req-held"
        for r in logs.records()
    )
    # the held connection was given back: the pool serves again
    assert client_one_conn.get("/healthz").status_code == 200


def test_the_app_starts_with_postgres_down_and_recovers_when_it_comes_back(
    head_dsn: str, monkeypatch: pytest.MonkeyPatch, logs: LogSink
) -> None:
    real = psycopg.conninfo.conninfo_to_dict(head_dsn)
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    dsn = psycopg.conninfo.make_conninfo(head_dsn, host="127.0.0.1", port=port)
    _env(monkeypatch, dsn, DB_POOL_MIN_SIZE="1", DB_POOL_MAX_SIZE="2", DB_POOL_TIMEOUT_SEC="0.3")

    with TestClient(create_app()) as client:
        assert client.get("/healthz").status_code == 503  # started anyway

        proxy = _Proxy(port, (str(real["host"]), int(str(real["port"]))))
        try:
            client.app.state.pool.wait(timeout=30)  # type: ignore[attr-defined]
            assert client.get("/healthz").status_code == 200
        finally:
            proxy.stop()


class _Proxy:
    """A one-purpose TCP forwarder, so 'Postgres comes back' needs no restart."""

    def __init__(self, port: int, target: tuple[str, int]) -> None:
        self.target = target
        self.server = socket.create_server(("127.0.0.1", port))
        self.server.settimeout(0.2)
        self.done = threading.Event()
        self.threads = [threading.Thread(target=self._accept, daemon=True)]
        self.threads[0].start()

    def _accept(self) -> None:
        while not self.done.is_set():
            try:
                client, _ = self.server.accept()
            except TimeoutError:
                continue
            upstream = socket.create_connection(self.target)
            for src, dst in ((client, upstream), (upstream, client)):
                thread = threading.Thread(target=self._pipe, args=(src, dst), daemon=True)
                thread.start()
                self.threads.append(thread)

    def _pipe(self, src: socket.socket, dst: socket.socket) -> None:
        try:
            while data := src.recv(65536):
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for sock in (src, dst):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def stop(self) -> None:
        self.done.set()
        self.server.close()
