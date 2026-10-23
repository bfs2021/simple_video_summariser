"""App layout, startup and auth/rate-limit seams (issue #39)."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import services.api.main as api_main
from common import db as common_db
from services.api import deps, routes_ops, routes_read, routes_write
from tests.services.api.conftest import LogSink

REPO_ROOT = Path(__file__).resolve().parents[3]


def _deny() -> None:
    raise HTTPException(status_code=401, detail="denied")


def _too_many() -> None:
    raise HTTPException(status_code=429, detail="slow down")


@pytest.fixture
def write_probe_route() -> Iterator[None]:
    """A test-only route on the write router, removed again afterwards."""
    before = list(routes_write.router.routes)

    @routes_write.router.post("/__test_write")
    def _probe() -> dict[str, str]:
        return {"ok": "yes"}

    yield
    routes_write.router.routes[:] = before


def test_module_exposes_create_app_and_an_app_instance() -> None:
    assert isinstance(api_main.app, FastAPI)
    assert isinstance(api_main.create_app(), FastAPI)
    assert api_main.create_app() is not api_main.create_app()


def _module_routes() -> list[APIRoute]:
    return [
        r
        for module in (routes_read, routes_write, routes_ops)
        for r in module.router.routes
        if isinstance(r, APIRoute)
    ]


def _app_operations(app: FastAPI) -> list[tuple[str, str]]:
    return [
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
    ]


def test_three_router_modules_are_all_included_and_healthz_is_an_ops_route(
    write_probe_route: None, logs: LogSink
) -> None:
    # /metrics is plain text for Prometheus and kept out of the OpenAPI schema (#60).
    ops_paths = {
        r.path
        for r in routes_ops.router.routes
        if isinstance(r, APIRoute) and r.include_in_schema
    }
    assert "/healthz" in ops_paths

    app = api_main.create_app()
    app_paths = {path for _, path in _app_operations(app)}
    assert ops_paths <= app_paths
    assert "/__test_write" in app_paths


def test_metrics_is_the_only_schema_excluded_route() -> None:
    # A second include_in_schema=False route would silently escape the
    # all-routes 401 sweep below, which walks app.openapi()["paths"] (#60).
    excluded = {
        (route.path, method)
        for module in (routes_read, routes_write, routes_ops)
        for route in module.router.routes
        if isinstance(route, APIRoute) and not route.include_in_schema
        for method in route.methods or ()
    }
    assert excluded == {("/metrics", "GET")}


def test_starting_the_app_configures_logging_exactly_once(
    app: FastAPI, logs: LogSink
) -> None:
    assert logs.configure_calls == 0
    with TestClient(app):
        assert logs.configure_calls == 1
    assert logs.configure_calls == 1


def test_starting_the_app_makes_no_synchronous_connect_call(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pool (psycopg.Connection.connect, #95) opens with open(wait=False): its
    # background workers may try to connect, but startup itself must not. Every
    # attempt fails, so a startup that waits for the pool raises instead of hanging.
    threads: list[int] = []

    def refuse(*args: object, **kwargs: object) -> psycopg.Connection[object]:
        threads.append(threading.get_ident())
        raise psycopg.OperationalError("no database in this test")

    monkeypatch.setattr(common_db, "connect", refuse)
    monkeypatch.setattr(psycopg, "connect", refuse)
    monkeypatch.setattr(psycopg.Connection, "connect", classmethod(lambda cls, *a, **kw: refuse()))

    with TestClient(app):
        pass

    assert threading.get_ident() not in threads


def test_importing_the_app_needs_no_settings_and_loads_no_processing_modules() -> None:
    code = (
        "import json, sys\n"
        "import services.api.main\n"
        "bad = sorted({m.split('.')[0] if not m.startswith('adapters') else m"
        " for m in sys.modules"
        " if m == 'adapters' or m.startswith('adapters.')"
        " or m.split('.')[0] in ('yt_dlp', 'faster_whisper', 'alembic')})\n"
        "print(json.dumps(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def test_database_route_handlers_are_sync_functions() -> None:
    db_routes = [
        r
        for r in _module_routes()
        if any(d.call is deps.get_conn for d in r.dependant.dependencies)
    ]
    assert db_routes, "expected at least /healthz to use deps.get_conn"
    for route in db_routes:
        assert not inspect.iscoroutinefunction(route.endpoint), route.path


def test_require_auth_and_rate_limit_let_requests_through(
    write_probe_route: None, logs: LogSink
) -> None:
    with TestClient(api_main.create_app()) as client:
        assert client.post("/__test_write").status_code == 200


def test_require_auth_is_attached_once_at_app_level(app: FastAPI) -> None:
    assert any(d.dependency is deps.require_auth for d in app.router.dependencies)
    for module in (routes_read, routes_write, routes_ops):
        assert not any(
            d.dependency is deps.require_auth for d in module.router.dependencies
        )


def test_overriding_require_auth_denies_every_route_except_openapi(
    write_probe_route: None, logs: LogSink
) -> None:
    # FastAPI 0.142 includes routers lazily, so app.routes holds include
    # wrappers rather than APIRoutes; the OpenAPI paths are the public list.
    app = api_main.create_app()
    app.dependency_overrides[deps.require_auth] = _deny
    operations = _app_operations(app)
    assert ("GET", "/healthz") in operations
    assert ("POST", "/__test_write") in operations

    with TestClient(app) as client:
        for method, path in operations:
            response = client.request(method, path)
            assert response.status_code == 401, (method, path)
        assert client.get("/openapi.json").status_code == 200


def test_docs_and_redoc_are_disabled(client: TestClient) -> None:
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404


def test_rate_limit_applies_to_the_write_router_only(
    write_probe_route: None, logs: LogSink
) -> None:
    app = api_main.create_app()
    app.dependency_overrides[deps.rate_limit] = _too_many

    def healthy_conn() -> Iterator[object]:
        yield _EmptyConn()

    app.dependency_overrides[deps.get_conn] = healthy_conn

    with TestClient(app) as client:
        assert client.post("/__test_write").status_code == 429
        assert client.get("/healthz").status_code == 200


def test_delete_video_inherits_auth_and_write_rate_limit(logs: LogSink) -> None:
    app = api_main.create_app()
    app.dependency_overrides[deps.require_auth] = _deny
    with TestClient(app) as client:
        assert client.delete("/videos/dQw4w9WgXcQ").status_code == 401

    app = api_main.create_app()
    app.dependency_overrides[deps.rate_limit] = _too_many
    with TestClient(app) as client:
        assert client.delete("/videos/dQw4w9WgXcQ").status_code == 429


def test_write_router_declares_rate_limit() -> None:
    assert any(
        d.dependency is deps.rate_limit for d in routes_write.router.dependencies
    )
    assert not any(
        d.dependency is deps.rate_limit for d in routes_read.router.dependencies
    )
    assert not any(
        d.dependency is deps.rate_limit for d in routes_ops.router.dependencies
    )


class _EmptyConn:
    read_only = False

    def execute(self, *args: object, **kwargs: object) -> _EmptyConn:
        return self

    def fetchall(self) -> list[tuple[str, str, int]]:
        return []
