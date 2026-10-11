"""Read routes, validation and error mapping (issue #42). No Postgres.

``get_conn`` is overridden with a dependency that fails the test if it is
ever resolved, which proves every 422 is decided before the database.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api import deps, routes_read
from tests.services.api.conftest import UNREACHABLE_DSN, FakePool, LogSink

MARKER = "ZZMARKER42"
VID = "dQw4w9WgXcQ"


def _no_db() -> Iterator[Any]:
    raise AssertionError("the database must not be touched")
    yield  # pragma: no cover


@pytest.fixture
def api(app: FastAPI) -> Iterator[TestClient]:
    app.dependency_overrides[deps.get_conn] = _no_db
    with TestClient(app) as client:
        yield client


def _assert_422_without_echo(response: Any, *markers: str) -> None:
    assert response.status_code == 422, response.text
    for marker in (MARKER, *markers):
        assert marker not in response.text
    detail = response.json()["detail"]
    if isinstance(detail, list):
        for item in detail:
            assert set(item) == {"loc", "msg", "type"}


# --- GET /videos ----------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"limit": "0"},
        {"limit": "201"},
        {"limit": "1.5"},
        {"limit": "abc"},
        {"limit": "1e3"},
        {"limit": str(2**63)},
        {"offset": "-1"},
        {"offset": "1000001"},
        {"offset": "1.5"},
        {"offset": "abc"},
        {"offset": "1e3"},
        {"offset": str(2**63 + 1)},
        {"channel": "UC" + "a" * 21},
        {"channel": "UC" + "a" * 23},
        {"channel": "UC" + "a" * 21 + "!"},
        {"channel": "@" + MARKER},
        {"channel": "UC" + MARKER + "a" * 13},
        {"channel": "UC" + "a" * 22 + "\n"},
        {"status": MARKER},
        {"status": "pending"},
        {"status": "DONE"},
        {"published_after": "2026-13-01"},
        {"published_after": "yesterday"},
        {"published_before": "2026-02-30"},
        {"published_after": MARKER},
        {"published_after": "2026-09-02", "published_before": "2026-09-01"},
    ],
)
def test_bad_library_query_params_are_422_before_the_database(
    api: TestClient, params: dict[str, str]
) -> None:
    _assert_422_without_echo(api.get("/videos", params=params))


@pytest.mark.parametrize(
    "path",
    [
        "a" * 10,
        "a" * 12,
        "dQw4w9WgXc!",
        "dQw4w9WgXc.",
        "%41Qw4w9WgXcQ",
        "dQw4w9WgX%20Q",
        "dQw4w9WgXc%0A",
        "dQw4w9WgXc%C3%A9",
        MARKER + "xy",
    ],
)
@pytest.mark.parametrize("suffix", ["", "/analyses", "/transcript"])
def test_a_malformed_video_id_is_422_before_the_database(
    api: TestClient, path: str, suffix: str
) -> None:
    response = api.get(f"/videos/{path}{suffix}")

    _assert_422_without_echo(response, path)
    assert response.json() == {
        "detail": [{"loc": ["path", "video_id"], "msg": routes_read.BAD_VIDEO_ID,
                    "type": "value_error"}]
    }


@pytest.mark.parametrize(
    ("suffix", "params"),
    [
        ("/analyses", {"limit": "0"}),
        ("/analyses", {"limit": "51"}),
        ("/analyses", {"offset": "-1"}),
        ("/analyses", {"offset": "1000001"}),
        ("/analyses", {"limit": MARKER}),
        ("/transcript", {"limit": "0"}),
        ("/transcript", {"limit": "1001"}),
        ("/transcript", {"offset": "-1"}),
        ("/transcript", {"offset": "1000001"}),
        ("/transcript", {"offset": MARKER}),
    ],
)
def test_bad_paging_params_are_422_before_the_database(
    api: TestClient, suffix: str, params: dict[str, str]
) -> None:
    _assert_422_without_echo(api.get(f"/videos/{VID}{suffix}", params=params))


def test_a_naive_datetime_is_read_as_utc() -> None:
    from datetime import UTC, datetime

    query = routes_read.VideoListQuery.model_validate(
        {"published_after": "2026-09-01T12:00:00", "published_before": "2026-09-02"}
    )

    assert query.published_after == datetime(2026, 9, 1, 12, tzinfo=UTC)
    assert query.published_after is not None and query.published_after.tzinfo is UTC
    assert query.published_before == datetime(2026, 9, 2, tzinfo=UTC)


def test_a_root_path_prefix_does_not_break_video_id_checks(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def stop(video_id: str) -> None:
        seen.append(video_id)
        raise AssertionError("reached")

    app.dependency_overrides[deps.get_conn] = _no_db
    with TestClient(app, root_path="/api", raise_server_exceptions=False) as client:
        good = client.get(f"/api/videos/{VID}")
        bad = client.get("/api/videos/%41Qw4w9WgXcQ")

    assert good.status_code == 500
    assert bad.status_code == 422


# --- errors -------------------------------------------------------------------


class _FailingConn:
    read_only = False

    def __init__(self, error: Exception) -> None:
        self.error = error

    def execute(self, *args: object, **kwargs: object) -> Any:
        raise self.error


@pytest.mark.parametrize(
    "path",
    ["/videos", f"/videos/{VID}", f"/videos/{VID}/analyses", f"/videos/{VID}/transcript"],
)
@pytest.mark.parametrize(
    "error",
    [
        psycopg.OperationalError(f"connection to {UNREACHABLE_DSN} lost"),
        psycopg.errors.QueryCanceled("canceling statement due to statement timeout"),
    ],
)
def test_an_operational_error_is_503_with_a_fixed_body_and_logged(
    app: FastAPI, logs: LogSink, path: str, error: Exception
) -> None:
    def failing() -> Iterator[Any]:
        yield _FailingConn(error)

    app.dependency_overrides[deps.get_conn] = failing
    with TestClient(app) as client:
        response = client.get(path, headers={"X-Request-Id": "req-42"})

    assert response.status_code == 503
    assert response.json() == {"detail": "database unavailable"}
    for secret in ("SECRETPW", "127.0.0.1", "postgresql://", "statement timeout"):
        assert secret not in response.text
    [line] = [r for r in logs.records() if r.get("event") == "database unavailable"]
    assert line["request_id"] == "req-42"
    assert "OperationalError" in line["exception"] or "QueryCanceled" in line["exception"]


@pytest.mark.parametrize("path", ["/videos", f"/videos/{VID}"])
def test_a_failing_connect_is_503_with_the_read_body(
    app: FastAPI, logs: LogSink, use_pool: Any, path: str
) -> None:
    use_pool(FakePool(getconn_error=psycopg.OperationalError(f"could not connect to {UNREACHABLE_DSN}")))
    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 503
    assert response.json() == {"detail": "database unavailable"}


def test_any_other_exception_keeps_the_generic_500(app: FastAPI, logs: LogSink) -> None:
    def failing() -> Iterator[Any]:
        yield _FailingConn(RuntimeError(f"boom {MARKER}"))

    app.dependency_overrides[deps.get_conn] = failing
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/videos")

    assert response.status_code == 500
    assert response.json() == {"detail": "internal error"}


def test_healthz_keeps_its_own_unavailable_body(
    app: FastAPI, logs: LogSink, use_pool: Any
) -> None:
    use_pool(FakePool(getconn_error=psycopg.OperationalError("no")))
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "unavailable"}


# --- wiring -------------------------------------------------------------------


def test_openapi_lists_the_four_paths_with_params_and_response_schemas(app: FastAPI) -> None:
    paths = app.openapi()["paths"]
    expected = {
        "/videos": {"channel", "status", "published_after", "published_before", "offset",
                    "limit"},
        "/videos/{video_id}": {"video_id"},
        "/videos/{video_id}/analyses": {"video_id", "offset", "limit"},
        "/videos/{video_id}/transcript": {"video_id", "offset", "limit"},
    }
    for path, params in expected.items():
        operation = paths[path]["get"]
        assert {p["name"] for p in operation["parameters"]} == params, path
        schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
        assert "$ref" in schema, path


def test_read_routes_are_sync_and_declare_response_models() -> None:
    import inspect

    routes = [r for r in routes_read.router.routes if getattr(r, "path", "").startswith("/videos")]
    assert len(routes) == 5
    for route in routes:
        assert not inspect.iscoroutinefunction(route.endpoint)  # type: ignore[attr-defined]
        if not route.path.endswith("/render"):  # type: ignore[attr-defined]  # HTML page, no JSON model (#45)
            assert route.response_model is not None  # type: ignore[attr-defined]
