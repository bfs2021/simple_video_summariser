"""``POST /ops/jobs/retry`` (issue #113).

Unit tests fake ``deps.get_conn`` and ``PostgresQueue.retry_dead_matching``;
the integration tests at the bottom run the route against seeded Postgres.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from common.config import get_settings
from common.queue import MAX_BULK_RETRY, BulkRetriedJob, BulkRetryResult, PostgresQueue
from common.repo.jobs import UNSET
from services.api import deps, routes_ops
from services.api.main import create_app
from tests.services.api.conftest import LogSink, head_dsn

__all__ = ["head_dsn"]

MARKER = "ZZMARKER42"
JSON = {"Content-Type": "application/json"}
URL = "/ops/jobs/retry"


class Spy:
    """Replaces ``PostgresQueue.retry_dead_matching``; records its keyword arguments."""

    def __init__(self, result: BulkRetryResult | Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result or BulkRetryResult(jobs=(), has_more=False)

    def __call__(self, **kwargs: Any) -> BulkRetryResult:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def install(monkeypatch: pytest.MonkeyPatch, fake: Spy) -> None:
    monkeypatch.setattr(PostgresQueue, "retry_dead_matching", lambda self, **kw: fake(**kw))


class Conns:
    def __init__(self) -> None:
        self.resolved = 0


@pytest.fixture
def conns(app: FastAPI) -> Conns:
    seen = Conns()

    def override() -> Iterator[object]:
        seen.resolved += 1
        yield object()

    app.dependency_overrides[deps.get_conn] = override
    return seen


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> Spy:
    fake = Spy()
    install(monkeypatch, fake)
    return fake


def post(app: FastAPI, body: Any = None, *, headers: dict[str, str] | None = None, raw: bytes | None = None) -> Any:
    with TestClient(app) as client:
        if raw is not None:
            return client.post(URL, content=raw, headers=headers or JSON)
        return client.post(URL, json=body, headers=headers or JSON)


# --- validation: limit ---------------------------------------------------------


@pytest.mark.parametrize("limit", [1, MAX_BULK_RETRY])
def test_the_limit_boundaries_are_accepted(app: FastAPI, conns: Conns, spy: Spy, limit: int) -> None:
    response = post(app, {"limit": limit})

    assert response.status_code == 200, response.text
    assert spy.calls[0]["limit"] == limit


@pytest.mark.parametrize("limit", [0, MAX_BULK_RETRY + 1, -1, "5", 1.5, True, None, [5], {"a": 1}])
def test_a_bad_limit_is_422_and_never_reaches_the_database(
    app: FastAPI, conns: Conns, spy: Spy, limit: Any
) -> None:
    response = post(app, {"limit": limit})

    assert response.status_code == 422
    assert conns.resolved == 0 and spy.calls == []


def test_the_limit_is_required_and_has_no_default(app: FastAPI, conns: Conns, spy: Spy) -> None:
    for body in ({}, {"kind": "ingest"}):
        assert post(app, body).status_code == 422
    assert conns.resolved == 0 and spy.calls == []


def test_the_cap_is_the_named_constant_and_is_fifty() -> None:
    limit = routes_ops.BulkRetryBody.model_json_schema()["properties"]["limit"]

    assert MAX_BULK_RETRY == 50
    assert (limit["minimum"], limit["maximum"]) == (1, MAX_BULK_RETRY)


# --- validation: the rest of the body -------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        {"state": "pending"}, {"state": "DEAD"}, {"state": ""}, {"state": 1},
        {"kind": "notify"}, {"kind": "INGEST"}, {"kind": ""}, {"kind": ["ingest"]}, {"kind": 5},
        {"error_class": "tool_failure"}, {"error_class": "NONE"}, {"error_class": ""},
        {"error_class": ["BUG"]}, {"error_class": 7},
        {"dry_run": "true"}, {"dry_run": 1}, {"dry_run": 0}, {"dry_run": None},
        {"error_clas": "TOOL_FAILURE"}, {"extra": 1}, {"LIMIT": 5},
    ],
)
def test_a_bad_field_or_unknown_key_is_422(app: FastAPI, conns: Conns, spy: Spy, extra: dict[str, Any]) -> None:
    response = post(app, {"limit": 5, **extra})

    assert response.status_code == 422, extra
    assert conns.resolved == 0 and spy.calls == []


@pytest.mark.parametrize("raw", [b"", b"{", b"[]", b"5", b'"x"', b"null", b"[1]", b"\xff\xfe"])
def test_a_body_that_is_not_a_json_object_is_422(app: FastAPI, conns: Conns, spy: Spy, raw: bytes) -> None:
    response = post(app, raw=raw)

    assert response.status_code == 422
    assert conns.resolved == 0 and spy.calls == []


@pytest.mark.parametrize(
    "body",
    [
        {"limit": MARKER}, {"limit": 5, "kind": MARKER}, {"limit": 5, "error_class": MARKER},
        {"limit": 5, "state": MARKER}, {"limit": 5, "dry_run": MARKER}, {"limit": 5, MARKER: 1},
    ],
)
def test_a_422_never_echoes_a_submitted_value(app: FastAPI, conns: Conns, spy: Spy, body: dict[str, Any]) -> None:
    response = post(app, body)

    assert response.status_code == 422
    assert MARKER not in response.text
    for item in response.json()["detail"]:
        assert set(item) == {"loc", "msg", "type"}


def test_a_json_decode_error_does_not_echo_the_bytes(app: FastAPI, conns: Conns, spy: Spy) -> None:
    response = post(app, raw=b'{"limit": ZZMARKER42')

    assert response.status_code == 422 and MARKER not in response.text


def test_explicit_nulls_are_the_same_as_omitting_the_keys(app: FastAPI, conns: Conns, spy: Spy) -> None:
    post(app, {"limit": 4})
    post(app, {"limit": 4, "state": None, "kind": None, "error_class": None})

    assert spy.calls[0] == spy.calls[1]
    assert spy.calls[0] == {"kind": None, "error_class": UNSET, "limit": 4, "dry_run": False}


def test_state_dead_is_accepted_so_the_listing_filter_can_be_sent_unchanged(
    app: FastAPI, conns: Conns, spy: Spy
) -> None:
    assert post(app, {"limit": 2, "state": "dead", "kind": "analyze", "error_class": "BUG"}).status_code == 200
    assert spy.calls == [{"kind": "analyze", "error_class": "BUG", "limit": 2, "dry_run": False}]


def test_error_class_none_means_the_reaped_jobs(app: FastAPI, conns: Conns, spy: Spy) -> None:
    post(app, {"limit": 2, "error_class": "none"})

    assert spy.calls[0]["error_class"] is None


def test_dry_run_is_passed_through(app: FastAPI, conns: Conns, spy: Spy) -> None:
    response = post(app, {"limit": 2, "dry_run": True})

    assert spy.calls[0]["dry_run"] is True and response.json()["dry_run"] is True


# --- content type, rate limit, routing ---------------------------------------------


@pytest.mark.parametrize("headers", [{"Content-Type": "text/plain"}, {"Content-Type": "application/x-www-form-urlencoded"}, {}])
def test_a_wrong_or_missing_content_type_is_415(app: FastAPI, conns: Conns, spy: Spy, headers: dict[str, str]) -> None:
    with TestClient(app) as client:
        response = client.post(URL, content=b'{"limit": 5}', headers=headers)

    assert response.status_code == 415
    assert conns.resolved == 0 and spy.calls == []


def test_the_rate_limit_covers_bulk_retry_but_not_the_listing(app: FastAPI, conns: Conns, spy: Spy) -> None:
    def too_many() -> None:
        raise HTTPException(429, "slow down")

    app.dependency_overrides[deps.rate_limit] = too_many
    with TestClient(app) as client:
        retry = client.post(URL, json={"limit": 5}, headers=JSON)
        listing = client.get("/ops/jobs")

    assert retry.status_code == 429
    assert listing.status_code != 429
    assert spy.calls == []


def test_the_bulk_route_never_reaches_the_single_job_route(app: FastAPI, conns: Conns, spy: Spy) -> None:
    response = post(app, {"limit": 3})

    assert response.status_code == 200  # not the single route's 422 for id="retry"
    assert len(spy.calls) == 1


def test_the_route_is_a_sync_function() -> None:
    import inspect

    assert not inspect.iscoroutinefunction(routes_ops.post_ops_jobs_retry)


def test_openapi_declares_the_json_request_body(app: FastAPI) -> None:
    operation = app.openapi()["paths"][URL]["post"]
    schema = operation["requestBody"]["content"]["application/json"]["schema"]

    assert operation["requestBody"]["required"] is True
    assert schema["$ref"].endswith("/BulkRetryBody")
    body = app.openapi()["components"]["schemas"]["BulkRetryBody"]
    assert body["required"] == ["limit"]
    assert body["additionalProperties"] is False


# --- responses -----------------------------------------------------------------------


def test_the_response_shape_and_the_log_lines(app: FastAPI, conns: Conns, logs: LogSink, monkeypatch: pytest.MonkeyPatch) -> None:
    result = BulkRetryResult(
        jobs=(
            BulkRetriedJob(9, "vb", "ingest", "TOOL_FAILURE", 3),
            BulkRetriedJob(4, "va", "ingest", "TOOL_FAILURE", 2),
        ),
        has_more=True,
    )
    install(monkeypatch, Spy(result))

    with TestClient(app) as client:
        response = client.post(
            URL, json={"limit": 2, "kind": "ingest"}, headers={**JSON, "X-Request-Id": "req-bulk"}
        )

    assert response.json() == {"retried": 2, "job_ids": [9, 4], "has_more": True, "dry_run": False}
    records = logs.records()
    per_job = [r for r in records if r.get("event") == "ops.job_retried"]
    assert [(r["job_id"], r["video_id"], r["kind"], r["prev_error_class"], r["prev_attempts"], r["bulk"], r["level"], r["request_id"]) for r in per_job] == [
        (9, "vb", "ingest", "TOOL_FAILURE", 3, True, "warning", "req-bulk"),
        (4, "va", "ingest", "TOOL_FAILURE", 2, True, "warning", "req-bulk"),
    ]
    [summary] = [r for r in records if r.get("event") == "ops.jobs_bulk_retried"]
    assert (summary["retried"], summary["kind"], summary["limit"], summary["has_more"], summary["level"], summary["request_id"]) == (
        2, "ingest", 2, True, "warning", "req-bulk",
    )


def test_nothing_matching_is_a_200_with_zero(app: FastAPI, conns: Conns, spy: Spy) -> None:
    response = post(app, {"limit": 5})

    assert response.status_code == 200
    assert response.json() == {"retried": 0, "job_ids": [], "has_more": False, "dry_run": False}


def test_a_dry_run_logs_one_info_line_and_no_retry_lines(app: FastAPI, conns: Conns, logs: LogSink, monkeypatch: pytest.MonkeyPatch) -> None:
    result = BulkRetryResult(jobs=(BulkRetriedJob(9, "vb", "ingest", "BUG", 3),), has_more=False)
    install(monkeypatch, Spy(result))

    post(app, {"limit": 2, "dry_run": True})

    events = [r for r in logs.records() if str(r.get("event", "")).startswith("ops.")]
    assert [(r["event"], r["level"]) for r in events] == [("ops.jobs_bulk_retry_dry_run", "info")]


def test_a_unique_violation_is_409_with_the_fixed_body(app: FastAPI, conns: Conns, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, Spy(psycopg.errors.UniqueViolation(f"dup {MARKER}")))

    response = post(app, {"limit": 5})

    assert response.status_code == 409
    assert response.json() == {"detail": "concurrent change, try again"}


# --- integration ---------------------------------------------------------------------


@pytest.fixture
def dsn(head_dsn: str, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("DATABASE_URL", head_dsn)
    get_settings.cache_clear()
    return head_dsn


@pytest.fixture
def db(dsn: str) -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(dsn, autocommit=True) as conn:
        yield conn


def seed(db: psycopg.Connection[Any], video: str, kind: str, state: str, error_class: str | None) -> int:
    row = db.execute(
        """
        INSERT INTO jobs (video_id, kind, dedupe_key, state, attempts, error_class, finished_at)
        VALUES (%s, %s, 'default', %s, 3, %s, CASE WHEN %s = 'dead' THEN now() END) RETURNING id
        """,
        (video, kind, state, error_class, state),
    ).fetchone()
    assert row is not None
    return int(row[0])


@pytest.mark.integration
def test_end_to_end_the_filter_limits_what_is_retried(dsn: str, db: psycopg.Connection[Any], logs: LogSink) -> None:
    hit1 = seed(db, "a", "ingest", "dead", "TOOL_FAILURE")
    hit2 = seed(db, "b", "ingest", "dead", "TOOL_FAILURE")
    others = [
        seed(db, "c", "ingest", "dead", "BUG"),
        seed(db, "d", "transcribe", "dead", "TOOL_FAILURE"),
        seed(db, "e", "ingest", "pending", None),
        seed(db, "f", "ingest", "done", "TOOL_FAILURE"),
    ]
    snapshot = db.execute("SELECT * FROM jobs WHERE id = ANY(%s) ORDER BY id", (others,)).fetchall()

    with TestClient(create_app()) as client:
        dry = client.post(URL, json={"limit": 50, "kind": "ingest", "error_class": "TOOL_FAILURE", "dry_run": True}, headers=JSON)
        assert db.execute("SELECT count(*) FROM jobs WHERE state = 'pending'").fetchone() == (1,)
        real = client.post(URL, json={"limit": 50, "kind": "ingest", "error_class": "TOOL_FAILURE"}, headers=JSON)

    assert dry.json() == {"retried": 2, "job_ids": [hit2, hit1], "has_more": False, "dry_run": True}
    assert real.json() == {"retried": 2, "job_ids": [hit2, hit1], "has_more": False, "dry_run": False}
    assert db.execute("SELECT * FROM jobs WHERE id = ANY(%s) ORDER BY id", (others,)).fetchall() == snapshot
    states = dict(db.execute("SELECT id, state FROM jobs WHERE id IN (%s, %s)", (hit1, hit2)).fetchall())
    assert states == {hit1: "pending", hit2: "pending"}
