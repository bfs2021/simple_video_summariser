"""``GET /metrics`` without a database (issue #60).

The aggregates themselves are covered against real Postgres in
``tests/common/repo/test_metrics_aggregates.py``. Here a fake connection answers
each statement, which pins the route's contract: headers, body shape, one
statement per family inside one read-only ``REPEATABLE READ`` transaction, and a
fixed, leak-free 503 for every database failure.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from common.config import get_settings
from common.metrics import DURATION_BUCKETS
from services.api import deps
from tests.services.api.conftest import UNREACHABLE_DSN, LogSink

SECRET = "SECRETPW"
BODY_503 = "database unavailable\n"


class FakeConn:
    """Answers each aggregate by a fragment of its SQL; records what it was asked."""

    def __init__(self, error: Exception | None = None, fail_on: int | None = None) -> None:
        self.error = error
        self.fail_on = fail_on
        self.read_only: bool | None = None
        self.isolation_level: Any = None
        self.rollbacks = 0
        self.queries: list[str] = []
        self.params: list[Any] = []
        self._current = ""

    def execute(self, query: str, params: Any = None) -> FakeConn:
        self.queries.append(query)
        self.params.append(params)
        if self.error is not None and (self.fail_on is None or self.fail_on == len(self.queries)):
            raise self.error
        self._current = query
        return self

    def rollback(self) -> None:
        self.rollbacks += 1

    def fetchall(self) -> list[tuple[Any, ...]]:
        q = self._current
        if "GROUP BY kind, state" in q:
            return [("ingest", "pending", 2), ("notify", "pending", 3)]
        if "extract(epoch" in q:
            return [("analyze", *(1,) * len(DURATION_BUCKETS), 1, 12.5)]
        if "GROUP BY model" in q:
            return [('evil"model\nx', 10, 20)]
        if "FROM planner_task_runs" in q:
            return []
        raise AssertionError(f"unexpected fetchall for {q!r}")

    def fetchone(self) -> tuple[Any, ...] | None:
        q = self._current
        if "feed_gap_reason = 'suspected'" in q:
            return (2, 1, 0)
        if "jsonb_typeof" in q:
            return (0.25,)
        if "SUM(cost_usd)" in q:
            return (None,)
        if "speakers_coerced" in q:
            return (4,)
        if "FROM media" in q:
            return (1234,)
        if "FROM videos v" in q:
            return (6,)
        raise AssertionError(f"unexpected fetchone for {q!r}")


def _use(app: FastAPI, conn: FakeConn) -> None:
    def override() -> Iterator[FakeConn]:
        yield conn

    app.dependency_overrides[deps.get_conn] = override


def _get(app: FastAPI) -> Any:
    with TestClient(app) as client:
        return client.get("/metrics")


def test_metrics_answers_200_with_the_text_format_content_type(app: FastAPI) -> None:
    _use(app, FakeConn())

    response = _get(app)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    body = response.text
    assert 'queue_depth{kind="ingest",state="pending"} 2' in body
    assert 'queue_depth{kind="other",state="pending"} 3' in body
    assert 'feed_gap_channels{reason="suspected"} 2' in body
    assert 'feed_gap_channels{reason="no_anchor"} 1' in body
    assert 'feed_gap_channels{reason="listing_window_exhausted"} 0' in body
    assert "channel_id" not in body
    assert 'job_duration_seconds_count{kind="analyze"} 1' in body
    assert "whisper_rtf 0.25\n" in body
    assert 'llm_tokens_total{model="evil\\"model\\nx",direction="output"} 20' in body
    assert "llm_cost_usd_total 0\n" in body
    assert "speaker_coercions_total 4\n" in body
    assert "audio_bytes_used 1234\n" in body
    assert "reanalysis_backlog_videos 6\n" in body


_FAMILIES = (
    "queue_depth",
    "feed_gap_channels",
    "job_duration_seconds",
    "whisper_rtf",
    "llm_tokens_total",
    "llm_cost_usd_total",
    "speaker_coercions_total",
    "audio_bytes_used",
    "reanalysis_backlog_videos",
    "reaped_jobs_total",
    "planner_task_last_run_timestamp_seconds",
    "planner_task_last_success",
)


def _deny() -> None:
    raise HTTPException(status_code=401, detail="denied")


def test_metrics_returns_401_when_require_auth_denies(app: FastAPI) -> None:
    # /metrics is include_in_schema=False, so the all-routes 401 test in
    # test_app.py (which walks the OpenAPI schema) skips it; this is its coverage.
    conn = FakeConn()
    _use(app, conn)
    app.dependency_overrides[deps.require_auth] = _deny

    response = _get(app)

    assert response.status_code == 401
    assert conn.queries == []
    # An auth failure must not leak any part of a scrape.
    assert not response.headers["content-type"].startswith("text/plain")
    assert "version=0.0.4" not in response.headers["content-type"]
    assert "# HELP" not in response.text
    assert "# TYPE" not in response.text
    for family in _FAMILIES:
        assert family not in response.text


def test_two_scrapes_of_an_unchanged_database_are_byte_identical(app: FastAPI) -> None:
    _use(app, FakeConn())
    with TestClient(app) as client:
        first = client.get("/metrics").content
        second = client.get("/metrics").content

    assert first == second


def test_one_statement_per_family_in_one_read_only_repeatable_read_transaction(
    app: FastAPI,
) -> None:
    conn = FakeConn()
    _use(app, conn)

    _get(app)

    assert len(conn.queries) == 10
    assert conn.read_only is True
    assert conn.isolation_level == psycopg.IsolationLevel.REPEATABLE_READ
    assert conn.rollbacks == 1


def test_the_backlog_uses_the_configured_prompt_version(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROMPT_VERSION", "v-test-42")
    get_settings.cache_clear()
    conn = FakeConn()
    _use(app, conn)

    _get(app)

    (backlog_params,) = [p for q, p in zip(conn.queries, conn.params, strict=True) if "videos v" in q]
    assert backlog_params == ("v-test-42",)


@pytest.mark.parametrize(
    "error",
    [
        psycopg.OperationalError(f"connection to server at 10.1.2.3 failed: {SECRET}"),
        psycopg.errors.QueryCanceled(f"canceling statement due to statement timeout {SECRET}"),
        psycopg.errors.UndefinedTable(f'relation "jobs" does not exist {SECRET}'),
        RuntimeError(f"postgresql://user:{SECRET}@db/x"),
    ],
    ids=["down", "timeout", "erroring", "unexpected"],
)
@pytest.mark.parametrize("fail_on", [1, 2, 6, 9])
def test_a_database_failure_gives_a_fixed_503_and_no_partial_metrics(
    app: FastAPI, logs: LogSink, error: Exception, fail_on: int
) -> None:
    _use(app, FakeConn(error=error, fail_on=fail_on))

    response = _get(app)

    assert response.status_code == 503
    assert response.text == BODY_503
    assert response.headers["content-type"].startswith("text/plain")
    assert "queue_depth" not in response.text
    assert SECRET not in response.text
    # the error and its traceback are logged server-side
    assert any(
        rec.get("event") == "metrics database query failed" and "Traceback" in rec.get("exception", "")
        for rec in logs.records()
    )


def test_a_connection_that_cannot_be_opened_gives_the_same_503(app: FastAPI) -> None:
    def override() -> Iterator[Any]:
        raise deps.DatabaseUnavailable
        yield  # pragma: no cover

    app.dependency_overrides[deps.get_conn] = override

    response = _get(app)

    assert response.status_code == 503
    assert response.text == BODY_503


def test_an_unreachable_database_answers_503_without_leaking_the_dsn(app: FastAPI) -> None:
    # No override: the real get_conn dials the closed port from the autouse env.
    response = _get(app)

    assert response.status_code == 503
    assert response.text == BODY_503
    assert SECRET not in response.text
    assert UNREACHABLE_DSN not in response.text
    assert "127.0.0.1" not in response.text


def test_healthz_keeps_its_own_503_body(app: FastAPI) -> None:
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json() == deps.UNAVAILABLE_BODY


def test_metrics_is_not_part_of_the_openapi_schema(app: FastAPI) -> None:
    assert "/metrics" not in app.openapi()["paths"]
