from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api import deps
from services.api.main import create_app


class FakeConn:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.read_only = False
        self.queries: list[str] = []

    def execute(self, query: str, *args: Any) -> FakeConn:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self

    def fetchall(self) -> list[tuple[Any, ...]]:
        return []


def _use(app: FastAPI, conn: FakeConn) -> None:
    def override() -> Iterator[FakeConn]:
        yield conn

    app.dependency_overrides[deps.get_conn] = override


def test_mutually_exclusive_filters_are_422_before_database_access(app: FastAPI) -> None:
    conn = FakeConn()
    _use(app, conn)
    with TestClient(app) as client:
        response = client.get("/ops/channels?failing=true&attention=true")
    assert response.status_code == 422
    assert conn.queries == []


@pytest.mark.parametrize("query", ["attention=TRUE", "attention=1", "attention=yes", "failing=FALSE"])
def test_boolean_filters_are_strict(app: FastAPI, query: str) -> None:
    conn = FakeConn()
    _use(app, conn)
    with TestClient(app) as client:
        response = client.get(f"/ops/channels?{query}")
    assert response.status_code == 422
    assert conn.queries == []


def test_database_failure_is_fixed_503(app: FastAPI) -> None:
    conn = FakeConn(psycopg.OperationalError("secret"))
    _use(app, conn)
    with TestClient(app) as client:
        response = client.get("/ops/channels?attention=true")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
    assert "secret" not in response.text


@pytest.mark.integration
def test_attention_filter_fields_and_order(head_dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", head_dsn)
    with psycopg.connect(head_dsn, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO channels (channel_id, title, poll_failing_since, poll_failures, last_poll_err)"
            " VALUES ('UCfail', 'Zulu', %s, 2, 'failed')",
            (datetime(2026, 10, 3, tzinfo=UTC),),
        )
        conn.execute(
            "INSERT INTO channels (channel_id, title, feed_gap_detected_at, feed_gap_reason)"
            " VALUES ('UCgap', 'Alpha', %s, 'no_anchor')",
            (datetime(2026, 10, 1, tzinfo=UTC),),
        )
        conn.execute("INSERT INTO channels (channel_id, title) VALUES ('UCok', 'Healthy')")
    with TestClient(create_app()) as client:
        response = client.get("/ops/channels?attention=true")
    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["channel_id"] for item in items] == ["UCfail", "UCgap"]
    assert set(items[0]) == {
        "channel_id",
        "title",
        "poll_failing_since",
        "poll_failures",
        "last_poll_err",
        "feed_gap_detected_at",
        "feed_gap_reason",
    }
    assert items[1]["feed_gap_reason"] == "no_anchor"
    assert items[1]["feed_gap_detected_at"] == "2026-10-01T00:00:00Z"
