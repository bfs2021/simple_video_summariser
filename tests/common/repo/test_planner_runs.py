"""``common/repo/planner_runs.py`` against a real Postgres (issue #98)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from common.repo import planner_runs

pytestmark = pytest.mark.integration

T0 = datetime(2026, 10, 10, 12, 0, 0, tzinfo=UTC)


def _one(conn: psycopg.Connection, name: str) -> planner_runs.PlannerTaskRun:
    (row,) = [r for r in planner_runs.list_task_runs(conn) if r.task_name == name]
    return row


def test_list_is_empty_on_a_fresh_database(conn: psycopg.Connection) -> None:
    assert planner_runs.list_task_runs(conn) == []


def test_started_creates_the_row_with_only_start_and_interval(conn: psycopg.Connection) -> None:
    planner_runs.record_started(conn, "reap", interval_sec=60, at=T0)

    assert _one(conn, "reap") == planner_runs.PlannerTaskRun(
        "reap", 60.0, T0, None, None, None, None, None, None
    )


def test_started_keeps_the_previous_finish_columns_and_updates_interval(
    conn: psycopg.Connection,
) -> None:
    planner_runs.record_started(conn, "reap", interval_sec=60, at=T0)
    planner_runs.record_finished(
        conn, "reap", interval_sec=60, at=T0 + timedelta(seconds=1),
        outcome="ok", error=None, count=3,
    )
    later = T0 + timedelta(minutes=5)

    planner_runs.record_started(conn, "reap", interval_sec=90, at=later)

    row = _one(conn, "reap")
    assert row.last_started_at == later
    assert row.interval_sec == 90.0
    assert row.last_finished_at == T0 + timedelta(seconds=1)
    assert row.last_outcome == "ok"
    assert row.last_count == 3
    assert row.last_ok_at == T0 + timedelta(seconds=1)


def test_ok_then_failed_keeps_last_ok_at_and_clears_count(conn: psycopg.Connection) -> None:
    ok_at = T0 + timedelta(seconds=1)
    fail_at = T0 + timedelta(minutes=1)
    planner_runs.record_started(conn, "a", interval_sec=60, at=T0)
    planner_runs.record_finished(conn, "a", interval_sec=60, at=ok_at, outcome="ok", error=None, count=4)
    planner_runs.record_finished(
        conn, "a", interval_sec=60, at=fail_at, outcome="failed", error="ValueError: x", count=None
    )

    row = _one(conn, "a")
    assert row.last_outcome == "failed"
    assert row.last_error == "ValueError: x"
    assert row.last_count is None
    assert row.last_finished_at == fail_at
    assert row.last_ok_at == ok_at


def test_failed_then_ok_clears_the_error_and_moves_last_ok_at(conn: psycopg.Connection) -> None:
    planner_runs.record_finished(
        conn, "a", interval_sec=60, at=T0, outcome="failed", error="E: x", count=None
    )
    assert _one(conn, "a").last_ok_at is None

    later = T0 + timedelta(minutes=1)
    planner_runs.record_finished(conn, "a", interval_sec=60, at=later, outcome="ok", error=None, count=None)

    row = _one(conn, "a")
    assert (row.last_error, row.last_ok_at, row.last_outcome) == (None, later, "ok")


def test_skipped_locked_creates_a_row_and_touches_nothing_else(conn: psycopg.Connection) -> None:
    planner_runs.record_skipped_locked(conn, "a", interval_sec=60, at=T0)
    assert _one(conn, "a") == planner_runs.PlannerTaskRun(
        "a", 60.0, None, None, None, None, None, None, T0
    )

    planner_runs.record_started(conn, "a", interval_sec=60, at=T0 + timedelta(seconds=1))
    planner_runs.record_finished(
        conn, "a", interval_sec=60, at=T0 + timedelta(seconds=2), outcome="ok", error=None, count=1
    )
    before = _one(conn, "a")
    skip_at = T0 + timedelta(minutes=1)
    planner_runs.record_skipped_locked(conn, "a", interval_sec=999, at=skip_at)

    after = _one(conn, "a")
    assert after == dataclasses.replace(before, last_skipped_locked_at=skip_at)
    assert after.interval_sec == 60.0


def test_list_is_ordered_by_task_name(conn: psycopg.Connection) -> None:
    for name in ("poll", "reap", "audio"):
        planner_runs.record_started(conn, name, interval_sec=1, at=T0)

    assert [r.task_name for r in planner_runs.list_task_runs(conn)] == ["audio", "poll", "reap"]


def test_repo_functions_do_not_commit(head_dsn: str) -> None:
    with psycopg.connect(head_dsn) as writer, psycopg.connect(head_dsn) as reader:
        planner_runs.record_started(writer, "a", interval_sec=1, at=T0)

        assert planner_runs.list_task_runs(reader) == []
