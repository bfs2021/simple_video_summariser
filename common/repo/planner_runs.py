"""Planner task run records (issue #98; architecture.md §6 ``planner_task_runs``).

One row per periodic planner task, overwritten in place: when it last
started and finished, how it ended and why it failed. The planner writes
through the ``record_*`` functions on an autocommit connection; read-side
callers (the ops router, metrics) use ``list_task_runs``.

Like every ``common/repo/`` function these take a psycopg connection first,
use parameterized SQL and never commit or roll back. Nothing deletes rows: a
task that leaves the registry keeps its old row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import psycopg

_COLUMNS = (
    "task_name, interval_sec, last_started_at, last_finished_at, last_outcome, "
    "last_error, last_count, last_ok_at, last_skipped_locked_at"
)


@dataclass(frozen=True, slots=True)
class PlannerTaskRun:
    """One ``planner_task_runs`` row."""

    task_name: str
    interval_sec: float
    last_started_at: datetime | None
    last_finished_at: datetime | None
    last_outcome: Literal["ok", "failed"] | None
    last_error: str | None
    last_count: int | None
    last_ok_at: datetime | None
    last_skipped_locked_at: datetime | None


def list_task_runs(conn: psycopg.Connection[Any]) -> list[PlannerTaskRun]:
    """Every recorded task, ordered by ``task_name``."""
    rows = conn.execute(f"SELECT {_COLUMNS} FROM planner_task_runs ORDER BY task_name").fetchall()
    return [PlannerTaskRun(*row) for row in rows]


def record_started(
    conn: psycopg.Connection[Any], name: str, *, interval_sec: float, at: datetime
) -> None:
    """Set ``last_started_at`` and ``interval_sec``; other columns keep their values."""
    conn.execute(
        """
        INSERT INTO planner_task_runs (task_name, interval_sec, last_started_at)
        VALUES (%s, %s, %s)
        ON CONFLICT (task_name) DO UPDATE
           SET interval_sec = EXCLUDED.interval_sec,
               last_started_at = EXCLUDED.last_started_at
        """,
        (name, interval_sec, at),
    )


def record_finished(
    conn: psycopg.Connection[Any],
    name: str,
    *,
    interval_sec: float,
    at: datetime,
    outcome: Literal["ok", "failed"],
    error: str | None,
    count: int | None,
) -> None:
    """Set the finish columns. ``last_ok_at`` moves only on ``ok``."""
    conn.execute(
        """
        INSERT INTO planner_task_runs
            (task_name, interval_sec, last_finished_at, last_outcome,
             last_error, last_count, last_ok_at)
        VALUES (%(name)s, %(interval)s, %(at)s, %(outcome)s, %(error)s, %(count)s,
                CASE WHEN %(outcome)s = 'ok' THEN %(at)s END)
        ON CONFLICT (task_name) DO UPDATE
           SET interval_sec = EXCLUDED.interval_sec,
               last_finished_at = EXCLUDED.last_finished_at,
               last_outcome = EXCLUDED.last_outcome,
               last_error = EXCLUDED.last_error,
               last_count = EXCLUDED.last_count,
               last_ok_at = CASE WHEN EXCLUDED.last_outcome = 'ok'
                                 THEN EXCLUDED.last_finished_at
                                 ELSE planner_task_runs.last_ok_at END
        """,
        {
            "name": name,
            "interval": interval_sec,
            "at": at,
            "outcome": outcome,
            "error": error,
            "count": count,
        },
    )


def record_skipped_locked(
    conn: psycopg.Connection[Any], name: str, *, interval_sec: float, at: datetime
) -> None:
    """Set only ``last_skipped_locked_at`` (creating the row if missing)."""
    conn.execute(
        """
        INSERT INTO planner_task_runs (task_name, interval_sec, last_skipped_locked_at)
        VALUES (%s, %s, %s)
        ON CONFLICT (task_name) DO UPDATE
           SET last_skipped_locked_at = EXCLUDED.last_skipped_locked_at
        """,
        (name, interval_sec, at),
    )
