"""Run records against a real Postgres (issue #98)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest

from common.config import Settings
from common.repo import planner_runs
from services.planner import main as planner_main
from services.planner.main import PostgresAdvisoryLocks, PostgresRunRecorder
from services.planner.scheduler import Scheduler, Task

pytestmark = pytest.mark.integration

T0 = datetime(2026, 10, 10, 12, 0, 0, tzinfo=UTC)


class Wall:
    def __init__(self) -> None:
        self.t = T0

    def __call__(self) -> datetime:
        value = self.t
        self.t += timedelta(seconds=1)
        return value


@pytest.fixture
def parts(head_dsn: str) -> Iterator[tuple[str, list[PostgresAdvisoryLocks | PostgresRunRecorder]]]:
    made: list[PostgresAdvisoryLocks | PostgresRunRecorder] = []
    yield head_dsn, made
    for item in made:
        item.close()


def _scheduler(
    parts: tuple[str, list[PostgresAdvisoryLocks | PostgresRunRecorder]],
    tasks: list[Task],
    wall: Wall,
) -> Scheduler:
    dsn, made = parts
    locks = PostgresAdvisoryLocks(lambda: psycopg.connect(dsn))
    recorder = PostgresRunRecorder(lambda: psycopg.connect(dsn))
    made += [locks, recorder]
    return Scheduler(
        tasks,
        locks,
        recorder=recorder,
        wall_clock=wall,
        sleep=lambda _s: None,
        install_signal_handlers=False,
    )


def _boom() -> int:
    raise RuntimeError("it broke")


def test_run_once_leaves_one_exact_row_per_task(parts, head_dsn: str) -> None:  # type: ignore[no-untyped-def]
    sched = _scheduler(
        parts,
        [Task("audio_retention", 60, lambda: 3), Task("poll_channels", 90, lambda: None), Task("reap", 120, _boom)],
        Wall(),
    )

    assert sched.run_once() is False

    with psycopg.connect(head_dsn) as conn:
        rows = planner_runs.list_task_runs(conn)
    s = timedelta(seconds=1)
    assert rows == [
        planner_runs.PlannerTaskRun("audio_retention", 60.0, T0, T0 + s, "ok", None, 3, T0 + s, None),
        planner_runs.PlannerTaskRun(
            "poll_channels", 90.0, T0 + 2 * s, T0 + 3 * s, "ok", None, None, T0 + 3 * s, None
        ),
        planner_runs.PlannerTaskRun(
            "reap", 120.0, T0 + 4 * s, T0 + 5 * s, "failed", "RuntimeError: it broke", None, None, None
        ),
    ]


def test_a_failure_after_an_ok_run_keeps_last_ok_at(parts, head_dsn: str) -> None:  # type: ignore[no-untyped-def]
    state = {"fail": False}

    def flaky() -> int:
        if state["fail"]:
            raise ValueError("later")
        return 1

    sched = _scheduler(parts, [Task("reap", 60, flaky)], Wall())
    sched.run_once()
    state["fail"] = True
    sched.run_once()

    with psycopg.connect(head_dsn) as conn:
        (row,) = planner_runs.list_task_runs(conn)
    s = timedelta(seconds=1)
    assert row.last_outcome == "failed"
    assert row.last_ok_at == T0 + s
    assert row.last_finished_at == T0 + 3 * s
    assert row.last_error == "ValueError: later"


def test_a_second_scheduler_only_sets_last_skipped_locked_at(parts, head_dsn: str) -> None:  # type: ignore[no-untyped-def]
    second_wall = Wall()
    second_wall.t = T0 + timedelta(hours=1)
    second = _scheduler(parts, [Task("reap", 60, lambda: 99)], second_wall)
    first_inside: list[bool] = []

    def first_task() -> int:
        first_inside.append(second.run_once())  # first still holds "a"
        return 7

    first = _scheduler(parts, [Task("reap", 60, first_task)], Wall())

    first.run_once()

    assert first_inside == [True]
    with psycopg.connect(head_dsn) as conn:
        (row,) = planner_runs.list_task_runs(conn)
    s = timedelta(seconds=1)
    assert row.last_skipped_locked_at == T0 + timedelta(hours=1)
    assert (row.last_outcome, row.last_count) == ("ok", 7)
    assert (row.last_started_at, row.last_finished_at) == (T0, T0 + s)


def test_a_killed_run_leaves_started_after_finished(parts, head_dsn: str) -> None:  # type: ignore[no-untyped-def]
    def killed() -> int:
        raise KeyboardInterrupt  # not swallowed, finish is never recorded

    sched = _scheduler(parts, [Task("reap", 60, lambda: 1)], Wall())
    sched.run_once()
    later = Wall()
    later.t = T0 + timedelta(minutes=5)
    sched = _scheduler(parts, [Task("reap", 60, killed)], later)

    with pytest.raises(KeyboardInterrupt):
        sched.run_once()

    with psycopg.connect(head_dsn) as conn:
        (row,) = planner_runs.list_task_runs(conn)
    assert row.last_started_at is not None and row.last_finished_at is not None
    assert row.last_started_at > row.last_finished_at


def test_unknown_tasks_keep_their_rows(parts, head_dsn: str) -> None:  # type: ignore[no-untyped-def]
    with psycopg.connect(head_dsn, autocommit=True) as conn:
        planner_runs.record_started(conn, "retired_task", interval_sec=60, at=T0)
    _scheduler(parts, [Task("reap", 60, lambda: 1)], Wall()).run_once()

    with psycopg.connect(head_dsn) as conn:
        assert [r.task_name for r in planner_runs.list_task_runs(conn)] == ["reap", "retired_task"]


def test_recorder_reconnects_after_the_database_drops_its_connection(head_dsn: str) -> None:
    recorder = PostgresRunRecorder(lambda: psycopg.connect(head_dsn))
    recorder.task_started("a", interval_sec=1, at=T0)
    assert recorder._conn is not None
    recorder._conn.close()  # simulate a dropped connection
    with pytest.raises(psycopg.Error):
        recorder.task_started("a", interval_sec=1, at=T0)

    recorder.task_started("a", interval_sec=1, at=T0 + timedelta(seconds=5))

    with psycopg.connect(head_dsn) as conn:
        (row,) = planner_runs.list_task_runs(conn)
    assert row.last_started_at == T0 + timedelta(seconds=5)
    recorder.close()


def test_build_scheduler_records_every_registry_task_once(
    head_dsn: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(planner_main, "connect", lambda: psycopg.connect(head_dsn))
    settings = Settings(DATABASE_URL="postgresql://u:p@h/db", AUDIO_DIR=tmp_path)  # type: ignore[arg-type]
    scheduler = planner_main.build_scheduler(settings)

    assert scheduler.run_once() is True

    with psycopg.connect(head_dsn) as conn:
        rows = planner_runs.list_task_runs(conn)
    assert [r.task_name for r in rows] == sorted(planner_main.LOCK_KEYS)
    assert all(r.last_outcome == "ok" and r.last_finished_at is not None for r in rows)
