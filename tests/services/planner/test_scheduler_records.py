"""Run recording in the scheduler (issue #98), driven with an in-memory recorder.

No Docker, no real sleep: the wall clock and the monotonic clock are both
injected, so every recorded timestamp is asserted exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import psycopg
import pytest
from structlog.testing import capture_logs

from services.planner.scheduler import Scheduler, Task
from tests.services.planner.test_scheduler import FakeLocks

T0 = datetime(2026, 10, 10, 12, 0, 0, tzinfo=UTC)


class FakeWallClock:
    """Each call returns the next second, so start and finish differ."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> datetime:
        self.calls += 1
        return T0 + timedelta(seconds=self.calls - 1)


class FakeRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []
        self.raise_on: set[str] = set()
        self.exc: Exception = RuntimeError("recorder down")

    def _maybe_raise(self, kind: str) -> None:
        if kind in self.raise_on or "all" in self.raise_on:
            raise self.exc

    def task_started(self, name: str, *, interval_sec: float, at: datetime) -> None:
        self.events.append(("started", name, interval_sec, at))
        self._maybe_raise("started")

    def task_finished(
        self,
        name: str,
        *,
        interval_sec: float,
        at: datetime,
        outcome: Literal["ok", "failed"],
        error: str | None,
        count: int | None,
    ) -> None:
        self.events.append(("finished", name, interval_sec, at, outcome, error, count))
        self._maybe_raise("finished")

    def task_skipped_locked(self, name: str, *, interval_sec: float, at: datetime) -> None:
        self.events.append(("skipped", name, interval_sec, at))
        self._maybe_raise("skipped")


def _build(
    tmp_path: Path,
    recorder: FakeRecorder,
    locks: FakeLocks,
    *tasks: Task,
) -> Scheduler:
    return Scheduler(
        list(tasks),
        locks,
        clock=lambda: 0.0,
        sleep=lambda _s: None,
        liveness_path=tmp_path / "hb",
        install_signal_handlers=False,
        recorder=recorder,
        wall_clock=FakeWallClock(),
    )


def _returns(value: int | None) -> Callable[[], int | None]:
    return lambda: value


def _raises(exc: BaseException) -> Callable[[], int | None]:
    def fn() -> int | None:
        raise exc

    return fn


def test_ok_run_records_start_then_finish_with_the_returned_count(tmp_path: Path) -> None:
    rec = FakeRecorder()
    sched = _build(tmp_path, rec, FakeLocks(), Task("a", 60, _returns(3)))

    sched.run_once()

    assert rec.events == [
        ("started", "a", 60, T0),
        ("finished", "a", 60, T0 + timedelta(seconds=1), "ok", None, 3),
    ]


def test_a_task_returning_none_records_a_null_count(tmp_path: Path) -> None:
    rec = FakeRecorder()
    _build(tmp_path, rec, FakeLocks(), Task("a", 60, _returns(None))).run_once()

    assert rec.events[1][-1] is None


def test_started_is_recorded_before_the_task_runs(tmp_path: Path) -> None:
    rec = FakeRecorder()
    seen: list[int] = []
    task = Task("a", 60, lambda: seen.append(len(rec.events)))

    _build(tmp_path, rec, FakeLocks(), task).run_once()

    assert seen == [1]


def test_failed_run_records_class_and_message_without_traceback(tmp_path: Path) -> None:
    rec = FakeRecorder()
    sched = _build(tmp_path, rec, FakeLocks(), Task("a", 60, _raises(ValueError("boom"))))

    assert sched.run_once() is False

    assert rec.events[1] == (
        "finished", "a", 60, T0 + timedelta(seconds=1), "failed", "ValueError: boom", None
    )


def test_error_text_is_truncated_to_500_characters(tmp_path: Path) -> None:
    rec = FakeRecorder()
    exc = ValueError("x" * 2000)
    _build(tmp_path, rec, FakeLocks(), Task("a", 60, _raises(exc))).run_once()

    error = rec.events[1][5]
    assert isinstance(error, str)
    assert len(error) == 500
    assert error.startswith("ValueError: xxx")


def test_error_text_of_exactly_500_characters_is_kept_whole(tmp_path: Path) -> None:
    rec = FakeRecorder()
    message = "x" * (500 - len("ValueError: "))
    _build(tmp_path, rec, FakeLocks(), Task("a", 60, _raises(ValueError(message)))).run_once()

    assert rec.events[1][5] == f"ValueError: {message}"


def test_skipped_locked_records_only_the_skip(tmp_path: Path) -> None:
    rec = FakeRecorder()
    locks = FakeLocks()
    locks.held_elsewhere.add("a")
    calls: list[str] = []
    task = Task("a", 60, lambda: calls.append("a"))

    _build(tmp_path, rec, locks, task).run_once()

    assert rec.events == [("skipped", "a", 60, T0)]
    assert calls == []


def test_database_unavailable_records_nothing(tmp_path: Path) -> None:
    rec = FakeRecorder()
    locks = FakeLocks()
    locks.fail_try_lock = 1

    _build(tmp_path, rec, locks, Task("a", 60, _returns(1))).run_once()

    assert rec.events == []


def test_run_loop_records_like_run_once(tmp_path: Path) -> None:
    rec = FakeRecorder()
    _build(tmp_path, rec, FakeLocks(), Task("a", 60, _returns(2))).run(max_iterations=1)

    assert [e[0] for e in rec.events] == ["started", "finished"]


def test_a_recorder_that_always_raises_does_not_stop_tasks_or_lock_release(
    tmp_path: Path,
) -> None:
    rec = FakeRecorder()
    rec.raise_on = {"all"}
    rec.exc = psycopg.OperationalError("db gone")
    locks = FakeLocks()
    calls: list[str] = []

    def fn_a() -> int:
        calls.append("a")
        return 1

    def fn_b() -> int:
        calls.append("b")
        raise ValueError("b fails")

    sched = _build(tmp_path, rec, locks, Task("a", 60, fn_a), Task("b", 60, fn_b))
    with capture_logs() as logs:
        ok = sched.run_once()

    assert calls == ["a", "b"]
    assert ok is False  # b's own failure still counts
    assert locks.released == ["a", "b"]
    failed = [e for e in logs if e["event"] == "planner.record_failed"]
    assert len(failed) == 4
    assert {e["log_level"] for e in failed} == {"error"}
    assert {e["task"] for e in failed} == {"a", "b"}
    assert {e["error_class"] for e in failed} == {"OperationalError"}


def test_a_failing_skip_record_is_logged_and_the_run_continues(tmp_path: Path) -> None:
    rec = FakeRecorder()
    rec.raise_on = {"skipped"}
    locks = FakeLocks()
    locks.held_elsewhere.add("a")
    calls: list[str] = []
    sched = _build(
        tmp_path, rec, locks, Task("a", 60, _returns(1)), Task("b", 60, lambda: calls.append("b"))
    )

    with capture_logs() as logs:
        assert sched.run_once() is True

    assert calls == ["b"]
    assert [e["task"] for e in logs if e["event"] == "planner.record_failed"] == ["a"]


def test_planner_task_log_line_is_unchanged(tmp_path: Path) -> None:
    rec = FakeRecorder()
    with capture_logs() as logs:
        _build(tmp_path, rec, FakeLocks(), Task("a", 60, _returns(3))).run_once()

    line = [e for e in logs if e["event"] == "planner.task"]
    assert len(line) == 1
    assert set(line[0]) == {"event", "log_level", "task", "duration_ms", "outcome", "count"}


def test_default_scheduler_without_a_recorder_still_runs(tmp_path: Path) -> None:
    sched = Scheduler(
        [Task("a", 60, _returns(1))],
        FakeLocks(),
        sleep=lambda _s: None,
        liveness_path=tmp_path / "hb",
        install_signal_handlers=False,
    )

    assert sched.run_once() is True


@pytest.mark.parametrize("which", ["started", "finished"])
def test_a_raising_recorder_does_not_change_the_task_outcome(tmp_path: Path, which: str) -> None:
    rec = FakeRecorder()
    rec.raise_on = {which}

    assert _build(tmp_path, rec, FakeLocks(), Task("a", 60, _returns(1))).run_once() is True


def test_default_wall_clock_is_utc_now(tmp_path: Path) -> None:
    rec = FakeRecorder()
    sched = Scheduler(
        [Task("a", 60, _returns(1))],
        FakeLocks(),
        sleep=lambda _s: None,
        liveness_path=tmp_path / "hb",
        install_signal_handlers=False,
        recorder=rec,
    )
    before = datetime.now(UTC)

    sched.run_once()

    after = datetime.now(UTC)
    started_at = rec.events[0][3]
    assert isinstance(started_at, datetime)
    assert started_at.utcoffset() == timedelta(0)
    assert before <= started_at <= after
