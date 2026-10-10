"""Unit tests for the planner's stale-job reaper pass (issue #35). No database."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Any

import pytest

from common.config import Settings
from common.queue import Job, JobQueue
from services.planner.reaper import reap_stale_jobs


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"DATABASE_URL": "postgresql://u:p@h/db"}
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


class FakeQueue:
    """A ``JobQueue`` whose ``reap_stale`` returns a canned count or raises."""

    def __init__(self, reaped: int = 0, error: Exception | None = None) -> None:
        self._reaped = reaped
        self._error = error
        self.reap_calls: list[tuple[Any, ...]] = []

    def enqueue(
        self,
        kind: str,
        video_id: str,
        *,
        dedupe_key: str = "default",
        payload: dict[str, Any] | None = None,
        priority: int = 0,
        run_after: datetime | None = None,
    ) -> int | None:
        raise AssertionError("the reaper must not enqueue")

    def claim(
        self, kinds: Sequence[str], *, worker: str, dedupe_key: str | None = None
    ) -> AbstractContextManager[Job | None]:
        raise AssertionError("the reaper must not claim")

    def heartbeat(self, job_id: int, worker: str) -> bool:
        raise AssertionError("the reaper must not heartbeat")

    def reap_stale(self, older_than_sec: int | None = None) -> int:
        self.reap_calls.append((older_than_sec,))
        if self._error is not None:
            raise self._error
        return self._reaped


def _reap_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == "planner.reap"]


def test_fake_queue_satisfies_the_job_queue_port() -> None:
    queue: JobQueue = FakeQueue()
    assert queue.reap_stale() == 0


@pytest.mark.parametrize("reaped", [0, 1, 7])
def test_returns_the_number_of_jobs_the_queue_reaped(reaped: int) -> None:
    assert reap_stale_jobs(FakeQueue(reaped=reaped), _settings()) == reaped


def test_makes_exactly_one_pass_with_the_default_threshold() -> None:
    queue = FakeQueue(reaped=2)

    reap_stale_jobs(queue, _settings())

    assert queue.reap_calls == [(None,)]


def test_one_reaped_job_logs_one_info_line_with_count_and_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        reap_stale_jobs(FakeQueue(reaped=1), _settings(REAP_AFTER_SEC=420))

    [record] = _reap_records(caplog)
    assert record.levelno == logging.INFO
    assert record.__dict__["reaped"] == 1
    assert record.__dict__["threshold_sec"] == 420


def test_many_reaped_jobs_log_one_info_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        reap_stale_jobs(FakeQueue(reaped=5), _settings())

    [record] = _reap_records(caplog)
    assert record.levelno == logging.INFO
    assert record.__dict__["reaped"] == 5
    assert record.__dict__["threshold_sec"] == 300


def test_zero_reaped_jobs_log_the_same_line_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        reap_stale_jobs(FakeQueue(reaped=0), _settings())

    [record] = _reap_records(caplog)
    assert record.levelno == logging.DEBUG
    assert record.__dict__["reaped"] == 0
    assert record.__dict__["threshold_sec"] == 300


def test_zero_reaped_jobs_are_silent_at_info_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        reap_stale_jobs(FakeQueue(reaped=0), _settings())

    assert _reap_records(caplog) == []


def test_a_raising_queue_propagates_the_same_exception_and_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    boom = ConnectionError("database is down")

    with caplog.at_level(logging.DEBUG), pytest.raises(ConnectionError) as exc_info:
        reap_stale_jobs(FakeQueue(error=boom), _settings())

    assert exc_info.value is boom
    assert _reap_records(caplog) == []
