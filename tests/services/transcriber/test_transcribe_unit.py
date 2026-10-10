"""Transcribe handler checks that need no database (issue #29).

Everything here happens before the handler reaches Postgres: id validation and the
progress callback. The rest is in ``test_transcribe.py`` (integration).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from common.errors import BugError, Cancelled
from common.models import AudioRef, TranscriptResult
from common.worker import JobContext
from services.transcriber.transcribe import (
    HEARTBEAT_SEC,
    make_transcribe_handler,
    progress_reporter,
)
from tests.services.transcriber.fakes import make_ctx, make_job, make_settings


class Reached(Exception):
    """Raised by the fake ``connect``: the handler got past validation."""


class Spy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch_normalized(self, video_id: str, dest: Path) -> AudioRef:
        self.calls.append("fetch")
        raise AssertionError("audio must not be fetched")

    def transcribe(self, audio: AudioRef, **_: Any) -> TranscriptResult:
        self.calls.append("transcribe")
        raise AssertionError("must not transcribe")

    def chunk(self, *_: Any, **__: Any) -> list[Any]:
        self.calls.append("chunk")
        return []


def build(spy: Spy) -> Any:
    def connect() -> Any:
        raise Reached

    return make_transcribe_handler(
        connect=connect,
        queue=cast(Any, object()),
        audio_source=spy,
        transcriber=spy,
        chunker=spy.chunk,
        settings=make_settings(),
    )


@pytest.mark.parametrize("bad", ["", "a" * 10, "a" * 12, "abc/../defgh", "abcdefgh ij", "abcdefghij\n"])
def test_an_invalid_video_id_is_a_bug_before_any_work(bad: str) -> None:
    spy = Spy()
    ctx, _ = make_ctx()

    with pytest.raises(BugError):
        build(spy)(make_job(bad), ctx)

    assert spy.calls == []


def test_a_video_id_with_a_leading_dash_is_valid() -> None:
    spy = Spy()
    ctx, _ = make_ctx()

    with pytest.raises(Reached):
        build(spy)(make_job("-wNyEUrxzFU"), ctx)


def progress_lines(logs: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = logs.records("info")
    return records


def test_progress_checks_for_cancellation_on_every_call() -> None:
    ctx, _ = make_ctx()
    on_progress = progress_reporter(ctx, 100.0)
    on_progress(1.0)

    ctx._cancel()

    with pytest.raises(Cancelled):
        on_progress(2.0)


def test_progress_logs_at_most_once_per_ten_percent() -> None:
    ctx, logs = make_ctx()
    on_progress = progress_reporter(ctx, 1000.0)

    for done in range(1000):
        on_progress(float(done))

    lines = progress_lines(logs)
    assert 1 < len(lines) <= 11


def test_progress_past_the_duration_still_logs_at_most_eleven_lines() -> None:
    ctx, logs = make_ctx()
    on_progress = progress_reporter(ctx, 10.0)

    for done in range(200):
        on_progress(done * 0.5)

    assert 1 < len(progress_lines(logs)) <= 11


def test_progress_with_a_zero_duration_does_not_raise() -> None:
    ctx, logs = make_ctx()
    on_progress = progress_reporter(ctx, 0.0)

    on_progress(0.0)
    on_progress(5.0)

    assert len(progress_lines(logs)) <= 11


def test_progress_ignores_a_non_finite_position() -> None:
    ctx, _ = make_ctx()
    on_progress = progress_reporter(ctx, 100.0)

    on_progress(float("nan"))
    on_progress(float("inf"))


# ------------------------------------------------- cadence (issue #96)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Events:
    """A progress hook that records ``(done, total, percent)`` per event."""

    def __init__(self) -> None:
        self.items: list[tuple[float, float, Any]] = []

    def __call__(self, stage: str, done: float, total: float, **fields: Any) -> None:
        assert stage == "transcribe"
        self.items.append((done, total, fields.get("percent")))


def reporter(duration: float, clock: Clock | None = None) -> tuple[Any, Events, Clock]:
    events = Events()
    clk = clock or Clock()
    ctx = JobContext(make_ctx()[1].logger, events)
    return progress_reporter(ctx, duration, clock=clk), events, clk


def test_a_line_is_reported_exactly_at_each_ten_percent_step() -> None:
    on_progress, events, _ = reporter(100.0)

    on_progress(0.0)
    on_progress(9.99)  # one short of the step: nothing new
    assert [e[2] for e in events.items] == [0]
    on_progress(10.0)  # exactly at the step
    on_progress(10.5)
    on_progress(19.99)
    on_progress(20.01)  # one past the step

    assert [e[2] for e in events.items] == [0, 10, 20]


def test_ten_percent_is_exact_for_floating_point_ratios() -> None:
    on_progress, events, _ = reporter(1000.0)

    on_progress(300.0)

    assert [e[2] for e in events.items] == [30]


def test_one_call_jumping_over_several_steps_reports_one_line() -> None:
    on_progress, events, _ = reporter(100.0)

    on_progress(55.0)

    assert events.items == [(55.0, 100.0, 55)]


def test_a_quiet_stretch_reports_a_heartbeat_line_once_the_audio_advanced() -> None:
    on_progress, events, clock = reporter(1000.0)
    on_progress(1.0)  # step 0 line
    clock.now += HEARTBEAT_SEC - 0.5
    on_progress(2.0)  # not yet
    assert len(events.items) == 1

    clock.now += 0.5
    on_progress(3.0)  # exactly HEARTBEAT_SEC since the last line

    assert [e[0] for e in events.items] == [1.0, 3.0]


def test_no_heartbeat_when_the_audio_position_did_not_advance() -> None:
    on_progress, events, clock = reporter(1000.0)
    on_progress(5.0)
    clock.now += HEARTBEAT_SEC * 3

    on_progress(5.0)
    on_progress(4.0)

    assert len(events.items) == 1


def test_a_heartbeat_restarts_the_wait_and_never_repeats_a_step() -> None:
    on_progress, events, clock = reporter(1000.0)
    on_progress(1.0)
    clock.now += HEARTBEAT_SEC
    on_progress(2.0)  # heartbeat, still step 0
    clock.now += HEARTBEAT_SEC - 1
    on_progress(3.0)  # only HEARTBEAT_SEC - 1 since the heartbeat

    assert [e[0] for e in events.items] == [1.0, 2.0]


def test_the_line_count_stays_within_eleven_plus_elapsed_over_heartbeat() -> None:
    on_progress, events, clock = reporter(1000.0)
    calls = 4000
    for i in range(calls):
        clock.now += 1.0
        on_progress(i / 4)  # 0..999.75, one call per wall second

    elapsed = calls * 1.0
    assert len(events.items) <= 11 + elapsed / HEARTBEAT_SEC


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf")])
def test_an_unknown_duration_follows_the_time_rule_alone(duration: float) -> None:
    on_progress, events, clock = reporter(duration)

    on_progress(0.0)
    on_progress(50.0)  # no step rule, no time has passed
    assert events.items == []

    clock.now += HEARTBEAT_SEC
    on_progress(60.0)
    on_progress(70.0)  # one line per callback, and the wait restarted

    assert events.items == [(60.0, 0.0, None)]


def test_a_position_past_the_duration_reports_at_most_one_hundred_percent() -> None:
    on_progress, events, _ = reporter(10.0)

    on_progress(25.0)

    assert [e[2] for e in events.items] == [100]


def test_cancellation_is_checked_before_anything_is_reported() -> None:
    events = Events()
    ctx = JobContext(make_ctx()[1].logger, events)
    on_progress = progress_reporter(ctx, 100.0, clock=Clock())
    ctx._cancel()

    with pytest.raises(Cancelled):
        on_progress(50.0)

    assert events.items == []


def test_the_worker_default_logs_the_documented_fields_per_line() -> None:
    ctx, logs = make_ctx()
    on_progress = progress_reporter(ctx, 200.0, clock=Clock())

    on_progress(100.0)

    (record,) = logs.records("info")
    assert record["event"] == "transcribe progress"
    assert (record["done_sec"], record["duration_sec"], record["percent"]) == (100.0, 200.0, 50)


def test_a_position_that_goes_backwards_does_not_repeat_a_step_line() -> None:
    on_progress, events, _ = reporter(100.0)

    on_progress(50.0)
    on_progress(10.0)  # a step below the last one
    on_progress(52.0)  # back in the same step as the first line

    assert [e[2] for e in events.items] == [50]
