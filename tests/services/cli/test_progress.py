"""``ProgressPrinter`` and its wiring into ``run_queue`` (issue #96)."""

from __future__ import annotations

import io
from typing import Any

import pytest

from common.errors import Cancelled
from common.queue import Job
from common.worker import JobContext
from services.cli.inprocess import InProcessQueue
from services.cli.main import PipelineError, run_queue
from services.cli.progress import ProgressPrinter

VID = "iG9CE55wbtY"


class Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def printer(clock: Clock | None = None) -> tuple[ProgressPrinter, io.StringIO]:
    out = io.StringIO()
    return ProgressPrinter(clock=clock or Clock(), stream=lambda: out), out


def test_a_transcribe_line_shows_position_total_percent_and_elapsed() -> None:
    clock = Clock()
    p, out = printer(clock)
    p.begin()
    clock.now += 245  # 4:05

    p("transcribe", 750.0, 3730.0)

    assert out.getvalue() == "[transcribe] 12:30 / 1:02:10 audio (20%) - elapsed 4:05\n"


def test_a_position_past_the_duration_shows_one_hundred_percent_and_the_total() -> None:
    p, out = printer()
    p.begin()

    p("transcribe", 500.0, 120.0)

    assert out.getvalue() == "[transcribe] 2:00 / 2:00 audio (100%) - elapsed 0:00\n"


@pytest.mark.parametrize("total", [0.0, -5.0, float("nan"), float("inf")])
def test_an_unknown_duration_shows_position_and_elapsed_only(total: float) -> None:
    p, out = printer()
    p.begin()

    p("transcribe", 75.0, total)

    assert out.getvalue() == "[transcribe] 1:15 audio - elapsed 0:00\n"


@pytest.mark.parametrize("done", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_position_prints_nothing(done: float) -> None:
    p, out = printer()

    p("transcribe", done, 100.0)

    assert out.getvalue() == ""


def test_elapsed_counts_from_the_first_event_when_begin_was_not_called() -> None:
    clock = Clock()
    p, out = printer(clock)
    p("transcribe", 10.0, 100.0)
    clock.now += 61

    p("transcribe", 20.0, 100.0)

    assert out.getvalue().splitlines()[1].endswith("(20%) - elapsed 1:01")


def test_analyze_lines_are_roster_chunk_k_of_n_and_reduce() -> None:
    p, out = printer()

    p("analyze", 0, 2, phase="roster")
    p("analyze", 1, 2, phase="chunk", chunk=1)
    p("analyze", 2, 2, phase="chunk", chunk=2)
    p("analyze", 2, 2, phase="reduce")

    assert out.getvalue().splitlines() == [
        "[analyze] roster",
        "[analyze] chunk 1/2",
        "[analyze] chunk 2/2",
        "[analyze] reduce",
    ]


def test_an_unknown_stage_or_phase_prints_nothing() -> None:
    p, out = printer()

    p("mystery", 1, 2)
    p("analyze", 1, 2, phase="mystery")
    p("analyze", 1, 2)

    assert out.getvalue() == ""


def test_every_line_is_one_plain_newline_terminated_line() -> None:
    p, out = printer()
    p("transcribe", 10.0, 100.0)
    p("transcribe", 20.0, 0.0)
    p("analyze", 0, 1, phase="roster")
    p("analyze", 1, 1, phase="chunk")
    p("analyze", 1, 1, phase="reduce")

    text = out.getvalue()
    assert text.endswith("\n")
    assert text.count("\n") == 5
    assert "\r" not in text and "\x1b" not in text
    assert all(line.isprintable() for line in text.splitlines())


def test_a_hostile_stage_name_cannot_inject_control_bytes() -> None:
    p, out = printer()
    # Only known stages render, so a hostile name prints nothing at all.
    p("analyze\x1b[2J\r", 1, 1, phase="roster")

    assert out.getvalue() == ""


class Broken:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def write(self, text: str) -> int:
        raise self.error

    def flush(self) -> None:
        raise self.error


@pytest.mark.parametrize("error", [BrokenPipeError(), OSError(5, "io"), ValueError("closed file")])
def test_a_failing_stderr_never_raises(error: Exception) -> None:
    p = ProgressPrinter(stream=lambda: Broken(error))  # type: ignore[arg-type, return-value]

    p("transcribe", 10.0, 100.0)
    p("analyze", 0, 1, phase="roster")


def test_the_default_stream_is_stderr_resolved_at_call_time(
    capsys: pytest.CaptureFixture[str],
) -> None:
    p = ProgressPrinter()

    p("analyze", 0, 1, phase="roster")

    out, err = capsys.readouterr()
    assert (out, err) == ("", "[analyze] roster\n")


# ------------------------------------------------------------------ run_queue


def seeded() -> InProcessQueue:
    queue = InProcessQueue()
    queue.enqueue("ingest", VID, priority=10)
    return queue


def test_run_queue_gives_every_handler_a_context_that_reports_to_the_hook() -> None:
    queue = seeded()
    events: list[tuple[str, float, float]] = []

    def hook(stage: str, done: float, total: float, **fields: Any) -> None:
        events.append((stage, done, total))

    def ingest(job: Job, ctx: JobContext) -> None:
        ctx.progress("analyze", 1, 3, phase="chunk")

    run_queue(queue, {"ingest": ingest}, VID, progress=hook)

    assert events == [("analyze", 1, 3)]


def test_progress_through_run_queue_goes_to_stderr_once_and_stdout_stays_clean(
    capsys: pytest.CaptureFixture[str],
) -> None:
    queue = seeded()
    p = ProgressPrinter()

    def ingest(job: Job, ctx: JobContext) -> None:
        ctx.progress("analyze", 0, 1, phase="roster")
        ctx.progress("analyze", 1, 1, phase="chunk")

    run_queue(queue, {"ingest": ingest}, VID, progress=p)

    out, err = capsys.readouterr()
    assert out == ""
    assert err.splitlines() == ["[analyze] roster", "[analyze] chunk 1/1"]


def test_a_cancelled_raised_by_the_hook_fails_the_stage_as_cancelled() -> None:
    queue = seeded()

    def hook(stage: str, done: float, total: float, **fields: Any) -> None:
        raise Cancelled()

    def ingest(job: Job, ctx: JobContext) -> None:
        ctx.progress("analyze", 0, 1, phase="roster")

    with pytest.raises(PipelineError) as caught:
        run_queue(queue, {"ingest": ingest}, VID, progress=hook)

    assert isinstance(caught.value.cause, Cancelled)


def test_a_keyboard_interrupt_from_the_hook_passes_through_run_queue() -> None:
    queue = seeded()

    def hook(stage: str, done: float, total: float, **fields: Any) -> None:
        raise KeyboardInterrupt

    def ingest(job: Job, ctx: JobContext) -> None:
        ctx.progress("analyze", 0, 1, phase="roster")

    with pytest.raises(KeyboardInterrupt):
        run_queue(queue, {"ingest": ingest}, VID, progress=hook)


def test_a_raising_hook_does_not_change_the_run() -> None:
    queue = seeded()
    ran: list[str] = []

    def hook(stage: str, done: float, total: float, **fields: Any) -> None:
        raise RuntimeError("renderer")

    def ingest(job: Job, ctx: JobContext) -> None:
        ctx.progress("analyze", 0, 1, phase="roster")
        ran.append("after")

    assert run_queue(queue, {"ingest": ingest}, VID, progress=hook) == ["ingest"]
    assert ran == ["after"]


def test_a_negative_position_shows_zero() -> None:
    p, out = printer()
    p.begin()

    p("transcribe", -5.0, 100.0)
    p("transcribe", -5.0, 0.0)

    assert out.getvalue().splitlines() == [
        "[transcribe] 0:00 / 1:40 audio (0%) - elapsed 0:00",
        "[transcribe] 0:00 audio - elapsed 0:00",
    ]
