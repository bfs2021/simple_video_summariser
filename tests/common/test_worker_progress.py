"""``JobContext.progress``: the one narrow progress method (issue #96)."""

from __future__ import annotations

from typing import Any

import pytest

from common.errors import Cancelled
from common.worker import JobContext
from tests.services.transcriber.fakes import RecordedLogs


def make(hook: Any = None) -> tuple[JobContext, RecordedLogs]:
    logs = RecordedLogs()
    return JobContext(logs.logger, hook), logs


def test_without_a_hook_progress_is_one_structured_info_line() -> None:
    ctx, logs = make()

    ctx.progress("transcribe", 30.0, 120.0, percent=25)

    assert logs.records() == [
        {
            "event": "transcribe progress",
            "stage": "transcribe",
            "done": 30.0,
            "total": 120.0,
            "percent": 25,
        }
    ]
    assert len(logs.records("info")) == 1


def test_with_a_hook_the_hook_gets_the_event_and_nothing_is_logged() -> None:
    seen: list[tuple[str, float, float, dict[str, Any]]] = []

    def hook(stage: str, done: float, total: float, **fields: Any) -> None:
        seen.append((stage, done, total, fields))

    ctx, logs = make(hook)

    ctx.progress("analyze", 2, 5, phase="chunk")

    assert seen == [("analyze", 2, 5, {"phase": "chunk"})]
    assert logs.records() == []


def test_a_raising_hook_is_swallowed_and_logged_once_per_job() -> None:
    calls = 0

    def hook(stage: str, done: float, total: float, **fields: Any) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("renderer broke")

    ctx, logs = make(hook)

    ctx.progress("a", 1, 2)
    ctx.progress("a", 2, 2)
    ctx.progress("a", 3, 2)

    assert calls == 3  # still called, never disabled
    assert len(logs.records("warning")) == 1


def test_the_warning_budget_is_per_job_context() -> None:
    def hook(stage: str, done: float, total: float, **fields: Any) -> None:
        raise OSError("closed")

    first, first_logs = make(hook)
    second, second_logs = make(hook)

    first.progress("a", 1, 2)
    second.progress("a", 1, 2)

    assert len(first_logs.records("warning")) == 1
    assert len(second_logs.records("warning")) == 1


@pytest.mark.parametrize("error", [Cancelled(), KeyboardInterrupt(), SystemExit(3)])
def test_cancellation_and_exit_signals_from_the_hook_propagate(error: BaseException) -> None:
    def hook(stage: str, done: float, total: float, **fields: Any) -> None:
        raise error

    ctx, logs = make(hook)

    with pytest.raises(type(error)):
        ctx.progress("a", 1, 2)

    assert logs.records("warning") == []
