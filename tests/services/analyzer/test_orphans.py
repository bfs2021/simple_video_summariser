"""The orphan pass (issue #160): a fake store and a fake canceller, no database, no network."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any

import pytest
from structlog.testing import capture_logs

from common.models import OrphanOutcome
from common.repo.llm_batches import OrphanBatch
from services.analyzer import orphans
from services.analyzer.orphans import handle_orphans, run_loop, run_pass
from tests.adapters.summarize.test_anthropic import KEY, make, raw

DAY = timedelta(hours=24)


class FakeStore:
    def __init__(self, *rows: OrphanBatch) -> None:
        self.rows = list(rows)
        self.listed_with: list[tuple[int, timedelta]] = []

    def list(self, limit: int, expired_after: timedelta) -> Sequence[OrphanBatch]:
        self.listed_with.append((limit, expired_after))
        return self.rows[:limit]

    def delete(self, row: OrphanBatch) -> bool:
        if row in self.rows:
            self.rows.remove(row)
            return True
        return False


class FakeCanceller:
    def __init__(self, *outcomes: OrphanOutcome) -> None:
        self.outcomes = list(outcomes)
        self.cancelled: list[str] = []

    def cancel_orphan_batch(self, batch_id: str) -> OrphanOutcome:
        self.cancelled.append(batch_id)
        return self.outcomes.pop(0) if self.outcomes else "cancelled"


def row(n: int = 1, *, expired: bool = False) -> OrphanBatch:
    return OrphanBatch(job_id=n, step_key="roster", batch_id=f"msgbatch_{n}", expired=expired)


def handle(
    store: FakeStore, canceller: FakeCanceller, *, limit: int = 5, stop: bool = False
) -> orphans.PassCounts:
    return handle_orphans(store, canceller, limit=limit, expired_after=DAY, stop=lambda: stop)


def events(logs: Sequence[Mapping[str, Any]]) -> list[str]:
    return [entry["event"] for entry in logs]


def test_a_dead_jobs_in_progress_batch_is_cancelled_once_and_its_row_deleted() -> None:
    store, canceller = FakeStore(row(1)), FakeCanceller("cancelled")

    with capture_logs() as logs:
        counts = handle(store, canceller)

    assert canceller.cancelled == ["msgbatch_1"]
    assert store.rows == []
    assert counts.cancelled == 1
    assert logs[0] == {
        "event": "anthropic_orphan_batch_cancelled",
        "job_id": 1,
        "step_key": "roster",
        "batch_id": "msgbatch_1",
        "log_level": "info",
    }


@pytest.mark.parametrize("outcome", ["not_cancellable", "cancelled"])
def test_a_batch_that_already_ended_has_its_row_deleted_and_nothing_raises(
    outcome: OrphanOutcome,
) -> None:
    store, canceller = FakeStore(row(1)), FakeCanceller(outcome)

    with capture_logs() as logs:
        handle(store, canceller)

    assert store.rows == []
    assert events(logs)[0] == f"anthropic_orphan_batch_{outcome}"


def test_retry_later_keeps_the_row_and_the_next_pass_cancels_again() -> None:
    store, canceller = FakeStore(row(1)), FakeCanceller("retry_later", "cancelled")

    with capture_logs() as logs:
        first = handle(store, canceller)
        assert store.rows == [row(1)]
        handle(store, canceller)

    assert first.retry_later == 1
    assert canceller.cancelled == ["msgbatch_1", "msgbatch_1"]
    assert store.rows == []
    assert events(logs)[0] == "anthropic_orphan_batch_retry_later"


def test_an_expired_row_is_deleted_without_any_api_call() -> None:
    store, canceller = FakeStore(row(1, expired=True)), FakeCanceller()

    with capture_logs() as logs:
        counts = handle(store, canceller)

    assert canceller.cancelled == []
    assert store.rows == []
    assert counts.expired == 1
    assert events(logs)[0] == "anthropic_orphan_batch_expired"


def test_an_empty_list_makes_no_api_call_and_logs_no_summary() -> None:
    store, canceller = FakeStore(), FakeCanceller()

    with capture_logs() as logs:
        counts = handle(store, canceller)

    assert canceller.cancelled == []
    assert counts.total == 0
    assert logs == []


def test_a_pass_that_handled_rows_logs_a_summary_with_counts() -> None:
    store = FakeStore(row(1), row(2, expired=True), row(3))
    canceller = FakeCanceller("cancelled", "retry_later")

    with capture_logs() as logs:
        handle(store, canceller)

    summary = logs[-1]
    assert summary["event"] == "anthropic_orphan_pass"
    assert (summary["cancelled"], summary["not_cancellable"]) == (1, 0)
    assert (summary["retry_later"], summary["expired"]) == (1, 1)


def test_the_limit_bounds_a_pass_and_is_passed_to_the_store() -> None:
    store, canceller = FakeStore(row(1), row(2), row(3)), FakeCanceller()

    handle(store, canceller, limit=2)

    assert store.listed_with == [(2, DAY)]
    assert canceller.cancelled == ["msgbatch_1", "msgbatch_2"]
    assert store.rows == [row(3)]


def test_once_shutdown_is_requested_no_further_row_is_handled() -> None:
    store, canceller = FakeStore(row(1), row(2), row(3)), FakeCanceller()
    state = {"checks": 0}

    def stop() -> bool:
        state["checks"] += 1
        return state["checks"] > 1  # allows the first row only

    handle_orphans(store, canceller, limit=5, expired_after=DAY, stop=stop)

    assert canceller.cancelled == ["msgbatch_1"]
    assert store.rows == [row(2), row(3)]


def test_production_defaults() -> None:
    assert (orphans.ORPHAN_LIMIT, orphans.ORPHAN_INTERVAL_SEC) == (20, 600)
    assert orphans.ORPHAN_EXPIRED_AFTER == timedelta(hours=24)


# --- a pass that cannot reach the database ---------------------------------------------


class Conn:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_a_database_error_is_logged_and_the_next_pass_opens_a_fresh_connection() -> None:
    conns: list[Conn] = []
    canceller = FakeCanceller()

    class Boom(FakeStore):
        def list(self, limit: int, expired_after: timedelta) -> Sequence[OrphanBatch]:
            raise OSError("connection lost")

    def connect() -> Conn:
        conns.append(Conn())
        return conns[-1]

    stores = iter([Boom(), FakeStore(row(1))])
    kwargs: dict[str, Any] = {"limit": 5, "expired_after": DAY, "stop": lambda: False}

    with capture_logs() as logs:
        first = run_pass(connect, canceller, store_factory=lambda c: next(stores), **kwargs)
        second = run_pass(connect, canceller, store_factory=lambda c: next(stores), **kwargs)

    assert first is None
    assert second is not None and second.cancelled == 1
    assert len(conns) == 2 and all(c.closed for c in conns)
    assert "analyzer.orphan_pass_failed" in events(logs)


def test_a_connect_failure_is_logged_and_does_not_raise() -> None:
    def connect() -> Conn:
        raise OSError("db down")

    with capture_logs() as logs:
        result = run_pass(
            connect,
            FakeCanceller(),
            store_factory=lambda c: FakeStore(),
            limit=5,
            expired_after=DAY,
            stop=lambda: False,
        )

    assert result is None
    assert events(logs) == ["analyzer.orphan_pass_failed"]


# --- the loop --------------------------------------------------------------------------


class FakeEvent:
    """An ``Event`` whose ``wait`` records the interval and is set after ``wakes`` waits."""

    def __init__(self, wakes: int) -> None:
        self.wakes = wakes
        self.waits: list[float] = []
        self._set = False

    def is_set(self) -> bool:
        return self._set

    def wait(self, timeout: float | None = None) -> bool:
        assert timeout is not None
        self.waits.append(timeout)
        if len(self.waits) >= self.wakes:
            self._set = True
        return self._set


def test_the_loop_runs_a_pass_at_once_then_one_per_interval_until_stopped() -> None:
    passes: list[int] = []
    event = FakeEvent(wakes=2)

    run_loop(lambda: passes.append(len(passes)), interval_sec=7.5, stop=event)

    assert passes == [0, 1]
    assert event.waits == [7.5, 7.5]


def test_the_loop_runs_no_pass_when_stopped_before_it_starts() -> None:
    passes: list[int] = []
    event = threading.Event()
    event.set()

    run_loop(lambda: passes.append(1), interval_sec=7.5, stop=event)

    assert passes == []


def test_the_loop_survives_a_pass_that_raises() -> None:
    calls: list[int] = []
    event = FakeEvent(wakes=2)

    def flaky() -> None:
        calls.append(1)
        raise RuntimeError("boom")

    with capture_logs() as logs:
        run_loop(flaky, interval_sec=1, stop=event)

    assert calls == [1, 1]
    assert events(logs).count("analyzer.orphan_pass_crashed") == 2


def test_the_key_appears_in_no_log_record_on_the_retry_later_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    leaky = raw(500, json.dumps({"type": "error", "error": {"type": "api_error", "message": f"key {KEY}"}}))
    summarizer, _ = make(leaky, RuntimeError(f"boom {KEY}"), batch=True)
    store = FakeStore(row(1), row(2))

    with capture_logs() as logs:
        counts = handle_orphans(store, summarizer, limit=5, expired_after=DAY, stop=lambda: False)

    assert counts.retry_later == 2
    assert KEY not in caplog.text
    assert KEY not in json.dumps(logs, default=str)
    for record in caplog.records:
        assert KEY not in json.dumps(record.__dict__, default=str)
