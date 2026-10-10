"""Tests for ``common.worker`` (issue #13, architecture.md §1, §4, §5, §11.5).

Everything here drives ``Worker`` through its public surface -
``run(max_iterations=...)`` and ``request_shutdown()`` - against a
``FakeQueue`` that reimplements just enough of ``PostgresQueue.claim``'s
contract (issue #11: plain exceptions are swallowed and recorded,
``Cancelled``/``KeyboardInterrupt``/``SystemExit`` are recorded then
re-raised) to exercise the loop without a database.

The heartbeat tick and the shutdown-grace deadline are both driven through
an injected ``timer_factory``: a ``FakeTimerFactory`` hands back a handle
whose scheduled callback the test fires by hand, so heartbeat and grace
tests need no real waiting at all (`poll_sec` idle-wait tests inject a
`sleep` spy for the same reason). The one exception is
``test_real_sigterm_stops_an_idle_worker``, which sends an actual
``SIGTERM`` to a subprocess to prove the signal handler is really
installed; a pipe (the child's stdout) makes that test wait on readiness,
not on a sleep.

The last section holds three ``integration`` tests against a real
``PostgresQueue`` - the behaviours that depend on genuine Postgres semantics
(a heartbeat on a second connection, ``Cancelled`` releasing a job without
spending an attempt, ``heartbeat()`` returning ``False`` for a row another
worker now owns). They live here, not in a separate file, because issue #13
names ``tests/common/test_worker.py`` as its only test file.
"""

from __future__ import annotations

import random
import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest
import structlog
from alembic import command
from alembic.config import Config
from psycopg.rows import dict_row
from structlog.testing import capture_logs

from common.config import Settings, get_settings
from common.errors import Cancelled
from common.queue import Job, PostgresQueue
from common.worker import (
    _MAX_PLAUSIBLE_DURATION_SEC,
    EXIT_HUNG_HANDLER,
    JobContext,
    Worker,
    default_runtime_limit_sec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# test doubles
# ---------------------------------------------------------------------------


def make_job(**overrides: Any) -> Job:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    defaults: dict[str, Any] = {
        "id": 1,
        "video_id": "v1",
        "kind": "ingest",
        "dedupe_key": "default",
        "state": "running",
        "priority": 0,
        "payload": {},
        "attempts": 1,
        "last_error": None,
        "error_class": None,
        "run_after": now,
        "locked_by": "w1",
        "locked_at": now,
        "heartbeat_at": now,
        "finished_at": None,
        "created_at": now,
    }
    defaults.update(overrides)
    return Job(**defaults)


class FakeQueue:
    """A ``JobQueue`` double that mimics ``PostgresQueue.claim``'s outcome
    handling (issue #11) closely enough to test the worker loop: a plain
    exception raised inside the ``with`` block is recorded and swallowed;
    ``Cancelled``/``KeyboardInterrupt``/``SystemExit`` are recorded and
    re-raised.
    """

    def __init__(self, script: Sequence[Job | BaseException | None] = ()) -> None:
        self._script = list(script)
        self.claim_calls: list[tuple[tuple[str, ...], str]] = []
        self.outcomes: list[tuple[Job, str]] = []
        self.heartbeat_calls: list[tuple[int, str]] = []
        self.heartbeat_result: bool | Callable[[], bool] | Exception = True
        self.close_calls = 0
        self.close_raises: Exception | None = None

    @contextmanager
    def claim(
        self, kinds: Sequence[str], *, worker: str, dedupe_key: str | None = None
    ) -> Iterator[Job | None]:
        self.claim_calls.append((tuple(kinds), worker))
        if not self._script:
            yield None
            return
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if item is None:
            yield None
            return
        job = item
        try:
            yield job
        except (Cancelled, KeyboardInterrupt, SystemExit):
            self.outcomes.append((job, "pending"))
            raise
        except Exception:  # noqa: BLE001 - mirrors PostgresQueue.claim's safety net
            self.outcomes.append((job, "failed"))
        else:
            self.outcomes.append((job, "done"))

    def heartbeat(self, job_id: int, worker: str) -> bool:
        self.heartbeat_calls.append((job_id, worker))
        result = self.heartbeat_result
        if isinstance(result, Exception):
            raise result
        if callable(result):
            return result()
        return result

    def close(self) -> None:
        self.close_calls += 1
        if self.close_raises is not None:
            raise self.close_raises

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
        raise NotImplementedError("unused by worker tests")

    def reap_stale(self, older_than_sec: int | None = None) -> int:
        raise NotImplementedError("unused by worker tests")


class FakeTimerHandle:
    def __init__(self, function: Callable[[], None]) -> None:
        self.function = function
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        self.function()


class FakeTimerFactory:
    def __init__(self) -> None:
        self.scheduled: list[tuple[float, FakeTimerHandle]] = []

    def __call__(self, interval: float, function: Callable[[], None]) -> FakeTimerHandle:
        handle = FakeTimerHandle(function)
        self.scheduled.append((interval, handle))
        return handle

    @property
    def last(self) -> FakeTimerHandle:
        return self.scheduled[-1][1]


def make_worker(
    queue: FakeQueue,
    handler: Callable[[Job, JobContext], None],
    *,
    heartbeat_queue: FakeQueue | None = None,
    heartbeat_queue_factory: Callable[[], FakeQueue] | None = None,
    liveness_path: Path,
    poll_sec: float = 5,
    heartbeat_sec: float = 60,
    shutdown_grace_sec: float = 20,
    sleep: Callable[[float], None] | None = None,
    rng: random.Random | None = None,
    timer_factory: FakeTimerFactory | None = None,
    reconnect: Callable[[], FakeQueue] | None = None,
    runtime_limit_sec: Callable[[Job], float | None] = lambda _job: None,
    exit_codes: list[int] | None = None,
) -> Worker:
    if heartbeat_queue_factory is None:
        heartbeat_queue_factory = (
            (lambda: heartbeat_queue) if heartbeat_queue is not None else (lambda: queue)
        )
    return Worker(
        "w1",
        ["ingest"],
        handler,
        queue,
        poll_sec=poll_sec,
        liveness_path=liveness_path,
        heartbeat_sec=heartbeat_sec,
        shutdown_grace_sec=shutdown_grace_sec,
        heartbeat_queue_factory=heartbeat_queue_factory,
        sleep=sleep if sleep is not None else (lambda _s: None),
        rng=rng if rng is not None else random.Random(0),
        timer_factory=timer_factory if timer_factory is not None else FakeTimerFactory(),
        reconnect=reconnect if reconnect is not None else (lambda: queue),
        runtime_limit_sec=runtime_limit_sec,
        exit_process=exit_codes.append if exit_codes is not None else (lambda _code: None),
        install_signal_handlers=False,
    )


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def test_run_claims_runs_handler_and_repeats(tmp_path: Path) -> None:
    job1, job2 = make_job(id=1), make_job(id=2)
    queue = FakeQueue([job1, job2])
    seen: list[int] = []

    worker = make_worker(queue, lambda job, ctx: seen.append(job.id), liveness_path=tmp_path / "hb")
    worker.run(max_iterations=2)

    assert seen == [1, 2]
    assert queue.outcomes == [(job1, "done"), (job2, "done")]


def test_idle_wait_uses_poll_sec_with_up_to_20_percent_jitter(tmp_path: Path) -> None:
    sleeps: list[float] = []
    queue = FakeQueue([None])
    worker = make_worker(
        queue,
        lambda job, ctx: None,
        liveness_path=tmp_path / "hb",
        poll_sec=10,
        sleep=sleeps.append,
        rng=random.Random(42),
    )
    worker.run(max_iterations=1)

    assert len(sleeps) == 1
    assert 8.0 <= sleeps[0] <= 12.0


def test_each_job_runs_inside_bound_context_that_is_gone_after(tmp_path: Path) -> None:
    job = make_job(id=7, video_id="abc", kind="ingest", attempts=3)
    queue = FakeQueue([job])
    captured: dict[str, Any] = {}

    def handler(job: Job, ctx: JobContext) -> None:
        captured.update(structlog.contextvars.get_contextvars())

    worker = make_worker(queue, handler, liveness_path=tmp_path / "hb")
    worker.run(max_iterations=1)

    assert captured == {"job_id": 7, "video_id": "abc", "kind": "ingest", "attempt": 3}
    assert structlog.contextvars.get_contextvars() == {}


def test_handler_that_raises_does_not_stop_the_loop(tmp_path: Path) -> None:
    job1, job2 = make_job(id=1), make_job(id=2)
    queue = FakeQueue([job1, job2])
    seen: list[int] = []

    def handler(job: Job, ctx: JobContext) -> None:
        seen.append(job.id)
        if job.id == 1:
            raise ValueError("boom")

    worker = make_worker(queue, handler, liveness_path=tmp_path / "hb")
    worker.run(max_iterations=2)

    assert seen == [1, 2]
    assert queue.outcomes == [(job1, "failed"), (job2, "done")]


def test_database_unreachable_backs_off_up_to_60s_and_carries_on(tmp_path: Path) -> None:
    queue = FakeQueue([ConnectionError("db down"), ConnectionError("still down"), make_job(id=9)])
    sleeps: list[float] = []
    seen: list[int] = []

    worker = make_worker(
        queue,
        lambda job, ctx: seen.append(job.id),
        liveness_path=tmp_path / "hb",
        sleep=sleeps.append,
    )
    worker.run(max_iterations=3)

    assert seen == [9]
    assert sleeps == [1.0, 2.0]  # doubles each failure, from _INITIAL_DB_BACKOFF_SEC


def test_database_backoff_is_capped_at_max_db_backoff_sec(tmp_path: Path) -> None:
    queue = FakeQueue([ConnectionError("x")] * 10 + [None])
    sleeps: list[float] = []
    worker = Worker(
        "w1",
        ["ingest"],
        lambda job, ctx: None,
        queue,
        liveness_path=tmp_path / "hb",
        heartbeat_sec=60,
        shutdown_grace_sec=20,
        sleep=sleeps.append,
        max_db_backoff_sec=8,
        reconnect=lambda: queue,
        install_signal_handlers=False,
        heartbeat_queue_factory=lambda: queue,
    )
    worker.run(max_iterations=10)

    assert max(sleeps) <= 8


def test_database_outage_reconnects_by_default_and_carries_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no ``reconnect=`` given, the worker opens a fresh connection itself
    (as it already does for the heartbeat) instead of reusing the dead one."""
    dead = FakeQueue([ConnectionError("db down"), ConnectionError("still down")])
    fresh = FakeQueue([make_job(id=9)])
    fake_conn = object()
    opened: list[object] = []

    def fake_connect() -> object:
        opened.append(fake_conn)
        return fake_conn

    def fake_postgres_queue(conn: object) -> FakeQueue:
        assert conn is fake_conn
        return fresh

    monkeypatch.setattr("common.db.connect", fake_connect)
    monkeypatch.setattr("common.queue.PostgresQueue", fake_postgres_queue)
    seen: list[int] = []

    worker = Worker(
        "w1",
        ["ingest"],
        lambda job, ctx: seen.append(job.id),
        dead,
        liveness_path=tmp_path / "hb",
        heartbeat_sec=60,
        shutdown_grace_sec=20,
        heartbeat_queue_factory=lambda: fresh,
        sleep=lambda _s: None,
        timer_factory=FakeTimerFactory(),
        install_signal_handlers=False,
    )
    worker.run(max_iterations=3)

    assert seen == [9]
    assert len(dead.claim_calls) == 1  # the first claim failed; the dead queue is never reused
    assert opened  # a new connection was opened by default


def test_failed_reconnect_is_retried_on_the_next_iteration(tmp_path: Path) -> None:
    dead = FakeQueue([ConnectionError("db down"), ConnectionError("still down")])
    fresh = FakeQueue([make_job(id=3)])
    reconnect_calls = 0

    def reconnect() -> FakeQueue:
        nonlocal reconnect_calls
        reconnect_calls += 1
        if reconnect_calls == 1:
            raise ConnectionError("still refusing connections")
        return fresh

    seen: list[int] = []
    worker = make_worker(dead, lambda job, ctx: seen.append(job.id), liveness_path=tmp_path / "hb", reconnect=reconnect)
    worker.run(max_iterations=3)

    assert reconnect_calls == 2
    assert seen == [3]


def test_liveness_file_touched_on_every_loop_iteration(tmp_path: Path) -> None:
    liveness = tmp_path / "hb"
    queue = FakeQueue([None, None])
    worker = make_worker(queue, lambda job, ctx: None, liveness_path=liveness)
    worker.run(max_iterations=2)

    assert liveness.exists()


# ---------------------------------------------------------------------------
# Heartbeats
# ---------------------------------------------------------------------------


def test_heartbeat_ticks_on_a_separate_queue_while_handler_runs(tmp_path: Path) -> None:
    job = make_job(id=5)
    queue = FakeQueue([job])
    hb_queue = FakeQueue()
    timer_factory = FakeTimerFactory()

    def handler(job: Job, ctx: JobContext) -> None:
        for _ in range(2):
            scheduled_before = len(timer_factory.scheduled)
            timer_factory.last.fire()
            # A successful tick arms exactly one next tick (``last`` is the
            # fired handle until it does, so re-firing alone proves nothing).
            assert len(timer_factory.scheduled) == scheduled_before + 1
            assert timer_factory.scheduled[-1][0] == 60

    worker = make_worker(
        queue,
        handler,
        heartbeat_queue=hb_queue,
        liveness_path=tmp_path / "hb",
        timer_factory=timer_factory,
    )
    worker.run(max_iterations=1)

    assert hb_queue.heartbeat_calls == [(5, "w1"), (5, "w1")]
    assert queue.heartbeat_calls == []  # never on the claim connection


def test_heartbeat_scheduled_at_heartbeat_sec_interval(tmp_path: Path) -> None:
    queue = FakeQueue([make_job()])
    timer_factory = FakeTimerFactory()

    def handler(job: Job, ctx: JobContext) -> None:
        pass

    worker = make_worker(
        queue, handler, liveness_path=tmp_path / "hb", heartbeat_sec=42, timer_factory=timer_factory
    )
    worker.run(max_iterations=1)

    assert timer_factory.scheduled[0][0] == 42


def test_heartbeat_stops_within_one_tick_after_handler_returns(tmp_path: Path) -> None:
    job = make_job()
    queue = FakeQueue([job])
    timer_factory = FakeTimerFactory()

    worker = make_worker(queue, lambda job, ctx: None, liveness_path=tmp_path / "hb", timer_factory=timer_factory)
    worker.run(max_iterations=1)

    assert timer_factory.last.cancelled is True
    # Firing the (already-cancelled) handle after the fact must not send a
    # heartbeat: no heartbeat is sent after the job's outcome is written.
    calls_before = len(queue.heartbeat_calls)
    timer_factory.last.fire()
    assert len(queue.heartbeat_calls) == calls_before


def test_heartbeat_returning_false_cancels_the_job_and_stops_ticking(tmp_path: Path) -> None:
    job = make_job(id=3)
    queue = FakeQueue([job])
    timer_factory = FakeTimerFactory()
    seen_cancelled: list[bool] = []

    def handler(job: Job, ctx: JobContext) -> None:
        queue.heartbeat_result = False
        timer_factory.last.fire()
        seen_cancelled.append(ctx.cancelled)

    worker = make_worker(queue, handler, liveness_path=tmp_path / "hb", timer_factory=timer_factory)
    worker.run(max_iterations=1)

    assert seen_cancelled == [True]
    # Reaped: heartbeat stopped rescheduling itself (only the manual fire happened).
    assert len(queue.heartbeat_calls) == 1


def test_heartbeat_failure_is_logged_and_retried_next_tick(tmp_path: Path) -> None:
    job = make_job(id=4)
    queue = FakeQueue([job])
    timer_factory = FakeTimerFactory()
    tick_count = 0

    def handler(job: Job, ctx: JobContext) -> None:
        nonlocal tick_count
        queue.heartbeat_result = ConnectionError("blip")
        scheduled_before = len(timer_factory.scheduled)
        timer_factory.last.fire()  # tick 1: fails
        assert len(timer_factory.scheduled) == scheduled_before + 1
        assert timer_factory.scheduled[-1][0] == 60
        tick_count += 1
        queue.heartbeat_result = True
        timer_factory.last.fire()  # tick 2: succeeds, rescheduled by tick 1
        tick_count += 1
        assert not ctx.cancelled

    worker = make_worker(queue, handler, liveness_path=tmp_path / "hb", timer_factory=timer_factory)
    worker.run(max_iterations=1)

    assert tick_count == 2
    assert len(queue.heartbeat_calls) == 2


def test_heartbeat_failure_discards_the_queue_and_reconnects_via_the_factory(tmp_path: Path) -> None:
    """Issue #106: a raising ``heartbeat()`` call must not be retried forever
    on the same (now broken) connection. The next tick gets a fresh queue
    from ``heartbeat_queue_factory`` and the old one is closed."""
    job = make_job(id=20)
    queue = FakeQueue([job])
    queue_a = FakeQueue()
    queue_b = FakeQueue()
    queues = [queue_a, queue_b]
    factory_calls: list[int] = []

    def factory() -> FakeQueue:
        factory_calls.append(1)
        return queues[len(factory_calls) - 1]

    timer_factory = FakeTimerFactory()

    def handler(job: Job, ctx: JobContext) -> None:
        queue_a.heartbeat_result = ConnectionError("dead")
        scheduled_before = len(timer_factory.scheduled)
        timer_factory.last.fire()  # tick 1: on A, fails
        assert len(timer_factory.scheduled) == scheduled_before + 1
        queue_b.heartbeat_result = True
        timer_factory.last.fire()  # tick 2: gets B fresh from the factory, succeeds
        assert not ctx.cancelled

    worker = make_worker(
        queue,
        handler,
        heartbeat_queue_factory=factory,
        liveness_path=tmp_path / "hb",
        timer_factory=timer_factory,
    )
    worker.run(max_iterations=1)

    assert len(factory_calls) == 2
    assert queue_a.heartbeat_calls == [(20, "w1")]
    assert queue_b.heartbeat_calls == [(20, "w1")]
    assert queue_a.close_calls == 1


def test_heartbeat_reconnect_failure_is_logged_and_next_tick_retries_the_factory(
    tmp_path: Path,
) -> None:
    """Issue #106: if the factory itself raises while reconnecting (the
    database is still down), that tick logs a distinct event, still touches
    liveness and still schedules the next tick - which tries the factory
    again."""
    liveness = tmp_path / "hb"
    job = make_job(id=21)
    queue = FakeQueue([job])
    queue_a = FakeQueue()
    queue_c = FakeQueue()
    factory_calls = 0

    def factory() -> FakeQueue:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            return queue_a
        if factory_calls == 2:
            raise ConnectionError("still down")
        return queue_c

    timer_factory = FakeTimerFactory()

    def handler(job: Job, ctx: JobContext) -> None:
        queue_a.heartbeat_result = ConnectionError("dead")
        timer_factory.last.fire()  # tick 1: on A (factory call 1), fails

        liveness.unlink()
        scheduled_before = len(timer_factory.scheduled)
        with capture_logs() as logs:
            timer_factory.last.fire()  # tick 2: factory call 2 raises
        assert liveness.exists()
        # The failed tick must arm exactly one next tick; re-firing ``last``
        # alone cannot show this, as it still points at the fired handle.
        assert len(timer_factory.scheduled) == scheduled_before + 1
        assert timer_factory.scheduled[-1][0] == 60
        reconnect_failed = [log for log in logs if log["event"] == "worker.heartbeat_reconnect_failed"]
        assert len(reconnect_failed) == 1
        assert reconnect_failed[0]["worker"] == "w1"
        assert reconnect_failed[0]["job_id"] == 21
        assert "error" in reconnect_failed[0]

        queue_c.heartbeat_result = True
        timer_factory.last.fire()  # tick 3: factory call 3 succeeds, uses C
        assert not ctx.cancelled

    worker = make_worker(
        queue,
        handler,
        heartbeat_queue_factory=factory,
        liveness_path=liveness,
        timer_factory=timer_factory,
    )
    worker.run(max_iterations=1)

    assert factory_calls == 3
    assert queue_a.heartbeat_calls == [(21, "w1")]
    assert queue_c.heartbeat_calls == [(21, "w1")]


def test_heartbeat_reaped_does_not_call_the_factory_again(tmp_path: Path) -> None:
    """Issue #106: 'reaped' (``heartbeat()`` returns ``False``) is unchanged
    from #13 - the queue is not discarded and the factory is not consulted
    again."""
    job = make_job(id=22)
    queue = FakeQueue([job])
    factory_calls = 0

    def factory() -> FakeQueue:
        nonlocal factory_calls
        factory_calls += 1
        return queue

    timer_factory = FakeTimerFactory()

    def handler(job: Job, ctx: JobContext) -> None:
        queue.heartbeat_result = False
        timer_factory.last.fire()
        assert ctx.cancelled

    worker = make_worker(
        queue,
        handler,
        heartbeat_queue_factory=factory,
        liveness_path=tmp_path / "hb",
        timer_factory=timer_factory,
    )
    worker.run(max_iterations=1)

    assert factory_calls == 1
    assert queue.heartbeat_calls == [(22, "w1")]
    assert queue.close_calls == 0  # reaped is not a connection failure


def test_heartbeat_queue_close_raising_still_discards_it_and_reconnects(tmp_path: Path) -> None:
    """Issue #106: ``close()`` on the broken queue is best-effort - if it
    raises, the queue is still thrown away and the next tick still gets a
    fresh one."""
    job = make_job(id=23)
    queue = FakeQueue([job])
    queue_a = FakeQueue()
    queue_a.close_raises = RuntimeError("already dead")
    queue_b = FakeQueue()
    queues = [queue_a, queue_b]
    factory_calls: list[int] = []

    def factory() -> FakeQueue:
        factory_calls.append(1)
        return queues[len(factory_calls) - 1]

    timer_factory = FakeTimerFactory()

    def handler(job: Job, ctx: JobContext) -> None:
        queue_a.heartbeat_result = ConnectionError("dead")
        timer_factory.last.fire()  # tick 1: fails, close() raises too
        queue_b.heartbeat_result = True
        timer_factory.last.fire()  # tick 2: still reconnects fine
        assert not ctx.cancelled

    worker = make_worker(
        queue,
        handler,
        heartbeat_queue_factory=factory,
        liveness_path=tmp_path / "hb",
        timer_factory=timer_factory,
    )
    worker.run(max_iterations=1)

    assert len(factory_calls) == 2
    assert queue_a.close_calls == 1
    assert queue_b.heartbeat_calls == [(23, "w1")]


def test_successful_heartbeat_reuses_the_same_queue_across_jobs_without_extra_factory_calls(
    tmp_path: Path,
) -> None:
    """Issue #106: when nothing has failed, the heartbeat queue is created
    once per process and reused across jobs - no extra factory call at the
    start of job B."""
    job_a = make_job(id=30)
    job_b = make_job(id=31)
    queue = FakeQueue([job_a, job_b])
    hb_queue = FakeQueue()
    factory_calls = 0

    def factory() -> FakeQueue:
        nonlocal factory_calls
        factory_calls += 1
        return hb_queue

    timer_factory = FakeTimerFactory()

    def handler(job: Job, ctx: JobContext) -> None:
        timer_factory.last.fire()

    worker = make_worker(
        queue,
        handler,
        heartbeat_queue_factory=factory,
        liveness_path=tmp_path / "hb",
        timer_factory=timer_factory,
    )
    worker.run(max_iterations=2)

    assert factory_calls == 1
    assert hb_queue.heartbeat_calls == [(30, "w1"), (31, "w1")]


def test_queue_recreated_during_job_a_carries_over_to_job_b_without_an_extra_factory_call(
    tmp_path: Path,
) -> None:
    """Issue #106: a queue re-created mid-job-A is the one job B uses."""
    job_a = make_job(id=32)
    job_b = make_job(id=33)
    queue = FakeQueue([job_a, job_b])
    queue_a = FakeQueue()
    queue_b = FakeQueue()
    queues = [queue_a, queue_b]
    factory_calls: list[int] = []

    def factory() -> FakeQueue:
        factory_calls.append(1)
        return queues[len(factory_calls) - 1]

    timer_factory = FakeTimerFactory()
    job_b_heartbeat_queue: list[FakeQueue] = []

    def handler(job: Job, ctx: JobContext) -> None:
        if job.id == 32:
            queue_a.heartbeat_result = ConnectionError("dead")
            timer_factory.last.fire()  # fails, discards A
            queue_b.heartbeat_result = True
            timer_factory.last.fire()  # reconnects to B (factory call 2)
        else:
            timer_factory.last.fire()  # job B's tick: must reuse B, no factory call
            job_b_heartbeat_queue.append(queue_b)

    worker = make_worker(
        queue,
        handler,
        heartbeat_queue_factory=factory,
        liveness_path=tmp_path / "hb",
        timer_factory=timer_factory,
    )
    worker.run(max_iterations=2)

    assert len(factory_calls) == 2
    assert queue_b.heartbeat_calls == [(32, "w1"), (33, "w1")]


def test_stop_after_a_failed_tick_leaves_no_queue_and_next_job_reconnects_on_first_tick(
    tmp_path: Path,
) -> None:
    """Issue #106: if a job ends (``stop()``) after a failed tick and before
    the next one, the next job's first tick creates the fresh queue - and no
    heartbeat/factory call happens after ``stop()`` returns."""
    job_a = make_job(id=40)
    job_b = make_job(id=41)
    queue = FakeQueue([job_a, job_b])
    queue_a = FakeQueue()
    queue_b = FakeQueue()
    queues = [queue_a, queue_b]
    factory_calls: list[int] = []

    def factory() -> FakeQueue:
        factory_calls.append(1)
        return queues[len(factory_calls) - 1]

    timer_factory = FakeTimerFactory()

    def handler(job: Job, ctx: JobContext) -> None:
        if job.id == 40:
            queue_a.heartbeat_result = ConnectionError("dead")
            timer_factory.last.fire()  # tick fails, queue discarded; job ends here
        else:
            queue_b.heartbeat_result = True
            timer_factory.last.fire()  # job B's first tick: reconnects via factory

    worker = make_worker(
        queue,
        handler,
        heartbeat_queue_factory=factory,
        liveness_path=tmp_path / "hb",
        timer_factory=timer_factory,
    )
    worker.run(max_iterations=2)

    # Job A's only tick failed and discarded its queue; job B's first tick is
    # the one and only reconnect - not a leftover call from job A, and
    # nothing after that reconnect calls heartbeat again for job A.
    assert len(factory_calls) == 2
    assert queue_a.heartbeat_calls == [(40, "w1")]
    assert queue_b.heartbeat_calls == [(41, "w1")]


def test_a_job_that_starts_while_the_heartbeat_database_is_down_after_a_failed_tick_still_runs(
    tmp_path: Path,
) -> None:
    """Issue #106 (with #32's eager creation kept): once a tick has discarded
    the queue, starting the next job must not call the factory eagerly - a
    still-down database would otherwise raise out of job start. The first tick
    of that job does the reconnect and logs ``worker.heartbeat_reconnect_failed``."""
    job_a = make_job(id=60)
    job_b = make_job(id=61)
    queue = FakeQueue([job_a, job_b])
    queue_a = FakeQueue()
    queue_b = FakeQueue()
    factory_calls = 0

    def factory() -> FakeQueue:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            return queue_a
        if factory_calls == 2:
            raise ConnectionError("still down")
        return queue_b

    timer_factory = FakeTimerFactory()
    ran: list[int] = []
    logs_b: list[Any] = []

    def handler(job: Job, ctx: JobContext) -> None:
        ran.append(job.id)
        if job.id == 60:
            queue_a.heartbeat_result = ConnectionError("dead")
            timer_factory.last.fire()  # fails, discards A
        else:
            assert factory_calls == 1  # job start did not call the factory
            with capture_logs() as logs:
                timer_factory.last.fire()  # factory call 2 raises
            logs_b.extend(logs)
            timer_factory.last.fire()  # factory call 3 succeeds

    worker = make_worker(
        queue,
        handler,
        heartbeat_queue_factory=factory,
        liveness_path=tmp_path / "hb",
        timer_factory=timer_factory,
    )
    worker.run(max_iterations=2)

    assert ran == [60, 61]
    assert [log["event"] for log in logs_b] == ["worker.heartbeat_reconnect_failed"]
    assert factory_calls == 3
    assert queue_b.heartbeat_calls == [(61, "w1")]


def test_claim_reconnect_never_touches_the_heartbeat_queue(tmp_path: Path) -> None:
    """Issue #106: the claim loop's reconnect and the heartbeat reconnect
    stay on separate connections in both directions."""
    job = make_job(id=50)
    queue = FakeQueue([ConnectionError("claim db down")])
    hb_queue = FakeQueue()
    fresh_claim_queue = FakeQueue([job])
    hb_factory_calls = 0

    def hb_factory() -> FakeQueue:
        nonlocal hb_factory_calls
        hb_factory_calls += 1
        return hb_queue

    timer_factory = FakeTimerFactory()

    def handler(job: Job, ctx: JobContext) -> None:
        timer_factory.last.fire()

    worker = make_worker(
        queue,
        handler,
        heartbeat_queue_factory=hb_factory,
        liveness_path=tmp_path / "hb",
        timer_factory=timer_factory,
        reconnect=lambda: fresh_claim_queue,
        sleep=lambda _s: None,
    )
    worker.run(max_iterations=2)

    assert hb_factory_calls == 1
    assert hb_queue.heartbeat_calls == [(50, "w1")]


def test_liveness_file_touched_on_heartbeat_tick(tmp_path: Path) -> None:
    liveness = tmp_path / "hb"
    job = make_job()
    queue = FakeQueue([job])
    timer_factory = FakeTimerFactory()
    exists_after_tick: list[bool] = []

    def handler(job: Job, ctx: JobContext) -> None:
        liveness.unlink()  # prove the *tick*, not job start, recreates it
        timer_factory.last.fire()
        exists_after_tick.append(liveness.exists())

    worker = make_worker(queue, handler, liveness_path=liveness, timer_factory=timer_factory)
    worker.run(max_iterations=1)

    assert exists_after_tick == [True]


def test_liveness_file_touched_even_when_the_heartbeat_call_raises(tmp_path: Path) -> None:
    """A run of failed ticks (DB outage during a long handler) must not leave
    the liveness file stale while the process and handler are alive."""
    liveness = tmp_path / "hb"
    queue = FakeQueue([make_job()])
    timer_factory = FakeTimerFactory()
    exists_after_tick: list[bool] = []

    def handler(job: Job, ctx: JobContext) -> None:
        liveness.unlink()
        queue.heartbeat_result = ConnectionError("db down")
        timer_factory.last.fire()
        exists_after_tick.append(liveness.exists())

    worker = make_worker(queue, handler, liveness_path=liveness, timer_factory=timer_factory)
    worker.run(max_iterations=1)

    assert exists_after_tick == [True]


def test_liveness_file_touched_when_the_heartbeat_reports_the_job_reaped(tmp_path: Path) -> None:
    liveness = tmp_path / "hb"
    queue = FakeQueue([make_job()])
    timer_factory = FakeTimerFactory()
    exists_after_tick: list[bool] = []

    def handler(job: Job, ctx: JobContext) -> None:
        liveness.unlink()
        queue.heartbeat_result = False
        timer_factory.last.fire()
        exists_after_tick.append(liveness.exists())

    worker = make_worker(queue, handler, liveness_path=liveness, timer_factory=timer_factory)
    worker.run(max_iterations=1)

    assert exists_after_tick == [True]


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def test_request_shutdown_while_idle_stops_the_loop_without_a_new_claim(tmp_path: Path) -> None:
    queue = FakeQueue([make_job()])  # would be claimable if the loop ran again
    worker = make_worker(queue, lambda job, ctx: None, liveness_path=tmp_path / "hb")

    worker.request_shutdown()
    worker.run()  # no max_iterations: must return on its own

    assert queue.claim_calls == []


def test_shutdown_signaled_during_idle_wait_stops_within_one_poll(tmp_path: Path) -> None:
    queue = FakeQueue([None])
    worker = make_worker(queue, lambda job, ctx: None, liveness_path=tmp_path / "hb", poll_sec=5)
    # Simulate the signal arriving while the idle wait is in progress.
    worker._sleep = lambda _s: worker.request_shutdown()

    worker.run()  # unbounded; must still return promptly

    assert queue.claim_calls == [(("ingest",), "w1")]


def test_shutdown_while_handler_runs_arms_grace_and_finishing_in_time_exits_clean(tmp_path: Path) -> None:
    job = make_job(id=11)
    queue = FakeQueue([job])
    timer_factory = FakeTimerFactory()

    def handler(job: Job, ctx: JobContext) -> None:
        worker.request_shutdown()  # signal arrives mid-handler
        # ... handler finishes its work before the grace period expires ...

    worker = make_worker(queue, handler, liveness_path=tmp_path / "hb", timer_factory=timer_factory, shutdown_grace_sec=20)
    worker.run()

    assert queue.outcomes == [(job, "done")]
    intervals = [interval for interval, _handle in timer_factory.scheduled]
    assert 20 in intervals  # grace timer armed
    assert all(handle.cancelled for _interval, handle in timer_factory.scheduled)


def test_grace_expiry_cancels_and_job_is_released_to_pending_without_spending_an_attempt(
    tmp_path: Path,
) -> None:
    job = make_job(id=12)
    queue = FakeQueue([job])
    timer_factory = FakeTimerFactory()

    def handler(job: Job, ctx: JobContext) -> None:
        worker.request_shutdown()
        timer_factory.last.fire()  # grace period expires
        ctx.check_cancelled()  # cooperative handler: raises Cancelled

    worker = make_worker(queue, handler, liveness_path=tmp_path / "hb", timer_factory=timer_factory)
    worker.run()  # must return normally (process exit 0), not raise

    assert queue.outcomes == [(job, "pending")]


def test_handler_ignoring_cancellation_keeps_running_worker_does_not_kill_it(tmp_path: Path) -> None:
    job = make_job(id=13)
    queue = FakeQueue([job])
    timer_factory = FakeTimerFactory()
    finished: list[bool] = []

    def handler(job: Job, ctx: JobContext) -> None:
        worker.request_shutdown()
        timer_factory.last.fire()  # grace expires; ctx.cancelled is now True
        assert ctx.cancelled
        finished.append(True)  # handler ignores it and completes anyway

    worker = make_worker(queue, handler, liveness_path=tmp_path / "hb", timer_factory=timer_factory)
    worker.run()

    assert finished == [True]
    assert queue.outcomes == [(job, "done")]


def test_second_signal_during_grace_cancels_immediately(tmp_path: Path) -> None:
    job = make_job(id=14)
    queue = FakeQueue([job])
    timer_factory = FakeTimerFactory()

    def handler(job: Job, ctx: JobContext) -> None:
        worker.request_shutdown()
        assert not ctx.cancelled
        assert timer_factory.last.cancelled is False
        worker.request_shutdown()  # second signal: cancel now, don't wait out the timer
        assert ctx.cancelled
        assert timer_factory.last.cancelled is True

    worker = make_worker(queue, handler, liveness_path=tmp_path / "hb", timer_factory=timer_factory)
    worker.run()


def test_run_returns_after_max_iterations_even_without_shutdown(tmp_path: Path) -> None:
    queue = FakeQueue([None, None, None])
    worker = make_worker(queue, lambda job, ctx: None, liveness_path=tmp_path / "hb")
    worker.run(max_iterations=2)

    assert len(queue.claim_calls) == 2


# ---------------------------------------------------------------------------
# Real SIGTERM (subprocess)
# ---------------------------------------------------------------------------


def test_real_sigterm_stops_an_idle_worker(tmp_path: Path) -> None:
    liveness = tmp_path / "heartbeat"
    script = tmp_path / "run_worker.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(REPO_ROOT)!r})
            from contextlib import contextmanager
            from pathlib import Path
            from common.worker import Worker

            class IdleQueue:
                def heartbeat(self, job_id, worker):
                    return True

                @contextmanager
                def claim(self, kinds, *, worker):
                    yield None

            worker = Worker(
                "w1", ["ingest"], lambda job, ctx: None, IdleQueue(),
                poll_sec=0.2, liveness_path=Path({str(liveness)!r}),
                heartbeat_sec=60, shutdown_grace_sec=20,
                heartbeat_queue_factory=lambda: IdleQueue(),
            )
            worker._install_signals()  # avoid a race between READY and run()'s own install
            print("READY", flush=True)
            worker.run()
            print("DONE", flush=True)
            """
        )
    )

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert line.strip() == "READY"

        proc.terminate()  # SIGTERM
        returncode = proc.wait(timeout=5)
        remaining_output = proc.stdout.read()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    assert returncode == 0, remaining_output
    assert "DONE" in remaining_output


# ---------------------------------------------------------------------------
# Against a real PostgresQueue (integration)
# ---------------------------------------------------------------------------


@pytest.fixture
def head_dsn(postgres_dsn: str, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("DATABASE_URL", postgres_dsn)
    get_settings.cache_clear()
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    command.upgrade(config, "head")
    return postgres_dsn


@pytest.fixture
def claim_conn(head_dsn: str) -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(head_dsn, autocommit=True) as connection:
        yield connection


@pytest.fixture
def heartbeat_conn(head_dsn: str) -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(head_dsn, autocommit=True) as connection:
        yield connection


def _pg_settings() -> Settings:
    return Settings(DATABASE_URL="postgresql://u:p@h/db")  # type: ignore[arg-type]


@pytest.fixture
def claim_queue(claim_conn: psycopg.Connection[Any]) -> PostgresQueue:
    return PostgresQueue(claim_conn, settings=_pg_settings())


@pytest.fixture
def heartbeat_queue(heartbeat_conn: psycopg.Connection[Any]) -> PostgresQueue:
    return PostgresQueue(heartbeat_conn, settings=_pg_settings())


def _row(conn: psycopg.Connection[Any], job_id: int) -> dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
        row = cur.fetchone()
    assert row is not None
    return row


def make_pg_worker(
    claim_queue: PostgresQueue,
    heartbeat_queue: PostgresQueue,
    handler: Callable[[Job, JobContext], None],
    *,
    liveness_path: Path,
    timer_factory: FakeTimerFactory,
    runtime_limit_sec: Callable[[Job], float | None] = lambda _job: None,
    exit_codes: list[int] | None = None,
) -> Worker:
    return Worker(
        "w1",
        ["ingest"],
        handler,
        claim_queue,
        liveness_path=liveness_path,
        heartbeat_sec=60,
        shutdown_grace_sec=20,
        heartbeat_queue_factory=lambda: heartbeat_queue,
        timer_factory=timer_factory,
        sleep=lambda _s: None,
        runtime_limit_sec=runtime_limit_sec,
        exit_process=exit_codes.append if exit_codes is not None else (lambda _code: None),
        install_signal_handlers=False,
    )


@pytest.mark.integration
def test_heartbeat_updates_heartbeat_at_via_a_connection_distinct_from_claim(
    claim_queue: PostgresQueue,
    heartbeat_queue: PostgresQueue,
    claim_conn: psycopg.Connection[Any],
    tmp_path: Path,
) -> None:
    job_id = claim_queue.enqueue("ingest", "v1")
    assert job_id is not None
    timer_factory = FakeTimerFactory()
    at_claim: dict[str, Any] = {}

    def handler(job: Job, ctx: JobContext) -> None:
        at_claim["heartbeat_at"] = _row(claim_conn, job_id)["heartbeat_at"]
        timer_factory.last.fire()

    worker = make_pg_worker(
        claim_queue, heartbeat_queue, handler, liveness_path=tmp_path / "hb", timer_factory=timer_factory
    )
    worker.run(max_iterations=1)

    # The claim connection is autocommit, so a fresh SELECT sees whatever the
    # heartbeat connection committed, even though claim_queue itself never
    # issued the UPDATE.
    after = _row(claim_conn, job_id)["heartbeat_at"]
    assert at_claim["heartbeat_at"] is not None
    assert after is not None
    assert after >= at_claim["heartbeat_at"]


@pytest.mark.integration
def test_grace_expiry_releases_the_job_to_pending_without_spending_an_attempt(
    claim_queue: PostgresQueue,
    heartbeat_queue: PostgresQueue,
    claim_conn: psycopg.Connection[Any],
    tmp_path: Path,
) -> None:
    job_id = claim_queue.enqueue("ingest", "v1")
    assert job_id is not None
    timer_factory = FakeTimerFactory()

    def handler(job: Job, ctx: JobContext) -> None:
        worker.request_shutdown()
        timer_factory.last.fire()  # grace period expires
        ctx.check_cancelled()  # raises Cancelled

    worker = make_pg_worker(
        claim_queue, heartbeat_queue, handler, liveness_path=tmp_path / "hb", timer_factory=timer_factory
    )
    attempts_before_claim = _row(claim_conn, job_id)["attempts"]
    worker.run()  # returns normally: process exit 0

    row = _row(claim_conn, job_id)
    assert row["state"] == "pending"
    # claim() itself increments attempts (architecture.md §5); Cancelled must
    # undo exactly that, leaving it where it started rather than "spent".
    assert row["attempts"] == attempts_before_claim
    assert row["locked_by"] is None


@pytest.mark.integration
def test_heartbeat_returns_false_once_the_job_is_genuinely_reaped(
    claim_queue: PostgresQueue,
    heartbeat_queue: PostgresQueue,
    claim_conn: psycopg.Connection[Any],
    tmp_path: Path,
) -> None:
    job_id = claim_queue.enqueue("ingest", "v1")
    assert job_id is not None
    timer_factory = FakeTimerFactory()
    cancelled_seen: list[bool] = []

    def handler(job: Job, ctx: JobContext) -> None:
        # Simulate the reaper (architecture.md §5) having handed this row to
        # another worker while this one's heartbeat connection stalled.
        with claim_conn.cursor() as cur:
            cur.execute("UPDATE jobs SET locked_by = 'other-worker' WHERE id = %s", (job_id,))

        timer_factory.last.fire()  # this worker's heartbeat tick, now stale
        cancelled_seen.append(ctx.cancelled)

    worker = make_pg_worker(
        claim_queue, heartbeat_queue, handler, liveness_path=tmp_path / "hb", timer_factory=timer_factory
    )
    worker.run(max_iterations=1)

    assert cancelled_seen == [True]


# ---------------------------------------------------------------------------
# Runtime limit (issue #66). The limit is a one-shot timer through the injected
# timer_factory, so a test "waits" for it by firing the handle by hand.
# ---------------------------------------------------------------------------


class FailRecordingQueue(FakeQueue):
    def __init__(self, script: Sequence[Job | BaseException | None] = ()) -> None:
        super().__init__(script)
        self.failed: list[tuple[Job, str, BaseException]] = []
        self.fail_error: Exception | None = None

    def fail_job(self, job: Job, worker: str, exc: BaseException) -> None:
        if self.fail_error is not None:
            raise self.fail_error
        self.failed.append((job, worker, exc))


def _limit_timer(factory: FakeTimerFactory, limit: float) -> FakeTimerHandle:
    matches = [h for interval, h in factory.scheduled if interval == limit]
    assert len(matches) == 1, f"expected one timer at {limit}s, got {factory.scheduled}"
    return matches[0]


def test_a_job_past_its_limit_stops_heartbeating_records_a_failed_attempt_and_exits_nonzero(
    tmp_path: Path,
) -> None:
    job = make_job(id=9, video_id="vid", kind="ingest", attempts=2)
    queue = FailRecordingQueue([job])
    timers = FakeTimerFactory()
    codes: list[int] = []

    def handler(job: Job, ctx: JobContext) -> None:
        _limit_timer(timers, 777).fire()

    worker = make_worker(
        queue,
        handler,
        liveness_path=tmp_path / "hb",
        timer_factory=timers,
        runtime_limit_sec=lambda _job: 777,
        exit_codes=codes,
    )
    with structlog.testing.capture_logs() as logs:
        worker.run(max_iterations=1)

    assert codes == [EXIT_HUNG_HANDLER]
    assert EXIT_HUNG_HANDLER > 0
    assert len(queue.failed) == 1
    failed_job, failed_by, exc = queue.failed[0]
    assert (failed_job.id, failed_by) == (9, "w1")
    assert "exceeded its runtime limit of 777" in str(exc)
    error_logs = [e for e in logs if e["log_level"] == "error" and e["event"] == "worker.runtime_limit_exceeded"]
    assert len(error_logs) == 1
    assert error_logs[0]["job_id"] == 9
    assert error_logs[0]["video_id"] == "vid"
    assert error_logs[0]["kind"] == "ingest"
    assert error_logs[0]["attempt"] == 2


def test_the_heartbeat_is_stopped_before_the_limit_is_acted_on(tmp_path: Path) -> None:
    queue = FailRecordingQueue([make_job()])
    timers = FakeTimerFactory()

    def handler(job: Job, ctx: JobContext) -> None:
        heartbeat_timer = timers.scheduled[0][1]
        _limit_timer(timers, 500).fire()
        assert heartbeat_timer.cancelled is True
        # A tick that was already queued must not send a heartbeat any more.
        heartbeat_timer.fire()

    worker = make_worker(
        queue, handler, liveness_path=tmp_path / "hb", timer_factory=timers, runtime_limit_sec=lambda _j: 500
    )
    worker.run(max_iterations=1)

    assert queue.heartbeat_calls == []


def test_a_job_that_finishes_in_time_disarms_the_limit(tmp_path: Path) -> None:
    queue = FailRecordingQueue([make_job()])
    timers = FakeTimerFactory()
    codes: list[int] = []

    worker = make_worker(
        queue,
        lambda job, ctx: None,
        liveness_path=tmp_path / "hb",
        timer_factory=timers,
        runtime_limit_sec=lambda _j: 500,
        exit_codes=codes,
    )
    worker.run(max_iterations=1)

    handle = _limit_timer(timers, 500)
    assert handle.cancelled is True
    # Even if the timer thread was already past its cancel check, the finished
    # job must not be reported as hung.
    handle.fire()
    assert codes == []
    assert queue.failed == []


def test_no_limit_means_no_timer_and_no_exit(tmp_path: Path) -> None:
    queue = FailRecordingQueue([make_job()])
    timers = FakeTimerFactory()

    worker = make_worker(
        queue, lambda job, ctx: None, liveness_path=tmp_path / "hb", timer_factory=timers
    )
    worker.run(max_iterations=1)

    assert [interval for interval, _ in timers.scheduled] == [60]  # the heartbeat only


def test_the_worker_still_exits_when_recording_the_failure_raises(tmp_path: Path) -> None:
    queue = FailRecordingQueue([make_job()])
    queue.fail_error = RuntimeError("database gone")
    timers = FakeTimerFactory()
    codes: list[int] = []

    def handler(job: Job, ctx: JobContext) -> None:
        _limit_timer(timers, 500).fire()

    worker = make_worker(
        queue,
        handler,
        liveness_path=tmp_path / "hb",
        timer_factory=timers,
        runtime_limit_sec=lambda _j: 500,
        exit_codes=codes,
    )
    with structlog.testing.capture_logs() as logs:
        worker.run(max_iterations=1)

    assert codes == [EXIT_HUNG_HANDLER]
    assert any(e["event"] == "worker.runtime_limit_record_failed" for e in logs)


def test_a_queue_without_fail_job_still_exits(tmp_path: Path) -> None:
    queue = FakeQueue([make_job()])
    timers = FakeTimerFactory()
    codes: list[int] = []

    def handler(job: Job, ctx: JobContext) -> None:
        _limit_timer(timers, 500).fire()

    worker = make_worker(
        queue,
        handler,
        liveness_path=tmp_path / "hb",
        timer_factory=timers,
        runtime_limit_sec=lambda _j: 500,
        exit_codes=codes,
    )
    worker.run(max_iterations=1)

    assert codes == [EXIT_HUNG_HANDLER]


class _InstrumentedLock:
    """A real lock that reports when an acquire has to wait for another thread."""

    def __init__(self, on_contended: Callable[[], None]) -> None:
        self._lock = threading.Lock()
        self._on_contended = on_contended

    def acquire(self) -> bool:
        if not self._lock.acquire(blocking=False):
            self._on_contended()
            self._lock.acquire()
        return True

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> None:
        self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.release()


class _ThreadingWithInstrumentedLock:
    """Stands in for ``threading`` inside ``common.worker`` only."""

    def __init__(self, on_contended: Callable[[], None]) -> None:
        self._on_contended = on_contended

    def Lock(self) -> _InstrumentedLock:  # mirrors threading.Lock
        return _InstrumentedLock(self._on_contended)

    def __getattr__(self, name: str) -> Any:
        return getattr(threading, name)


def test_the_handler_returning_waits_for_a_limit_that_is_already_being_acted_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forces the race the watchdog lock arbitrates, deterministically.

    The timer thread takes the lock, passes the "already finished?" check and
    stops inside ``fail_job``. The handler then returns on the main thread and
    calls ``finish()``. With the lock, ``finish()`` has to wait for the timer
    thread (contention), so the timer thread wins and the job is reported hung
    exactly once. Without it ``finish()`` runs straight through, flips the flag
    and cancels the timer while the timer thread is mid-report: the job is
    both "finished" and "hung".
    """
    progress = threading.Event()  # finish() contended for the lock, or ran past it
    contended = threading.Event()
    timer_thread_inside = threading.Event()

    def _on_contended() -> None:
        contended.set()
        progress.set()

    monkeypatch.setattr("common.worker.threading", _ThreadingWithInstrumentedLock(_on_contended))

    class BlockingQueue(FailRecordingQueue):
        def fail_job(self, job: Job, worker: str, exc: BaseException) -> None:
            timer_thread_inside.set()  # the lock is held and the flag was already checked
            assert progress.wait(timeout=30), "finish() never reached the lock or the cancel"
            super().fail_job(job, worker, exc)

    queue = BlockingQueue([make_job()])
    timers = FakeTimerFactory()
    codes: list[int] = []
    threads: list[threading.Thread] = []

    def handler(job: Job, ctx: JobContext) -> None:
        handle = _limit_timer(timers, 500)
        real_cancel = handle.cancel

        def cancel() -> None:
            real_cancel()
            progress.set()

        handle.cancel = cancel  # type: ignore[method-assign]
        thread = threading.Thread(target=handle.fire)
        threads.append(thread)
        thread.start()
        assert timer_thread_inside.wait(timeout=30)

    worker = make_worker(
        queue,
        handler,
        liveness_path=tmp_path / "hb",
        timer_factory=timers,
        runtime_limit_sec=lambda _j: 500,
        exit_codes=codes,
    )
    worker.run(max_iterations=1)
    threads[0].join(timeout=30)

    assert contended.is_set(), "finish() did not have to wait for the timer thread: no lock arbitrates"
    assert codes == [EXIT_HUNG_HANDLER]
    assert len(queue.failed) == 1


def test_a_worker_built_without_settings_runs_unlimited_when_no_environment_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_settings.cache_clear()
    queue = FailRecordingQueue([make_job(kind="ingest")])
    timers = FakeTimerFactory()
    codes: list[int] = []
    seen: list[int] = []

    worker = Worker(
        "w1",
        ["ingest"],
        lambda job, ctx: seen.append(job.id),
        queue,
        liveness_path=tmp_path / "hb",
        heartbeat_sec=60,
        shutdown_grace_sec=20,
        heartbeat_queue_factory=lambda: queue,
        sleep=lambda _s: None,
        timer_factory=timers,
        exit_process=codes.append,
        install_signal_handlers=False,
    )
    with structlog.testing.capture_logs() as logs:
        worker.run(max_iterations=1)

    assert len(seen) == 1
    assert [interval for interval, _ in timers.scheduled] == [60]  # no limit timer
    assert codes == []
    assert any(e["event"] == "worker.runtime_limit_settings_unavailable" for e in logs)


@pytest.mark.parametrize(
    "duration",
    [10**400, 3 * 10**9 + 1, 10**12, float("inf"), float("nan"), 1e308],
)
def test_an_absurd_transcribe_duration_gets_no_limit_instead_of_crashing_or_never_firing(
    duration: float,
) -> None:
    with structlog.testing.capture_logs() as logs:
        limit = default_runtime_limit_sec(
            _settings(), make_job(kind="transcribe", payload={"duration_sec": duration})
        )

    assert limit is None
    assert any(e["event"] == "worker.runtime_limit_duration_unusable" for e in logs)


def test_the_largest_accepted_transcribe_duration_still_fits_in_a_timer() -> None:
    limit = default_runtime_limit_sec(
        _settings(), make_job(kind="transcribe", payload={"duration_sec": _MAX_PLAUSIBLE_DURATION_SEC})
    )
    assert limit is not None
    assert limit < threading.TIMEOUT_MAX
    timer = threading.Timer(limit, lambda: None)  # must be constructible and startable
    timer.daemon = True
    timer.start()
    timer.cancel()
    timer.join()


def _settings(**overrides: Any) -> Settings:
    return Settings(DATABASE_URL="postgresql://u:p@h/db", **overrides)  # type: ignore[arg-type]


def test_default_limits_are_ingest_10_minutes_and_analyze_1_hour() -> None:
    settings = _settings()
    assert default_runtime_limit_sec(settings, make_job(kind="ingest")) == 600
    assert default_runtime_limit_sec(settings, make_job(kind="analyze")) == 3600


def test_the_per_kind_limits_are_configurable() -> None:
    settings = _settings(INGEST_MAX_RUNTIME_SEC=30, ANALYZE_MAX_RUNTIME_SEC=40)
    assert default_runtime_limit_sec(settings, make_job(kind="ingest")) == 30
    assert default_runtime_limit_sec(settings, make_job(kind="analyze")) == 40


def test_the_transcribe_limit_is_the_audio_duration_times_the_rtf_ceiling_plus_overhead() -> None:
    settings = _settings(TRANSCRIBE_MAX_RTF=2.0)
    job = make_job(kind="transcribe", payload={"duration_sec": 1000})
    limit = default_runtime_limit_sec(settings, job)
    assert limit is not None
    assert limit > 2000
    assert limit - 2000 < 3600  # a fixed allowance, not another multiple


def test_a_healthy_nine_hour_transcription_is_not_killed_at_a_realistic_speed() -> None:
    settings = _settings()  # default RTF ceiling
    nine_hours = 9 * 3600
    job = make_job(kind="transcribe", payload={"duration_sec": nine_hours})
    limit = default_runtime_limit_sec(settings, job)
    assert limit is not None
    healthy_rtf = 1.0  # as slow as real time
    assert nine_hours * healthy_rtf < limit


def test_the_transcribe_limit_scales_with_the_duration() -> None:
    settings = _settings()
    short = default_runtime_limit_sec(settings, make_job(kind="transcribe", payload={"duration_sec": 600}))
    long = default_runtime_limit_sec(settings, make_job(kind="transcribe", payload={"duration_sec": 36000}))
    assert short is not None and long is not None
    assert long > short


@pytest.mark.parametrize("payload", [{}, {"duration_sec": None}, {"duration_sec": 0}, {"duration_sec": "x"}, {"duration_sec": True}])
def test_a_transcribe_job_without_a_usable_duration_has_no_limit(payload: dict[str, Any]) -> None:
    assert default_runtime_limit_sec(_settings(), make_job(kind="transcribe", payload=payload)) is None


@pytest.mark.parametrize("name", ["INGEST_MAX_RUNTIME_SEC", "ANALYZE_MAX_RUNTIME_SEC", "TRANSCRIBE_MAX_RTF"])
@pytest.mark.parametrize("bad", ["0", "-1"])
def test_a_non_positive_limit_setting_is_rejected(name: str, bad: str) -> None:
    with pytest.raises(ValueError):
        Settings(DATABASE_URL="postgresql://u:p@h/db", **{name: bad})  # type: ignore[arg-type]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("attempts_before", "expected_state"),
    [(0, "pending"), (1, "pending"), (2, "dead")],
)
def test_a_hung_job_is_failed_with_a_runtime_limit_message_and_dead_letters_at_max(
    claim_queue: PostgresQueue,
    heartbeat_queue: PostgresQueue,
    claim_conn: psycopg.Connection[Any],
    tmp_path: Path,
    attempts_before: int,
    expected_state: str,
) -> None:
    job_id = claim_queue.enqueue("ingest", "v1")
    assert job_id is not None
    with claim_conn.cursor() as cur:
        cur.execute("UPDATE jobs SET attempts = %s WHERE id = %s", (attempts_before, job_id))
    timers = FakeTimerFactory()
    codes: list[int] = []

    def handler(job: Job, ctx: JobContext) -> None:
        _limit_timer(timers, 321).fire()

    worker = make_pg_worker(
        claim_queue,
        heartbeat_queue,
        handler,
        liveness_path=tmp_path / "hb",
        timer_factory=timers,
        runtime_limit_sec=lambda _j: 321,
        exit_codes=codes,
    )
    worker.run(max_iterations=1)

    row = _row(claim_conn, job_id)
    assert codes == [EXIT_HUNG_HANDLER]
    assert row["state"] == expected_state
    assert row["attempts"] == attempts_before + 1  # the attempt is spent, not refunded
    assert row["locked_by"] is None
    assert "exceeded its runtime limit of 321" in row["last_error"]
    assert row["error_class"] == "TOOL_FAILURE"
    # The handler returns normally after the (fake) exit, which must not
    # overwrite the recorded outcome: the claim UPDATE is guarded by locked_by.
    assert (row["finished_at"] is not None) == (expected_state == "dead")


@pytest.mark.integration
def test_a_terminated_heartbeat_connection_is_replaced_and_heartbeats_resume(
    claim_queue: PostgresQueue,
    claim_conn: psycopg.Connection[Any],
    head_dsn: str,
    tmp_path: Path,
) -> None:
    """Issue #106: kill the heartbeat backend mid-handler; the next tick fails,
    the one after reconnects on a different backend and moves ``heartbeat_at``."""
    job_id = claim_queue.enqueue("ingest", "v1")
    assert job_id is not None
    opened: list[psycopg.Connection[Any]] = []

    def factory() -> PostgresQueue:
        connection = psycopg.connect(head_dsn, autocommit=True)
        opened.append(connection)
        return PostgresQueue(connection, settings=_pg_settings())

    timer_factory = FakeTimerFactory()
    seen: dict[str, Any] = {}

    def handler(job: Job, ctx: JobContext) -> None:
        with claim_conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET heartbeat_at = now() - interval '1 hour' WHERE id = %s",
                (job_id,),
            )
        seen["stale_heartbeat_at"] = _row(claim_conn, job_id)["heartbeat_at"]
        assert len(opened) == 1  # created eagerly at job start (#32)
        dead_pid = opened[0].info.backend_pid
        with claim_conn.cursor() as cur:
            cur.execute("SELECT pg_terminate_backend(%s)", (dead_pid,))
        timer_factory.last.fire()  # tick 1: fails on the dead connection
        seen["after_failed_tick"] = _row(claim_conn, job_id)["heartbeat_at"]
        timer_factory.last.fire()  # tick 2: fresh connection
        seen["after_recovery"] = _row(claim_conn, job_id)["heartbeat_at"]
        seen["dead_pid"] = dead_pid

    try:
        worker = Worker(
            "w1",
            ["ingest"],
            handler,
            claim_queue,
            liveness_path=tmp_path / "hb",
            heartbeat_sec=60,
            shutdown_grace_sec=20,
            heartbeat_queue_factory=factory,
            timer_factory=timer_factory,
            sleep=lambda _s: None,
            runtime_limit_sec=lambda _job: None,
            install_signal_handlers=False,
        )
        worker.run(max_iterations=1)

        assert len(opened) == 2
        assert opened[1].info.backend_pid != seen["dead_pid"]
        assert seen["after_failed_tick"] == seen["stale_heartbeat_at"]
        assert seen["after_recovery"] > seen["stale_heartbeat_at"]
        assert opened[0].closed
    finally:
        for connection in opened:
            connection.close()
