"""Interval scheduler for the planner's periodic tasks (issue #38).

``Scheduler`` runs an ordered registry of ``Task(name, interval_sec, fn)``
entries on one thread. It knows nothing about channels, reaping or audio; it
only decides *when* each callable runs and wraps every run in the concerns the
callables must not think about:

- **Schedule.** Every task runs once at startup, in registry order. After
  that a task is due at ``last_start + interval``: an overrunning task runs
  again straight after it finishes, once, never several times to catch up.
  Time comes from an injected monotonic clock, so a wall-clock step (NTP,
  DST) neither skips nor bursts runs.
- **Isolation.** A task that raises is logged (task, exception class,
  traceback) and waits for its next interval; the others carry on.
  ``KeyboardInterrupt``/``SystemExit`` are not swallowed.
- **Locks.** Each run holds ``locks.try_lock(name)`` and releases it in a
  ``finally``. A held lock means another planner may be running: the task is
  skipped with a WARNING and retried at its next interval.
- **Database outage.** ``psycopg.OperationalError`` from the lock provider
  is logged and answered with a doubling backoff (capped), after which the
  provider reconnects. The loop never crash-loops the container (mirrors
  ``common.worker.Worker``, #13).
- **Run records.** An injected ``RunRecorder`` is told when a task starts,
  finishes (ok or failed) or is skipped because its lock is held (#98), with
  timestamps from an injected wall clock. A recorder that raises is logged
  (``planner.record_failed``) and otherwise ignored.
- **Liveness.** The liveness file is touched on every loop iteration and idle
  waits are cut into slices of at most ``MAX_SLEEP_SLICE_SEC``.
- **Shutdown.** SIGTERM/SIGINT call ``request_shutdown()``: a task in flight
  finishes and releases its lock, nothing else starts, ``run()`` returns.

``sleep`` and ``clock`` are injected so tests never wait for real.
"""

from __future__ import annotations

import os
import select
import signal
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Literal, Protocol

import psycopg
import structlog

_log = structlog.get_logger(__name__)

#: Longest single idle wait, so the liveness file stays fresh (issue #38 AC:
#: under 180 s old across a 3600 s gap).
MAX_SLEEP_SLICE_SEC = 30.0

_INITIAL_DB_BACKOFF_SEC = 1.0
_DEFAULT_MAX_DB_BACKOFF_SEC = 60.0

type Outcome = Literal["ok", "failed", "skipped_locked"]


@dataclass(frozen=True)
class Task:
    """One periodic job. ``fn`` may return a count (jobs reaped, enqueued ...)."""

    name: str
    interval_sec: float
    fn: Callable[[], int | None]


class LockProvider(Protocol):
    """Named, non-blocking, session-level locks (Postgres advisory locks)."""

    def try_lock(self, name: str) -> bool:
        """Take the lock for ``name`` if free. May raise ``psycopg.OperationalError``."""
        ...

    def release(self, name: str) -> None:
        """Release a lock taken by ``try_lock``."""
        ...

    def reset(self) -> None:
        """Drop the underlying connection; the next ``try_lock`` reconnects."""
        ...


#: Longest ``last_error`` text a recorder is given (issue #98).
MAX_RECORDED_ERROR_CHARS = 500


class RunRecorder(Protocol):
    """Where the last run of each task is remembered (``planner_task_runs``, #98).

    Any method may raise; the scheduler logs ``planner.record_failed`` and
    carries on, so a recorder never needs to guard itself.
    """

    def task_started(self, name: str, *, interval_sec: float, at: datetime) -> None:
        """The lock was acquired and ``fn()`` is about to run."""
        ...

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
        """``fn()`` returned (``ok``, ``count`` = its value) or raised (``failed``)."""
        ...

    def task_skipped_locked(self, name: str, *, interval_sec: float, at: datetime) -> None:
        """Another planner held the lock; nothing ran."""
        ...


class NoopRecorder:
    """The default: records nothing."""

    def task_started(self, name: str, *, interval_sec: float, at: datetime) -> None:
        pass

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
        pass

    def task_skipped_locked(self, name: str, *, interval_sec: float, at: datetime) -> None:
        pass


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _DatabaseUnavailable(Exception):
    """Internal: the lock provider could not reach the database."""


class Scheduler:
    def __init__(
        self,
        tasks: Sequence[Task],
        locks: LockProvider,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] | None = None,
        liveness_path: Path = Path("/tmp/heartbeat"),
        max_db_backoff_sec: float = _DEFAULT_MAX_DB_BACKOFF_SEC,
        install_signal_handlers: bool = True,
        recorder: RunRecorder | None = None,
        wall_clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._tasks = list(tasks)
        self._recorder: RunRecorder = recorder if recorder is not None else NoopRecorder()
        self._wall_clock = wall_clock
        self._locks = locks
        self._clock = clock
        # A plain flag, not threading.Event: a signal handler runs on the main
        # thread and Event.set() takes a lock the interrupted code may hold.
        self._shutdown_requested = False
        self._injected_sleep = sleep
        self._wake_read: int | None = None
        self._wake_write: int | None = None
        self._prev_wakeup_fd = -1
        self._liveness_path = liveness_path
        self._max_db_backoff_sec = max_db_backoff_sec
        self._install_signal_handlers_enabled = install_signal_handlers
        self._last_start: dict[str, float] = {}
        self._backoff = _INITIAL_DB_BACKOFF_SEC
        self._prev_handlers: dict[int, object] = {}

    # -- public API ---------------------------------------------------------

    def request_shutdown(self) -> None:
        """Stop starting tasks; called by the signal handler or a test.

        Only sets a flag and writes one byte to the wake-up pipe, both safe
        inside a signal handler. An idle wait then ends at once.
        """
        self._shutdown_requested = True
        if self._wake_write is not None:
            try:
                os.write(self._wake_write, b"x")
            except OSError:
                pass  # pipe full or closed: the flag is what counts

    def _is_shutdown(self) -> bool:
        return self._shutdown_requested

    def run(self, max_iterations: int | None = None) -> None:
        """Run due tasks and wait, until shutdown or ``max_iterations``.

        One iteration is: touch liveness, run every task that is due, then
        (if not shutting down) sleep until the next one is due, in a slice of
        at most ``MAX_SLEEP_SLICE_SEC``. ``max_iterations`` bounds the loop
        for tests; state persists across calls.
        """
        if self._install_signal_handlers_enabled:
            self._install_signals()
        self._open_wake_pipe()
        try:
            iterations = 0
            while max_iterations is None or iterations < max_iterations:
                iterations += 1
                if self._is_shutdown():
                    return
                self._touch_liveness()
                try:
                    self._run_due_tasks()
                except _DatabaseUnavailable:
                    self._back_off()
                    continue
                if self._is_shutdown():
                    return
                self._sleep_until_next_due()
        finally:
            self._close_wake_pipe()
            if self._install_signal_handlers_enabled:
                self._restore_signals()

    def run_once(self) -> bool:
        """Run every task once in registry order; ``True`` if none failed.

        ``ok`` and ``skipped_locked`` both count as success. An unreachable
        database counts as a failure and is not retried.
        """
        if self._install_signal_handlers_enabled:
            self._install_signals()
        try:
            all_ok = True
            for task in self._tasks:
                if self._is_shutdown():
                    break
                try:
                    outcome = self._run_task(task)
                except _DatabaseUnavailable:
                    all_ok = False
                    continue
                if outcome == "failed":
                    all_ok = False
            return all_ok
        finally:
            if self._install_signal_handlers_enabled:
                self._restore_signals()

    # -- loop body ------------------------------------------------------------

    def _is_due(self, task: Task) -> bool:
        last = self._last_start.get(task.name)
        return last is None or self._clock() >= last + task.interval_sec

    def _run_due_tasks(self) -> None:
        for task in self._tasks:
            if self._is_shutdown():
                return
            if self._is_due(task):
                self._run_task(task)

    def _sleep_until_next_due(self) -> None:
        if not self._tasks:
            wait = MAX_SLEEP_SLICE_SEC
        else:
            now = self._clock()
            next_due = min(
                self._last_start[t.name] + t.interval_sec if t.name in self._last_start else now
                for t in self._tasks
            )
            wait = next_due - now
        if wait > 0:
            self._sleep(min(wait, MAX_SLEEP_SLICE_SEC))

    def _run_task(self, task: Task) -> Outcome:
        """Run one task under its lock. Raises ``_DatabaseUnavailable``."""
        started = self._clock()
        try:
            acquired = self._locks.try_lock(task.name)
        except psycopg.OperationalError as exc:
            _log.error(
                "planner.database_unavailable",
                task=task.name,
                error_class=type(exc).__name__,
                error=str(exc),
                retry_in_sec=self._backoff,
            )
            self._locks.reset()
            raise _DatabaseUnavailable from exc
        self._backoff = _INITIAL_DB_BACKOFF_SEC
        self._last_start[task.name] = started

        if not acquired:
            _log.warning(
                "planner task skipped, lock is held: another planner may be running",
                task=task.name,
            )
            self._safely_record(
                task,
                lambda: self._recorder.task_skipped_locked(
                    task.name, interval_sec=task.interval_sec, at=self._wall_clock()
                ),
            )
            self._log_run(task, started, "skipped_locked", None)
            return "skipped_locked"

        self._safely_record(
            task,
            lambda: self._recorder.task_started(
                task.name, interval_sec=task.interval_sec, at=self._wall_clock()
            ),
        )
        outcome: Outcome = "ok"
        result: int | None = None
        error: str | None = None
        try:
            result = task.fn()
        except Exception as exc:  # noqa: BLE001 - a failing task must not stop the others
            outcome = "failed"
            error = f"{type(exc).__name__}: {exc}"[:MAX_RECORDED_ERROR_CHARS]
            _log.error(
                "planner.task_failed",
                task=task.name,
                error_class=type(exc).__name__,
                exc_info=True,
            )
        finally:
            self._release(task)
        self._safely_record(
            task,
            lambda: self._recorder.task_finished(
                task.name,
                interval_sec=task.interval_sec,
                at=self._wall_clock(),
                outcome="failed" if error is not None else "ok",
                error=error,
                count=result,
            ),
        )
        self._log_run(task, started, outcome, result)
        return outcome

    def _safely_record(self, task: Task, write: Callable[[], None]) -> None:
        try:
            write()
        except Exception as exc:  # noqa: BLE001 - recording must never break the planner
            _log.error(
                "planner.record_failed",
                task=task.name,
                error_class=type(exc).__name__,
            )

    def _release(self, task: Task) -> None:
        try:
            self._locks.release(task.name)
        except psycopg.OperationalError as exc:
            # The connection is gone, and Postgres frees its locks with it.
            _log.error(
                "planner.unlock_failed",
                task=task.name,
                error_class=type(exc).__name__,
                error=str(exc),
            )
            self._locks.reset()

    def _log_run(self, task: Task, started: float, outcome: Outcome, result: int | None) -> None:
        fields: dict[str, object] = {
            "task": task.name,
            "duration_ms": round((self._clock() - started) * 1000),
            "outcome": outcome,
        }
        if result is not None:
            fields["count"] = result
        _log.info("planner.task", **fields)

    # -- outage backoff ---------------------------------------------------------

    def _back_off(self) -> None:
        remaining = self._backoff
        self._backoff = min(self._backoff * 2, self._max_db_backoff_sec)
        while remaining > 0 and not self._is_shutdown():
            chunk = min(remaining, MAX_SLEEP_SLICE_SEC)
            self._sleep(chunk)
            remaining -= chunk
            self._touch_liveness()

    # -- waiting ------------------------------------------------------------------

    def _sleep(self, seconds: float) -> None:
        if self._injected_sleep is not None:
            self._injected_sleep(seconds)
        elif self._wake_read is not None:
            # select is retried with the remaining timeout after a signal
            # handler ran, and by then the handler has written the wake byte.
            ready, _, _ = select.select([self._wake_read], [], [], seconds)
            if ready:
                os.read(self._wake_read, 4096)
        else:
            time.sleep(seconds)

    def _open_wake_pipe(self) -> None:
        if self._injected_sleep is not None or self._wake_read is not None:
            return
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)
        self._wake_read, self._wake_write = read_fd, write_fd
        if self._install_signal_handlers_enabled:
            # A signal that lands just before select() starts gets no EINTR
            # and its Python handler would wait until select() times out.
            # Having the C-level handler write to the pipe closes that race.
            self._prev_wakeup_fd = signal.set_wakeup_fd(write_fd, warn_on_full_buffer=False)
        if self._shutdown_requested:
            os.write(write_fd, b"x")

    def _close_wake_pipe(self) -> None:
        if self._install_signal_handlers_enabled and self._wake_write is not None:
            signal.set_wakeup_fd(self._prev_wakeup_fd)
            self._prev_wakeup_fd = -1
        fds = (self._wake_read, self._wake_write)
        self._wake_read = self._wake_write = None
        for fd in fds:
            if fd is not None:
                os.close(fd)

    # -- liveness ---------------------------------------------------------------

    def _touch_liveness(self) -> None:
        try:
            self._liveness_path.touch()
        except OSError as exc:
            _log.error("planner.liveness_touch_failed", error=str(exc))

    # -- signals ------------------------------------------------------------------

    def _install_signals(self) -> None:
        self._prev_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._on_signal)
        self._prev_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._on_signal)

    def _restore_signals(self) -> None:
        for sig, prev in self._prev_handlers.items():
            signal.signal(sig, prev)  # type: ignore[arg-type]
        self._prev_handlers.clear()

    def _on_signal(self, signum: int, frame: FrameType | None) -> None:
        self.request_shutdown()
