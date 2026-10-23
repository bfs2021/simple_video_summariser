"""Live progress of a ``transcribe`` job (issue #245, architecture.md §7.2).

``ProgressTicker`` keeps the audio position the model has reached and, every
``PROGRESS_INTERVAL_SEC`` of wall time, stores the progress figures through a
``write`` callable. The tick is time-based, not tied to segment boundaries, so a
long silent stretch still refreshes ``updated_at`` and the UI can tell a quiet job
from a stalled one. The write is a single small ``UPDATE`` of ``jobs.progress`` on
the ticker's own connection (``JobProgressWriter``), separate from the queue's
heartbeat.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
import structlog

from common.progress import PROGRESS_INTERVAL_SEC, Estimate, progress_figures
from common.queue import Job
from common.repo.jobs import write_progress

_log = structlog.get_logger(__name__)

#: How long ``ticking`` waits for the ticker thread to wind down.
_JOIN_TIMEOUT_SEC = 10.0


class ProgressTicker:
    """Evaluates progress figures and hands them to ``write``; see the module docstring."""

    def __init__(
        self,
        write: Callable[[dict[str, Any]], object],
        *,
        duration_sec: float | None,
        estimate: Estimate | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._write = write
        self._clock = clock
        self._lock = threading.Lock()
        self._duration_sec = duration_sec
        self._estimate = estimate
        self._started_at: float | None = None
        self._done_sec = 0.0
        self._write_failure_logged = False

    def set_target(self, duration_sec: float | None, estimate: Estimate | None) -> None:
        """Replace the duration and estimate, e.g. once the real audio length is known."""
        with self._lock:
            self._duration_sec = duration_sec
            self._estimate = estimate

    def begin(self) -> None:
        """Transcription starts now: ``elapsed_sec`` counts from here, so the ETA is
        not skewed by the audio download or the model load before it."""
        with self._lock:
            self._started_at = self._clock()

    def note(self, done_sec: float) -> None:
        """The audio position (seconds) the model has reached."""
        with self._lock:
            self._done_sec = done_sec

    def tick(self) -> None:
        """Evaluate and store the figures once. A failing write is logged once, never raised."""
        with self._lock:
            elapsed = 0.0 if self._started_at is None else self._clock() - self._started_at
            figures = progress_figures(self._done_sec, self._duration_sec, elapsed, self._estimate)
        try:
            self._write(figures)
        except Exception:  # noqa: BLE001 - progress is best effort; the job carries on
            if not self._write_failure_logged:
                self._write_failure_logged = True
                _log.warning("transcribe progress write failed; the job carries on", exc_info=True)

    def run(self, wait: Callable[[float], bool]) -> None:
        """Tick every ``PROGRESS_INTERVAL_SEC`` until ``wait`` returns True (``Event.wait``)."""
        while not wait(PROGRESS_INTERVAL_SEC):
            self.tick()


class JobProgressWriter:
    """``write`` for a ticker: ``write_progress`` on its own lazily opened connection."""

    def __init__(self, connect: Callable[[], psycopg.Connection[Any]], job: Job) -> None:
        self._connect = connect
        self._job = job
        self._conn: psycopg.Connection[Any] | None = None

    def __call__(self, figures: dict[str, Any]) -> bool:
        if self._conn is None:
            self._conn = self._connect()
        # ``locked_by`` is the worker that claimed the job; without one nothing matches.
        return write_progress(self._conn, self._job.id, self._job.locked_by or "", figures)

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - already finished; nothing to recover
                _log.warning("could not close the progress connection", exc_info=True)


@contextmanager
def ticking(ticker: ProgressTicker) -> Iterator[None]:
    """Run ``ticker`` on a daemon thread for the length of the block."""
    stop = threading.Event()
    thread = threading.Thread(
        target=ticker.run, args=(stop.wait,), name="transcribe-progress", daemon=True
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=_JOIN_TIMEOUT_SEC)
