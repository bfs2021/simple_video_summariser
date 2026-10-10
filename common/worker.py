"""Worker runtime (issue #13; architecture.md §1, §4, §5, §11.5).

``Worker`` is the claim → handle → repeat loop every job-consuming service
(``analyzer``, ``transcriber``) runs. It owns three concerns the handler
itself must not have to think about:

- **The claim loop.** Claim a job of the right kind, run the handler, let
  the queue (#11) record the outcome, repeat. An idle worker waits
  ``poll_sec`` with jitter so N replicas don't poll in lockstep. A database
  outage backs off (capped) instead of crash-looping the container.
- **Liveness.** ``bound_context`` (#6) wraps every job so its logs carry
  ``job_id``/``video_id``/``kind``/``attempt``, and the §11.5 heartbeat file
  is touched every loop iteration and every heartbeat tick, so a
  multi-hour handler never looks wedged to Docker's healthcheck.
- **Graceful shutdown.** SIGTERM/SIGINT stop new claims immediately. A
  handler already running gets ``WORKER_SHUTDOWN_GRACE_SEC`` to finish; past
  that, ``ctx.cancelled`` is set so a cooperative handler can raise
  ``Cancelled`` (#15) and hand its job back without spending an attempt. A
  second signal during the grace period cancels immediately. A handler that
  never checks ``ctx.check_cancelled()`` is left running for Docker's
  SIGKILL to reap (#12 already proves the queue recovers that job).

The job heartbeat (``queue.heartbeat``, keeping ``jobs.heartbeat_at`` fresh
so the reaper in #11 leaves a live job alone) runs on a background timer
against its *own* database connection - never the connection the claim loop
uses - so a slow or blocked claim connection can't stall the heartbeat, and
vice versa.

Every timing knob (the poll sleep, the heartbeat tick, the grace deadline)
is driven through injected ``sleep``/``timer_factory``/``rng`` callables so
tests can drive the state machine without waiting on a real clock. Only one
test (a real ``SIGTERM`` sent to a subprocess) needs the real clock, to prove
the signal handler is actually installed.
"""

from __future__ import annotations

import contextvars
import os
import random
import signal
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

import structlog

from common.config import Settings, get_settings
from common.errors import Cancelled, ToolFailureError
from common.logging import bound_context
from common.queue import Job, JobQueue

_log = structlog.get_logger(__name__)

#: Idle poll wait is poll_sec scaled by 1 +/- this fraction (architecture.md
#: §1: "so N replicas don't poll in lockstep").
_JITTER_FRACTION = 0.20

#: First backoff when the database is unreachable; doubled each consecutive
#: failure up to ``max_db_backoff_sec``.
_INITIAL_DB_BACKOFF_SEC = 1.0

#: Default cap for the database-unreachable backoff (issue #13 AC).
_DEFAULT_MAX_DB_BACKOFF_SEC = 60.0


#: Process exit status of a worker that killed itself over a hung handler (#66).
EXIT_HUNG_HANDLER = 70

#: Fixed allowance added to ``duration x RTF`` for the transcribe limit: model
#: load, audio decode and the final write all happen outside the audio loop.
_TRANSCRIBE_OVERHEAD_SEC = 600.0

#: Longest audio duration (one year) a transcribe limit is derived from. Above
#: this the value is nonsense, and the limit would exceed what a timer thread
#: can wait, so such a job gets no limit instead.
_MAX_PLAUSIBLE_DURATION_SEC = 365 * 24 * 3600


class RuntimeLimitExceeded(ToolFailureError):
    """A handler ran past its kind's maximum wall-clock runtime (#66).

    A ``ToolFailureError`` so it retries like a stuck tool (at most three
    attempts, alerting) and a job that always hangs ends up dead-lettered.
    """


def default_runtime_limit_sec(settings: Settings, job: Job) -> float | None:
    """Maximum runtime for ``job`` in seconds; ``None`` means no limit (#66).

    ``ingest`` and ``analyze`` use a fixed per-kind setting. ``transcribe``
    is ``payload["duration_sec"] x TRANSCRIBE_MAX_RTF`` plus a fixed
    allowance; with no usable duration there is no basis for a limit, so none
    is applied rather than risk killing a healthy long transcription.
    """
    if job.kind == "ingest":
        return float(settings.INGEST_MAX_RUNTIME_SEC)
    if job.kind == "analyze":
        return float(settings.ANALYZE_MAX_RUNTIME_SEC)
    if job.kind == "transcribe":
        duration = job.payload.get("duration_sec")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            return None
        if not 0 < duration <= _MAX_PLAUSIBLE_DURATION_SEC:
            # Also catches nan/inf and ints too big for a float. An absurd
            # value must not crash the claim or push the timer past what a
            # threading.Timer can wait (which would silently never fire).
            _log.warning("worker.runtime_limit_duration_unusable", job_id=job.id)
            return None
        return float(duration) * settings.TRANSCRIBE_MAX_RTF + _TRANSCRIBE_OVERHEAD_SEC
    return None


class TimerHandle(Protocol):
    """What ``timer_factory`` must hand back: something cancellable."""

    def cancel(self) -> None: ...


#: Schedules ``function`` to run once, after ``interval`` seconds, returning
#: a handle whose ``cancel()`` prevents that if it hasn't fired yet. Injected
#: so heartbeat and grace-period tests can fire ticks/expiry synchronously,
#: with no real waiting (issue #13's "Tests" AC).
TimerFactory = Callable[[float, Callable[[], None]], TimerHandle]


def _default_timer_factory(interval: float, function: Callable[[], None]) -> TimerHandle:
    timer = threading.Timer(interval, function)
    timer.daemon = True
    timer.start()
    return timer


def _default_reconnect() -> JobQueue:
    """Open a fresh connection and wrap it in a ``PostgresQueue``.

    Used after a claim failure when the caller gave no ``reconnect=``: the
    connection that just failed is not trusted again. Imported locally for
    the same reason as the heartbeat's default (see ``_get_heartbeat_queue``).
    """
    from common.db import connect
    from common.queue import PostgresQueue

    return PostgresQueue(connect())


class ProgressHook(Protocol):
    """Where a handler's progress events go (issue #96).

    ``stage`` names the long step (``"transcribe"``, ``"analyze"``), ``done`` and
    ``total`` say how far it is, ``fields`` carry anything else the stage reports.
    """

    def __call__(self, stage: str, done: float, total: float, **fields: Any) -> None: ...


class JobContext:
    """Passed to every handler call. Deliberately narrow (issue #13, #96):

    ``cancelled``, ``check_cancelled()``, ``progress()`` and a bound ``logger``
    are the *only* things a handler can reach - it cannot see the queue, the
    connection, or anything else about the worker running it.
    """

    def __init__(
        self,
        logger: structlog.typing.FilteringBoundLogger,
        progress: ProgressHook | None = None,
    ) -> None:
        self.logger = logger
        self._cancel_event = threading.Event()
        self._progress_hook = progress
        self._progress_failure_logged = False

    def progress(self, stage: str, done: float, total: float, **fields: Any) -> None:
        """Report progress inside a long stage.

        Without a hook (the workers) this is one structured ``info`` log line with
        the job's bound context. With a hook (the CLI) the hook replaces the log
        line. A failing hook never fails the job: an ``Exception`` from it is
        logged once per job at ``warning`` and swallowed. ``Cancelled`` and
        anything that is not an ``Exception`` (``KeyboardInterrupt``,
        ``SystemExit``) propagate.
        """
        hook = self._progress_hook
        if hook is None:
            self.logger.info(f"{stage} progress", stage=stage, done=done, total=total, **fields)
            return
        try:
            hook(stage, done, total, **fields)
        except Cancelled:
            raise
        except Exception:
            if not self._progress_failure_logged:
                self._progress_failure_logged = True
                self.logger.warning(
                    "progress hook failed; the job carries on", exc_info=True
                )

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def check_cancelled(self) -> None:
        """Raise ``Cancelled`` if the worker has cancelled this job."""
        if self._cancel_event.is_set():
            raise Cancelled()

    def _cancel(self) -> None:
        """Worker-internal: mark this job cancelled. Not part of the handler API."""
        self._cancel_event.set()


class Worker:
    """Claims ``kinds`` jobs from ``queue`` and runs ``handler`` on each.

    ``handler(job, ctx)`` runs synchronously on the calling thread - this
    worker never runs more than one job at a time (architecture.md §1:
    concurrency comes from replicas, not from a process running jobs in
    parallel).
    """

    def __init__(
        self,
        name: str,
        kinds: Sequence[str],
        handler: Callable[[Job, JobContext], None],
        queue: JobQueue,
        *,
        poll_sec: float = 5,
        liveness_path: Path = Path("/tmp/heartbeat"),
        heartbeat_sec: float | None = None,
        shutdown_grace_sec: float | None = None,
        settings: Settings | None = None,
        heartbeat_queue_factory: Callable[[], JobQueue] | None = None,
        reconnect: Callable[[], JobQueue] | None = None,
        max_db_backoff_sec: float = _DEFAULT_MAX_DB_BACKOFF_SEC,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        timer_factory: TimerFactory = _default_timer_factory,
        runtime_limit_sec: Callable[[Job], float | None] | None = None,
        exit_process: Callable[[int], None] = os._exit,
        install_signal_handlers: bool = True,
        dedupe_key: str | None = None,
    ) -> None:
        self._name = name
        self._kinds = list(kinds)
        # Claim only jobs with exactly this ``dedupe_key`` (#100). Passed to
        # ``claim`` only when set, so a queue whose ``claim`` predates the
        # keyword (a fake, the in-process CLI queue) keeps working.
        self._dedupe_key = dedupe_key
        self._handler = handler
        self._queue = queue
        self._poll_sec = poll_sec
        self._liveness_path = liveness_path

        def _resolved_settings() -> Settings:
            return settings if settings is not None else get_settings()

        self._heartbeat_sec = (
            heartbeat_sec if heartbeat_sec is not None else float(_resolved_settings().HEARTBEAT_SEC)
        )
        self._grace_sec = (
            shutdown_grace_sec
            if shutdown_grace_sec is not None
            else float(_resolved_settings().WORKER_SHUTDOWN_GRACE_SEC)
        )

        if runtime_limit_sec is None:

            def runtime_limit_sec(job: Job) -> float | None:
                # Resolved per job, not at construction, and defensively: a
                # Worker built without settings (and without an environment)
                # must still work; it just runs without a limit, loudly.
                try:
                    resolved = _resolved_settings()
                except Exception as exc:  # noqa: BLE001 - config errors must not fail the job
                    _log.error(
                        "worker.runtime_limit_settings_unavailable",
                        worker=name,
                        job_id=job.id,
                        error=type(exc).__name__,
                    )
                    return None
                return default_runtime_limit_sec(resolved, job)

        self._runtime_limit_sec = runtime_limit_sec
        self._exit_process = exit_process

        self._heartbeat_queue_factory = heartbeat_queue_factory
        self._heartbeat_queue: JobQueue | None = None
        # Set once a failed tick has discarded the heartbeat queue. Only
        # consulted while the queue is None; see ``_start_heartbeat`` (#106).
        self._heartbeat_discarded = False
        self._reconnect = reconnect
        self._max_db_backoff_sec = max_db_backoff_sec

        self._sleep = sleep
        self._rng = rng if rng is not None else random.Random()
        self._timer_factory = timer_factory
        self._install_signal_handlers_enabled = install_signal_handlers

        self._shutdown_event = threading.Event()
        self._state_lock = threading.Lock()
        self._grace_timer: TimerHandle | None = None
        self._current_ctx: JobContext | None = None
        self._prev_handlers: dict[int, signal.Handlers | Callable[..., object] | int | None] = {}

    # -- public API -----------------------------------------------------

    def request_shutdown(self) -> None:
        """Stop claiming new jobs; called by the signal handler or a test.

        A first call while a job is running arms the
        ``WORKER_SHUTDOWN_GRACE_SEC`` timer. A second call while that timer
        is still pending cancels the job immediately instead of waiting out
        the rest of the grace period (issue #13 AC).
        """
        self._shutdown_event.set()
        with self._state_lock:
            ctx = self._current_ctx
            timer = self._grace_timer
        if ctx is None:
            return
        if timer is None:
            self._arm_grace(ctx)
        else:
            self._cancel_grace_immediately(ctx, timer)

    def run(self, max_iterations: int | None = None) -> None:
        """Claim → handle → repeat, until shutdown or ``max_iterations``.

        ``max_iterations`` bounds the loop for tests; production callers
        omit it and run until a signal calls ``request_shutdown()``.
        """
        if self._install_signal_handlers_enabled:
            self._install_signals()
        try:
            backoff = _INITIAL_DB_BACKOFF_SEC
            iterations = 0
            while max_iterations is None or iterations < max_iterations:
                iterations += 1
                if self._shutdown_event.is_set():
                    return
                self._touch_liveness()
                try:
                    should_stop = self._run_once()
                except Cancelled:
                    return
                except Exception as exc:  # noqa: BLE001 - database unreachable, keep the loop alive
                    _log.error(
                        "worker.iteration_failed",
                        worker=self._name,
                        kinds=self._kinds,
                        error=str(exc),
                    )
                    self._sleep(backoff)
                    backoff = min(backoff * 2, self._max_db_backoff_sec)
                    self._reconnect_queue()
                    continue
                backoff = _INITIAL_DB_BACKOFF_SEC
                if should_stop:
                    return
        finally:
            if self._install_signal_handlers_enabled:
                self._restore_signals()

    # -- the loop body ----------------------------------------------------

    def _run_once(self) -> bool:
        """Claim and, if there was a job, run it. Returns True to stop the loop."""
        claim = (
            self._queue.claim(self._kinds, worker=self._name)
            if self._dedupe_key is None
            else self._queue.claim(self._kinds, worker=self._name, dedupe_key=self._dedupe_key)
        )
        with claim as job:
            if job is None:
                if self._shutdown_event.is_set():
                    return True
                jitter = self._rng.uniform(1 - _JITTER_FRACTION, 1 + _JITTER_FRACTION)
                self._sleep(self._poll_sec * jitter)
                return False
            self._handle_job(job)
        return self._shutdown_event.is_set()

    def _handle_job(self, job: Job) -> None:
        with bound_context(job_id=job.id, video_id=job.video_id, kind=job.kind, attempt=job.attempts):
            ctx = JobContext(structlog.get_logger("worker"))
            self._set_current_ctx(ctx)
            if self._shutdown_event.is_set():
                # Shutdown arrived in the window between the top-of-loop
                # check and the claim landing; the job is already
                # 'running' in the database, so give it the grace period
                # rather than abandoning it silently.
                self._arm_grace(ctx)
            stop_heartbeat = self._start_heartbeat(job, ctx)
            finish_watchdog = self._start_runtime_watchdog(job, stop_heartbeat)
            try:
                self._handler(job, ctx)
            finally:
                finish_watchdog()
                stop_heartbeat()
                self._disarm_grace()
                self._set_current_ctx(None)

    def _set_current_ctx(self, ctx: JobContext | None) -> None:
        with self._state_lock:
            self._current_ctx = ctx

    # -- liveness -----------------------------------------------------------

    def _touch_liveness(self) -> None:
        try:
            self._liveness_path.touch()
        except OSError as exc:
            _log.error("worker.liveness_touch_failed", worker=self._name, error=str(exc))

    # -- heartbeat ------------------------------------------------------------

    def _get_heartbeat_queue(self) -> JobQueue:
        if self._heartbeat_queue is None:
            if self._heartbeat_queue_factory is not None:
                self._heartbeat_queue = self._heartbeat_queue_factory()
            else:
                # Local import: keeps a plain psycopg dependency out of the
                # module for callers who inject a heartbeat_queue_factory
                # (every non-integration test) and avoids a cycle at import
                # time (common.db has no reason to know about workers).
                from common.db import connect
                from common.queue import PostgresQueue

                self._heartbeat_queue = PostgresQueue(connect())
        return self._heartbeat_queue

    def _discard_heartbeat_queue(self) -> None:
        """Throw away the cached heartbeat queue after a failed tick.

        Best-effort ``close()`` (issue #106): ``JobQueue`` has no ``close``,
        so it is looked up dynamically; if it raises, the queue is still
        discarded - the point is to release the dead connection, not to
        guarantee it.
        """
        queue = self._heartbeat_queue
        self._heartbeat_queue = None
        self._heartbeat_discarded = True
        close = getattr(queue, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - already discarded, just log
                _log.error(
                    "worker.heartbeat_queue_close_failed",
                    worker=self._name,
                    error=str(exc),
                )

    def _start_heartbeat(self, job: Job, ctx: JobContext) -> Callable[[], None]:
        """Start ticking ``queue.heartbeat`` every ``HEARTBEAT_SEC``.

        Returns a ``stop()`` callable. Calling it blocks until any tick
        already in flight has finished, so the caller can be sure no
        heartbeat is sent after it returns (issue #13 AC: "No heartbeat is
        sent after the job's outcome is written").
        """
        # Created eagerly while healthy (issue #32 pins this: the first job
        # opens the connection before its handler runs). After a failed tick
        # discarded the queue it is NOT created here: the database may still
        # be down, and raising out of job start would fail a job over a
        # heartbeat-only problem. The tick reconnects instead (issue #106).
        if not self._heartbeat_discarded:
            self._get_heartbeat_queue()
        stop_flag = threading.Event()
        tick_lock = threading.Lock()
        handle_box: dict[str, TimerHandle] = {}
        captured_context = contextvars.copy_context()

        def _tick() -> None:
            with tick_lock:
                if stop_flag.is_set():
                    return
                # The process and handler are alive whatever the database
                # says, so the §11.5 liveness file is touched on every tick,
                # including ones whose heartbeat call fails, whose reconnect
                # fails, or that report the job reaped (a long DB outage must
                # not look wedged).
                self._touch_liveness()
                # issue #106: fetched at tick time, not captured when the job
                # started, so a queue re-created by an earlier tick (or job)
                # is picked up; at most one factory call and one heartbeat()
                # call per tick.
                try:
                    hb_queue = self._get_heartbeat_queue()
                except Exception as exc:  # noqa: BLE001 - retried on the next tick
                    _log.error(
                        "worker.heartbeat_reconnect_failed",
                        worker=self._name,
                        job_id=job.id,
                        error=str(exc),
                    )
                else:
                    try:
                        ok = hb_queue.heartbeat(job.id, self._name)
                    except Exception as exc:  # noqa: BLE001 - retried on the next tick
                        _log.error(
                            "worker.heartbeat_failed",
                            worker=self._name,
                            job_id=job.id,
                            error=str(exc),
                        )
                        self._discard_heartbeat_queue()
                    else:
                        if not ok:
                            ctx._cancel()
                            _log.warning(
                                "worker.heartbeat_reaped",
                                worker=self._name,
                                job_id=job.id,
                            )
                            stop_flag.set()
                            return
                if not stop_flag.is_set():
                    handle_box["handle"] = self._timer_factory(self._heartbeat_sec, _tick_in_context)

        def _tick_in_context() -> None:
            captured_context.run(_tick)

        handle_box["handle"] = self._timer_factory(self._heartbeat_sec, _tick_in_context)

        def stop() -> None:
            stop_flag.set()
            handle_box["handle"].cancel()
            # Block until an in-flight tick (one that passed the stop_flag
            # check before we set it above) has released the lock, so no
            # heartbeat call is still running once stop() returns.
            with tick_lock:
                pass

        return stop

    # -- runtime limit (#66) ---------------------------------------------------

    def _start_runtime_watchdog(
        self, job: Job, stop_heartbeat: Callable[[], None]
    ) -> Callable[[], None]:
        """Arm a one-shot timer at the job's runtime limit; returns ``finish()``.

        Neither the job heartbeat nor the liveness file notices a handler that
        is wedged (the heartbeat thread keeps both fresh), so this is the only
        thing that does. When it fires the worker stops heartbeating, records
        the attempt as failed (so a job that always hangs dead-letters), logs
        at ERROR and exits the process: Python cannot kill a thread, and the
        container restart plus the reaper (#35) recover anything left over.

        ``finish()`` disarms it. A single lock decides the race between the
        handler returning and the timer firing: whichever takes it first wins,
        so a job that finished is never reported as hung.
        """
        limit = self._runtime_limit_sec(job)
        if limit is None:
            return lambda: None
        lock = threading.Lock()
        finished = False
        captured_context = contextvars.copy_context()

        def _on_limit() -> None:
            with lock:
                if finished:
                    return
                self._handler_hung(job, limit, stop_heartbeat)

        def _on_limit_in_context() -> None:
            captured_context.run(_on_limit)

        handle = self._timer_factory(limit, _on_limit_in_context)

        def finish() -> None:
            nonlocal finished
            with lock:
                finished = True
            handle.cancel()

        return finish

    def _handler_hung(self, job: Job, limit: float, stop_heartbeat: Callable[[], None]) -> None:
        stop_heartbeat()
        exc = RuntimeLimitExceeded(
            f"job exceeded its runtime limit of {limit:g} s (kind={job.kind}, "
            f"attempt={job.attempts}); the worker is exiting"
        )
        _log.error(
            "worker.runtime_limit_exceeded",
            worker=self._name,
            job_id=job.id,
            video_id=job.video_id,
            kind=job.kind,
            attempt=job.attempts,
            limit_sec=limit,
        )
        try:
            queue = self._get_heartbeat_queue()
            fail_job = getattr(queue, "fail_job", None)
            if fail_job is None:
                raise RuntimeError("queue cannot record a failed attempt")
            fail_job(job, self._name, exc)
        except Exception as err:  # noqa: BLE001 - exit regardless; the reaper recovers the job
            _log.error(
                "worker.runtime_limit_record_failed",
                worker=self._name,
                job_id=job.id,
                error=str(err),
            )
        self._exit_process(EXIT_HUNG_HANDLER)

    # -- shutdown / grace period --------------------------------------------

    def _arm_grace(self, ctx: JobContext) -> None:
        with self._state_lock:
            if self._grace_timer is not None:
                return
            captured_context = contextvars.copy_context()

            def _on_expire() -> None:
                captured_context.run(self._grace_expired, ctx)

            self._grace_timer = self._timer_factory(self._grace_sec, _on_expire)

    def _grace_expired(self, ctx: JobContext) -> None:
        with self._state_lock:
            self._grace_timer = None
        ctx._cancel()
        _log.warning("worker.shutdown.grace_expired", worker=self._name)

    def _cancel_grace_immediately(self, ctx: JobContext, timer: TimerHandle) -> None:
        with self._state_lock:
            self._grace_timer = None
        timer.cancel()
        ctx._cancel()
        _log.warning("worker.shutdown.immediate_cancel", worker=self._name)

    def _disarm_grace(self) -> None:
        with self._state_lock:
            timer = self._grace_timer
            self._grace_timer = None
        if timer is not None:
            timer.cancel()

    # -- database reconnect -------------------------------------------------

    def _reconnect_queue(self) -> None:
        reconnect = self._reconnect if self._reconnect is not None else _default_reconnect
        try:
            self._queue = reconnect()
        except Exception as exc:  # noqa: BLE001 - stay up, retry next iteration
            _log.error("worker.reconnect_failed", worker=self._name, error=str(exc))

    # -- signal handling ------------------------------------------------------

    def _install_signals(self) -> None:
        self._prev_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._on_signal)
        self._prev_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._on_signal)

    def _restore_signals(self) -> None:
        for sig, prev in self._prev_handlers.items():
            signal.signal(sig, prev)
        self._prev_handlers.clear()

    def _on_signal(self, signum: int, frame: FrameType | None) -> None:
        self.request_shutdown()
