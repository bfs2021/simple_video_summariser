"""Planner service entrypoint (issue #38; architecture.md §1, §11.1).

``python -m services.planner.main`` runs the six periodic maintenance tasks
in one long-lived process, each on its own interval:

============================  =====================  =============================
task                          interval               what it does
============================  =====================  =============================
``reap``                      ``REAP_INTERVAL_SEC``  hand back jobs of dead workers
``auto_caption_fallback``     ``POLL_INTERVAL_SEC``  re-enqueue ingest for a video whose
                                                     transcribe job dead-lettered (#92)
``audio_retention``           ``POLL_INTERVAL_SEC``  expire / cap / clean audio
``reanalysis_sweep``          ``POLL_INTERVAL_SEC``  enqueue analyses for a new prompt
``poll_channels``             ``POLL_INTERVAL_SEC``  discover new videos from RSS
``dead_letter_alerts``        ``REAP_INTERVAL_SEC``  alert on TOOL_FAILURE / RESOURCE
                                                     dead-letters (#67); a no-op
                                                     without ``ALERT_WEBHOOK_URL``
============================  =====================  =============================

Every task runs once at startup, in that order (``PROMPT_VERSION`` changes
only through a restart, so the sweep must fire on start). Scheduling lives in
``services.planner.scheduler``; this module wires the tasks, owns the lock
keys and the CLI.

**Exactly one replica.** The planner must run as exactly one replica
(architecture §1, §11.1): the tasks are idempotent, but two planners would
poll every feed and sweep twice for no gain. Each task therefore runs under a
session-level PostgreSQL advisory lock on a dedicated connection, and a second
planner skips a locked task with a WARNING instead of doing it twice. The
locks are defence in depth that make an accidental second replica harmless and
visible; they are not permission to scale the service out. If the process
dies Postgres releases the locks with its connection, so nothing needs cleanup.

``--once`` runs every task once and exits 0 (all ``ok`` or ``skipped_locked``)
or 1 (any task raised), so an operator can trigger a sweep by hand::

    docker compose run --rm planner python -m services.planner.main --once

The planner never runs migrations (AGENTS.md).
"""

from __future__ import annotations

import signal
import sys

# Handlers first (issue #108): when run as ``python -m``, a SIGTERM/SIGINT that
# arrives while the heavy imports below are still loading sets a flag instead
# of killing the process, and ``main()`` then exits 0. Only ``signal`` and
# ``sys`` may be imported above this point. Importing the module (as tests do)
# installs nothing.
_early_stop = False


def _early_on_signal(signum: int, frame: object) -> None:
    global _early_stop
    _early_stop = True


def _install_early_handlers() -> None:
    if __name__ != "__main__":
        return  # imported, not run: leave the process's signals alone
    signal.signal(signal.SIGTERM, _early_on_signal)
    signal.signal(signal.SIGINT, _early_on_signal)


_install_early_handlers()

import argparse
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

import psycopg
import structlog
from pydantic import ValidationError

from adapters.alerts.webhook import WebhookAlerter
from adapters.youtube.feed import YouTubeFeed
from common.config import Settings, get_settings
from common.db import connect
from common.logging import configure_logging
from common.queue import PostgresQueue
from common.repo import planner_runs
from common.shutdown import SignalLatch
from services.planner.alerts import send_dead_letter_alerts
from services.planner.fallback import run_auto_caption_fallback_sweep
from services.planner.poll import poll_channels
from services.planner.reaper import reap_stale_jobs
from services.planner.retention import enforce_audio_retention, max_bytes_from_gb
from services.planner.scheduler import Scheduler, Task
from services.planner.sweep import run_reanalysis_sweep

_log = structlog.get_logger(__name__)

#: ``pg_try_advisory_lock(namespace, task_id)`` keys. Fixed constants, never
#: derived from ``hash()``, which differs per process and would give two
#: replicas different keys. The namespace spells "YTPD" in ASCII.
LOCK_NAMESPACE = 0x59545044
LOCK_KEYS: dict[str, tuple[int, int]] = {
    "reap": (LOCK_NAMESPACE, 1),
    "audio_retention": (LOCK_NAMESPACE, 2),
    "reanalysis_sweep": (LOCK_NAMESPACE, 3),
    "poll_channels": (LOCK_NAMESPACE, 4),
    "dead_letter_alerts": (LOCK_NAMESPACE, 5),
    "auto_caption_fallback": (LOCK_NAMESPACE, 6),
}

Connect = Callable[[], psycopg.Connection[Any]]


class PostgresAdvisoryLocks:
    """Session-level advisory locks on one dedicated, lazily opened connection.

    The connection is autocommit, so no transaction lingers between calls.
    ``reset()`` and ``close()`` drop it; Postgres then frees every lock the
    session held.
    """

    def __init__(self, connect: Connect) -> None:
        self._connect = connect
        self._conn: psycopg.Connection[Any] | None = None

    def try_lock(self, name: str) -> bool:
        conn = self._ensure_connection()
        row = conn.execute("SELECT pg_try_advisory_lock(%s, %s)", LOCK_KEYS[name]).fetchone()
        return bool(row is not None and row[0])

    def release(self, name: str) -> None:
        conn = self._ensure_connection()
        row = conn.execute("SELECT pg_advisory_unlock(%s, %s)", LOCK_KEYS[name]).fetchone()
        if row is None or not row[0]:
            _log.warning("planner lock was not held at release", task=name)

    def reset(self) -> None:
        self.close()

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except psycopg.Error:
                pass

    def _ensure_connection(self) -> psycopg.Connection[Any]:
        if self._conn is None:
            conn = self._connect()
            conn.autocommit = True
            self._conn = conn
        return self._conn


class PostgresRunRecorder:
    """Writes ``planner_task_runs`` on its own lazily opened autocommit connection.

    A failed write drops the connection (the next write reconnects) and
    re-raises; the scheduler logs ``planner.record_failed`` and carries on.
    """

    def __init__(self, connect: Connect) -> None:
        self._connect = connect
        self._conn: psycopg.Connection[Any] | None = None

    def task_started(self, name: str, *, interval_sec: float, at: datetime) -> None:
        self._write(lambda c: planner_runs.record_started(c, name, interval_sec=interval_sec, at=at))

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
        self._write(
            lambda c: planner_runs.record_finished(
                c, name, interval_sec=interval_sec, at=at, outcome=outcome, error=error, count=count
            )
        )

    def task_skipped_locked(self, name: str, *, interval_sec: float, at: datetime) -> None:
        self._write(
            lambda c: planner_runs.record_skipped_locked(c, name, interval_sec=interval_sec, at=at)
        )

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except psycopg.Error:
                pass

    def _write(self, work: Callable[[psycopg.Connection[Any]], None]) -> None:
        try:
            if self._conn is None:
                conn = self._connect()
                conn.autocommit = True
                self._conn = conn
            work(self._conn)
        except psycopg.Error:
            self.close()
            raise


def build_registry(settings: Settings, connect: Connect) -> list[Task]:
    """The ordered task registry. Opens no connection; each run opens its own."""

    def with_conn(work: Callable[[psycopg.Connection[Any]], int | None]) -> Callable[[], int | None]:
        def run() -> int | None:
            conn = connect()
            try:
                return work(conn)
            finally:
                conn.close()

        return run

    def reap(conn: psycopg.Connection[Any]) -> int:
        return reap_stale_jobs(PostgresQueue(conn, settings=settings), settings)

    def auto_caption_fallback(conn: psycopg.Connection[Any]) -> int:
        return run_auto_caption_fallback_sweep(
            conn, PostgresQueue(conn, settings=settings), settings=settings
        )

    def audio_retention(conn: psycopg.Connection[Any]) -> int:
        report = enforce_audio_retention(
            conn,
            settings.AUDIO_DIR,
            max_bytes=max_bytes_from_gb(settings.AUDIO_MAX_GB),
            now=datetime.now(UTC),
        )
        return report.expired_deleted + report.evicted + report.orphans_deleted

    def reanalysis_sweep(conn: psycopg.Connection[Any]) -> int:
        return run_reanalysis_sweep(
            conn,
            PostgresQueue(conn, settings=settings),
            prompt_version=settings.PROMPT_VERSION,
            summarizer_name=settings.SUMMARIZER,
        )

    def poll(conn: psycopg.Connection[Any]) -> int:
        result = poll_channels(conn, YouTubeFeed(), PostgresQueue(conn, settings=settings))
        return result.jobs_enqueued

    webhook = settings.ALERT_WEBHOOK_URL
    sink = (
        WebhookAlerter(webhook.get_secret_value(), fmt=settings.ALERT_WEBHOOK_FORMAT)
        if webhook is not None
        else None
    )

    def dead_letter_alerts(conn: psycopg.Connection[Any]) -> int:
        return send_dead_letter_alerts(
            conn, sink, min_interval_sec=settings.ALERT_MIN_INTERVAL_SEC
        )

    def no_channel() -> int:
        return 0  # no alert channel: do not even open a connection

    return [
        Task("reap", settings.REAP_INTERVAL_SEC, with_conn(reap)),
        Task("auto_caption_fallback", settings.POLL_INTERVAL_SEC, with_conn(auto_caption_fallback)),
        Task("audio_retention", settings.POLL_INTERVAL_SEC, with_conn(audio_retention)),
        Task("reanalysis_sweep", settings.POLL_INTERVAL_SEC, with_conn(reanalysis_sweep)),
        Task("poll_channels", settings.POLL_INTERVAL_SEC, with_conn(poll)),
        Task(
            "dead_letter_alerts",
            settings.REAP_INTERVAL_SEC,
            with_conn(dead_letter_alerts) if sink is not None else no_channel,
        ),
    ]


def build_scheduler(settings: Settings) -> Scheduler:
    return Scheduler(
        build_registry(settings, connect),
        PostgresAdvisoryLocks(connect),
        recorder=PostgresRunRecorder(connect),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    scheduler_factory: Callable[[Settings], Scheduler] = build_scheduler,
) -> int:
    latch = SignalLatch.start(lambda: _early_stop)  # first: see ``common.shutdown`` (#108)
    try:
        return _run(argv, scheduler_factory, latch)
    finally:
        latch.restore()


def _run(
    argv: Sequence[str] | None,
    scheduler_factory: Callable[[Settings], Scheduler],
    latch: SignalLatch,
) -> int:
    if latch.requested:
        return 0
    parser = argparse.ArgumentParser(
        prog="python -m services.planner.main",
        description="Run the planner's periodic tasks (exactly one replica).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run every task once in registry order and exit (1 if any task raised)",
    )
    args = parser.parse_args(argv)
    if latch.requested:
        return 0

    try:
        settings = get_settings()
    except ValidationError as exc:
        print(f"planner: invalid settings:\n{exc}", file=sys.stderr)
        return 1
    configure_logging(settings)
    if latch.requested:
        return 0

    scheduler = scheduler_factory(settings)
    latch.attach(scheduler.request_shutdown)
    if latch.requested:
        return 0
    if args.once:
        return 0 if scheduler.run_once() else 1
    scheduler.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
