"""Analyzer service entrypoint (issue #33; architecture.md 1, 11.1, 11.5).

``python -m services.analyzer.main`` runs one long-lived worker (#13) that
claims only ``analyze`` jobs and hands each to the analyze handler (#30). The
summarizer backend comes from the single ``SUMMARIZER`` setting through
``adapters.summarize.factory.build_summarizer``; it is built once and reused
for every job.

Startup order matters, because a bad configuration must fail here and not on
the first job:

1. read the settings and configure logging; invalid settings exit with code 2;
2. build the summarizer (this makes no network call); a configuration that can
   never work, such as ``SUMMARIZER=anthropic`` without a key, exits with
   code 2 -- still before any database connection is opened;
3. log ``analyzer.started``;
4. with ``SUMMARIZER=ollama``, probe Ollama once (``GET /api/tags``, 5 s) and
   log one ``analyzer.ollama_*`` line saying whether it is reachable and
   ``OLLAMA_MODEL`` is pulled (#84). The probe only reports: it never changes
   the exit code or delays the worker for more than its timeout, because Ollama
   runs on the host and may start after the stack, and the job loop already
   retries ``LLM_UNAVAILABLE``. It needs no database connection;
5. run the worker.

A database that is down at startup is handled like one that goes down mid-run:
the connection is opened lazily by the worker's own claim loop, so the process
logs, backs off (capped at 60 s) and retries instead of exiting.

The analyzer shares no code path with the transcriber: this package imports
nothing from ``services.transcriber`` or ``adapters.transcription``, so it
runs in the backend image, which has no Whisper (architecture.md 11.1).
"""

from __future__ import annotations

import os
import signal
import sys

# Handlers first (issue #108): when run as ``python -m``, a SIGTERM/SIGINT that
# arrives while the heavy imports below are still loading sets a flag instead
# of killing the process, and ``main()`` then exits 0. Only ``signal``, ``sys``
# and ``os`` may be imported above this point. Importing the module (as tests
# do) installs nothing.
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

import re
import socket
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import structlog
from pydantic import ValidationError

from adapters.summarize.factory import build_summarizer
from adapters.summarize.ollama import GetTransport, probe_ollama, urllib_get_transport
from common.config import Settings, get_settings
from common.db import connect as default_connect
from common.errors import BugError, Defer
from common.logging import configure_logging
from common.queue import Job, JobQueue, PostgresQueue, analyze_dedupe_key
from common.shutdown import SignalLatch
from common.worker import JobContext, Worker
from services.analyzer.handler import make_analyze_handler

_log = structlog.get_logger(__name__)

#: Exit code for a configuration that can never work (like ``EX_USAGE``).
EXIT_CONFIG = 2

#: How long a job whose ``dedupe_key`` this process is not configured for waits.
MISMATCH_DEFER = timedelta(minutes=15)

#: ``<part>:<part>``: exactly what ``analyze_dedupe_key`` can produce (non-empty
#: parts, one colon), and no whitespace, which no configured value contains.
_KEY_RE = re.compile(r"[^:\s]+:[^:\s]+")

Connect = Callable[[], psycopg.Connection[Any]]
Handler = Callable[[Job, JobContext], None]
HandlerFactory = Callable[..., Handler]


def worker_name() -> str:
    """``analyzer-<hostname>-<pid>``: tells replicas apart in ``jobs.locked_by``."""
    return f"analyzer-{socket.gethostname()}-{os.getpid()}"


def build_job_handler(
    settings: Settings,
    summarizer: Any,
    connect: Connect,
    *,
    handler_factory: HandlerFactory = make_analyze_handler,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Handler:
    """The analyze handler behind the job / configuration guard.

    The job's ``dedupe_key`` is ``<PROMPT_VERSION>:<summarizer name>`` (C1). A
    job enqueued for another prompt version or backend must not be analyzed
    with this process's summarizer, so it is handed back for later (another
    replica, or this one after a restart, may own it) without spending an
    attempt. A key that is not even ``<slug>:<slug>`` was never produced by
    ``analyze_dedupe_key``: that is a bug and dead-letters at once.

    Defence in depth since #100: the worker claims by its own key
    (``Worker(dedupe_key=...)``), so the mismatch branch is reachable only
    through a caller that claims without the key, and a malformed key is never
    claimed by this entrypoint at all.
    """
    inner = handler_factory(connect=connect, summarizer=summarizer, settings=settings)
    expected = analyze_dedupe_key(settings.PROMPT_VERSION, summarizer.name)

    def handler(job: Job, ctx: JobContext) -> None:
        key = job.dedupe_key
        if _KEY_RE.fullmatch(key) is None:
            raise BugError(f"analyze job has a malformed dedupe_key {key!r}; expected <version>:<summarizer>")
        if key != expected:
            ctx.logger.warning(
                "analyzer.job_config_mismatch",
                job_dedupe_key=key,
                configured_dedupe_key=expected,
            )
            raise Defer(now() + MISMATCH_DEFER)
        inner(job, ctx)

    return handler


class _LazyQueue:
    """A ``PostgresQueue`` on a connection opened by the first ``claim``.

    Opening the connection inside ``claim`` puts a database that is down at
    startup on the worker's outage path (log, back off, retry). ``reset``
    drops the connection; the worker calls it after a failed iteration and the
    next ``claim`` opens a fresh one, once per backoff cycle.
    """

    def __init__(self, connect: Connect, settings: Settings) -> None:
        self._connect = connect
        self._settings = settings
        self._inner: PostgresQueue | None = None
        self._conn: psycopg.Connection[Any] | None = None

    def claim(
        self, kinds: Sequence[str], *, worker: str, dedupe_key: str | None = None
    ) -> AbstractContextManager[Job | None]:
        if self._inner is None:
            self._conn = self._connect()
            self._inner = PostgresQueue(self._conn, settings=self._settings)
        return self._inner.claim(kinds, worker=worker, dedupe_key=dedupe_key)

    def reset(self) -> _LazyQueue:
        self.close()
        return self

    def close(self) -> None:
        conn, self._conn, self._inner = self._conn, None, None
        if conn is not None:
            with suppress(Exception):  # a dead connection is what we are dropping
                conn.close()

    # The worker only ever calls ``claim`` on this queue; the heartbeat has its own.
    def enqueue(self, *args: Any, **kwargs: Any) -> int | None:
        raise NotImplementedError

    def heartbeat(self, job_id: int, worker: str) -> bool:
        raise NotImplementedError

    def reap_stale(self, older_than_sec: int | None = None) -> int:
        raise NotImplementedError


def _model_name(settings: Settings) -> str:
    return settings.OLLAMA_MODEL if settings.SUMMARIZER == "ollama" else settings.ANTHROPIC_MODEL


def _report_ollama(settings: Settings, transport: GetTransport) -> None:
    """Probe Ollama once and log the outcome (#84). Never raises, never exits."""
    probe = probe_ollama(settings.OLLAMA_HOST, settings.OLLAMA_MODEL, transport=transport)
    if probe.outcome == "ready":
        _log.info("analyzer.ollama_ready", host=probe.host, model=probe.model)
    elif probe.outcome == "model_missing":
        _log.warning(
            "analyzer.ollama_model_missing",
            host=probe.host,
            model=probe.model,
            hint=f"ollama pull {probe.model}",
        )
    else:
        _log.warning(
            "analyzer.ollama_unreachable",
            host=probe.host,
            model=probe.model,
            error=probe.error,
            note="analyze jobs will be retried when Ollama is up",
        )


def _invalid_settings(exc: ValidationError) -> tuple[list[str], list[str]]:
    """Variable names and ``NAME: problem`` lines, without any input value."""
    variables: list[str] = []
    problems: list[str] = []
    for error in exc.errors(include_input=False, include_url=False, include_context=False):
        name = ".".join(str(part) for part in error["loc"])
        if name and name not in variables:
            variables.append(name)
        problems.append(f"{name}: {error['msg']}" if name else error["msg"])
    return variables, problems


def main(
    *,
    settings: Settings | None = None,
    connect: Connect | None = None,
    summarizer_factory: Callable[[Settings], Any] = build_summarizer,
    handler_factory: HandlerFactory = make_analyze_handler,
    worker_options: Mapping[str, Any] | None = None,
    max_iterations: int | None = None,
    probe_transport: GetTransport | None = None,
) -> int:
    """Run the analyzer until SIGTERM/SIGINT; returns the process exit code.

    The keyword arguments are the test seams: ``settings``, the database
    ``connect``, the summarizer and handler factories, extra ``Worker``
    options (``poll_sec``, ``sleep``, ``liveness_path`` ...), a bound on the
    claim loop and the HTTP round-trip of the Ollama startup probe.

    A SIGTERM/SIGINT received while starting up (see ``common.shutdown``, #108)
    makes this return 0 at the next startup step, without claiming a job.
    """
    latch = SignalLatch.start(lambda: _early_stop)  # first, before anything else
    try:
        return _run(
            latch,
            settings=settings,
            connect=connect,
            summarizer_factory=summarizer_factory,
            handler_factory=handler_factory,
            worker_options=worker_options,
            max_iterations=max_iterations,
            probe_transport=probe_transport,
        )
    finally:
        latch.restore()


def _run(
    latch: SignalLatch,
    *,
    settings: Settings | None,
    connect: Connect | None,
    summarizer_factory: Callable[[Settings], Any],
    handler_factory: HandlerFactory,
    worker_options: Mapping[str, Any] | None,
    max_iterations: int | None,
    probe_transport: GetTransport | None,
) -> int:
    if latch.requested:
        return 0
    if settings is None:
        try:
            settings = get_settings()
        except ValidationError as exc:
            # Nothing valid to configure logging from: use the defaults, and
            # stderr, where an operator looks for why the process died.
            configure_logging(Settings.model_construct(), stream=sys.stderr)
            variables, problems = _invalid_settings(exc)
            _log.error("analyzer.invalid_settings", variables=variables, problems=problems)
            return EXIT_CONFIG
    configure_logging(settings)
    if latch.requested:
        return 0

    try:
        summarizer = summarizer_factory(settings)
        # The one key this process claims (#100); built here so a name that cannot
        # form a key is a startup failure, not a crash after ``analyzer.started``.
        dedupe_key = analyze_dedupe_key(settings.PROMPT_VERSION, summarizer.name)
    except Exception as exc:  # noqa: BLE001 - any failure to build is a startup failure
        _log.error("analyzer.invalid_config", error=str(exc))
        return EXIT_CONFIG

    if latch.requested:
        return 0

    name = worker_name()
    fields: dict[str, Any] = {
        "worker": name,
        "summarizer": settings.SUMMARIZER,
        "model": _model_name(settings),
        "prompt_version": settings.PROMPT_VERSION,
        "dedupe_key": dedupe_key,
        "chunk_sec": settings.CHUNK_SEC,
        "overlap_sec": settings.OVERLAP_SEC,
    }
    if settings.SUMMARIZER == "anthropic":
        fields["anthropic_batch"] = settings.ANTHROPIC_BATCH
    _log.info("analyzer.started", **fields)
    if settings.SUMMARIZER == "ollama" and not latch.requested:
        _report_ollama(settings, probe_transport or urllib_get_transport)
    if latch.requested:
        return 0

    open_connection = connect if connect is not None else default_connect
    handler = build_job_handler(
        settings, summarizer, open_connection, handler_factory=handler_factory
    )
    claim_queue = _LazyQueue(open_connection, settings)

    def heartbeat_queue() -> JobQueue:
        # Its own connection: a slow claim must not stall the heartbeat.
        return PostgresQueue(open_connection(), settings=settings)

    worker = Worker(
        name,
        ["analyze"],
        handler,
        claim_queue,
        settings=settings,
        heartbeat_queue_factory=heartbeat_queue,
        reconnect=claim_queue.reset,
        dedupe_key=dedupe_key,
        **(worker_options or {}),
    )
    # Late-bound: resolved when a signal arrives, not when attached.
    latch.attach(lambda: worker.request_shutdown())
    try:
        if latch.requested:
            return 0
        worker.run(max_iterations=max_iterations)
    finally:
        claim_queue.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
