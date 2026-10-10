"""Transcriber service entrypoint (issue #32; architecture.md §1, §7.2, §11.5).

``python -m services.transcriber.main`` runs one long-lived worker that claims
``ingest`` and ``transcribe`` jobs and hands each to the #28 or #29 handler
through #13's ``Worker``. Any number of replicas can run side by side, because
a process keeps nothing between jobs except:

* the queue connections (one to claim, one for the heartbeat), and
* one lazily built ``Transcriber``, shared by every ``transcribe`` job. The
  Whisper model behind it (2-3 GB for ``large-v3``) loads on its first use, so
  a replica that only ingests never pays for it, and a failed load is never
  cached: the next ``transcribe`` job tries again.

Ingest hands off to transcribe only through the queue (a ``transcribe`` row),
so a different replica can pick it up. Each job opens its own database
connection for the handler and closes it when the job ends.

Importing this module has no side effects: no settings, no connection, no
``faster_whisper``. ``main()`` does the work, in this order: validate settings
(exit 2 on a bad value, naming the field and never its value), configure
logging, make sure ``AUDIO_DIR`` exists and is writable (exit 2 otherwise,
because every ``transcribe`` job would dead-letter), log ``transcriber.started``,
then open the database. An unreachable database at startup is retried with a
capped backoff instead of exiting, and SIGTERM during that wait exits 0.

Migrations never run here (AGENTS.md).

``handlers``, ``transcriber_factory`` and ``queue_factory`` are for tests. With
``handlers`` given, the production handlers (and so the transcriber) are not
built at all.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from types import FrameType
from typing import Any
from urllib.parse import urlsplit

import psycopg
import structlog
from pydantic import ValidationError

from adapters.transcription.faster_whisper import FasterWhisperTranscriber
from adapters.youtube.audio import YouTubeAudio
from adapters.youtube.metadata import YouTubeMetadata
from adapters.youtube.subtitles import YouTubeSubtitles
from adapters.youtube.ytdlp import CookieFileError, check_cookie_file
from common.chunking import chunk_segments
from common.config import Settings, get_settings
from common.db import connect as default_connect
from common.errors import BugError
from common.logging import configure_logging
from common.models import (
    AudioRef,
    AudioSource,
    MetadataSource,
    Transcriber,
    TranscriptResult,
)
from common.queue import Job, JobQueue, PostgresQueue
from common.worker import JobContext, Worker
from services.transcriber.ingest import SubtitleSource, make_ingest_handler
from services.transcriber.persist import Chunker
from services.transcriber.transcribe import make_transcribe_handler

_log = structlog.get_logger(__name__)

Handler = Callable[[Job, JobContext], None]
Connect = Callable[[], psycopg.Connection[Any]]

#: The liveness file the compose healthcheck reads (architecture.md §11.5).
LIVENESS_PATH = Path("/tmp/heartbeat")

#: Same ceiling as the worker's own database-outage backoff (#13).
_MAX_DB_BACKOFF_SEC = 60.0
_INITIAL_DB_BACKOFF_SEC = 1.0

_EXIT_BAD_CONFIG = 2


def worker_name() -> str:
    """``transcriber:<hostname>:<pid>``: unique per replica, so ``locked_by`` never collides."""
    return f"transcriber:{socket.gethostname()}:{os.getpid()}"


# -- the transcriber ---------------------------------------------------------


def default_transcriber_factory(settings: Settings) -> Callable[[], Transcriber]:
    """Builds the faster-whisper transcriber. It does not import or load anything itself."""

    def build() -> Transcriber:
        return FasterWhisperTranscriber(
            settings.WHISPER_MODEL, settings.WHISPER_COMPUTE, settings.WHISPER_THREADS
        )

    return build


class LazyTranscriber:
    """Builds the real transcriber on first use and reuses it afterwards.

    A factory that raises leaves nothing cached, so the next call builds again.
    """

    def __init__(self, factory: Callable[[], Transcriber]) -> None:
        self._factory = factory
        self._transcriber: Transcriber | None = None

    def transcribe(
        self,
        audio: AudioRef,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> TranscriptResult:
        if self._transcriber is None:
            self._transcriber = self._factory()
        return self._transcriber.transcribe(audio, language=language, on_progress=on_progress)


# -- queues and their connections ---------------------------------------------


class _ClosablePostgresQueue(PostgresQueue):
    def close(self) -> None:
        self._conn.close()


def production_queue_factory(
    settings: Settings, *, connect: Connect | None = None
) -> Callable[[], JobQueue]:
    """Each call opens a new connection and wraps it in a ``PostgresQueue``."""
    open_connection = connect if connect is not None else default_connect

    def open_queue() -> JobQueue:
        return _ClosablePostgresQueue(open_connection(), settings=settings)

    return open_queue


class QueuePool:
    """Opens the process's queues and remembers them, so all can be closed on exit.

    ``open_main`` is the claim loop's queue and ``reconnect`` replaces it (closing
    the one it replaces); ``open_heartbeat`` gives the heartbeat its own.
    """

    def __init__(self, factory: Callable[[], JobQueue]) -> None:
        self._factory = factory
        self._open: list[JobQueue] = []
        self._main: JobQueue | None = None

    def open_main(self) -> JobQueue:
        self._main = self._new()
        return self._main

    def open_heartbeat(self) -> JobQueue:
        return self._new()

    def reconnect(self) -> JobQueue:
        old = self._main
        self._main = self._new()
        if old is not None:
            self._discard(old)
        return self._main

    def close_all(self) -> None:
        for queue in self._open[:]:
            self._discard(queue)
        self._main = None

    def _new(self) -> JobQueue:
        queue = self._factory()
        self._open.append(queue)
        return queue

    def _discard(self, queue: JobQueue) -> None:
        if queue in self._open:
            self._open.remove(queue)
        close = getattr(queue, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception as exc:  # noqa: BLE001 - closing is best effort
            _log.warning("transcriber.queue_close_failed", error=str(exc))


def open_queue_with_backoff(
    open_queue: Callable[[], JobQueue],
    stop: threading.Event,
    *,
    wait: Callable[[float], bool] | None = None,
    max_backoff_sec: float = _MAX_DB_BACKOFF_SEC,
    redact: Callable[[str], str] = str,
) -> JobQueue | None:
    """Open the queue, retrying with a capped, doubling backoff while the database is down.

    Returns ``None`` if ``stop`` is set (or ``wait`` reports it) before a queue opens.
    """
    wait_for = wait if wait is not None else stop.wait
    delay = _INITIAL_DB_BACKOFF_SEC
    while not stop.is_set():
        try:
            return open_queue()
        except Exception as exc:  # noqa: BLE001 - any failure to connect is retried
            _log.error(
                "transcriber.db_unreachable",
                error=redact(str(exc)),
                retry_in_sec=delay,
            )
        if wait_for(delay):
            return None
        delay = min(delay * 2, max_backoff_sec)
    return None


# -- handlers -------------------------------------------------------------------


def dispatcher(handlers: Mapping[str, Handler]) -> Handler:
    """One handler that routes each job to the handler for its kind."""
    table = dict(handlers)

    def dispatch(job: Job, ctx: JobContext) -> None:
        handler = table.get(job.kind)
        if handler is None:
            raise BugError(f"no handler for job kind {job.kind!r}; this worker has {sorted(table)}")
        handler(job, ctx)

    return dispatch


def build_handlers(
    settings: Settings,
    *,
    transcriber: Transcriber,
    connect: Connect | None = None,
    audio_source: AudioSource | None = None,
    metadata: MetadataSource | None = None,
    subtitles: SubtitleSource | None = None,
    chunker: Chunker | None = None,
) -> dict[str, Handler]:
    """The kind -> handler map. Every job opens, uses and closes its own connection."""
    open_connection = connect if connect is not None else default_connect
    cookie_file = settings.YTDLP_COOKIE_FILE
    audio = audio_source if audio_source is not None else YouTubeAudio(cookie_file=cookie_file)
    meta_source = metadata if metadata is not None else YouTubeMetadata(cookie_file=cookie_file)
    subtitle_source = (
        subtitles if subtitles is not None else YouTubeSubtitles(cookie_file=cookie_file)
    )
    split = chunker if chunker is not None else chunk_segments

    def ingest(job: Job, ctx: JobContext) -> None:
        with closing(open_connection()) as conn:
            make_ingest_handler(
                conn=conn,
                queue=PostgresQueue(conn, settings=settings),
                metadata=meta_source,
                subtitles=subtitle_source,
                settings=settings,
            )(job, ctx)

    def transcribe(job: Job, ctx: JobContext) -> None:
        with closing(open_connection()) as conn:
            make_transcribe_handler(
                connect=open_connection,
                queue=PostgresQueue(conn, settings=settings),
                audio_source=audio,
                transcriber=transcriber,
                chunker=split,
                settings=settings,
            )(job, ctx)

    return {"ingest": ingest, "transcribe": transcribe}


def build_worker(
    settings: Settings,
    *,
    name: str,
    handlers: Mapping[str, Handler],
    pool: QueuePool,
    queue: JobQueue,
    sleep: Callable[[float], None] | None = None,
) -> Worker:
    """The claim loop: kinds are the handler map's keys, the heartbeat has its own queue."""
    return Worker(
        name,
        list(handlers),
        dispatcher(handlers),
        queue,
        liveness_path=LIVENESS_PATH,
        settings=settings,
        heartbeat_queue_factory=pool.open_heartbeat,
        reconnect=pool.reconnect,
        sleep=sleep if sleep is not None else time.sleep,
    )


# -- startup checks -----------------------------------------------------------------


def prepare_audio_dir(path: Path) -> None:
    """Create ``path`` if missing and prove a file can be written there.

    Raises ``OSError`` when it cannot be created, is not a directory or is not writable.
    """
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path):
        pass


def _invalid_settings_line(exc: ValidationError) -> str:
    """One line naming each bad field and why, never the value it had."""
    problems: list[str] = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        field = ".".join(str(part) for part in error["loc"])
        message = str(error["msg"]).replace("\n", " ")
        problems.append(f"{field}: {message}" if field else message)
    return "transcriber: invalid settings: " + "; ".join(problems)


def _redactor(settings: Settings) -> Callable[[str], str]:
    secrets: list[str] = []
    dsn = settings.DATABASE_URL.get_secret_value()
    secrets.append(dsn)
    try:
        password = urlsplit(dsn).password
    except ValueError:
        password = None
    if password:
        secrets.append(password)
    if settings.ANTHROPIC_API_KEY is not None:
        secrets.append(settings.ANTHROPIC_API_KEY.get_secret_value())
    ordered = sorted((s for s in secrets if s), key=len, reverse=True)

    def redact(text: str) -> str:
        for secret in ordered:
            text = text.replace(secret, "***")
        return text

    return redact


# -- main -------------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    handlers: Mapping[str, Handler] | None = None,
    transcriber_factory: Callable[[], Transcriber] | None = None,
    queue_factory: Callable[[], JobQueue] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m services.transcriber.main",
        description="Run a transcriber worker: claims ingest and transcribe jobs.",
    )
    parser.parse_args(argv)

    try:
        settings = get_settings()
    except ValidationError as exc:
        print(_invalid_settings_line(exc), file=sys.stderr)
        return _EXIT_BAD_CONFIG
    configure_logging(settings)

    try:
        check_cookie_file(settings.YTDLP_COOKIE_FILE)
    except CookieFileError as exc:
        # Once, at startup: a bad path must not turn into a failure per job.
        print(f"transcriber: {exc}", file=sys.stderr)
        return _EXIT_BAD_CONFIG

    try:
        prepare_audio_dir(settings.AUDIO_DIR)
    except OSError as exc:
        _log.error(
            "transcriber.audio_dir_unusable",
            path=str(settings.AUDIO_DIR),
            error=str(exc),
        )
        return _EXIT_BAD_CONFIG

    if handlers is None:
        factory = (
            transcriber_factory
            if transcriber_factory is not None
            else default_transcriber_factory(settings)
        )
        handlers = build_handlers(settings, transcriber=LazyTranscriber(factory), connect=default_connect)
    name = worker_name()
    _log.info(
        "transcriber.started",
        worker=name,
        kinds=list(handlers),
        whisper_model=settings.WHISPER_MODEL,
        whisper_compute=settings.WHISPER_COMPUTE,
        whisper_threads=settings.WHISPER_THREADS,
        audio_dir=str(settings.AUDIO_DIR),
        audio_keep=settings.AUDIO_KEEP,
    )

    pool = QueuePool(
        queue_factory if queue_factory is not None else production_queue_factory(settings)
    )
    stop = threading.Event()
    running: list[Worker] = []

    def on_signal(signum: int, frame: FrameType | None) -> None:
        stop.set()
        for worker in running:
            worker.request_shutdown()

    previous = {
        sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        queue = open_queue_with_backoff(pool.open_main, stop, redact=_redactor(settings))
        if queue is None:
            return 0
        worker = build_worker(settings, name=name, handlers=handlers, pool=pool, queue=queue)
        running.append(worker)
        if stop.is_set():
            return 0
        worker.run()
        return 0
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        pool.close_all()


if __name__ == "__main__":
    sys.exit(main())
