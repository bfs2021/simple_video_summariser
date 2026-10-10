"""``ytdigest run`` - one video through ingest, transcribe and analyze (issue #31).

``ytdigest run <url-or-id>`` (or ``python -m services.cli run ...``) runs the
same handlers the workers run (#28, #29, #30), in one foreground process, on an
in-process queue (``InProcessQueue``) that never touches the ``jobs`` table. It
writes the results to Postgres, reports each stage on stderr and prints the
resulting analysis on stdout. It re-implements no pipeline stage.

Exit codes: 0 done, 1 a stage failed (or the setup did), 2 bad input or usage,
3 the video is not available yet (premiere or live, ``Defer``), 130 Ctrl-C.
The CLI fails fast: nothing is retried and nothing sleeps. It never runs
migrations (AGENTS.md).

``build_inprocess_handlers`` wires the three handler factories onto the in-process
queue and one connection. The summarizer comes from the shared
``adapters.summarize.factory.build_summarizer`` (#33). The transcriber entrypoint
(#32) has its own ``build_handlers`` for the Postgres queue; the two differ only in
the queue and connection they hand the ingest and transcribe handlers.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

import psycopg
import structlog
from pydantic import ValidationError

from adapters.summarize.factory import SummarizerConfigError, build_summarizer
from adapters.transcription.faster_whisper import FasterWhisperTranscriber
from adapters.youtube.audio import YouTubeAudio
from adapters.youtube.metadata import YouTubeMetadata
from adapters.youtube.subtitles import YouTubeSubtitles
from adapters.youtube.ytdlp import CookieFileError, check_cookie_file
from common.chunking import chunk_segments, chunk_strategy
from common.config import Settings, get_settings
from common.db import connect
from common.errors import (
    BugError,
    Cancelled,
    ControlFlow,
    Defer,
    JobError,
    classify,
    format_error,
)
from common.logging import bound_context, configure_logging
from common.models import (
    Analysis,
    AudioSource,
    MetadataSource,
    Summarizer,
    Transcriber,
)
from common.queue import PRIORITY_INTERACTIVE, Job
from common.repo.analyses import latest_analysis
from common.repo.transcripts import get_best_transcript, get_chunks, get_transcript
from common.repo.videos import get_video_meta
from common.worker import JobContext, ProgressHook
from common.youtube_refs import parse_video_ref
from services.analyzer.handler import make_analyze_handler
from services.cli.inprocess import InProcessQueue
from services.cli.output import format_result, sanitize_text
from services.cli.progress import ProgressPrinter
from services.transcriber.ingest import SubtitleSource, make_ingest_handler
from services.transcriber.transcribe import make_transcribe_handler

Handler = Callable[[Job, JobContext], None]

#: The Interactive band (architecture.md §4): a human is waiting for this one.
PRIORITY = PRIORITY_INTERACTIVE
#: A single video needs ingest, transcribe and analyze; ten is generous.
MAX_JOBS = 10
#: Cap for one error message on stderr.
_MAX_MESSAGE_CHARS = 500
#: Tables the pipeline reads or writes; all of them must exist (migrations applied).
_REQUIRED_TABLES = (
    "videos",
    "transcripts",
    "transcript_chunks",
    "analyses",
    "topics",
    "claims",
    "quotes",
    "jobs",
    "media",
)
#: The transcript sources ``get_transcript`` knows, for finding the analyzed one.
_TRANSCRIPT_SOURCES = ("youtube_manual", "whisper", "youtube_auto")

_EXIT_FAILED = 1
_EXIT_USAGE = 2
_EXIT_DEFERRED = 3
_EXIT_INTERRUPTED = 130


@dataclass
class Deps:
    """Everything ``main`` needs from outside; tests pass fakes, production passes ``default_deps``."""

    settings: Settings
    connect: Callable[[], psycopg.Connection[Any]]
    metadata: MetadataSource
    subtitles: SubtitleSource
    audio: AudioSource
    transcriber: Transcriber
    summarizer: Summarizer


class PipelineError(Exception):
    """A stage failed. ``stage`` is the job kind (or ``pipeline``); ``cause`` is what it raised."""

    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(f"{stage}: {cause!r}")
        self.stage = stage
        self.cause = cause


# --------------------------------------------------------------------- wiring


def default_deps(settings: Settings) -> Deps:
    """The real adapters. Nothing here loads a model or starts a process.

    Raises ``CookieFileError`` if ``YTDLP_COOKIE_FILE`` is set but unusable.
    """
    check_cookie_file(settings.YTDLP_COOKIE_FILE)
    return Deps(
        settings=settings,
        connect=connect,
        metadata=YouTubeMetadata(cookie_file=settings.YTDLP_COOKIE_FILE),
        subtitles=YouTubeSubtitles(cookie_file=settings.YTDLP_COOKIE_FILE),
        audio=YouTubeAudio(cookie_file=settings.YTDLP_COOKIE_FILE),
        transcriber=FasterWhisperTranscriber(),
        summarizer=build_summarizer(settings),
    )


def _check_transcriber_first(transcriber: Transcriber, handler: Handler) -> Handler:
    """Fail the transcribe stage on a missing speech-to-text install before any audio work.

    Otherwise the audio directory is touched first, and on a host whose ``AUDIO_DIR``
    is not writable the clear "install requirements.whisper.txt" message is lost.
    """
    if not isinstance(transcriber, FasterWhisperTranscriber):
        return handler

    def checked(job: Job, ctx: JobContext) -> None:
        transcriber.check_available()
        handler(job, ctx)

    return checked


def build_inprocess_handlers(
    deps: Deps, queue: InProcessQueue, conn: psycopg.Connection[Any]
) -> dict[str, Handler]:
    """One handler per job kind, the same factories the worker entrypoints use."""
    return {
        "ingest": make_ingest_handler(
            conn=conn,
            queue=queue,
            metadata=deps.metadata,
            subtitles=deps.subtitles,
            settings=deps.settings,
        ),
        "transcribe": _check_transcriber_first(
            deps.transcriber,
            make_transcribe_handler(
                connect=deps.connect,
                queue=queue,
                audio_source=deps.audio,
                transcriber=deps.transcriber,
                chunker=chunk_segments,
                settings=deps.settings,
            ),
        ),
        "analyze": make_analyze_handler(
            connect=deps.connect, summarizer=deps.summarizer, settings=deps.settings
        ),
    }


# --------------------------------------------------------------------- runner


def run_queue(
    queue: InProcessQueue,
    handlers: Mapping[str, Handler],
    video_id: str,
    *,
    on_start: Callable[[Job], None] | None = None,
    on_finish: Callable[[Job], None] | None = None,
    progress: ProgressHook | None = None,
    max_jobs: int = MAX_JOBS,
) -> list[str]:
    """Run queued jobs FIFO until the queue is empty; returns the kinds run, in order.

    Stops at the first failure with ``PipelineError``: no later job runs and
    nothing is retried. Handlers that keep enqueueing, a job for another video
    and a job of a kind without a handler are bugs (``BugError``), not work.
    ``KeyboardInterrupt`` and ``SystemExit`` pass through untouched.
    """
    ran: list[str] = []
    while True:
        if len(ran) >= max_jobs and queue.pending:
            raise PipelineError(
                "pipeline",
                BugError(f"more than {max_jobs} jobs for one video: the handlers keep enqueueing"),
            )
        with queue.claim(list(handlers), worker="cli") as job:
            if job is None:
                break
            try:
                if job.video_id != video_id:
                    raise BugError(
                        f"a {job.kind!r} job was enqueued for video {job.video_id!r}, "
                        f"not {video_id!r}"
                    )
                if on_start is not None:
                    on_start(job)
                with bound_context(
                    job_id=job.id, video_id=job.video_id, kind=job.kind, attempt=job.attempts
                ):
                    handlers[job.kind](
                        job, JobContext(structlog.get_logger("worker"), progress)
                    )
                if on_finish is not None:
                    on_finish(job)
            except Exception as exc:  # every failure is reported by stage
                raise PipelineError(job.kind, exc) from exc
        ran.append(job.kind)
    leftover = queue.leftover()
    if leftover:
        kind, other = leftover[0]
        raise PipelineError(
            "pipeline", BugError(f"a {kind!r} job for video {other!r} has no handler")
        )
    return ran


# --------------------------------------------------------------------- command line


class _Parser(argparse.ArgumentParser):
    """Usage errors exit 2. The ``run`` parser says how to pass an ID that starts with ``-``."""

    def __init__(self, *args: Any, dash_hint: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._dash_hint = dash_hint

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        text = f"{self.prog}: error: {message}\n"
        if self._dash_hint:
            text += (
                "hint: a video ID that starts with '-' must follow '--', "
                "e.g. ytdigest run -- -wNyEUrxzFU\n"
            )
        self.exit(_EXIT_USAGE, text)


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="ytdigest", description="Structured summaries of YouTube videos."
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    run = commands.add_parser(
        "run",
        help="run one video through ingest, transcribe and analyze",
        description=(
            "Run one video through the pipeline in this process, save the results "
            "to Postgres and print the analysis."
        ),
        dash_hint=True,
    )
    run.add_argument("video", help="a YouTube URL or an 11-character video ID")
    run.add_argument(
        "-v", "--verbose", action="store_true", help="also print the traceback of a failure"
    )
    # Lazy: services.eval imports this module.
    from services.eval.cli import add_eval_parser

    add_eval_parser(commands)
    return parser


def _say(text: str = "") -> None:
    print(text, file=sys.stderr, flush=True)


def _one_line(text: str) -> str:
    """Untrusted ``text`` as a single, printable, length-capped line."""
    line = " ".join(sanitize_text(text).split())
    if len(line) > _MAX_MESSAGE_CHARS:
        line = line[: _MAX_MESSAGE_CHARS - 3] + "..."
    return line


def _describe(exc: BaseException) -> str:
    text = str(exc)
    if isinstance(exc, JobError):
        return _one_line(text or type(exc).__name__)
    return _one_line(f"{type(exc).__name__}: {text}" if text else type(exc).__name__)


def _report_failure(stage: str, exc: BaseException, *, verbose: bool) -> int:
    if isinstance(exc, Defer):
        _say(f"[{stage}] not available yet, retry after {exc.until.isoformat()}")
        return _EXIT_DEFERRED
    if isinstance(exc, Cancelled):
        _say(f"[{stage}] cancelled")
        return _EXIT_INTERRUPTED
    if isinstance(exc, ControlFlow):
        _say(f"[{stage}] failed: BUG: unexpected {type(exc).__name__}")
    else:
        _say(f"[{stage}] failed: {classify(exc).value}: {_describe(exc)}")
    if verbose:
        _say(format_error(exc).rstrip("\n"))
    return _EXIT_FAILED


def _progress_detail(conn: psycopg.Connection[Any], settings: Settings, video_id: str) -> str:
    transcript = get_best_transcript(conn, video_id)
    if transcript is None:
        conn.rollback()
        return "no transcript yet"
    chunks = get_chunks(conn, transcript.id, chunk_strategy(settings.CHUNK_SEC, settings.OVERLAP_SEC))
    conn.rollback()
    return f"transcript: {transcript.source}, {len(chunks)} chunks"


def _missing_tables(conn: psycopg.Connection[Any]) -> list[str]:
    rows = conn.execute(
        "SELECT t, to_regclass(t) IS NULL FROM unnest(%s::text[]) AS t",
        (list(_REQUIRED_TABLES),),
    ).fetchall()
    conn.rollback()
    return [str(name) for name, missing in rows if missing]


def _schema_missing(missing: Sequence[str]) -> int:
    _say(
        f"the database schema is not ready (missing: {', '.join(missing)}); "
        "apply the migrations with: docker compose run --rm migrate"
    )
    return _EXIT_FAILED


def _transcript_source(conn: psycopg.Connection[Any], analysis: Analysis) -> str:
    for source in _TRANSCRIPT_SOURCES:
        transcript = get_transcript(conn, analysis.video_id, source)
        if transcript is not None and transcript.id == analysis.transcript_id:
            conn.rollback()
            return source
    conn.rollback()
    return "unknown"


def _print_result(conn: psycopg.Connection[Any], video_id: str) -> int:
    analysis = latest_analysis(conn, video_id)
    meta = get_video_meta(conn, video_id)
    conn.rollback()
    if analysis is None or meta is None:
        _say(
            "[pipeline] finished without an analysis: the video has no usable transcript "
            "(no captions, or nothing was said)"
        )
        return _EXIT_FAILED
    text = format_result(
        title=meta.title,
        video_id=video_id,
        transcript_source=_transcript_source(conn, analysis),
        analysis=analysis,
    )
    sys.stdout.write(text)
    sys.stdout.flush()
    return 0


def _settings_from_environment() -> Settings | None:
    try:
        return get_settings()
    except ValidationError as exc:
        # Field names only: the invalid value may be a DSN with a password.
        fields = sorted({str(error["loc"][0]) for error in exc.errors() if error["loc"]})
        _say(f"invalid configuration: check {', '.join(fields) or 'the environment'}")
        return None


def main(argv: Sequence[str] | None = None, *, deps: Deps | None = None) -> int:
    """Run the CLI and return its exit code. ``deps`` lets tests inject fake ports."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else _EXIT_USAGE)

    if args.command == "eval":
        # Lazy: services.eval imports this module.
        from services.eval.cli import eval_main

        try:
            return eval_main(args, deps)
        except KeyboardInterrupt:
            _say("interrupted (Ctrl-C)")
            return _EXIT_INTERRUPTED

    try:
        video_id = parse_video_ref(args.video)
    except ValueError as exc:
        _say(f"ytdigest: error: {_one_line(str(exc))}")
        return _EXIT_USAGE

    try:
        return _run(video_id, deps, verbose=args.verbose)
    except KeyboardInterrupt:
        _say("interrupted (Ctrl-C)")
        return _EXIT_INTERRUPTED


def _run(video_id: str, deps: Deps | None, *, verbose: bool) -> int:
    if deps is None:
        settings = _settings_from_environment()
        if settings is None:
            return _EXIT_FAILED
        configure_logging(settings, stream=sys.stderr)
        try:
            deps = default_deps(settings)
        except (SummarizerConfigError, CookieFileError) as exc:
            _say(f"invalid configuration: {_one_line(str(exc))}")
            return _EXIT_FAILED
        except Exception as exc:  # noqa: BLE001 - anything else that cannot be built
            return _report_failure("setup", exc, verbose=verbose)
    else:
        configure_logging(deps.settings, stream=sys.stderr)
    settings = deps.settings

    try:
        conn = deps.connect()
    except (psycopg.Error, OSError):
        # No detail: a connection error can carry the DSN, and with it the password.
        _say(
            "cannot connect to the database: check DATABASE_URL and that Postgres is running"
        )
        return _EXIT_FAILED
    try:
        try:
            missing = _missing_tables(conn)
        except psycopg.Error:
            _say("cannot query the database: check DATABASE_URL and that Postgres is running")
            return _EXIT_FAILED
        if missing:
            return _schema_missing(missing)

        queue = InProcessQueue()
        queue.enqueue("ingest", video_id, priority=PRIORITY)

        printer = ProgressPrinter()

        def started(job: Job) -> None:
            printer.begin()
            _say(f"[{job.kind}] started")

        def finished(job: Job) -> None:
            _say(f"[{job.kind}] finished - {_progress_detail(conn, settings, video_id)}")

        try:
            run_queue(
                queue,
                build_inprocess_handlers(deps, queue, conn),
                video_id,
                on_start=started,
                on_finish=finished,
                progress=printer,
            )
        except PipelineError as err:
            if isinstance(err.cause, psycopg.errors.UndefinedTable):
                return _schema_missing(["(a table the pipeline needs)"])
            return _report_failure(err.stage, err.cause, verbose=verbose)
        return _print_result(conn, video_id)
    finally:
        conn.close()
