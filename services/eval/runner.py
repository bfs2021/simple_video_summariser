"""``eval run``: analyse every golden-set video with one summarizer, model and prompt (issue #62).

The analysis is done by the existing analyze handler, called in process with a
synthetic ``Job``. Nothing here enqueues, ingests, transcribes or downloads.
The rows written are exactly what the analyzer worker would write.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import psycopg
import structlog

from common.chunking import chunk_strategy
from common.config import Settings
from common.errors import ControlFlow, JobError, classify
from common.models import Summarizer
from common.queue import PRIORITY_INTERACTIVE, Job, analyze_dedupe_key
from common.repo.analyses import analysis_exists, analysis_run_for
from common.repo.transcripts import get_best_transcript, get_chunks
from common.repo.videos import get_video_meta
from common.worker import JobContext
from services.cli.output import sanitize_text

Status = Literal["analyzed", "skipped", "missing", "failed"]
Handler = Callable[[Job, JobContext], None]
_MAX_MESSAGE_CHARS = 500


@dataclass(frozen=True, slots=True)
class VideoOutcome:
    """What happened to one video; ``text`` is the part of the stderr line after the ID."""

    status: Status
    text: str


def apply_overrides(
    settings: Settings,
    *,
    summarizer: Literal["ollama", "anthropic"] | None,
    model: str | None,
    prompt_version: str | None,
) -> Settings:
    """``settings`` with the command-line overrides applied and re-validated.

    ``--model`` sets the model of whichever summarizer is effective. Validation
    runs again over the whole set (``model_copy`` would skip it), so
    ``--prompt-version "../x"`` is refused by the same rule as ``PROMPT_VERSION``.
    Raises ``pydantic.ValidationError``.
    """
    updates: dict[str, Any] = {}
    if summarizer is not None:
        updates["SUMMARIZER"] = summarizer
    if prompt_version is not None:
        updates["PROMPT_VERSION"] = prompt_version
    effective = updates.get("SUMMARIZER", settings.SUMMARIZER)
    if model is not None:
        updates["OLLAMA_MODEL" if effective == "ollama" else "ANTHROPIC_MODEL"] = model
    return type(settings).model_validate({**settings.model_dump(), **updates})


def _one_line(text: str) -> str:
    line = " ".join(sanitize_text(text).split())
    return line if len(line) <= _MAX_MESSAGE_CHARS else line[: _MAX_MESSAGE_CHARS - 3] + "..."


def describe_failure(exc: BaseException) -> str:
    """``CLASS: message`` for the status line; the message is sanitized and capped."""
    if isinstance(exc, ControlFlow):
        return f"BUG: unexpected {type(exc).__name__}"
    text = str(exc)
    if isinstance(exc, JobError):
        message = text or type(exc).__name__
    else:
        message = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    return f"{classify(exc).value}: {_one_line(message)}"


def synthetic_job(video_id: str, settings: Settings, summarizer: Summarizer) -> Job:
    """The job the handler is called with. It is never written to ``jobs``."""
    now = datetime.now(UTC)
    return Job(
        id=0,
        video_id=video_id,
        kind="analyze",
        dedupe_key=analyze_dedupe_key(settings.PROMPT_VERSION, summarizer.name),
        state="running",
        priority=PRIORITY_INTERACTIVE,
        payload={},
        attempts=1,
        last_error=None,
        error_class=None,
        run_after=now,
        locked_by="eval",
        locked_at=now,
        heartbeat_at=now,
        finished_at=None,
        created_at=now,
    )


def evaluate_video(
    conn: psycopg.Connection[Any],
    handler: Handler | None,
    settings: Settings,
    summarizer: Summarizer,
    video_id: str,
    *,
    dry_run: bool,
) -> VideoOutcome:
    """Analyse (or, with ``dry_run``, only inspect) one video. Never raises ``Exception``.

    ``KeyboardInterrupt`` passes through. ``handler`` may be ``None`` only for a dry run.
    """
    strategy = chunk_strategy(settings.CHUNK_SEC, settings.OVERLAP_SEC)
    model = summarizer.model
    try:
        meta = get_video_meta(conn, video_id)
        if meta is None:
            conn.rollback()
            return VideoOutcome("missing", "missing: not in database")
        transcript = get_best_transcript(conn, video_id)
        if transcript is None:
            conn.rollback()
            return VideoOutcome("missing", "missing: no transcript")
        exists = analysis_exists(
            conn, video_id, transcript.id, strategy, model, settings.PROMPT_VERSION
        )
        chunks = get_chunks(conn, transcript.id, strategy)
        conn.rollback()
        if dry_run:
            title = _one_line(meta.title)
            chunked = f"{len(chunks)} chunks" if chunks else "not chunked yet"
            verdict = "would skip" if exists else "would analyze"
            return VideoOutcome(
                "skipped" if exists else "analyzed",
                f"{verdict} (title: {title!r}; transcript: {transcript.source}; chunks: {chunked})",
            )
        if exists:
            return VideoOutcome("skipped", "skipped: already analyzed")
        assert handler is not None
        handler(
            synthetic_job(video_id, settings, summarizer),
            JobContext(structlog.get_logger("eval")),
        )
        run = analysis_run_for(conn, video_id, model, settings.PROMPT_VERSION)
        chunk_count = len(get_chunks(conn, transcript.id, strategy))
        conn.rollback()
        if run is None:  # the handler returned without writing: not expected
            return VideoOutcome("failed", "failed: BUG: the analysis was not stored")
        a = run.analysis
        return VideoOutcome(
            "analyzed",
            f"analyzed ({chunk_count} chunks, {a.input_tokens + a.output_tokens} tokens, "
            f"{a.duration_ms} ms)",
        )
    except Exception as exc:  # noqa: BLE001 - one video's failure must not stop the rest
        try:
            conn.rollback()
        except psycopg.Error:
            pass
        return VideoOutcome("failed", f"failed: {describe_failure(exc)}")


def tally(outcomes: Sequence[VideoOutcome], *, dry_run: bool) -> str:
    """The final summary line."""
    count = {s: sum(1 for o in outcomes if o.status == s) for s in ("analyzed", "skipped", "missing", "failed")}
    if dry_run:
        return (
            f"dry run: would analyze {count['analyzed']}, would skip {count['skipped']}, "
            f"missing {count['missing']}"
        )
    return (
        f"summary: analyzed {count['analyzed']}, skipped {count['skipped']}, "
        f"missing {count['missing']}, failed {count['failed']}"
    )
