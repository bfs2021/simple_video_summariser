"""The ``transcribe`` job handler (issue #29, architecture.md §7.2).

Gets normalized audio, registers it in ``media``, runs speech-to-text, saves a
``whisper`` transcript with its chunks and enqueues ``analyze``. Every
collaborator is injected, so the handler imports neither ``faster_whisper`` nor
an adapter and stays importable in the backend image. It holds no SQL: the
database is reached through ``common.repo`` and the ``JobQueue`` port.

Retry safety, step by step:

* A whisper transcript already saved (a crash between commit and enqueue) skips
  audio and speech-to-text; the handler only makes sure chunks exist and
  enqueues ``analyze``.
* The ``media`` row is committed before speech-to-text, so retention (#36) can
  see the audio in use, and no transaction is open while the model runs.
* The transcript and its chunks are one transaction.
* With ``AUDIO_KEEP=0`` the file is deleted first and the row second, on every
  exit path once audio was fetched.

The attempt cap is the queue's business (#11), not the handler's.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psycopg

from adapters.youtube.metadata import watch_url
from common.chunking import chunk_strategy
from common.config import Settings
from common.errors import BugError, PermanentSourceError
from common.models import AudioRef, AudioSource, Transcriber
from common.queue import Job, JobQueue
from common.repo.media import delete_media, register_media
from common.repo.transcripts import get_chunks, get_transcript, save_chunks
from common.repo.videos import mark_unavailable, video_exists
from common.worker import JobContext
from services.transcriber.persist import (
    Chunker,
    enqueue_analyze,
    save_transcript_with_chunks,
)

_PROGRESS_STEPS = 10
#: Wall seconds after which a quiet transcription prints another line (issue #96).
HEARTBEAT_SEC = 30.0


def _validate(job: Job) -> str:
    try:
        watch_url(job.video_id)  # #16's rule: 11 chars of [A-Za-z0-9_-]
    except ValueError as exc:
        raise BugError(str(exc)) from exc
    return job.video_id


def progress_reporter(
    ctx: JobContext, duration_sec: float, *, clock: Callable[[], float] = time.monotonic
) -> Callable[[float], None]:
    """The ``on_progress`` callback: a cancellation check, plus ``ctx.progress`` lines.

    ``check_cancelled`` runs on every call, before anything is reported. A line is
    reported when progress crosses each 10% step, and also when ``HEARTBEAT_SEC`` of
    wall time have passed since the last line and the audio position advanced: at most
    ``_PROGRESS_STEPS + 1 + elapsed / HEARTBEAT_SEC`` lines, one per call at most.
    With an unknown or non-positive ``duration_sec`` only the time rule applies.
    """
    known = math.isfinite(duration_sec) and duration_sec > 0
    last_step = -1
    last_done = 0.0
    last_line_at = clock()

    def on_progress(done_sec: float) -> None:
        nonlocal last_step, last_done, last_line_at
        ctx.check_cancelled()
        if not math.isfinite(done_sec):
            return
        now = clock()
        step = -1
        if known:
            fraction = min(max(done_sec / duration_sec, 0.0), 1.0)
            step = int(fraction * _PROGRESS_STEPS)
        crossed = known and step > last_step
        waited = done_sec > last_done and now - last_line_at >= HEARTBEAT_SEC
        if not (crossed or waited):
            return
        last_step = max(last_step, step)
        last_done = done_sec
        last_line_at = now
        if known:
            percent = int(fraction * 100 + 1e-9)
            ctx.progress(
                "transcribe",
                done_sec,
                duration_sec,
                done_sec=round(done_sec, 1),
                duration_sec=round(duration_sec, 1),
                percent=percent,
            )
        else:
            ctx.progress("transcribe", done_sec, 0.0, done_sec=round(done_sec, 1))

    return on_progress


def _language(job: Job, ctx: JobContext) -> str | None:
    """``payload["language"]`` when it is a non-empty string, else ``None`` (auto-detect)."""
    if "language" not in job.payload:
        return None
    value = job.payload["language"]
    if isinstance(value, str) and value.strip():
        return value
    ctx.logger.warning(
        "ignoring invalid language in the job payload; the language will be detected",
        language=repr(value)[:80],
    )
    return None


def _unlink_under(audio_dir: Path, rel_path: str, ctx: JobContext) -> None:
    """Delete ``audio_dir / rel_path``; a missing file is fine, a path outside is refused."""
    root = audio_dir.resolve()
    target = (audio_dir / rel_path).resolve()
    if not target.is_relative_to(root) or target == root:
        ctx.logger.error("refusing to delete audio outside AUDIO_DIR", rel_path=rel_path)
        return
    target.unlink(missing_ok=True)


def make_transcribe_handler(
    *,
    connect: Callable[[], psycopg.Connection[Any]],
    queue: JobQueue,
    audio_source: AudioSource,
    transcriber: Transcriber,
    chunker: Chunker,
    settings: Settings,
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[Job, JobContext], None]:
    """Build the handler. ``connect`` opens one connection per job, closed when it ends."""
    strategy = chunk_strategy(settings.CHUNK_SEC, settings.OVERLAP_SEC)

    def ensure_chunks_and_hand_off(
        conn: psycopg.Connection[Any], job: Job, ctx: JobContext
    ) -> bool:
        """Retry path: a whisper transcript exists. Returns whether it was handled."""
        existing = get_transcript(conn, job.video_id, "whisper")
        chunks_exist = existing is not None and bool(get_chunks(conn, existing.id, strategy))
        conn.commit()
        if existing is None:
            return False
        if not chunks_exist and existing.segments:
            chunks = chunker(
                existing.segments,
                chunk_sec=settings.CHUNK_SEC,
                overlap_sec=settings.OVERLAP_SEC,
            )
            with conn.transaction():
                save_chunks(conn, existing.id, strategy, chunks)
            chunks_exist = bool(chunks)
        hand_off(job, ctx, has_chunks=chunks_exist)
        return True

    def hand_off(job: Job, ctx: JobContext, *, has_chunks: bool) -> None:
        if not has_chunks:
            ctx.logger.warning(
                "transcript has no speech: nothing to analyze, no analyze job enqueued"
            )
            return
        if enqueue_analyze(queue, settings, job.video_id, priority=job.priority) is None:
            ctx.logger.info("analyze job already pending or running")

    def process(conn: psycopg.Connection[Any], job: Job, ctx: JobContext, audio: AudioRef) -> None:
        register_media(conn, job.video_id, audio.rel_path, audio.bytes, settings.AUDIO_TTL_DAYS)
        conn.commit()  # before speech-to-text: retention must see this audio
        ctx.check_cancelled()

        result = transcriber.transcribe(
            audio,
            language=_language(job, ctx),
            on_progress=progress_reporter(ctx, audio.duration_sec, clock=clock),
        )

        save_transcript_with_chunks(
            conn,
            settings,
            job.video_id,
            source="whisper",
            language=result.language,
            speaker_source="none",
            segments=result.segments,
            engine_meta=result.engine_meta,
            chunker=chunker,
        )
        hand_off(job, ctx, has_chunks=bool(result.segments))

    def discard_audio(conn: psycopg.Connection[Any], job: Job, ctx: JobContext, audio: AudioRef) -> None:
        # File first, row second: a crash in between leaves a row pointing at a
        # missing file (visible to #36), never an orphan file nobody can see.
        _unlink_under(settings.AUDIO_DIR, audio.rel_path, ctx)
        conn.rollback()
        delete_media(conn, job.video_id)
        conn.commit()

    def handler(job: Job, ctx: JobContext) -> None:
        video_id = _validate(job)
        with connect() as conn:
            found = video_exists(conn, video_id)
            conn.commit()
            if not found:
                raise BugError(f"no videos row for {video_id!r}; ingest must run first")
            if ensure_chunks_and_hand_off(conn, job, ctx):
                return

            try:
                audio = audio_source.fetch_normalized(video_id, settings.AUDIO_DIR)
            except PermanentSourceError as exc:
                mark_unavailable(conn, video_id, exc.reason.value)
                conn.commit()
                raise

            try:
                process(conn, job, ctx, audio)
            except BaseException:
                if not settings.AUDIO_KEEP:
                    try:
                        discard_audio(conn, job, ctx, audio)
                    except Exception:
                        ctx.logger.warning("could not discard audio after a failure", exc_info=True)
                raise
            if not settings.AUDIO_KEEP:
                discard_audio(conn, job, ctx, audio)

    return handler
