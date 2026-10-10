"""The ``ingest`` job handler (issue #28, architecture.md §7.1).

Fetches metadata, upserts the ``videos`` row, then picks exactly one
transcript path (see ``choose_transcript_path``). The handler holds no state
between jobs and writes no SQL: the database is reached through
``common.repo`` and the ``JobQueue`` port.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any, Protocol

import psycopg

from adapters.youtube.subtitles import SubtitleAvailability, speaker_source
from common.config import Settings
from common.errors import PermanentSourceError, ToolFailureError
from common.models import MetadataSource, Segment, VideoMeta
from common.queue import Job, JobQueue
from common.repo.jobs import latest_job_state
from common.repo.transcripts import get_best_transcript
from common.repo.videos import clear_unavailable, record_unavailable, upsert_video
from common.worker import JobContext
from common.youtube_refs import VIDEO_ID_PATTERN
from services.transcriber.persist import enqueue_analyze, save_transcript_with_chunks

_VIDEO_ID = VIDEO_ID_PATTERN
_ORIGINS = frozenset({"adhoc", "rss", "backfill"})
_STRONG_SOURCES = frozenset({"youtube_manual", "whisper"})


class SubtitleSource(Protocol):
    """The subtitle port (architecture.md §3)."""

    def available(self, meta: VideoMeta) -> SubtitleAvailability: ...

    def fetch(self, video_id: str, lang: str, kind: str) -> list[Segment]: ...


class TranscriptPath(StrEnum):
    MANUAL = "manual"
    TRANSCRIBE = "transcribe"
    AUTO_FALLBACK = "auto_fallback"
    HAVE_TRANSCRIPT = "have_transcript"
    NONE = "none"


def choose_transcript_path(
    availability: SubtitleAvailability,
    *,
    prefer_whisper: bool,
    auto_caption_fallback: bool,
    existing_source: str | None,
    transcribe_state: str | None,
) -> TranscriptPath:
    """The ingest decision table (issue #28), as a pure function."""
    if existing_source in _STRONG_SOURCES:
        return TranscriptPath.HAVE_TRANSCRIPT
    if availability.manual is not None and not prefer_whisper:
        return TranscriptPath.MANUAL
    if transcribe_state != "dead":
        return TranscriptPath.TRANSCRIBE
    if availability.auto is not None and auto_caption_fallback:
        return TranscriptPath.AUTO_FALLBACK
    return TranscriptPath.NONE


def _validate(job: Job) -> str:
    video_id = job.video_id
    if not isinstance(video_id, str) or _VIDEO_ID.fullmatch(video_id) is None:
        raise ValueError(f"invalid video id: {video_id!r}")
    origin = job.payload.get("origin", "adhoc")
    if not isinstance(origin, str) or origin not in _ORIGINS:
        raise ValueError(f"invalid origin {origin!r}; expected one of {sorted(_ORIGINS)}")
    return origin


def make_ingest_handler(
    *,
    conn: psycopg.Connection[Any],
    queue: JobQueue,
    metadata: MetadataSource,
    subtitles: SubtitleSource,
    settings: Settings,
) -> Callable[[Job, JobContext], None]:
    def handler(job: Job, ctx: JobContext) -> None:
        origin = _validate(job)
        video_id = job.video_id

        def permanent(exc: PermanentSourceError) -> None:
            record_unavailable(conn, video_id, exc.reason.value, origin=origin)
            conn.commit()

        try:
            meta = metadata.fetch(video_id)
        except PermanentSourceError as exc:
            permanent(exc)
            raise
        if meta.video_id != video_id:
            raise ToolFailureError(
                f"metadata is for {meta.video_id!r}, but the job is for {video_id!r}"
            )
        ctx.check_cancelled()

        upsert_video(conn, meta, origin)
        clear_unavailable(conn, video_id)
        conn.commit()

        def download(lang: str, kind: str) -> list[Segment]:
            ctx.check_cancelled()
            try:
                return subtitles.fetch(video_id, lang, kind)
            except PermanentSourceError as exc:
                permanent(exc)
                raise

        available = subtitles.available(meta)
        best = get_best_transcript(conn, video_id)
        existing_source = best.source if best is not None else None
        transcribe_state = latest_job_state(conn, video_id, "transcribe")
        conn.commit()  # end the read transaction: enqueue() opens its own

        def choose(availability: SubtitleAvailability) -> TranscriptPath:
            return choose_transcript_path(
                availability,
                prefer_whisper=settings.PREFER_WHISPER,
                auto_caption_fallback=settings.AUTO_CAPTION_FALLBACK,
                existing_source=existing_source,
                transcribe_state=transcribe_state,
            )

        path = choose(available)
        saved_source: str | None = None
        segments: list[Segment] = []
        kind = ""
        lang = ""

        if path is TranscriptPath.MANUAL:
            assert available.manual is not None
            kind, lang = "manual", available.manual
            segments = download(lang, kind)
            if not segments:
                path = choose(SubtitleAvailability(manual=None, auto=available.auto))
        if path is TranscriptPath.AUTO_FALLBACK:
            assert available.auto is not None
            kind, lang = "auto", available.auto
            segments = download(lang, kind)
            if not segments:
                path = TranscriptPath.NONE

        if segments:
            saved_source = "youtube_manual" if kind == "manual" else "youtube_auto"
            if kind == "auto":
                ctx.logger.warning(
                    "using auto-captions as a last resort: transcript is degraded",
                    track=lang,
                )
            save_transcript_with_chunks(
                conn,
                settings,
                video_id,
                source=saved_source,
                language=lang,
                speaker_source=speaker_source(segments),
                segments=segments,
                engine_meta={"track": lang, "kind": kind},
            )
            enqueue_analyze(queue, settings, video_id, priority=job.priority)
        elif path is TranscriptPath.HAVE_TRANSCRIPT:
            enqueue_analyze(queue, settings, video_id, priority=job.priority)
        elif path is TranscriptPath.TRANSCRIBE:
            # duration_sec lets the worker derive this job's runtime limit (#66).
            payload: dict[str, Any] = {}
            if meta.language:
                payload["language"] = meta.language
            if meta.duration_sec:
                payload["duration_sec"] = meta.duration_sec
            job_id = queue.enqueue(
                "transcribe", video_id, payload=payload or None, priority=job.priority
            )
            if job_id is None:
                ctx.logger.info("transcribe job already pending or running")
        elif path is TranscriptPath.NONE:
            ctx.logger.warning(
                "no transcript source left: transcription dead-lettered and "
                "auto-captions are unavailable or disabled"
            )

        summary: dict[str, Any] = {"transcript_path": path.value}
        if saved_source is not None:
            summary["transcript_source"] = saved_source
        ctx.logger.info("ingest finished", **summary)

    return handler
