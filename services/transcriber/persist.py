"""Persist helpers shared by the ``ingest`` and ``transcribe`` handlers (issues #28, #29)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import psycopg

from common.chunking import chunk_segments, chunk_strategy
from common.config import Settings
from common.models import Chunk, Segment
from common.queue import JobQueue, analyze_dedupe_key
from common.repo.transcripts import save_chunks, save_transcript


class Chunker(Protocol):
    """The chunker port: ``common.chunking.chunk_segments`` (#21) or a test double."""

    def __call__(
        self, segments: Sequence[Segment], *, chunk_sec: int, overlap_sec: int
    ) -> list[Chunk]: ...


def save_transcript_with_chunks(
    conn: psycopg.Connection[Any],
    settings: Settings,
    video_id: str,
    *,
    source: str,
    language: str | None,
    speaker_source: str,
    segments: Sequence[Segment],
    engine_meta: dict[str, Any] | None,
    chunker: Chunker | None = None,
) -> int:
    """Save a transcript and its chunks in one transaction, returning the transcript id.

    Any read transaction the caller left open is committed first, so the write
    below is a real transaction: a failure while saving chunks rolls the
    transcript back too, and a transcript is never left without its chunks.
    """
    chunks = (chunker or chunk_segments)(
        segments, chunk_sec=settings.CHUNK_SEC, overlap_sec=settings.OVERLAP_SEC
    )
    conn.commit()
    with conn.transaction():
        transcript_id = save_transcript(
            conn, video_id, source, language, speaker_source, segments, engine_meta
        )
        save_chunks(
            conn,
            transcript_id,
            chunk_strategy(settings.CHUNK_SEC, settings.OVERLAP_SEC),
            chunks,
        )
    return transcript_id


def enqueue_analyze(
    queue: JobQueue,
    settings: Settings,
    parent_job_id: int,
    video_id: str,
    *,
    priority: int,
) -> int | None:
    """Hand off to ``analyze`` under the configured dedupe key."""
    return queue.handoff(
        parent_job_id,
        "analyze",
        video_id,
        dedupe_key=analyze_dedupe_key(settings.PROMPT_VERSION, settings.SUMMARIZER),
        priority=priority,
    )
