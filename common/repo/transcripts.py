"""Transcript and chunk repository functions (issue #14, architecture.md §6, §7.3)."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from common.models import (
    Chunk,
    IndexedSegment,
    Segment,
    Transcript,
    TranscriptInfo,
    TranscriptPage,
)
from common.repo._hygiene import clean_json, clean_text

_SOURCES = {"youtube_manual", "youtube_auto", "whisper"}
_SPEAKER_SOURCES = {"subtitle_labels", "none"}


def save_transcript(
    conn: psycopg.Connection[Any],
    video_id: str,
    source: str,
    language: str | None,
    speaker_source: str,
    segments: Sequence[Segment],
    engine_meta: dict[str, Any] | None,
) -> int:
    """Persist a transcript for ``video_id`` from ``source``, returning its id.

    ``full_text`` is built by joining the segments' text with a single
    space. If ``(video_id, source)`` already has a transcript, the
    **existing** row wins - nothing is changed and its id is returned, so
    a retried job can never overwrite a transcript an analysis already
    points at.

    ``source`` must be ``youtube_manual``, ``youtube_auto`` or ``whisper``;
    ``speaker_source`` must be ``subtitle_labels`` or ``none``. Anything
    else raises ``ValueError``.
    """
    if source not in _SOURCES:
        raise ValueError(f"source must be one of {sorted(_SOURCES)}, got {source!r}")
    if speaker_source not in _SPEAKER_SOURCES:
        raise ValueError(
            f"speaker_source must be one of {sorted(_SPEAKER_SOURCES)}, got {speaker_source!r}"
        )

    full_text = clean_text(" ".join(seg.text for seg in segments)) or ""
    segments_json = [clean_json(_segment_to_json(seg)) for seg in segments]

    row = conn.execute(
        """
        INSERT INTO transcripts (video_id, source, language, speaker_source,
                                  segments, full_text, engine_meta)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (video_id, source) DO NOTHING
        RETURNING id
        """,
        (
            video_id,
            source,
            language,
            speaker_source,
            Jsonb(segments_json),
            full_text,
            Jsonb(clean_json(engine_meta)) if engine_meta is not None else None,
        ),
    ).fetchone()
    if row is not None:
        return int(row[0])

    # ON CONFLICT DO NOTHING skipped the insert only because the row
    # already exists - first write wins (issue #14).
    existing = conn.execute(
        "SELECT id FROM transcripts WHERE video_id = %s AND source = %s",
        (video_id, source),
    ).fetchone()
    assert existing is not None
    return int(existing[0])


def get_best_transcript(conn: psycopg.Connection[Any], video_id: str) -> Transcript | None:
    """The transcript for ``video_id`` ranked best by ``transcript_rank()``.

    Manual beats whisper beats auto-captions (D2). ``None`` if the video
    has no transcript at all.
    """
    row = conn.execute(
        """
        SELECT id, video_id, source, language, speaker_source, segments,
               full_text, engine_meta, created_at
        FROM transcripts
        WHERE video_id = %s
        ORDER BY transcript_rank(source)
        LIMIT 1
        """,
        (video_id,),
    ).fetchone()
    if row is None:
        return None
    return _transcript(row)


def save_chunks(
    conn: psycopg.Connection[Any],
    transcript_id: int,
    strategy: str,
    chunks: Sequence[Chunk],
) -> None:
    """Insert ``chunks`` for ``transcript_id`` under ``strategy``.

    Uses ``ON CONFLICT DO NOTHING`` (architecture.md §7.3), so saving the
    same chunks twice never fails.
    """
    for chunk in chunks:
        conn.execute(
            """
            INSERT INTO transcript_chunks
                (transcript_id, seq, start_sec, end_sec, text, chunk_strategy)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (transcript_id, chunk_strategy, seq) DO NOTHING
            """,
            (
                transcript_id,
                chunk.seq,
                chunk.start_sec,
                chunk.end_sec,
                clean_text(chunk.text),
                strategy,
            ),
        )


def get_chunks(conn: psycopg.Connection[Any], transcript_id: int, strategy: str) -> list[Chunk]:
    """Chunks for ``transcript_id`` under ``strategy``, ordered by ``seq``.

    An empty list when that strategy has none saved.
    """
    rows = conn.execute(
        """
        SELECT id, transcript_id, seq, start_sec, end_sec, text, chunk_strategy
        FROM transcript_chunks
        WHERE transcript_id = %s AND chunk_strategy = %s
        ORDER BY seq
        """,
        (transcript_id, strategy),
    ).fetchall()
    return [_chunk(row) for row in rows]


def _segment_to_json(seg: Segment) -> dict[str, Any]:
    data: dict[str, Any] = {"start": seg.start, "end": seg.end, "text": seg.text}
    if seg.speaker is not None:
        data["speaker"] = seg.speaker
    return data


def _segment_from_json(data: dict[str, Any]) -> Segment:
    return Segment(
        start=float(data["start"]),
        end=float(data["end"]),
        text=data["text"],
        speaker=data.get("speaker"),
    )


def _transcript(row: tuple[Any, ...]) -> Transcript:
    (
        transcript_id,
        video_id,
        source,
        language,
        speaker_source,
        segments,
        full_text,
        engine_meta,
        created_at,
    ) = row
    return Transcript(
        id=int(transcript_id),
        video_id=video_id,
        source=source,
        language=language,
        speaker_source=speaker_source,
        segments=tuple(_segment_from_json(item) for item in segments),
        full_text=full_text,
        engine_meta=engine_meta,
        created_at=created_at,
    )


def _chunk(row: tuple[Any, ...]) -> Chunk:
    chunk_id, transcript_id, seq, start_sec, end_sec, text, chunk_strategy = row
    return Chunk(
        seq=seq,
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        text=text,
        chunk_strategy=chunk_strategy,
        transcript_id=int(transcript_id),
        id=int(chunk_id),
    )


def get_transcript(conn: psycopg.Connection[Any], video_id: str, source: str) -> Transcript | None:
    """The transcript of ``video_id`` from exactly ``source``, or ``None`` (issue #29).

    Unlike ``get_best_transcript`` this does not rank: a ``whisper`` transcript is
    found even when a better (manual) one exists.
    """
    row = conn.execute(
        """
        SELECT id, video_id, source, language, speaker_source, segments,
               full_text, engine_meta, created_at
        FROM transcripts
        WHERE video_id = %s AND source = %s
        """,
        (video_id, source),
    ).fetchone()
    return None if row is None else _transcript(row)


RTF_SAMPLE_SIZE = 20
_MIN_RTF = Decimal("1e-300")
_MAX_RTF = Decimal("1e300")


def median_whisper_rtf(conn: psycopg.Connection[Any]) -> tuple[float | None, int]:
    """Median ``engine_meta.rtf`` of the newest usable whisper transcripts.

    Returns ``(median, samples)``. Only the newest ``RTF_SAMPLE_SIZE``
    ``whisper`` transcripts whose ``rtf`` is a positive finite JSON number
    count; with none, the result is ``(None, 0)``.
    """
    # The CASE guards each cast, since Postgres may evaluate WHERE terms in
    # any order. The bounds keep the float8 conversion finite and non-zero.
    rows = conn.execute(
        """
        SELECT rtf FROM (
            SELECT id, created_at,
                   CASE WHEN jsonb_typeof(engine_meta) = 'object'
                         AND jsonb_typeof(engine_meta->'rtf') = 'number'
                        THEN (engine_meta->>'rtf')::numeric
                   END AS rtf
            FROM transcripts
            WHERE source = 'whisper'
        ) t
        WHERE rtf > %s AND rtf < %s
        ORDER BY created_at DESC, id DESC
        LIMIT %s
        """,
        (_MIN_RTF, _MAX_RTF, RTF_SAMPLE_SIZE),
    ).fetchall()
    values = [float(row[0]) for row in rows]
    if not values:
        return None, 0
    return statistics.median(values), len(values)


CHARS_SAMPLE_SIZE = 20


def median_chars_per_audio_sec(conn: psycopg.Connection[Any]) -> tuple[float | None, int]:
    """Median transcript characters per second of audio, newest 20 (#117).

    ``length(full_text) / videos.duration_sec`` over transcripts of any
    ``source`` whose video has a positive ``duration_sec`` and whose text is
    not empty. Returns ``(median, samples)``; with no usable row ``(None, 0)``.
    A read only: no lock, no write.
    """
    rows = conn.execute(
        """
        SELECT length(t.full_text)::float8 / v.duration_sec::float8
        FROM transcripts t JOIN videos v ON v.video_id = t.video_id
        WHERE v.duration_sec > 0 AND length(t.full_text) > 0
        ORDER BY t.created_at DESC, t.id DESC
        LIMIT %s
        """,
        (CHARS_SAMPLE_SIZE,),
    ).fetchall()
    values = [float(row[0]) for row in rows if math.isfinite(float(row[0])) and row[0] > 0]
    if not values:
        return None, 0
    return statistics.median(values), len(values)


_BEST_INFO = """
    SELECT id, source, language, speaker_source, jsonb_array_length(segments)
    FROM transcripts
    WHERE video_id = %s
    ORDER BY transcript_rank(source), id
    LIMIT 1
"""


def best_transcript_info(conn: psycopg.Connection[Any], video_id: str) -> TranscriptInfo | None:
    """The best transcript's identity and segment count, without its text (#42).

    Ranked like ``get_best_transcript`` (D2: manual > whisper > auto).
    """
    row = conn.execute(_BEST_INFO, (video_id,)).fetchone()
    if row is None:
        return None
    transcript_id, source, language, speaker_source, count = row
    return TranscriptInfo(int(transcript_id), source, language, speaker_source, int(count))


def best_transcript_page(
    conn: psycopg.Connection[Any], video_id: str, *, offset: int, limit: int
) -> TranscriptPage | None:
    """One page of the best transcript's segments, sliced in SQL (#42).

    ``None`` when the video has no transcript. Neither the whole ``segments``
    array nor ``full_text`` is ever fetched.
    """
    info = best_transcript_info(conn, video_id)
    if info is None:
        return None
    rows = conn.execute(
        """
        SELECT s.ord - 1, (s.seg->>'start')::float8, (s.seg->>'end')::float8,
               s.seg->>'text', s.seg->>'speaker'
        FROM transcripts t,
             jsonb_array_elements(t.segments) WITH ORDINALITY AS s(seg, ord)
        WHERE t.id = %s
        ORDER BY s.ord
        OFFSET %s LIMIT %s
        """,
        (info.id, offset, limit),
    ).fetchall()
    segments = tuple(
        IndexedSegment(int(i), float(start), float(end), text, speaker)
        for i, start, end, text, speaker in rows
    )
    return TranscriptPage(video_id=video_id, info=info, segments=segments)


def latest_whisper_rtf(conn: psycopg.Connection[Any]) -> float | None:
    """``engine_meta.rtf`` of the newest ``whisper`` transcript with a usable ``rtf`` (#60).

    Usable means a positive, finite JSON number; missing, null, non-numeric,
    zero or negative values are skipped. ``None`` when no transcript qualifies.
    """
    row = conn.execute(
        """
        SELECT rtf FROM (
            SELECT id, created_at,
                   CASE WHEN jsonb_typeof(engine_meta) = 'object'
                         AND jsonb_typeof(engine_meta->'rtf') = 'number'
                        THEN (engine_meta->>'rtf')::numeric
                   END AS rtf
            FROM transcripts
            WHERE source = 'whisper'
        ) t
        WHERE rtf > %s AND rtf < %s
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (_MIN_RTF, _MAX_RTF),
    ).fetchone()
    return None if row is None else float(row[0])
