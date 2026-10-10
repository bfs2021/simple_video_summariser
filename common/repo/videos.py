"""Video repository functions (issue #14, architecture.md §6)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg

from common.models import (
    ActiveJob,
    CatalogEntry,
    FeedEntry,
    JobFailure,
    VideoMeta,
    VideoPage,
    VideoSummary,
)
from common.repo._hygiene import clean_text

_UNAVAILABLE_REASONS = {"removed", "private", "geoblocked", "agegated"}


def upsert_video(conn: psycopg.Connection[Any], meta: VideoMeta, origin: str) -> None:
    """Insert ``meta`` as a video, or update its mutable fields on a later call.

    ``title``, ``duration_sec``, ``published_at`` and ``description`` are
    refreshed on every call, and a missing ``channel_id`` (a stub row from
    ``record_unavailable``, or a submitted video from
    ``insert_submitted_video``) is filled in; a known ``channel_id`` is never
    overwritten. ``origin`` and ``discovered_at`` are set only
    on the first insert - they record the video's *first* discovery, so a
    later call, even with a different ``origin``, never changes them.

    When ``meta.channel_id`` has no ``channels`` row yet, one is created
    with ``active = false`` so the foreign key holds without starting
    monitoring for an ad-hoc submission. An existing channel row, active or
    not, is left untouched.
    """
    conn.execute(
        """
        INSERT INTO channels (channel_id, active)
        VALUES (%s, false)
        ON CONFLICT (channel_id) DO NOTHING
        """,
        (meta.channel_id,),
    )
    conn.execute(
        """
        INSERT INTO videos (video_id, channel_id, title, duration_sec,
                             published_at, description, origin)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (video_id) DO UPDATE SET
            channel_id   = COALESCE(videos.channel_id, EXCLUDED.channel_id),
            title        = EXCLUDED.title,
            duration_sec = EXCLUDED.duration_sec,
            published_at = EXCLUDED.published_at,
            description  = EXCLUDED.description
        """,
        (
            meta.video_id,
            meta.channel_id,
            clean_text(meta.title),
            meta.duration_sec,
            meta.published_at,
            clean_text(meta.description),
            origin,
        ),
    )


def _check_reason(reason: str) -> None:
    if reason not in _UNAVAILABLE_REASONS:
        raise ValueError(
            f"unavailable reason must be one of {sorted(_UNAVAILABLE_REASONS)}, got {reason!r}"
        )


def mark_unavailable(conn: psycopg.Connection[Any], video_id: str, reason: str) -> None:
    """Record why ``video_id`` can no longer be fetched.

    ``reason`` must be one of ``removed``, ``private``, ``geoblocked`` or
    ``agegated``; anything else raises ``ValueError``.
    """
    _check_reason(reason)
    conn.execute(
        "UPDATE videos SET unavailable = %s WHERE video_id = %s",
        (reason, video_id),
    )


def get_discovered_at(conn: psycopg.Connection[Any], video_id: str) -> datetime | None:
    """When ``video_id`` was first discovered, or ``None`` if it has no row (#65)."""
    row = conn.execute(
        "SELECT discovered_at FROM videos WHERE video_id = %s", (video_id,)
    ).fetchone()
    return row[0] if row is not None else None


def clear_unavailable(conn: psycopg.Connection[Any], video_id: str) -> None:
    """Reset ``videos.unavailable`` to NULL, e.g. a private video went public.

    A no-op when ``video_id`` has no row.
    """
    conn.execute("UPDATE videos SET unavailable = NULL WHERE video_id = %s", (video_id,))


def record_unavailable(
    conn: psycopg.Connection[Any], video_id: str, reason: str, *, origin: str
) -> None:
    """Mark ``video_id`` unavailable, creating a stub row if it has none.

    The stub has a NULL ``channel_id`` and the given ``origin`` (first
    ingest of an already-removed video). An existing row keeps its
    ``origin`` and metadata; only ``unavailable`` changes. ``reason`` is
    validated like ``mark_unavailable``.
    """
    _check_reason(reason)
    conn.execute(
        """
        INSERT INTO videos (video_id, origin, unavailable)
        VALUES (%s, %s, %s)
        ON CONFLICT (video_id) DO UPDATE SET unavailable = EXCLUDED.unavailable
        """,
        (video_id, origin, reason),
    )


def insert_discovered_video(conn: psycopg.Connection[Any], entry: FeedEntry, origin: str) -> bool:
    """Insert the video ``entry`` lists, unless it already has a row.

    Returns whether a row was inserted. An existing row is left exactly as it is
    (``origin``, ``title``, ``description``, ``discovered_at`` and the rest), which is
    what makes polling the same feed twice harmless. ``upsert_video`` is the wrong tool
    here: it needs full metadata and overwrites ``title`` and ``description``.

    A missing ``channels`` row is created inactive, as ``upsert_video`` does.
    """
    conn.execute(
        """
        INSERT INTO channels (channel_id, active)
        VALUES (%s, false)
        ON CONFLICT (channel_id) DO NOTHING
        """,
        (entry.channel_id,),
    )
    cur = conn.execute(
        """
        INSERT INTO videos (video_id, channel_id, title, published_at, origin)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (video_id) DO NOTHING
        """,
        (entry.video_id, entry.channel_id, clean_text(entry.title), entry.published_at, origin),
    )
    return cur.rowcount == 1


def insert_submitted_video(conn: psycopg.Connection[Any], video_id: str) -> bool:
    """Insert a bare ``origin = 'adhoc'`` row for a submitted video (#40).

    Only ``video_id`` and ``origin`` are set; ``channel_id``, ``title`` and the
    rest stay NULL until ingest (#28) runs ``upsert_video``. Returns whether a
    row was inserted. An existing row (from RSS, backfill or an earlier
    submission) is left exactly as it is, including ``origin`` and
    ``discovered_at``. No ``channels`` row is created.
    """
    cur = conn.execute(
        """
        INSERT INTO videos (video_id, origin)
        VALUES (%s, 'adhoc')
        ON CONFLICT (video_id) DO NOTHING
        """,
        (video_id,),
    )
    return cur.rowcount == 1


def video_exists(conn: psycopg.Connection[Any], video_id: str) -> bool:
    """Whether ``videos`` has a row for ``video_id`` (issue #29)."""
    row = conn.execute(
        "SELECT EXISTS (SELECT 1 FROM videos WHERE video_id = %s)", (video_id,)
    ).fetchone()
    assert row is not None
    return bool(row[0])


def get_video_meta(conn: psycopg.Connection[Any], video_id: str) -> VideoMeta | None:
    """The stored fields of ``video_id`` as a ``VideoMeta``, or ``None`` without a row.

    Fields the ``videos`` table does not store (``language``, ``live_status``, the
    subtitle language lists) are ``None`` or ``()``. NULL text columns (a stub row
    from ``record_unavailable``) come back as empty strings.
    """
    row = conn.execute(
        """
        SELECT video_id, channel_id, title, description, duration_sec, published_at
        FROM videos
        WHERE video_id = %s
        """,
        (video_id,),
    ).fetchone()
    if row is None:
        return None
    found_id, channel_id, title, description, duration_sec, published_at = row
    return VideoMeta(
        video_id=found_id,
        channel_id=channel_id or "",
        title=title or "",
        description=description or "",
        duration_sec=duration_sec,
        published_at=published_at,
        language=None,
        live_status=None,
        manual_subtitle_langs=(),
        auto_caption_langs=(),
    )


@dataclass(frozen=True, slots=True)
class KnownVideo:
    """What a ``videos`` row says about its owner and date (issue #93)."""

    channel_id: str | None  # NULL for a stub row (a submission, an unavailable video)
    published_at: datetime | None  # NULL until ingest (#28) fills it in


def known_videos(
    conn: psycopg.Connection[Any], video_ids: Sequence[str]
) -> dict[str, KnownVideo]:
    """Map each of ``video_ids`` that already has a ``videos`` row to its facts.

    Any row counts, whatever its origin, job state or ``unavailable`` flag. One
    query for the whole list; unknown IDs are simply absent from the result.
    """
    if not video_ids:
        return {}
    rows = conn.execute(
        "SELECT video_id, channel_id, published_at FROM videos WHERE video_id = ANY(%s)",
        (list(video_ids),),
    ).fetchall()
    return {row[0]: KnownVideo(channel_id=row[1], published_at=row[2]) for row in rows}


def known_video_ids(conn: psycopg.Connection[Any], video_ids: Sequence[str]) -> set[str]:
    """Return the subset of ``video_ids`` that already has a ``videos`` row.

    The keys of ``known_videos``: one query for the whole list.
    """
    return set(known_videos(conn, video_ids))


def insert_catalog_video(
    conn: psycopg.Connection[Any], entry: CatalogEntry, *, channel_id: str, origin: str
) -> bool:
    """Insert a row for a catalog-listed video with the given ``origin`` (#41, #93).

    Shared by channel backfill (``origin='backfill'``) and RSS-gap
    reconciliation (``origin='rss'``). The listing has no publish date, so
    ``published_at`` stays NULL until ingest fills it in; that keeps a row made
    here from ever acting as a reconciliation anchor. The title is cleaned. A
    missing ``channels`` row is created inactive, as ``insert_discovered_video``
    does. An existing video row is left untouched. Returns whether this call
    inserted the row.
    """
    conn.execute(
        """
        INSERT INTO channels (channel_id, active)
        VALUES (%s, false)
        ON CONFLICT (channel_id) DO NOTHING
        """,
        (channel_id,),
    )
    row = conn.execute(
        """
        INSERT INTO videos (video_id, channel_id, title, duration_sec, origin)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (video_id) DO NOTHING
        RETURNING video_id
        """,
        (entry.video_id, channel_id, clean_text(entry.title), entry.duration_sec, origin),
    ).fetchone()
    return row is not None


def insert_backfill_video(
    conn: psycopg.Connection[Any], entry: CatalogEntry, *, channel_id: str
) -> bool:
    """Insert a channel-backfill row from a catalog entry (issue #41).

    ``insert_catalog_video`` with ``origin='backfill'``.
    """
    return insert_catalog_video(conn, entry, channel_id=channel_id, origin="backfill")


# The one definition of a video's processing status (#42). Both list_videos and
# get_video select from this, so the rules cannot drift apart. ``v`` is the
# videos row; the LATERAL joins see only that video's jobs and analyses.
_VIDEO_READ = """
    SELECT v.video_id, v.title, v.channel_id, c.title AS channel_title,
           v.published_at, v.duration_sec, v.origin, v.unavailable,
           CASE
               WHEN la.created_at IS NOT NULL THEN 'done'
               WHEN aj.id IS NOT NULL THEN 'processing'
               WHEN v.unavailable IS NOT NULL THEN 'unavailable'
               WHEN nj.state = 'dead' THEN 'failed'
               ELSE 'idle'
           END AS status,
           aj.id AS aj_id, aj.kind AS aj_kind, aj.state AS aj_state,
           lf.id AS lf_id, lf.kind AS lf_kind, lf.error_class AS lf_error_class,
           lf.finished_at AS lf_finished_at,
           la.created_at AS latest_analysis_at,
           v.discovered_at
    FROM videos v
    LEFT JOIN channels c ON c.channel_id = v.channel_id
    LEFT JOIN LATERAL (
        SELECT a.created_at FROM analyses a
        WHERE a.video_id = v.video_id
        ORDER BY a.created_at DESC, a.id DESC LIMIT 1
    ) la ON true
    LEFT JOIN LATERAL (
        SELECT j.id, j.kind, j.state FROM jobs j
        WHERE j.video_id = v.video_id AND j.state IN ('pending', 'running')
        ORDER BY CASE j.kind WHEN 'ingest' THEN 0 WHEN 'transcribe' THEN 1
                             WHEN 'analyze' THEN 2 ELSE 3 END,
                 j.created_at, j.id
        LIMIT 1
    ) aj ON true
    LEFT JOIN LATERAL (
        SELECT j.state FROM jobs j
        WHERE j.video_id = v.video_id
        ORDER BY j.created_at DESC, j.id DESC LIMIT 1
    ) nj ON true
    LEFT JOIN LATERAL (
        SELECT j.id, j.kind, j.error_class, j.finished_at FROM jobs j
        WHERE j.video_id = v.video_id AND j.state = 'dead'
        ORDER BY j.created_at DESC, j.id DESC LIMIT 1
    ) lf ON true
"""


def list_videos(
    conn: psycopg.Connection[Any],
    *,
    channel_id: str | None = None,
    status: str | None = None,
    published_after: datetime | None = None,
    published_before: datetime | None = None,
    offset: int = 0,
    limit: int = 50,
) -> VideoPage:
    """One page of the library plus the total match count (#42).

    Filters combine with AND; any date filter excludes videos with no
    ``published_at``. ``published_after`` is inclusive, ``published_before``
    exclusive. Ordered by ``published_at DESC NULLS LAST, discovered_at DESC,
    video_id``. One statement, however large ``limit`` is.
    """
    rows = conn.execute(
        f"""
        WITH m AS (
            SELECT * FROM ({_VIDEO_READ}) r
            WHERE (%(channel)s::text IS NULL OR channel_id = %(channel)s)
              AND (%(status)s::text IS NULL OR status = %(status)s)
              AND (%(after)s::timestamptz IS NULL OR published_at >= %(after)s)
              AND (%(before)s::timestamptz IS NULL OR published_at < %(before)s)
        )
        SELECT t.total, p.*
        FROM (SELECT count(*) AS total FROM m) t
        LEFT JOIN LATERAL (
            SELECT * FROM m
            ORDER BY published_at DESC NULLS LAST, discovered_at DESC, video_id ASC
            OFFSET %(offset)s LIMIT %(limit)s
        ) p ON true
        """,
        {
            "channel": channel_id,
            "status": status,
            "after": published_after,
            "before": published_before,
            "offset": offset,
            "limit": limit,
        },
    ).fetchall()
    total = int(rows[0][0]) if rows else 0
    items = tuple(_summary(row[1:]) for row in rows if row[1] is not None)
    return VideoPage(items=items, total=total)


def get_video(conn: psycopg.Connection[Any], video_id: str) -> VideoSummary | None:
    """One video with its derived status, or ``None`` without a ``videos`` row (#42)."""
    row = conn.execute(f"{_VIDEO_READ} WHERE v.video_id = %s", (video_id,)).fetchone()
    return None if row is None else _summary(row)


def _summary(row: Sequence[Any]) -> VideoSummary:
    (
        video_id, title, channel_id, channel_title, published_at, duration_sec, origin,
        unavailable, status, aj_id, aj_kind, aj_state, lf_id, lf_kind, lf_class,
        lf_finished, latest_analysis_at, _discovered_at,
    ) = row
    return VideoSummary(
        video_id=video_id,
        title=title,
        channel_id=channel_id,
        channel_title=channel_title,
        published_at=published_at,
        duration_sec=duration_sec,
        origin=origin,
        unavailable=unavailable,
        status=status,
        active_job=None if aj_id is None else ActiveJob(int(aj_id), aj_kind, aj_state),
        last_failure=(
            None if lf_id is None else JobFailure(int(lf_id), lf_kind, lf_class, lf_finished)
        ),
        latest_analysis_at=latest_analysis_at,
    )

