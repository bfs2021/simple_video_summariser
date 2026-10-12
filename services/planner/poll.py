"""Channel polling (issue #34; architecture.md D9b, §4 priority bands).

``poll_channels`` is one pass: it fetches every active channel's RSS feed
once, records each unseen video published *after* the channel's
``monitor_from`` with ``origin='rss'`` and queues an ``ingest`` job for it.
Older entries are ignored completely, so a later backfill (#41) still finds
them unseen.

Each new video's row and its job are written in one transaction, so a crash
never leaves a video recorded without its job. One broken feed, or one
failing channel, is recorded on that channel and never stops the others.
Database connection errors are the exception: they propagate, and the
planner loop (#38) handles the retry.

With ``skip_shorts`` (``SKIP_SHORTS``, #80) a new Short is still recorded, but
as ``unavailable = 'short'`` and with no job, so the next poll sees it as seen.

Scheduling, the singleton guard and reconnecting belong to #38.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg

from common.errors import MAX_ERROR_BYTES, UnavailableReason
from common.models import Channel, FeedSource
from common.queue import PRIORITY_NORMAL, JobQueue
from common.repo.channels import list_active_channels, record_poll
from common.repo.videos import insert_discovered_video
from common.shorts import should_skip_short

_logger = logging.getLogger(__name__)

_ORIGIN = "rss"
_INGEST_PRIORITY = PRIORITY_NORMAL  # architecture.md §4


@dataclass(frozen=True)
class PollResult:
    """What one ``poll_channels`` call did.

    ``videos_discovered`` counts inserted ``videos`` rows and ``jobs_enqueued``
    counts ``enqueue`` calls that returned an id (not the C1-deduped ones).
    ``shorts_skipped`` counts the inserted rows that got no job because they
    are Shorts (#80); they are in ``videos_discovered`` too.
    """

    channels_polled: int
    channels_failed: int
    videos_discovered: int
    jobs_enqueued: int
    shorts_skipped: int = 0


@dataclass
class _Counts:
    discovered: int = 0
    enqueued: int = 0
    shorts_skipped: int = 0


def poll_channels(
    conn: psycopg.Connection[Any],
    feed: FeedSource,
    queue: JobQueue,
    *,
    skip_shorts: bool = False,
) -> PollResult:
    """Poll every active channel's feed once.

    ``skip_shorts`` is ``Settings.SKIP_SHORTS``: a new Short is recorded with
    ``unavailable = 'short'`` and gets no ``ingest`` job.
    """
    channels = list_active_channels(conn)
    conn.commit()  # end the read transaction; nothing may stay open across a fetch
    polled = failed = 0
    counts = _Counts()
    for channel in channels:
        try:
            entries = feed.fetch(channel.channel_id)
            _process_entries(conn, queue, channel, entries, counts, skip_shorts)
        except psycopg.OperationalError:
            raise
        except Exception as exc:
            conn.rollback()
            health = record_poll(conn, channel.channel_id, error=_error_text(exc))
            since = health.poll_failing_since if health is not None else None
            _logger.warning(
                "planner.poll.channel_failed",
                exc_info=exc,
                extra={
                    "channel_id": channel.channel_id,
                    "poll_failing_since": (
                        since.astimezone(UTC).isoformat() if since is not None else None
                    ),
                    "poll_failures": health.poll_failures if health is not None else None,
                },
            )
            failed += 1
        else:
            record_poll(conn, channel.channel_id)
            polled += 1
        conn.commit()
    result = PollResult(polled, failed, counts.discovered, counts.enqueued, counts.shorts_skipped)
    _logger.info(
        "planner.poll",
        extra={
            "channels_polled": result.channels_polled,
            "channels_failed": result.channels_failed,
            "videos_discovered": result.videos_discovered,
            "jobs_enqueued": result.jobs_enqueued,
            "shorts_skipped": result.shorts_skipped,
        },
    )
    return result


def _process_entries(
    conn: psycopg.Connection[Any],
    queue: JobQueue,
    channel: Channel,
    entries: list[Any],
    counts: _Counts,
    skip_shorts: bool,
) -> None:
    seen: set[str] = set()
    all_new = True
    considered = 0
    oldest: datetime | None = None
    for entry in entries:
        if entry.channel_id != channel.channel_id:
            _logger.warning(
                "planner.poll.foreign_entry",
                extra={
                    "channel_id": channel.channel_id,
                    "entry_channel_id": entry.channel_id,
                    "video_id": entry.video_id,
                },
            )
            continue
        published_at = entry.published_at
        if published_at is None or published_at.utcoffset() is None:
            _logger.warning(
                "planner.poll.bad_timestamp",
                extra={"channel_id": channel.channel_id, "video_id": entry.video_id},
            )
            continue
        if entry.video_id in seen:
            continue
        seen.add(entry.video_id)
        considered += 1
        if published_at <= channel.monitor_from:
            all_new = False
            continue
        skip = should_skip_short(is_short=entry.is_short, origin=_ORIGIN, skip_shorts=skip_shorts)
        job_id = None
        with conn.transaction():
            if skip:
                inserted = insert_discovered_video(
                    conn, entry, _ORIGIN, unavailable=UnavailableReason.SHORT.value
                )
            else:
                inserted = insert_discovered_video(conn, entry, _ORIGIN)
                if inserted:
                    job_id = queue.enqueue("ingest", entry.video_id, priority=_INGEST_PRIORITY)
        if not inserted:
            all_new = False
            continue
        counts.discovered += 1
        if skip:
            counts.shorts_skipped += 1
            _logger.info(
                "planner.poll.short_skipped",
                extra={"channel_id": channel.channel_id, "video_id": entry.video_id},
            )
        elif job_id is not None:
            counts.enqueued += 1
        if oldest is None or published_at < oldest:
            oldest = published_at
    if all_new and considered and oldest is not None:
        floor = channel.monitor_from
        if channel.last_polled is not None and channel.last_polled > floor:
            floor = channel.last_polled
        if oldest > floor:
            _logger.warning(
                "possible feed gap",
                extra={"channel_id": channel.channel_id, "oldest_published_at": oldest},
            )


def _error_text(exc: Exception) -> str:
    """``"<ExceptionClass>: <message>"``, cut to at most ``MAX_ERROR_BYTES`` bytes."""
    text = f"{type(exc).__name__}: {exc}"
    return text.encode("utf-8")[:MAX_ERROR_BYTES].decode("utf-8", "ignore")
