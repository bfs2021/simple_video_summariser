"""Channel repository functions (issue #14, architecture.md §6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
import structlog

from common.models import Channel, ChannelListItem, ChannelPage
from common.repo._hygiene import clean_text

_log = structlog.get_logger(__name__)
FEED_GAP_REASONS = ("suspected", "no_anchor", "listing_window_exhausted")


def add_channel(conn: psycopg.Connection[Any], channel_id: str, title: str | None) -> None:
    """Create an active channel with ``monitor_from = now()``.

    A no-op if ``channel_id`` already has a row, active or not. Calling
    this again for an already-active channel changes nothing, not even
    ``monitor_from``. To also re-activate an inactive channel (one that
    ``upsert_video`` created for an ad-hoc video, or one that was
    deactivated), use ``register_channel``.
    """
    conn.execute(
        """
        INSERT INTO channels (channel_id, title, active, monitor_from)
        VALUES (%s, %s, true, now())
        ON CONFLICT (channel_id) DO NOTHING
        """,
        (channel_id, clean_text(title)),
    )


@dataclass(frozen=True, slots=True)
class RegisteredChannel:
    channel: Channel
    created: bool


def register_channel(conn: psycopg.Connection[Any], channel_id: str) -> RegisteredChannel:
    """Start forward-only monitoring of ``channel_id`` (#40, D9b) in one statement.

    - Unknown channel: inserted with ``active = true`` and ``monitor_from = now()``.
    - Inactive channel: set ``active = true`` and ``monitor_from = now()``, so
      the gap is never backfilled; ``title`` and ``added_at`` are kept.
    - Active channel: nothing changes, not even ``monitor_from``.

    The state change is the single ``INSERT ... ON CONFLICT DO UPDATE ... WHERE``
    statement; an already-active channel is then read back. ``created`` is true
    only for the insert. Concurrent calls for the same new channel are safe:
    ``ON CONFLICT`` waits for the other insert, then takes the no-op path.
    """
    row = conn.execute(
        """
        INSERT INTO channels AS c (channel_id, active, monitor_from)
        VALUES (%(id)s, true, now())
        ON CONFLICT (channel_id) DO UPDATE
            SET active = true, monitor_from = now()
            WHERE c.active = false
        RETURNING c.channel_id, c.title, c.active, c.monitor_from, c.last_polled,
                   c.last_poll_err, c.added_at, c.poll_failing_since, c.poll_failures,
                   c.feed_gap_detected_at, c.feed_gap_reason, (c.xmax = 0) AS created
        """,
        {"id": channel_id},
    ).fetchone()
    if row is not None:
        return RegisteredChannel(channel=_channel(row[:11]), created=bool(row[11]))
    # Already active: the upsert changed nothing and returned nothing. Read it
    # in a new statement, whose snapshot also sees a row a concurrent
    # registration has just committed.
    existing = conn.execute(
        """
        SELECT channel_id, title, active, monitor_from, last_polled, last_poll_err, added_at,
               poll_failing_since, poll_failures, feed_gap_detected_at, feed_gap_reason
        FROM channels WHERE channel_id = %(id)s
        """,
        {"id": channel_id},
    ).fetchone()
    assert existing is not None
    return RegisteredChannel(channel=_channel(existing), created=False)


def channel_exists(conn: psycopg.Connection[Any], channel_id: str) -> bool:
    """Whether ``channel_id`` has a ``channels`` row, active or not."""
    row = conn.execute(
        "SELECT 1 FROM channels WHERE channel_id = %s", (channel_id,)
    ).fetchone()
    return row is not None


def list_active_channels(conn: psycopg.Connection[Any]) -> list[Channel]:
    """Every channel with ``active = true``."""
    rows = conn.execute(
        """
        SELECT channel_id, title, active, monitor_from, last_polled,
               last_poll_err, added_at, poll_failing_since, poll_failures,
               feed_gap_detected_at, feed_gap_reason
        FROM channels
        WHERE active = true
        ORDER BY channel_id
        """
    ).fetchall()
    return [_channel(row) for row in rows]


def list_channels(conn: psycopg.Connection[Any], *, offset: int, limit: int) -> ChannelPage:
    """One page of every channel, active or not, plus the total count (#119).

    Ordered by ``title ASC NULLS LAST, channel_id ASC`` (a total order).
    ``video_count`` is the number of ``videos`` rows with that ``channel_id``;
    videos without a channel count nowhere. Two statements, however large
    ``limit`` is: the page with its counts, and the total.
    """
    rows = conn.execute(
        """
        SELECT c.channel_id, c.title, c.active, count(v.video_id)
        FROM (
            SELECT channel_id, title, active FROM channels
            ORDER BY title ASC NULLS LAST, channel_id ASC
            OFFSET %(offset)s LIMIT %(limit)s
        ) c
        LEFT JOIN videos v ON v.channel_id = c.channel_id
        GROUP BY c.channel_id, c.title, c.active
        ORDER BY c.title ASC NULLS LAST, c.channel_id ASC
        """,
        {"offset": offset, "limit": limit},
    ).fetchall()
    total_row = conn.execute("SELECT count(*) FROM channels").fetchone()
    assert total_row is not None
    return ChannelPage(
        items=tuple(
            ChannelListItem(channel_id=r[0], title=r[1], active=r[2], video_count=int(r[3]))
            for r in rows
        ),
        total=int(total_row[0]),
    )


@dataclass(frozen=True, slots=True)
class PollHealth:
    """A channel's failure run after ``record_poll`` (see its docstring)."""

    poll_failing_since: datetime | None
    poll_failures: int


def record_poll(
    conn: psycopg.Connection[Any], channel_id: str, error: str | None = None
) -> PollHealth | None:
    """Record one RSS poll outcome in a single ``UPDATE`` (#34, #90).

    Always sets ``last_polled = now()`` and ``last_poll_err = error`` (``None``
    clears the previous error). The failure-run columns mean:

    - ``poll_failing_since``: when the current run of failed polls started
      (the ``now()`` of its first failure); ``NULL`` when the last poll
      succeeded or the channel was never polled.
    - ``poll_failures``: how many polls in a row have failed; ``0`` iff
      ``poll_failing_since`` is ``NULL``.

    A failure (``error`` is not ``None``) on a healthy channel starts a run
    (``poll_failing_since = now()``, ``poll_failures = 1``); on an already
    failing channel it keeps the start and adds 1. A success resets both. Every
    failure class counts the same. List failing feeds with::

        SELECT channel_id, poll_failing_since, poll_failures, last_poll_err
        FROM channels WHERE poll_failing_since IS NOT NULL
        ORDER BY poll_failing_since

    Returns the two columns as stored after this call, or ``None`` when
    ``channel_id`` has no row.
    """
    row = conn.execute(
        """
        UPDATE channels
        SET last_polled = now(),
            last_poll_err = %(err)s,
            poll_failing_since = CASE WHEN %(failed)s
                THEN coalesce(poll_failing_since, now()) END,
            poll_failures = CASE WHEN %(failed)s THEN poll_failures + 1 ELSE 0 END
        WHERE channel_id = %(id)s
        RETURNING poll_failing_since, poll_failures
        """,
        {"err": clean_text(error), "failed": error is not None, "id": channel_id},
    ).fetchone()
    if row is None:
        return None
    return PollHealth(poll_failing_since=row[0], poll_failures=row[1])


def _channel(row: tuple[Any, ...]) -> Channel:
    (
        channel_id,
        title,
        active,
        monitor_from,
        last_polled,
        last_poll_err,
        added_at,
        poll_failing_since,
        poll_failures,
        feed_gap_detected_at,
        feed_gap_reason,
    ) = row
    return Channel(
        channel_id=channel_id,
        title=title,
        active=active,
        monitor_from=monitor_from,
        last_polled=last_polled,
        last_poll_err=last_poll_err,
        added_at=added_at,
        poll_failing_since=poll_failing_since,
        poll_failures=poll_failures,
        feed_gap_detected_at=feed_gap_detected_at,
        feed_gap_reason=feed_gap_reason,
    )


@dataclass(frozen=True, slots=True)
class FeedGapState:
    detected_at: datetime
    reason: str


@dataclass(frozen=True, slots=True)
class OpsChannel:
    channel_id: str
    title: str | None
    poll_failing_since: datetime | None
    poll_failures: int
    last_poll_err: str | None
    feed_gap_detected_at: datetime | None
    feed_gap_reason: str | None


def _log_gap_transition(
    channel_id: str, previous_reason: str | None, reason: str | None, source: str
) -> None:
    if previous_reason != reason:
        _log.info(
            "channel.feed_gap_state_changed",
            channel_id=channel_id,
            previous_reason=previous_reason,
            reason=reason,
            source=source,
        )


def mark_feed_gap_suspected(
    conn: psycopg.Connection[Any], channel_id: str, *, source: str
) -> FeedGapState | None:
    row = conn.execute(
        """
        WITH previous AS (
            SELECT channel_id, feed_gap_reason FROM channels WHERE channel_id = %s FOR UPDATE
        )
        UPDATE channels c
        SET feed_gap_detected_at = coalesce(c.feed_gap_detected_at, now()),
            feed_gap_reason = 'suspected'
        FROM previous p
        WHERE c.channel_id = p.channel_id
        RETURNING c.feed_gap_detected_at, c.feed_gap_reason, p.feed_gap_reason
        """,
        (channel_id,),
    ).fetchone()
    if row is None:
        return None
    previous = row[2]
    _log_gap_transition(channel_id, previous, "suspected", source)
    return FeedGapState(row[0], row[1])


def set_feed_gap_reason(
    conn: psycopg.Connection[Any], channel_id: str, reason: str, *, source: str
) -> FeedGapState | None:
    if reason not in FEED_GAP_REASONS:
        raise ValueError("invalid feed gap reason")
    row = conn.execute(
        """
        WITH previous AS (
            SELECT channel_id, feed_gap_reason FROM channels
            WHERE channel_id = %(id)s AND feed_gap_reason IS NOT NULL FOR UPDATE
        )
        UPDATE channels c
        SET feed_gap_reason = %(reason)s
        FROM previous p
        WHERE c.channel_id = p.channel_id
        RETURNING c.feed_gap_detected_at, c.feed_gap_reason, p.feed_gap_reason
        """,
        {"id": channel_id, "reason": reason},
    ).fetchone()
    if row is None:
        return None
    previous = row[2]
    _log_gap_transition(channel_id, previous, reason, source)
    return FeedGapState(row[0], row[1])


def clear_feed_gap(
    conn: psycopg.Connection[Any], channel_id: str, *, source: str
) -> FeedGapState | None:
    row = conn.execute(
        """
        WITH previous AS (
            SELECT channel_id, feed_gap_detected_at, feed_gap_reason FROM channels
            WHERE channel_id = %s AND feed_gap_reason IS NOT NULL FOR UPDATE
        )
        UPDATE channels c
        SET feed_gap_detected_at = NULL, feed_gap_reason = NULL
        FROM previous p
        WHERE c.channel_id = p.channel_id
        RETURNING p.feed_gap_detected_at, p.feed_gap_reason
        """,
        (channel_id,),
    ).fetchone()
    if row is None:
        return None
    _log_gap_transition(channel_id, row[1], None, source)
    return FeedGapState(row[0], row[1])


def list_ops_channels(
    conn: psycopg.Connection[Any], *, failing: bool, attention: bool
) -> list[OpsChannel]:
    where = "poll_failing_since IS NOT NULL" if failing else (
        "poll_failing_since IS NOT NULL OR feed_gap_reason IS NOT NULL" if attention else "true"
    )
    rows = conn.execute(
        f"""
        SELECT channel_id, title, poll_failing_since, poll_failures, last_poll_err,
               feed_gap_detected_at, feed_gap_reason
        FROM channels
        WHERE {where}
        ORDER BY CASE WHEN poll_failing_since IS NOT NULL THEN 0
                      WHEN feed_gap_reason IS NOT NULL THEN 1 ELSE 2 END,
                 CASE WHEN poll_failing_since IS NOT NULL THEN poll_failing_since END ASC,
                 CASE WHEN poll_failing_since IS NULL THEN feed_gap_detected_at END ASC,
                 title ASC NULLS LAST, channel_id ASC
        """
    ).fetchall()
    return [OpsChannel(*row) for row in rows]


def feed_gap_counts(conn: psycopg.Connection[Any]) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT count(*) FILTER (WHERE feed_gap_reason = 'suspected'),
               count(*) FILTER (WHERE feed_gap_reason = 'no_anchor'),
               count(*) FILTER (WHERE feed_gap_reason = 'listing_window_exhausted')
        FROM channels
        """
    ).fetchone()
    assert row is not None
    return dict(zip(FEED_GAP_REASONS, (int(value) for value in row), strict=True))


def get_channel_title(conn: psycopg.Connection[Any], channel_id: str) -> str | None:
    """The stored title of ``channel_id`` (#45), read-only.

    ``None`` for an empty or unknown ``channel_id`` and for a channel whose
    title is NULL or empty.
    """
    if not channel_id:
        return None
    row = conn.execute(
        "SELECT title FROM channels WHERE channel_id = %s", (channel_id,)
    ).fetchone()
    if row is None or not row[0]:
        return None
    return str(row[0])


def fill_channel_title(conn: psycopg.Connection[Any], channel_id: str, title: str | None) -> None:
    """Set ``title`` on a channel whose title is NULL; a title already set is kept (#81)."""
    if title is None:
        return
    conn.execute(
        "UPDATE channels SET title = %(title)s WHERE channel_id = %(id)s AND title IS NULL",
        {"id": channel_id, "title": title},
    )
