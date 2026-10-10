"""Channel repository functions (issue #14, architecture.md §6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg

from common.models import Channel
from common.repo._hygiene import clean_text


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
                  (c.xmax = 0) AS created
        """,
        {"id": channel_id},
    ).fetchone()
    if row is not None:
        return RegisteredChannel(channel=_channel(row[:9]), created=bool(row[9]))
    # Already active: the upsert changed nothing and returned nothing. Read it
    # in a new statement, whose snapshot also sees a row a concurrent
    # registration has just committed.
    existing = conn.execute(
        """
        SELECT channel_id, title, active, monitor_from, last_polled, last_poll_err, added_at,
               poll_failing_since, poll_failures
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
               last_poll_err, added_at, poll_failing_since, poll_failures
        FROM channels
        WHERE active = true
        ORDER BY channel_id
        """
    ).fetchall()
    return [_channel(row) for row in rows]


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
    )


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
