"""RSS-gap reconciliation (issue #93; architecture.md D9b, §4 priority bands).

A channel's RSS feed holds only its newest ~15 uploads, so a channel that
publishes more than that between two polls (#34) loses videos it never saw;
#34 only logs ``possible feed gap``. ``reconcile_channels`` is one pass, called
by the planner loop (#38) every ``RECONCILE_INTERVAL_SEC``: it lists each
active channel's newest uploads with the catalog adapter (#27), bounded to
``RECONCILE_LIST_LIMIT``, and records the ones the feed dropped.

Why the selection is anchor-based. The flat catalog listing has no publish
date, so "published after ``monitor_from``" cannot be read from it. The
uploads playlist is ordered newest first, though, so every entry positioned
above a video that is *known* to be published after ``monitor_from`` was
uploaded after the cutoff too. That known video is the **anchor**: the entry
positioned lowest in the listing whose ``videos`` row belongs to this channel
and has a ``published_at`` strictly greater than ``monitor_from``. Candidates
are the entries above the anchor with no ``videos`` row at all. Unknown entries
below the anchor may be pre-``monitor_from`` history, which D9b leaves unseen
until an explicit backfill (#41), so they are never touched. With no anchor
nothing is written (the cases that leaves open are #156's).

A candidate is inserted with ``origin='rss'`` and a NULL ``published_at``
(a made-up date would be trusted as an anchor later), and an ``ingest`` job at
the Normal priority is queued only if this call inserted the row, so a video
that a poll or a backfill got to first keeps its own row and job. Each video
is written in its own transaction. No transaction is open while the catalog
runs. The pass never touches ``channels.last_polled``, ``last_poll_err``,
``active`` or ``monitor_from``.

Failures. One channel failing never stops the others, except a rate limit,
which ends the pass so YouTube is not hammered; the next run resumes at the
channel that was rate-limited. ``psycopg.OperationalError`` propagates, and the
scheduler handles it.

Bounded run time. The scheduler touches the planner's heartbeat file only
between tasks, so a pass must stay well under 180 s: the catalog is built with
``RECONCILE_LIST_TIMEOUT_SEC`` and no new channel starts once
``RECONCILE_PASS_BUDGET_SEC`` has elapsed on the injected monotonic ``clock``.
Channels are visited in ``channel_id`` order after the ``start_after`` cursor,
wrapping around, each at most once per pass; ``resume_after`` in the result is
the cursor for the next call, so deferred channels go first next time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psycopg

from common.errors import PermanentSourceError, RateLimitedError
from common.models import CatalogEntry, CatalogSource, Channel
from common.queue import PRIORITY_NORMAL, JobQueue
from common.repo.channels import (
    clear_feed_gap,
    list_active_channels,
    set_feed_gap_reason,
)
from common.repo.videos import insert_catalog_video, known_videos

_logger = logging.getLogger(__name__)

#: How many of a channel's newest uploads one listing covers.
RECONCILE_LIST_LIMIT = 50
#: ``YouTubeCatalog(timeout=...)``: one listing's ceiling (built in ``main.py``).
RECONCILE_LIST_TIMEOUT_SEC = 60
#: No new channel starts once this long has elapsed since the pass began.
RECONCILE_PASS_BUDGET_SEC = 90
#: Recovered videos are forward-monitored content, so the Normal band (§4).
RECONCILE_PRIORITY = PRIORITY_NORMAL

_ORIGIN = "rss"


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    """What one ``reconcile_channels`` call did.

    ``channels_checked`` counts channels whose listing was read (including ones
    with no anchor and ones the catalog reports as nothing listable);
    ``channels_failed`` those that failed, the rate-limited one included;
    ``channels_deferred`` those not reached because the time budget ran out or a
    rate limit ended the pass. ``listed`` is the entries listed, ``videos_recovered``
    the ``videos`` rows inserted and ``jobs_enqueued`` the ``enqueue`` calls that
    returned an id (not the C1-deduplicated ones). ``resume_after`` is the cursor
    for the next call: the last channel finished, or the one before the
    rate-limited channel, or the cursor passed in when none finished.
    """

    channels_checked: int
    channels_failed: int
    channels_deferred: int
    listed: int
    videos_recovered: int
    jobs_enqueued: int
    resume_after: str | None


@dataclass
class _Counts:
    checked: int = 0
    failed: int = 0
    deferred: int = 0
    listed: int = 0
    recovered: int = 0
    enqueued: int = 0


def reconcile_channels(
    conn: psycopg.Connection[Any],
    catalog: CatalogSource,
    queue: JobQueue,
    *,
    clock: Callable[[], float],
    start_after: str | None = None,
    list_limit: int = RECONCILE_LIST_LIMIT,
) -> ReconcileResult:
    """Reconcile every active channel's newest uploads once."""
    started = clock()
    channels = sorted(list_active_channels(conn), key=lambda c: c.channel_id)
    conn.commit()  # end the read transaction; nothing may stay open across a listing
    if start_after is not None:
        channels = [c for c in channels if c.channel_id > start_after] + [
            c for c in channels if c.channel_id <= start_after
        ]

    counts = _Counts()
    resume_after = start_after
    for index, channel in enumerate(channels):
        if clock() - started >= RECONCILE_PASS_BUDGET_SEC:
            counts.deferred = len(channels) - index
            break
        try:
            _reconcile_channel(conn, catalog, queue, channel, counts, list_limit)
        except psycopg.OperationalError:
            raise
        except PermanentSourceError as exc:
            _logger.info(
                "planner.reconcile.channel_unlistable",
                extra={"channel_id": channel.channel_id, "reason": exc.reason.value},
            )
            counts.checked += 1
        except RateLimitedError as exc:
            conn.rollback()
            _log_failure(channel, exc, "planner.reconcile.rate_limited")
            counts.failed += 1
            counts.deferred = len(channels) - index - 1
            break
        except Exception as exc:  # noqa: BLE001 - one broken channel must not stop the others
            conn.rollback()
            _log_failure(channel, exc, "planner.reconcile.channel_failed")
            counts.failed += 1
        else:
            counts.checked += 1
        resume_after = channel.channel_id
        conn.commit()

    result = ReconcileResult(
        channels_checked=counts.checked,
        channels_failed=counts.failed,
        channels_deferred=counts.deferred,
        listed=counts.listed,
        videos_recovered=counts.recovered,
        jobs_enqueued=counts.enqueued,
        resume_after=resume_after,
    )
    _logger.info(
        "planner.reconcile",
        extra={
            "channels_checked": result.channels_checked,
            "channels_failed": result.channels_failed,
            "channels_deferred": result.channels_deferred,
            "listed": result.listed,
            "videos_recovered": result.videos_recovered,
            "jobs_enqueued": result.jobs_enqueued,
            "resume_after": result.resume_after,
            "duration_ms": round((clock() - started) * 1000),
        },
    )
    return result


def _log_failure(channel: Channel, exc: Exception, event: str) -> None:
    _logger.warning(
        event, extra={"channel_id": channel.channel_id, "error_class": type(exc).__name__}
    )


def _reconcile_channel(
    conn: psycopg.Connection[Any],
    catalog: CatalogSource,
    queue: JobQueue,
    channel: Channel,
    counts: _Counts,
    list_limit: int,
) -> None:
    listing = catalog.list_uploads(channel.channel_id, limit=list_limit)
    counts.listed += len(listing.entries)
    entries: list[CatalogEntry] = []
    seen: set[str] = set()
    for entry in listing.entries:
        if entry.video_id not in seen:
            seen.add(entry.video_id)
            entries.append(entry)

    known = known_videos(conn, [e.video_id for e in entries])
    conn.commit()  # the lookup's implicit transaction ends before any write block

    anchor: int | None = None
    for position, entry in enumerate(entries):
        row = known.get(entry.video_id)
        if (
            row is not None
            and row.channel_id == channel.channel_id
            and row.published_at is not None
            and row.published_at > channel.monitor_from
        ):
            anchor = position  # keep going: the lowest anchor bounds the candidates
    if anchor is None:
        _logger.info(
            "planner.reconcile.no_anchor",
            extra={"channel_id": channel.channel_id, "listed": len(entries)},
        )
        with conn.transaction():
            set_feed_gap_reason(conn, channel.channel_id, "no_anchor", source="reconcile")
        return

    recovered_here = 0
    enqueued_here = 0
    try:
        with conn.transaction():
            for entry in entries[:anchor]:
                if entry.video_id in known:
                    continue
                inserted = insert_catalog_video(
                    conn, entry, channel_id=channel.channel_id, origin=_ORIGIN
                )
                job_id = (
                    queue.enqueue("ingest", entry.video_id, priority=RECONCILE_PRIORITY)
                    if inserted
                    else None
                )
                if inserted:
                    recovered_here += 1
                    if job_id is not None:
                        enqueued_here += 1
            if len(entries) == list_limit and anchor == len(entries) - 1:
                set_feed_gap_reason(
                    conn,
                    channel.channel_id,
                    "listing_window_exhausted",
                    source="reconcile",
                )
            else:
                clear_feed_gap(conn, channel.channel_id, source="reconcile")
        counts.recovered += recovered_here
        counts.enqueued += enqueued_here
    finally:
        if recovered_here:
            _logger.info(
                "planner.reconcile.recovered",
                extra={"channel_id": channel.channel_id, "count": recovered_here},
            )
