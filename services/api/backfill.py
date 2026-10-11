"""Channel backfill (issue #41; architecture.md 4, 8.3; D9b, C4).

``backfill`` lists a channel's newest ``limit`` uploads through a
``CatalogSource`` and, unless it is a dry run, gives every video the system
does not know yet a ``videos`` row (``origin = 'backfill'``) and one
``ingest`` job at ``PRIORITY_BACKFILL``. It is plain Python with no HTTP, so
the route only maps its result and errors.

Transactions:

- The channel-existence check is committed before the listing starts, so a
  slow yt-dlp run holds no lock or snapshot.
- Each new video is one transaction: row, then job only if this call
  inserted the row. A crash or a database error leaves every video either
  complete or absent, and a repeat call fills in the rest. A concurrent
  writer (RSS, another backfill) that inserted the row first wins; its job
  and origin are left alone.

A ``PermanentSourceError`` from the catalog means "The playlist does not
exist", which yt-dlp prints for a channel with no uploads as well as for a
removed one (#27, #107), so it is reported as an empty listing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import psycopg
import structlog
from psycopg.pq import TransactionStatus

from common.errors import PermanentSourceError
from common.models import CatalogEntry, CatalogSource, ChannelCatalog
from common.queue import PRIORITY_BACKFILL, JobQueue, PostgresQueue
from common.repo.channels import channel_exists
from common.repo.transcripts import median_whisper_rtf
from common.repo.videos import (
    delete_unused_backfill_video,
    insert_backfill_video,
    known_video_ids,
)

_log = structlog.get_logger(__name__)

MAX_BACKFILL_LIMIT = 500
ESTIMATE_ASSUMES = "every new video needs speech-to-text"


class ChannelNotRegistered(Exception):
    """The channel has no ``channels`` row; register it with ``POST /channels``."""


@dataclass(frozen=True, slots=True)
class Estimate:
    audio_sec: int
    unknown_duration: int
    rtf: float | None
    rtf_samples: int
    transcription_sec: int | None
    assumes: str = ESTIMATE_ASSUMES


@dataclass(frozen=True, slots=True)
class BackfillResult:
    channel_id: str
    dry_run: bool
    limit: int
    listed: int
    total_count: int | None
    already_known: int
    new_videos: int
    enqueued: int
    video_ids: tuple[str, ...]
    estimate: Estimate
    backfill_id: str | None


def backfill(
    conn: psycopg.Connection[Any],
    queue: JobQueue,
    catalog: CatalogSource,
    channel_id: str,
    *,
    limit: int,
    dry_run: bool,
) -> BackfillResult:
    """Select the channel's unknown uploads and, unless ``dry_run``, enqueue them.

    ``conn`` must have no open transaction; ``queue`` must write through it.
    Raises ``ChannelNotRegistered`` before listing if the channel has no row.
    Catalog errors other than ``PermanentSourceError`` propagate unchanged.
    """
    if conn.info.transaction_status != TransactionStatus.IDLE:
        raise RuntimeError("backfill expects a connection with no open transaction")
    with conn.transaction():
        registered = channel_exists(conn, channel_id)
    if not registered:
        raise ChannelNotRegistered(channel_id)

    listing = _list(catalog, channel_id, limit)

    with conn.transaction():
        known = known_video_ids(conn, [e.video_id for e in listing.entries])
        rtf, rtf_samples = median_whisper_rtf(conn)
    new = [e for e in listing.entries if e.video_id not in known]

    backfill_id = None if dry_run else str(uuid.uuid4())
    enqueued = 0
    if backfill_id is not None:
        payload = {"origin": "backfill", "backfill_id": backfill_id}
        for entry in new:
            with conn.transaction():
                if insert_backfill_video(conn, entry, channel_id=channel_id):
                    job = queue.enqueue(
                        "ingest", entry.video_id, payload=payload, priority=PRIORITY_BACKFILL
                    )
                    if job is not None:
                        enqueued += 1

    return BackfillResult(
        channel_id=channel_id,
        dry_run=dry_run,
        limit=limit,
        listed=len(listing.entries),
        total_count=listing.total_count,
        already_known=len(listing.entries) - len(new),
        new_videos=len(new),
        enqueued=enqueued,
        video_ids=tuple(e.video_id for e in new),
        estimate=_estimate(new, rtf, rtf_samples),
        backfill_id=backfill_id,
    )


@dataclass(frozen=True, slots=True)
class CancelResult:
    channel_id: str
    backfill_id: str
    cancelled: int
    videos_removed: int
    videos_kept: int
    video_ids: tuple[str, ...]


def cancel_backfill(
    conn: psycopg.Connection[Any], channel_id: str, backfill_id: str
) -> CancelResult:
    """Undo one real backfill's still-pending ``ingest`` jobs (#116).

    Deletes the jobs (``PostgresQueue.cancel_backfill``) and, for each, its
    ``videos`` row when nothing else depends on it, all in one transaction:
    either the job and its row go together or neither does. A video whose row
    is kept (a transcript, or another job, exists) is counted in
    ``videos_kept``. ``conn`` must have no open transaction.
    Raises ``ChannelNotRegistered`` if the channel has no row. An unknown
    ``backfill_id`` is not an error: it cancels nothing.
    """
    if conn.info.transaction_status != TransactionStatus.IDLE:
        raise RuntimeError("cancel_backfill expects a connection with no open transaction")
    with conn.transaction():
        if not channel_exists(conn, channel_id):
            raise ChannelNotRegistered(channel_id)
        jobs = PostgresQueue(conn).cancel_backfill(channel_id, backfill_id)
        removed = sum(delete_unused_backfill_video(conn, job.video_id) for job in jobs)
    return CancelResult(
        channel_id=channel_id,
        backfill_id=backfill_id,
        cancelled=len(jobs),
        videos_removed=removed,
        videos_kept=len(jobs) - removed,
        video_ids=tuple(job.video_id for job in jobs),
    )


def _list(catalog: CatalogSource, channel_id: str, limit: int) -> ChannelCatalog:
    try:
        return catalog.list_uploads(channel_id, limit=limit)
    except PermanentSourceError as exc:
        _log.info(
            "backfill listing empty: playlist does not exist",
            channel_id=channel_id,
            reason=exc.reason.value,
        )
        return ChannelCatalog(channel_id, (), None)


def _estimate(new: list[CatalogEntry], rtf: float | None, rtf_samples: int) -> Estimate:
    durations = [e.duration_sec for e in new if e.duration_sec is not None]
    audio_sec = sum(durations)
    return Estimate(
        audio_sec=audio_sec,
        unknown_duration=len(new) - len(durations),
        rtf=rtf,
        rtf_samples=rtf_samples,
        transcription_sec=None if rtf is None else round(audio_sec * rtf),
    )
