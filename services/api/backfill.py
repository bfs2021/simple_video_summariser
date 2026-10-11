"""Channel backfill, API side (issues #41, #116, #115; architecture.md 8.3).

The listing, selection, estimate and per-video writes live in
``common.backfill`` since #115, so the planner's ``channel_backfill`` task can
share them; this module re-exports the names the routes and tests use and keeps
``cancel_backfill``, which only the API calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.pq import TransactionStatus

from common.backfill import (
    BACKFILL_DRY_RUN_DEADLINE_SEC,
    BACKFILL_PROBE_CONCURRENCY,
    BACKFILL_PROBE_TIMEOUT_SEC,
    ESTIMATE_ASSUMES,
    LLM_BASIS,
    MAX_BACKFILL_LIMIT,
    MAX_BACKFILL_REQUEST_LIMIT,
    SUBTITLES_MANUAL,
    SUBTITLES_TRANSCRIBE,
    SUBTITLES_UNCHECKED,
    BackfillResult,
    ChannelNotRegistered,
    Estimate,
    LlmEstimate,
    SubtitleProbe,
    VideoEstimate,
    backfill,
)
from common.queue import PostgresQueue
from common.repo.backfill_requests import cancel_active
from common.repo.channels import channel_exists
from common.repo.videos import delete_unused_backfill_video

__all__ = [
    "BACKFILL_DRY_RUN_DEADLINE_SEC",
    "BACKFILL_PROBE_CONCURRENCY",
    "BACKFILL_PROBE_TIMEOUT_SEC",
    "ESTIMATE_ASSUMES",
    "LLM_BASIS",
    "MAX_BACKFILL_LIMIT",
    "MAX_BACKFILL_REQUEST_LIMIT",
    "SUBTITLES_MANUAL",
    "SUBTITLES_TRANSCRIBE",
    "SUBTITLES_UNCHECKED",
    "BackfillResult",
    "CancelResult",
    "ChannelNotRegistered",
    "Estimate",
    "LlmEstimate",
    "SubtitleProbe",
    "VideoEstimate",
    "backfill",
    "cancel_backfill",
]


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
    ``videos_kept``. If ``backfill_id`` is the id of the channel's active
    backfill request (#115), that request is ended too, in the same
    transaction, so the planner stops enqueueing more (it ends ``failed`` with
    ``error_class = 'CANCELLED'``). ``conn`` must have no open transaction.
    Raises ``ChannelNotRegistered`` if the channel has no row. An unknown
    ``backfill_id`` is not an error: it cancels nothing.
    """
    if conn.info.transaction_status != TransactionStatus.IDLE:
        raise RuntimeError("cancel_backfill expects a connection with no open transaction")
    with conn.transaction():
        if not channel_exists(conn, channel_id):
            raise ChannelNotRegistered(channel_id)
        cancel_active(conn, channel_id, backfill_id)
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
