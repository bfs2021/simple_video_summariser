"""Planner task ``channel_backfill`` (issue #115; architecture.md 4, 8.3).

``POST /channels/{id}/backfill-requests`` only records a row in
``backfill_requests``. Each run of this task takes **at most one** request,
the oldest ``pending`` or ``running`` one, and does one bounded slice of it:
list the channel's newest ``limit_n`` uploads once, then write the videos the
system does not know yet and one ``ingest`` job each, exactly as the
synchronous ``POST /channels/{id}/backfill`` does (``common.backfill.backfill``
is the shared code). With no request it does nothing and returns ``0``.

Slices. A real request enqueues at most
``min(BACKFILL_ENQUEUE_PER_RUN, BACKFILL_MAX_PENDING - P)`` videos per run,
``P`` being the ``pending`` ``ingest`` jobs at the backfill priority (-10). More
new videos left over means the request stays ``running``; the next run lists
again, skips what is known now and carries on. A run whose allowance covered
every new video it listed (including none) ends the request (``done``).
When ``P`` already fills the budget the run enqueues nothing, does not list, does not count as a failed attempt and leaves the
request ``running``. A dry run writes ``result`` and ends ``done`` in one run.

Failures. ``TransientNetworkError`` and ``RateLimitedError`` count one
attempt and leave the request claimable; the ``BACKFILL_REQUEST_MAX_ATTEMPTS``-th
ends it ``failed``. ``ToolFailureError`` ends it ``failed`` at once. A
``PermanentSourceError`` is an empty listing (#107) handled inside
``common.backfill``. Only the exception class is stored or logged, never
yt-dlp's stderr or the message. Any other exception propagates (the scheduler
records the task as failed) and the request stays ``running`` for the next run.

Crash safety. The request is committed ``running`` before the listing, and every
video is its own transaction with ``ON CONFLICT DO NOTHING``, so a run that dies
part-way leaves complete videos only and the next run resumes without a second
row or job. A video that RSS or a concurrent backfill inserted first keeps its
own origin and priority. Backfill never touches ``channels``.

Run time. Planner tasks run one after another in one process, so a run delays
the others by at most ``BACKFILL_REQUEST_LIST_TIMEOUT_SEC`` for the listing plus
the writes for at most ``BACKFILL_ENQUEUE_PER_RUN`` videos (one short
transaction each). A dry run probes no subtitles, so it adds no further
network time. No transaction is open while the catalog runs.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from typing import Any

import psycopg
import structlog

from common.backfill import BackfillResult, ChannelNotRegistered, backfill
from common.config import Settings
from common.errors import RateLimitedError, ToolFailureError, TransientNetworkError
from common.models import CatalogSource
from common.queue import JobQueue
from common.repo import backfill_requests as requests
from common.repo.backfill_requests import BackfillRequestRow, request_backfill_id

_log = structlog.get_logger(__name__)

#: The catalog's timeout for these listings (the synchronous route's is 90 s).
BACKFILL_REQUEST_LIST_TIMEOUT_SEC = 300
BACKFILL_ENQUEUE_PER_RUN = 500
BACKFILL_MAX_PENDING = 1000
BACKFILL_REQUEST_MAX_ATTEMPTS = 3


def run_allowance(pending_backfill_jobs: int) -> int:
    """How many videos one run may enqueue, given ``P`` pending backfill ingests."""
    return max(0, min(BACKFILL_ENQUEUE_PER_RUN, BACKFILL_MAX_PENDING - pending_backfill_jobs))


def run_channel_backfill(
    conn: psycopg.Connection[Any],
    queue: JobQueue,
    catalog: CatalogSource,
    *,
    settings: Settings,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Do one slice of the oldest active request; return the jobs it enqueued.

    ``conn`` must have no open transaction and ``queue`` must write through it.
    Returns with the connection idle: every write is committed.
    """
    started = clock()
    with conn.transaction():
        request = requests.claim_next(conn)
        allowance = (
            0
            if request is None or request.dry_run
            else run_allowance(requests.count_pending_backfill_ingests(conn))
        )
    if request is None:
        return 0

    if not request.dry_run and allowance == 0:
        with conn.transaction():
            requests.touch(conn, request.id)
        _log_run(
            request,
            started,
            clock,
            listed=0,
            new_videos=0,
            enqueued_run=0,
            total=request.enqueued,
            state="running",
        )
        return 0

    try:
        result = backfill(
            conn,
            queue,
            catalog,
            request.channel_id,
            limit=request.limit_n,
            dry_run=request.dry_run,
            prefer_whisper=settings.PREFER_WHISPER,
            summarizer=settings.SUMMARIZER,
            model=(
                settings.ANTHROPIC_MODEL
                if settings.SUMMARIZER == "anthropic"
                else settings.OLLAMA_MODEL
            ),
            clock=clock,
            backfill_id=request_backfill_id(request.id),
            max_enqueue=allowance,
        )
    except (TransientNetworkError, RateLimitedError, ToolFailureError, ChannelNotRegistered) as exc:
        fatal = isinstance(exc, ToolFailureError | ChannelNotRegistered)
        error_class = type(exc).__name__
        with conn.transaction():
            state = requests.record_failure(
                conn,
                request.id,
                error_class=error_class,
                max_attempts=BACKFILL_REQUEST_MAX_ATTEMPTS,
                fatal=fatal,
            )
        _log.warning(
            "planner.channel_backfill.failed",
            request_id=request.id,
            channel_id=request.channel_id,
            error_class=error_class,
            state=state,
        )
        return 0

    # Every new video of this listing was attempted when the allowance covered
    # them all (concurrent writers' videos count as known); otherwise more remain.
    done = request.dry_run or result.new_videos <= allowance
    with conn.transaction():
        requests.record_progress(
            conn,
            request.id,
            enqueued_this_run=result.enqueued,
            result=_result_json(result),
            done=done,
        )
    _log_run(
        request,
        started,
        clock,
        listed=result.listed,
        new_videos=result.new_videos,
        enqueued_run=result.enqueued,
        total=request.enqueued + result.enqueued,
        state="done" if done else "running",
    )
    return result.enqueued


def _result_json(result: BackfillResult) -> dict[str, Any]:
    """The synchronous route's response body, key for key."""
    body = dataclasses.asdict(result)
    body["video_ids"] = list(result.video_ids)
    body["videos"] = list(body["videos"])
    return body


def _log_run(
    request: BackfillRequestRow,
    started: float,
    clock: Callable[[], float],
    *,
    listed: int,
    new_videos: int,
    enqueued_run: int,
    total: int,
    state: str,
) -> None:
    _log.info(
        "planner.channel_backfill",
        request_id=request.id,
        channel_id=request.channel_id,
        dry_run=request.dry_run,
        listed=listed,
        new_videos=new_videos,
        enqueued_this_run=enqueued_run,
        enqueued_total=total,
        state=state,
        duration_ms=round((clock() - started) * 1000, 1),
    )
