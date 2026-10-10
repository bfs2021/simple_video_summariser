"""Auto-caption fallback sweep (issue #92; architecture.md §7.1, §8.4).

``run_auto_caption_fallback_sweep`` is one pass, called by the planner loop
(#38) on each tick. It finds videos whose newest ``transcribe`` job is ``dead``
and enqueues one fresh ``ingest`` job for each. With ``AUTO_CAPTION_FALLBACK``
on, that ingest takes #28's ``AUTO_FALLBACK`` path (it reads the dead state
from the database, so the job needs no flag in its payload) and the video still
gets a ``youtube_auto`` transcript and an ``analyze`` job. The sweep does no
caption work itself.

Level-triggered on purpose: both ways a job dies (``PostgresQueue._fail`` and
``reap_stale``) are covered, a crash between "marked dead" and "enqueued"
cannot lose the trigger, and jobs that were already dead before the deploy are
picked up too. Likewise, a video that dead-lettered while
``AUTO_CAPTION_FALLBACK=0`` is picked up on the first pass after the flag goes
back to ``1``; that is intended.

A video is a candidate when its newest ``transcribe`` job (``created_at DESC,
id DESC``, the order ``latest_job_state`` uses) is ``dead`` with an
``error_class`` other than ``PERMANENT_SOURCE`` (NULL, as a reaped job has, is
eligible), the video is not marked unavailable, it has no transcript with
non-blank text from any source, and no ``ingest`` job in any state was created
after the dead job finished. The last rule is the loop guard: each dead
transcribe job earns at most one fallback ingest, and an ad-hoc resubmission
after the dead-letter counts as that attempt. A dead fallback ingest is left
for the operator (#44).

A database error propagates unchanged; ingests enqueued before it stay.
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg

from common.config import Settings
from common.queue import JobQueue

_logger = logging.getLogger(__name__)

#: The ``trigger`` payload value that marks an ingest enqueued by this sweep.
TRIGGER = "transcribe_dead"

_CANDIDATES_SQL = r"""
    SELECT d.video_id, d.priority, v.origin
    FROM (
        SELECT DISTINCT ON (j.video_id)
               j.video_id, j.state, j.error_class, j.priority,
               COALESCE(j.finished_at, j.created_at) AS finished_at
        FROM jobs j
        WHERE j.kind = 'transcribe'
          AND j.video_id IN (SELECT video_id FROM jobs WHERE kind = 'transcribe' AND state = 'dead')
        ORDER BY j.video_id, j.created_at DESC, j.id DESC
    ) d
    JOIN videos v ON v.video_id = d.video_id
    WHERE d.state = 'dead'
      AND d.error_class IS DISTINCT FROM 'PERMANENT_SOURCE'
      AND v.unavailable IS NULL
      AND NOT EXISTS (
            SELECT 1 FROM transcripts t
            WHERE t.video_id = d.video_id AND t.full_text ~ '\S')
      AND NOT EXISTS (
            SELECT 1 FROM jobs i
            WHERE i.video_id = d.video_id AND i.kind = 'ingest'
              AND i.created_at > d.finished_at)
    ORDER BY d.finished_at, d.video_id
    LIMIT %(limit)s
"""


def run_auto_caption_fallback_sweep(
    conn: psycopg.Connection[Any],
    queue: JobQueue,
    *,
    settings: Settings,
    limit: int = 500,
) -> int:
    """Enqueue up to ``limit`` fallback ingests and return how many were enqueued."""
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    candidates: list[tuple[str, int, str]] = []
    if settings.AUTO_CAPTION_FALLBACK:
        with conn.cursor() as cur:
            cur.execute(_CANDIDATES_SQL, {"limit": limit})
            candidates = [(str(r[0]), int(r[1]), str(r[2])) for r in cur.fetchall()]
        # End the read transaction: enqueue() must open its own, not a savepoint
        # inside this one that a closing connection would roll back.
        conn.commit()

    enqueued = 0
    for video_id, priority, origin in candidates:
        job_id = queue.enqueue(
            "ingest",
            video_id,
            payload={"origin": origin, "trigger": TRIGGER},
            priority=priority,
        )
        if job_id is not None:
            enqueued += 1

    # An idle system enqueues nothing on every tick, so that line is DEBUG.
    _logger.log(
        logging.INFO if enqueued else logging.DEBUG,
        "planner.auto_caption_fallback_sweep",
        extra={"enqueued": enqueued, "limit": limit},
    )
    return enqueued
