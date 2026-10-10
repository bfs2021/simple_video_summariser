"""Re-analysis sweep (issue #37; architecture.md §4, C4).

``run_reanalysis_sweep`` is one pass, called by the planner loop (#38) on each
tick. It enqueues one low-priority ``analyze`` job for every video that has a
stored transcript but no analysis at the current ``PROMPT_VERSION``, so bumping
the version re-analyses the corpus without re-downloading or re-transcribing.

Videos whose ``analyze`` job at this key is ``pending``, ``running`` or
``dead`` are skipped: ``dead`` because re-enqueueing would bypass
``max_attempts`` (the ops view, #44, is the retry path). A database error
propagates unchanged; jobs enqueued before it stay, and the next run skips them.
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg

from common.queue import PRIORITY_REANALYSIS, JobQueue, analyze_dedupe_key

_logger = logging.getLogger(__name__)

#: Below new videos (0) and interactive submissions (+10), so a sweep never starves them.
REANALYSIS_PRIORITY = PRIORITY_REANALYSIS

_CANDIDATES_SQL = r"""
    SELECT v.video_id
    FROM videos v
    WHERE EXISTS (
            SELECT 1 FROM transcripts t
            WHERE t.video_id = v.video_id AND t.full_text ~ '\S')
      AND NOT EXISTS (
            SELECT 1 FROM analyses a
            WHERE a.video_id = v.video_id AND a.prompt_version = %(prompt_version)s)
      AND NOT EXISTS (
            SELECT 1 FROM jobs j
            WHERE j.video_id = v.video_id AND j.kind = 'analyze'
              AND j.dedupe_key = %(dedupe_key)s
              AND j.state IN ('pending', 'running', 'dead'))
    ORDER BY v.published_at DESC NULLS LAST, v.video_id
    LIMIT %(limit)s
"""


def run_reanalysis_sweep(
    conn: psycopg.Connection[Any],
    queue: JobQueue,
    *,
    prompt_version: str,
    summarizer_name: str,
    limit: int = 500,
) -> int:
    """Enqueue up to ``limit`` re-analysis jobs and return how many were enqueued."""
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    dedupe_key = analyze_dedupe_key(prompt_version, summarizer_name)
    with conn.cursor() as cur:
        cur.execute(
            _CANDIDATES_SQL,
            {"prompt_version": prompt_version, "dedupe_key": dedupe_key, "limit": limit},
        )
        candidates = [str(row[0]) for row in cur.fetchall()]
    # End the read transaction: enqueue() must open its own, not a savepoint inside
    # this one that the planner's closing connection would roll back (#199; same
    # reason as services/planner/fallback.py, #92).
    conn.commit()

    enqueued = 0
    for video_id in candidates:
        job_id = queue.enqueue(
            "analyze", video_id, dedupe_key=dedupe_key, priority=REANALYSIS_PRIORITY
        )
        if job_id is not None:
            enqueued += 1

    _logger.info(
        "planner.reanalysis_sweep",
        extra={
            "prompt_version": prompt_version,
            "summarizer": summarizer_name,
            "enqueued": enqueued,
            "limit": limit,
            "limit_reached": len(candidates) >= limit,
        },
    )
    return enqueued
