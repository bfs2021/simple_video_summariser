"""Re-analysis sweep (issue #37; architecture.md §4, C4).

``run_reanalysis_sweep`` is one pass, called by the planner loop (#38) on each
tick. It enqueues one low-priority ``analyze`` job for every video that has a
stored transcript but no analysis at the current ``PROMPT_VERSION``, so bumping
the version re-analyses the corpus without re-downloading or re-transcribing.

Videos whose ``analyze`` job at this key is ``pending``, ``running`` or
``dead`` are skipped: ``dead`` because re-enqueueing would bypass
``max_attempts`` (the ops view, #44, is the retry path). A database error
propagates unchanged; jobs enqueued before it stay, and the next run skips them.

Spend cap (#99). With ``SUMMARIZER=anthropic`` the sweep also stops at
``SWEEP_MAX_JOBS_ANTHROPIC`` jobs *in total per prompt version*: the number of
``analyze`` jobs already enqueued at this key and ``REANALYSIS_PRIORITY``, in any
state, is read from ``jobs`` on every run, so the cap survives a planner restart.
A run enqueues at most ``min(limit, cap - already_enqueued)``. Jobs enqueued by
the transcribe handler or interactively use other priorities and are not counted.
``plan_reanalysis_sweep`` is the one computation behind both the live run and the
read-only ``--sweep-dry-run`` report, so they cannot disagree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import psycopg

from common.config import Settings
from common.queue import PRIORITY_REANALYSIS, JobQueue, analyze_dedupe_key

_logger = logging.getLogger(__name__)

#: Below new videos (0) and interactive submissions (+10), so a sweep never starves them.
REANALYSIS_PRIORITY = PRIORITY_REANALYSIS

#: The cap's default, read from the setting so the two cannot drift apart.
DEFAULT_MAX_JOBS_ANTHROPIC = int(Settings.model_fields["SWEEP_MAX_JOBS_ANTHROPIC"].default)

_ELIGIBLE_SQL = r"""
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
"""

_CANDIDATES_SQL = (
    "SELECT v.video_id "
    + _ELIGIBLE_SQL
    + " ORDER BY v.published_at DESC NULLS LAST, v.video_id LIMIT %(limit)s"
)
_COUNT_SQL = "SELECT count(*) " + _ELIGIBLE_SQL

_ALREADY_ENQUEUED_SQL = """
    SELECT count(*) FROM jobs
    WHERE kind = 'analyze' AND dedupe_key = %(dedupe_key)s AND priority = %(priority)s
"""


@dataclass(frozen=True)
class SweepPlan:
    """What a sweep pass would do right now (#99)."""

    prompt_version: str
    summarizer: str
    candidates: int  #: every eligible video, not truncated at ``limit``
    already_enqueued: int
    cap: int | None  #: ``None`` when the summarizer is not capped
    limit: int
    would_enqueue_this_run: int
    would_remain_after_cap: int  #: candidates the cap leaves unanalysed
    cap_stopped: bool  #: the cap, not ``limit`` or running out, ends this run
    video_ids: tuple[str, ...]  #: the videos a live run would enqueue


def plan_reanalysis_sweep(
    conn: psycopg.Connection[Any],
    *,
    prompt_version: str,
    summarizer_name: str,
    limit: int = 500,
    max_jobs_anthropic: int = DEFAULT_MAX_JOBS_ANTHROPIC,
) -> SweepPlan:
    """Read the database and work out one pass. Writes nothing; ends its read transaction."""
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    dedupe_key = analyze_dedupe_key(prompt_version, summarizer_name)
    cap = max_jobs_anthropic if summarizer_name == "anthropic" else None
    params = {"prompt_version": prompt_version, "dedupe_key": dedupe_key}
    with conn.cursor() as cur:
        cur.execute(
            _ALREADY_ENQUEUED_SQL, {"dedupe_key": dedupe_key, "priority": REANALYSIS_PRIORITY}
        )
        row = cur.fetchone()
        already = int(row[0]) if row is not None else 0
        cur.execute(_COUNT_SQL, params)
        row = cur.fetchone()
        total = int(row[0]) if row is not None else 0
        room = None if cap is None else max(0, cap - already)
        take = min(limit, total, total if room is None else room)
        video_ids: tuple[str, ...] = ()
        if take > 0:
            cur.execute(_CANDIDATES_SQL, {**params, "limit": take})
            video_ids = tuple(str(r[0]) for r in cur.fetchall())
    # End the read transaction: enqueue() must open its own, not a savepoint inside
    # this one that the planner's closing connection would roll back (#199; same
    # reason as services/planner/fallback.py, #92).
    conn.commit()
    return SweepPlan(
        prompt_version=prompt_version,
        summarizer=summarizer_name,
        candidates=total,
        already_enqueued=already,
        cap=cap,
        limit=limit,
        would_enqueue_this_run=len(video_ids),
        would_remain_after_cap=0 if room is None else max(0, total - room),
        cap_stopped=room is not None and room < min(limit, total),
        video_ids=video_ids,
    )


def run_reanalysis_sweep(
    conn: psycopg.Connection[Any],
    queue: JobQueue,
    *,
    prompt_version: str,
    summarizer_name: str,
    limit: int = 500,
    max_jobs_anthropic: int = DEFAULT_MAX_JOBS_ANTHROPIC,
) -> int:
    """Enqueue up to ``limit`` re-analysis jobs (within the cap) and return how many."""
    plan = plan_reanalysis_sweep(
        conn,
        prompt_version=prompt_version,
        summarizer_name=summarizer_name,
        limit=limit,
        max_jobs_anthropic=max_jobs_anthropic,
    )
    dedupe_key = analyze_dedupe_key(prompt_version, summarizer_name)

    enqueued = 0
    for video_id in plan.video_ids:
        job_id = queue.enqueue(
            "analyze", video_id, dedupe_key=dedupe_key, priority=REANALYSIS_PRIORITY
        )
        if job_id is not None:
            enqueued += 1

    if plan.cap_stopped:
        _logger.warning(
            "planner.reanalysis_sweep.cap_reached",
            extra={
                "prompt_version": prompt_version,
                "summarizer": summarizer_name,
                "cap": plan.cap,
                "already_enqueued": plan.already_enqueued,
                "unenqueued": plan.candidates - len(plan.video_ids),
            },
        )
    _logger.info(
        "planner.reanalysis_sweep",
        extra={
            "prompt_version": prompt_version,
            "summarizer": summarizer_name,
            "enqueued": enqueued,
            "limit": limit,
            "limit_reached": len(plan.video_ids) >= limit,
            "cap": plan.cap,
            "already_enqueued": plan.already_enqueued,
        },
    )
    return enqueued
