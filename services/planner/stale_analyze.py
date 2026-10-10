"""Stale ``analyze`` jobs after a PROMPT_VERSION rollout (issue #100; architecture.md §11.6).

``resolve_stale_analyze_jobs`` is one pass, called by the planner loop (#38).
An analyzer claims only jobs whose ``dedupe_key`` is its own
``<PROMPT_VERSION>:<summarizer>``, so a pending ``analyze`` job left on another
version would otherwise wait forever. Each tick this re-keys such a job to the
current version, keeping its priority, or marks it superseded (see
``PostgresQueue.resolve_stale_analyze`` for the rules). The task runs before
``reanalysis_sweep`` so the sweep sees a re-keyed job as active and adds no
duplicate at the lower re-analysis priority.

A database error propagates unchanged: scheduling is #38's job.
"""

from __future__ import annotations

import logging

from common.config import Settings
from common.queue import PostgresQueue

_logger = logging.getLogger(__name__)


def resolve_stale_analyze_jobs(queue: PostgresQueue, settings: Settings) -> int:
    """Run one pass and return how many jobs it re-keyed plus superseded."""
    result = queue.resolve_stale_analyze(settings.PROMPT_VERSION)
    _logger.info(
        "planner.stale_analyze_resolved",
        extra={
            "prompt_version": settings.PROMPT_VERSION,
            "rekeyed": result.rekeyed,
            "superseded": result.superseded,
        },
    )
    return result.rekeyed + result.superseded
