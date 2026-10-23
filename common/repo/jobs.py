"""Job read helpers (issues #28, #39, #40, #44, architecture.md §2).

Writes go through the ``JobQueue`` port; this module only reads what a
handler needs to decide what to do next.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from common.metrics import DURATION_BUCKETS, DurationHistogram


class _Unset:
    """Sentinel distinguishing "no filter" from an explicit ``None`` (#44)."""

    def __repr__(self) -> str:
        return "UNSET"


#: Default for ``list_jobs(error_class=...)``: no filter at all. Passing
#: ``None`` explicitly means "only rows with a NULL ``error_class``" (#114).
UNSET: Final = _Unset()


def latest_job_state(conn: psycopg.Connection[Any], video_id: str, kind: str) -> str | None:
    """State of the newest ``kind`` job for ``video_id``, or ``None`` if there is none.

    "Newest" is by ``created_at`` then ``id``, so a pending job created after a
    dead one (an operator retry) is the latest.
    """
    row = conn.execute(
        """
        SELECT state FROM jobs
        WHERE video_id = %s AND kind = %s
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (video_id, kind),
    ).fetchone()
    return None if row is None else str(row[0])


def latest_job(conn: psycopg.Connection[Any], video_id: str, kind: str) -> tuple[int, str] | None:
    """``(id, state)`` of the newest ``kind`` job for ``video_id``, or ``None`` (#40).

    Ordered like ``latest_job_state``: ``created_at`` then ``id``.
    """
    row = conn.execute(
        """
        SELECT id, state FROM jobs
        WHERE video_id = %s AND kind = %s
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (video_id, kind),
    ).fetchone()
    return None if row is None else (int(row[0]), str(row[1]))


@dataclass(frozen=True, slots=True)
class ActiveJob:
    """The ``pending`` or ``running`` job at one ``(video, kind, key)`` (#101).

    ``forced`` is true only for a JSON ``true`` at ``payload.force``, the exact
    test the analyze handler applies (#30).
    """

    id: int
    state: str
    priority: int
    forced: bool


def active_job(
    conn: psycopg.Connection[Any], video_id: str, kind: str, dedupe_key: str
) -> ActiveJob | None:
    """The active job at ``(video_id, kind, dedupe_key)``, or ``None`` (#101).

    ``jobs_active_uniq`` allows at most one, so there is no ordering. Read-only.
    """
    row = conn.execute(
        """
        SELECT id, state, priority, payload -> 'force' = 'true'::jsonb
        FROM jobs
        WHERE video_id = %s AND kind = %s AND dedupe_key = %s
          AND state IN ('pending', 'running')
        """,
        (video_id, kind, dedupe_key),
    ).fetchone()
    if row is None:
        return None
    return ActiveJob(id=int(row[0]), state=str(row[1]), priority=int(row[2]), forced=bool(row[3]))


def queue_depth(conn: psycopg.Connection[Any]) -> dict[tuple[str, str], int]:
    """Number of jobs per ``(kind, state)``, for ``/healthz`` (#39) and ``/metrics`` (#60).

    Only groups that have at least one row appear; zero-filling the known
    kinds and states is the caller's job. Read-only.
    """
    rows = conn.execute("SELECT kind, state, count(*) FROM jobs GROUP BY kind, state").fetchall()
    return {(str(kind), str(state)): int(count) for kind, state, count in rows}


@dataclass(frozen=True, slots=True)
class JobRow:
    """One ``jobs`` row for the Ops view (#44). ``payload`` is never exposed."""

    id: int
    video_id: str
    video_title: str | None
    kind: str
    dedupe_key: str
    state: str
    priority: int
    attempts: int
    error_class: str | None
    last_error: str | None
    run_after: datetime
    locked_by: str | None
    heartbeat_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    #: ``jobs.progress`` (#245): live figures of a running transcribe job, else ``None``.
    progress: dict[str, Any] | None
    #: ``payload.duration_sec`` when it is a plausible number, else ``None`` (#245).
    #: Lets the API estimate a pending transcribe job without exposing ``payload``.
    payload_duration_sec: float | None


#: The columns ``JobRow`` is built from, shared by the list and the single-row read.
_JOB_COLUMNS = """
    j.id, j.video_id, v.title AS video_title, j.kind, j.dedupe_key,
    j.state, j.priority, j.attempts, j.error_class, j.last_error,
    j.run_after, j.locked_by, j.heartbeat_at, j.finished_at, j.created_at,
    j.progress,
    CASE WHEN jsonb_typeof(j.payload -> 'duration_sec') = 'number'
          AND abs((j.payload ->> 'duration_sec')::numeric) < 1e9
         THEN (j.payload ->> 'duration_sec')::float8
    END AS payload_duration_sec"""

_LIST_JOBS_SQL = f"""
    SELECT {_JOB_COLUMNS}
    FROM jobs j
    LEFT JOIN videos v ON v.video_id = j.video_id
    WHERE (%(state)s::text IS NULL OR j.state = %(state)s)
      AND (%(kind)s::text IS NULL OR j.kind = %(kind)s)
      AND (
          %(error_class_set)s = false
          OR (%(error_class)s::text IS NULL AND j.error_class IS NULL)
          OR j.error_class = %(error_class)s
      )
      AND (%(before_id)s::bigint IS NULL OR j.id < %(before_id)s)
    ORDER BY j.id DESC
    LIMIT %(limit)s
"""


def list_jobs(
    conn: psycopg.Connection[Any],
    *,
    state: str | None = None,
    kind: str | None = None,
    error_class: str | None | _Unset = UNSET,
    before_id: int | None = None,
    limit: int,
) -> list[JobRow]:
    """One page of ``jobs``, newest first, for the Ops view (#44).

    Filters combine with AND. ``error_class`` is a three-way switch: the
    default ``UNSET`` means "no filter", ``None`` means "only rows whose
    ``error_class`` is NULL" (#114's reaped-but-unclassed jobs), and any other
    string filters on that exact value. ``before_id`` is a keyset cursor:
    only rows with ``id < before_id``. Fetches ``limit + 1`` rows so the
    caller can see whether another page follows without a separate
    ``count(*)``; trimming to ``limit`` is the caller's job. One statement,
    bound parameters only, read-only.
    """
    error_class_set = error_class is not UNSET
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            _LIST_JOBS_SQL,
            {
                "state": state,
                "kind": kind,
                "error_class_set": error_class_set,
                "error_class": None if error_class is UNSET else error_class,
                "before_id": before_id,
                "limit": limit,
            },
        )
        rows = cur.fetchall()
    return [JobRow(**row) for row in rows]


def get_job(conn: psycopg.Connection[Any], job_id: int) -> JobRow | None:
    """One job in the Ops view's item shape (#44), or ``None`` without a row.

    Used after ``PostgresQueue.retry_dead`` to answer with the shared
    ``JobOut`` shape (``video_title`` included) instead of duplicating the
    join inline in the route.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT {_JOB_COLUMNS}
            FROM jobs j
            LEFT JOIN videos v ON v.video_id = j.video_id
            WHERE j.id = %s
            """,
            (job_id,),
        )
        row = cur.fetchone()
    return None if row is None else JobRow(**row)


def write_progress(
    conn: psycopg.Connection[Any], job_id: int, worker: str, figures: dict[str, Any]
) -> bool:
    """Store ``figures`` (+ the database's ``updated_at``) as a running job's progress (#245).

    One small ``UPDATE`` of ``jobs.progress`` only: ``heartbeat_at`` is the queue's
    business and is not touched. It locks just this job's row for the length of the
    statement, and the claim only looks at ``pending`` rows, so it never blocks a
    claim. Guarded by ``locked_by`` and ``state = 'running'`` like every outcome
    write, so a job reaped and reclaimed meanwhile is left alone: returns whether a
    row was written. Commits through its own transaction block.
    """
    with conn.transaction():
        cur = conn.execute(
            """
            UPDATE jobs
            SET progress = %(figures)s || jsonb_build_object('updated_at', now())
            WHERE id = %(id)s AND state = 'running' AND locked_by = %(worker)s
            """,
            {"figures": Jsonb(figures), "id": job_id, "worker": worker},
        )
        return cur.rowcount > 0


def duration_histograms(conn: psycopg.Connection[Any]) -> dict[str, DurationHistogram]:
    """Duration histogram per job kind for ``job_duration_seconds`` (#60), one statement.

    Over ``done`` jobs with a recorded ``started_at`` (older rows are excluded,
    not counted as 0 s): ``finished_at - started_at`` of the last attempt, in
    exact numeric seconds, so 60 s is in ``le="60"`` and 60.001 s is not. The
    ``FILTER`` counts are cumulative, as the exposition format wants. Only
    kinds with at least one such job appear.
    """
    filters = ", ".join(f"count(*) FILTER (WHERE s <= {bound})" for bound in DURATION_BUCKETS)
    rows = conn.execute(
        f"""
        WITH d AS (
            SELECT kind, extract(epoch FROM finished_at - started_at) AS s
            FROM jobs
            WHERE state = 'done' AND started_at IS NOT NULL AND finished_at IS NOT NULL
              AND finished_at >= started_at
        )
        SELECT kind, {filters}, count(*), COALESCE(sum(s), 0)::float8
        FROM d GROUP BY kind
        """
    ).fetchall()
    n = len(DURATION_BUCKETS)
    return {
        str(row[0]): DurationHistogram(
            tuple(int(c) for c in row[1 : 1 + n]), int(row[1 + n]), float(row[2 + n])
        )
        for row in rows
    }
