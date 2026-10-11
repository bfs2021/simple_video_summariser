"""Backfill request rows (issue #115; architecture.md 8.3, 4).

``POST /channels/{id}/backfill-requests`` only records a row here; the
planner's ``channel_backfill`` task does the listing and the enqueueing. Like
the other repo modules, nothing in here commits: the caller owns the
transaction.

At most one ``pending`` or ``running`` request exists per channel, decided by
the partial unique index ``backfill_requests_active_uniq`` and not by a read
before the insert.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
from psycopg.rows import class_row
from psycopg.types.json import Jsonb

#: ``jobs.priority`` of a backfill ``ingest`` job (``common.queue.PRIORITY_BACKFILL``).
#: Kept literal here so a read query does not import the queue module.
_BACKFILL_PRIORITY = -10

#: Marks a request that an operator cancelled (#116); the state is ``failed``.
CANCELLED = "CANCELLED"

_NAMESPACE = uuid.UUID("5d0b8e0c-6a7e-4c2e-9f55-1b1c2d3e4f50")


def request_backfill_id(request_id: int) -> str:
    """The ``backfill_id`` every ``ingest`` job of one request carries.

    Derived from the request id, so all slices of a request share one id and
    ``POST /channels/{id}/backfill/cancel`` can name the whole request.
    """
    return str(uuid.uuid5(_NAMESPACE, f"backfill-request:{request_id}"))


@dataclass(frozen=True, slots=True)
class BackfillRequestRow:
    id: int
    channel_id: str
    limit_n: int
    dry_run: bool
    state: str
    attempts: int
    enqueued: int
    result: dict[str, Any] | None
    error_class: str | None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None


_COLUMNS = (
    "id, channel_id, limit_n, dry_run, state, attempts, enqueued, result, "
    "error_class, created_at, updated_at, finished_at"
)

_ACTIVE = "state IN ('pending', 'running')"


@dataclass(frozen=True, slots=True)
class CreateOutcome:
    """``created`` is false when the channel already has an active request."""

    request_id: int
    created: bool


def create_request(
    conn: psycopg.Connection[Any], channel_id: str, *, limit_n: int, dry_run: bool
) -> CreateOutcome:
    """Insert a ``pending`` request, or report the channel's active one.

    One ``INSERT ... ON CONFLICT DO NOTHING`` against the partial unique
    index. A concurrent insert for the same channel waits for the other
    transaction and then takes the conflict path, so exactly one wins. The
    read of the winner's id follows the conflict; if that request finished in
    between, the insert is tried again.
    """
    for _ in range(3):
        row = conn.execute(
            f"""
            INSERT INTO backfill_requests (channel_id, limit_n, dry_run, state)
            VALUES (%s, %s, %s, 'pending')
            ON CONFLICT (channel_id) WHERE {_ACTIVE} DO NOTHING
            RETURNING id
            """,
            (channel_id, limit_n, dry_run),
        ).fetchone()
        if row is not None:
            return CreateOutcome(int(row[0]), True)
        existing = conn.execute(
            f"SELECT id FROM backfill_requests WHERE channel_id = %s AND {_ACTIVE}",
            (channel_id,),
        ).fetchone()
        if existing is not None:
            return CreateOutcome(int(existing[0]), False)
    raise RuntimeError("backfill request kept appearing and finishing")  # pragma: no cover


def get_request(
    conn: psycopg.Connection[Any], channel_id: str, request_id: int
) -> BackfillRequestRow | None:
    """The request, only if it belongs to ``channel_id``."""
    with conn.cursor(row_factory=class_row(BackfillRequestRow)) as cur:
        cur.execute(
            f"SELECT {_COLUMNS} FROM backfill_requests WHERE id = %s AND channel_id = %s",
            (request_id, channel_id),
        )
        return cur.fetchone()


def claim_next(conn: psycopg.Connection[Any]) -> BackfillRequestRow | None:
    """The oldest ``pending`` or ``running`` request, now ``running``; else ``None``.

    ``FOR UPDATE SKIP LOCKED``: a second claimer skips a request that is
    locked instead of waiting for it. The caller commits to make it durable.
    """
    with conn.cursor(row_factory=class_row(BackfillRequestRow)) as cur:
        cur.execute(
            f"""
            UPDATE backfill_requests SET state = 'running', updated_at = now()
            WHERE id = (
                SELECT id FROM backfill_requests
                WHERE {_ACTIVE}
                ORDER BY created_at, id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING {_COLUMNS}
            """
        )
        return cur.fetchone()


def count_pending_backfill_ingests(conn: psycopg.Connection[Any]) -> int:
    """How many ``pending`` ``ingest`` jobs sit at the backfill priority."""
    row = conn.execute(
        "SELECT count(*) FROM jobs WHERE kind = 'ingest' AND state = 'pending' AND priority = %s",
        (_BACKFILL_PRIORITY,),
    ).fetchone()
    return 0 if row is None else int(row[0])


def record_progress(
    conn: psycopg.Connection[Any],
    request_id: int,
    *,
    enqueued_this_run: int,
    result: dict[str, Any],
    done: bool,
) -> bool:
    """Add a run's ``enqueued`` and store its ``result``; ``done`` ends the request.

    Touches only an active request, so a request cancelled while the run was
    listing stays terminal. Returns whether a row was updated.
    """
    cur = conn.execute(
        f"""
        UPDATE backfill_requests
        SET enqueued = enqueued + %s,
            result = %s,
            error_class = NULL,
            state = CASE WHEN %s THEN 'done' ELSE 'running' END,
            finished_at = CASE WHEN %s THEN now() ELSE NULL END,
            updated_at = now()
        WHERE id = %s AND {_ACTIVE}
        """,
        (enqueued_this_run, Jsonb(result), done, done, request_id),
    )
    return cur.rowcount == 1


def touch(conn: psycopg.Connection[Any], request_id: int) -> None:
    """Bump ``updated_at`` of an active request (a run that did no work)."""
    conn.execute(
        f"UPDATE backfill_requests SET updated_at = now() WHERE id = %s AND {_ACTIVE}",
        (request_id,),
    )


def record_failure(
    conn: psycopg.Connection[Any],
    request_id: int,
    *,
    error_class: str,
    max_attempts: int,
    fatal: bool,
) -> str | None:
    """Count one failed attempt; return the new state, or ``None`` if not active.

    ``failed`` when ``fatal`` or the attempt count reaches ``max_attempts``;
    otherwise the request stays ``running`` and the next run retries it. Only
    the exception's class name is stored.
    """
    row = conn.execute(
        f"""
        UPDATE backfill_requests
        SET attempts = attempts + 1,
            error_class = %s,
            state = CASE WHEN %s OR attempts + 1 >= %s THEN 'failed' ELSE 'running' END,
            finished_at = CASE WHEN %s OR attempts + 1 >= %s THEN now() ELSE NULL END,
            updated_at = now()
        WHERE id = %s AND {_ACTIVE}
        RETURNING state
        """,
        (error_class, fatal, max_attempts, fatal, max_attempts, request_id),
    ).fetchone()
    return None if row is None else str(row[0])


def cancel_active(conn: psycopg.Connection[Any], channel_id: str, backfill_id: str) -> int | None:
    """End the channel's active request if ``backfill_id`` is its id (#116).

    The request becomes ``failed`` with ``error_class = 'CANCELLED'``. Returns
    the request id, or ``None`` when the channel has no active request or the
    id names something else (a synchronous backfill, say), which is left alone.
    """
    row = conn.execute(
        f"SELECT id FROM backfill_requests WHERE channel_id = %s AND {_ACTIVE} FOR UPDATE",
        (channel_id,),
    ).fetchone()
    if row is None or request_backfill_id(int(row[0])) != backfill_id:
        return None
    conn.execute(
        """
        UPDATE backfill_requests
        SET state = 'failed', error_class = %s, finished_at = now(), updated_at = now()
        WHERE id = %s
        """,
        (CANCELLED, row[0]),
    )
    return int(row[0])
