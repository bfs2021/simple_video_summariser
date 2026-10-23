"""The Anthropic batches an analyze job has submitted (issue #89).

Each call is one statement in its own transaction, so no transaction is held
open across a poll.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import psycopg

from common.errors import LLMUnavailableError
from common.models import BatchCheckpoint

_COLUMNS = "step_key, batch_id, request_sha256, submitted_at"


def get_checkpoint(
    conn: psycopg.Connection[Any], job_id: int, step_key: str
) -> BatchCheckpoint | None:
    with conn.transaction():
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM llm_batches WHERE job_id = %s AND step_key = %s",
            (job_id, step_key),
        ).fetchone()
    return _checkpoint(row) if row is not None else None


def insert_checkpoint(
    conn: psycopg.Connection[Any],
    job_id: int,
    step_key: str,
    batch_id: str,
    request_sha256: str,
) -> BatchCheckpoint:
    """Store ``batch_id`` for the step; if a row already exists, return that row instead."""
    with conn.transaction():
        row = conn.execute(
            f"""
            INSERT INTO llm_batches (job_id, step_key, batch_id, request_sha256)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (job_id, step_key) DO NOTHING
            RETURNING {_COLUMNS}
            """,
            (job_id, step_key, batch_id, request_sha256),
        ).fetchone()
    if row is not None:
        return _checkpoint(row)
    existing = get_checkpoint(conn, job_id, step_key)
    if existing is None:
        raise LLMUnavailableError(f"checkpoint for step {step_key!r} changed under the insert")
    return existing


def delete_checkpoint(conn: psycopg.Connection[Any], job_id: int, step_key: str) -> None:
    with conn.transaction():
        conn.execute(
            "DELETE FROM llm_batches WHERE job_id = %s AND step_key = %s",
            (job_id, step_key),
        )


def delete_job_checkpoints(conn: psycopg.Connection[Any], job_id: int) -> None:
    with conn.transaction():
        conn.execute("DELETE FROM llm_batches WHERE job_id = %s", (job_id,))


@dataclass(frozen=True, slots=True)
class OrphanBatch:
    """A row whose job is over (#160). ``expired``: older than ``expired_after``, by the DB clock."""

    job_id: int
    step_key: str
    batch_id: str
    expired: bool


def list_orphans(
    conn: psycopg.Connection[Any], *, limit: int, expired_after: timedelta
) -> list[OrphanBatch]:
    """Rows of ``dead`` / ``done`` jobs, oldest ``submitted_at`` first, at most ``limit``.

    Rows of ``pending`` and ``running`` jobs are never returned: #89's resume
    depends on them.
    """
    with conn.transaction():
        rows = conn.execute(
            """
            SELECT b.job_id, b.step_key, b.batch_id, b.submitted_at < now() - %s
            FROM llm_batches b
            JOIN jobs j ON j.id = b.job_id
            WHERE j.state IN ('dead', 'done')
            ORDER BY b.submitted_at, b.job_id, b.step_key
            LIMIT %s
            """,
            (expired_after, limit),
        ).fetchall()
    return [OrphanBatch(job_id, step_key, batch_id, bool(expired)) for job_id, step_key, batch_id, expired in rows]


def delete_orphan(
    conn: psycopg.Connection[Any], job_id: int, step_key: str, batch_id: str
) -> bool:
    """Delete one orphan row, only while its job is still ``dead`` or ``done``.

    One statement: the job's state is re-checked (and share-locked, so a
    concurrent ``retry_dead`` waits) in the same ``DELETE``. Returns whether a
    row went.
    """
    with conn.transaction():
        row = conn.execute(
            """
            DELETE FROM llm_batches b
            WHERE b.job_id = %s AND b.step_key = %s AND b.batch_id = %s
              AND EXISTS (
                SELECT 1 FROM jobs j
                WHERE j.id = b.job_id AND j.state IN ('dead', 'done')
                FOR SHARE OF j
              )
            RETURNING 1
            """,
            (job_id, step_key, batch_id),
        ).fetchone()
    return row is not None


def _checkpoint(row: Any) -> BatchCheckpoint:
    step_key, batch_id, request_sha256, submitted_at = row
    return BatchCheckpoint(
        step_key=step_key,
        batch_id=batch_id,
        request_sha256=request_sha256,
        submitted_at=submitted_at,
    )
