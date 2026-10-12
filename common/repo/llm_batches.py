"""The Anthropic batches an analyze job has submitted (issue #89).

Each call is one statement in its own transaction, so no transaction is held
open across a poll.
"""

from __future__ import annotations

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


def _checkpoint(row: Any) -> BatchCheckpoint:
    step_key, batch_id, request_sha256, submitted_at = row
    return BatchCheckpoint(
        step_key=step_key,
        batch_id=batch_id,
        request_sha256=request_sha256,
        submitted_at=submitted_at,
    )
