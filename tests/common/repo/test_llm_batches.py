"""``common.repo.llm_batches`` against Postgres (issue #89)."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest

from common.queue import PostgresQueue
from common.repo.llm_batches import (
    delete_checkpoint,
    delete_job_checkpoints,
    get_checkpoint,
    insert_checkpoint,
)

pytestmark = pytest.mark.integration

VIDEO = "dQw4w9WgXcQ"


@pytest.fixture
def job_id(conn: psycopg.Connection[Any]) -> int:
    queue = PostgresQueue(conn)
    created = queue.enqueue("analyze", VIDEO, dedupe_key="v1:anthropic")
    conn.commit()
    assert created is not None
    return created


def test_a_step_is_stored_and_read_back(conn: psycopg.Connection[Any], job_id: int) -> None:
    inserted = insert_checkpoint(conn, job_id, "roster", "msgbatch_a", "sha-1")

    got = get_checkpoint(conn, job_id, "roster")

    assert got == inserted
    assert got is not None
    assert (got.batch_id, got.request_sha256) == ("msgbatch_a", "sha-1")
    assert got.submitted_at is not None


def test_a_second_insert_for_the_same_step_returns_the_stored_row(
    conn: psycopg.Connection[Any], job_id: int
) -> None:
    first = insert_checkpoint(conn, job_id, "roster", "msgbatch_a", "sha-1")

    second = insert_checkpoint(conn, job_id, "roster", "msgbatch_b", "sha-2")

    assert second == first
    assert get_checkpoint(conn, job_id, "roster") == first


def test_steps_are_independent(conn: psycopg.Connection[Any], job_id: int) -> None:
    insert_checkpoint(conn, job_id, "roster", "msgbatch_a", "sha-1")

    assert get_checkpoint(conn, job_id, "reduce") is None


def test_delete_removes_one_step(conn: psycopg.Connection[Any], job_id: int) -> None:
    insert_checkpoint(conn, job_id, "roster", "msgbatch_a", "sha-1")
    insert_checkpoint(conn, job_id, "reduce", "msgbatch_b", "sha-2")

    delete_checkpoint(conn, job_id, "roster")

    assert get_checkpoint(conn, job_id, "roster") is None
    assert get_checkpoint(conn, job_id, "reduce") is not None


def test_delete_for_a_job_removes_all_its_steps(
    conn: psycopg.Connection[Any], job_id: int
) -> None:
    insert_checkpoint(conn, job_id, "roster", "msgbatch_a", "sha-1")
    insert_checkpoint(conn, job_id, "chunks", "msgbatch_b", "sha-2")

    delete_job_checkpoints(conn, job_id)

    assert get_checkpoint(conn, job_id, "roster") is None
    assert get_checkpoint(conn, job_id, "chunks") is None


def test_deleting_the_job_cascades_to_its_checkpoints(
    conn: psycopg.Connection[Any], job_id: int
) -> None:
    insert_checkpoint(conn, job_id, "roster", "msgbatch_a", "sha-1")
    conn.execute("DELETE FROM jobs WHERE id = %s", (job_id,))
    conn.commit()

    row = conn.execute("SELECT count(*) FROM llm_batches WHERE job_id = %s", (job_id,)).fetchone()
    assert row is not None and row[0] == 0
