"""``common.repo.llm_batches`` against Postgres (issue #89)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import psycopg
import pytest

from common.queue import PostgresQueue
from common.repo.llm_batches import (
    delete_checkpoint,
    delete_job_checkpoints,
    delete_orphan,
    get_checkpoint,
    insert_checkpoint,
    list_orphans,
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


# --- orphans: rows of dead / done jobs (issue #160) ----------------------------------


def _job(conn: psycopg.Connection[Any], state: str, n: int) -> int:
    vid = f"orphan{n:05d}"[:11]
    conn.execute("INSERT INTO videos (video_id) VALUES (%s)", (vid,))
    row = conn.execute(
        "INSERT INTO jobs (kind, video_id, dedupe_key, state) VALUES ('analyze', %s, 'v1:anthropic', %s) "
        "RETURNING id",
        (vid, state),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _age(conn: psycopg.Connection[Any], job: int, step: str, hours: int) -> None:
    conn.execute(
        "UPDATE llm_batches SET submitted_at = now() - make_interval(hours => %s) "
        "WHERE job_id = %s AND step_key = %s",
        (hours, job, step),
    )


def test_only_rows_of_dead_and_done_jobs_are_orphans(conn: psycopg.Connection[Any]) -> None:
    ids = {s: _job(conn, s, i) for i, s in enumerate(["pending", "running", "dead", "done"])}
    for state, job in ids.items():
        insert_checkpoint(conn, job, "roster", f"msgbatch_{state}", "sha")

    got = list_orphans(conn, limit=10, expired_after=timedelta(hours=24))

    assert {(o.job_id, o.step_key, o.batch_id) for o in got} == {
        (ids["dead"], "roster", "msgbatch_dead"),
        (ids["done"], "roster", "msgbatch_done"),
    }


def test_orphans_come_oldest_first_and_limit_bounds_them(conn: psycopg.Connection[Any]) -> None:
    jobs = [_job(conn, "dead", i) for i in range(3)]
    for job, hours in zip(jobs, (1, 3, 2), strict=True):
        insert_checkpoint(conn, job, "roster", f"msgbatch_{job}", "sha")
        _age(conn, job, "roster", hours)
    day = timedelta(hours=24)

    first = list_orphans(conn, limit=2, expired_after=day)
    assert [o.job_id for o in first] == [jobs[1], jobs[2]]

    for o in first:
        assert delete_orphan(conn, o.job_id, o.step_key, o.batch_id) is True
    assert [o.job_id for o in list_orphans(conn, limit=2, expired_after=day)] == [jobs[0]]


def test_expired_is_judged_against_expired_after(conn: psycopg.Connection[Any]) -> None:
    old, fresh = _job(conn, "dead", 1), _job(conn, "dead", 2)
    for job in (old, fresh):
        insert_checkpoint(conn, job, "roster", f"msgbatch_{job}", "sha")
    _age(conn, old, "roster", 5)

    got = {o.job_id: o.expired for o in list_orphans(conn, limit=10, expired_after=timedelta(hours=4))}

    assert got == {old: True, fresh: False}


def test_delete_orphan_removes_the_row_of_a_dead_job(conn: psycopg.Connection[Any]) -> None:
    job = _job(conn, "dead", 1)
    insert_checkpoint(conn, job, "roster", "msgbatch_a", "sha")

    assert delete_orphan(conn, job, "roster", "msgbatch_a") is True
    assert get_checkpoint(conn, job, "roster") is None
    assert delete_orphan(conn, job, "roster", "msgbatch_a") is False  # already gone


def test_a_job_revived_after_listing_keeps_its_row(conn: psycopg.Connection[Any]) -> None:
    queue = PostgresQueue(conn)
    created = queue.enqueue("analyze", VIDEO, dedupe_key="v1:anthropic")
    assert created is not None
    conn.execute("UPDATE jobs SET state = 'dead' WHERE id = %s", (created,))
    insert_checkpoint(conn, created, "roster", "msgbatch_a", "sha")
    (orphan,) = list_orphans(conn, limit=10, expired_after=timedelta(hours=24))

    queue.retry_dead(created)
    assert delete_orphan(conn, orphan.job_id, orphan.step_key, orphan.batch_id) is False

    assert get_checkpoint(conn, created, "roster") is not None


def test_a_replaced_batch_id_is_not_deleted(conn: psycopg.Connection[Any]) -> None:
    job = _job(conn, "dead", 1)
    insert_checkpoint(conn, job, "roster", "msgbatch_new", "sha")

    assert delete_orphan(conn, job, "roster", "msgbatch_old") is False

    assert get_checkpoint(conn, job, "roster") is not None
