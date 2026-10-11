"""``common.repo.backfill_requests`` against Postgres (issue #115)."""

from __future__ import annotations

import threading
import time
from typing import Any

import psycopg
import pytest

from common.repo.backfill_requests import (
    CANCELLED,
    cancel_active,
    claim_next,
    count_pending_backfill_ingests,
    create_request,
    get_request,
    record_failure,
    record_progress,
    request_backfill_id,
)

pytestmark = pytest.mark.integration

CHANNEL = "UCuAXFkgsw1L7xaCfnd5JJOw"
OTHER = "UCxxxxxxxxxxxxxxxxxxxxxx"


@pytest.fixture
def channels(conn: psycopg.Connection[Any]) -> None:
    conn.execute("INSERT INTO channels (channel_id) VALUES (%s), (%s)", (CHANNEL, OTHER))
    conn.commit()


def _state(conn: psycopg.Connection[Any], rid: int) -> str:
    row = conn.execute("SELECT state FROM backfill_requests WHERE id = %s", (rid,)).fetchone()
    assert row is not None
    return str(row[0])


def test_second_create_for_a_channel_reports_the_existing_request(
    conn: psycopg.Connection[Any], channels: None
) -> None:
    first = create_request(conn, CHANNEL, limit_n=10, dry_run=True)
    second = create_request(conn, CHANNEL, limit_n=20, dry_run=False)
    other = create_request(conn, OTHER, limit_n=5, dry_run=True)

    assert first.created and not second.created and other.created
    assert second.request_id == first.request_id
    row = get_request(conn, CHANNEL, first.request_id)
    assert row is not None and (row.limit_n, row.dry_run, row.state) == (10, True, "pending")


def test_a_finished_request_frees_the_channel(
    conn: psycopg.Connection[Any], channels: None
) -> None:
    first = create_request(conn, CHANNEL, limit_n=10, dry_run=True)
    conn.execute("UPDATE backfill_requests SET state = 'done' WHERE id = %s", (first.request_id,))

    again = create_request(conn, CHANNEL, limit_n=10, dry_run=True)

    assert again.created and again.request_id != first.request_id


def test_get_request_is_scoped_to_its_channel(
    conn: psycopg.Connection[Any], channels: None
) -> None:
    created = create_request(conn, CHANNEL, limit_n=10, dry_run=True)

    assert get_request(conn, OTHER, created.request_id) is None
    assert get_request(conn, CHANNEL, created.request_id + 1) is None


def test_claim_next_takes_the_oldest_active_request_and_marks_it_running(
    conn: psycopg.Connection[Any], channels: None
) -> None:
    conn.execute(
        "INSERT INTO backfill_requests (channel_id, limit_n, dry_run, state, created_at)"
        " VALUES (%s, 5, true, 'done', now() - interval '3 hours'),"
        "        (%s, 5, true, 'pending', now() - interval '1 hour'),"
        "        (%s, 5, true, 'pending', now() - interval '2 hours')",
        (CHANNEL, CHANNEL, OTHER),
    )

    claimed = claim_next(conn)

    assert claimed is not None and claimed.channel_id == OTHER
    assert claimed.state == "running"
    assert _state(conn, claimed.id) == "running"


def test_claim_next_resumes_a_running_request_and_is_empty_without_one(
    conn: psycopg.Connection[Any], channels: None
) -> None:
    assert claim_next(conn) is None
    created = create_request(conn, CHANNEL, limit_n=5, dry_run=False)
    conn.execute("UPDATE backfill_requests SET state = 'running'")

    claimed = claim_next(conn)

    assert claimed is not None and claimed.id == created.request_id


def test_claim_next_skips_a_request_another_session_holds_locked(
    conn: psycopg.Connection[Any], head_dsn: str, channels: None
) -> None:
    oldest = create_request(conn, CHANNEL, limit_n=5, dry_run=True).request_id
    conn.execute("UPDATE backfill_requests SET created_at = now() - interval '1 hour'")
    newer = create_request(conn, OTHER, limit_n=5, dry_run=True).request_id
    conn.commit()
    with psycopg.connect(head_dsn) as holder:
        holder.execute("SELECT 1 FROM backfill_requests WHERE id = %s FOR UPDATE", (oldest,))

        conn.execute("SET LOCAL lock_timeout = '3s'")  # without SKIP LOCKED: fail, not hang
        claimed = claim_next(conn)

        assert claimed is not None and claimed.id == newer


def test_concurrent_creates_for_one_channel_yield_exactly_one_row(
    conn: psycopg.Connection[Any], head_dsn: str, channels: None
) -> None:
    """Forced interleaving: B's insert is blocked on A's uncommitted row, then A commits."""
    conn.commit()
    a = psycopg.connect(head_dsn)
    b = psycopg.connect(head_dsn)
    out: dict[str, Any] = {}
    try:
        first = create_request(a, CHANNEL, limit_n=10, dry_run=True)  # uncommitted

        def second() -> None:
            out["b"] = create_request(b, CHANNEL, limit_n=20, dry_run=False)
            b.commit()

        thread = threading.Thread(target=second)
        thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            blocked = conn.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
                " AND pid <> pg_backend_pid() AND wait_event_type = 'Lock'"
                " AND query ILIKE '%INSERT INTO backfill_requests%'"
            ).fetchone()
            conn.rollback()
            if blocked is not None and blocked[0] == 1:
                break
        else:
            pytest.fail("the second insert never blocked on the first")
        a.commit()
        thread.join(10)
        assert not thread.is_alive()
    finally:
        a.close()
        b.close()

    assert first.created and not out["b"].created
    assert out["b"].request_id == first.request_id
    assert conn.execute("SELECT count(*) FROM backfill_requests").fetchone() == (1,)


def test_record_progress_adds_enqueued_and_ends_only_when_done(
    conn: psycopg.Connection[Any], channels: None
) -> None:
    rid = create_request(conn, CHANNEL, limit_n=10, dry_run=False).request_id

    assert record_progress(conn, rid, enqueued_this_run=4, result={"n": 1}, done=False)
    assert record_progress(conn, rid, enqueued_this_run=3, result={"n": 2}, done=False)
    mid = get_request(conn, CHANNEL, rid)
    assert mid is not None
    assert (mid.state, mid.enqueued, mid.result, mid.finished_at) == ("running", 7, {"n": 2}, None)

    assert record_progress(conn, rid, enqueued_this_run=0, result={"n": 3}, done=True)
    end = get_request(conn, CHANNEL, rid)
    assert end is not None and end.state == "done" and end.finished_at is not None
    assert end.enqueued == 7
    # A terminal request is not touched again.
    assert not record_progress(conn, rid, enqueued_this_run=9, result={"n": 4}, done=False)
    assert get_request(conn, CHANNEL, rid) == end


@pytest.mark.parametrize(
    ("fatal", "prior_attempts", "expected"),
    [
        (False, 0, "running"),
        (False, 1, "running"),
        (False, 2, "failed"),  # the third failed attempt, with max_attempts=3
        (True, 0, "failed"),
    ],
)
def test_record_failure_counts_attempts_and_fails_at_the_limit(
    conn: psycopg.Connection[Any], channels: None, fatal: bool, prior_attempts: int, expected: str
) -> None:
    rid = create_request(conn, CHANNEL, limit_n=10, dry_run=False).request_id
    conn.execute("UPDATE backfill_requests SET attempts = %s", (prior_attempts,))

    state = record_failure(conn, rid, error_class="RateLimitedError", max_attempts=3, fatal=fatal)

    row = get_request(conn, CHANNEL, rid)
    assert state == expected and row is not None
    assert (row.state, row.attempts, row.error_class) == (
        expected,
        prior_attempts + 1,
        "RateLimitedError",
    )
    assert (row.finished_at is not None) == (expected == "failed")


def test_count_pending_backfill_ingests_counts_only_pending_ingest_at_minus_10(
    conn: psycopg.Connection[Any], channels: None
) -> None:
    for i in range(6):
        conn.execute("INSERT INTO videos (video_id) VALUES (%s)", (f"cv{i:09d}",))
    rows = [
        ("cv000000000", "ingest", "pending", -10),
        ("cv000000001", "ingest", "pending", -10),
        ("cv000000002", "ingest", "pending", 0),
        ("cv000000003", "ingest", "running", -10),
        ("cv000000004", "analyze", "pending", -10),
        ("cv000000005", "ingest", "done", -10),
    ]
    for vid, kind, state, prio in rows:
        conn.execute(
            "INSERT INTO jobs (video_id, kind, state, priority) VALUES (%s, %s, %s, %s)",
            (vid, kind, state, prio),
        )

    assert count_pending_backfill_ingests(conn) == 2


def test_cancel_active_ends_the_matching_request_only(
    conn: psycopg.Connection[Any], channels: None
) -> None:
    rid = create_request(conn, CHANNEL, limit_n=10, dry_run=False).request_id

    assert cancel_active(conn, CHANNEL, request_backfill_id(rid + 1)) is None  # someone else's id
    assert cancel_active(conn, OTHER, request_backfill_id(rid)) is None  # no request there
    assert _state(conn, rid) == "pending"

    assert cancel_active(conn, CHANNEL, request_backfill_id(rid)) == rid
    row = get_request(conn, CHANNEL, rid)
    assert row is not None
    assert (row.state, row.error_class) == ("failed", CANCELLED) and row.finished_at is not None
    assert create_request(conn, CHANNEL, limit_n=1, dry_run=True).created  # the channel is free


def test_request_backfill_id_is_a_stable_canonical_lowercase_uuid() -> None:
    first = request_backfill_id(7)

    assert first == request_backfill_id(7) != request_backfill_id(8)
    assert first == first.lower() and len(first) == 36
