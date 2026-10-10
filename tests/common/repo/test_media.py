"""Tests for the retention queries in common/repo/media.py (issue #36)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from common.repo.media import (
    LockOutcome,
    delete_media,
    delete_row,
    list_expired,
    list_oldest_first,
    list_referenced_paths,
    lock_for_deletion,
    register_media,
    total_bytes,
)

pytestmark = pytest.mark.integration

NOW = datetime(2026, 6, 1, tzinfo=UTC)
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def add(
    conn: psycopg.Connection,
    video_id: str,
    path: str,
    *,
    size: int = 10,
    created_at: datetime = T0,
    expires_at: datetime = NOW + timedelta(days=30),
) -> int:
    row = conn.execute(
        "INSERT INTO media (video_id, path, bytes, created_at, expires_at) "
        "VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (video_id, path, size, created_at, expires_at),
    ).fetchone()
    assert row is not None
    return int(row[0])


def new_video(conn: psycopg.Connection, vid: str) -> str:
    conn.execute("INSERT INTO videos (video_id) VALUES (%s)", (vid,))
    return vid


def test_total_bytes_is_zero_for_an_empty_table(conn: psycopg.Connection) -> None:
    assert total_bytes(conn) == 0


def test_total_bytes_sums_all_rows(conn: psycopg.Connection) -> None:
    add(conn, new_video(conn, "v1"), "a", size=3)
    add(conn, new_video(conn, "v2"), "b", size=4)

    assert total_bytes(conn) == 7


def test_list_expired_includes_the_boundary_and_excludes_later_rows(
    conn: psycopg.Connection,
) -> None:
    exact = add(conn, new_video(conn, "v1"), "a", expires_at=NOW)
    add(conn, new_video(conn, "v2"), "b", expires_at=NOW + timedelta(microseconds=1))
    before = add(conn, new_video(conn, "v3"), "c", expires_at=NOW - timedelta(days=1))

    rows = list_expired(conn, NOW)

    assert sorted(r.id for r in rows) == sorted([exact, before])


def test_list_oldest_first_orders_by_created_at_then_id(
    conn: psycopg.Connection,
) -> None:
    newer = add(conn, new_video(conn, "v1"), "a", created_at=T0 + timedelta(days=1))
    tie_a = add(conn, new_video(conn, "v2"), "b", created_at=T0)
    tie_b = add(conn, new_video(conn, "v3"), "c", created_at=T0)

    assert [r.id for r in list_oldest_first(conn)] == [tie_a, tie_b, newer]


def test_list_referenced_paths_returns_every_path(conn: psycopg.Connection) -> None:
    add(conn, new_video(conn, "v1"), "ab/a.opus")
    add(conn, new_video(conn, "v2"), "cd/b.opus")

    assert list_referenced_paths(conn) == {"ab/a.opus", "cd/b.opus"}


def test_lock_for_deletion_expired_row_is_locked_and_returned(
    conn: psycopg.Connection,
) -> None:
    mid = add(conn, new_video(conn, "v1"), "a", size=5, expires_at=NOW)

    outcome, row = lock_for_deletion(conn, mid, expired_at=NOW)

    assert outcome is LockOutcome.LOCKED
    assert row is not None
    assert (row.id, row.video_id, row.path, row.bytes) == (mid, "v1", "a", 5)


def test_lock_for_deletion_rechecks_expiry(conn: psycopg.Connection) -> None:
    mid = add(conn, new_video(conn, "v1"), "a", expires_at=NOW + timedelta(seconds=1))

    outcome, row = lock_for_deletion(conn, mid, expired_at=NOW)

    assert outcome is LockOutcome.NOT_ELIGIBLE
    assert row is None


def test_lock_for_deletion_rechecks_the_cap(conn: psycopg.Connection) -> None:
    mid = add(conn, new_video(conn, "v1"), "a", size=10)

    assert lock_for_deletion(conn, mid, max_bytes=10)[0] is LockOutcome.NOT_ELIGIBLE
    assert lock_for_deletion(conn, mid, max_bytes=9)[0] is LockOutcome.LOCKED


def test_lock_for_deletion_reports_running_jobs(conn: psycopg.Connection) -> None:
    vid = new_video(conn, "v1")
    mid = add(conn, vid, "a", expires_at=NOW)
    conn.execute(
        "INSERT INTO jobs (video_id, kind, state) VALUES (%s, 'transcribe', 'running')",
        (vid,),
    )

    outcome, row = lock_for_deletion(conn, mid, expired_at=NOW)

    assert outcome is LockOutcome.RUNNING
    assert row is None


def test_lock_for_deletion_missing_row_is_unavailable(conn: psycopg.Connection) -> None:
    assert lock_for_deletion(conn, 999, expired_at=NOW) == (
        LockOutcome.UNAVAILABLE,
        None,
    )


def test_lock_for_deletion_skips_a_row_locked_elsewhere(
    conn: psycopg.Connection, head_dsn: str
) -> None:
    mid = add(conn, new_video(conn, "v1"), "a", expires_at=NOW)
    conn.commit()

    with psycopg.connect(head_dsn) as other:
        other.execute("SELECT id FROM media WHERE id = %s FOR UPDATE", (mid,))
        outcome, row = lock_for_deletion(conn, mid, expired_at=NOW)
        other.rollback()

    assert (outcome, row) == (LockOutcome.UNAVAILABLE, None)


def test_lock_for_deletion_requires_exactly_one_criterion(
    conn: psycopg.Connection,
) -> None:
    with pytest.raises(ValueError):
        lock_for_deletion(conn, 1)
    with pytest.raises(ValueError):
        lock_for_deletion(conn, 1, expired_at=NOW, max_bytes=1)
    with pytest.raises(ValueError):
        lock_for_deletion(conn, 1, low_space=True, expired_at=NOW)
    with pytest.raises(ValueError):
        lock_for_deletion(conn, 1, low_space=True, max_bytes=1)
    with pytest.raises(ValueError):
        lock_for_deletion(conn, 1, low_space=True, expired_at=NOW, max_bytes=1)
    with pytest.raises(ValueError):
        lock_for_deletion(conn, 1, low_space=False)


def test_lock_for_deletion_low_space_locks_a_row_that_is_neither_expired_nor_over_cap(
    conn: psycopg.Connection,
) -> None:
    mid = add(conn, new_video(conn, "v1"), "a", size=1, expires_at=NOW + timedelta(days=365))

    outcome, row = lock_for_deletion(conn, mid, low_space=True)

    assert outcome is LockOutcome.LOCKED
    assert row is not None and row.id == mid


def test_lock_for_deletion_low_space_reports_running_jobs(conn: psycopg.Connection) -> None:
    vid = new_video(conn, "v1")
    mid = add(conn, vid, "a")
    conn.execute(
        "INSERT INTO jobs (video_id, kind, state) VALUES (%s, 'transcribe', 'running')", (vid,)
    )

    assert lock_for_deletion(conn, mid, low_space=True) == (LockOutcome.RUNNING, None)


def test_lock_for_deletion_low_space_missing_row_is_unavailable(conn: psycopg.Connection) -> None:
    assert lock_for_deletion(conn, 999, low_space=True) == (LockOutcome.UNAVAILABLE, None)


def test_lock_for_deletion_low_space_skips_a_row_locked_elsewhere(
    conn: psycopg.Connection, head_dsn: str
) -> None:
    mid = add(conn, new_video(conn, "v1"), "a")
    conn.commit()

    with psycopg.connect(head_dsn) as other:
        other.execute("SELECT id FROM media WHERE id = %s FOR UPDATE", (mid,))
        outcome, row = lock_for_deletion(conn, mid, low_space=True)
        other.rollback()

    assert (outcome, row) == (LockOutcome.UNAVAILABLE, None)


def test_delete_row_removes_only_that_row_and_does_not_commit(
    conn: psycopg.Connection, head_dsn: str
) -> None:
    keep = add(conn, new_video(conn, "v1"), "a")
    gone = add(conn, new_video(conn, "v2"), "b")
    conn.commit()

    delete_row(conn, gone)

    with psycopg.connect(head_dsn) as other:
        seen = {r[0] for r in other.execute("SELECT id FROM media").fetchall()}
    assert seen == {keep, gone}  # uncommitted
    conn.rollback()
    assert list_referenced_paths(conn) == {"a", "b"}


# --- register_media / delete_media (issue #29) --------------------------------------


def media_rows(conn: psycopg.Connection, vid: str) -> list[tuple[Any, ...]]:
    return conn.execute(
        "SELECT path, bytes, format, created_at, expires_at FROM media WHERE video_id = %s",
        (vid,),
    ).fetchall()


def test_register_media_expiry_is_exactly_ttl_days_after_created_at(
    conn: psycopg.Connection,
) -> None:
    vid = new_video(conn, "v1")

    register_media(conn, vid, "ab/v1.opus", 123, 30)

    ((path, size, fmt, created_at, expires_at),) = media_rows(conn, vid)
    assert (path, size, fmt) == ("ab/v1.opus", 123, "opus16k")
    assert expires_at - created_at == timedelta(days=30)


def test_register_media_twice_updates_the_row_instead_of_raising(
    conn: psycopg.Connection,
) -> None:
    vid = new_video(conn, "v1")
    conn.execute(
        "INSERT INTO media (video_id, path, bytes, created_at, expires_at) "
        "VALUES (%s, 'old', 1, %s, %s)",
        (vid, T0, T0 + timedelta(days=1)),
    )

    register_media(conn, vid, "new/v1.opus", 999, 7)
    register_media(conn, vid, "newer/v1.opus", 1000, 7)

    ((path, size, _fmt, created_at, expires_at),) = media_rows(conn, vid)
    assert (path, size) == ("newer/v1.opus", 1000)
    assert created_at > T0
    assert expires_at - created_at == timedelta(days=7)


def test_register_media_for_an_unknown_video_violates_the_foreign_key(
    conn: psycopg.Connection,
) -> None:
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        register_media(conn, "nosuchvideo", "a", 1, 30)


def test_register_media_does_not_commit(conn: psycopg.Connection, head_dsn: str) -> None:
    vid = new_video(conn, "v1")
    conn.commit()

    register_media(conn, vid, "a", 1, 30)

    with psycopg.connect(head_dsn) as other:
        assert media_rows(other, vid) == []
    conn.rollback()


def test_delete_media_removes_only_that_videos_row_and_tolerates_none(
    conn: psycopg.Connection,
) -> None:
    gone = new_video(conn, "v1")
    keep = new_video(conn, "v2")
    register_media(conn, gone, "a", 1, 30)
    register_media(conn, keep, "b", 1, 30)

    delete_media(conn, gone)
    delete_media(conn, gone)

    assert list_referenced_paths(conn) == {"b"}
