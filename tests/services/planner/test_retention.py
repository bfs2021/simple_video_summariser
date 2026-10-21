"""Integration tests for planner audio retention (issue #36).

Real Postgres (fixtures from #7) and real files under ``tmp_path``. ``now``
is injected, so nothing depends on the wall clock.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest
from structlog.testing import capture_logs

from services.planner.retention import (
    ORPHAN_GRACE,
    RetentionReport,
    enforce_audio_retention,
)

pytestmark = pytest.mark.integration

NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
FAR = 10**12  # a cap nobody reaches
T0 = datetime(2026, 1, 1, tzinfo=UTC)


class Audio:
    """Helper that seeds ``media`` rows and the matching files."""

    def __init__(self, conn: psycopg.Connection, audio_dir: Path) -> None:
        self.conn = conn
        self.dir = audio_dir

    def add(
        self,
        rel: str,
        *,
        size: int = 10,
        expires_at: datetime | None = None,
        created_at: datetime = T0,
        video_id: str | None = None,
        file: bool = True,
        fmt: str = "opus16k",
    ) -> tuple[int, str]:
        vid = video_id or "v" + uuid.uuid4().hex[:10]
        self.conn.execute(
            "INSERT INTO videos (video_id) VALUES (%s) ON CONFLICT DO NOTHING", (vid,)
        )
        row = self.conn.execute(
            """
            INSERT INTO media (video_id, path, bytes, format, created_at, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
            """,
            (vid, rel, size, fmt, created_at, expires_at or NOW + timedelta(days=30)),
        ).fetchone()
        assert row is not None
        if file:
            self.write(rel, size)
        self.conn.commit()
        return int(row[0]), vid

    def write(self, rel: str, size: int = 10) -> Path:
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        return path

    def job(self, video_id: str, state: str, kind: str = "transcribe") -> None:
        self.conn.execute(
            "INSERT INTO jobs (video_id, kind, state) VALUES (%s, %s, %s)",
            (video_id, kind, state),
        )
        self.conn.commit()

    def paths(self) -> list[str]:
        rows = self.conn.execute("SELECT path FROM media ORDER BY id").fetchall()
        self.conn.commit()
        return [r[0] for r in rows]

    def exists(self, rel: str) -> bool:
        return (self.dir / rel).exists()


@pytest.fixture
def audio_dir(tmp_path: Path) -> Path:
    path = tmp_path / "audio"
    path.mkdir()
    return path


@pytest.fixture
def audio(conn: psycopg.Connection, audio_dir: Path) -> Audio:
    return Audio(conn, audio_dir)


def run(
    conn: psycopg.Connection,
    audio_dir: Path,
    *,
    max_bytes: int = FAR,
    now: datetime = NOW,
) -> RetentionReport:
    return enforce_audio_retention(conn, audio_dir, max_bytes=max_bytes, now=now)


# ---------------------------------------------------------------- empty / no-op


def test_empty_media_table_reports_zeros(
    conn: psycopg.Connection, audio_dir: Path
) -> None:
    assert run(conn, audio_dir) == RetentionReport(0, 0, 0, 0, 0, 0, 0)


def test_rows_inside_window_and_cap_are_untouched(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/a.opus", size=5)

    report = run(conn, audio.dir, max_bytes=100)

    assert report == RetentionReport(0, 0, 0, 0, 5, 0, 0)
    assert audio.paths() == ["ab/a.opus"]
    assert audio.exists("ab/a.opus")


# ------------------------------------------------------------------ phase 1


def test_expired_row_has_file_unlinked_and_row_deleted(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/a.opus", size=7, expires_at=NOW - timedelta(seconds=1))

    report = run(conn, audio.dir)

    assert report.expired_deleted == 1
    assert report.bytes_freed == 7
    assert report.bytes_retained == 0
    assert audio.paths() == []
    assert not audio.exists("ab/a.opus")


def test_row_expiring_exactly_now_is_expired(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/a.opus", expires_at=NOW)

    assert run(conn, audio.dir).expired_deleted == 1
    assert audio.paths() == []


def test_row_expiring_one_microsecond_after_now_is_kept(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/a.opus", expires_at=NOW + timedelta(microseconds=1))

    assert run(conn, audio.dir).expired_deleted == 0
    assert audio.paths() == ["ab/a.opus"]
    assert audio.exists("ab/a.opus")


def test_expired_row_with_missing_file_is_deleted_without_error(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/a.opus", expires_at=NOW - timedelta(days=1), file=False)

    report = run(conn, audio.dir)

    assert report.expired_deleted == 1
    assert report.errors == 0
    assert audio.paths() == []


# ------------------------------------------------------------------ phase 2


def test_total_exactly_at_cap_evicts_nothing(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/a.opus", size=10)
    audio.add("ab/b.opus", size=10)

    report = run(conn, audio.dir, max_bytes=20)

    assert report.evicted == 0
    assert report.bytes_retained == 20


def test_total_one_byte_over_cap_evicts_exactly_the_oldest(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/new.opus", size=10, created_at=T0 + timedelta(days=2))
    audio.add("ab/old.opus", size=10, created_at=T0)

    report = run(conn, audio.dir, max_bytes=19)

    assert report.evicted == 1
    assert report.bytes_freed == 10
    assert report.bytes_retained == 10
    assert audio.paths() == ["ab/new.opus"]
    assert not audio.exists("ab/old.opus")
    assert audio.exists("ab/new.opus")


def test_eviction_stops_as_soon_as_total_is_within_cap(
    audio: Audio, conn: psycopg.Connection
) -> None:
    for i in range(4):
        audio.add(f"ab/{i}.opus", size=10, created_at=T0 + timedelta(days=i))

    report = run(conn, audio.dir, max_bytes=25)

    assert report.evicted == 2
    assert audio.paths() == ["ab/2.opus", "ab/3.opus"]


def test_created_at_ties_are_broken_by_id_ascending(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/first.opus", size=10)
    audio.add("ab/second.opus", size=10)

    run(conn, audio.dir, max_bytes=10)

    assert audio.paths() == ["ab/second.opus"]


def test_eviction_ignores_expires_at(audio: Audio, conn: psycopg.Connection) -> None:
    far_future = NOW + timedelta(days=3650)
    audio.add("ab/a.opus", size=10, expires_at=far_future, created_at=T0)
    audio.add(
        "ab/b.opus", size=10, expires_at=far_future, created_at=T0 + timedelta(days=1)
    )

    report = run(conn, audio.dir, max_bytes=10)

    assert report.evicted == 1
    assert report.expired_deleted == 0
    assert audio.paths() == ["ab/b.opus"]


def test_cap_applies_to_rows_remaining_after_expiry(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/expired.opus", size=50, expires_at=NOW - timedelta(days=1))
    audio.add("ab/keep.opus", size=10)

    report = run(conn, audio.dir, max_bytes=10)

    assert report.expired_deleted == 1
    assert report.evicted == 0
    assert audio.paths() == ["ab/keep.opus"]


def test_total_comes_from_the_database_not_from_stat(
    audio: Audio, conn: psycopg.Connection
) -> None:
    # Recorded bytes say 100, the file on disk is tiny: the database wins.
    audio.add("ab/a.opus", size=100)
    (audio.dir / "ab/a.opus").write_bytes(b"x")

    report = run(conn, audio.dir, max_bytes=50)

    assert report.evicted == 1
    assert report.bytes_freed == 100


def test_eviction_of_a_row_with_missing_file_is_not_an_error(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/a.opus", size=10, file=False)

    report = run(conn, audio.dir, max_bytes=0)

    assert report.evicted == 1
    assert report.errors == 0
    assert audio.paths() == []


def test_cap_unreachable_warns_and_returns(
    audio: Audio, conn: psycopg.Connection
) -> None:
    _, vid = audio.add("ab/a.opus", size=30)
    audio.job(vid, "running")

    with capture_logs() as logs:
        report = run(conn, audio.dir, max_bytes=10)

    assert report.evicted == 0
    assert report.skipped_running == 1
    assert report.bytes_retained == 30
    warnings = [
        e for e in logs if e["event"] == "planner.audio_retention.cap_unreachable"
    ]
    assert len(warnings) == 1
    assert warnings[0]["log_level"] == "warning"
    assert warnings[0]["total_bytes"] == 30


# --------------------------------------------------------------- protection


def test_running_job_protects_an_expired_row(
    audio: Audio, conn: psycopg.Connection
) -> None:
    _, vid = audio.add("ab/a.opus", expires_at=NOW - timedelta(days=1))
    audio.job(vid, "running")

    report = run(conn, audio.dir)

    assert report.expired_deleted == 0
    assert report.skipped_running == 1
    assert audio.paths() == ["ab/a.opus"]
    assert audio.exists("ab/a.opus")


def test_eviction_skips_a_running_row_and_moves_to_the_next_oldest(
    audio: Audio, conn: psycopg.Connection
) -> None:
    _, running = audio.add("ab/oldest.opus", size=10, created_at=T0)
    audio.add("ab/middle.opus", size=10, created_at=T0 + timedelta(days=1))
    audio.add("ab/newest.opus", size=10, created_at=T0 + timedelta(days=2))
    audio.job(running, "running")

    report = run(conn, audio.dir, max_bytes=20)

    assert report.skipped_running == 1
    assert report.evicted == 1
    assert audio.paths() == ["ab/oldest.opus", "ab/newest.opus"]


@pytest.mark.parametrize("state", ["pending", "done", "dead"])
def test_non_running_jobs_do_not_protect(
    audio: Audio, conn: psycopg.Connection, state: str
) -> None:
    _, vid = audio.add("ab/a.opus", expires_at=NOW - timedelta(days=1))
    audio.job(vid, state)

    report = run(conn, audio.dir)

    assert report.expired_deleted == 1
    assert report.skipped_running == 0


def test_a_row_locked_by_another_connection_is_skipped_not_waited_on(
    audio: Audio, conn: psycopg.Connection, head_dsn: str
) -> None:
    locked_id, _ = audio.add(
        "ab/locked.opus", size=10, expires_at=NOW - timedelta(days=1)
    )
    audio.add("ab/free.opus", size=10, expires_at=NOW - timedelta(days=1))

    with psycopg.connect(head_dsn) as other:
        other.execute("SELECT id FROM media WHERE id = %s FOR UPDATE", (locked_id,))
        report = run(conn, audio.dir)
        other.rollback()

    assert report.expired_deleted == 1
    assert audio.paths() == ["ab/locked.opus"]
    assert audio.exists("ab/locked.opus")
    assert not audio.exists("ab/free.opus")


def test_row_no_longer_over_cap_when_locked_is_left_alone(
    audio: Audio, conn: psycopg.Connection
) -> None:
    # Two rows over the cap; deleting the oldest brings the total under it,
    # so the re-check on the second row must not delete it.
    audio.add("ab/a.opus", size=10, created_at=T0)
    audio.add("ab/b.opus", size=10, created_at=T0 + timedelta(days=1))

    report = run(conn, audio.dir, max_bytes=10)

    assert report.evicted == 1
    assert audio.exists("ab/b.opus")


# -------------------------------------------------------------- path safety


def test_path_escaping_audio_dir_is_never_unlinked(
    audio: Audio, conn: psycopg.Connection, tmp_path: Path
) -> None:
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    victim = sibling / "victim.opus"
    victim.write_bytes(b"precious")
    audio.add(
        "../sibling/victim.opus", size=8, expires_at=NOW - timedelta(days=1), file=False
    )

    with capture_logs() as logs:
        report = run(conn, audio.dir)

    assert victim.exists()
    assert audio.paths() == []
    assert report.errors == 1
    assert report.bytes_freed == 0
    assert any(e["log_level"] == "error" for e in logs)


def test_absolute_path_is_never_unlinked(
    audio: Audio, conn: psycopg.Connection, tmp_path: Path
) -> None:
    victim = tmp_path / "abs.opus"
    victim.write_bytes(b"precious")
    audio.add(str(victim), size=8, expires_at=NOW - timedelta(days=1), file=False)

    report = run(conn, audio.dir)

    assert victim.exists()
    assert report.errors == 1
    assert audio.paths() == []


def test_permission_error_on_unlink_keeps_row_and_continues(
    audio: Audio, conn: psycopg.Connection
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    audio.add(
        "aa/stuck.opus", size=10, created_at=T0, expires_at=NOW - timedelta(days=1)
    )
    audio.add(
        "bb/ok.opus",
        size=20,
        created_at=T0 + timedelta(days=1),
        expires_at=NOW - timedelta(days=1),
    )
    locked_dir = audio.dir / "aa"
    locked_dir.chmod(0o500)
    try:
        report = run(conn, audio.dir)
    finally:
        locked_dir.chmod(0o700)

    assert report.errors == 1
    assert report.expired_deleted == 1
    assert report.bytes_freed == 20
    assert audio.paths() == ["aa/stuck.opus"]
    assert audio.exists("aa/stuck.opus")
    assert not audio.exists("bb/ok.opus")


def test_failure_on_one_row_does_not_undo_earlier_rows(
    audio: Audio, conn: psycopg.Connection
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    audio.add("aa/done.opus", created_at=T0, expires_at=NOW - timedelta(days=1))
    audio.add(
        "bb/stuck.opus",
        created_at=T0 + timedelta(days=1),
        expires_at=NOW - timedelta(days=1),
    )
    (audio.dir / "bb").chmod(0o500)
    try:
        run(conn, audio.dir)
    finally:
        (audio.dir / "bb").chmod(0o700)

    assert audio.paths() == ["bb/stuck.opus"]


# ------------------------------------------------------------------ phase 3


def age(path: Path, at: datetime) -> None:
    ts = at.timestamp()
    os.utime(path, (ts, ts))


def test_old_unreferenced_file_is_deleted(
    audio: Audio, conn: psycopg.Connection
) -> None:
    orphan = audio.write("ab/orphan.opus", 12)
    age(orphan, NOW - ORPHAN_GRACE - timedelta(hours=1))

    report = run(conn, audio.dir)

    assert report.orphans_deleted == 1
    assert report.bytes_freed == 12
    assert not orphan.exists()


def test_orphan_exactly_at_grace_edge_is_deleted(
    audio: Audio, conn: psycopg.Connection
) -> None:
    orphan = audio.write("ab/edge.opus")
    age(orphan, NOW - ORPHAN_GRACE)

    assert run(conn, audio.dir).orphans_deleted == 1
    assert not orphan.exists()


def test_orphan_one_second_inside_grace_is_kept(
    audio: Audio, conn: psycopg.Connection
) -> None:
    orphan = audio.write("ab/fresh.opus")
    age(orphan, NOW - ORPHAN_GRACE + timedelta(seconds=1))

    assert run(conn, audio.dir).orphans_deleted == 0
    assert orphan.exists()


def test_download_and_tmp_leftovers_are_orphans(
    audio: Audio, conn: psycopg.Connection
) -> None:
    files = [audio.write("ab/.download-x.part"), audio.write("ab/.tmp-y.opus")]
    for f in files:
        age(f, NOW - timedelta(days=3))

    assert run(conn, audio.dir).orphans_deleted == 2


def test_referenced_file_is_never_deleted_however_old(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/kept.opus")
    age(audio.dir / "ab/kept.opus", NOW - timedelta(days=400))

    report = run(conn, audio.dir)

    assert report.orphans_deleted == 0
    assert audio.exists("ab/kept.opus")


def test_orphan_search_is_recursive(audio: Audio, conn: psycopg.Connection) -> None:
    deep = audio.write("ab/cd/ef/deep.opus")
    age(deep, NOW - timedelta(days=3))

    assert run(conn, audio.dir).orphans_deleted == 1
    assert not deep.exists()
    assert (audio.dir / "ab/cd/ef").is_dir()


def test_symlinks_and_directories_are_never_deleted(
    audio: Audio, conn: psycopg.Connection, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    target_file = outside / "t.opus"
    target_file.write_bytes(b"x")
    age(target_file, NOW - timedelta(days=9))
    (audio.dir / "ab").mkdir()
    link_file = audio.dir / "ab" / "link.opus"
    link_file.symlink_to(target_file)
    link_dir = audio.dir / "dirlink"
    link_dir.symlink_to(outside, target_is_directory=True)
    (audio.dir / "empty").mkdir()
    os.utime(link_file, (0, 0), follow_symlinks=False)

    report = run(conn, audio.dir)

    assert report.orphans_deleted == 0
    assert link_file.is_symlink()
    assert link_dir.is_symlink()
    assert target_file.exists()
    assert (audio.dir / "empty").is_dir()
    assert (audio.dir / "ab").is_dir()


def test_orphan_bytes_are_freed_but_do_not_count_toward_the_cap(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/kept.opus", size=10)
    orphan = audio.write("ab/orphan.opus", 1000)
    age(orphan, NOW - timedelta(days=3))

    report = run(conn, audio.dir, max_bytes=10)

    assert report.evicted == 0
    assert report.orphans_deleted == 1
    assert report.bytes_freed == 1000
    assert report.bytes_retained == 10
    assert audio.exists("ab/kept.opus")


def test_file_of_deleted_video_becomes_an_orphan(
    audio: Audio, conn: psycopg.Connection
) -> None:
    _, vid = audio.add("ab/gone.opus")
    age(audio.dir / "ab/gone.opus", NOW - timedelta(days=3))
    conn.execute("DELETE FROM videos WHERE video_id = %s", (vid,))
    conn.commit()

    report = run(conn, audio.dir)

    assert report.orphans_deleted == 1
    assert not audio.exists("ab/gone.opus")


def test_a_file_evicted_this_run_is_not_counted_again_as_an_orphan(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/a.opus", size=10, expires_at=NOW - timedelta(days=1))

    report = run(conn, audio.dir)

    assert report.expired_deleted == 1
    assert report.orphans_deleted == 0
    assert report.bytes_freed == 10


# --------------------------------------------------------------- robustness


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_unavailable_audio_dir_raises_before_touching_the_database(
    audio: Audio, conn: psycopg.Connection, tmp_path: Path, kind: str
) -> None:
    audio.add("ab/a.opus", expires_at=NOW - timedelta(days=1))
    bad = tmp_path / "not-mounted"
    if kind == "file":
        bad.write_text("x")

    with capture_logs() as logs, pytest.raises(OSError):
        run(conn, bad)

    assert audio.paths() == ["ab/a.opus"]
    assert any(e["log_level"] == "error" for e in logs)


def test_second_run_is_a_no_op(audio: Audio, conn: psycopg.Connection) -> None:
    audio.add("ab/exp.opus", size=10, expires_at=NOW - timedelta(days=1))
    audio.add("ab/old.opus", size=10, created_at=T0)
    audio.add("ab/new.opus", size=10, created_at=T0 + timedelta(days=1))
    orphan = audio.write("ab/orphan.opus", 5)
    age(orphan, NOW - timedelta(days=3))

    first = run(conn, audio.dir, max_bytes=10)
    second = run(conn, audio.dir, max_bytes=10)

    assert first.expired_deleted == 1
    assert first.evicted == 1
    assert first.orphans_deleted == 1
    assert second == RetentionReport(0, 0, 0, 0, first.bytes_retained, 0, 0)


def test_bytes_retained_equals_sum_of_remaining_rows(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/a.opus", size=3, created_at=T0)
    audio.add("ab/b.opus", size=4, created_at=T0 + timedelta(days=1))

    report = run(conn, audio.dir)

    assert report.bytes_retained == 7


def test_run_logs_one_structured_line_with_the_report_fields(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/a.opus", size=10, expires_at=NOW - timedelta(days=1))

    with capture_logs() as logs:
        report = run(conn, audio.dir)

    lines = [e for e in logs if e["event"] == "planner.audio_retention"]
    assert len(lines) == 1
    line = lines[0]
    for field in (
        "expired_deleted",
        "evicted",
        "orphans_deleted",
        "bytes_freed",
        "bytes_retained",
        "skipped_running",
        "errors",
    ):
        assert line[field] == getattr(report, field)


def test_naive_now_is_rejected(conn: psycopg.Connection, audio_dir: Path) -> None:
    with pytest.raises(ValueError):
        run(conn, audio_dir, now=datetime(2026, 6, 1))  # naive on purpose


@pytest.fixture(autouse=True)
def _no_leftover_perms(audio_dir: Path) -> Iterator[None]:
    yield
    for p in audio_dir.rglob("*"):
        if p.is_dir() and not p.is_symlink():
            p.chmod(0o700)
