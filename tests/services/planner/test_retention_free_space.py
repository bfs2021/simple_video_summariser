"""Phase 4 of audio retention: the free-space guard, against a real Postgres (issue #105).

``tmp_path`` is ``audio_dir`` with real files; the disk is a fake whose
``free`` follows the real bytes on disk, so no test needs a nearly full disk.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import psycopg
import pytest
from structlog.testing import capture_logs

from services.planner.retention import RetentionReport, enforce_audio_retention
from tests.services.planner.test_retention import FAR, NOW, T0, Audio

pytestmark = pytest.mark.integration

GB = 10**9
DAY = timedelta(days=1)


class Usage(NamedTuple):
    total: int
    used: int
    free: int


def _bytes_on_disk(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


class FakeDisk:
    """``free`` is ``free_now`` at construction and grows by every byte unlinked since."""

    def __init__(self, root: Path, *, total: int, free: int) -> None:
        self.root = root
        self.total = total
        self.base = free + _bytes_on_disk(root)
        self.calls: list[Path] = []

    def __call__(self, path: Path) -> Usage:
        self.calls.append(path)
        free = self.base - _bytes_on_disk(self.root)
        return Usage(self.total, self.total - free, free)


class StuckDisk:
    """Reports a constant ``free``: unlinked space is never returned."""

    def __init__(self, *, total: int, free: int) -> None:
        self.usage = Usage(total, total - free, free)
        self.calls = 0

    def __call__(self, path: Path) -> Usage:
        self.calls += 1
        return self.usage


@pytest.fixture
def audio_dir(tmp_path: Path) -> Path:
    path = tmp_path / "audio"
    path.mkdir()
    return path


@pytest.fixture
def audio(conn: psycopg.Connection, audio_dir: Path) -> Audio:
    return Audio(conn, audio_dir)


def guard(
    conn: psycopg.Connection,
    audio_dir: Path,
    disk: object,
    *,
    floor: int,
    max_bytes: int = FAR,
    now: datetime = NOW,
) -> RetentionReport:
    """Run with a floor of exactly ``floor`` bytes (``total`` is large, ``pct`` is 0)."""
    return enforce_audio_retention(
        conn,
        audio_dir,
        max_bytes=max_bytes,
        now=now,
        min_free_gb=floor / GB,
        min_free_pct=0,
        disk_usage=disk,  # type: ignore[arg-type]
    )


def seed(audio: Audio, n: int, *, size: int = 10) -> list[str]:
    """``n`` rows, oldest first, ``created_at`` one day apart."""
    paths = [f"ab/f{i}.opus" for i in range(n)]
    for i, rel in enumerate(paths):
        audio.add(rel, size=size, created_at=T0 + i * DAY)
    return paths


# ----------------------------------------------------------------- guard off


def test_floor_zero_reads_nothing_and_evicts_nothing(
    audio: Audio, conn: psycopg.Connection
) -> None:
    seed(audio, 3)
    disk = FakeDisk(audio.dir, total=100 * GB, free=0)

    report = enforce_audio_retention(
        conn, audio.dir, max_bytes=FAR, now=NOW, min_free_gb=0, min_free_pct=0, disk_usage=disk
    )

    assert disk.calls == []
    assert (report.evicted_for_space, report.free_bytes) == (0, None)
    assert len(audio.paths()) == 3


def test_default_arguments_keep_the_guard_off(audio: Audio, conn: psycopg.Connection) -> None:
    seed(audio, 2)

    report = enforce_audio_retention(conn, audio.dir, max_bytes=FAR, now=NOW)

    assert (report.evicted_for_space, report.free_bytes) == (0, None)
    assert len(audio.paths()) == 2


# ---------------------------------------------------------------- the boundary


def test_free_exactly_at_the_floor_evicts_nothing(audio: Audio, conn: psycopg.Connection) -> None:
    seed(audio, 3)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB)

    report = guard(conn, audio.dir, disk, floor=2 * GB)

    assert report.evicted_for_space == 0
    assert report.free_bytes == 2 * GB
    assert len(audio.paths()) == 3


def test_free_one_byte_below_the_floor_evicts_exactly_the_oldest_row(
    audio: Audio, conn: psycopg.Connection
) -> None:
    paths = seed(audio, 3)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 1)

    report = guard(conn, audio.dir, disk, floor=2 * GB)

    assert report.evicted_for_space == 1
    assert report.evicted == 0
    assert audio.paths() == paths[1:]
    assert not audio.exists(paths[0])
    assert report.free_bytes == 2 * GB - 1 + 10


def test_eviction_is_oldest_first_and_stops_as_soon_as_the_floor_is_met(
    audio: Audio, conn: psycopg.Connection
) -> None:
    paths = seed(audio, 6)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 25)  # needs 25 bytes: 3 rows

    report = guard(conn, audio.dir, disk, floor=2 * GB)

    assert report.evicted_for_space == 3
    assert audio.paths() == paths[3:]
    assert report.free_bytes == 2 * GB + 5
    assert report.bytes_freed == 30


def test_free_space_is_measured_again_after_every_unlinked_file(
    audio: Audio, conn: psycopg.Connection
) -> None:
    seed(audio, 6)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 25)

    guard(conn, audio.dir, disk, floor=2 * GB)

    # once to start, then after each of the 3 evicted files
    assert len(disk.calls) == 4
    assert set(disk.calls) == {audio.dir}


def test_free_space_landing_exactly_on_the_floor_stops_the_guard_without_a_warning(
    audio: Audio, conn: psycopg.Connection
) -> None:
    paths = seed(audio, 8)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 30)  # 3 rows reach it exactly

    with capture_logs() as logs:
        report = guard(conn, audio.dir, disk, floor=2 * GB)

    assert report.evicted_for_space == 3
    assert report.free_bytes == 2 * GB
    assert audio.paths() == paths[3:]
    assert not [e for e in logs if "free_space" in e["event"]]


def test_the_floor_is_judged_on_real_free_space_not_on_recorded_bytes(
    audio: Audio, conn: psycopg.Connection
) -> None:
    # Files are twice as big as their recorded size, so the floor is met after
    # two rows while the recorded bytes evicted (10) are still below the deficit (20).
    paths = [f"ab/f{i}.opus" for i in range(5)]
    for i, rel in enumerate(paths):
        audio.add(rel, size=5, created_at=T0 + i * DAY)
        audio.write(rel, 10)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 20)

    report = guard(conn, audio.dir, disk, floor=2 * GB)

    assert report.evicted_for_space == 2
    assert audio.paths() == paths[2:]
    assert report.free_bytes == 2 * GB


def test_created_at_ties_are_broken_by_id(audio: Audio, conn: psycopg.Connection) -> None:
    audio.add("ab/first.opus", created_at=T0)
    audio.add("ab/second.opus", created_at=T0)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 1)

    guard(conn, audio.dir, disk, floor=2 * GB)

    assert audio.paths() == ["ab/second.opus"]


def test_rows_inside_their_retention_window_are_evicted_too(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/a.opus", expires_at=NOW + 300 * DAY)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 1)

    report = guard(conn, audio.dir, disk, floor=2 * GB)

    assert report.evicted_for_space == 1


# ------------------------------------------------------ guard vs the byte cap


def test_under_the_cap_but_low_on_disk_the_guard_still_evicts(
    audio: Audio, conn: psycopg.Connection
) -> None:
    seed(audio, 3)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 1)

    report = guard(conn, audio.dir, disk, floor=2 * GB, max_bytes=10_000)

    assert (report.evicted, report.evicted_for_space) == (0, 1)


def test_plenty_of_disk_but_over_the_cap_phase_two_evicts_as_before(
    audio: Audio, conn: psycopg.Connection
) -> None:
    seed(audio, 3)
    disk = FakeDisk(audio.dir, total=100 * GB, free=50 * GB)

    report = guard(conn, audio.dir, disk, floor=2 * GB, max_bytes=10)

    assert (report.evicted, report.evicted_for_space) == (2, 0)
    assert report.bytes_retained == 10


def test_phase_one_to_three_deletions_count_before_the_guard_evicts_anything(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/expired.opus", size=100, created_at=T0, expires_at=NOW - DAY)
    audio.add("ab/kept.opus", size=10, created_at=T0 + DAY)
    audio.write("ab/orphan.opus", 50)
    old = (NOW - timedelta(days=3)).timestamp()
    os.utime(audio.dir / "ab/orphan.opus", (old, old))
    # 150 bytes short before the pass; expiry (100) and the orphan (50) cover it.
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 150)

    report = guard(conn, audio.dir, disk, floor=2 * GB)

    assert (report.expired_deleted, report.orphans_deleted) == (1, 1)
    assert report.evicted_for_space == 0
    assert audio.paths() == ["ab/kept.opus"]
    assert report.free_bytes == 2 * GB


def test_both_floors_hold_after_the_pass(audio: Audio, conn: psycopg.Connection) -> None:
    seed(audio, 8)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 25)

    report = guard(conn, audio.dir, disk, floor=2 * GB, max_bytes=40)

    assert report.bytes_retained <= 40
    assert report.free_bytes is not None and report.free_bytes >= 2 * GB


# ---------------------------------------------------------------- protection


def test_a_running_row_is_never_evicted_and_the_next_oldest_goes(
    audio: Audio, conn: psycopg.Connection
) -> None:
    _, vid = audio.add("ab/a.opus", created_at=T0)
    audio.add("ab/b.opus", created_at=T0 + DAY)
    audio.job(vid, "running")
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 1)

    report = guard(conn, audio.dir, disk, floor=2 * GB)

    assert audio.paths() == ["ab/a.opus"]
    assert audio.exists("ab/a.opus")
    assert (report.evicted_for_space, report.skipped_running) == (1, 1)


def test_a_row_skipped_by_the_cap_and_the_guard_counts_once(
    audio: Audio, conn: psycopg.Connection
) -> None:
    _, vid = audio.add("ab/a.opus", size=30, created_at=T0)
    audio.job(vid, "running")
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 1)

    report = guard(conn, audio.dir, disk, floor=2 * GB, max_bytes=10)

    assert report.skipped_running == 1


def test_a_row_locked_by_another_connection_is_skipped_not_waited_on(
    audio: Audio, conn: psycopg.Connection, head_dsn: str
) -> None:
    locked_id, _ = audio.add("ab/locked.opus", created_at=T0)
    audio.add("ab/free.opus", created_at=T0 + DAY)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 1)

    with psycopg.connect(head_dsn) as other:
        other.execute("SELECT id FROM media WHERE id = %s FOR UPDATE", (locked_id,))
        report = guard(conn, audio.dir, disk, floor=2 * GB)
        other.rollback()

    assert report.evicted_for_space == 1
    assert audio.paths() == ["ab/locked.opus"]


# ----------------------------------------------------------------- unreachable


def test_all_rows_protected_warns_unreachable_and_returns(
    audio: Audio, conn: psycopg.Connection
) -> None:
    for i in range(2):
        _, vid = audio.add(f"ab/r{i}.opus", created_at=T0 + i * DAY)
        audio.job(vid, "running")
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 5)

    with capture_logs() as logs:
        report = guard(conn, audio.dir, disk, floor=2 * GB)

    assert report.evicted_for_space == 0
    (line,) = [e for e in logs if e["event"] == "planner.audio_retention.free_space_unreachable"]
    assert line["log_level"] == "warning"
    assert (line["free_bytes"], line["floor_bytes"]) == (2 * GB - 5, 2 * GB)


def test_no_rows_left_warns_unreachable(audio: Audio, conn: psycopg.Connection) -> None:
    seed(audio, 2)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 1000)

    with capture_logs() as logs:
        report = guard(conn, audio.dir, disk, floor=2 * GB)

    assert report.evicted_for_space == 2
    assert [e for e in logs if e["event"] == "planner.audio_retention.free_space_unreachable"]


def test_a_met_floor_logs_no_warning(audio: Audio, conn: psycopg.Connection) -> None:
    seed(audio, 2)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 1)

    with capture_logs() as logs:
        guard(conn, audio.dir, disk, floor=2 * GB)

    assert not [e for e in logs if "free_space" in e["event"]]


# ---------------------------------------------------------------- runaway bound


def test_the_guard_stops_once_the_recorded_bytes_evicted_cover_the_deficit(
    audio: Audio, conn: psycopg.Connection
) -> None:
    paths = seed(audio, 8)
    disk = StuckDisk(total=100 * GB, free=2 * GB - 25)  # deficit 25: 3 rows of 10

    with capture_logs() as logs:
        report = guard(conn, audio.dir, disk, floor=2 * GB)

    assert report.evicted_for_space == 3
    assert audio.paths() == paths[3:]
    assert [e for e in logs if e["event"] == "planner.audio_retention.free_space_unreachable"]


def test_a_deficit_covered_exactly_stops_there(audio: Audio, conn: psycopg.Connection) -> None:
    paths = seed(audio, 8)
    disk = StuckDisk(total=100 * GB, free=2 * GB - 30)  # deficit 30: exactly 3 rows

    report = guard(conn, audio.dir, disk, floor=2 * GB)

    assert report.evicted_for_space == 3
    assert audio.paths() == paths[3:]


def test_a_deficit_one_byte_more_needs_a_fourth_row(audio: Audio, conn: psycopg.Connection) -> None:
    paths = seed(audio, 8)
    disk = StuckDisk(total=100 * GB, free=2 * GB - 31)

    report = guard(conn, audio.dir, disk, floor=2 * GB)

    assert report.evicted_for_space == 4
    assert audio.paths() == paths[4:]


# ------------------------------------------------------------ floor vs volume


@pytest.mark.parametrize("floor", [10 * GB, 10 * GB + 1])
def test_a_floor_at_or_above_the_volume_never_wipes_audio(
    audio: Audio, conn: psycopg.Connection, floor: int
) -> None:
    seed(audio, 3)
    disk = FakeDisk(audio.dir, total=10 * GB, free=GB)

    with capture_logs() as logs:
        report = guard(conn, audio.dir, disk, floor=floor)

    assert report.evicted_for_space == 0
    assert len(audio.paths()) == 3
    (line,) = [e for e in logs if e["event"] == "planner.audio_retention.free_space_floor_invalid"]
    assert (line["floor"], line["total"]) == (floor, 10 * GB)
    assert report.free_bytes == GB


def test_a_floor_one_byte_below_the_volume_is_a_real_floor(
    audio: Audio, conn: psycopg.Connection
) -> None:
    seed(audio, 3)
    disk = FakeDisk(audio.dir, total=10 * GB, free=GB)

    with capture_logs() as logs:
        report = guard(conn, audio.dir, disk, floor=10 * GB - 1)

    assert report.evicted_for_space == 3  # unreachable, so everything evictable goes
    assert not [e for e in logs if e["event"] == "planner.audio_retention.free_space_floor_invalid"]


def test_the_percent_floor_is_taken_from_the_measured_total(
    audio: Audio, conn: psycopg.Connection
) -> None:
    seed(audio, 3)
    disk = FakeDisk(audio.dir, total=1000, free=49)  # 5 % of 1000 = 50; 2 GB would be invalid

    report = enforce_audio_retention(
        conn, audio.dir, max_bytes=FAR, now=NOW, min_free_gb=0, min_free_pct=5, disk_usage=disk
    )

    assert report.evicted_for_space == 1
    assert report.free_bytes == 59


# ------------------------------------------------------------ unlink behaviour


def test_an_unlink_error_rolls_back_that_row_counts_an_error_and_moves_on(
    audio: Audio, conn: psycopg.Connection
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    audio.add("aa/stuck.opus", created_at=T0)
    audio.add("bb/ok.opus", created_at=T0 + DAY)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 1)
    (audio.dir / "aa").chmod(0o500)
    try:
        report = guard(conn, audio.dir, disk, floor=2 * GB)
    finally:
        (audio.dir / "aa").chmod(0o700)

    assert report.errors == 1
    assert report.evicted_for_space == 1
    assert audio.paths() == ["aa/stuck.opus"]


def test_a_missing_file_still_loses_its_row_and_free_space_is_measured_again(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/gone.opus", size=2, created_at=T0, file=False)
    audio.add("ab/real.opus", size=10, created_at=T0 + DAY)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 5)

    report = guard(conn, audio.dir, disk, floor=2 * GB)

    assert report.evicted_for_space == 2  # the missing file's row, then the real one
    assert report.bytes_freed == 10  # only the real file returned space
    assert len(disk.calls) == 3


# ------------------------------------------------------ measurement failures


def test_a_failing_disk_usage_leaves_phases_one_to_three_intact(
    audio: Audio, conn: psycopg.Connection
) -> None:
    audio.add("ab/expired.opus", created_at=T0, expires_at=NOW - DAY)
    audio.add("ab/kept.opus", created_at=T0 + DAY)

    def broken(_path: Path) -> Usage:
        raise PermissionError("no stat for you")

    with capture_logs() as logs:
        report = guard(conn, audio.dir, broken, floor=2 * GB)

    assert report.expired_deleted == 1
    assert (report.evicted_for_space, report.free_bytes, report.errors) == (0, None, 1)
    assert audio.paths() == ["ab/kept.opus"]
    (line,) = [e for e in logs if e["event"] == "planner.audio_retention.disk_usage_unavailable"]
    assert "no stat for you" in line["error"]


def test_a_failing_remeasurement_stops_the_guard_but_keeps_what_was_evicted(
    audio: Audio, conn: psycopg.Connection
) -> None:
    paths = seed(audio, 4)
    real = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 100)

    def flaky(path: Path) -> Usage:
        if len(real.calls) >= 2:  # first call and the first re-measure succeed
            raise OSError("went away")
        return real(path)

    with capture_logs() as logs:
        report = guard(conn, audio.dir, flaky, floor=2 * GB)

    assert report.evicted_for_space == 2
    assert audio.paths() == paths[2:]
    assert (report.free_bytes, report.errors) == (None, 1)
    assert len([e for e in logs if e["event"] == "planner.audio_retention.disk_usage_unavailable"]) == 1


@pytest.mark.parametrize("usage", [Usage(0, 0, 0), Usage(100 * GB, 0, -1)])
def test_a_nonsensical_reading_is_treated_as_a_failed_measurement(
    audio: Audio, conn: psycopg.Connection, usage: Usage
) -> None:
    seed(audio, 2)

    report = guard(conn, audio.dir, lambda _p: usage, floor=2 * GB)

    assert (report.evicted_for_space, report.free_bytes, report.errors) == (0, None, 1)
    assert len(audio.paths()) == 2


# ----------------------------------------------------------- report and log


def test_the_report_and_the_log_line_carry_the_new_fields(
    audio: Audio, conn: psycopg.Connection
) -> None:
    seed(audio, 3)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 1)

    with capture_logs() as logs:
        report = guard(conn, audio.dir, disk, floor=2 * GB)

    (line,) = [e for e in logs if e["event"] == "planner.audio_retention"]
    assert line["evicted_for_space"] == report.evicted_for_space == 1
    assert line["free_bytes"] == report.free_bytes == 2 * GB - 1 + 10
    assert report.bytes_freed == 10


def test_a_second_run_with_the_same_disk_state_evicts_nothing(
    audio: Audio, conn: psycopg.Connection
) -> None:
    seed(audio, 5)
    disk = FakeDisk(audio.dir, total=100 * GB, free=2 * GB - 25)
    first = guard(conn, audio.dir, disk, floor=2 * GB)

    second = guard(conn, audio.dir, disk, floor=2 * GB)

    assert first.evicted_for_space == 3
    assert second.evicted_for_space == 0
    assert second.free_bytes == first.free_bytes


# ------------------------------------------------------------ the real call


def test_the_default_measurement_is_the_real_disk_usage_of_audio_dir(
    audio: Audio, conn: psycopg.Connection
) -> None:
    seed(audio, 2)

    report = enforce_audio_retention(
        conn, audio.dir, max_bytes=FAR, now=NOW, min_free_gb=1e-9, min_free_pct=0
    )

    assert report.evicted_for_space == 0
    assert report.free_bytes is not None
    assert report.free_bytes == pytest.approx(shutil.disk_usage(audio.dir).free, abs=10**9)
