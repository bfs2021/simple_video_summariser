"""Audio retention (issue #36; architecture.md D6b, §6 ``media``, §7.2).

``enforce_audio_retention`` is one pass that keeps ``AUDIO_DIR`` bounded:

1. **Expiry** - retained audio whose ``expires_at`` has passed is deleted.
2. **Cap** - the oldest audio (``created_at``, then ``id``) is evicted until
   the total *recorded* size (``SUM(media.bytes)``, never a ``stat``) is at or
   below ``max_bytes``. Rows still inside their retention window are evicted
   too when the cap demands it.
3. **Orphans** - regular files under ``AUDIO_DIR`` that no ``media`` row
   references are deleted once their mtime is older than ``ORPHAN_GRACE``:
   files left behind by deleted videos, and ``.download-*`` / ``.tmp-*``
   leftovers of a crashed fetch.
4. **Free-space guard** (#105) - after phases 1-3, the real free space on the
   ``AUDIO_DIR`` filesystem is read (``disk_usage``). Below the floor
   (``min_free_bytes``: the larger of ``AUDIO_MIN_FREE_GB`` and
   ``AUDIO_MIN_FREE_PCT`` of the volume) the oldest audio is evicted, in the
   order of phase 2 and whatever the byte cap says, until free space is back
   at the floor. Free space is re-read after every unlinked file. The cap
   cannot see untracked files, other data on the volume, or an
   ``AUDIO_MAX_GB`` larger than the volume; this can.

Audio of a video with a ``running`` job is never deleted in phases 1, 2 and 4.
Each row is handled in its own transaction: lock it with ``FOR UPDATE SKIP
LOCKED``, re-check eligibility on the locked row, unlink the file, delete the
row, commit. A row another connection holds is skipped, not waited on.

``media.path`` is data, not trusted: see ``resolve_media_path``.

Scheduling and the singleton guard belong to the planner entrypoint (#38).
"""

from __future__ import annotations

import math
import os
import shutil
import stat
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

import psycopg
import structlog

from common.repo import media

UnsafeMediaPath = media.UnsafeMediaPath
resolve_media_path = media.resolve_media_path

_log = structlog.get_logger(__name__)

#: An unreferenced file must be older than this before phase 3 deletes it.
#: Well above the 2 h download timeout of #19, so an in-flight fetch that has
#: not registered its ``media`` row yet is never mistaken for an orphan.
ORPHAN_GRACE = timedelta(hours=24)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class RetentionReport:
    """What one ``enforce_audio_retention`` run did.

    ``bytes_freed`` counts recorded bytes of files actually unlinked (phases
    1 and 2) plus the on-disk size of deleted orphans. ``bytes_retained`` is
    ``SUM(media.bytes)`` after the run. ``errors`` counts rows or files that
    could not be handled and were logged. ``evicted_for_space`` counts rows
    deleted by the free-space guard (not in ``evicted``); ``free_bytes`` is
    the free space measured at the end of the pass, ``None`` when the guard
    was off or the measurement failed.
    """

    expired_deleted: int
    evicted: int
    orphans_deleted: int
    bytes_freed: int
    bytes_retained: int
    skipped_running: int
    errors: int
    evicted_for_space: int = 0
    free_bytes: int | None = None


def max_bytes_from_gb(gb: float) -> int:
    """Convert ``AUDIO_MAX_GB`` to bytes, with 1 GB = 10^9 bytes (decimal, not GiB).

    ``20`` -> ``20_000_000_000`` and ``0.5`` -> ``500_000_000``.
    """
    return round(gb * 10**9)


def min_free_bytes(total_bytes: int, *, min_free_gb: float, min_free_pct: float) -> int:
    """The free-space floor in bytes: the larger of the GB floor and the percent floor.

    ``min_free_gb`` uses the decimal-GB rule of ``max_bytes_from_gb``. The
    percent floor is ``ceil(total_bytes * min_free_pct / 100)``, computed on
    the decimal value of the percentage so that 0.07 % of 10000 is 7, not the 8
    that a float product (7.000000000000001) would round up to.
    """
    from_pct = math.ceil(Decimal(repr(min_free_pct)) * total_bytes / 100)
    return max(max_bytes_from_gb(min_free_gb), from_pct)


class _Counters:
    def __init__(self) -> None:
        self.expired_deleted = 0
        self.evicted = 0
        self.orphans_deleted = 0
        self.evicted_for_space = 0
        self.bytes_freed = 0
        self.errors = 0
        self.running: set[int] = set()


def enforce_audio_retention(
    conn: psycopg.Connection[Any],
    audio_dir: Path,
    *,
    max_bytes: int,
    now: datetime,
    min_free_gb: float = 0,
    min_free_pct: float = 0,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> RetentionReport:
    """Run expiry, cap and orphan cleanup once and return what happened.

    ``now`` must be timezone-aware and is the only clock used. The free-space
    guard (phase 4) is off unless ``min_free_gb`` or ``min_free_pct`` is
    non-zero; ``disk_usage(audio_dir)`` returns an object with ``.total`` and
    ``.free`` (the unprivileged-process free space, as ``shutil.disk_usage``
    gives). Raises
    ``OSError`` (after logging an error) if ``audio_dir`` is not a directory,
    before anything is deleted. Commits the connection's transaction as it
    goes (one commit per handled row).
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not audio_dir.is_dir():
        _log.error(
            "planner.audio_retention.audio_dir_unavailable", audio_dir=str(audio_dir)
        )
        if audio_dir.exists():
            raise NotADirectoryError(f"{audio_dir} is not a directory")
        raise FileNotFoundError(f"{audio_dir} does not exist")

    counters = _Counters()

    # Phase 1: expiry.
    expired = media.list_expired(conn, now)
    conn.commit()
    for row in expired:
        removed = _delete_one(conn, audio_dir, row.id, counters, expired_at=now)
        if removed is not None:
            counters.expired_deleted += 1

    # Phase 2: cap, on the rows that remain.
    total = media.total_bytes(conn)
    if total > max_bytes:
        candidates = media.list_oldest_first(conn)
        conn.commit()
        for row in candidates:
            if total <= max_bytes:
                break
            removed = _delete_one(
                conn, audio_dir, row.id, counters, max_bytes=max_bytes
            )
            if removed is not None:
                counters.evicted += 1
                total -= removed
    conn.commit()
    total = media.total_bytes(conn)
    conn.commit()
    if total > max_bytes:
        _log.warning(
            "planner.audio_retention.cap_unreachable",
            total_bytes=total,
            max_bytes=max_bytes,
        )

    # Phase 3: orphan files.
    _delete_orphans(conn, audio_dir, now, counters)

    # Phase 4: free-space guard, after phases 1-3 so their deletions count.
    free_bytes = _free_space_guard(
        conn,
        audio_dir,
        counters,
        min_free_gb=min_free_gb,
        min_free_pct=min_free_pct,
        disk_usage=disk_usage,
    )

    retained = media.total_bytes(conn)
    conn.commit()
    report = RetentionReport(
        expired_deleted=counters.expired_deleted,
        evicted=counters.evicted,
        orphans_deleted=counters.orphans_deleted,
        bytes_freed=counters.bytes_freed,
        bytes_retained=retained,
        skipped_running=len(counters.running),
        errors=counters.errors,
        evicted_for_space=counters.evicted_for_space,
        free_bytes=free_bytes,
    )
    _log.info(
        "planner.audio_retention",
        expired_deleted=report.expired_deleted,
        evicted=report.evicted,
        orphans_deleted=report.orphans_deleted,
        bytes_freed=report.bytes_freed,
        bytes_retained=report.bytes_retained,
        skipped_running=report.skipped_running,
        errors=report.errors,
        evicted_for_space=report.evicted_for_space,
        free_bytes=report.free_bytes,
    )
    return report


def _measure(
    audio_dir: Path, disk_usage: Callable[[Path], Any], counters: _Counters
) -> tuple[int, int] | None:
    """``(total, free)`` of the volume of ``audio_dir``, or ``None`` after logging a failure."""
    try:
        usage = disk_usage(audio_dir)
        total, free = int(usage.total), int(usage.free)
        if total <= 0 or free < 0:
            raise OSError(f"nonsensical disk usage: total={total} free={free}")
    except OSError as exc:
        _log.error("planner.audio_retention.disk_usage_unavailable", error=str(exc))
        counters.errors += 1
        return None
    return total, free


def _free_space_guard(
    conn: psycopg.Connection[Any],
    audio_dir: Path,
    counters: _Counters,
    *,
    min_free_gb: float,
    min_free_pct: float,
    disk_usage: Callable[[Path], Any],
) -> int | None:
    """Phase 4: evict oldest-first until free space is back at the floor (#105).

    Returns the free space last measured, or ``None`` when nothing was
    measured or a measurement failed. Never raises for a disk problem.
    """
    if min_free_gb == 0 and min_free_pct == 0:
        return None  # guard off: read nothing, evict nothing
    measured = _measure(audio_dir, disk_usage, counters)
    if measured is None:
        return None
    total, free = measured
    floor = min_free_bytes(total, min_free_gb=min_free_gb, min_free_pct=min_free_pct)
    if floor >= total:
        # A floor the volume can never meet would evict everything.
        _log.error("planner.audio_retention.free_space_floor_invalid", floor=floor, total=total)
        return free
    if free >= floor:
        return free

    deficit = floor - free
    evicted_bytes = 0  # recorded bytes of rows deleted by this phase
    candidates = media.list_oldest_first(conn)
    conn.commit()
    for row in candidates:
        if free >= floor or evicted_bytes >= deficit:
            break
        removed = _delete_one(conn, audio_dir, row.id, counters, low_space=True)
        if removed is None:
            continue
        counters.evicted_for_space += 1
        evicted_bytes += removed
        measured = _measure(audio_dir, disk_usage, counters)
        if measured is None:
            return None
        free = measured[1]
    if free < floor:
        _log.warning(
            "planner.audio_retention.free_space_unreachable", free_bytes=free, floor_bytes=floor
        )
    return free


def _delete_one(
    conn: psycopg.Connection[Any],
    audio_dir: Path,
    media_id: int,
    counters: _Counters,
    *,
    expired_at: datetime | None = None,
    max_bytes: int | None = None,
    low_space: bool = False,
) -> int | None:
    """Delete one row (and its file) in its own transaction.

    Returns the row's recorded bytes if the row was deleted, else ``None``
    (skipped, no longer eligible, or failed and rolled back).
    """
    try:
        outcome, row = media.lock_for_deletion(
            conn, media_id, expired_at=expired_at, max_bytes=max_bytes, low_space=low_space
        )
        if row is None:
            conn.rollback()
            if outcome is media.LockOutcome.RUNNING:
                counters.running.add(media_id)
            return None

        try:
            target = resolve_media_path(audio_dir, row.path)
        except UnsafeMediaPath:
            # Never touch the file; drop the row that points nowhere safe.
            _log.error(
                "planner.audio_retention.unsafe_path", media_id=row.id, path=row.path
            )
            counters.errors += 1
            media.delete_row(conn, row.id)
            conn.commit()
            return row.bytes

        try:
            target.unlink()
            counters.bytes_freed += row.bytes
        except FileNotFoundError:
            pass
        except OSError as exc:
            conn.rollback()
            _log.error(
                "planner.audio_retention.unlink_failed",
                media_id=row.id,
                path=row.path,
                error=str(exc),
            )
            counters.errors += 1
            return None

        media.delete_row(conn, row.id)
        conn.commit()
        return row.bytes
    except psycopg.Error:
        conn.rollback()
        _log.exception("planner.audio_retention.row_failed", media_id=media_id)
        counters.errors += 1
        return None


def _delete_orphans(
    conn: psycopg.Connection[Any], audio_dir: Path, now: datetime, counters: _Counters
) -> None:
    referenced = {
        PurePosixPath(p).as_posix() for p in media.list_referenced_paths(conn)
    }
    conn.commit()
    cutoff_ns = ((now - ORPHAN_GRACE - _EPOCH) // timedelta(microseconds=1)) * 1000

    def on_walk_error(exc: OSError) -> None:
        _log.error("planner.audio_retention.walk_failed", error=str(exc))
        counters.errors += 1

    for dirpath, _dirnames, filenames in os.walk(audio_dir, onerror=on_walk_error):
        for name in filenames:
            full = Path(dirpath) / name
            try:
                st = os.lstat(full)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):  # symlinks, sockets, ...
                continue
            if full.relative_to(audio_dir).as_posix() in referenced:
                continue
            if st.st_mtime_ns > cutoff_ns:
                continue
            try:
                full.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                _log.error(
                    "planner.audio_retention.orphan_unlink_failed",
                    path=str(full),
                    error=str(exc),
                )
                counters.errors += 1
                continue
            counters.orphans_deleted += 1
            counters.bytes_freed += st.st_size
