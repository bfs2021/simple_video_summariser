"""Fast unit tests for the free-space floor of audio retention (issue #105). No database.

The guard's measurement paths that end before any row is touched need no
connection, so they pass a sentinel and would fail loudly if it were used.
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from structlog.testing import capture_logs

from services.planner import retention
from services.planner.retention import enforce_audio_retention, min_free_bytes

GB = 10**9


class Usage(NamedTuple):
    total: int
    used: int
    free: int


class NoConnection:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the database was used ({name})")


def _guard(
    tmp_path: Path, disk_usage: Any, *, gb: float = 1, pct: float = 0
) -> tuple[int | None, retention._Counters]:
    counters = retention._Counters()
    free = retention._free_space_guard(
        NoConnection(),  # type: ignore[arg-type]
        tmp_path,
        counters,
        min_free_gb=gb,
        min_free_pct=pct,
        disk_usage=disk_usage,
    )
    return free, counters


# ------------------------------------------------------------------ min_free_bytes


@pytest.mark.parametrize(
    ("total", "gb", "pct", "expected"),
    [
        (100 * GB, 2, 5, 5 * GB),  # percent wins
        (10 * GB, 2, 5, 2 * GB),  # bytes win
        (100 * GB, 0, 0, 0),
        (100 * GB, 5, 5, 5 * GB),  # tie
        (100 * GB, 0, 5, 5 * GB),
        (100 * GB, 2, 0, 2 * GB),
        (1000, 0, 0.5, 5),
        (10_001, 0, 5, 501),  # ceil(500.05)
        (10_000, 0, 5, 500),  # exact: no rounding up
        (10_000, 0, 0.07, 7),  # a float product is 7.000000000000001 and would round up to 8
        (3, 0, 33.333, 1),  # ceil(0.99999)
        (0, 2, 5, 2 * GB),
    ],
)
def test_min_free_bytes_is_the_larger_of_the_byte_and_percent_floors(
    total: int, gb: float, pct: float, expected: int
) -> None:
    assert min_free_bytes(total, min_free_gb=gb, min_free_pct=pct) == expected


def test_min_free_bytes_returns_an_int() -> None:
    assert isinstance(min_free_bytes(10 * GB, min_free_gb=0.5, min_free_pct=1.5), int)


# -------------------------------------------------------------------- signature


def test_enforce_audio_retention_keeps_the_guard_off_by_default() -> None:
    params = inspect.signature(enforce_audio_retention).parameters

    assert params["min_free_gb"].default == 0
    assert params["min_free_pct"].default == 0
    assert params["disk_usage"].default is shutil.disk_usage
    for name in ("min_free_gb", "min_free_pct", "disk_usage"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


# ------------------------------------------------------- guard: nothing measured


def test_floor_zero_reads_nothing(tmp_path: Path) -> None:
    def disk_usage(path: Path) -> Usage:
        raise AssertionError("disk_usage must not be called")

    free, counters = _guard(tmp_path, disk_usage, gb=0, pct=0)

    assert free is None
    assert counters.errors == 0


def test_disk_usage_is_called_on_audio_dir_itself(tmp_path: Path) -> None:
    seen: list[Path] = []

    def disk_usage(path: Path) -> Usage:
        seen.append(path)
        return Usage(100 * GB, 0, 100 * GB)

    _guard(tmp_path, disk_usage)

    assert seen == [tmp_path]


# ------------------------------------------------------- guard: failed measurement


@pytest.mark.parametrize(
    "exc", [PermissionError("denied"), FileNotFoundError("gone"), OSError(38, "ENOSYS")]
)
def test_a_failing_measurement_skips_the_guard_and_counts_one_error(
    tmp_path: Path, exc: OSError
) -> None:
    def disk_usage(path: Path) -> Usage:
        raise exc

    with capture_logs() as logs:
        free, counters = _guard(tmp_path, disk_usage)

    assert free is None
    assert counters.errors == 1
    lines = [e for e in logs if e["event"] == "planner.audio_retention.disk_usage_unavailable"]
    assert len(lines) == 1
    assert str(exc) in lines[0]["error"]


@pytest.mark.parametrize(
    "usage", [Usage(0, 0, 0), Usage(-1, 0, 5), Usage(100, 0, -1), Usage(0, 0, 100)]
)
def test_a_nonsensical_reading_is_a_measurement_failure(tmp_path: Path, usage: Usage) -> None:
    with capture_logs() as logs:
        free, counters = _guard(tmp_path, lambda _p: usage)

    assert free is None
    assert counters.errors == 1
    assert [e for e in logs if e["event"] == "planner.audio_retention.disk_usage_unavailable"]


def test_free_zero_on_a_sane_volume_is_a_valid_reading(tmp_path: Path) -> None:
    # free == 0 is a full disk, not a failure; the floor (1 GB < 100 GB) is
    # unmet, so it would evict, which needs the database.
    with pytest.raises(AssertionError, match="the database was used"):
        _guard(tmp_path, lambda _p: Usage(100 * GB, 100 * GB, 0))


# ----------------------------------------------------- guard: floor not below total


@pytest.mark.parametrize("floor_gb", [10, 11])
def test_a_floor_at_or_above_the_volume_size_evicts_nothing(
    tmp_path: Path, floor_gb: float
) -> None:
    with capture_logs() as logs:
        free, counters = _guard(
            tmp_path, lambda _p: Usage(10 * GB, 9 * GB, GB), gb=floor_gb, pct=0
        )

    assert free == GB
    assert counters.errors == 0
    (line,) = [e for e in logs if e["event"] == "planner.audio_retention.free_space_floor_invalid"]
    assert (line["floor"], line["total"]) == (floor_gb * GB, 10 * GB)


def test_a_floor_one_byte_below_the_volume_size_is_valid(tmp_path: Path) -> None:
    # free is below the floor, so the guard goes on to the database.
    with pytest.raises(AssertionError, match="the database was used"):
        _guard(tmp_path, lambda _p: Usage(10 * GB, 9 * GB, GB), gb=9.999999999)


def test_free_at_the_floor_evicts_nothing_and_returns_the_reading(tmp_path: Path) -> None:
    free, counters = _guard(tmp_path, lambda _p: Usage(10 * GB, 8 * GB, 2 * GB), gb=2)

    assert free == 2 * GB
    assert counters.errors == 0
