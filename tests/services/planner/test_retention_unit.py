"""Fast unit tests for planner audio retention helpers (issue #36). No database."""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from pathlib import Path

import pytest

from services.planner import retention
from services.planner.retention import (
    ORPHAN_GRACE,
    RetentionReport,
    UnsafeMediaPath,
    max_bytes_from_gb,
    resolve_media_path,
)


@pytest.mark.parametrize(
    ("gb", "expected"),
    [(20, 20_000_000_000), (0.5, 500_000_000), (1, 1_000_000_000), (0.000001, 1000)],
)
def test_max_bytes_from_gb_uses_decimal_gigabytes(gb: float, expected: int) -> None:
    assert max_bytes_from_gb(gb) == expected


def test_max_bytes_from_gb_returns_an_int() -> None:
    assert isinstance(max_bytes_from_gb(0.5), int)


def test_orphan_grace_is_24_hours() -> None:
    assert ORPHAN_GRACE == timedelta(hours=24)


def test_report_is_a_frozen_dataclass_with_the_documented_fields() -> None:
    names = [f.name for f in dataclasses.fields(RetentionReport)]
    assert names == [
        "expired_deleted",
        "evicted",
        "orphans_deleted",
        "bytes_freed",
        "bytes_retained",
        "skipped_running",
        "errors",
        "evicted_for_space",
        "free_bytes",
    ]
    report = RetentionReport(0, 0, 0, 0, 0, 0, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.errors = 1  # type: ignore[misc]


def test_resolve_media_path_joins_a_relative_path_under_audio_dir(
    tmp_path: Path,
) -> None:
    assert (
        resolve_media_path(tmp_path, "ab/abc.opus")
        == tmp_path.resolve() / "ab" / "abc.opus"
    )


@pytest.mark.parametrize(
    "rel",
    [
        "/etc/passwd",
        "../outside.opus",
        "ab/../../outside.opus",
        "ab/../../../etc/passwd",
        "..",
        "",
        ".",
        "ab/..",
    ],
)
def test_resolve_media_path_rejects_unsafe_paths(tmp_path: Path, rel: str) -> None:
    with pytest.raises(UnsafeMediaPath):
        resolve_media_path(tmp_path, rel)


def test_resolve_media_path_rejects_a_symlink_that_escapes_audio_dir(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.opus").write_bytes(b"x")
    (audio / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafeMediaPath):
        resolve_media_path(audio, "link/x.opus")


def test_resolve_media_path_rejects_a_file_symlink_that_escapes_audio_dir(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    target = tmp_path / "secret.opus"
    target.write_bytes(b"x")
    (audio / "ab.opus").symlink_to(target)

    with pytest.raises(UnsafeMediaPath):
        resolve_media_path(audio, "ab.opus")


def test_resolve_media_path_allows_a_missing_file(tmp_path: Path) -> None:
    # A row whose file is already gone must still resolve, so it can be deleted.
    assert (
        resolve_media_path(tmp_path, "ab/gone.opus").parent == tmp_path.resolve() / "ab"
    )


def test_module_exposes_the_single_public_entrypoint() -> None:
    assert callable(retention.enforce_audio_retention)
