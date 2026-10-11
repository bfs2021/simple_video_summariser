"""DB_POOL_MIN_SIZE / DB_POOL_MAX_SIZE / DB_POOL_TIMEOUT_SEC (issue #95)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from common.config import Settings

DB = "postgresql://u:p@h/db"
ROOT = Path(__file__).resolve().parents[2]


def _settings(**values: Any) -> Settings:
    return Settings(**{"DATABASE_URL": DB, **values})


def test_defaults_are_1_5_and_2_seconds() -> None:
    s = _settings()

    assert (s.DB_POOL_MIN_SIZE, s.DB_POOL_MAX_SIZE, s.DB_POOL_TIMEOUT_SEC) == (1, 5, 2)


def test_they_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_POOL_MIN_SIZE", "2")
    monkeypatch.setenv("DB_POOL_MAX_SIZE", "7")
    monkeypatch.setenv("DB_POOL_TIMEOUT_SEC", "3")

    s = Settings(DATABASE_URL=DB)  # type: ignore[arg-type]

    assert (s.DB_POOL_MIN_SIZE, s.DB_POOL_MAX_SIZE, s.DB_POOL_TIMEOUT_SEC) == (2, 7, 3)


def test_min_equal_to_max_is_accepted() -> None:
    s = _settings(DB_POOL_MIN_SIZE=3, DB_POOL_MAX_SIZE=3)

    assert s.DB_POOL_MIN_SIZE == s.DB_POOL_MAX_SIZE == 3


def test_min_above_max_is_rejected_and_names_the_variable() -> None:
    with pytest.raises(ValidationError) as info:
        _settings(DB_POOL_MIN_SIZE=6, DB_POOL_MAX_SIZE=5)

    assert "DB_POOL_MAX_SIZE" in str(info.value)


@pytest.mark.parametrize("name", ["DB_POOL_MIN_SIZE", "DB_POOL_MAX_SIZE", "DB_POOL_TIMEOUT_SEC"])
@pytest.mark.parametrize("bad", [0, -1])
def test_zero_and_negative_values_are_rejected(name: str, bad: int) -> None:
    with pytest.raises(ValidationError) as info:
        _settings(**{name: bad})

    assert name in str(info.value)


@pytest.mark.parametrize("name", ["DB_POOL_MIN_SIZE", "DB_POOL_MAX_SIZE"])
@pytest.mark.parametrize("bad", ["abc", "", 1.5])
def test_sizes_must_be_integers(name: str, bad: Any) -> None:
    with pytest.raises(ValidationError):
        _settings(**{name: bad})


def test_the_timeout_ceiling_is_four_seconds_inclusive() -> None:
    assert _settings(DB_POOL_TIMEOUT_SEC=4).DB_POOL_TIMEOUT_SEC == 4
    for bad in (4.001, 5, 60):
        with pytest.raises(ValidationError) as info:
            _settings(DB_POOL_TIMEOUT_SEC=bad)
        assert "DB_POOL_TIMEOUT_SEC" in str(info.value)


def test_the_architecture_doc_lists_the_settings_and_the_sizing_rule() -> None:
    text = (ROOT / "_docs" / "architecture.md").read_text("utf-8")

    assert "| `DB_POOL_MIN_SIZE` / `DB_POOL_MAX_SIZE` | `1` / `5` |" in text
    assert "| `DB_POOL_TIMEOUT_SEC` | `2` |" in text
    assert "api replicas x DB_POOL_MAX_SIZE + workers + planner + migrate" in text
    assert "max_connections" in text


def test_env_example_lists_the_settings_with_their_defaults() -> None:
    lines = (ROOT / ".env.example").read_text("utf-8").splitlines()

    for line in ("DB_POOL_MIN_SIZE=1", "DB_POOL_MAX_SIZE=5", "DB_POOL_TIMEOUT_SEC=2"):
        assert line in lines
