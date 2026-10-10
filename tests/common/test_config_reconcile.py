"""RECONCILE_INTERVAL_SEC (issue #93)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from common.config import Settings

DB = "postgresql://u:p@h/db"


def _settings(**values: Any) -> Settings:
    return Settings(**{"DATABASE_URL": DB, **values})


def test_the_default_is_six_hours() -> None:
    assert _settings().RECONCILE_INTERVAL_SEC == 21600


def test_it_can_be_set_and_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECONCILE_INTERVAL_SEC", "900")

    assert Settings(DATABASE_URL=DB).RECONCILE_INTERVAL_SEC == 900  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1, -21600, "abc", "", 1.5, "1.5"])
def test_zero_negative_and_non_integer_values_are_rejected(bad: Any) -> None:
    with pytest.raises(ValidationError) as info:
        _settings(RECONCILE_INTERVAL_SEC=bad)

    assert "RECONCILE_INTERVAL_SEC" in str(info.value)


def test_it_is_documented_in_the_architecture_config_table() -> None:
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / "_docs" / "architecture.md").read_text("utf-8")

    assert "| `RECONCILE_INTERVAL_SEC` | `21600` | planner" in text
