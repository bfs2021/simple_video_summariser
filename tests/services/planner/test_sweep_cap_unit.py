"""Settings and failure handling of the sweep cap and dry run (issue #99). No Postgres."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from common.config import Settings, get_settings
from services.planner import main as planner_main


def test_a_database_error_in_the_dry_run_goes_to_stderr_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:secretpw@127.0.0.1:1/db")
    get_settings.cache_clear()
    try:
        code = planner_main.main(["--sweep-dry-run"])
    finally:
        get_settings.cache_clear()

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "sweep dry run failed" in captured.err
    assert "secretpw" not in captured.err


def _settings(**values: Any) -> Settings:
    return Settings(**{"DATABASE_URL": "postgresql://u:p@h/db", **values})


def test_the_cap_defaults_to_100_and_accepts_zero() -> None:
    assert _settings().SWEEP_MAX_JOBS_ANTHROPIC == 100
    assert _settings(SWEEP_MAX_JOBS_ANTHROPIC="0").SWEEP_MAX_JOBS_ANTHROPIC == 0


@pytest.mark.parametrize("value", ["-1", "ten", "1.5", ""])
def test_a_negative_or_non_integer_cap_fails_validation(value: str) -> None:
    with pytest.raises(ValidationError, match="SWEEP_MAX_JOBS_ANTHROPIC"):
        _settings(SWEEP_MAX_JOBS_ANTHROPIC=value)
