"""Migration ``0009_planner_task_runs`` (issue #98)."""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from common.config import get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
integration = pytest.mark.integration


def _config_for(dsn: str, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("DATABASE_URL", dsn)
    get_settings.cache_clear()
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    return config


def _columns(dsn: str) -> dict[str, tuple[str, str]]:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_name='planner_task_runs'"
        ).fetchall()
    return {name: (dtype, nullable) for name, dtype, nullable in rows}


def test_0009_is_chained_to_0008() -> None:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))

    revision = ScriptDirectory.from_config(config).get_revision("0009")

    assert revision is not None
    assert revision.down_revision == "0008"


@integration
def test_upgrade_creates_the_table_and_downgrade_drops_it(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_for(postgres_dsn, monkeypatch)
    command.upgrade(config, "head")

    assert _columns(postgres_dsn) == {
        "task_name": ("text", "NO"),
        "interval_sec": ("double precision", "NO"),
        "last_started_at": ("timestamp with time zone", "YES"),
        "last_finished_at": ("timestamp with time zone", "YES"),
        "last_outcome": ("text", "YES"),
        "last_error": ("text", "YES"),
        "last_count": ("integer", "YES"),
        "last_ok_at": ("timestamp with time zone", "YES"),
        "last_skipped_locked_at": ("timestamp with time zone", "YES"),
    }

    command.downgrade(config, "-1")
    assert _columns(postgres_dsn) == {}

    command.upgrade(config, "head")
    assert "task_name" in _columns(postgres_dsn)


@integration
def test_outcome_check_rejects_unknown_values_and_task_name_is_the_key(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    command.upgrade(_config_for(postgres_dsn, monkeypatch), "head")
    with psycopg.connect(postgres_dsn, autocommit=True) as conn:
        conn.execute("INSERT INTO planner_task_runs (task_name, interval_sec) VALUES ('a', 1)")
        with pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute("INSERT INTO planner_task_runs (task_name, interval_sec) VALUES ('a', 1)")
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(
                "INSERT INTO planner_task_runs (task_name, interval_sec, last_outcome) "
                "VALUES ('b', 1, 'skipped_locked')"
            )
