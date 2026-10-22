"""Migration ``0009_planner_task_runs`` (issue #98)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

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


def _column_default(dsn: str, name: str) -> str | None:
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "SELECT column_default FROM information_schema.columns "
            "WHERE table_name='planner_task_runs' AND column_name=%s",
            (name,),
        ).fetchone()
    assert row is not None
    return cast(str | None, row[0])


def test_0009_is_chained_to_0008() -> None:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))

    revision = ScriptDirectory.from_config(config).get_revision("0009")

    assert revision is not None
    assert revision.down_revision == "0008"


def test_0014_is_chained_to_0013() -> None:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))

    revisions = {revision.revision: revision for revision in ScriptDirectory.from_config(config).walk_revisions()}

    assert "0014" in revisions
    assert revisions["0014"].down_revision == "0013"


@integration
def test_upgrade_creates_the_table_and_downgrade_drops_it(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_for(postgres_dsn, monkeypatch)
    command.upgrade(config, "0013")
    with psycopg.connect(postgres_dsn, autocommit=True) as conn:
        conn.execute("INSERT INTO planner_task_runs (task_name, interval_sec) VALUES ('existing', 1)")
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
        "cumulative_count": ("bigint", "NO"),
    }
    assert _column_default(postgres_dsn, "cumulative_count") == "0"
    with psycopg.connect(postgres_dsn) as conn:
        assert conn.execute(
            "SELECT cumulative_count FROM planner_task_runs WHERE task_name='existing'"
        ).fetchone() == (0,)

    command.downgrade(config, "0013")
    assert "cumulative_count" not in _columns(postgres_dsn)

    command.downgrade(config, "0008")
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
