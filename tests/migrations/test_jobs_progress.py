"""Migration ``0016_jobs_progress`` (issue #245)."""

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


def _column_type(dsn: str) -> tuple[str, str] | None:
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "SELECT data_type, is_nullable FROM information_schema.columns "
            "WHERE table_name='jobs' AND column_name='progress'"
        ).fetchone()
    return None if row is None else (row[0], row[1])


def test_0016_is_chained_to_0015() -> None:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))

    revision = ScriptDirectory.from_config(config).get_revision("0016")

    assert revision is not None
    assert revision.down_revision == "0015"


@integration
def test_upgrade_adds_a_nullable_jsonb_column_and_downgrade_removes_it(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_for(postgres_dsn, monkeypatch)
    command.upgrade(config, "0015")
    assert _column_type(postgres_dsn) is None

    command.upgrade(config, "0016")

    assert _column_type(postgres_dsn) == ("jsonb", "YES")

    command.downgrade(config, "0015")

    assert _column_type(postgres_dsn) is None
