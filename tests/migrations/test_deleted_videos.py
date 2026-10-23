from __future__ import annotations

from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from common.config import get_settings

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config_for(dsn: str, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("DATABASE_URL", dsn)
    get_settings.cache_clear()
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    return config


def _table_exists(dsn: str) -> bool:
    with psycopg.connect(dsn) as conn:
        row = conn.execute("SELECT to_regclass('deleted_videos')").fetchone()
    assert row is not None
    return bool(row[0] == "deleted_videos")


def test_0016_is_the_only_head_and_is_chained_to_0015() -> None:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    script = ScriptDirectory.from_config(config)

    revision = script.get_revision("0016")

    assert revision is not None
    assert revision.down_revision == "0015"
    assert script.get_heads() == ["0016"]


def test_upgrade_creates_tombstones_and_downgrade_removes_only_them(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_for(postgres_dsn, monkeypatch)
    command.upgrade(config, "0015")
    with psycopg.connect(postgres_dsn) as conn:
        conn.execute("INSERT INTO channels (channel_id) VALUES ('UCpreserved')")
        conn.commit()

    command.upgrade(config, "head")
    assert _table_exists(postgres_dsn)
    with psycopg.connect(postgres_dsn) as conn:
        conn.execute("INSERT INTO deleted_videos (video_id) VALUES ('dQw4w9WgXcQ')")
        row = conn.execute(
            "SELECT video_id, deleted_at IS NOT NULL FROM deleted_videos"
        ).fetchone()
    assert row == ("dQw4w9WgXcQ", True)

    command.downgrade(config, "0015")
    assert not _table_exists(postgres_dsn)
    with psycopg.connect(postgres_dsn) as conn:
        assert conn.execute(
            "SELECT channel_id FROM channels WHERE channel_id = 'UCpreserved'"
        ).fetchone() == ("UCpreserved",)

    command.upgrade(config, "head")
    assert _table_exists(postgres_dsn)
