"""Migration ``0008_jobs_alerted_at`` (issue #67)."""

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


def _has_column(dsn: str) -> bool:
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "SELECT 1 FROM information_schema.columns WHERE table_name='jobs' "
            "AND column_name='alerted_at'"
        ).fetchone()
    return row is not None


def test_0008_is_chained_to_0007() -> None:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))

    revision = ScriptDirectory.from_config(config).get_revision("0008")

    assert revision is not None
    assert revision.down_revision == "0007"


@integration
def test_upgrade_adds_the_column_backfills_dead_rows_and_downgrade_reverses(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_for(postgres_dsn, monkeypatch)
    command.upgrade(config, "0007")
    with psycopg.connect(postgres_dsn) as conn:
        conn.execute("INSERT INTO videos (video_id) VALUES ('mig08video')")
        conn.execute(
            "INSERT INTO jobs (video_id, kind, state, error_class, finished_at) VALUES "
            "('mig08video', 'ingest', 'dead', 'TOOL_FAILURE', '2026-01-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO jobs (video_id, kind, state) VALUES ('mig08video', 'transcribe', 'pending')"
        )
    assert not _has_column(postgres_dsn)

    command.upgrade(config, "0008")

    assert _has_column(postgres_dsn)
    with psycopg.connect(postgres_dsn) as conn:
        rows: dict[str, bool] = dict(
            conn.execute(
                "SELECT state, alerted_at IS NOT NULL FROM jobs WHERE video_id='mig08video'"
            ).fetchall()
        )
    # Old dead rows are not re-announced after the upgrade; others stay NULL.
    assert rows == {"dead": True, "pending": False}

    command.downgrade(config, "0007")

    assert not _has_column(postgres_dsn)
