"""Migration ``0006_metrics_columns`` (issue #60).

The first two tests read the revision file only and run without a database. The
integration tests need Postgres (``postgres_dsn``).
"""

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


def _column(dsn: str, table: str, column: str) -> tuple[str, str] | None:
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            """
            SELECT data_type, is_nullable FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s AND column_name = %s
            """,
            (table, column),
        ).fetchone()
    return None if row is None else (str(row[0]), str(row[1]))


def test_0006_is_chained_to_0005() -> None:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    scripts = ScriptDirectory.from_config(config)

    revision = scripts.get_revision("0006")

    assert revision is not None
    assert revision.down_revision == "0005"


def test_0006_downgrade_drops_exactly_what_upgrade_adds() -> None:
    source = (REPO_ROOT / "migrations" / "versions" / "0006_metrics_columns.py").read_text()

    assert "ADD COLUMN started_at TIMESTAMPTZ NULL" in source
    assert "ADD COLUMN speakers_coerced INTEGER NULL" in source
    assert "ALTER TABLE jobs DROP COLUMN started_at" in source
    assert "ALTER TABLE analyses DROP COLUMN speakers_coerced" in source


@integration
def test_upgrade_adds_both_nullable_columns_and_downgrade_drops_them(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_for(postgres_dsn, monkeypatch)
    command.upgrade(config, "0005")
    assert _column(postgres_dsn, "jobs", "started_at") is None
    assert _column(postgres_dsn, "analyses", "speakers_coerced") is None

    command.upgrade(config, "0006")

    assert _column(postgres_dsn, "jobs", "started_at") == ("timestamp with time zone", "YES")
    assert _column(postgres_dsn, "analyses", "speakers_coerced") == ("integer", "YES")

    command.downgrade(config, "0005")

    assert _column(postgres_dsn, "jobs", "started_at") is None
    assert _column(postgres_dsn, "analyses", "speakers_coerced") is None

    command.upgrade(config, "head")
    assert _column(postgres_dsn, "jobs", "started_at") is not None


@integration
def test_existing_rows_stay_null_not_zero(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_for(postgres_dsn, monkeypatch)
    command.upgrade(config, "0005")
    with psycopg.connect(postgres_dsn) as conn:
        conn.execute("INSERT INTO videos (video_id) VALUES ('v1')")
        conn.execute(
            "INSERT INTO jobs (video_id, kind, state, finished_at)"
            " VALUES ('v1', 'ingest', 'done', now())"
        )
        conn.execute(
            "INSERT INTO transcripts (video_id, source, segments, full_text)"
            " VALUES ('v1', 'whisper', '[]', 'text')"
        )
        conn.execute(
            "INSERT INTO analyses (video_id, transcript_id, chunk_strategy, model,"
            " prompt_version, tldr) SELECT 'v1', id, 's', 'm', 'p', 't' FROM transcripts"
        )
        conn.commit()

    command.upgrade(config, "0006")

    with psycopg.connect(postgres_dsn) as conn:
        assert conn.execute("SELECT started_at FROM jobs").fetchone() == (None,)
        assert conn.execute("SELECT speakers_coerced FROM analyses").fetchone() == (None,)
