"""Migration ``0012_backfill_requests`` (issue #115)."""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from psycopg import errors

from common.config import get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
CHANNEL = "UCuAXFkgsw1L7xaCfnd5JJOw"


def _config_for(dsn: str, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("DATABASE_URL", dsn)
    get_settings.cache_clear()
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    return config


def _table_exists(dsn: str) -> bool:
    with psycopg.connect(dsn) as conn:
        row = conn.execute("SELECT to_regclass('backfill_requests')").fetchone()
    return row is not None and row[0] is not None


def test_0012_is_chained_to_0011_and_is_the_only_head_after_it() -> None:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    script = ScriptDirectory.from_config(config)

    revision = script.get_revision("0012")

    assert revision is not None
    assert revision.down_revision == "0011"
    assert len(script.get_heads()) == 1


@pytest.mark.integration
def test_upgrade_downgrade_upgrade(postgres_dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config_for(postgres_dsn, monkeypatch)

    command.upgrade(config, "0011")
    assert not _table_exists(postgres_dsn)
    command.upgrade(config, "0012")
    assert _table_exists(postgres_dsn)
    command.downgrade(config, "0011")
    assert not _table_exists(postgres_dsn)
    command.upgrade(config, "head")
    assert _table_exists(postgres_dsn)


@pytest.mark.integration
def test_columns_checks_and_unique_index(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    command.upgrade(_config_for(postgres_dsn, monkeypatch), "head")
    with psycopg.connect(postgres_dsn, autocommit=True) as conn:
        conn.execute("INSERT INTO channels (channel_id) VALUES (%s)", (CHANNEL,))
        cols = conn.execute(
            "SELECT column_name, is_nullable, column_default FROM information_schema.columns"
            " WHERE table_name = 'backfill_requests' ORDER BY ordinal_position"
        ).fetchall()
        names = [c[0] for c in cols]
        assert names == [
            "id", "channel_id", "limit_n", "dry_run", "state", "attempts", "enqueued",
            "result", "error_class", "created_at", "updated_at", "finished_at",
        ]  # fmt: skip
        defaults = {c[0]: c[2] for c in cols}
        assert defaults["attempts"] == "0" and defaults["enqueued"] == "0"

        def insert(limit: int = 5, state: str = "pending", channel: str = CHANNEL) -> None:
            conn.execute(
                "INSERT INTO backfill_requests (channel_id, limit_n, dry_run, state)"
                " VALUES (%s, %s, true, %s)",
                (channel, limit, state),
            )

        for bad in (0, 5001):
            with pytest.raises(errors.CheckViolation):
                insert(limit=bad)
        with pytest.raises(errors.CheckViolation):
            insert(state="cancelled")
        with pytest.raises(errors.ForeignKeyViolation):
            insert(channel="UCzzzzzzzzzzzzzzzzzzzzzz")

        insert(limit=1)
        # A second active request for the channel is refused, in either state...
        with pytest.raises(errors.UniqueViolation):
            insert(limit=5000, state="running")
        # ...but finished ones do not count.
        insert(state="done")
        insert(state="failed")
        conn.execute("UPDATE backfill_requests SET state = 'done' WHERE state = 'pending'")
        insert(limit=5000, state="running")
