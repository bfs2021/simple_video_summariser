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


def test_0015_is_the_only_head_and_is_chained_to_0014() -> None:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    script = ScriptDirectory.from_config(config)

    revision = script.get_revision("0015")

    assert revision is not None
    assert revision.down_revision == "0014"
    assert script.get_heads() == ["0015"]


def _current_revision(dsn: str) -> str:
    with psycopg.connect(dsn) as conn:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    assert row is not None
    return str(row[0])


def _columns(dsn: str) -> set[str]:
    with psycopg.connect(dsn) as conn:
        return {
            str(row[0])
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'channels'"
            ).fetchall()
        }


def test_upgrade_adds_nullable_gap_columns_and_preserves_existing_rows(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_for(postgres_dsn, monkeypatch)
    command.upgrade(config, "0014")
    with psycopg.connect(postgres_dsn) as conn:
        conn.execute("INSERT INTO channels (channel_id) VALUES ('UCexisting')")
        conn.commit()

    command.upgrade(config, "0015")

    with psycopg.connect(postgres_dsn) as conn:
        row = conn.execute(
            "SELECT feed_gap_detected_at, feed_gap_reason FROM channels WHERE channel_id = 'UCexisting'"
        ).fetchone()
    assert row == (None, None)


def test_constraints_accept_only_paired_finite_gap_states(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    command.upgrade(_config_for(postgres_dsn, monkeypatch), "0015")
    with psycopg.connect(postgres_dsn, autocommit=True) as conn:
        for reason in ("suspected", "no_anchor", "listing_window_exhausted"):
            conn.execute(
                "INSERT INTO channels (channel_id, feed_gap_detected_at, feed_gap_reason) VALUES (%s, now(), %s)",
                (f"UC{reason}", reason),
            )
        for detected, reason in (("now()", "NULL"), ("NULL", "'suspected'"), ("now()", "'other'")):
            with pytest.raises(psycopg.errors.CheckViolation):
                conn.execute(
                    f"INSERT INTO channels (channel_id, feed_gap_detected_at, feed_gap_reason) VALUES ('UCbad', {detected}, {reason})"
                )


def test_downgrade_removes_both_gap_columns_and_upgrade_restores_them(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_for(postgres_dsn, monkeypatch)
    command.upgrade(config, "0015")
    assert _current_revision(postgres_dsn) == "0015"
    command.downgrade(config, "0014")

    assert _current_revision(postgres_dsn) == "0014"
    assert not {"feed_gap_detected_at", "feed_gap_reason"} & _columns(postgres_dsn)

    command.upgrade(config, "0015")
    assert _current_revision(postgres_dsn) == "0015"
    assert {"feed_gap_detected_at", "feed_gap_reason"} <= _columns(postgres_dsn)
