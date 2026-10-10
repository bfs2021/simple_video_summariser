"""Migration ``0007_channel_poll_failures`` (issue #90)."""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config

from common.config import get_settings

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config_for(dsn: str, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("DATABASE_URL", dsn)
    get_settings.cache_clear()
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    return config


def _has_column(dsn: str, column: str) -> bool:
    with psycopg.connect(dsn) as conn:
        return conn.execute(
            "SELECT 1 FROM information_schema.columns"
            " WHERE table_name = 'channels' AND column_name = %s",
            (column,),
        ).fetchone() is not None


def test_backfill_marks_a_failing_row_with_a_lower_bound_and_leaves_healthy_rows(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_for(postgres_dsn, monkeypatch)
    command.upgrade(config, "0006")
    with psycopg.connect(postgres_dsn) as conn:
        conn.execute(
            "INSERT INTO channels (channel_id, last_polled, last_poll_err)"
            " VALUES ('UCbad', '2026-09-01T10:00:00Z', 'FeedNotFoundError: 404')"
        )
        conn.execute(
            "INSERT INTO channels (channel_id, last_polled) VALUES ('UCok', '2026-09-02T10:00:00Z')"
        )
        conn.execute("INSERT INTO channels (channel_id) VALUES ('UCnever')")
        conn.commit()

    command.upgrade(config, "0007")

    with psycopg.connect(postgres_dsn) as conn:
        rows = conn.execute(
            "SELECT channel_id, poll_failing_since = last_polled, poll_failing_since IS NULL,"
            " poll_failures FROM channels ORDER BY channel_id"
        ).fetchall()
    assert rows == [
        ("UCbad", True, False, 1),
        ("UCnever", None, True, 0),
        ("UCok", None, True, 0),
    ]


def test_check_constraints_reject_inconsistent_rows(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    command.upgrade(_config_for(postgres_dsn, monkeypatch), "head")
    with psycopg.connect(postgres_dsn, autocommit=True) as conn:
        for since, failures in (
            ("now()", 0),  # a start without failures
            ("NULL", 1),  # failures without a start
            ("now()", -1),  # negative
            ("NULL", -1),
        ):
            with pytest.raises(psycopg.errors.CheckViolation):
                conn.execute(
                    f"INSERT INTO channels (channel_id, poll_failing_since, poll_failures)"
                    f" VALUES ('UCx', {since}, {failures})"
                )
        conn.execute(
            "INSERT INTO channels (channel_id, poll_failing_since, poll_failures)"
            " VALUES ('UCx', now(), 2)"
        )


def test_upgrade_downgrade_upgrade_succeeds_with_channel_rows(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_for(postgres_dsn, monkeypatch)
    command.upgrade(config, "head")
    with psycopg.connect(postgres_dsn) as conn:
        conn.execute("INSERT INTO channels (channel_id) VALUES ('UC1')")
        conn.commit()

    command.downgrade(config, "-1")
    assert not _has_column(postgres_dsn, "poll_failing_since")
    assert not _has_column(postgres_dsn, "poll_failures")
    command.upgrade(config, "head")

    assert _has_column(postgres_dsn, "poll_failing_since")
    assert _has_column(postgres_dsn, "poll_failures")
