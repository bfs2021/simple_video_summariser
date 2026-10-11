"""Migration ``0010_search_docs`` (issue #111)."""

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


def _objects(dsn: str) -> dict[str, bool]:
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            """
            SELECT to_regclass('search_docs') IS NOT NULL,
                   to_regclass('search_docs_fts_idx') IS NOT NULL,
                   to_regprocedure('refresh_search_doc(text)') IS NOT NULL,
                   to_regprocedure('search_docs_sync()') IS NOT NULL,
                   to_regprocedure('tsquery_positive_lexemes(tsquery)') IS NOT NULL,
                   (SELECT count(*) FROM pg_trigger
                    WHERE tgname LIKE '%\\_search\\_docs\\_trg') = 3
            """
        ).fetchone()
    assert row is not None
    names = ("table", "index", "refresh", "sync", "lexemes", "triggers")
    return dict(zip(names, row, strict=True))


def test_0010_is_chained_to_0009() -> None:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))

    revision = ScriptDirectory.from_config(config).get_revision("0010")

    assert revision is not None
    assert revision.down_revision == "0009"


@integration
def test_upgrade_backfills_every_video_and_downgrade_round_trips(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_for(postgres_dsn, monkeypatch)
    command.upgrade(config, "0009")
    with psycopg.connect(postgres_dsn) as conn:
        conn.execute("INSERT INTO videos (video_id, title) VALUES ('mig10full01', 'Zebra Talk')")
        conn.execute("INSERT INTO videos (video_id) VALUES ('mig10bare01')")
        tid = conn.execute(
            "INSERT INTO transcripts (video_id, source, segments, full_text)"
            " VALUES ('mig10full01', 'whisper', '[]', 'giraffes graze') RETURNING id"
        ).fetchone()
        assert tid is not None
        conn.execute(
            "INSERT INTO analyses (video_id, transcript_id, chunk_strategy, model,"
            " prompt_version, tldr) VALUES ('mig10full01', %s, 'x', 'm', 'p', 'about okapis')",
            (tid[0],),
        )
        conn.commit()
    assert not any(_objects(postgres_dsn).values())

    command.upgrade(config, "0010")

    assert all(_objects(postgres_dsn).values())
    with psycopg.connect(postgres_dsn) as conn:
        rows: dict[str, str] = dict(
            conn.execute(
                "SELECT video_id, fts::text FROM search_docs ORDER BY video_id"
            ).fetchall()
        )
        weights = conn.execute(
            "SELECT fts FROM search_docs WHERE video_id = 'mig10full01'"
        ).fetchone()
    assert rows["mig10bare01"] == ""
    full = rows["mig10full01"]
    assert "'zebra':1A" in full and "'okapi':" in full and "'giraff':" in full
    assert weights is not None

    command.downgrade(config, "0009")

    assert not any(_objects(postgres_dsn).values())
    # the previous schema is intact and the upgrade can be repeated
    command.upgrade(config, "0010")
    assert all(_objects(postgres_dsn).values())
    with psycopg.connect(postgres_dsn) as conn:
        count = conn.execute("SELECT count(*) FROM search_docs").fetchone()
    assert count == (2,)


@integration
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("zebra", ["zebra"]),
        ("zebras running", ["zebra", "run"]),
        ("zebra -dogs", ["zebra"]),
        ('"machine learning"', ["machin", "learn"]),
        ('zebra -"machine learning"', ["zebra"]),
        ("-dogs", []),
        ("the", []),
    ],
)
def test_tsquery_positive_lexemes(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch, query: str, expected: list[str]
) -> None:
    command.upgrade(_config_for(postgres_dsn, monkeypatch), "0010")
    with psycopg.connect(postgres_dsn) as conn:
        row = conn.execute(
            "SELECT tsquery_positive_lexemes(websearch_to_tsquery('english', %s))", (query,)
        ).fetchone()
    assert row is not None
    assert sorted(row[0]) == sorted(expected)


@integration
def test_tsquery_positive_lexemes_unescapes_quotes_and_backslashes(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    command.upgrade(_config_for(postgres_dsn, monkeypatch), "0010")
    with psycopg.connect(postgres_dsn) as conn:
        row = conn.execute(
            "SELECT tsquery_positive_lexemes(CAST(%s AS tsquery))",
            (r"'it''s' & 'a\\b' & !'gone'",),
        ).fetchone()
    assert row is not None
    assert sorted(row[0]) == ["a\\b", "it's"]
