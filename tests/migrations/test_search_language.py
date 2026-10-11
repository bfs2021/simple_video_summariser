"""Migration ``0011_search_language`` (issue #112)."""

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

#: Every language the issue lists, with its Snowball configuration.
SUPPORTED = {
    "ar": "arabic", "hy": "armenian", "eu": "basque", "ca": "catalan", "da": "danish",
    "nl": "dutch", "en": "english", "et": "estonian", "fi": "finnish", "fr": "french",
    "de": "german", "el": "greek", "hi": "hindi", "hu": "hungarian", "id": "indonesian",
    "ga": "irish", "it": "italian", "lt": "lithuanian", "ne": "nepali", "no": "norwegian",
    "nb": "norwegian", "nn": "norwegian", "pt": "portuguese", "ro": "romanian",
    "ru": "russian", "sr": "serbian", "es": "spanish", "sv": "swedish", "ta": "tamil",
    "tr": "turkish", "yi": "yiddish",
}  # fmt: skip

CASES: list[tuple[str | None, str]] = [
    *SUPPORTED.items(),
    ("EN", "english"), ("en-US", "english"), ("en_GB", "english"), ("en-orig", "english"),
    ("  De  ", "german"), ("pt-BR", "portuguese"), ("PT_br", "portuguese"),
    ("nb-NO", "norwegian"),
    (None, "english"), ("", "english"), ("   ", "english"), ("\t\n", "english"),
    ("pl", "simple"), ("uk", "simple"), ("cs", "simple"), ("ja", "simple"), ("zh", "simple"),
    ("ko", "simple"), ("vi", "simple"), ("th", "simple"), ("haw", "simple"),
    ("yue", "simple"), ("xx", "simple"), ("123", "simple"), ("x" * 200, "simple"),
    ("-US", "english"), ("en-", "english"), ("-", "english"),
]  # fmt: skip


def _config_for(dsn: str, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("DATABASE_URL", dsn)
    get_settings.cache_clear()
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    return config


def _fts_expression(dsn: str) -> str:
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "SELECT pg_get_expr(d.adbin, d.adrelid) FROM pg_attrdef d"
            " JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum"
            " WHERE d.adrelid = 'transcripts'::regclass AND a.attname = 'fts'"
        ).fetchone()
    assert row is not None
    return str(row[0])


def test_0011_is_chained_to_0010() -> None:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))

    revision = ScriptDirectory.from_config(config).get_revision("0011")

    assert revision is not None
    assert revision.down_revision == "0010"


@integration
def test_upgrade_recomputes_rows_and_downgrade_restores_the_expression(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_for(postgres_dsn, monkeypatch)
    command.upgrade(config, "0010")
    with psycopg.connect(postgres_dsn) as conn:
        conn.execute("INSERT INTO videos (video_id, title) VALUES ('mig11video1', 'Titel')")
        conn.execute("INSERT INTO videos (video_id) VALUES ('mig11video2')")
        for vid, lang, text in (
            ("mig11video1", "en", "she was running home"),
            ("mig11video2", "de", "Die Häuser sind alt"),
        ):
            conn.execute(
                "INSERT INTO transcripts (video_id, source, language, segments, full_text)"
                " VALUES (%s, 'whisper', %s, '[]', %s)",
                (vid, lang, text),
            )
        conn.commit()
        before: dict[str, str] = dict(
            conn.execute("SELECT video_id, fts::text FROM transcripts").fetchall()
        )
        before_docs: dict[str, str] = dict(conn.execute("SELECT video_id, fts::text FROM search_docs").fetchall())
    assert "'english'" in _fts_expression(postgres_dsn)

    command.upgrade(config, "0011")

    assert "transcript_ts_config(language)" in _fts_expression(postgres_dsn)
    with psycopg.connect(postgres_dsn) as conn:
        after: dict[str, str] = dict(conn.execute("SELECT video_id, fts::text FROM transcripts").fetchall())
        german = conn.execute(
            "SELECT fts = to_tsvector('german', full_text) FROM transcripts"
            " WHERE video_id = 'mig11video2'"
        ).fetchone()
        docs: dict[str, str] = dict(conn.execute("SELECT video_id, fts::text FROM search_docs").fetchall())
        index = conn.execute(
            "SELECT 1 FROM pg_indexes WHERE indexname = 'transcripts_fts_idx'"
        ).fetchone()
    assert after["mig11video1"] == before["mig11video1"]  # English is byte-identical
    assert german == (True,)
    assert after["mig11video2"] != before["mig11video2"]
    assert docs["mig11video1"] == before_docs["mig11video1"]
    assert docs["mig11video2"] != before_docs["mig11video2"]  # search_docs follows
    assert index is not None

    command.downgrade(config, "0010")

    assert "'english'" in _fts_expression(postgres_dsn)
    with psycopg.connect(postgres_dsn) as conn:
        restored: dict[str, str] = dict(conn.execute("SELECT video_id, fts::text FROM transcripts").fetchall())
        restored_docs: dict[str, str] = dict(conn.execute("SELECT video_id, fts::text FROM search_docs").fetchall())
        gone = conn.execute(
            "SELECT to_regprocedure('transcript_ts_config(text)'),"
            " to_regprocedure('transcript_search_tsquery(text)')"
        ).fetchone()
    assert restored == before
    assert restored_docs == before_docs
    assert gone == (None, None)
    command.upgrade(config, "0011")


@integration
def test_save_transcript_and_updates_use_the_language(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from common.models import Segment
    from common.repo.transcripts import save_transcript

    command.upgrade(_config_for(postgres_dsn, monkeypatch), "head")
    with psycopg.connect(postgres_dsn) as conn:
        conn.execute("INSERT INTO videos (video_id) VALUES ('mig11save01')")
        text = "Les maisons sont anciennes"
        save_transcript(conn, "mig11save01", "whisper", "fr", "none", (Segment(0, 1, text),), None)
        row = conn.execute(
            "SELECT fts = to_tsvector('french', full_text) FROM transcripts"
        ).fetchone()
        assert row == (True,)

        conn.execute("UPDATE transcripts SET language = 'de'")
        row = conn.execute(
            "SELECT fts = to_tsvector('german', full_text) FROM transcripts"
        ).fetchone()
        assert row == (True,)


@integration
def test_mapping_table(postgres_dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    command.upgrade(_config_for(postgres_dsn, monkeypatch), "head")
    with psycopg.connect(postgres_dsn) as conn:
        for lang, expected in CASES:
            row = conn.execute(
                "SELECT transcript_ts_config(%s) = CAST(%s AS regconfig)",
                (lang, f"pg_catalog.{expected}"),
            ).fetchone()
            assert row == (True,), (lang, expected)


@integration
def test_mapping_is_schema_qualified_immutable_parallel_safe_and_not_strict(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    command.upgrade(_config_for(postgres_dsn, monkeypatch), "head")
    with psycopg.connect(postgres_dsn) as conn:
        conn.execute("SET search_path = ''")
        row = conn.execute(
            "SELECT public.transcript_ts_config('de') = CAST('pg_catalog.german' AS regconfig)"
        ).fetchone()
        assert row == (True,)
        props = conn.execute(
            "SELECT provolatile, proparallel, proisstrict FROM pg_proc"
            " WHERE proname IN ('transcript_ts_config')"
        ).fetchone()
    assert props == ("i", "s", False)


@integration
def test_every_config_the_mapping_returns_exists_and_is_in_the_search_tsquery(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    command.upgrade(_config_for(postgres_dsn, monkeypatch), "head")
    with psycopg.connect(postgres_dsn) as conn:
        returned = set()
        for lang, _ in CASES:
            row = conn.execute("SELECT transcript_ts_config(%s)::oid", (lang,)).fetchone()
            assert row is not None
            returned.add(row[0])
        existing = {r[0] for r in conn.execute("SELECT oid FROM pg_catalog.pg_ts_config")}
        assert returned <= existing
        assert {c for c in SUPPORTED.values()} | {"simple"} == {
            name for (name,) in conn.execute(
                "SELECT cfgname FROM pg_ts_config WHERE oid = ANY(%s)", (list(returned),)
            )
        }
        # The OR'd index condition must hold every configuration's own query.
        for lang, _ in CASES:
            ok = conn.execute(
                "SELECT to_tsvector(transcript_ts_config(%s), 'Häuser running') @@"
                " transcript_search_tsquery('Häuser running')",
                (lang,),
            ).fetchone()
            assert ok == (True,), repr(lang)
