"""The wiring around ``run_channel_backfill`` (issue #115, QA round 2).

Everything in ``test_backfill.py`` calls ``run_channel_backfill`` directly, so a
wrong keyword in ``build_registry``'s ``channel_backfill`` closure, or a wrong
setting handed to ``backfill()``, would pass the suite while the feature is dead
or mis-configured in production (the #92 failure class). These run the real
registry closure and the real settings pass-through. Needs Postgres.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import psycopg
import pytest
from pydantic import SecretStr

from common.config import Settings
from common.models import CatalogEntry, ChannelCatalog
from common.queue import PostgresQueue
from common.repo.backfill_requests import create_request, get_request
from services.planner import main as planner_main
from services.planner.backfill import run_channel_backfill

pytestmark = pytest.mark.integration

CHANNEL = "UCuAXFkgsw1L7xaCfnd5JJOw"


def _entries(n: int) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(f"wr{i:09d}", f"title {i}", 100) for i in range(n))


class RecordingCatalog:
    """Stands in for ``YouTubeCatalog``: remembers its constructor kwargs, no network."""

    instances: ClassVar[list[RecordingCatalog]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        RecordingCatalog.instances.append(self)

    def list_uploads(self, channel_id: str, *, limit: int) -> ChannelCatalog:
        entries = _entries(3)
        return ChannelCatalog(channel_id, entries[:limit], len(entries))


@pytest.fixture
def db(conn: psycopg.Connection[Any]) -> psycopg.Connection[Any]:
    conn.execute("INSERT INTO channels (channel_id, active) VALUES (%s, true)", (CHANNEL,))
    conn.commit()
    return conn


def _request(conn: psycopg.Connection[Any], *, dry_run: bool = False) -> int:
    rid = create_request(conn, CHANNEL, limit_n=10, dry_run=dry_run).request_id
    conn.commit()
    return rid


def test_the_registered_task_runs_end_to_end_with_the_configured_catalog(
    db: psycopg.Connection[Any], head_dsn: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The closure, through the real ``with_conn``, read back on a fresh connection."""
    RecordingCatalog.instances = []
    monkeypatch.setattr(planner_main, "YouTubeCatalog", RecordingCatalog)
    cookies = tmp_path / "cookies.txt"
    settings = Settings(DATABASE_URL=SecretStr(head_dsn), YTDLP_COOKIE_FILE=cookies)
    rid = _request(db)
    task = {t.name: t for t in planner_main.build_registry(settings, lambda: psycopg.connect(head_dsn))}[
        "channel_backfill"
    ]

    built_by_registry = len(RecordingCatalog.instances)  # the reconcile task's catalog

    assert task.fn() == 3

    with psycopg.connect(head_dsn) as fresh:
        assert fresh.execute("SELECT count(*) FROM videos").fetchone() == (3,)
        assert fresh.execute("SELECT count(*) FROM jobs WHERE kind = 'ingest'").fetchone() == (3,)
        row = fresh.execute(
            "SELECT state, enqueued FROM backfill_requests WHERE id = %s", (rid,)
        ).fetchone()
    assert row == ("done", 3)
    # the architecture's 300 s listing timeout and the cookie file reach the adapter
    assert [c.kwargs for c in RecordingCatalog.instances[built_by_registry:]] == [
        {"timeout": 300, "cookie_file": cookies}
    ]


def test_a_dry_run_prefers_whisper_when_configured(
    db: psycopg.Connection[Any], head_dsn: str
) -> None:
    rid = _request(db, dry_run=True)
    settings = Settings(DATABASE_URL=SecretStr(head_dsn), PREFER_WHISPER=True)

    run_channel_backfill(db, PostgresQueue(db), RecordingCatalog(), settings=settings)

    row = get_request(db, CHANNEL, rid)
    db.commit()
    assert row is not None and row.result is not None
    assert [v["subtitles"] for v in row.result["videos"]] == ["transcribe"] * 3


def test_the_default_dry_run_leaves_subtitles_unchecked(
    db: psycopg.Connection[Any], head_dsn: str
) -> None:
    rid = _request(db, dry_run=True)
    settings = Settings(DATABASE_URL=SecretStr(head_dsn), PREFER_WHISPER=False)

    run_channel_backfill(db, PostgresQueue(db), RecordingCatalog(), settings=settings)

    row = get_request(db, CHANNEL, rid)
    db.commit()
    assert row is not None and row.result is not None
    assert [v["subtitles"] for v in row.result["videos"]] == ["unchecked"] * 3


@pytest.mark.parametrize(
    ("summarizer", "priced_model", "expect_cost"),
    [("anthropic", "claude-test", True), ("ollama", "ollama-test", False)],
)
def test_the_cost_estimate_uses_the_summarizers_own_model(
    db: psycopg.Connection[Any],
    head_dsn: str,
    summarizer: str,
    priced_model: str,
    expect_cost: bool,
) -> None:
    # history priced under the anthropic model only: the anthropic estimate must find it
    # (a swap to OLLAMA_MODEL would not); ollama never prices at all
    db.execute("INSERT INTO videos (video_id, duration_sec) VALUES ('hist0000001', 10)")
    t = db.execute(
        "INSERT INTO transcripts (video_id, source, segments, full_text)"
        " VALUES ('hist0000001', 'whisper', '[]', %s) RETURNING id",
        ("x" * 1000,),
    ).fetchone()
    assert t is not None
    db.execute(
        "INSERT INTO analyses (video_id, transcript_id, chunk_strategy, model, prompt_version,"
        " tldr, cost_usd) VALUES ('hist0000001', %s, 's', 'claude-test', 'v1', 't', 0.01)",
        (t[0],),
    )
    db.commit()
    rid = _request(db, dry_run=True)
    settings = Settings(
        DATABASE_URL=SecretStr(head_dsn),
        SUMMARIZER=summarizer,  # type: ignore[arg-type]
        ANTHROPIC_MODEL="claude-test",
        OLLAMA_MODEL="ollama-test",
    )

    run_channel_backfill(db, PostgresQueue(db), RecordingCatalog(), settings=settings)

    row = get_request(db, CHANNEL, rid)
    db.commit()
    assert row is not None and row.result is not None
    cost = row.result["estimate"]["llm"]["cost_usd"]
    assert (cost is not None) is expect_cost
