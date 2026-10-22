"""``GET /metrics`` end to end on Postgres (issue #60).

Seeds every table the endpoint reads and asserts every sample value, then checks
that no seeded marker string leaks. Timestamps are explicit.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from common.config import Settings, get_settings
from services.api import routes_ops
from services.api.main import create_app
from services.planner.main import (
    PostgresAdvisoryLocks,
    PostgresRunRecorder,
    build_registry,
)
from services.planner.scheduler import Scheduler
from tests.services.api.conftest import LogSink

pytestmark = pytest.mark.integration

T0 = datetime(2026, 1, 1, tzinfo=UTC)
MARK = "LEAKMARK"
API_KEY = "sk-ant-LEAKMARK-key"


@pytest.fixture
def db(
    head_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> Iterator[psycopg.Connection[Any]]:
    monkeypatch.setenv("DATABASE_URL", head_dsn)
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    monkeypatch.setenv("PROMPT_VERSION", "v2")
    get_settings.cache_clear()
    with psycopg.connect(head_dsn, autocommit=True) as conn:
        yield conn


@pytest.fixture
def api(db: psycopg.Connection[Any], logs: LogSink) -> Iterator[TestClient]:
    with TestClient(create_app()) as client:
        yield client


def _sample(body: str, line_prefix: str) -> str:
    (line,) = [ln for ln in body.splitlines() if ln.startswith(line_prefix + " ")]
    return line.rsplit(" ", 1)[1]


def _job(db: psycopg.Connection[Any], vid: str, kind: str, state: str, secs: float | None) -> None:
    db.execute(
        "INSERT INTO jobs (video_id, kind, state, dedupe_key, started_at, finished_at,"
        " last_error, error_class, locked_by)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            vid,
            kind,
            state,
            f"k{vid}{kind}{state}{secs}",
            None if secs is None else T0,
            None if secs is None else T0 + timedelta(seconds=secs),
            f"{MARK}-last-error",
            f"{MARK}-class",
            f"{MARK}-worker",
        ),
    )


def test_an_empty_database_gives_zero_filled_metrics(api: TestClient) -> None:
    response = api.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert len(re.findall(r"^queue_depth\{", body, re.MULTILINE)) == 12
    assert [
        line for line in body.splitlines() if line.startswith("feed_gap_channels{")
    ] == [
        'feed_gap_channels{reason="suspected"} 0',
        'feed_gap_channels{reason="no_anchor"} 0',
        'feed_gap_channels{reason="listing_window_exhausted"} 0',
    ]
    assert _sample(body, "llm_cost_usd_total") == "0"
    assert _sample(body, "speaker_coercions_total") == "0"
    assert _sample(body, "audio_bytes_used") == "0"
    assert _sample(body, "reanalysis_backlog_videos") == "0"
    # Samples only: the "# HELP"/"# TYPE" lines of an empty family are valid
    # exposition, so anchor to the start of a line.
    assert re.search(r"^whisper_rtf ", body, re.MULTILINE) is None
    assert re.search(r"^llm_tokens_total\{", body, re.MULTILINE) is None


def test_every_metric_is_derived_from_the_seeded_rows(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    for channel_id, reason in (
        ("UCgap1", "suspected"),
        ("UCgap2", "suspected"),
        ("UCgap3", "no_anchor"),
        ("UCgap4", "listing_window_exhausted"),
    ):
        db.execute(
            "INSERT INTO channels (channel_id, feed_gap_detected_at, feed_gap_reason) VALUES (%s, %s, %s)",
            (channel_id, T0, reason),
        )
    for vid in ("v1", "v2", "v3", "v4"):
        db.execute("INSERT INTO videos (video_id, title) VALUES (%s, %s)", (vid, f"{MARK}-title"))
    # jobs: several kinds and states, one unknown kind, durations on bucket edges
    _job(db, "v1", "ingest", "pending", None)
    _job(db, "v1", "ingest", "dead", None)
    _job(db, "v2", "transcribe", "running", None)
    _job(db, "v3", "mystery", "pending", None)
    _job(db, "v1", "analyze", "done", 60.0)
    _job(db, "v2", "analyze", "done", 60.001)
    _job(db, "v3", "analyze", "done", 4.0)
    # transcripts: an older and a newer whisper row, a non-numeric rtf, a manual row
    ids: dict[str, int] = {}
    for vid, source, meta, minutes in (
        ("v1", "whisper", '{"rtf": 0.5}', 1),
        ("v2", "whisper", '{"rtf": 0.125}', 2),
        ("v3", "whisper", '{"rtf": "fast"}', 3),
        ("v4", "youtube_manual", '{"rtf": 9.0}', 4),
    ):
        row = db.execute(
            "INSERT INTO transcripts (video_id, source, segments, full_text, engine_meta, created_at)"
            " VALUES (%s, %s, '[]', 'text', %s::jsonb, %s) RETURNING id",
            (vid, source, meta, T0 + timedelta(minutes=minutes)),
        ).fetchone()
        assert row is not None
        ids[vid] = int(row[0])
    # analyses: two models, NULL and non-NULL cost and speakers_coerced; PROMPT_VERSION is v2
    for vid, model, version, tokens, cost, coerced in (
        ("v1", 'qwen"x\n', "v2", (100, 10), 0.5, 3),
        ("v2", "claude", "v2", (7, 3), None, None),
        ("v3", "claude", "v1", (1, 1), 0.25, 0),
    ):
        db.execute(
            "INSERT INTO analyses (video_id, transcript_id, chunk_strategy, model, prompt_version,"
            " tldr, input_tokens, output_tokens, cost_usd, speakers_coerced)"
            " VALUES (%s, %s, 's', %s, %s, %s, %s, %s, %s, %s)",
            (vid, ids[vid], model, version, f"{MARK}-tldr", *tokens, cost, coerced),
        )
    db.execute(
        "INSERT INTO media (video_id, path, bytes, expires_at) VALUES"
        " ('v1', %s, 1000, %s), ('v2', %s, 234, %s)",
        (f"{MARK}/a.opus", T0, f"{MARK}/b.opus", T0),
    )

    response = api.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    body = response.text
    assert _sample(body, 'queue_depth{kind="ingest",state="pending"}') == "1"
    assert _sample(body, 'queue_depth{kind="ingest",state="dead"}') == "1"
    assert _sample(body, 'queue_depth{kind="transcribe",state="running"}') == "1"
    assert _sample(body, 'queue_depth{kind="other",state="pending"}') == "1"
    assert _sample(body, 'queue_depth{kind="analyze",state="done"}') == "3"
    assert _sample(body, 'feed_gap_channels{reason="suspected"}') == "2"
    assert _sample(body, 'feed_gap_channels{reason="no_anchor"}') == "1"
    assert _sample(body, 'feed_gap_channels{reason="listing_window_exhausted"}') == "1"
    assert _sample(body, 'job_duration_seconds_bucket{kind="analyze",le="5"}') == "1"
    assert _sample(body, 'job_duration_seconds_bucket{kind="analyze",le="15"}') == "1"
    assert _sample(body, 'job_duration_seconds_bucket{kind="analyze",le="60"}') == "2"  # 60.0 yes, 60.001 no
    assert _sample(body, 'job_duration_seconds_bucket{kind="analyze",le="300"}') == "3"
    assert _sample(body, 'job_duration_seconds_bucket{kind="analyze",le="+Inf"}') == "3"
    assert _sample(body, 'job_duration_seconds_count{kind="analyze"}') == "3"
    assert float(_sample(body, 'job_duration_seconds_sum{kind="analyze"}')) == pytest.approx(124.001)
    assert _sample(body, 'job_duration_seconds_count{kind="ingest"}') == "0"
    assert _sample(body, "whisper_rtf") == "0.125"
    assert _sample(body, 'llm_tokens_total{model="claude",direction="input"}') == "8"
    assert _sample(body, 'llm_tokens_total{model="claude",direction="output"}') == "4"
    assert _sample(body, 'llm_tokens_total{model="qwen\\"x\\n",direction="input"}') == "100"
    assert _sample(body, "llm_cost_usd_total") == "0.75"
    assert _sample(body, "speaker_coercions_total") == "3"
    assert _sample(body, "audio_bytes_used") == "1234"
    # v3 (rtf 'fast' whisper) and v1/v2 are at... v1, v2 have a v2 analysis; v3 only v1 -> backlog 1.
    # v4 has a manual transcript and no analysis at all -> backlog 2.
    assert _sample(body, "reanalysis_backlog_videos") == "2"

    for line in body.splitlines():
        assert "\r" not in line
    # nothing free-text, secret or identifying leaks into the body
    for marker in (MARK, API_KEY, "v1", "v2", "v3", "v4", "password", "@"):
        assert marker not in body.replace("# ", ""), marker
    label_keys = set(re.findall(r"[{,]([a-z]+)=", body))
    assert label_keys == {"kind", "state", "model", "direction", "le", "reason"}


def test_registered_reaper_run_is_persisted_and_scraped(
    api: TestClient,
    db: psycopg.Connection[Any],
    head_dsn: str,
    tmp_path: Path,
) -> None:
    db.execute("INSERT INTO videos (video_id) VALUES ('stale-job')")
    db.execute(
        "INSERT INTO jobs (video_id, kind, state, dedupe_key, attempts, locked_by, locked_at, heartbeat_at) "
        "VALUES ('stale-job', 'ingest', 'running', 'default', 1, 'lost', now() - interval '10 minutes', "
        "now() - interval '10 minutes')"
    )
    settings = Settings(DATABASE_URL=head_dsn, AUDIO_DIR=tmp_path, REAP_AFTER_SEC=300)  # type: ignore[arg-type]
    connect = lambda: psycopg.connect(head_dsn)
    reap_task = build_registry(settings, connect)[0]
    locks = PostgresAdvisoryLocks(connect)
    recorder = PostgresRunRecorder(connect)
    scheduler = Scheduler(
        [reap_task],
        locks,
        recorder=recorder,
        sleep=lambda _seconds: None,
        install_signal_handlers=False,
    )
    try:
        assert scheduler.run_once() is True
        first = api.get("/metrics").text
        assert _sample(first, "reaped_jobs_total") == "1"
        assert _sample(first, 'planner_task_last_success{task="reap"}') == "1"
        float(_sample(first, 'planner_task_last_run_timestamp_seconds{task="reap"}'))

        assert scheduler.run_once() is True
        second = api.get("/metrics").text
        assert _sample(second, "reaped_jobs_total") == "1"
    finally:
        recorder.close()
        locks.close()


def test_two_scrapes_of_an_unchanged_database_are_byte_identical(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    db.execute("INSERT INTO videos (video_id) VALUES ('v1')")
    _job(db, "v1", "ingest", "done", 3.5)

    assert api.get("/metrics").content == api.get("/metrics").content


def test_twenty_five_models_are_capped_at_twenty_plus_other(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    db.execute("INSERT INTO videos (video_id) VALUES ('v1')")
    row = db.execute(
        "INSERT INTO transcripts (video_id, source, segments, full_text)"
        " VALUES ('v1', 'whisper', '[]', 't') RETURNING id"
    ).fetchone()
    assert row is not None
    for i in range(25):
        db.execute(
            "INSERT INTO analyses (video_id, transcript_id, chunk_strategy, model, prompt_version,"
            " tldr, input_tokens, output_tokens) VALUES ('v1', %s, 's', %s, 'v2', 't', %s, 0)",
            (row[0], f"model{i:02d}", 1000 - i),
        )

    body = api.get("/metrics").text

    models = set(re.findall(r'llm_tokens_total\{model="([^"]*)"', body))
    assert models == {f"model{i:02d}" for i in range(20)} | {"other"}
    assert _sample(body, 'llm_tokens_total{model="other",direction="input"}') == str(
        sum(1000 - i for i in range(20, 25))
    )


def test_a_scrape_cannot_write_to_the_database(
    api: TestClient, db: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def writing_query(conn: psycopg.Connection[Any]) -> dict[tuple[str, str], int]:
        conn.execute("INSERT INTO videos (video_id) VALUES ('sneaky')")
        return {}

    monkeypatch.setattr(routes_ops, "queue_depth", writing_query)

    response = api.get("/metrics")

    assert response.status_code == 503
    assert response.text == "database unavailable\n"
    assert db.execute("SELECT count(*) FROM videos").fetchone() == (0,)
