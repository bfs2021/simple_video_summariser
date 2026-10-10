"""Integration tests for the re-analysis sweep (issue #37). Needs Postgres."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from pydantic import SecretStr

from common.config import Settings
from common.queue import PostgresQueue, analyze_dedupe_key
from services.planner.sweep import REANALYSIS_PRIORITY, run_reanalysis_sweep

pytestmark = pytest.mark.integration

VERSION = "v2"
SUMMARIZER = "ollama"
KEY = analyze_dedupe_key(VERSION, SUMMARIZER)
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _queue(conn: psycopg.Connection[Any]) -> PostgresQueue:
    # Settings are only read by claim/reap; pass explicitly to avoid the environment.
    settings = Settings(DATABASE_URL=SecretStr("postgresql://u:p@h/db"))
    return PostgresQueue(conn, settings=settings)


def _sweep(conn: psycopg.Connection[Any], **kwargs: Any) -> int:
    params: dict[str, Any] = {"prompt_version": VERSION, "summarizer_name": SUMMARIZER}
    params.update(kwargs)
    return run_reanalysis_sweep(conn, _queue(conn), **params)


def _video(
    conn: psycopg.Connection[Any],
    video_id: str,
    *,
    age_days: int | None = 0,
    unavailable: str | None = None,
) -> None:
    published = None if age_days is None else BASE - timedelta(days=age_days)
    conn.execute(
        "INSERT INTO videos (video_id, published_at, unavailable) VALUES (%s, %s, %s)",
        (video_id, published, unavailable),
    )


def _transcript(
    conn: psycopg.Connection[Any], video_id: str, source: str = "youtube_manual", text: str = "hi"
) -> int:
    row = conn.execute(
        "INSERT INTO transcripts (video_id, source, segments, full_text) "
        "VALUES (%s, %s, '[]', %s) RETURNING id",
        (video_id, source, text),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _analysis(
    conn: psycopg.Connection[Any], video_id: str, transcript_id: int, version: str, model: str = "m"
) -> None:
    conn.execute(
        "INSERT INTO analyses (video_id, transcript_id, chunk_strategy, model, prompt_version, "
        "tldr) VALUES (%s, %s, 'c', %s, %s, 't')",
        (video_id, transcript_id, model, version),
    )


def _ready(conn: psycopg.Connection[Any], video_id: str, **video: Any) -> int:
    _video(conn, video_id, **video)
    return _transcript(conn, video_id)


def _jobs(conn: psycopg.Connection[Any]) -> list[tuple[str, str, str, int, str]]:
    return conn.execute(
        "SELECT video_id, kind, dedupe_key, priority, state FROM jobs ORDER BY id"
    ).fetchall()


def test_priority_constant_is_minus_five() -> None:
    assert REANALYSIS_PRIORITY == -5


def test_empty_database_returns_zero(conn: psycopg.Connection[Any]) -> None:
    assert _sweep(conn) == 0
    assert _jobs(conn) == []


def test_enqueues_one_analyze_job_per_candidate_at_low_priority(
    conn: psycopg.Connection[Any],
) -> None:
    _ready(conn, "a")

    assert _sweep(conn) == 1

    assert _jobs(conn) == [("a", "analyze", KEY, -5, "pending")]


def test_video_without_transcript_is_skipped(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "a")

    assert _sweep(conn) == 0


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_video_with_only_blank_transcripts_is_skipped(
    conn: psycopg.Connection[Any], text: str
) -> None:
    _video(conn, "a")
    _transcript(conn, "a", text=text)

    assert _sweep(conn) == 0


def test_blank_transcript_next_to_a_real_one_still_qualifies(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "a")
    _transcript(conn, "a", "youtube_auto", text="")
    _transcript(conn, "a", "whisper", text="words")

    assert _sweep(conn) == 1


def test_video_analysed_at_current_version_is_skipped_whatever_the_model(
    conn: psycopg.Connection[Any],
) -> None:
    tid = _ready(conn, "a")
    _analysis(conn, "a", tid, VERSION, model="some-other-model")

    assert _sweep(conn) == 0


def test_video_with_only_older_version_analyses_is_a_candidate(
    conn: psycopg.Connection[Any],
) -> None:
    tid = _ready(conn, "a")
    _analysis(conn, "a", tid, "v1")

    assert _sweep(conn) == 1


def test_switching_summarizer_alone_does_not_trigger_a_sweep(
    conn: psycopg.Connection[Any],
) -> None:
    tid = _ready(conn, "a")
    _analysis(conn, "a", tid, VERSION, model="qwen")

    assert _sweep(conn, summarizer_name="anthropic") == 0


def test_rolling_back_to_a_version_every_video_has_enqueues_nothing(
    conn: psycopg.Connection[Any],
) -> None:
    for vid in ("a", "b"):
        tid = _ready(conn, vid)
        _analysis(conn, vid, tid, "v1")
        _analysis(conn, vid, tid, "v2")

    assert _sweep(conn, prompt_version="v1") == 0


def test_one_job_per_video_even_with_several_transcripts(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "a")
    _transcript(conn, "a", "youtube_manual")
    _transcript(conn, "a", "whisper")

    assert _sweep(conn) == 1
    assert len(_jobs(conn)) == 1


def test_unavailable_video_is_still_a_candidate(conn: psycopg.Connection[Any]) -> None:
    _ready(conn, "a", unavailable="private")

    assert _sweep(conn) == 1


@pytest.mark.parametrize("state", ["pending", "running", "dead"])
def test_video_with_an_analyze_job_at_this_key_in_state_is_skipped(
    conn: psycopg.Connection[Any], state: str
) -> None:
    _ready(conn, "a")
    conn.execute(
        "INSERT INTO jobs (video_id, kind, dedupe_key, state) VALUES ('a', 'analyze', %s, %s)",
        (KEY, state),
    )

    assert _sweep(conn) == 0
    assert len(_jobs(conn)) == 1


def test_done_job_does_not_block_a_video_still_lacking_an_analysis(
    conn: psycopg.Connection[Any],
) -> None:
    _ready(conn, "a")
    conn.execute(
        "INSERT INTO jobs (video_id, kind, dedupe_key, state) VALUES ('a', 'analyze', %s, 'done')",
        (KEY,),
    )

    assert _sweep(conn) == 1


def test_jobs_with_another_key_or_kind_do_not_block(conn: psycopg.Connection[Any]) -> None:
    _ready(conn, "a")
    conn.execute(
        "INSERT INTO jobs (video_id, kind, dedupe_key, state) VALUES "
        "('a', 'analyze', 'v1:ollama', 'dead'), ('a', 'ingest', %s, 'pending')",
        (KEY,),
    )

    assert _sweep(conn) == 1


def test_pending_priority_zero_job_is_left_alone(conn: psycopg.Connection[Any]) -> None:
    _ready(conn, "a")
    _queue(conn).enqueue("analyze", "a", dedupe_key=KEY, priority=0)

    assert _sweep(conn) == 0
    assert [(j[0], j[3]) for j in _jobs(conn)] == [("a", 0)]


def test_candidates_are_ordered_newest_first_nulls_last_then_video_id(
    conn: psycopg.Connection[Any],
) -> None:
    _ready(conn, "old", age_days=10)
    _ready(conn, "nodate", age_days=None)
    _ready(conn, "new", age_days=1)
    _ready(conn, "tie_b", age_days=5)
    _ready(conn, "tie_a", age_days=5)

    assert _sweep(conn) == 5

    assert [j[0] for j in _jobs(conn)] == ["new", "tie_a", "tie_b", "old", "nodate"]


@pytest.mark.parametrize(
    ("candidates", "limit", "expected"),
    [(2, 3, 2), (3, 3, 3), (4, 3, 3)],
)
def test_limit_boundaries(
    conn: psycopg.Connection[Any], candidates: int, limit: int, expected: int
) -> None:
    for i in range(candidates):
        _ready(conn, f"v{i}", age_days=i)

    assert _sweep(conn, limit=limit) == expected
    assert len(_jobs(conn)) == expected


def test_repeated_calls_work_through_the_corpus_by_limit(
    conn: psycopg.Connection[Any],
) -> None:
    for i in range(5):
        _ready(conn, f"v{i}", age_days=i)

    assert [_sweep(conn, limit=2) for _ in range(4)] == [2, 2, 1, 0]
    assert [j[0] for j in _jobs(conn)] == ["v0", "v1", "v2", "v3", "v4"]


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_raises_value_error(
    conn: psycopg.Connection[Any], limit: int
) -> None:
    with pytest.raises(ValueError):
        _sweep(conn, limit=limit)


def test_running_twice_enqueues_nothing_the_second_time(
    conn: psycopg.Connection[Any],
) -> None:
    _ready(conn, "a")
    _ready(conn, "b")

    assert _sweep(conn) == 2
    assert _sweep(conn) == 0


def test_dead_lettered_job_is_not_re_enqueued_on_the_next_tick(
    conn: psycopg.Connection[Any],
) -> None:
    _ready(conn, "a")
    assert _sweep(conn) == 1
    conn.execute("UPDATE jobs SET state = 'dead', finished_at = now()")

    assert _sweep(conn) == 0
    assert [j[4] for j in _jobs(conn)] == ["dead"]


def test_database_error_propagates_and_earlier_jobs_stay_enqueued(
    conn: psycopg.Connection[Any],
) -> None:
    _ready(conn, "a", age_days=1)
    _ready(conn, "b", age_days=2)

    class FailingSecond:
        def __init__(self, inner: PostgresQueue) -> None:
            self.inner = inner
            self.calls = 0

        def enqueue(self, *args: Any, **kwargs: Any) -> int | None:
            self.calls += 1
            if self.calls == 2:
                raise psycopg.OperationalError("boom")
            return self.inner.enqueue(*args, **kwargs)

    queue = FailingSecond(_queue(conn))
    with pytest.raises(psycopg.OperationalError):
        run_reanalysis_sweep(
            conn,
            queue,  # type: ignore[arg-type]
            prompt_version=VERSION,
            summarizer_name=SUMMARIZER,
        )

    assert [j[0] for j in _jobs(conn)] == ["a"]
    assert _sweep(conn) == 1


def test_logs_exactly_one_info_line_with_the_summary(
    conn: psycopg.Connection[Any], caplog: pytest.LogCaptureFixture
) -> None:
    for i in range(3):
        _ready(conn, f"v{i}", age_days=i)

    with caplog.at_level(logging.INFO, logger="services.planner.sweep"):
        _sweep(conn, limit=2)

    records = [r for r in caplog.records if r.name == "services.planner.sweep"]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.INFO
    assert record.__dict__["prompt_version"] == VERSION
    assert record.__dict__["summarizer"] == SUMMARIZER
    assert record.__dict__["enqueued"] == 2
    assert record.__dict__["limit_reached"] is True
    assert record.__dict__["limit"] == 2


def test_limit_reached_is_false_when_candidates_run_out(
    conn: psycopg.Connection[Any], caplog: pytest.LogCaptureFixture
) -> None:
    _ready(conn, "a")

    with caplog.at_level(logging.INFO, logger="services.planner.sweep"):
        _sweep(conn, limit=5)

    record = next(r for r in caplog.records if r.name == "services.planner.sweep")
    assert record.__dict__["limit_reached"] is False


# --- durability across the connection lifetime (#199) ------------------------------------


def test_enqueued_jobs_survive_the_connection_closing(
    conn: psycopg.Connection[Any], head_dsn: str
) -> None:
    # The planner closes the task's connection without committing, so a job that is
    # only visible on the sweep's own connection is lost. Commit the seed data the
    # way earlier ticks would have, run the sweep, close, and look from a fresh one.
    _ready(conn, "a")
    _ready(conn, "b")
    conn.commit()

    assert _sweep(conn) == 2
    conn.close()

    with psycopg.connect(head_dsn) as other:
        assert other.execute(
            "SELECT video_id, kind, dedupe_key, priority, state FROM jobs ORDER BY video_id"
        ).fetchall() == [
            ("a", "analyze", KEY, -5, "pending"),
            ("b", "analyze", KEY, -5, "pending"),
        ]


def test_the_registered_planner_task_persists_its_jobs(
    conn: psycopg.Connection[Any], head_dsn: str
) -> None:
    # Through the real ``with_conn`` wrapper (open, run, close without commit),
    # which is the path production takes.
    from services.planner.main import build_registry

    _ready(conn, "a")
    conn.commit()
    settings = Settings(
        DATABASE_URL=SecretStr(head_dsn), PROMPT_VERSION=VERSION, SUMMARIZER="ollama"
    )
    registry = build_registry(settings, lambda: psycopg.connect(head_dsn))
    task = next(t for t in registry if t.name == "reanalysis_sweep")

    assert task.fn() == 1

    with psycopg.connect(head_dsn) as other:
        assert other.execute("SELECT count(*) FROM jobs WHERE kind = 'analyze'").fetchone() == (1,)
