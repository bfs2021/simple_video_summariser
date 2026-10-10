"""Stale ``analyze`` jobs after a PROMPT_VERSION rollout (issue #100). Needs Postgres."""

from __future__ import annotations

import logging
from typing import Any

import psycopg
import pytest
from pydantic import SecretStr

from common.config import Settings
from common.queue import PostgresQueue, StaleAnalyzeResult
from services.planner.stale_analyze import resolve_stale_analyze_jobs
from services.planner.sweep import run_reanalysis_sweep

pytestmark = pytest.mark.integration

V = "aaaaaaaaaaa"
W = "bbbbbbbbbbb"
SUPERSEDED = "superseded: PROMPT_VERSION is now v2"


def _settings() -> Settings:
    return Settings(DATABASE_URL=SecretStr("postgresql://u:p@h/db"))


def _queue(conn: psycopg.Connection[Any]) -> PostgresQueue:
    return PostgresQueue(conn, settings=_settings())


def _resolve(conn: psycopg.Connection[Any], version: str = "v2") -> StaleAnalyzeResult:
    return _queue(conn).resolve_stale_analyze(version)


def _video(conn: psycopg.Connection[Any], video_id: str = V) -> None:
    conn.execute("INSERT INTO videos (video_id) VALUES (%s) ON CONFLICT DO NOTHING", (video_id,))


def _job(
    conn: psycopg.Connection[Any],
    key: str,
    *,
    video_id: str = V,
    kind: str = "analyze",
    state: str = "pending",
    priority: int = 0,
    attempts: int = 0,
    payload: str = "{}",
) -> int:
    row = conn.execute(
        "INSERT INTO jobs (video_id, kind, dedupe_key, state, priority, attempts, payload, "
        "run_after, finished_at) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, "
        "now() + interval '7 minutes', CASE WHEN %s IN ('done','dead') THEN now() END) "
        "RETURNING id",
        (video_id, kind, key, state, priority, attempts, payload, state),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _analysis(
    conn: psycopg.Connection[Any], version: str, video_id: str = V, model: str = "m"
) -> None:
    row = conn.execute(
        "INSERT INTO transcripts (video_id, source, segments, full_text) "
        "VALUES (%s, 'whisper', '[]', 'hi') RETURNING id",
        (video_id,),
    ).fetchone()
    assert row is not None
    conn.execute(
        "INSERT INTO analyses (video_id, transcript_id, chunk_strategy, model, prompt_version, "
        "tldr) VALUES (%s, %s, 'c', %s, %s, 't')",
        (video_id, row[0], model, version),
    )


def _row(conn: psycopg.Connection[Any], job_id: int) -> dict[str, Any]:
    cur = conn.execute(
        "SELECT dedupe_key, state, priority, attempts, payload, error_class, last_error, "
        "finished_at IS NOT NULL AS finished, run_after > now() AS deferred "
        "FROM jobs WHERE id = %s",
        (job_id,),
    )
    values = cur.fetchone()
    assert values is not None and cur.description is not None
    return dict(zip([d.name for d in cur.description], values, strict=True))


def _table(conn: psycopg.Connection[Any]) -> list[tuple[Any, ...]]:
    return conn.execute("SELECT * FROM jobs ORDER BY id").fetchall()


def test_an_empty_table_and_a_repeat_call_return_zeros(conn: psycopg.Connection[Any]) -> None:
    assert _resolve(conn) == StaleAnalyzeResult(0, 0)

    _video(conn)
    _job(conn, "v1:ollama")
    assert _resolve(conn) == StaleAnalyzeResult(1, 0)
    assert _resolve(conn) == StaleAnalyzeResult(0, 0)


def test_a_stale_job_is_rekeyed_in_place_keeping_everything_else_and_the_sweep_sees_it(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn)
    conn.execute(
        "INSERT INTO transcripts (video_id, source, segments, full_text) "
        "VALUES (%s, 'whisper', '[]', 'hi')",
        (V,),
    )
    job = _job(conn, "v1:ollama", priority=10, attempts=2, payload='{"force": true}')
    before = conn.execute(
        "SELECT id, priority, payload, attempts, run_after, state FROM jobs WHERE id = %s", (job,)
    ).fetchone()

    assert _resolve(conn) == StaleAnalyzeResult(1, 0)

    row = conn.execute(
        "SELECT id, priority, payload, attempts, run_after, state, dedupe_key FROM jobs "
        "WHERE id = %s",
        (job,),
    ).fetchone()
    assert row is not None and before is not None
    assert row[:6] == before
    assert row[6] == "v2:ollama"
    # The sweep sees the re-keyed job as active and adds no duplicate at -5.
    assert run_reanalysis_sweep(
        conn, _queue(conn), prompt_version="v2", summarizer_name="ollama"
    ) == 0
    assert conn.execute("SELECT count(*) FROM jobs").fetchone() == (1,)


def test_a_stale_job_beside_an_active_job_at_the_target_key_is_superseded(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn)
    old = _job(conn, "v1:ollama", priority=10)
    conn.execute("UPDATE jobs SET error_class = 'TOOL_FAILURE' WHERE id = %s", (old,))
    new = _job(conn, "v2:ollama")

    assert _resolve(conn) == StaleAnalyzeResult(0, 1)

    assert _row(conn, new)["state"] == "pending" and _row(conn, new)["dedupe_key"] == "v2:ollama"
    row = _row(conn, old)
    assert (row["state"], row["dedupe_key"], row["error_class"], row["last_error"]) == (
        "dead",
        "v1:ollama",
        None,
        SUPERSEDED,
    )
    assert row["finished"] is True


def test_a_running_job_at_the_target_key_also_blocks_the_rekey(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn)
    old = _job(conn, "v1:ollama")
    _job(conn, "v2:ollama", state="running")

    assert _resolve(conn) == StaleAnalyzeResult(0, 1)
    assert _row(conn, old)["state"] == "dead"


def test_work_that_dead_lettered_at_the_target_key_is_never_revived(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn)
    old = _job(conn, "v1:ollama")
    dead = _job(conn, "v2:ollama", state="dead")
    before = _row(conn, dead)

    assert _resolve(conn) == StaleAnalyzeResult(0, 1)

    assert _row(conn, dead) == before
    assert _row(conn, old)["state"] == "dead" and _row(conn, old)["dedupe_key"] == "v1:ollama"


def test_a_done_job_at_the_target_key_does_not_block_the_rekey(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn)
    old = _job(conn, "v1:ollama")
    _job(conn, "v2:ollama", state="done")

    assert _resolve(conn) == StaleAnalyzeResult(1, 0)
    assert _row(conn, old)["dedupe_key"] == "v2:ollama"


@pytest.mark.parametrize("model", ["m", "some-other-model"])
def test_an_analysis_at_the_new_version_supersedes_whatever_the_model(
    conn: psycopg.Connection[Any], model: str
) -> None:
    _video(conn)
    old = _job(conn, "v1:ollama")
    _analysis(conn, "v2", model=model)

    assert _resolve(conn) == StaleAnalyzeResult(0, 1)
    assert _row(conn, old)["state"] == "dead"


def test_an_analysis_at_another_version_does_not_block_the_rekey(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn)
    old = _job(conn, "v1:ollama")
    _analysis(conn, "v1")

    assert _resolve(conn) == StaleAnalyzeResult(1, 0)
    assert _row(conn, old)["dedupe_key"] == "v2:ollama"


def test_a_rollback_treats_the_newer_version_as_stale_the_same_way(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn)
    old = _job(conn, "v2:ollama")
    _analysis(conn, "v1")

    assert _resolve(conn, "v1") == StaleAnalyzeResult(0, 1)

    row = _row(conn, old)
    assert (row["state"], row["last_error"]) == ("dead", "superseded: PROMPT_VERSION is now v1")


def test_a_rollback_without_an_analysis_rekeys_down(conn: psycopg.Connection[Any]) -> None:
    _video(conn)
    old = _job(conn, "v2:ollama", priority=10)

    assert _resolve(conn, "v1") == StaleAnalyzeResult(1, 0)
    assert (_row(conn, old)["dedupe_key"], _row(conn, old)["priority"]) == ("v1:ollama", 10)


def test_each_summarizer_is_rekeyed_to_its_own_target(conn: psycopg.Connection[Any]) -> None:
    _video(conn)
    ollama = _job(conn, "v1:ollama")
    anthropic = _job(conn, "v1:anthropic")

    assert _resolve(conn) == StaleAnalyzeResult(2, 0)

    assert _row(conn, ollama)["dedupe_key"] == "v2:ollama"
    assert _row(conn, anthropic)["dedupe_key"] == "v2:anthropic"


def test_two_stale_rows_for_one_target_rekey_the_highest_priority_and_supersede_the_rest(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn)
    low = _job(conn, "v0:ollama", priority=0)
    high = _job(conn, "v1:ollama", priority=10)

    assert _resolve(conn) == StaleAnalyzeResult(1, 1)  # and no unique-index violation

    assert _row(conn, high)["dedupe_key"] == "v2:ollama" and _row(conn, high)["state"] == "pending"
    assert _row(conn, low)["state"] == "dead" and _row(conn, low)["dedupe_key"] == "v0:ollama"


def test_equal_priorities_rekey_the_lowest_id(conn: psycopg.Connection[Any]) -> None:
    _video(conn)
    first = _job(conn, "v1:ollama", priority=5)
    second = _job(conn, "v0:ollama", priority=5)
    third = _job(conn, "vx:ollama", priority=5)

    assert _resolve(conn) == StaleAnalyzeResult(1, 2)

    assert _row(conn, first)["dedupe_key"] == "v2:ollama"
    assert _row(conn, second)["state"] == "dead" and _row(conn, third)["state"] == "dead"


def test_videos_are_resolved_independently(conn: psycopg.Connection[Any]) -> None:
    _video(conn, V)
    _video(conn, W)
    a = _job(conn, "v1:ollama", video_id=V)
    b = _job(conn, "v1:ollama", video_id=W)
    _job(conn, "v2:ollama", video_id=W)

    assert _resolve(conn) == StaleAnalyzeResult(1, 1)

    assert _row(conn, a)["dedupe_key"] == "v2:ollama"
    assert _row(conn, b)["state"] == "dead"


def test_the_same_target_key_on_different_videos_is_rekeyed_for_each(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, V)
    _video(conn, W)
    a = _job(conn, "v1:ollama", video_id=V, priority=0)
    b = _job(conn, "v1:ollama", video_id=W, priority=10)

    assert _resolve(conn) == StaleAnalyzeResult(2, 0)

    assert _row(conn, a)["dedupe_key"] == "v2:ollama" and _row(conn, b)["dedupe_key"] == "v2:ollama"


def test_only_pending_analyze_jobs_at_another_version_are_touched(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn)
    _job(conn, "v1:ollama", state="running")
    _job(conn, "v1:anthropic", state="done")
    _job(conn, "v1:other", state="dead")
    _job(conn, "default", kind="ingest")
    _job(conn, "v1:ollama", kind="transcribe")
    _job(conn, "v2:ollama", video_id=W)  # already at the version (its video is not V)
    before = _table(conn)

    assert _resolve(conn) == StaleAnalyzeResult(0, 0)

    assert _table(conn) == before


@pytest.mark.parametrize("key", ["default", "fake", "v1:", ":ollama", "v1", "v1:fake:extra", "v1::x"])
def test_a_malformed_key_is_left_alone(conn: psycopg.Connection[Any], key: str) -> None:
    _video(conn)
    job = _job(conn, key)
    before = _row(conn, job)

    assert _resolve(conn) == StaleAnalyzeResult(0, 0)

    assert _row(conn, job) == before


def test_a_reaped_job_is_handled_by_the_next_call(conn: psycopg.Connection[Any]) -> None:
    _video(conn)
    job = _job(conn, "v1:ollama", state="running")
    assert _resolve(conn) == StaleAnalyzeResult(0, 0)

    conn.execute("UPDATE jobs SET state = 'pending' WHERE id = %s", (job,))  # the reaper

    assert _resolve(conn) == StaleAnalyzeResult(1, 0)
    assert _row(conn, job)["dedupe_key"] == "v2:ollama"


def test_a_job_enqueued_at_the_old_version_after_the_switch_is_rekeyed_next_call(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn)
    assert _resolve(conn) == StaleAnalyzeResult(0, 0)
    late = _queue(conn).enqueue("analyze", V, dedupe_key="v1:ollama", priority=0)  # old transcriber
    assert late is not None

    assert _resolve(conn) == StaleAnalyzeResult(1, 0)
    assert _row(conn, late)["dedupe_key"] == "v2:ollama"


@pytest.mark.parametrize("version", ["", "a:b", ":"])
def test_an_invalid_prompt_version_raises_before_writing(
    conn: psycopg.Connection[Any], version: str
) -> None:
    _video(conn)
    _job(conn, "v1:ollama")
    before = _table(conn)

    with pytest.raises(ValueError, match="prompt_version"):
        _resolve(conn, version)

    assert _table(conn) == before


def test_the_version_is_a_bound_value_not_sql(conn: psycopg.Connection[Any]) -> None:
    _video(conn)
    job = _job(conn, "v1:ollama")

    assert _resolve(conn, "v2'; DROP TABLE jobs; --") == StaleAnalyzeResult(1, 0)

    assert _row(conn, job)["dedupe_key"] == "v2'; DROP TABLE jobs; --:ollama"


def test_a_row_another_transaction_holds_is_skipped_and_picked_up_next_call(
    conn: psycopg.Connection[Any], head_dsn: str
) -> None:
    _video(conn)
    held = _job(conn, "v1:ollama")
    free = _job(conn, "v1:anthropic")
    conn.commit()

    with psycopg.connect(head_dsn) as claimer:
        claimer.execute("SELECT id FROM jobs WHERE id = %s FOR UPDATE", (held,))

        result = _resolve(conn)  # returns at once: SKIP LOCKED, no waiting

        assert result == StaleAnalyzeResult(1, 0)
        assert _row(conn, held)["dedupe_key"] == "v1:ollama"
        assert _row(conn, free)["dedupe_key"] == "v2:anthropic"
        claimer.rollback()

    assert _resolve(conn) == StaleAnalyzeResult(1, 0)
    assert _row(conn, held)["dedupe_key"] == "v2:ollama"


def test_the_planner_task_logs_one_info_line_with_the_counts_and_returns_their_sum(
    conn: psycopg.Connection[Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="services.planner.stale_analyze")
    _video(conn)
    kept = _job(conn, "v1:ollama")
    _job(conn, "v0:ollama")
    settings = Settings(DATABASE_URL=SecretStr("postgresql://u:p@h/db"), PROMPT_VERSION="v3")

    total = resolve_stale_analyze_jobs(_queue(conn), settings)

    lines = [r for r in caplog.records if r.getMessage() == "planner.stale_analyze_resolved"]
    assert [(r.levelno, r.prompt_version, r.rekeyed, r.superseded) for r in lines] == [  # type: ignore[attr-defined]
        (logging.INFO, "v3", 1, 1)
    ]
    assert _row(conn, kept)["dedupe_key"] == "v3:ollama"
    assert total == 2
    assert len(caplog.records) == 1  # nothing per row
