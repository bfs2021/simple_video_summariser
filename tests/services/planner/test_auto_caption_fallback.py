"""Auto-caption fallback sweep (issue #92). Integration tests need Postgres."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import psycopg
import pytest
from pydantic import SecretStr

from common.config import Settings
from common.errors import PermanentSourceError, ToolFailureError, UnavailableReason
from common.queue import PostgresQueue
from services.planner.fallback import run_auto_caption_fallback_sweep
from services.transcriber.ingest import make_ingest_handler
from tests.services.transcriber.fakes import (
    AUTO_ID,
    NO_CAPTIONS_ID,
    FakeMetadata,
    FakeSubtitles,
    auto_segments,
    fixture_meta,
    make_ctx,
    make_job,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _settings(**values: Any) -> Settings:
    return Settings(DATABASE_URL=SecretStr("postgresql://u:p@h/db"), **values)


def _queue(conn: psycopg.Connection[Any], settings: Settings | None = None) -> PostgresQueue:
    return PostgresQueue(conn, settings=settings or _settings())


def _sweep(
    conn: psycopg.Connection[Any], settings: Settings | None = None, **kwargs: Any
) -> int:
    settings = settings or _settings()
    return run_auto_caption_fallback_sweep(
        conn, _queue(conn, settings), settings=settings, **kwargs
    )


def _video(
    conn: psycopg.Connection[Any],
    video_id: str,
    *,
    origin: str = "rss",
    unavailable: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO videos (video_id, origin, unavailable) VALUES (%s, %s, %s)",
        (video_id, origin, unavailable),
    )


def _job(
    conn: psycopg.Connection[Any],
    video_id: str,
    kind: str = "transcribe",
    *,
    state: str = "dead",
    error_class: str | None = "TOOL_FAILURE",
    priority: int = 0,
    created: datetime = BASE,
    finished: datetime | None = None,
    dedupe_key: str = "default",
) -> int:
    if finished is None and state in ("dead", "done"):
        finished = created + timedelta(hours=1)
    row = conn.execute(
        "INSERT INTO jobs (video_id, kind, dedupe_key, state, error_class, priority, "
        "created_at, finished_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (video_id, kind, dedupe_key, state, error_class, priority, created, finished),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _transcript(
    conn: psycopg.Connection[Any], video_id: str, source: str = "whisper", text: str = "hi"
) -> None:
    conn.execute(
        "INSERT INTO transcripts (video_id, source, segments, full_text) "
        "VALUES (%s, %s, '[]', %s)",
        (video_id, source, text),
    )


def _dead(conn: psycopg.Connection[Any], video_id: str, **video: Any) -> None:
    """A video whose only transcribe job is dead: the plain candidate."""
    _video(conn, video_id, **video)
    _job(conn, video_id)


def _ingests(conn: psycopg.Connection[Any]) -> list[tuple[Any, ...]]:
    return conn.execute(
        "SELECT video_id, dedupe_key, state, priority, payload FROM jobs "
        "WHERE kind = 'ingest' ORDER BY id"
    ).fetchall()


# --- unit: no database -------------------------------------------------------------


class _Untouchable:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"touched {name}")


@pytest.mark.parametrize("limit", [0, -1])
def test_a_non_positive_limit_raises_before_touching_anything(limit: int) -> None:
    with pytest.raises(ValueError, match="limit"):
        run_auto_caption_fallback_sweep(
            cast(Any, _Untouchable()), cast(Any, _Untouchable()), settings=_settings(), limit=limit
        )


def test_flag_off_selects_and_enqueues_nothing_without_touching_the_database() -> None:
    result = run_auto_caption_fallback_sweep(
        cast(Any, _Untouchable()),
        cast(Any, _Untouchable()),
        settings=_settings(AUTO_CAPTION_FALLBACK=False),
    )

    assert result == 0


# --- eligibility -------------------------------------------------------------------

@pytest.mark.integration
def test_a_dead_transcribe_gets_one_ingest_with_its_priority_origin_and_trigger(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "a", origin="backfill")
    _job(conn, "a", priority=-10)

    assert _sweep(conn) == 1

    assert _ingests(conn) == [
        ("a", "default", "pending", -10, {"origin": "backfill", "trigger": "transcribe_dead"})
    ]


@pytest.mark.integration
@pytest.mark.parametrize("origin", ["adhoc", "rss", "backfill"])
def test_every_video_origin_is_accepted_by_the_ingest_validator(
    conn: psycopg.Connection[Any], origin: str
) -> None:
    _dead(conn, AUTO_ID, origin=origin)
    _sweep(conn)
    ((_, _, _, _, payload),) = _ingests(conn)

    handler = make_ingest_handler(
        conn=conn,
        queue=_queue(conn),
        metadata=FakeMetadata(PermanentSourceError(UnavailableReason.REMOVED, "x")),
        subtitles=FakeSubtitles(),
        settings=_settings(),
    )
    ctx, _ = make_ctx()
    with pytest.raises(PermanentSourceError):  # got past _validate: reached the adapter
        handler(make_job(AUTO_ID, payload=payload), ctx)


@pytest.mark.integration
def test_permanent_source_is_not_a_candidate(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "a")
    _job(conn, "a", error_class="PERMANENT_SOURCE")

    assert _sweep(conn) == 0
    assert _ingests(conn) == []


@pytest.mark.integration
@pytest.mark.parametrize("error_class", [None, "TOOL_FAILURE", "RESOURCE", "TRANSIENT_NETWORK"])
def test_null_and_other_error_classes_are_candidates(
    conn: psycopg.Connection[Any], error_class: str | None
) -> None:
    _video(conn, "a")
    _job(conn, "a", error_class=error_class)

    assert _sweep(conn) == 1


@pytest.mark.integration
def test_an_unavailable_video_is_not_a_candidate(conn: psycopg.Connection[Any]) -> None:
    _dead(conn, "a", unavailable="REMOVED")

    assert _sweep(conn) == 0


@pytest.mark.integration
@pytest.mark.parametrize("source", ["whisper", "youtube_manual", "youtube_auto"])
def test_a_non_blank_transcript_of_any_source_excludes_the_video(
    conn: psycopg.Connection[Any], source: str
) -> None:
    _dead(conn, "a")
    _transcript(conn, "a", source)

    assert _sweep(conn) == 0


@pytest.mark.integration
@pytest.mark.parametrize("text", ["", "  \n\t "])
def test_a_blank_transcript_does_not_exclude_the_video(
    conn: psycopg.Connection[Any], text: str
) -> None:
    _dead(conn, "a")
    _transcript(conn, "a", "youtube_auto", text)

    assert _sweep(conn) == 1


@pytest.mark.integration
def test_an_ingest_created_after_the_dead_job_finished_blocks_in_any_state(
    conn: psycopg.Connection[Any],
) -> None:
    for i, state in enumerate(["pending", "running", "done", "dead"]):
        vid = f"v{i}"
        _video(conn, vid)
        _job(conn, vid, created=BASE, finished=BASE + timedelta(hours=1))
        _job(conn, vid, "ingest", state=state, created=BASE + timedelta(hours=2))

    assert _sweep(conn) == 0


@pytest.mark.integration
def test_an_ingest_created_at_the_very_instant_the_dead_job_finished_is_not_after_it(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "a")
    _job(conn, "a", created=BASE, finished=BASE + timedelta(hours=1))
    _job(conn, "a", "ingest", state="done", created=BASE + timedelta(hours=1))

    assert _sweep(conn) == 1


@pytest.mark.integration
def test_a_dead_job_without_finished_at_is_guarded_by_its_created_at(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "a")
    conn.execute(
        "INSERT INTO jobs (video_id, kind, state, error_class, created_at, finished_at) "
        "VALUES ('a', 'transcribe', 'dead', NULL, %s, NULL)",
        (BASE,),
    )
    _job(conn, "a", "ingest", state="done", created=BASE + timedelta(hours=1))

    assert _sweep(conn) == 0  # the later ingest counts as the fallback attempt


@pytest.mark.integration
def test_an_ingest_created_before_the_dead_job_finished_does_not_block(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "a")
    _job(conn, "a", "ingest", state="done", created=BASE)  # the original ingest
    _job(conn, "a", created=BASE + timedelta(minutes=1), finished=BASE + timedelta(hours=1))

    assert _sweep(conn) == 1


@pytest.mark.integration
def test_the_newest_transcribe_decides_not_an_older_dead_one(
    conn: psycopg.Connection[Any],
) -> None:
    for i, state in enumerate(["pending", "running", "done"]):
        vid = f"v{i}"
        _video(conn, vid)
        _job(conn, vid, created=BASE)  # old dead
        _job(conn, vid, state=state, created=BASE + timedelta(days=1))

    assert _sweep(conn) == 0


@pytest.mark.integration
def test_a_tie_on_created_at_is_broken_by_id_like_latest_job_state(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "a")
    _job(conn, "a", state="pending", error_class=None)  # lower id, same created_at
    _job(conn, "a")  # higher id: newest, dead
    _video(conn, "b")
    _job(conn, "b")  # lower id, dead
    _job(conn, "b", state="pending", error_class=None)  # higher id: newest

    assert _sweep(conn) == 1
    assert [row[0] for row in _ingests(conn)] == ["a"]


@pytest.mark.integration
def test_a_tie_among_many_jobs_is_broken_by_the_highest_id(
    conn: psycopg.Connection[Any],
) -> None:
    # Many rows share one created_at, so only the id decides which is newest.
    # Sorting on created_at alone would pick an arbitrary one of them.
    _video(conn, "dead_last")
    for _ in range(30):
        _job(conn, "dead_last", state="done", error_class=None)
    _job(conn, "dead_last")  # highest id: newest, dead
    _video(conn, "done_last")
    _job(conn, "done_last")  # lowest id, dead
    for _ in range(30):
        _job(conn, "done_last", state="done", error_class=None)  # newest is done

    assert _sweep(conn) == 1
    assert [row[0] for row in _ingests(conn)] == ["dead_last"]


@pytest.mark.integration
def test_a_dead_newest_with_an_older_done_transcribe_and_transcript_is_not_a_candidate(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "a")
    _job(conn, "a", state="done", error_class=None, created=BASE)
    _transcript(conn, "a")
    _job(conn, "a", created=BASE + timedelta(days=1))

    assert _sweep(conn) == 0


@pytest.mark.integration
def test_a_video_with_no_transcribe_job_is_not_a_candidate(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "a")
    _job(conn, "a", "analyze", state="dead")

    assert _sweep(conn) == 0


@pytest.mark.integration
def test_flag_off_enqueues_nothing_and_flipping_it_on_picks_the_video_up(
    conn: psycopg.Connection[Any],
) -> None:
    _dead(conn, "a")

    assert _sweep(conn, _settings(AUTO_CAPTION_FALLBACK=False)) == 0
    assert _ingests(conn) == []
    assert _sweep(conn, _settings(AUTO_CAPTION_FALLBACK=True)) == 1


# --- both dead paths, through the real queue ---------------------------------------


def _claim_and(
    conn: psycopg.Connection[Any], settings: Settings, video_id: str, how: str
) -> None:
    """Dead-letter a real transcribe job the way production does."""
    queue = _queue(conn, settings)
    queue.enqueue("transcribe", video_id)
    # One attempt short of the cap, so the claim below makes the job's last attempt.
    conn.execute(
        "UPDATE jobs SET attempts = %s WHERE video_id = %s",
        (settings.MAX_ATTEMPTS_TRANSCRIBE - 1, video_id),
    )
    conn.commit()
    with queue.claim(["transcribe"], worker="w") as job:
        assert job is not None
        assert job.attempts == settings.MAX_ATTEMPTS_TRANSCRIBE
        if how == "non_retrying_class":
            queue.fail_job(job, "w", PermanentSourceError(UnavailableReason.REMOVED, "gone"))
        elif how == "attempts_max":
            queue.fail_job(job, "w", ToolFailureError("boom"))
        elif how == "reaped":
            conn.execute(
                "UPDATE jobs SET heartbeat_at = now() - interval '1 day' WHERE id = %s",
                (job.id,),
            )
            assert queue.reap_stale() == 1
    conn.commit()


@pytest.mark.integration
@pytest.mark.parametrize("how", ["attempts_max", "reaped"])
def test_dead_letters_from_fail_at_max_attempts_and_from_the_reaper_are_both_picked_up(
    conn: psycopg.Connection[Any], how: str
) -> None:
    settings = _settings()
    _video(conn, "a")
    _claim_and(conn, settings, "a", how)
    row = conn.execute("SELECT state, error_class FROM jobs WHERE video_id = 'a'").fetchone()
    assert row is not None and row[0] == "dead"
    if how == "reaped":
        assert row[1] is None  # #114: the NULL class is eligible
    assert _sweep(conn, settings) == 1


@pytest.mark.integration
def test_a_permanent_source_dead_letter_through_the_queue_is_not_picked_up(
    conn: psycopg.Connection[Any],
) -> None:
    settings = _settings()
    _video(conn, "a")
    _claim_and(conn, settings, "a", "non_retrying_class")

    assert _sweep(conn, settings) == 0


# --- idempotency and loop guards -----------------------------------------------------


@pytest.mark.integration
def test_a_second_pass_enqueues_nothing(conn: psycopg.Connection[Any]) -> None:
    _dead(conn, "a")

    assert _sweep(conn) == 1
    assert _sweep(conn) == 0
    assert len(_ingests(conn)) == 1


@pytest.mark.integration
def test_a_finished_or_dead_fallback_ingest_is_not_repeated(
    conn: psycopg.Connection[Any],
) -> None:
    _dead(conn, "a")
    _sweep(conn)
    conn.execute("UPDATE jobs SET state = 'dead', finished_at = now() WHERE kind = 'ingest'")

    assert _sweep(conn) == 0
    assert len(_ingests(conn)) == 1


@pytest.mark.integration
def test_a_retried_then_dead_again_transcribe_earns_a_new_fallback(
    conn: psycopg.Connection[Any],
) -> None:
    _dead(conn, "a")
    _sweep(conn)
    conn.execute("UPDATE jobs SET state = 'done', finished_at = now() WHERE kind = 'ingest'")
    _job(conn, "a", created=datetime.now(UTC) + timedelta(hours=1))  # operator retry dies again

    assert _sweep(conn) == 1
    assert len(_ingests(conn)) == 2


# --- limit, order, logging, errors ---------------------------------------------------


@pytest.mark.integration
def test_limit_caps_the_pass_and_the_oldest_finished_go_first(
    conn: psycopg.Connection[Any],
) -> None:
    for vid, hours in [("c", 3), ("a", 5), ("b", 5), ("d", 1)]:
        _video(conn, vid)
        _job(conn, vid, created=BASE, finished=BASE + timedelta(hours=hours))

    assert _sweep(conn, limit=3) == 3
    assert [row[0] for row in _ingests(conn)] == ["d", "c", "a"]
    assert _sweep(conn, limit=3) == 1
    assert [row[0] for row in _ingests(conn)] == ["d", "c", "a", "b"]


@pytest.mark.integration
def test_ties_on_finished_at_are_broken_by_video_id_before_the_limit_cuts(
    conn: psycopg.Connection[Any],
) -> None:
    ids = [f"v{i:02d}" for i in range(30)]
    for vid in reversed(ids):  # insert in the opposite order to the expected one
        _video(conn, vid)
        _job(conn, vid, finished=BASE + timedelta(hours=1))

    assert _sweep(conn, limit=5) == 5
    assert sorted(row[0] for row in _ingests(conn)) == ids[:5]


@pytest.mark.integration
def test_a_pass_logs_one_line_at_info_when_it_enqueued_and_debug_when_idle(
    conn: psycopg.Connection[Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="services.planner.fallback")
    _sweep(conn, limit=7)
    _dead(conn, "a")
    _sweep(conn, limit=7)

    records = [r for r in caplog.records if r.getMessage() == "planner.auto_caption_fallback_sweep"]
    assert [(r.levelno, r.enqueued, r.limit) for r in records] == [  # type: ignore[attr-defined]
        (logging.DEBUG, 0, 7),
        (logging.INFO, 1, 7),
    ]


@pytest.mark.integration
def test_the_enqueued_ingest_is_committed_and_visible_to_another_connection(
    conn: psycopg.Connection[Any], head_dsn: str
) -> None:
    # The pass runs on a connection the planner closes without committing, so the
    # job must already be durable when it returns (the read must not leave a
    # transaction open for enqueue() to nest inside).
    _dead(conn, "a")
    conn.commit()
    assert _sweep(conn) == 1

    with psycopg.connect(head_dsn) as other:
        assert other.execute("SELECT count(*) FROM jobs WHERE kind = 'ingest'").fetchone() == (1,)


class _DedupingQueue:
    """Stands in for a queue whose enqueue lost a race to an active job."""

    def __init__(self) -> None:
        self.calls = 0

    def enqueue(self, *args: Any, **kwargs: Any) -> int | None:
        self.calls += 1
        return None


@pytest.mark.integration
def test_a_deduplicated_enqueue_is_not_counted(conn: psycopg.Connection[Any]) -> None:
    _dead(conn, "a")
    queue = _DedupingQueue()

    result = run_auto_caption_fallback_sweep(conn, cast(Any, queue), settings=_settings())

    assert (result, queue.calls) == (0, 1)


@pytest.mark.integration
def test_a_database_error_propagates_unchanged(conn: psycopg.Connection[Any]) -> None:
    conn.close()

    with pytest.raises(psycopg.OperationalError):
        _sweep(conn)


# --- end to end ------------------------------------------------------------------------


def _run_ingest(conn: psycopg.Connection[Any], video_id: str, meta_name: str, **subs: Any) -> None:
    settings = _settings()
    handler = make_ingest_handler(
        conn=conn,
        queue=_queue(conn, settings),
        metadata=FakeMetadata(fixture_meta(meta_name, video_id)),
        subtitles=FakeSubtitles(**subs),
        settings=settings,
    )
    ctx, _ = make_ctx()
    row = conn.execute(
        "SELECT id, payload, priority FROM jobs WHERE video_id = %s AND kind = 'ingest' "
        "AND state = 'pending'",
        (video_id,),
    ).fetchone()
    assert row is not None
    handler(make_job(video_id, payload=row[1], priority=row[2]), ctx)
    conn.execute("UPDATE jobs SET state = 'done', finished_at = now() WHERE id = %s", (row[0],))
    conn.commit()


@pytest.mark.integration
def test_end_to_end_the_fallback_ingest_saves_auto_captions_and_queues_analyze(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, AUTO_ID)
    _job(conn, AUTO_ID, "ingest", state="done", error_class=None, created=BASE)
    _job(conn, AUTO_ID, created=BASE + timedelta(minutes=1))

    assert _sweep(conn) == 1
    _run_ingest(conn, AUTO_ID, "auto_only", auto=auto_segments())

    assert conn.execute(
        "SELECT source FROM transcripts WHERE video_id = %s", (AUTO_ID,)
    ).fetchall() == [("youtube_auto",)]
    assert conn.execute(
        "SELECT kind, state FROM jobs WHERE video_id = %s AND kind = 'analyze'", (AUTO_ID,)
    ).fetchall() == [("analyze", "pending")]
    assert _sweep(conn) == 0


@pytest.mark.integration
def test_end_to_end_with_no_auto_captions_leaves_no_transcript_and_no_loop(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, NO_CAPTIONS_ID)
    _job(conn, NO_CAPTIONS_ID, "ingest", state="done", error_class=None, created=BASE)
    _job(conn, NO_CAPTIONS_ID, created=BASE + timedelta(minutes=1))

    assert _sweep(conn) == 1
    _run_ingest(conn, NO_CAPTIONS_ID, "no_captions")

    assert conn.execute("SELECT count(*) FROM transcripts").fetchone() == (0,)
    assert conn.execute(
        "SELECT count(*) FROM jobs WHERE kind = 'transcribe' AND state <> 'dead'"
    ).fetchone() == (0,)
    assert conn.execute("SELECT count(*) FROM jobs WHERE kind = 'analyze'").fetchone() == (0,)
    assert _sweep(conn) == 0
