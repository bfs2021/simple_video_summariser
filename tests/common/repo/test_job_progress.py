"""``jobs.progress``: the write, its clearing on every exit from ``running``, and the RTF
history the estimate reads (issue #245)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import psycopg
import pytest

from common.errors import Cancelled, Defer, PermanentSourceError, TransientNetworkError
from common.queue import PostgresQueue
from common.repo.jobs import list_jobs, write_progress
from common.repo.transcripts import rtf_history

pytestmark = pytest.mark.integration

_FIGURES: dict[str, Any] = {"percent": 40, "done_sec": 40.0, "duration_sec": 100.0}


@pytest.fixture
def ac(head_dsn: str) -> Iterator[psycopg.Connection[Any]]:
    """Autocommit, so raw setup SQL needs no commits and a second connection sees it."""
    with psycopg.connect(head_dsn, autocommit=True) as connection:
        yield connection


@pytest.fixture
def queue(ac: psycopg.Connection[Any]) -> PostgresQueue:
    return PostgresQueue(ac)


@pytest.fixture
def vid(ac: psycopg.Connection[Any]) -> str:
    ac.execute("INSERT INTO videos (video_id) VALUES ('progressvid')")
    return "progressvid"


def _claim_and_keep_running(
    ac: psycopg.Connection[Any], queue: PostgresQueue, vid: str, kind: str = "transcribe"
) -> int:
    """A job that is `running` under worker ``w1``, as a claim leaves it."""
    row = ac.execute(
        "INSERT INTO jobs (video_id, kind, state, locked_by, locked_at, started_at, "
        "heartbeat_at, attempts) VALUES (%s, %s, 'running', 'w1', now(), now(), now(), 1) "
        "RETURNING id",
        (vid, kind),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _progress(ac: psycopg.Connection[Any], job_id: int) -> Any:
    row = ac.execute("SELECT progress FROM jobs WHERE id=%s", (job_id,)).fetchone()
    assert row is not None
    return row[0]


def test_write_stores_the_figures_and_the_database_time(
    ac: psycopg.Connection[Any], queue: PostgresQueue, vid: str
) -> None:
    job_id = _claim_and_keep_running(ac, queue, vid)

    assert write_progress(ac, job_id, "w1", _FIGURES) is True

    stored = _progress(ac, job_id)
    assert {k: stored[k] for k in _FIGURES} == _FIGURES
    assert "updated_at" in stored


def test_write_leaves_heartbeat_at_alone(
    ac: psycopg.Connection[Any], queue: PostgresQueue, vid: str
) -> None:
    job_id = _claim_and_keep_running(ac, queue, vid)
    ac.execute("UPDATE jobs SET heartbeat_at = now() - interval '1 hour' WHERE id=%s", (job_id,))
    before = ac.execute("SELECT heartbeat_at FROM jobs WHERE id=%s", (job_id,)).fetchone()

    write_progress(ac, job_id, "w1", _FIGURES)

    after = ac.execute("SELECT heartbeat_at FROM jobs WHERE id=%s", (job_id,)).fetchone()
    assert after == before


def test_write_is_refused_for_another_worker_or_a_job_that_is_not_running(
    ac: psycopg.Connection[Any], queue: PostgresQueue, vid: str
) -> None:
    job_id = _claim_and_keep_running(ac, queue, vid)

    assert write_progress(ac, job_id, "someone-else", _FIGURES) is False
    ac.execute("UPDATE jobs SET state='pending', locked_by=NULL WHERE id=%s", (job_id,))
    assert write_progress(ac, job_id, "w1", _FIGURES) is False
    assert _progress(ac, job_id) is None


@pytest.mark.parametrize(
    "outcome",
    ["done", "dead", "retry", "defer", "cancel"],
)
def test_progress_is_cleared_when_a_job_leaves_running(
    ac: psycopg.Connection[Any], queue: PostgresQueue, vid: str, outcome: str
) -> None:
    job_id = queue.enqueue("transcribe", vid)
    assert job_id is not None
    errors: dict[str, BaseException | None] = {
        "done": None,
        "dead": PermanentSourceError("removed", "gone"),
        "retry": TransientNetworkError("blip"),
        "defer": Defer(_later(ac)),
        "cancel": Cancelled(),
    }
    raised = errors[outcome]
    try:
        with queue.claim(["transcribe"], worker="w1") as job:
            assert job is not None
            assert write_progress(ac, job.id, "w1", _FIGURES) is True
            if raised is not None:
                raise raised
    except Cancelled:
        pass

    assert _progress(ac, job_id) is None


def test_progress_is_cleared_when_the_reaper_takes_a_job_back(
    ac: psycopg.Connection[Any], queue: PostgresQueue, vid: str
) -> None:
    job_id = _claim_and_keep_running(ac, queue, vid)
    write_progress(ac, job_id, "w1", _FIGURES)
    ac.execute("UPDATE jobs SET heartbeat_at = now() - interval '1 day' WHERE id=%s", (job_id,))

    assert queue.reap_stale() == 1

    assert _progress(ac, job_id) is None


def test_a_claim_is_not_blocked_by_an_open_progress_update(
    head_dsn: str, ac: psycopg.Connection[Any], queue: PostgresQueue, vid: str
) -> None:
    running_id = _claim_and_keep_running(ac, queue, vid)
    ac.execute("INSERT INTO videos (video_id) VALUES ('progressvid2')")
    pending_id = queue.enqueue("transcribe", "progressvid2")
    assert pending_id is not None

    claimed: list[int] = []
    with psycopg.connect(head_dsn) as writer:
        # The progress UPDATE is executed and its transaction held open: the row lock stays.
        writer.execute(
            "UPDATE jobs SET progress = '{}'::jsonb WHERE id=%s AND state='running'", (running_id,)
        )

        def claim() -> None:
            with psycopg.connect(head_dsn, autocommit=True) as other:
                other.execute("SET lock_timeout = '3s'")
                with PostgresQueue(other).claim(["transcribe"], worker="w2") as job:
                    if job is not None:
                        claimed.append(job.id)

        thread = threading.Thread(target=claim)
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive(), "claim blocked behind the progress update"
        writer.rollback()

    assert claimed == [pending_id]


def test_list_jobs_returns_progress_or_none(
    ac: psycopg.Connection[Any], queue: PostgresQueue, vid: str
) -> None:
    job_id = _claim_and_keep_running(ac, queue, vid)
    assert [r.progress for r in list_jobs(ac, limit=5)] == [None]

    write_progress(ac, job_id, "w1", _FIGURES)

    (row,) = list_jobs(ac, limit=5)
    assert row.progress is not None and row.progress["percent"] == 40


def test_list_jobs_exposes_the_payload_duration_only_as_a_number(
    ac: psycopg.Connection[Any], queue: PostgresQueue, vid: str
) -> None:
    queue.enqueue("transcribe", vid, payload={"duration_sec": 90})
    ac.execute("INSERT INTO videos (video_id) VALUES ('progressvid2')")
    queue.enqueue("transcribe", "progressvid2", payload={"duration_sec": "soon"})

    durations = {r.video_id: r.payload_duration_sec for r in list_jobs(ac, limit=5)}

    assert durations == {"progressvid": 90.0, "progressvid2": None}


def _add_transcript(
    ac: psycopg.Connection[Any], n: int, engine_meta: Any, source: str = "whisper"
) -> None:
    vid = f"rtfvid{n:05d}"
    ac.execute("INSERT INTO videos (video_id) VALUES (%s)", (vid,))
    ac.execute(
        "INSERT INTO transcripts (video_id, source, segments, full_text, engine_meta, created_at)"
        " VALUES (%s, %s, '[]', '', %s::jsonb, now() - (%s * interval '1 minute'))",
        (vid, source, json.dumps(engine_meta), n),
    )


def _meta(rtf: Any, model: str = "base", compute: str = "int8") -> dict[str, Any]:
    return {"model": model, "compute_type": compute, "rtf": rtf}


def test_rtf_history_filters_by_model_and_compute_type_and_usable_rtf(
    ac: psycopg.Connection[Any],
) -> None:
    _add_transcript(ac, 1, _meta(0.5))
    _add_transcript(ac, 2, _meta(0.7, model="large-v3"))
    _add_transcript(ac, 3, _meta(0.9, compute="float16"))
    _add_transcript(ac, 4, _meta(0))
    _add_transcript(ac, 5, _meta(-1))
    _add_transcript(ac, 6, _meta(None))
    _add_transcript(ac, 7, _meta("fast"))
    _add_transcript(ac, 8, _meta(0.6), source="youtube_manual")

    assert rtf_history(ac, "base", "int8", 10) == [0.5]


def test_rtf_history_takes_the_newest_n(ac: psycopg.Connection[Any]) -> None:
    for n, rtf in enumerate([0.1, 0.2, 0.3], start=1):
        _add_transcript(ac, n, _meta(rtf))

    assert rtf_history(ac, "base", "int8", 2) == [0.1, 0.2]
    assert rtf_history(ac, "other", "int8", 2) == []


def _later(ac: psycopg.Connection[Any]) -> Any:
    row = ac.execute("SELECT now()").fetchone()
    assert row is not None
    return row[0] + timedelta(minutes=5)
