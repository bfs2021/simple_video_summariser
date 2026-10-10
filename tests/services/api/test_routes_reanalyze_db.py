"""``POST /videos/{id}/reanalyze`` against Postgres (issue #101).

Every case checks the ``jobs`` rows actually written. The claim race is
forced through ``request_reanalysis``'s ``after_lookup`` seam, which runs
between the active-job lookup and the upgrade, so no test sleeps. The
concurrent-request test holds one transaction open and waits (bounded) until
Postgres reports the second one blocked.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from common.config import get_settings
from common.models import Segment
from common.queue import PRIORITY_INTERACTIVE, PostgresQueue, analyze_dedupe_key
from common.repo.transcripts import save_transcript
from common.repo.videos import record_unavailable
from services.analyzer import main as analyzer_main
from services.api.main import create_app
from services.api.routes_write import (
    Reanalysis,
    ReanalysisConflict,
    ReanalysisContended,
    request_reanalysis,
)
from services.planner.sweep import run_reanalysis_sweep
from tests.services.analyzer.fakes import FakeSummarizer
from tests.services.api.conftest import LogSink
from tests.services.transcriber.fakes import make_settings

pytestmark = pytest.mark.integration

VID = "dQw4w9WgXcQ"
DASH = "-wNyEUrxzFU"
SEGMENTS = (Segment(0, 10, "Alpha opening."), Segment(100, 110, "Beta middle."))
RUNNING_MESSAGE = "an analysis is already running for this video"


@pytest.fixture
def dsn(head_dsn: str, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("DATABASE_URL", head_dsn)
    get_settings.cache_clear()
    return head_dsn


@pytest.fixture
def db(dsn: str) -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(dsn, autocommit=True) as conn:
        yield conn


@pytest.fixture
def api(dsn: str, logs: LogSink) -> Iterator[TestClient]:
    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def key(dsn: str) -> str:
    settings = get_settings()
    return analyze_dedupe_key(settings.PROMPT_VERSION, settings.SUMMARIZER)


def seed(db: psycopg.Connection[Any], video_id: str = VID, *, transcript: bool = True) -> None:
    db.execute(
        "INSERT INTO videos (video_id, origin) VALUES (%s, 'adhoc') ON CONFLICT DO NOTHING",
        (video_id,),
    )
    if transcript:
        save_transcript(db, video_id, "whisper", "en", "none", SEGMENTS, None)


def add_job(
    db: psycopg.Connection[Any],
    key: str,
    state: str,
    *,
    priority: int = 0,
    payload: str = "{}",
    video_id: str = VID,
) -> int:
    row = db.execute(
        "INSERT INTO jobs (video_id, kind, dedupe_key, state, priority, payload)"
        " VALUES (%s, 'analyze', %s, %s, %s, %s::jsonb) RETURNING id",
        (video_id, key, state, priority, payload),
    ).fetchone()
    assert row is not None
    return int(row[0])


def jobs(db: psycopg.Connection[Any]) -> list[tuple[Any, ...]]:
    return db.execute(
        "SELECT id, video_id, kind, dedupe_key, state, priority, payload FROM jobs ORDER BY id"
    ).fetchall()


def job_count(db: psycopg.Connection[Any]) -> int:
    row = db.execute("SELECT count(*) FROM jobs").fetchone()
    assert row is not None
    return int(row[0])


def active(db: psycopg.Connection[Any], key: str) -> list[tuple[Any, ...]]:
    return db.execute(
        "SELECT id FROM jobs WHERE kind = 'analyze' AND dedupe_key = %s"
        " AND state IN ('pending', 'running')",
        (key,),
    ).fetchall()


def wait_until_blocked(db: psycopg.Connection[Any], app: str) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = db.execute(
            "SELECT count(*) FROM pg_stat_activity"
            " WHERE application_name = %s AND wait_event_type = 'Lock'",
            (app,),
        ).fetchone()
        if row is not None and row[0] > 0:
            return
    raise AssertionError(f"{app} never blocked on a lock")


# --- refusals ----------------------------------------------------------------


def test_an_unknown_video_is_404_with_a_fixed_message_and_writes_nothing(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    response = api.post(f"/videos/{VID}/reanalyze")

    assert response.status_code == 404
    assert response.json() == {"detail": "video not found"}
    assert VID not in response.text
    assert job_count(db) == 0


def test_a_video_without_a_transcript_is_409_and_writes_nothing(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    seed(db, transcript=False)

    response = api.post(f"/videos/{VID}/reanalyze")

    assert response.status_code == 409
    assert response.json() == {"detail": "no transcript yet"}
    assert job_count(db) == 0


def test_a_running_unforced_job_is_409_and_nothing_is_written(
    api: TestClient, db: psycopg.Connection[Any], key: str
) -> None:
    seed(db)
    job_id = add_job(db, key, "running", priority=-5)
    before = jobs(db)

    response = api.post(f"/videos/{VID}/reanalyze")

    assert response.status_code == 409
    assert response.json() == {"detail": RUNNING_MESSAGE, "job_id": job_id}
    assert jobs(db) == before


# --- accepted ----------------------------------------------------------------


def test_a_video_with_a_transcript_and_no_analysis_gets_one_forced_interactive_job(
    api: TestClient, db: psycopg.Connection[Any], key: str
) -> None:
    seed(db)

    response = api.post(f"/videos/{VID}/reanalyze")

    assert response.status_code == 202
    [(job_id, video_id, kind, dedupe_key, state, priority, payload)] = jobs(db)
    assert response.json() == {
        "video_id": VID,
        "job_id": job_id,
        "state": "pending",
        "created": True,
    }
    assert (video_id, kind, dedupe_key, state, priority, payload) == (
        VID,
        "analyze",
        key,
        "pending",
        PRIORITY_INTERACTIVE,
        {"force": True},
    )


def test_a_dash_leading_id_is_accepted(api: TestClient, db: psycopg.Connection[Any]) -> None:
    seed(db, DASH)

    response = api.post(f"/videos/{DASH}/reanalyze")

    assert response.status_code == 202
    assert response.json()["video_id"] == DASH


def test_an_unavailable_video_with_a_stored_transcript_is_accepted(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    seed(db)
    record_unavailable(db, VID, "private", origin="adhoc")

    assert api.post(f"/videos/{VID}/reanalyze").status_code == 202


@pytest.mark.parametrize("state", ["done", "dead"])
def test_a_finished_job_at_the_key_does_not_block_and_is_left_unchanged(
    api: TestClient, db: psycopg.Connection[Any], key: str, state: str
) -> None:
    seed(db)
    old = add_job(db, key, state, priority=-5)
    [before] = jobs(db)

    response = api.post(f"/videos/{VID}/reanalyze")

    assert response.status_code == 202
    new_id = response.json()["job_id"]
    assert new_id != old
    assert jobs(db)[0] == before
    assert len(jobs(db)) == 2


def test_a_request_body_is_ignored(
    api: TestClient, db: psycopg.Connection[Any], key: str
) -> None:
    seed(db)

    response = api.post(
        f"/videos/{VID}/reanalyze",
        json={"force": False, "priority": 99, "dedupe_key": "x:y", "payload": {"a": 1}},
    )

    assert response.status_code == 202
    [(_, _, kind, dedupe_key, _, priority, payload)] = jobs(db)
    assert (kind, dedupe_key, priority, payload) == (
        "analyze",
        key,
        PRIORITY_INTERACTIVE,
        {"force": True},
    )


# --- reuse of an active job at the key ------------------------------------------


def test_a_pending_forced_job_is_reused_unchanged(
    api: TestClient, db: psycopg.Connection[Any], key: str
) -> None:
    seed(db)
    job_id = add_job(db, key, "pending", priority=PRIORITY_INTERACTIVE, payload='{"force": true}')
    before = jobs(db)

    response = api.post(f"/videos/{VID}/reanalyze")

    assert response.status_code == 200
    assert response.json() == {
        "video_id": VID,
        "job_id": job_id,
        "state": "pending",
        "created": False,
    }
    assert jobs(db) == before


@pytest.mark.parametrize(("start", "expected"), [(-5, 10), (0, 10), (9, 10), (10, 10), (20, 20)])
def test_a_pending_unforced_job_is_upgraded_in_place_and_never_lowered(
    api: TestClient, db: psycopg.Connection[Any], key: str, start: int, expected: int
) -> None:
    seed(db)
    job_id = add_job(db, key, "pending", priority=start, payload='{"other": 3}')

    response = api.post(f"/videos/{VID}/reanalyze")

    assert response.status_code == 200
    assert response.json() == {
        "video_id": VID,
        "job_id": job_id,
        "state": "pending",
        "created": False,
    }
    assert jobs(db) == [
        (job_id, VID, "analyze", key, "pending", expected, {"other": 3, "force": True})
    ]


def test_a_running_forced_job_is_reused(
    api: TestClient, db: psycopg.Connection[Any], key: str
) -> None:
    seed(db)
    job_id = add_job(db, key, "running", payload='{"force": true}')
    before = jobs(db)

    response = api.post(f"/videos/{VID}/reanalyze")

    assert response.status_code == 200
    assert response.json() == {
        "video_id": VID,
        "job_id": job_id,
        "state": "running",
        "created": False,
    }
    assert jobs(db) == before


def test_a_job_at_another_key_does_not_count_as_active(
    api: TestClient, db: psycopg.Connection[Any], key: str
) -> None:
    seed(db)
    other = add_job(db, "v0:other", "pending")

    response = api.post(f"/videos/{VID}/reanalyze")

    assert response.status_code == 202
    assert response.json()["job_id"] != other
    assert len(jobs(db)) == 2


# --- the claim race --------------------------------------------------------------


def claim_on_another_connection(dsn: str, job_id: int, state: str) -> Any:
    def hook() -> None:
        with psycopg.connect(dsn, autocommit=True) as other:
            other.execute(
                "UPDATE jobs SET state = %s, locked_by = 'w' WHERE id = %s", (state, job_id)
            )

    return hook


def test_a_job_claimed_between_the_lookup_and_the_upgrade_is_a_409_not_a_500(
    dsn: str, db: psycopg.Connection[Any], key: str
) -> None:
    seed(db)
    job_id = add_job(db, key, "pending", priority=-5)
    calls: list[int] = []

    def hook() -> None:
        calls.append(1)
        claim_on_another_connection(dsn, job_id, "running")()

    with psycopg.connect(dsn) as conn, pytest.raises(ReanalysisConflict) as caught:
        request_reanalysis(
            conn, VID, queue=PostgresQueue(conn), dedupe_key=key, after_lookup=hook
        )

    assert caught.value.job_id == job_id
    assert calls == [1]
    assert jobs(db) == [(job_id, VID, "analyze", key, "running", -5, {})]


def test_a_job_finished_between_the_lookup_and_the_upgrade_gets_a_new_job(
    dsn: str, db: psycopg.Connection[Any], key: str
) -> None:
    seed(db)
    job_id = add_job(db, key, "pending", priority=-5)

    with psycopg.connect(dsn) as conn:
        result = request_reanalysis(
            conn,
            VID,
            queue=PostgresQueue(conn),
            dedupe_key=key,
            after_lookup=claim_on_another_connection(dsn, job_id, "done"),
        )

    assert result.created is True
    assert result.job_id != job_id
    assert [row[4] for row in jobs(db)] == ["done", "pending"]
    assert len(active(db, key)) == 1


def test_a_job_claimed_and_forced_in_between_is_reused(
    dsn: str, db: psycopg.Connection[Any], key: str
) -> None:
    seed(db)
    job_id = add_job(db, key, "pending", priority=-5)

    def hook() -> None:
        with psycopg.connect(dsn, autocommit=True) as other:
            other.execute(
                "UPDATE jobs SET state = 'running',"
                " payload = payload || '{\"force\": true}' WHERE id = %s",
                (job_id,),
            )

    with psycopg.connect(dsn) as conn:
        result = request_reanalysis(
            conn, VID, queue=PostgresQueue(conn), dedupe_key=key, after_lookup=hook
        )

    assert result == Reanalysis(job_id=job_id, state="running", created=False)
    assert len(jobs(db)) == 1


def test_the_after_lookup_seam_is_not_called_when_there_is_nothing_to_upgrade(
    dsn: str, db: psycopg.Connection[Any], key: str
) -> None:
    seed(db)

    def hook() -> None:
        raise AssertionError("no active job, so no upgrade to race")

    with psycopg.connect(dsn) as conn:
        result = request_reanalysis(
            conn, VID, queue=PostgresQueue(conn), dedupe_key=key, after_lookup=hook
        )

    assert result.created is True


class _VanishingQueue(PostgresQueue):
    """``enqueue`` reports a conflict although no active job is visible."""

    def enqueue(self, *args: Any, **kwargs: Any) -> int | None:
        return None


def test_a_conflict_that_never_shows_an_active_job_ends_in_contended_not_a_loop(
    dsn: str, db: psycopg.Connection[Any], key: str
) -> None:
    seed(db)

    with psycopg.connect(dsn) as conn, pytest.raises(ReanalysisContended):
        request_reanalysis(conn, VID, queue=_VanishingQueue(conn), dedupe_key=key)

    assert job_count(db) == 0


# --- concurrency -------------------------------------------------------------------


def test_two_concurrent_requests_share_one_active_job(
    dsn: str, db: psycopg.Connection[Any], key: str
) -> None:
    seed(db)
    with psycopg.connect(dsn) as first:
        with first.transaction():
            winner = request_reanalysis(
                first, VID, queue=PostgresQueue(first), dedupe_key=key
            )
            result: dict[str, Reanalysis] = {}

            def second() -> None:
                with psycopg.connect(dsn, application_name="second-reanalyze") as conn:
                    result["value"] = request_reanalysis(
                        conn, VID, queue=PostgresQueue(conn), dedupe_key=key
                    )

            thread = threading.Thread(target=second)
            thread.start()
            wait_until_blocked(db, "second-reanalyze")
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert winner.created is True
    assert result["value"] == Reanalysis(job_id=winner.job_id, state="pending", created=False)
    assert len(active(db, key)) == 1


def test_two_concurrent_http_requests_both_succeed_with_one_job(
    dsn: str, db: psycopg.Connection[Any], key: str, logs: LogSink
) -> None:
    seed(db)
    app = create_app()
    barrier = threading.Barrier(2)
    responses: list[Any] = []

    def post() -> None:
        with TestClient(app) as client:
            barrier.wait(timeout=10)
            responses.append(client.post(f"/videos/{VID}/reanalyze"))

    threads = [threading.Thread(target=post) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads)

    [(job_id, *_)] = active(db, key)
    assert sorted(r.status_code for r in responses) == [200, 202]
    assert {r.json()["job_id"] for r in responses} == {job_id}
    assert sorted(r.json()["created"] for r in responses) == [False, True]


# --- interplay ------------------------------------------------------------------------


def test_the_sweep_after_a_reanalyze_creates_no_second_active_job(
    api: TestClient, db: psycopg.Connection[Any], key: str
) -> None:
    seed(db)
    job_id = api.post(f"/videos/{VID}/reanalyze").json()["job_id"]
    settings = get_settings()

    enqueued = run_reanalysis_sweep(
        db,
        PostgresQueue(db),
        prompt_version=settings.PROMPT_VERSION,
        summarizer_name=settings.SUMMARIZER,
    )

    assert enqueued == 0
    assert [row[0] for row in jobs(db)] == [job_id]


def test_the_sweep_before_a_reanalyze_leaves_one_upgraded_job(
    api: TestClient, db: psycopg.Connection[Any], key: str
) -> None:
    seed(db)
    settings = get_settings()
    assert (
        run_reanalysis_sweep(
            db,
            PostgresQueue(db),
            prompt_version=settings.PROMPT_VERSION,
            summarizer_name=settings.SUMMARIZER,
        )
        == 1
    )

    response = api.post(f"/videos/{VID}/reanalyze")

    assert response.status_code == 200
    [(job_id, _, _, _, _, priority, payload)] = jobs(db)
    assert response.json()["job_id"] == job_id
    assert (priority, payload) == (PRIORITY_INTERACTIVE, {"force": True})


def test_a_reanalyze_job_processed_by_the_analyzer_adds_an_analysis_row(
    api: TestClient, db: psycopg.Connection[Any], key: str, dsn: str, tmp_path: Path
) -> None:
    seed(db)
    settings = get_settings()

    def run_analyzer() -> None:
        analyzer_main.main(
            settings=make_settings(
                PROMPT_VERSION=settings.PROMPT_VERSION, SUMMARIZER=settings.SUMMARIZER
            ),
            connect=lambda: psycopg.connect(dsn),
            summarizer_factory=lambda s: FakeSummarizer(
                name=settings.SUMMARIZER, model="fake-model"
            ),
            worker_options={
                "liveness_path": tmp_path / "heartbeat",
                "install_signal_handlers": False,
                "sleep": lambda _s: None,
            },
            max_iterations=2,
        )

    PostgresQueue(db).enqueue("analyze", VID, dedupe_key=key)
    run_analyzer()
    assert _analysis_count(db) == 1
    PostgresQueue(db).enqueue("analyze", VID, dedupe_key=key)
    run_analyzer()
    assert _analysis_count(db) == 1, "an unforced job at the same key skips as already analyzed"

    assert api.post(f"/videos/{VID}/reanalyze").status_code == 202
    run_analyzer()

    assert _analysis_count(db) == 2
    assert {row[4] for row in jobs(db)} == {"done"}


def _analysis_count(db: psycopg.Connection[Any]) -> int:
    row = db.execute("SELECT count(*) FROM analyses").fetchone()
    assert row is not None
    return int(row[0])
