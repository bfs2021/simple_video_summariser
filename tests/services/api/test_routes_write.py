"""``POST /videos`` and ``POST /channels`` (issue #40).

Integration tests against the testcontainers Postgres: every case checks the
rows actually written. Races are forced deterministically: one connection
holds an uncommitted transaction, the other request is started in a thread,
and the test waits (bounded, no sleep) until Postgres reports it blocked.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from common.config import get_settings
from common.queue import PRIORITY_INTERACTIVE, PRIORITY_NORMAL, PostgresQueue
from common.repo.channels import register_channel
from services.api.main import create_app
from services.api.routes_write import submit_video
from tests.services.api.conftest import LogSink

pytestmark = pytest.mark.integration

VID = "dQw4w9WgXcQ"
DASH = "-wNyEUrxzFU"
CHANNEL = "UCuAXFkgsw1L7xaCfnd5JJOw"
MARKER = "ZZMARKER42"


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


def _jobs(db: psycopg.Connection[Any]) -> list[tuple[Any, ...]]:
    return db.execute(
        "SELECT id, video_id, kind, dedupe_key, priority, payload, state FROM jobs ORDER BY id"
    ).fetchall()


def _videos(db: psycopg.Connection[Any]) -> list[tuple[Any, ...]]:
    return db.execute(
        "SELECT video_id, channel_id, title, origin, discovered_at FROM videos ORDER BY video_id"
    ).fetchall()


def _counts(db: psycopg.Connection[Any]) -> tuple[int, int, int]:
    row = db.execute(
        "SELECT (SELECT count(*) FROM videos), (SELECT count(*) FROM jobs),"
        " (SELECT count(*) FROM channels)"
    ).fetchone()
    assert row is not None
    return (int(row[0]), int(row[1]), int(row[2]))


def _wait_until_blocked(db: psycopg.Connection[Any], app: str) -> None:
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


# --- POST /videos -------------------------------------------------------------


@pytest.mark.parametrize("ref", [VID, f"https://www.youtube.com/watch?v={VID}&t=5"])
def test_a_new_video_creates_a_bare_row_and_one_interactive_ingest_job(
    api: TestClient, db: psycopg.Connection[Any], ref: str
) -> None:
    response = api.post("/videos", json={"url": ref})

    assert response.status_code == 202
    [(job_id, video_id, kind, key, priority, payload, state)] = _jobs(db)
    assert response.json() == {
        "video_id": VID,
        "job_id": job_id,
        "state": "pending",
        "created": True,
    }
    assert (video_id, kind, key, priority, payload, state) == (
        VID,
        "ingest",
        "default",
        PRIORITY_INTERACTIVE,
        {},
        "pending",
    )
    [(row_id, channel_id, title, origin, _)] = _videos(db)
    assert (row_id, channel_id, title, origin) == (VID, None, None, "adhoc")
    assert _counts(db)[2] == 0


def test_a_video_id_starting_with_a_dash_is_accepted(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    response = api.post("/videos", json={"url": f"https://youtu.be/{DASH}"})

    assert response.status_code == 202
    assert response.json()["video_id"] == DASH


@pytest.mark.parametrize("state", ["pending", "running"])
def test_resubmitting_an_active_job_returns_it_unchanged(
    api: TestClient, db: psycopg.Connection[Any], state: str
) -> None:
    first = api.post("/videos", json={"url": VID}).json()
    db.execute("UPDATE jobs SET state = %s", (state,))

    again = api.post("/videos", json={"url": f"https://youtu.be/{VID}"})

    assert again.status_code == 200
    assert again.json() == {
        "video_id": VID,
        "job_id": first["job_id"],
        "state": state,
        "created": False,
    }
    assert len(_jobs(db)) == 1
    assert len(_videos(db)) == 1


def test_resubmitting_a_pending_lower_priority_ingest_raises_it_in_place(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    # Was "does not raise" under #40; #103 reverses it: a human is waiting now.
    db.execute("INSERT INTO videos (video_id, origin) VALUES (%s, 'rss')", (VID,))
    job_id = PostgresQueue(db).enqueue("ingest", VID, priority=PRIORITY_NORMAL)
    db.execute("UPDATE jobs SET attempts = 2")

    response = api.post("/videos", json={"url": VID})

    assert response.status_code == 200
    assert response.json() == {
        "video_id": VID,
        "job_id": job_id,
        "state": "pending",
        "created": False,
    }
    assert db.execute("SELECT id, priority, attempts FROM jobs").fetchall() == [
        (job_id, PRIORITY_INTERACTIVE, 2)
    ]


def test_resubmitting_a_running_ingest_leaves_its_priority(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    db.execute("INSERT INTO videos (video_id, origin) VALUES (%s, 'rss')", (VID,))
    job_id = PostgresQueue(db).enqueue("ingest", VID, priority=PRIORITY_NORMAL)
    db.execute("UPDATE jobs SET state = 'running'")

    response = api.post("/videos", json={"url": VID})

    assert response.json() == {
        "video_id": VID,
        "job_id": job_id,
        "state": "running",
        "created": False,
    }
    assert db.execute("SELECT priority FROM jobs").fetchall() == [(PRIORITY_NORMAL,)]


def test_resubmitting_after_a_done_ingest_raises_the_pending_transcribe(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    db.execute("INSERT INTO videos (video_id, origin) VALUES (%s, 'backfill')", (VID,))
    queue = PostgresQueue(db)
    ingest = queue.enqueue("ingest", VID, priority=-10)
    transcribe = queue.enqueue("transcribe", VID, priority=-10)
    db.execute("UPDATE jobs SET state = 'done' WHERE id = %s", (ingest,))

    response = api.post("/videos", json={"url": VID})

    assert response.status_code == 200
    assert response.json() == {
        "video_id": VID,
        "job_id": ingest,
        "state": "done",
        "created": False,
    }
    assert db.execute("SELECT id, state, priority FROM jobs ORDER BY id").fetchall() == [
        (ingest, "done", -10),
        (transcribe, "pending", PRIORITY_INTERACTIVE),
    ]


def test_resubmitting_after_a_dead_ingest_still_raises_other_pending_jobs(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    db.execute("INSERT INTO videos (video_id, origin) VALUES (%s, 'rss')", (VID,))
    queue = PostgresQueue(db)
    ingest = queue.enqueue("ingest", VID, priority=0)
    queue.enqueue("analyze", VID, dedupe_key="v1:fake", priority=-5)
    db.execute("UPDATE jobs SET state = 'dead' WHERE id = %s", (ingest,))

    api.post("/videos", json={"url": VID})

    assert db.execute("SELECT kind, priority FROM jobs ORDER BY id").fetchall() == [
        ("ingest", 0),
        ("analyze", PRIORITY_INTERACTIVE),
    ]


def test_resubmitting_when_everything_is_already_interactive_changes_no_row(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    api.post("/videos", json={"url": VID})
    PostgresQueue(db).enqueue("transcribe", VID, priority=PRIORITY_INTERACTIVE + 5)
    before = db.execute("SELECT * FROM jobs ORDER BY id").fetchall()

    response = api.post("/videos", json={"url": VID})

    assert response.status_code == 200
    assert db.execute("SELECT * FROM jobs ORDER BY id").fetchall() == before


def test_another_videos_pending_jobs_are_not_raised(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    db.execute("INSERT INTO videos (video_id, origin) VALUES (%s, 'rss')", (VID,))
    PostgresQueue(db).enqueue("ingest", VID, priority=0)
    PostgresQueue(db).enqueue("ingest", DASH, priority=-10)

    api.post("/videos", json={"url": VID})

    assert db.execute("SELECT video_id, priority FROM jobs ORDER BY id").fetchall() == [
        (VID, PRIORITY_INTERACTIVE),
        (DASH, -10),
    ]


def test_a_first_submission_leaves_the_single_job_at_interactive(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    assert api.post("/videos", json={"url": VID}).status_code == 202

    assert db.execute("SELECT priority FROM jobs").fetchall() == [(PRIORITY_INTERACTIVE,)]


def _spy_raise_priority(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[Any, ...]]) -> None:
    def spy(self: PostgresQueue, *args: Any) -> list[int]:
        calls.append(args)
        return []

    monkeypatch.setattr(PostgresQueue, "raise_priority", spy)


def test_an_invalid_submission_never_calls_raise_priority(
    api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, ...]] = []
    _spy_raise_priority(monkeypatch, calls)

    assert api.post("/videos", json={"url": "not a video"}).status_code == 422

    assert calls == []


def test_a_resubmission_raises_through_the_interactive_constant(
    api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.post("/videos", json={"url": VID})
    calls: list[tuple[Any, ...]] = []
    _spy_raise_priority(monkeypatch, calls)

    api.post("/videos", json={"url": VID})

    assert calls == [(VID, PRIORITY_INTERACTIVE)]


@pytest.mark.parametrize("state", ["done", "dead"])
def test_resubmitting_after_a_finished_job_enqueues_nothing(
    api: TestClient, db: psycopg.Connection[Any], state: str
) -> None:
    first = api.post("/videos", json={"url": VID}).json()
    db.execute("UPDATE jobs SET state = %s", (state,))

    again = api.post("/videos", json={"url": VID})

    assert again.status_code == 200
    assert again.json() == {
        "video_id": VID,
        "job_id": first["job_id"],
        "state": state,
        "created": False,
    }
    assert len(_jobs(db)) == 1


@pytest.mark.parametrize("origin", ["rss", "backfill"])
def test_an_existing_video_keeps_its_origin_and_discovered_at(
    api: TestClient, db: psycopg.Connection[Any], origin: str
) -> None:
    db.execute("INSERT INTO channels (channel_id, active) VALUES ('UCx', true)")
    discovered = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    db.execute(
        "INSERT INTO videos (video_id, channel_id, title, origin, discovered_at)"
        " VALUES (%s, 'UCx', 'T', %s, %s)",
        (VID, origin, discovered),
    )

    response = api.post("/videos", json={"url": VID})

    assert response.status_code == 202
    assert _videos(db) == [(VID, "UCx", "T", origin, discovered)]


def test_concurrent_submissions_while_the_first_is_uncommitted_share_one_job(
    dsn: str, db: psycopg.Connection[Any]
) -> None:
    with psycopg.connect(dsn) as first:
        with first.transaction():
            first_result = submit_video(first, VID, queue=PostgresQueue(first))
            result: dict[str, Any] = {}

            def second() -> None:
                with psycopg.connect(dsn, application_name="second-submit") as conn:
                    result["value"] = submit_video(conn, VID, queue=PostgresQueue(conn))
                    conn.commit()

            thread = threading.Thread(target=second)
            thread.start()
            _wait_until_blocked(db, "second-submit")
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert first_result.created is True
    assert result["value"].job_id == first_result.job_id
    assert result["value"].created is False
    assert len(_jobs(db)) == 1
    assert len(_videos(db)) == 1


def test_a_submission_that_loses_the_enqueue_race_returns_the_winners_job(
    dsn: str, db: psycopg.Connection[Any]
) -> None:
    db.execute("INSERT INTO videos (video_id, origin) VALUES (%s, 'rss')", (VID,))
    with psycopg.connect(dsn) as first:
        with first.transaction():
            winner = PostgresQueue(first).enqueue(
                "ingest", VID, priority=PRIORITY_INTERACTIVE
            )
            result: dict[str, Any] = {}

            def second() -> None:
                with psycopg.connect(dsn, application_name="loser-submit") as conn:
                    result["value"] = submit_video(conn, VID, queue=PostgresQueue(conn))
                    conn.commit()

            thread = threading.Thread(target=second)
            thread.start()
            _wait_until_blocked(db, "loser-submit")
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert result["value"].job_id == winner
    assert result["value"].created is False
    assert len(_jobs(db)) == 1


def test_two_concurrent_http_submissions_both_succeed_with_one_job(
    dsn: str, db: psycopg.Connection[Any], logs: LogSink
) -> None:
    app = create_app()
    barrier = threading.Barrier(2)
    responses: list[Any] = []

    def submit(ref: str) -> None:
        with TestClient(app) as client:
            barrier.wait(timeout=10)
            responses.append(client.post("/videos", json={"url": ref}))

    threads = [
        threading.Thread(target=submit, args=(ref,))
        for ref in (VID, f"https://www.youtube.com/watch?v={VID}")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads)
    assert len(responses) == 2

    [(job_id, *_)] = _jobs(db)
    assert sorted(r.status_code for r in responses) == [200, 202]
    assert {r.json()["job_id"] for r in responses} == {job_id}
    assert len(_videos(db)) == 1


_INVALID_BODIES: list[Any] = [
    {},
    {"url": None},
    {"url": 12345678901},
    {"url": {"v": VID}},
    {"url": [VID]},
    {"other": VID},
    [VID],
    VID,
    None,
]

_INVALID_REFS = [
    "",
    "   ",
    VID[:10],
    VID + "x",
    f"{VID[:10]};",
    f"{VID[:10]}%",
    f"{VID[:10]}.",
    f"{VID[:5]} {VID[6:]}",
    f"{VID[:10]}é",
    "https://www.youtube.com/watch",
    "https://www.youtube.com/watch?v=",
    f"https://www.youtube.com/watch?v={VID}&v=aaaaaaaaaaa",
    "https://www.youtube.com/playlist?list=PLabc123",
    f"https://www.youtube.com/channel/{CHANNEL}",
    f"https://youtube.com.evil.com/watch?v={VID}",
    f"https://evilyoutube.com/watch?v={VID}",
    f"https://youtu.be.evil.com/{VID}",
    f"https://youtube.com@evil.com/watch?v={VID}",
    f"https://music.youtube.com/watch?v={VID}",
    "javascript:alert(1)",
    f"ftp://www.youtube.com/watch?v={VID}",
    f"https://www.youtube.com/watch?v={VID}&x=" + "a" * 2048,
]


@pytest.mark.parametrize("body", _INVALID_BODIES)
def test_a_malformed_video_body_is_rejected_and_writes_nothing(
    api: TestClient, db: psycopg.Connection[Any], body: Any
) -> None:
    response = api.post("/videos", json=body)

    assert response.status_code == 422
    assert VID not in response.text
    assert _counts(db) == (0, 0, 0)


@pytest.mark.parametrize("ref", _INVALID_REFS)
def test_an_invalid_video_reference_is_rejected_and_writes_nothing(
    api: TestClient, db: psycopg.Connection[Any], ref: str
) -> None:
    response = api.post("/videos", json={"url": ref})

    assert response.status_code == 422
    assert response.json() == {"detail": "url must be a YouTube video ID or video URL"}
    assert _counts(db) == (0, 0, 0)


@pytest.mark.parametrize(
    "ref",
    [
        MARKER,
        f"https://evil.example/{MARKER}?v={VID}",
        f"https://www.youtube.com/watch?v={MARKER}",
        f"javascript:alert('{MARKER}')",
    ],
)
def test_a_rejected_video_reference_is_never_echoed(
    api: TestClient, db: psycopg.Connection[Any], ref: str
) -> None:
    response = api.post("/videos", json={"url": ref})

    assert response.status_code == 422
    assert MARKER not in response.text
    assert _counts(db) == (0, 0, 0)


# --- POST /channels -------------------------------------------------------------


def _channel_rows(db: psycopg.Connection[Any]) -> list[tuple[Any, ...]]:
    return db.execute(
        "SELECT channel_id, title, active, monitor_from, last_polled, last_poll_err, added_at"
        " FROM channels ORDER BY channel_id"
    ).fetchall()


@pytest.mark.parametrize(
    "ref", [CHANNEL, f"https://www.youtube.com/channel/{CHANNEL}/videos"]
)
def test_an_unknown_channel_is_registered_active_from_now(
    api: TestClient, db: psycopg.Connection[Any], ref: str
) -> None:
    before = db.execute("SELECT now()").fetchone()
    assert before is not None

    response = api.post("/channels", json={"url": ref})

    assert response.status_code == 201
    [(channel_id, title, active, monitor_from, last_polled, err, added_at)] = (
        _channel_rows(db)
    )
    assert (channel_id, title, active, last_polled, err) == (
        CHANNEL,
        None,
        True,
        None,
        None,
    )
    assert monitor_from >= before[0]
    body = response.json()
    assert body == {
        "channel_id": CHANNEL,
        "active": True,
        "monitor_from": body["monitor_from"],
        "added_at": body["added_at"],
        "created": True,
    }
    assert datetime.fromisoformat(body["monitor_from"]) == monitor_from
    assert datetime.fromisoformat(body["added_at"]) == added_at
    assert _jobs(db) == []


def test_an_inactive_channel_is_reactivated_from_now_without_backfill(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    old = datetime(2020, 1, 1, tzinfo=UTC)
    db.execute(
        "INSERT INTO channels (channel_id, title, active, monitor_from, added_at)"
        " VALUES (%s, 'Kept', false, %s, %s)",
        (CHANNEL, old, old),
    )
    db.execute(
        "INSERT INTO videos (video_id, channel_id, published_at, origin)"
        " VALUES (%s, %s, %s, 'adhoc')",
        (VID, CHANNEL, datetime(2025, 1, 1, tzinfo=UTC)),
    )
    before = db.execute("SELECT now()").fetchone()
    assert before is not None

    response = api.post("/channels", json={"url": CHANNEL})

    assert response.status_code == 200
    [(_, title, active, monitor_from, _, _, added_at)] = _channel_rows(db)
    assert (title, active, added_at) == ("Kept", True, old)
    assert monitor_from >= before[0]
    assert response.json()["created"] is False
    assert datetime.fromisoformat(response.json()["monitor_from"]) == monitor_from
    assert _jobs(db) == []


def test_an_active_channel_is_left_exactly_as_it_is(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    old = datetime(2020, 1, 1, tzinfo=UTC)
    polled = datetime(2021, 1, 1, tzinfo=UTC)
    db.execute(
        "INSERT INTO channels (channel_id, title, active, monitor_from, last_polled, added_at)"
        " VALUES (%s, 'T', true, %s, %s, %s)",
        (CHANNEL, old, polled, old),
    )
    before = _channel_rows(db)

    response = api.post("/channels", json={"url": f"youtube.com/channel/{CHANNEL}"})

    assert response.status_code == 200
    assert _channel_rows(db) == before
    assert response.json()["created"] is False
    assert datetime.fromisoformat(response.json()["monitor_from"]) == old
    assert _jobs(db) == []


def test_concurrent_registrations_of_a_new_channel_leave_one_row(
    dsn: str, db: psycopg.Connection[Any], logs: LogSink
) -> None:
    app = create_app()
    barrier = threading.Barrier(2)
    responses: list[Any] = []

    def register() -> None:
        with TestClient(app) as client:
            barrier.wait(timeout=10)
            responses.append(client.post("/channels", json={"url": CHANNEL}))

    threads = [threading.Thread(target=register) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads)
    assert len(responses) == 2

    assert sorted(r.status_code for r in responses) == [200, 201]
    assert len(_channel_rows(db)) == 1
    assert _jobs(db) == []


def test_a_registration_racing_an_uncommitted_insert_returns_the_committed_row(
    dsn: str, db: psycopg.Connection[Any]
) -> None:
    with psycopg.connect(dsn) as first:
        with first.transaction():
            winner = register_channel(first, CHANNEL)
            result: dict[str, Any] = {}

            def second() -> None:
                with psycopg.connect(dsn, application_name="second-register") as conn:
                    result["value"] = register_channel(conn, CHANNEL)
                    conn.commit()

            thread = threading.Thread(target=second)
            thread.start()
            _wait_until_blocked(db, "second-register")
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert winner.created is True
    assert result["value"].created is False
    assert result["value"].channel == winner.channel
    [(_, _, _, monitor_from, _, _, _)] = _channel_rows(db)
    assert monitor_from == winner.channel.monitor_from


@pytest.mark.parametrize(
    "ref",
    [
        "@somehandle",
        "https://www.youtube.com/@somehandle",
        "https://www.youtube.com/c/somename",
        "https://www.youtube.com/user/somename",
    ],
)
def test_handle_urls_are_rejected_with_a_channel_id_hint(
    api: TestClient, db: psycopg.Connection[Any], ref: str
) -> None:
    response = api.post("/channels", json={"url": ref + MARKER})

    assert response.status_code == 422
    assert "channel-ID" in response.json()["detail"]
    assert MARKER not in response.text
    assert _counts(db) == (0, 0, 0)


@pytest.mark.parametrize(
    "body",
    [
        *_INVALID_BODIES,
        {"url": ""},
        {"url": CHANNEL[:-1]},
        {"url": CHANNEL + "x"},
        {"url": f"https://youtube.com.evil.com/channel/{CHANNEL}"},
        {"url": f"https://youtube.com@evil.com/channel/{CHANNEL}"},
        {"url": f"https://www.youtube.com/watch?v={VID}"},
        {"url": "x" * 2049},
    ],
)
def test_invalid_channel_input_is_rejected_and_writes_nothing(
    api: TestClient, db: psycopg.Connection[Any], body: Any
) -> None:
    response = api.post("/channels", json=body)

    assert response.status_code == 422
    assert CHANNEL not in response.text
    assert _counts(db) == (0, 0, 0)
