"""Backfill cancel end to end on Postgres (issue #116).

Real ``get_conn`` and real SQL; only the catalog is faked. Races are forced
with held transactions and ``pg_stat_activity``, never ``sleep``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb

from common.config import get_settings
from common.models import CatalogEntry, ChannelCatalog
from common.queue import PRIORITY_BACKFILL, PRIORITY_INTERACTIVE, PostgresQueue
from common.repo.videos import delete_unused_backfill_video
from services.api.backfill import ChannelNotRegistered, backfill, cancel_backfill
from services.api.main import create_app
from tests.services.api.conftest import LogSink

pytestmark = pytest.mark.integration

CHANNEL = "UCuAXFkgsw1L7xaCfnd5JJOw"
OTHER = "UCxxxxxxxxxxxxxxxxxxxxxx"
BID = "0f8fad5b-d9cb-469f-a165-70867728950e"
OTHER_BID = "1f8fad5b-d9cb-469f-a165-70867728950e"
ENTRIES = tuple(CatalogEntry(f"rt{i:09d}", f"secret title {i}", 60) for i in range(3))


class Catalog:
    def __init__(self, entries: tuple[CatalogEntry, ...] = ENTRIES) -> None:
        self.entries = entries

    def list_uploads(self, channel_id: str, *, limit: int) -> ChannelCatalog:
        return ChannelCatalog(channel_id, self.entries[:limit], len(self.entries))


@pytest.fixture
def db(head_dsn: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[psycopg.Connection[Any]]:
    monkeypatch.setenv("DATABASE_URL", head_dsn)
    get_settings.cache_clear()
    with psycopg.connect(head_dsn, autocommit=True) as conn:
        conn.execute("INSERT INTO channels (channel_id) VALUES (%s), (%s)", (CHANNEL, OTHER))
        yield conn


@pytest.fixture
def api(db: psycopg.Connection[Any], logs: LogSink) -> Iterator[TestClient]:
    with TestClient(create_app()) as client:
        yield client


def _seed(
    db: psycopg.Connection[Any],
    video_id: str,
    *,
    channel: str = CHANNEL,
    origin: str = "backfill",
    kind: str = "ingest",
    state: str = "pending",
    priority: int = PRIORITY_BACKFILL,
    backfill_id: str | None = BID,
    attempts: int = 0,
) -> int:
    db.execute(
        "INSERT INTO videos (video_id, channel_id, origin) VALUES (%s, %s, %s)"
        " ON CONFLICT DO NOTHING",
        (video_id, channel, origin),
    )
    payload: dict[str, Any] = {"origin": "backfill"}
    if backfill_id is not None:
        payload["backfill_id"] = backfill_id
    row = db.execute(
        """
        INSERT INTO jobs (video_id, kind, state, priority, payload, attempts, run_after,
                          locked_by, locked_at, heartbeat_at, finished_at)
        VALUES (%(v)s, %(k)s, %(s)s, %(p)s, %(pl)s, %(a)s,
                CASE WHEN %(a)s > 0 THEN now() + interval '1 hour' ELSE now() END,
                CASE WHEN %(s)s = 'running' THEN 'w1' END,
                CASE WHEN %(s)s = 'running' THEN now() END,
                CASE WHEN %(s)s = 'running' THEN now() END,
                CASE WHEN %(s)s IN ('done', 'dead') THEN now() END)
        RETURNING id
        """,
        {"v": video_id, "k": kind, "s": state, "p": priority, "pl": Jsonb(payload), "a": attempts},
    ).fetchone()
    assert row is not None
    return int(row[0])


def _jobs(db: psycopg.Connection[Any]) -> list[tuple[Any, ...]]:
    return db.execute("SELECT * FROM jobs ORDER BY id").fetchall()


def _videos(db: psycopg.Connection[Any]) -> set[str]:
    return {r[0] for r in db.execute("SELECT video_id FROM videos").fetchall()}


def _cancel(api: TestClient, channel: str = CHANNEL, backfill_id: str = BID) -> dict[str, Any]:
    response = api.post(f"/channels/{channel}/backfill/cancel", json={"backfill_id": backfill_id})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


# --- selection --------------------------------------------------------------------


def test_only_qualifying_jobs_are_deleted_and_every_other_row_is_identical(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    qualifying = [
        _seed(db, "q0000000001"),
        _seed(db, "q0000000002", attempts=2),  # pending in backoff still qualifies
    ]
    untouched = [
        _seed(db, "r0000000001", state="running"),
        _seed(db, "r0000000002", state="done"),
        _seed(db, "r0000000003", state="dead"),
        _seed(db, "q0000000001", kind="transcribe"),
        _seed(db, "q0000000001", kind="analyze", state="running"),
        _seed(db, "s0000000001", priority=0, backfill_id=None),  # RSS
        _seed(db, "s0000000002", priority=PRIORITY_INTERACTIVE, backfill_id=None),  # POST /videos
        _seed(db, "s0000000003", backfill_id=OTHER_BID),  # another backfill, same channel
        _seed(db, "s0000000004", priority=PRIORITY_INTERACTIVE),  # raised by #103, same id
        _seed(db, "t0000000001", channel=OTHER),  # same backfill_id, another channel
        _seed(db, "u0000000001", backfill_id=None),  # backfill priority, no tag
    ]
    before = {row[0]: row for row in _jobs(db)}

    body = _cancel(api)

    assert body["cancelled"] == 2
    after = {row[0]: row for row in _jobs(db)}
    assert set(before) - set(after) == set(qualifying)
    for job_id in untouched:
        assert after[job_id] == before[job_id]


def test_the_cancel_never_widens_beyond_the_channel_and_backfill_id(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    _seed(db, "a0000000001", channel=OTHER, backfill_id=OTHER_BID)
    _seed(db, "a0000000002", channel=OTHER, backfill_id=BID)
    before = _jobs(db)

    # Neither the other channel's jobs nor any LIKE-style input is matched.
    assert _cancel(api, CHANNEL, BID)["cancelled"] == 0
    assert _cancel(api, CHANNEL, OTHER_BID)["cancelled"] == 0
    assert _jobs(db) == before


def test_a_blank_argument_to_the_queue_is_refused_not_matched(
    db: psycopg.Connection[Any],
) -> None:
    _seed(db, "a0000000001", backfill_id="")
    before = _jobs(db)

    queue = PostgresQueue(db)
    with pytest.raises(ValueError):
        queue.cancel_backfill(CHANNEL, "")
    with pytest.raises(ValueError):
        queue.cancel_backfill("", BID)
    assert _jobs(db) == before


def test_an_unknown_or_foreign_backfill_id_is_200_zero_and_writes_nothing(
    api: TestClient, db: psycopg.Connection[Any], logs: LogSink
) -> None:
    _seed(db, "a0000000001", channel=OTHER)
    before = (_jobs(db), _videos(db))

    unknown = _cancel(api, CHANNEL, OTHER_BID)
    foreign = _cancel(api, CHANNEL, BID)  # BID belongs to the other channel's job

    for body in (unknown, foreign):
        assert body["cancelled"] == 0
        assert body["video_ids"] == []
    assert (_jobs(db), _videos(db)) == before


def test_an_unregistered_channel_is_404_and_writes_nothing(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    _seed(db, "a0000000001", channel=OTHER)
    before = (_jobs(db), _videos(db))

    response = api.post(
        "/channels/UCyyyyyyyyyyyyyyyyyyyyyy/backfill/cancel", json={"backfill_id": BID}
    )

    assert response.status_code == 404
    assert (_jobs(db), _videos(db)) == before


def test_a_non_json_request_deletes_nothing(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    _seed(db, "a0000000001")
    before = _jobs(db)

    response = api.post(
        f"/channels/{CHANNEL}/backfill/cancel",
        content=f"backfill_id={BID}".encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 415
    assert _jobs(db) == before


# --- video rows -------------------------------------------------------------------


def test_a_video_row_is_removed_only_when_nothing_else_depends_on_it(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    _seed(db, "plain000001")
    _seed(db, "transcr0001")
    db.execute(
        "INSERT INTO transcripts (video_id, source, segments, full_text)"
        " VALUES ('transcr0001', 'manual_subs', '[]', 'text')"
    )
    _seed(db, "otherjob001")
    _seed(db, "otherjob001", kind="transcribe", state="done")
    _seed(db, "rssorigin01", origin="rss")
    _seed(db, "mediarow001")
    db.execute(
        "INSERT INTO media (video_id, path, bytes, expires_at)"
        " VALUES ('mediarow001', '/x', 1, now() + interval '1 day')"
    )

    body = _cancel(api)

    assert body["cancelled"] == 5
    assert body["videos_removed"] == 1
    assert body["videos_kept"] == 4
    assert body["cancelled"] == body["videos_removed"] + body["videos_kept"]
    assert body["video_ids"] == sorted(body["video_ids"])
    assert _videos(db) == {"transcr0001", "otherjob001", "rssorigin01", "mediarow001"}
    assert db.execute("SELECT count(*) FROM transcripts").fetchone() == (1,)
    assert db.execute("SELECT count(*) FROM media").fetchone() == (1,)
    assert db.execute("SELECT count(*) FROM jobs WHERE video_id = 'otherjob001'").fetchone() == (1,)


def test_a_cancelled_backfill_can_be_run_again_with_a_new_id(
    db: psycopg.Connection[Any],
) -> None:
    first = backfill(db, PostgresQueue(db), Catalog(), CHANNEL, limit=5, dry_run=False)
    assert first.backfill_id is not None

    cancelled = cancel_backfill(db, CHANNEL, first.backfill_id)
    second = backfill(db, PostgresQueue(db), Catalog(), CHANNEL, limit=5, dry_run=False)

    assert cancelled.cancelled == 3
    assert (second.already_known, second.enqueued) == (0, 3)
    assert second.backfill_id != first.backfill_id


def test_cancel_after_a_real_run_with_one_running_and_one_done_is_idempotent(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    result = backfill(db, PostgresQueue(db), Catalog(), CHANNEL, limit=5, dry_run=False)
    assert result.backfill_id is not None
    db.execute(
        "UPDATE jobs SET state='running', locked_by='w', locked_at=now(), heartbeat_at=now()"
        " WHERE video_id = %s",
        (ENTRIES[0].video_id,),
    )
    db.execute(
        "UPDATE jobs SET state='done', finished_at=now() WHERE video_id = %s",
        (ENTRIES[1].video_id,),
    )

    first = _cancel(api, CHANNEL, result.backfill_id)
    second = _cancel(api, CHANNEL, result.backfill_id)

    assert first["cancelled"] == 1
    assert first["video_ids"] == [ENTRIES[2].video_id]
    assert second["cancelled"] == 0
    assert _videos(db) == {ENTRIES[0].video_id, ENTRIES[1].video_id}


def test_the_cancel_is_logged_once_at_warning_without_titles(
    api: TestClient, db: psycopg.Connection[Any], logs: LogSink
) -> None:
    result = backfill(db, PostgresQueue(db), Catalog(), CHANNEL, limit=5, dry_run=False)
    assert result.backfill_id is not None

    _cancel(api, CHANNEL, result.backfill_id)

    [line] = [r for r in logs.records() if r.get("event") == "backfill.cancelled"]
    assert line["level"] == "warning"
    assert line["cancelled"] == 3
    assert "secret title" not in logs.text()


def test_cancel_expects_an_idle_connection_and_a_registered_channel(
    db: psycopg.Connection[Any],
) -> None:
    with pytest.raises(ChannelNotRegistered):
        cancel_backfill(db, "UCyyyyyyyyyyyyyyyyyyyyyy", BID)
    with db.transaction(), pytest.raises(RuntimeError):
        cancel_backfill(db, CHANNEL, BID)


def test_the_cancel_is_committed_and_visible_to_a_fresh_connection(
    head_dsn: str, db: psycopg.Connection[Any]
) -> None:
    _seed(db, "a0000000001")
    cancel_backfill(db, CHANNEL, BID)
    db.close()

    with psycopg.connect(head_dsn) as fresh:
        assert fresh.execute("SELECT count(*) FROM jobs").fetchone() == (0,)
        assert fresh.execute("SELECT count(*) FROM videos").fetchone() == (0,)


# --- races ------------------------------------------------------------------------


def _wait_until_blocked(db: psycopg.Connection[Any], count: int = 1) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = db.execute(
            "SELECT count(*) FROM pg_stat_activity"
            " WHERE datname = current_database() AND wait_event_type = 'Lock'"
        ).fetchone()
        if row is not None and row[0] >= count:
            return
        time.sleep(0.01)
    raise AssertionError("the competing statement never blocked on a lock")


def _in_thread(dsn: str, fn: Callable[[psycopg.Connection[Any]], Any]) -> tuple[threading.Thread, list[Any]]:
    out: list[Any] = []

    def run() -> None:
        with psycopg.connect(dsn, autocommit=True) as conn:
            try:
                out.append(fn(conn))
            except BaseException as exc:  # noqa: BLE001 - surfaced by the caller's assert
                out.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, out


def test_a_claim_in_flight_wins_over_the_cancel(
    head_dsn: str, db: psycopg.Connection[Any]
) -> None:
    claimed = _seed(db, "claimed0001")
    free = _seed(db, "free0000001")
    holder = psycopg.connect(head_dsn)  # the worker's claim, transaction held open
    try:
        holder.execute(
            "UPDATE jobs SET state='running', locked_by='w', locked_at=now(),"
            " heartbeat_at=now() WHERE id = %s",
            (claimed,),
        )
        thread, out = _in_thread(head_dsn, lambda c: cancel_backfill(c, CHANNEL, BID))
        _wait_until_blocked(db)
        holder.commit()
        thread.join(timeout=10)
    finally:
        holder.close()

    [result] = out
    assert result.video_ids == ("free0000001",)
    assert (result.cancelled, result.videos_removed) == (1, 1)
    row = db.execute("SELECT state, locked_by FROM jobs WHERE id = %s", (claimed,)).fetchone()
    assert row == ("running", "w")
    assert "claimed0001" in _videos(db)
    assert free not in {r[0] for r in _jobs(db)}


def test_a_claim_cannot_take_a_job_the_cancel_has_deleted_but_not_committed(
    head_dsn: str, db: psycopg.Connection[Any]
) -> None:
    _seed(db, "cancelled001")
    canceller = psycopg.connect(head_dsn)
    try:
        with canceller.transaction():
            assert len(PostgresQueue(canceller).cancel_backfill(CHANNEL, BID)) == 1
            with PostgresQueue(db).claim(["ingest"], worker="w2") as job:
                during = job
    finally:
        canceller.close()

    assert during is None
    with PostgresQueue(db).claim(["ingest"], worker="w2") as job:
        assert job is None
    assert _jobs(db) == []


def test_two_simultaneous_cancels_delete_each_job_exactly_once(
    head_dsn: str, db: psycopg.Connection[Any]
) -> None:
    for i in range(4):
        _seed(db, f"dup0000000{i}")
    first = psycopg.connect(head_dsn)
    try:
        with first.transaction():
            deleted = PostgresQueue(first).cancel_backfill(CHANNEL, BID)
            thread, out = _in_thread(head_dsn, lambda c: cancel_backfill(c, CHANNEL, BID))
            _wait_until_blocked(db)
            removed = sum(delete_unused_backfill_video(first, j.video_id) for j in deleted)
        thread.join(timeout=10)
    finally:
        first.close()

    [second] = out
    assert not isinstance(second, BaseException), second
    assert len(deleted) + second.cancelled == 4
    assert second.cancelled == 0
    assert removed == 4
    assert _jobs(db) == []


def test_a_cancel_racing_a_new_backfill_never_touches_the_new_backfills_jobs(
    head_dsn: str, db: psycopg.Connection[Any]
) -> None:
    _seed(db, "old00000001")
    new_entries = (CatalogEntry("new00000001", "t", 60), CatalogEntry("new00000002", "t", 60))
    canceller = psycopg.connect(head_dsn)
    try:
        with canceller.transaction():
            assert len(PostgresQueue(canceller).cancel_backfill(CHANNEL, BID)) == 1
            result = backfill(
                db, PostgresQueue(db), Catalog(new_entries), CHANNEL, limit=5, dry_run=False
            )
    finally:
        canceller.close()

    assert result.backfill_id not in (None, BID)
    ids = {r[0] for r in db.execute("SELECT video_id FROM jobs").fetchall()}
    assert ids == {"new00000001", "new00000002"}
