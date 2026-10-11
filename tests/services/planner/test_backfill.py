"""The planner's ``channel_backfill`` task against Postgres (issue #115).

A fake ``CatalogSource`` everywhere: no network, no yt-dlp, no sleep. The
catalog asserts that no transaction is open while it runs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import psycopg
import pytest
import structlog
from psycopg.pq import TransactionStatus
from pydantic import SecretStr

from common.config import Settings
from common.errors import (
    PermanentSourceError,
    RateLimitedError,
    ToolFailureError,
    TransientNetworkError,
    UnavailableReason,
)
from common.models import CatalogEntry, ChannelCatalog
from common.queue import PRIORITY_BACKFILL, PRIORITY_NORMAL, PostgresQueue
from common.repo.backfill_requests import (
    CANCELLED,
    create_request,
    get_request,
    request_backfill_id,
)
from services.api.routes_write import BackfillResponse
from services.planner.backfill import (
    BACKFILL_ENQUEUE_PER_RUN,
    BACKFILL_MAX_PENDING,
    run_channel_backfill,
)
from services.planner.main import run_on_fresh_connection

pytestmark = pytest.mark.integration

CHANNEL = "UCuAXFkgsw1L7xaCfnd5JJOw"
OTHER = "UCxxxxxxxxxxxxxxxxxxxxxx"
SECRET = "yt-dlp stderr https://www.youtube.com/playlist?list=UUsecret"


def _vid(i: int) -> str:
    return f"pb{i:09d}"


def _entries(n: int) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(_vid(i), f"title {i}", 100) for i in range(n))


class FakeCatalog:
    """Records every call and fails the test if a transaction is open."""

    def __init__(self, conn: psycopg.Connection[Any], entries: tuple[CatalogEntry, ...]) -> None:
        self.conn = conn
        self.entries = entries
        self.calls: list[tuple[str, int]] = []
        self.error: Exception | None = None
        self.during: Callable[[], None] | None = None

    def list_uploads(self, channel_id: str, *, limit: int) -> ChannelCatalog:
        assert self.conn.info.transaction_status == TransactionStatus.IDLE
        self.calls.append((channel_id, limit))
        if self.during is not None:
            self.during()
        if self.error is not None:
            raise self.error
        return ChannelCatalog(channel_id, self.entries[:limit], len(self.entries))


@pytest.fixture
def settings(head_dsn: str) -> Settings:
    return Settings(DATABASE_URL=SecretStr(head_dsn))


@pytest.fixture
def db(conn: psycopg.Connection[Any]) -> psycopg.Connection[Any]:
    conn.execute("INSERT INTO channels (channel_id, active) VALUES (%s, true), (%s, true)", (CHANNEL, OTHER))
    conn.commit()
    return conn


def _run(
    conn: psycopg.Connection[Any], catalog: FakeCatalog, settings: Settings
) -> int:
    return run_channel_backfill(conn, PostgresQueue(conn), catalog, settings=settings)


def _request(conn: psycopg.Connection[Any], *, limit: int = 5000, dry_run: bool = False,
             channel: str = CHANNEL) -> int:
    rid = create_request(conn, channel, limit_n=limit, dry_run=dry_run).request_id
    conn.commit()
    return rid


def _row(conn: psycopg.Connection[Any], rid: int, channel: str = CHANNEL) -> Any:
    row = get_request(conn, channel, rid)
    conn.commit()
    assert row is not None
    return row


def _count(conn: psycopg.Connection[Any], table: str) -> int:
    row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
    conn.commit()
    assert row is not None
    return int(row[0])


def _drain(conn: psycopg.Connection[Any]) -> None:
    conn.execute("UPDATE jobs SET state = 'done' WHERE kind = 'ingest' AND state = 'pending'")
    conn.commit()


def test_with_no_request_it_does_nothing_and_returns_zero(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    catalog = FakeCatalog(db, _entries(3))

    assert _run(db, catalog, settings) == 0
    assert catalog.calls == []
    assert db.info.transaction_status == TransactionStatus.IDLE


def test_it_lists_once_with_the_requests_limit_and_marks_running_before_listing(
    db: psycopg.Connection[Any], settings: Settings, head_dsn: str
) -> None:
    rid = _request(db, limit=7)
    catalog = FakeCatalog(db, _entries(3))
    seen: list[str] = []

    def peek() -> None:
        # A separate connection sees the claim only if it was committed.
        with psycopg.connect(head_dsn) as other:
            row = other.execute("SELECT state FROM backfill_requests WHERE id = %s", (rid,)).fetchone()
        assert row is not None
        seen.append(str(row[0]))

    catalog.during = peek

    _run(db, catalog, settings)

    assert catalog.calls == [(CHANNEL, 7)]
    assert seen == ["running"]


def test_a_real_run_writes_backfill_rows_and_jobs_in_catalog_order(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    rid = _request(db)
    catalog = FakeCatalog(db, _entries(3))

    assert _run(db, catalog, settings) == 3

    jobs = db.execute(
        "SELECT video_id, priority, payload, v.origin FROM jobs j JOIN videos v USING (video_id)"
        " WHERE kind = 'ingest' ORDER BY j.id"
    ).fetchall()
    assert [j[0] for j in jobs] == [_vid(0), _vid(1), _vid(2)]
    assert {j[1] for j in jobs} == {PRIORITY_BACKFILL}
    assert {j[3] for j in jobs} == {"backfill"}
    assert all(j[2] == {"origin": "backfill", "backfill_id": request_backfill_id(rid)} for j in jobs)
    row = _row(db, rid)
    assert (row.state, row.enqueued, row.attempts) == ("done", 3, 0)
    assert row.finished_at is not None


def test_a_dry_run_writes_the_result_and_no_videos_or_jobs(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    rid = _request(db, dry_run=True)
    catalog = FakeCatalog(db, _entries(4))
    before = (_count(db, "videos"), _count(db, "jobs"))

    assert _run(db, catalog, settings) == 0

    assert (_count(db, "videos"), _count(db, "jobs")) == before == (0, 0)
    row = _row(db, rid)
    assert (row.state, row.enqueued) == ("done", 0)
    assert row.result["enqueued"] == 0 and row.result["new_videos"] == 4
    assert row.result["backfill_id"] is None
    assert row.finished_at is not None


def test_result_has_exactly_the_synchronous_routes_response_keys(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    rid = _request(db, dry_run=True)
    _run(db, FakeCatalog(db, _entries(2)), settings)

    result = _row(db, rid).result

    assert set(result) == set(BackfillResponse.model_fields)
    assert set(result["estimate"]) == set(BackfillResponse.model_fields["estimate"].annotation.model_fields)  # type: ignore[union-attr]
    BackfillResponse.model_validate(result)  # same shape, same types


def test_1200_new_videos_take_three_runs_of_500_500_200(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    rid = _request(db)
    catalog = FakeCatalog(db, _entries(1200))
    per_run: list[int] = []
    states: list[str] = []
    for _ in range(3):
        per_run.append(_run(db, catalog, settings))
        _drain(db)  # the -10 jobs drain between runs
        states.append(_row(db, rid).state)

    assert per_run == [500, 500, 200]
    assert states == ["running", "running", "done"]
    row = _row(db, rid)
    assert row.enqueued == 1200
    assert _count(db, "videos") == 1200 and _count(db, "jobs") == 1200
    assert _run(db, catalog, settings) == 0  # nothing left to claim
    assert len(catalog.calls) == 3


def test_each_runs_result_describes_that_runs_listing_and_the_column_is_cumulative(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    rid = _request(db)
    catalog = FakeCatalog(db, _entries(700))

    _run(db, catalog, settings)
    first = _row(db, rid)
    _drain(db)
    _run(db, catalog, settings)
    second = _row(db, rid)

    assert first.result["enqueued"] == 500 and first.enqueued == 500
    assert first.result["new_videos"] == 700 and len(first.result["video_ids"]) == 700
    assert second.result["enqueued"] == 200 and second.enqueued == 700
    assert second.result["new_videos"] == 200
    assert second.result["video_ids"] == [_vid(i) for i in range(500, 700)]
    assert second.result["already_known"] == 500


def test_exactly_the_per_run_cap_of_new_videos_finishes_in_one_run(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    rid = _request(db)

    assert _run(db, FakeCatalog(db, _entries(BACKFILL_ENQUEUE_PER_RUN)), settings) == 500

    assert _row(db, rid).state == "done"


def test_one_over_the_per_run_cap_needs_a_second_run(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    rid = _request(db)
    catalog = FakeCatalog(db, _entries(BACKFILL_ENQUEUE_PER_RUN + 1))

    assert _run(db, catalog, settings) == 500
    assert _row(db, rid).state == "running"
    _drain(db)
    assert _run(db, catalog, settings) == 1
    assert _row(db, rid).state == "done"


def _seed_pending_backfill_jobs(conn: psycopg.Connection[Any], n: int) -> None:
    conn.execute(
        "INSERT INTO videos (video_id, channel_id, origin)"
        " SELECT 'sd' || lpad(g::text, 9, '0'), %s, 'backfill' FROM generate_series(1, %s) g",
        (CHANNEL, n),
    )
    conn.execute(
        "INSERT INTO jobs (video_id, kind, state, priority)"
        " SELECT 'sd' || lpad(g::text, 9, '0'), 'ingest', 'pending', %s FROM generate_series(1, %s) g",
        (PRIORITY_BACKFILL, n),
    )
    conn.commit()


@pytest.mark.parametrize(("pending", "expected"), [(1000, 0), (1200, 0), (999, 1), (700, 300)])
def test_backpressure_shrinks_the_run_to_the_free_budget(
    db: psycopg.Connection[Any], settings: Settings, pending: int, expected: int
) -> None:
    _seed_pending_backfill_jobs(db, pending)
    rid = _request(db)
    catalog = FakeCatalog(db, _entries(400))

    assert _run(db, catalog, settings) == expected

    row = _row(db, rid)
    assert row.attempts == 0 and row.error_class is None
    assert row.enqueued == expected
    if expected == 0:
        assert row.state == "running"
        assert catalog.calls == []  # no 300 s listing for a run that may not write
    else:
        assert catalog.calls == [(CHANNEL, 5000)]


def test_only_pending_ingests_at_the_backfill_priority_count_toward_backpressure(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    _seed_pending_backfill_jobs(db, BACKFILL_MAX_PENDING)
    db.execute("UPDATE jobs SET state = 'done' WHERE id % 2 = 0")
    db.execute("UPDATE jobs SET priority = %s WHERE id %% 2 = 1 AND id %% 4 = 1", (PRIORITY_NORMAL,))
    db.commit()
    rid = _request(db)

    ran = _run(db, FakeCatalog(db, _entries(2000)), settings)

    assert ran == 500  # 250 pending at -10 remain: the budget is 750, the run cap 500
    assert _row(db, rid).state == "running"


@pytest.mark.parametrize("error_type", [TransientNetworkError, RateLimitedError])
def test_transient_failures_count_attempts_and_the_third_fails_the_request(
    db: psycopg.Connection[Any], settings: Settings, error_type: type[Exception]
) -> None:
    rid = _request(db)
    catalog = FakeCatalog(db, _entries(3))
    catalog.error = error_type(SECRET)
    states = []
    for _ in range(3):
        assert _run(db, catalog, settings) == 0
        row = _row(db, rid)
        states.append((row.state, row.attempts, row.error_class))

    assert states == [
        ("running", 1, error_type.__name__),
        ("running", 2, error_type.__name__),
        ("failed", 3, error_type.__name__),
    ]
    assert _row(db, rid).finished_at is not None
    assert _run(db, catalog, settings) == 0 and len(catalog.calls) == 3  # failed: not claimed
    assert _count(db, "videos") == 0


def test_a_success_after_a_transient_failure_finishes_the_request(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    rid = _request(db)
    catalog = FakeCatalog(db, _entries(2))
    catalog.error = TransientNetworkError(SECRET)
    _run(db, catalog, settings)
    catalog.error = None

    assert _run(db, catalog, settings) == 2

    row = _row(db, rid)
    assert (row.state, row.attempts, row.error_class) == ("done", 1, None)


def test_a_tool_failure_fails_the_request_at_once(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    rid = _request(db)
    catalog = FakeCatalog(db, _entries(3))
    catalog.error = ToolFailureError(SECRET)

    assert _run(db, catalog, settings) == 0

    row = _row(db, rid)
    assert (row.state, row.attempts, row.error_class) == ("failed", 1, "ToolFailureError")


def test_a_permanent_source_error_is_an_empty_listing_and_marks_nothing(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    db.execute("INSERT INTO videos (video_id, channel_id) VALUES ('known000001', %s)", (CHANNEL,))
    db.commit()
    channels_before = db.execute("SELECT * FROM channels ORDER BY channel_id").fetchall()
    videos_before = db.execute("SELECT * FROM videos ORDER BY video_id").fetchall()
    rid = _request(db)
    catalog = FakeCatalog(db, ())
    catalog.error = PermanentSourceError(UnavailableReason.REMOVED, SECRET)

    assert _run(db, catalog, settings) == 0

    row = _row(db, rid)
    assert (row.state, row.attempts, row.error_class) == ("done", 0, None)
    assert row.result["listed"] == 0 and row.result["enqueued"] == 0
    assert db.execute("SELECT * FROM channels ORDER BY channel_id").fetchall() == channels_before
    assert db.execute("SELECT * FROM videos ORDER BY video_id").fetchall() == videos_before


def test_no_failure_stores_or_logs_the_exception_message(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    rid = _request(db)
    catalog = FakeCatalog(db, _entries(3))
    catalog.error = ToolFailureError(SECRET)

    with structlog.testing.capture_logs() as logs:
        _run(db, catalog, settings)

    stored = db.execute("SELECT row_to_json(r)::text FROM backfill_requests r").fetchone()
    db.commit()
    assert stored is not None and "yt-dlp" not in stored[0] and "UUsecret" not in stored[0]
    failed = [e for e in logs if e["event"] == "planner.channel_backfill.failed"]
    assert len(failed) == 1
    assert failed[0]["request_id"] == rid and failed[0]["error_class"] == "ToolFailureError"
    assert SECRET not in repr(logs) and "UUsecret" not in repr(logs)


class _ExplodingQueue(PostgresQueue):
    """Raises on the ``n``-th enqueue, as if the planner died there."""

    def __init__(self, conn: psycopg.Connection[Any], explode_on: int) -> None:
        super().__init__(conn)
        self.explode_on = explode_on
        self.count = 0

    def enqueue(self, *args: Any, **kwargs: Any) -> Any:
        self.count += 1
        if self.count == self.explode_on:
            raise RuntimeError("planner died")
        return super().enqueue(*args, **kwargs)


def test_a_crash_after_n_of_m_writes_resumes_without_duplicates(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    rid = _request(db)
    catalog = FakeCatalog(db, _entries(10))

    with pytest.raises(RuntimeError, match="planner died"):
        run_channel_backfill(db, _ExplodingQueue(db, explode_on=5), catalog, settings=settings)
    db.rollback()

    crashed = _row(db, rid)
    assert crashed.state == "running"  # resumable, not lost
    assert _count(db, "videos") == 4 and _count(db, "jobs") == 4  # only complete videos

    assert _run(db, catalog, settings) == 6

    assert _row(db, rid).state == "done"
    assert _count(db, "videos") == 10 and _count(db, "jobs") == 10
    dupes = db.execute("SELECT count(*) FROM (SELECT video_id FROM jobs GROUP BY video_id HAVING count(*) > 1) d").fetchone()
    assert dupes == (0,)


def test_a_video_rss_inserted_first_keeps_its_origin_and_priority(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    rid = _request(db)
    catalog = FakeCatalog(db, _entries(3))

    def rss_wins() -> None:
        with db.transaction():
            db.execute(
                "INSERT INTO videos (video_id, channel_id, origin) VALUES (%s, %s, 'rss')",
                (_vid(1), CHANNEL),
            )
            PostgresQueue(db).enqueue("ingest", _vid(1), payload={"origin": "rss"}, priority=PRIORITY_NORMAL)

    catalog.during = rss_wins

    assert _run(db, catalog, settings) == 2

    row = db.execute(
        "SELECT v.origin, j.priority FROM videos v JOIN jobs j USING (video_id) WHERE video_id = %s",
        (_vid(1),),
    ).fetchall()
    assert row == [("rss", PRIORITY_NORMAL)]
    assert _count(db, "jobs") == 3
    assert _row(db, rid).result["already_known"] == 1


def test_backfill_never_changes_a_channels_row(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    before = db.execute("SELECT * FROM channels ORDER BY channel_id").fetchall()
    _request(db)
    _run(db, FakeCatalog(db, _entries(5)), settings)

    assert db.execute("SELECT * FROM channels ORDER BY channel_id").fetchall() == before


def test_the_oldest_active_request_goes_first(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    newer = _request(db, channel=CHANNEL)
    older = _request(db, channel=OTHER)
    db.execute("UPDATE backfill_requests SET created_at = now() - interval '1 day' WHERE id = %s", (older,))
    db.commit()
    catalog = FakeCatalog(db, _entries(1))

    _run(db, catalog, settings)

    assert catalog.calls == [(OTHER, 5000)]
    assert _row(db, older, OTHER).state == "done" and _row(db, newer).state == "pending"


def test_a_request_cancelled_during_the_listing_stays_cancelled(
    db: psycopg.Connection[Any], settings: Settings, head_dsn: str
) -> None:
    from services.api.backfill import cancel_backfill

    rid = _request(db)
    catalog = FakeCatalog(db, _entries(3))

    def cancel() -> None:
        with psycopg.connect(head_dsn) as other:
            cancel_backfill(other, CHANNEL, request_backfill_id(rid))

    catalog.during = cancel

    _run(db, catalog, settings)

    row = _row(db, rid)
    assert (row.state, row.error_class) == ("failed", CANCELLED)
    assert row.enqueued == 0 and row.result is None


def test_one_info_line_per_run_without_titles(
    db: psycopg.Connection[Any], settings: Settings
) -> None:
    rid = _request(db)
    ticks = iter([10.0, 10.25, 10.5, 11.0, 12.0])

    with structlog.testing.capture_logs() as logs:
        run_channel_backfill(
            db, PostgresQueue(db), FakeCatalog(db, _entries(2)), settings=settings,
            clock=lambda: next(ticks),
        )

    lines = [e for e in logs if e["event"] == "planner.channel_backfill"]
    assert len(lines) == 1
    line = lines[0]
    assert line["log_level"] == "info"
    expected = {"request_id", "channel_id", "dry_run", "listed", "new_videos",
                "enqueued_this_run", "enqueued_total", "state", "duration_ms"}
    assert expected <= set(line) - {"event", "log_level"}
    assert (line["request_id"], line["listed"], line["new_videos"]) == (rid, 2, 2)
    assert (line["enqueued_this_run"], line["enqueued_total"], line["state"]) == (2, 2, "done")
    assert "title" not in repr(line)


def test_enqueued_work_survives_the_connection_closing(
    db: psycopg.Connection[Any], settings: Settings, head_dsn: str
) -> None:
    """The #199 failure mode: run through the planner's own wrapper, then read on a fresh connection."""
    rid = _request(db)
    catalog = FakeCatalog(db, _entries(3))

    def task(conn: psycopg.Connection[Any]) -> int:
        catalog.conn = conn
        return run_channel_backfill(conn, PostgresQueue(conn), catalog, settings=settings)

    assert run_on_fresh_connection(lambda: psycopg.connect(head_dsn), task) == 3

    with psycopg.connect(head_dsn) as fresh:
        assert fresh.execute("SELECT count(*) FROM videos").fetchone() == (3,)
        assert fresh.execute("SELECT count(*) FROM jobs WHERE kind = 'ingest'").fetchone() == (3,)
        row = fresh.execute(
            "SELECT state, enqueued FROM backfill_requests WHERE id = %s", (rid,)
        ).fetchone()
    assert row == ("done", 3)


@pytest.mark.parametrize("scenario", ["idle", "dry_run", "backpressure", "failure"])
def test_every_path_returns_with_an_idle_connection(
    db: psycopg.Connection[Any], settings: Settings, scenario: str
) -> None:
    catalog = FakeCatalog(db, _entries(3))
    if scenario == "dry_run":
        _request(db, dry_run=True)
    elif scenario == "backpressure":
        _seed_pending_backfill_jobs(db, BACKFILL_MAX_PENDING)
        _request(db)
    elif scenario == "failure":
        _request(db)
        catalog.error = TransientNetworkError("x")

    _run(db, catalog, settings)

    assert db.info.transaction_status == TransactionStatus.IDLE
