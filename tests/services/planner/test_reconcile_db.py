"""RSS-gap reconciliation against real tables (issue #93). Needs Postgres."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from psycopg.pq import TransactionStatus
from pydantic import SecretStr

from common.config import Settings
from common.errors import ToolFailureError
from common.models import CatalogEntry, ChannelCatalog, FeedEntry
from common.queue import PostgresQueue
from common.repo import videos as videos_repo
from common.repo.videos import insert_backfill_video, insert_discovered_video
from services.planner import reconcile
from services.planner.reconcile import ReconcileResult, reconcile_channels

pytestmark = pytest.mark.integration

CUTOFF = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
AFTER = CUTOFF + timedelta(days=1)


def _e(i: int) -> CatalogEntry:
    return CatalogEntry(f"vid{i:08d}", f"Title {i}", 100 + i)


class Catalog:
    def __init__(self, listings: dict[str, list[CatalogEntry] | BaseException]) -> None:
        self.listings = listings
        self.calls: list[tuple[str, int]] = []
        self.statuses: list[TransactionStatus] = []
        self.conn: psycopg.Connection[Any] | None = None

    def list_uploads(self, channel_id: str, *, limit: int) -> ChannelCatalog:
        self.calls.append((channel_id, limit))
        if self.conn is not None:
            self.statuses.append(self.conn.info.transaction_status)
        result = self.listings[channel_id]
        if isinstance(result, BaseException):
            raise result
        return ChannelCatalog(channel_id, tuple(result), len(result))


def _channel(
    conn: psycopg.Connection[Any], channel_id: str = "UCa", *, active: bool = True
) -> None:
    conn.execute(
        "INSERT INTO channels (channel_id, active, monitor_from) VALUES (%s, %s, %s)",
        (channel_id, active, CUTOFF),
    )
    conn.commit()


def _known(
    conn: psycopg.Connection[Any],
    entry: CatalogEntry,
    *,
    channel_id: str | None = "UCa",
    published_at: datetime | None = AFTER,
    origin: str = "rss",
) -> None:
    conn.execute(
        "INSERT INTO videos (video_id, channel_id, published_at, origin) VALUES (%s, %s, %s, %s)",
        (entry.video_id, channel_id, published_at, origin),
    )
    conn.commit()


def _queue(conn: psycopg.Connection[Any]) -> PostgresQueue:
    return PostgresQueue(conn, settings=Settings(DATABASE_URL=SecretStr("postgresql://u:p@h/db")))


def _run(
    conn: psycopg.Connection[Any], catalog: Catalog, start_after: str | None = None
) -> ReconcileResult:
    catalog.conn = conn
    return reconcile_channels(
        conn, catalog, _queue(conn), clock=lambda: 0.0, start_after=start_after
    )


def _jobs(conn: psycopg.Connection[Any]) -> list[tuple[Any, ...]]:
    conn.commit()
    return conn.execute(
        "SELECT video_id, kind, dedupe_key, priority, state, payload FROM jobs ORDER BY id"
    ).fetchall()


def test_a_recovered_video_has_the_catalog_columns_and_one_priority_zero_ingest_job(
    conn: psycopg.Connection[Any],
) -> None:
    _channel(conn)
    _known(conn, _e(3))
    catalog = Catalog({"UCa": [_e(1), _e(2), _e(3), _e(4)]})

    result = _run(conn, catalog)

    conn.commit()
    assert conn.execute(
        "SELECT video_id, channel_id, title, duration_sec, published_at, origin, unavailable "
        "FROM videos WHERE video_id IN (%s, %s) ORDER BY video_id",
        (_e(1).video_id, _e(2).video_id),
    ).fetchall() == [
        (_e(1).video_id, "UCa", "Title 1", 101, None, "rss", None),
        (_e(2).video_id, "UCa", "Title 2", 102, None, "rss", None),
    ]
    assert conn.execute("SELECT count(*) FROM videos").fetchone() == (3,)  # e4 untouched
    assert _jobs(conn) == [
        (_e(1).video_id, "ingest", "default", 0, "pending", {}),
        (_e(2).video_id, "ingest", "default", 0, "pending", {}),
    ]
    assert result == ReconcileResult(1, 0, 0, 4, 2, 2, "UCa")


def test_running_twice_over_the_same_catalog_creates_nothing_the_second_time(
    conn: psycopg.Connection[Any],
) -> None:
    _channel(conn)
    _known(conn, _e(2))
    catalog = Catalog({"UCa": [_e(1), _e(2)]})

    first = _run(conn, catalog)
    jobs_after_first = _jobs(conn)
    second = _run(conn, catalog)

    assert (first.videos_recovered, second.videos_recovered, second.jobs_enqueued) == (1, 0, 0)
    assert _jobs(conn) == jobs_after_first
    assert conn.execute("SELECT count(*) FROM videos").fetchone() == (2,)


def test_a_video_this_pass_inserted_never_acts_as_an_anchor(
    conn: psycopg.Connection[Any],
) -> None:
    _channel(conn)
    _known(conn, _e(2))
    _run(conn, Catalog({"UCa": [_e(1), _e(2)]}))

    # e1 now has a row with a NULL published_at. In a listing where it is the only known
    # entry, it must not anchor: e0 above it stays unrecovered.
    result = _run(conn, Catalog({"UCa": [_e(0), _e(1), _e(5), _e(6)]}))

    assert result.videos_recovered == 0


def test_a_catalog_call_never_sees_an_open_transaction(conn: psycopg.Connection[Any]) -> None:
    _channel(conn, "UCa")
    _channel(conn, "UCb")
    _known(conn, _e(2), channel_id="UCa")
    _known(conn, _e(12), channel_id="UCb")
    catalog = Catalog({"UCa": [_e(1), _e(2)], "UCb": [_e(11), _e(12)]})

    _run(conn, catalog)

    assert catalog.statuses == [TransactionStatus.IDLE, TransactionStatus.IDLE]
    assert conn.info.transaction_status == TransactionStatus.IDLE


def test_inactive_channels_are_never_listed(conn: psycopg.Connection[Any]) -> None:
    _channel(conn, "UCa")
    _channel(conn, "UCoff", active=False)
    catalog = Catalog({"UCa": []})

    _run(conn, catalog)

    assert [c for c, _ in catalog.calls] == ["UCa"]


def test_the_pass_leaves_poll_health_active_and_monitor_from_alone(
    conn: psycopg.Connection[Any],
) -> None:
    _channel(conn)
    conn.execute(
        "UPDATE channels SET last_polled = %s, last_poll_err = 'old', poll_failures = 3, "
        "poll_failing_since = %s",
        (CUTOFF, CUTOFF),
    )
    conn.commit()
    _known(conn, _e(2))
    before = conn.execute("SELECT * FROM channels").fetchall()

    _run(conn, Catalog({"UCa": [_e(1), _e(2)]}))
    conn.commit()

    assert conn.execute("SELECT * FROM channels").fetchall() == before


def test_a_video_the_poll_inserted_first_keeps_its_row_and_its_single_job(
    conn: psycopg.Connection[Any], head_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _channel(conn)
    _known(conn, _e(3))
    real = videos_repo.known_videos

    def lookup_then_lose_the_race(c: Any, ids: Any) -> Any:
        found = real(c, ids)  # reconcile decides on this (stale) view...
        with psycopg.connect(head_dsn) as poller:  # ...then #34's poll gets there first
            insert_discovered_video(poller, FeedEntry(_e(1).video_id, "UCa", "Feed", AFTER), "rss")
            PostgresQueue(
                poller, settings=Settings(DATABASE_URL=SecretStr("postgresql://u:p@h/db"))
            ).enqueue("ingest", _e(1).video_id, priority=0)
        return found

    monkeypatch.setattr(reconcile, "known_videos", lookup_then_lose_the_race)

    result = _run(conn, Catalog({"UCa": [_e(1), _e(3)]}))

    conn.commit()
    assert (result.videos_recovered, result.jobs_enqueued) == (0, 0)
    assert conn.execute(
        "SELECT title, origin, published_at FROM videos WHERE video_id = %s", (_e(1).video_id,)
    ).fetchone() == ("Feed", "rss", AFTER)
    assert _jobs(conn) == [(_e(1).video_id, "ingest", "default", 0, "pending", {})]


def test_a_video_backfill_inserted_first_keeps_origin_backfill_and_gets_no_second_job(
    conn: psycopg.Connection[Any], head_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _channel(conn)
    _known(conn, _e(3))
    real = videos_repo.known_videos

    def lookup_then_lose_the_race(c: Any, ids: Any) -> Any:
        found = real(c, ids)
        with psycopg.connect(head_dsn) as backfiller:
            insert_backfill_video(backfiller, _e(1), channel_id="UCa")
            PostgresQueue(
                backfiller, settings=Settings(DATABASE_URL=SecretStr("postgresql://u:p@h/db"))
            ).enqueue("ingest", _e(1).video_id, priority=-10)
        return found

    monkeypatch.setattr(reconcile, "known_videos", lookup_then_lose_the_race)

    _run(conn, Catalog({"UCa": [_e(1), _e(3)]}))

    conn.commit()
    assert conn.execute(
        "SELECT origin FROM videos WHERE video_id = %s", (_e(1).video_id,)
    ).fetchone() == ("backfill",)
    assert _jobs(conn) == [(_e(1).video_id, "ingest", "default", -10, "pending", {})]


def test_a_failing_channel_keeps_work_committed_for_the_earlier_one(
    conn: psycopg.Connection[Any],
) -> None:
    _channel(conn, "UCa")
    _channel(conn, "UCb")
    _known(conn, _e(2), channel_id="UCa")
    catalog = Catalog({"UCa": [_e(1), _e(2)], "UCb": ToolFailureError("x")})

    result = _run(conn, catalog)

    assert (result.channels_checked, result.channels_failed) == (1, 1)
    assert [row[0] for row in _jobs(conn)] == [_e(1).video_id]
    assert conn.execute("SELECT count(*) FROM videos").fetchone() == (2,)


def test_an_insert_that_fails_after_the_row_leaves_neither_row_nor_job(
    conn: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _channel(conn)
    _known(conn, _e(2))

    class Boom:
        def enqueue(self, *args: Any, **kwargs: Any) -> int | None:
            raise RuntimeError("queue down")

    catalog = Catalog({"UCa": [_e(1), _e(2)]})
    catalog.conn = conn
    result = reconcile_channels(conn, catalog, Boom(), clock=lambda: 0.0)  # type: ignore[arg-type]

    conn.commit()
    assert result.channels_failed == 1 and result.videos_recovered == 0
    assert conn.execute("SELECT count(*) FROM videos").fetchone() == (1,)  # only the anchor
    assert _jobs(conn) == []


def test_a_failure_on_a_later_video_keeps_the_earlier_one_committed(
    conn: psycopg.Connection[Any],
) -> None:
    # One transaction per video: e1 is durable even though e2's job write blows up and
    # the channel as a whole is reported failed.
    _channel(conn)
    _known(conn, _e(3))
    real = _queue(conn)

    class FailsOnSecond:
        def enqueue(self, kind: str, video_id: str, **kwargs: Any) -> int | None:
            if video_id == _e(2).video_id:
                raise RuntimeError("queue down")
            return real.enqueue(kind, video_id, **kwargs)

    catalog = Catalog({"UCa": [_e(1), _e(2), _e(3)]})
    catalog.conn = conn
    result = reconcile_channels(conn, catalog, FailsOnSecond(), clock=lambda: 0.0)  # type: ignore[arg-type]

    conn.rollback()  # whatever the pass left open must not be what makes e1 durable
    assert result.channels_failed == 1 and result.videos_recovered == 1
    assert conn.execute(
        "SELECT video_id FROM videos WHERE origin = 'rss' AND published_at IS NULL"
    ).fetchall() == [(_e(1).video_id,)]
    assert [row[0] for row in _jobs(conn)] == [_e(1).video_id]


def test_recovering_a_video_leaves_the_channel_row_active(
    conn: psycopg.Connection[Any],
) -> None:
    # The shared insert creates a *missing* channel row inactive; an existing one must stay.
    _channel(conn)
    _known(conn, _e(2))
    catalog = Catalog({"UCa": [_e(1), _e(2)]})

    _run(conn, catalog)

    conn.commit()
    assert conn.execute("SELECT active FROM channels WHERE channel_id = 'UCa'").fetchone() == (True,)
