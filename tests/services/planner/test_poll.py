"""Tests for the planner's channel polling pass (issue #34).

The first half runs without Postgres: a fake feed, queue and connection, with
the three repository functions ``poll`` imports replaced by in-memory fakes.
The second half (``integration``) runs against real tables.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from common.errors import MAX_ERROR_BYTES
from common.models import Channel, FeedEntry, FeedSource
from common.queue import PostgresQueue
from common.repo.channels import PollHealth, list_active_channels
from services.planner import poll
from services.planner.poll import PollResult, poll_channels

CUTOFF = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
LATER = CUTOFF + timedelta(seconds=1)
EARLIER = CUTOFF - timedelta(seconds=1)


def _entry(
    video_id: str = "vid00000001",
    channel_id: str = "UCa",
    published_at: datetime | None = LATER,
    title: str = "A title",
) -> FeedEntry:
    return FeedEntry(
        video_id=video_id,
        channel_id=channel_id,
        title=title,
        published_at=published_at,  # type: ignore[arg-type]
    )


def _channel(
    channel_id: str = "UCa",
    monitor_from: datetime = CUTOFF,
    last_polled: datetime | None = None,
) -> Channel:
    return Channel(
        channel_id=channel_id,
        title=None,
        active=True,
        monitor_from=monitor_from,
        last_polled=last_polled,
        last_poll_err=None,
        added_at=CUTOFF,
        poll_failing_since=None,
        poll_failures=0,
    )


class FakeFeed:
    def __init__(self, feeds: dict[str, list[FeedEntry] | Exception]) -> None:
        self._feeds = feeds
        self.fetched: list[str] = []

    def fetch(self, channel_id: str) -> list[FeedEntry]:
        self.fetched.append(channel_id)
        result = self._feeds[channel_id]
        if isinstance(result, Exception):
            raise result
        return result


class FakeQueue:
    def __init__(self, existing_jobs: set[str] | None = None) -> None:
        self.existing_jobs = existing_jobs or set()
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def enqueue(self, kind: str, video_id: str, **kwargs: Any) -> int | None:
        self.calls.append((kind, video_id, kwargs))
        if video_id in self.existing_jobs:
            return None
        return len(self.calls)


class FakeConn:
    def __init__(self) -> None:
        self.events: list[str] = []

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.events.append("begin")
        try:
            yield
        except BaseException:
            self.events.append("rollback")
            raise
        self.events.append("commit")

    def commit(self) -> None:
        self.events.append("commit")

    def rollback(self) -> None:
        self.events.append("rollback")


class FakeRepo:
    """In-memory stand-ins for the repo functions ``poll`` calls."""

    def __init__(self, channels: list[Channel], known: set[str] | None = None) -> None:
        self.channels = channels
        self.known = known or set()
        self.inserted: list[tuple[FeedEntry, str]] = []
        self.polls: list[tuple[str, str | None]] = []
        self.health: dict[str, PollHealth] = {}

    def list_active_channels(self, conn: Any) -> list[Channel]:
        return list(self.channels)

    def insert_discovered_video(self, conn: Any, entry: FeedEntry, origin: str) -> bool:
        if entry.video_id in self.known:
            return False
        self.known.add(entry.video_id)
        self.inserted.append((entry, origin))
        return True

    def record_poll(
        self, conn: Any, channel_id: str, error: str | None = None
    ) -> PollHealth | None:
        """Same rules as the real function (#90), with a fixed fake clock."""
        self.polls.append((channel_id, error))
        before = self.health.get(channel_id, PollHealth(None, 0))
        if error is None:
            after = PollHealth(None, 0)
        else:
            since = before.poll_failing_since or CUTOFF + timedelta(minutes=len(self.polls))
            after = PollHealth(since, before.poll_failures + 1)
        self.health[channel_id] = after
        return after


@pytest.fixture
def make_repo(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(channels: list[Channel], known: set[str] | None = None) -> FakeRepo:
        repo = FakeRepo(channels, known)
        monkeypatch.setattr(poll, "list_active_channels", repo.list_active_channels)
        monkeypatch.setattr(poll, "insert_discovered_video", repo.insert_discovered_video)
        monkeypatch.setattr(poll, "record_poll", repo.record_poll)
        return repo

    return install


def _run(feed: FakeFeed, queue: FakeQueue | None = None) -> tuple[PollResult, FakeQueue]:
    queue = queue or FakeQueue()
    result = poll_channels(FakeConn(), feed, queue)  # type: ignore[arg-type]
    return result, queue


def test_fake_feed_satisfies_the_feed_source_port() -> None:
    feed: FeedSource = FakeFeed({"UCa": []})
    assert feed.fetch("UCa") == []


# -- discovery and the cutoff ------------------------------------------------


def test_no_active_channels_fetches_nothing_and_returns_zeros(make_repo: Any) -> None:
    make_repo([])
    feed = FakeFeed({})

    result, queue = _run(feed)

    assert result == PollResult(0, 0, 0, 0)
    assert feed.fetched == []
    assert queue.calls == []


def test_each_active_channel_is_fetched_exactly_once(make_repo: Any) -> None:
    make_repo([_channel("UCa"), _channel("UCb")])
    feed = FakeFeed({"UCa": [], "UCb": []})

    result, _ = _run(feed)

    assert feed.fetched == ["UCa", "UCb"]
    assert result.channels_polled == 2


def test_new_entry_is_inserted_as_rss_and_queued_as_a_normal_ingest_job(
    make_repo: Any,
) -> None:
    repo = make_repo([_channel()])
    entry = _entry()

    result, queue = _run(FakeFeed({"UCa": [entry]}))

    assert repo.inserted == [(entry, "rss")]
    assert queue.calls == [("ingest", "vid00000001", {"priority": 0})]
    assert result == PollResult(1, 0, 1, 1)


def test_entry_published_exactly_at_the_cutoff_is_not_queued(make_repo: Any) -> None:
    repo = make_repo([_channel(monitor_from=CUTOFF)])

    result, queue = _run(FakeFeed({"UCa": [_entry(published_at=CUTOFF)]}))

    assert repo.inserted == []
    assert queue.calls == []
    assert result.videos_discovered == 0


def test_entry_one_second_after_the_cutoff_is_queued(make_repo: Any) -> None:
    repo = make_repo([_channel(monitor_from=CUTOFF)])

    _run(FakeFeed({"UCa": [_entry(published_at=CUTOFF + timedelta(seconds=1))]}))

    assert len(repo.inserted) == 1


def test_entry_before_the_cutoff_is_ignored_completely(make_repo: Any) -> None:
    repo = make_repo([_channel()])

    _, queue = _run(FakeFeed({"UCa": [_entry(published_at=EARLIER)]}))

    assert repo.inserted == []
    assert queue.calls == []


def test_video_that_already_has_a_row_gets_no_job(make_repo: Any) -> None:
    make_repo([_channel()], known={"vid00000001"})

    result, queue = _run(FakeFeed({"UCa": [_entry()]}))

    assert queue.calls == []
    assert result == PollResult(1, 0, 0, 0)


def test_video_with_an_active_job_is_inserted_but_not_counted_as_enqueued(
    make_repo: Any,
) -> None:
    repo = make_repo([_channel()])
    queue = FakeQueue(existing_jobs={"vid00000001"})

    result, _ = _run(FakeFeed({"UCa": [_entry()]}), queue)

    assert len(repo.inserted) == 1
    assert result == PollResult(1, 0, 1, 0)


def test_same_video_twice_in_one_feed_gives_one_row_and_one_job(make_repo: Any) -> None:
    repo = make_repo([_channel()])

    result, queue = _run(FakeFeed({"UCa": [_entry(), _entry()]}))

    assert len(repo.inserted) == 1
    assert len(queue.calls) == 1
    assert result == PollResult(1, 0, 1, 1)


def test_each_channel_uses_its_own_cutoff(make_repo: Any) -> None:
    repo = make_repo(
        [_channel("UCa", monitor_from=CUTOFF), _channel("UCb", monitor_from=CUTOFF + timedelta(days=1))]
    )
    feed = FakeFeed(
        {
            "UCa": [_entry("vid00000001", "UCa")],
            "UCb": [_entry("vid00000002", "UCb")],
        }
    )

    _run(feed)

    assert [e.video_id for e, _ in repo.inserted] == ["vid00000001"]


# -- defensive handling ------------------------------------------------------


def test_entry_of_another_channel_is_skipped_with_a_warning(
    make_repo: Any, caplog: pytest.LogCaptureFixture
) -> None:
    repo = make_repo([_channel("UCa")])
    caplog.set_level(logging.WARNING)

    result, _ = _run(FakeFeed({"UCa": [_entry(channel_id="UCother")]}))

    assert repo.inserted == []
    assert result.channels_failed == 0
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.parametrize(
    "published_at",
    [None, datetime(2030, 1, 1)],  # noqa: DTZ001 - naive on purpose
    ids=["none", "naive"],
)
def test_entry_with_a_missing_or_naive_timestamp_is_skipped_with_a_warning(
    make_repo: Any, caplog: pytest.LogCaptureFixture, published_at: datetime | None
) -> None:
    repo = make_repo([_channel()])
    caplog.set_level(logging.WARNING)

    result, _ = _run(FakeFeed({"UCa": [_entry(published_at=published_at)]}))

    assert repo.inserted == []
    assert result.channels_failed == 0
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


# -- per-channel outcome and isolation ---------------------------------------


def test_successful_fetch_records_a_clean_poll_even_when_empty(make_repo: Any) -> None:
    repo = make_repo([_channel()])

    _run(FakeFeed({"UCa": []}))

    assert repo.polls == [("UCa", None)]


def test_fetch_failure_is_recorded_and_the_other_channels_are_still_polled(
    make_repo: Any, caplog: pytest.LogCaptureFixture
) -> None:
    repo = make_repo([_channel("UCa"), _channel("UCb")])
    caplog.set_level(logging.WARNING)
    feed = FakeFeed({"UCa": ValueError("boom"), "UCb": [_entry("vid00000002", "UCb")]})

    result, _ = _run(feed)

    assert ("UCa", "ValueError: boom") in repo.polls
    assert ("UCb", None) in repo.polls
    assert [e.video_id for e, _ in repo.inserted] == ["vid00000002"]
    assert result == PollResult(channels_polled=1, channels_failed=1, videos_discovered=1, jobs_enqueued=1)
    failure_logs = [r for r in caplog.records if r.levelno == logging.WARNING and r.exc_info]
    assert failure_logs
    assert getattr(failure_logs[0], "channel_id", None) == "UCa"


def _failed_logs(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == "planner.poll.channel_failed"]


def test_channel_failing_over_three_polls_then_recovering_logs_each_failure_run(
    make_repo: Any, caplog: pytest.LogCaptureFixture
) -> None:
    repo = make_repo([_channel()])
    caplog.set_level(logging.WARNING)
    failing = FakeFeed({"UCa": ValueError("boom")})

    for _ in range(3):
        _run(failing)

    logs = _failed_logs(caplog)
    assert [r.__dict__["poll_failures"] for r in logs] == [1, 2, 3]
    starts = {r.__dict__["poll_failing_since"] for r in logs}
    assert starts == {(CUTOFF + timedelta(minutes=1)).isoformat()}  # ISO 8601 UTC, unchanged

    _run(FakeFeed({"UCa": []}))

    assert repo.health["UCa"] == PollHealth(None, 0)
    assert len(_failed_logs(caplog)) == 3  # success logs no failure line


def test_a_404_and_a_timeout_count_the_same(
    make_repo: Any, caplog: pytest.LogCaptureFixture
) -> None:
    from adapters.youtube.feed import FeedNotFoundError

    repo = make_repo([_channel("UCa"), _channel("UCb")])
    caplog.set_level(logging.WARNING)

    _run(FakeFeed({"UCa": FeedNotFoundError("404"), "UCb": TimeoutError("timed out")}))

    assert repo.health["UCa"].poll_failures == repo.health["UCb"].poll_failures == 1
    assert [r.__dict__["poll_failures"] for r in _failed_logs(caplog)] == [1, 1]


def test_recorded_error_is_capped_at_max_error_bytes(make_repo: Any) -> None:
    repo = make_repo([_channel()])

    _run(FakeFeed({"UCa": RuntimeError("é" * MAX_ERROR_BYTES)}))

    ((_, error),) = repo.polls
    assert error is not None
    assert error.startswith("RuntimeError: ")
    assert len(error.encode()) <= MAX_ERROR_BYTES


def test_failure_while_processing_entries_is_recorded_and_keeps_earlier_work(
    make_repo: Any,
) -> None:
    repo = make_repo([_channel("UCa"), _channel("UCb")])

    class ExplodingQueue(FakeQueue):
        def enqueue(self, kind: str, video_id: str, **kwargs: Any) -> int | None:
            if video_id == "vid00000002":
                raise RuntimeError("queue broke")
            return super().enqueue(kind, video_id, **kwargs)

    feed = FakeFeed(
        {
            "UCa": [_entry("vid00000001"), _entry("vid00000002"), _entry("vid00000003")],
            "UCb": [_entry("vid00000004", "UCb")],
        }
    )

    result, queue = _run(feed, ExplodingQueue())

    assert ("UCa", "RuntimeError: queue broke") in repo.polls
    assert ("UCb", None) in repo.polls
    assert [c[1] for c in queue.calls] == ["vid00000001", "vid00000004"]
    assert result.channels_failed == 1
    assert result.channels_polled == 1
    assert result.jobs_enqueued == 2


def test_database_connection_errors_propagate(make_repo: Any) -> None:
    make_repo([_channel()])

    class DownQueue(FakeQueue):
        def enqueue(self, kind: str, video_id: str, **kwargs: Any) -> int | None:
            raise psycopg.OperationalError("connection lost")

    with pytest.raises(psycopg.OperationalError):
        _run(FakeFeed({"UCa": [_entry()]}), DownQueue())


# -- feed window overflow ----------------------------------------------------


def _gap_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "possible feed gap" in r.getMessage()]


def test_all_new_entries_after_the_last_poll_log_a_feed_gap(
    make_repo: Any, caplog: pytest.LogCaptureFixture
) -> None:
    last_polled = CUTOFF + timedelta(hours=1)
    make_repo([_channel(last_polled=last_polled)])
    caplog.set_level(logging.WARNING)
    oldest = last_polled + timedelta(minutes=1)
    entries = [
        _entry("vid00000001", published_at=oldest + timedelta(hours=1)),
        _entry("vid00000002", published_at=oldest),
    ]

    _run(FakeFeed({"UCa": entries}))

    (record,) = _gap_warnings(caplog)
    assert record.levelno == logging.WARNING
    assert getattr(record, "channel_id", None) == "UCa"
    assert oldest.isoformat() in str(getattr(record, "oldest_published_at", "")) or (
        getattr(record, "oldest_published_at", None) == oldest
    )


def test_feed_gap_uses_the_cutoff_when_the_channel_was_never_polled(
    make_repo: Any, caplog: pytest.LogCaptureFixture
) -> None:
    make_repo([_channel(last_polled=None)])
    caplog.set_level(logging.WARNING)

    _run(FakeFeed({"UCa": [_entry(published_at=LATER)]}))

    assert len(_gap_warnings(caplog)) == 1


def test_no_feed_gap_when_the_oldest_entry_is_not_after_the_last_poll(
    make_repo: Any, caplog: pytest.LogCaptureFixture
) -> None:
    last_polled = CUTOFF + timedelta(hours=1)
    make_repo([_channel(last_polled=last_polled)])
    caplog.set_level(logging.WARNING)

    _run(FakeFeed({"UCa": [_entry(published_at=last_polled)]}))

    assert _gap_warnings(caplog) == []


def test_no_feed_gap_when_some_entry_already_has_a_row(
    make_repo: Any, caplog: pytest.LogCaptureFixture
) -> None:
    make_repo([_channel()], known={"vid00000002"})
    caplog.set_level(logging.WARNING)

    _run(FakeFeed({"UCa": [_entry("vid00000001"), _entry("vid00000002")]}))

    assert _gap_warnings(caplog) == []


def test_no_feed_gap_when_some_entry_predates_the_cutoff(
    make_repo: Any, caplog: pytest.LogCaptureFixture
) -> None:
    make_repo([_channel()])
    caplog.set_level(logging.WARNING)

    _run(FakeFeed({"UCa": [_entry("vid00000001"), _entry("vid00000002", published_at=EARLIER)]}))

    assert _gap_warnings(caplog) == []


def test_no_feed_gap_for_an_empty_feed(make_repo: Any, caplog: pytest.LogCaptureFixture) -> None:
    make_repo([_channel()])
    caplog.set_level(logging.WARNING)

    _run(FakeFeed({"UCa": []}))

    assert _gap_warnings(caplog) == []


# -- result and logging ------------------------------------------------------


def test_poll_result_is_a_frozen_dataclass() -> None:
    result = PollResult(1, 2, 3, 4)

    with pytest.raises(FrozenInstanceError):
        result.channels_polled = 9  # type: ignore[misc]


def test_one_info_line_carries_the_four_numbers(
    make_repo: Any, caplog: pytest.LogCaptureFixture
) -> None:
    make_repo([_channel("UCa"), _channel("UCb")])
    caplog.set_level(logging.INFO)
    feed = FakeFeed({"UCa": [_entry()], "UCb": RuntimeError("x")})

    _run(feed)

    records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(records) == 1
    record = records[0]
    assert (
        getattr(record, "channels_polled", None),
        getattr(record, "channels_failed", None),
        getattr(record, "videos_discovered", None),
        getattr(record, "jobs_enqueued", None),
    ) == (1, 1, 1, 1)


# -- integration -------------------------------------------------------------


def _add_channel(
    conn: psycopg.Connection[Any],
    channel_id: str = "UCa",
    monitor_from: datetime = CUTOFF,
    active: bool = True,
) -> None:
    conn.execute(
        "INSERT INTO channels (channel_id, active, monitor_from) VALUES (%s, %s, %s)",
        (channel_id, active, monitor_from),
    )
    conn.commit()


def _videos(conn: psycopg.Connection[Any]) -> list[tuple[Any, ...]]:
    conn.commit()
    return conn.execute(
        "SELECT video_id, channel_id, title, published_at, origin FROM videos ORDER BY video_id"
    ).fetchall()


def _jobs(conn: psycopg.Connection[Any]) -> list[tuple[Any, ...]]:
    conn.commit()
    return conn.execute(
        "SELECT video_id, kind, priority, dedupe_key, payload, state FROM jobs ORDER BY video_id"
    ).fetchall()


@pytest.fixture
def queue(conn: psycopg.Connection[Any]) -> PostgresQueue:
    from common.config import Settings

    return PostgresQueue(conn, settings=Settings(DATABASE_URL="postgresql://u:p@h/db"))  # type: ignore[arg-type]


@pytest.mark.integration
def test_poll_inserts_the_row_and_the_job(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    _add_channel(conn)
    feed = FakeFeed({"UCa": [_entry(title="Hello")]})

    result = poll_channels(conn, feed, queue)

    assert result == PollResult(1, 0, 1, 1)
    assert _videos(conn) == [("vid00000001", "UCa", "Hello", LATER, "rss")]
    assert _jobs(conn) == [("vid00000001", "ingest", 0, "default", {}, "pending")]
    row = conn.execute("SELECT last_polled, last_poll_err FROM channels").fetchone()
    assert row is not None
    assert row[0] is not None
    assert row[1] is None


@pytest.mark.integration
def test_polling_the_same_feed_twice_queues_nothing_the_second_time(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    _add_channel(conn)
    feed = FakeFeed({"UCa": [_entry()]})
    poll_channels(conn, feed, queue)
    before = conn.execute("SELECT * FROM videos").fetchall()

    result = poll_channels(conn, feed, queue)

    assert result == PollResult(1, 0, 0, 0)
    assert len(_jobs(conn)) == 1
    assert conn.execute("SELECT * FROM videos").fetchall() == before


@pytest.mark.integration
def test_existing_video_row_is_left_as_it_is(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    _add_channel(conn)
    conn.execute(
        """
        INSERT INTO videos (video_id, channel_id, title, description, origin)
        VALUES ('vid00000001', 'UCa', 'Full title', 'Desc', 'adhoc')
        """
    )
    conn.commit()
    before = conn.execute("SELECT * FROM videos").fetchall()

    result = poll_channels(conn, FakeFeed({"UCa": [_entry(title="Feed title")]}), queue)

    assert result.jobs_enqueued == 0
    assert _jobs(conn) == []
    assert conn.execute("SELECT * FROM videos").fetchall() == before


@pytest.mark.integration
def test_existing_active_job_dedupes_but_the_row_is_still_inserted(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    _add_channel(conn)
    assert queue.enqueue("ingest", "vid00000001", priority=10) is not None

    result = poll_channels(conn, FakeFeed({"UCa": [_entry()]}), queue)

    assert result == PollResult(1, 0, 1, 0)
    assert len(_videos(conn)) == 1
    assert [(j[0], j[2]) for j in _jobs(conn)] == [("vid00000001", 10)]


@pytest.mark.integration
def test_cutoff_boundary_against_real_rows(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    _add_channel(conn)
    entries = [
        _entry("vid00000001", published_at=EARLIER),
        _entry("vid00000002", published_at=CUTOFF),
        _entry("vid00000003", published_at=LATER),
    ]

    poll_channels(conn, FakeFeed({"UCa": entries}), queue)

    assert [v[0] for v in _videos(conn)] == ["vid00000003"]
    assert [j[0] for j in _jobs(conn)] == ["vid00000003"]


@pytest.mark.integration
def test_inactive_channels_are_not_fetched(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    _add_channel(conn, "UCa")
    _add_channel(conn, "UCoff", active=False)
    feed = FakeFeed({"UCa": [], "UCoff": []})

    poll_channels(conn, feed, queue)

    assert feed.fetched == ["UCa"]
    assert [c.channel_id for c in list_active_channels(conn)] == ["UCa"]


@pytest.mark.integration
def test_enqueue_failure_rolls_the_row_back_and_the_next_poll_recovers(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    _add_channel(conn)

    class FlakyQueue:
        def __init__(self) -> None:
            self.failed = False

        def enqueue(self, kind: str, video_id: str, **kwargs: Any) -> int | None:
            if not self.failed:
                self.failed = True
                raise RuntimeError("enqueue broke")
            return queue.enqueue(kind, video_id, **kwargs)

    flaky = FlakyQueue()
    feed = FakeFeed({"UCa": [_entry()]})

    first = poll_channels(conn, feed, flaky)  # type: ignore[arg-type]

    assert first.channels_failed == 1
    assert _videos(conn) == []
    assert _jobs(conn) == []
    err = conn.execute("SELECT last_poll_err FROM channels").fetchone()
    assert err == ("RuntimeError: enqueue broke",)

    second = poll_channels(conn, feed, flaky)  # type: ignore[arg-type]

    assert second == PollResult(1, 0, 1, 1)
    assert len(_videos(conn)) == 1
    assert len(_jobs(conn)) == 1
    assert conn.execute("SELECT last_poll_err FROM channels").fetchone() == (None,)


@pytest.mark.integration
def test_failed_fetch_stores_the_error_and_a_later_success_clears_it(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    _add_channel(conn)

    poll_channels(conn, FakeFeed({"UCa": ValueError("bad feed")}), queue)
    conn.commit()
    stored = conn.execute("SELECT last_polled, last_poll_err FROM channels").fetchone()
    assert stored is not None
    assert stored[0] is not None
    assert stored[1] == "ValueError: bad feed"

    poll_channels(conn, FakeFeed({"UCa": []}), queue)
    conn.commit()
    assert conn.execute("SELECT last_poll_err FROM channels").fetchone() == (None,)


def _health(conn: psycopg.Connection[Any]) -> tuple[Any, ...]:
    conn.commit()
    row = conn.execute(
        "SELECT poll_failing_since, poll_failures, last_polled, last_poll_err, now() FROM channels"
    ).fetchone()
    assert row is not None
    return tuple(row)


@pytest.mark.integration
def test_channel_failing_three_polls_keeps_its_start_then_recovers(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    _add_channel(conn)

    poll_channels(conn, FakeFeed({"UCa": ValueError("bad feed")}), queue)
    since, failures, last_polled, _, db_now = _health(conn)
    assert (failures, since) == (1, last_polled)  # the same now() as last_polled
    assert since <= db_now

    for expected in (2, 3):
        poll_channels(conn, FakeFeed({"UCa": ValueError("bad feed")}), queue)
        new_since, failures, last_polled, err, db_now = _health(conn)
        assert failures == expected
        assert new_since == since  # run start unchanged
        assert since < last_polled <= db_now
        assert err == "ValueError: bad feed"

    poll_channels(conn, FakeFeed({"UCa": []}), queue)
    assert _health(conn)[:2] == (None, 0)


@pytest.mark.integration
def test_a_404_and_a_timeout_count_the_same_in_the_database(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    from adapters.youtube.feed import FeedNotFoundError

    _add_channel(conn, "UCa")
    _add_channel(conn, "UCb")

    poll_channels(
        conn, FakeFeed({"UCa": FeedNotFoundError("404"), "UCb": TimeoutError("slow")}), queue
    )

    conn.commit()
    rows = conn.execute(
        "SELECT channel_id, poll_failures, poll_failing_since IS NOT NULL, last_poll_err"
        " FROM channels ORDER BY channel_id"
    ).fetchall()
    assert rows == [
        ("UCa", 1, True, "FeedNotFoundError: 404"),
        ("UCb", 1, True, "TimeoutError: slow"),
    ]


@pytest.mark.integration
def test_inactive_channel_columns_do_not_change(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    _add_channel(conn, "UCoff", active=False)

    poll_channels(conn, FakeFeed({"UCoff": ValueError("x")}), queue)

    assert _health(conn)[:2] == (None, 0)
