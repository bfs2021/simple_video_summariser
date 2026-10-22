"""Unit tests for RSS-gap reconciliation (issue #93). No Docker.

Fakes in the style of ``test_poll.py``: a fake catalog, queue and connection, with
the three repository functions ``reconcile`` imports replaced by in-memory fakes.
The real-SQL half is ``test_reconcile_db.py``.
"""

from __future__ import annotations

import ast
import inspect
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from common.errors import (
    PermanentSourceError,
    RateLimitedError,
    ToolFailureError,
    TransientNetworkError,
    UnavailableReason,
)
from common.models import CatalogEntry, CatalogSource, Channel, ChannelCatalog
from common.queue import PRIORITY_NORMAL
from common.repo.videos import KnownVideo
from services.planner import reconcile
from services.planner.reconcile import (
    RECONCILE_LIST_LIMIT,
    RECONCILE_PASS_BUDGET_SEC,
    ReconcileResult,
    reconcile_channels,
)

CUTOFF = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
AFTER = CUTOFF + timedelta(days=1)
BEFORE = CUTOFF - timedelta(days=1)


def _e(i: int) -> CatalogEntry:
    return CatalogEntry(f"vid{i:08d}", f"Title {i}", 100 + i)


def _ids(*numbers: int) -> list[str]:
    return [_e(n).video_id for n in numbers]


def _channel(
    channel_id: str = "UCa",
    monitor_from: datetime = CUTOFF,
    gap_reason: str | None = None,
) -> Channel:
    return Channel(
        channel_id=channel_id,
        title=None,
        active=True,
        monitor_from=monitor_from,
        last_polled=None,
        last_poll_err=None,
        added_at=CUTOFF,
        poll_failing_since=None,
        poll_failures=0,
        feed_gap_detected_at=CUTOFF if gap_reason is not None else None,
        feed_gap_reason=gap_reason,
    )


class FakeCatalog:
    def __init__(
        self,
        listings: dict[str, list[CatalogEntry] | BaseException],
        on_call: Callable[[str], None] | None = None,
    ) -> None:
        self.listings = listings
        self.on_call = on_call
        self.calls: list[tuple[str, int]] = []

    def list_uploads(self, channel_id: str, *, limit: int) -> ChannelCatalog:
        self.calls.append((channel_id, limit))
        if self.on_call is not None:
            self.on_call(channel_id)
        result = self.listings[channel_id]
        if isinstance(result, BaseException):
            raise result
        return ChannelCatalog(channel_id, tuple(result), len(result))

    @property
    def listed(self) -> list[str]:
        return [c for c, _ in self.calls]


class FakeQueue:
    def __init__(self, existing_jobs: set[str] | None = None) -> None:
        self.existing_jobs = existing_jobs or set()
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def enqueue(self, kind: str, video_id: str, **kwargs: Any) -> int | None:
        self.calls.append((kind, video_id, kwargs))
        return None if video_id in self.existing_jobs else len(self.calls)


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
    """In-memory stand-ins for the repo functions ``reconcile`` calls."""

    def __init__(self, channels: list[Channel], known: dict[str, KnownVideo]) -> None:
        self.channels = channels
        self.known = dict(known)
        self.lookups: list[list[str]] = []
        self.inserted: list[tuple[CatalogEntry, str, str]] = []
        self.fail_insert: set[str] = set()
        self.insert_losing_race: set[str] = set()
        self.gap_changes: list[tuple[str, str | None, str]] = []

    def list_active_channels(self, conn: Any) -> list[Channel]:
        return list(self.channels)

    def known_videos(self, conn: Any, video_ids: list[str]) -> dict[str, KnownVideo]:
        self.lookups.append(list(video_ids))
        return {v: self.known[v] for v in video_ids if v in self.known}

    def set_feed_gap_reason(self, conn: Any, channel_id: str, reason: str, *, source: str) -> object | None:
        channel = next(c for c in self.channels if c.channel_id == channel_id)
        if channel.feed_gap_reason is None:
            return None
        self.gap_changes.append((channel_id, reason, source))
        return object()

    def clear_feed_gap(self, conn: Any, channel_id: str, *, source: str) -> object | None:
        channel = next(c for c in self.channels if c.channel_id == channel_id)
        if channel.feed_gap_reason is None:
            return None
        self.gap_changes.append((channel_id, None, source))
        return object()

    def insert_catalog_video(
        self, conn: Any, entry: CatalogEntry, *, channel_id: str, origin: str
    ) -> bool:
        if entry.video_id in self.fail_insert:
            raise RuntimeError("boom")
        if entry.video_id in self.insert_losing_race or entry.video_id in self.known:
            return False
        self.known[entry.video_id] = KnownVideo(channel_id, None)
        self.inserted.append((entry, channel_id, origin))
        return True


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def make_repo(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeRepo]:
    def install(
        channels: list[Channel] | None = None, known: dict[str, KnownVideo] | None = None
    ) -> FakeRepo:
        repo = FakeRepo(channels if channels is not None else [_channel()], known or {})
        monkeypatch.setattr(reconcile, "list_active_channels", repo.list_active_channels)
        monkeypatch.setattr(reconcile, "known_videos", repo.known_videos)
        monkeypatch.setattr(reconcile, "insert_catalog_video", repo.insert_catalog_video)
        monkeypatch.setattr(reconcile, "set_feed_gap_reason", repo.set_feed_gap_reason, raising=False)
        monkeypatch.setattr(reconcile, "clear_feed_gap", repo.clear_feed_gap, raising=False)
        return repo

    return install


def _anchor(channel_id: str = "UCa") -> KnownVideo:
    return KnownVideo(channel_id=channel_id, published_at=AFTER)


def _run(
    catalog: FakeCatalog,
    *,
    queue: FakeQueue | None = None,
    conn: FakeConn | None = None,
    clock: Clock | None = None,
    start_after: str | None = None,
    list_limit: int = RECONCILE_LIST_LIMIT,
) -> tuple[ReconcileResult, FakeQueue, FakeConn]:
    queue = queue or FakeQueue()
    conn = conn or FakeConn()
    result = reconcile_channels(
        conn,  # type: ignore[arg-type]
        catalog,
        queue,  # type: ignore[arg-type]
        clock=clock or Clock(),
        start_after=start_after,
        list_limit=list_limit,
    )
    return result, queue, conn


def test_the_fake_catalog_satisfies_the_catalog_port() -> None:
    catalog: CatalogSource = FakeCatalog({"UCa": []})
    assert catalog.list_uploads("UCa", limit=1).entries == ()


def test_the_named_constants() -> None:
    assert reconcile.RECONCILE_LIST_LIMIT == 50
    assert reconcile.RECONCILE_LIST_TIMEOUT_SEC == 60
    assert reconcile.RECONCILE_PASS_BUDGET_SEC == 90
    assert reconcile.RECONCILE_PRIORITY == PRIORITY_NORMAL == 0


# -- selection ------------------------------------------------------------------


def test_the_catalog_is_called_once_per_channel_with_the_list_limit(
    make_repo: Callable[..., FakeRepo],
) -> None:
    make_repo([_channel("UCa"), _channel("UCb")])
    catalog = FakeCatalog({"UCa": [], "UCb": []})

    _run(catalog)

    assert catalog.calls == [("UCa", RECONCILE_LIST_LIMIT), ("UCb", RECONCILE_LIST_LIMIT)]


def test_an_anchor_in_the_middle_recovers_exactly_the_unknown_entries_above_it(
    make_repo: Callable[..., FakeRepo],
) -> None:
    # newest first: 1, 2 unknown; 3 anchor; 4, 5 unknown history below it
    repo = make_repo(known={_e(3).video_id: _anchor()})
    catalog = FakeCatalog({"UCa": [_e(1), _e(2), _e(3), _e(4), _e(5)]})

    result, queue, _ = _run(catalog)

    assert [i[0].video_id for i in repo.inserted] == _ids(1, 2)
    assert [c[1] for c in queue.calls] == _ids(1, 2)
    assert (result.videos_recovered, result.jobs_enqueued, result.listed) == (2, 2, 5)


def test_known_entries_above_the_anchor_are_skipped_whatever_they_are(
    make_repo: Callable[..., FakeRepo],
) -> None:
    known = {
        _e(1).video_id: KnownVideo("UCa", None),  # polled, awaiting ingest
        _e(3).video_id: KnownVideo(None, None),  # a stub row
        _e(4).video_id: _anchor(),
    }
    repo = make_repo(known=known)

    result, queue, _ = _run(FakeCatalog({"UCa": [_e(1), _e(2), _e(3), _e(4)]}))

    assert [i[0].video_id for i in repo.inserted] == _ids(2)
    assert [c[1] for c in queue.calls] == _ids(2)
    assert result.videos_recovered == 1


def test_the_anchor_as_the_top_entry_leaves_nothing_to_recover(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo(known={_e(1).video_id: _anchor()})

    result, queue, _ = _run(FakeCatalog({"UCa": [_e(1), _e(2), _e(3)]}))

    assert repo.inserted == [] and queue.calls == []
    assert result.videos_recovered == 0


def test_the_anchor_as_the_last_of_fifty_entries_recovers_the_forty_nine_above_it(
    make_repo: Callable[..., FakeRepo],
) -> None:
    entries = [_e(i) for i in range(50)]
    repo = make_repo(known={entries[-1].video_id: _anchor()})

    result, queue, _ = _run(FakeCatalog({"UCa": entries}))

    assert [i[0].video_id for i in repo.inserted] == [e.video_id for e in entries[:49]]
    assert result.videos_recovered == 49 and len(queue.calls) == 49


def test_with_several_anchors_the_lowest_one_bounds_the_candidates(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo(known={_e(2).video_id: _anchor(), _e(4).video_id: _anchor()})

    _run(FakeCatalog({"UCa": [_e(1), _e(2), _e(3), _e(4), _e(5)]}))

    assert [i[0].video_id for i in repo.inserted] == _ids(1, 3)


def test_no_anchor_writes_nothing_and_logs_one_info_line(
    make_repo: Callable[..., FakeRepo], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="services.planner.reconcile")
    repo = make_repo()  # nothing known at all: unknown entries may be pre-cutoff history

    result, queue, _ = _run(FakeCatalog({"UCa": [_e(1), _e(2), _e(3)]}))

    assert repo.inserted == [] and queue.calls == []
    assert result.videos_recovered == 0
    lines = [r for r in caplog.records if r.getMessage() == "planner.reconcile.no_anchor"]
    assert [(r.levelno, r.channel_id, r.listed) for r in lines] == [  # type: ignore[attr-defined]
        (logging.INFO, "UCa", 3)
    ]


def test_an_empty_listing_has_no_anchor_and_writes_nothing(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo()

    result, queue, _ = _run(FakeCatalog({"UCa": []}))

    assert repo.inserted == [] and queue.calls == []
    assert (result.channels_checked, result.listed) == (1, 0)


def test_published_exactly_at_monitor_from_is_not_an_anchor(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo(known={_e(2).video_id: KnownVideo("UCa", CUTOFF)})

    _run(FakeCatalog({"UCa": [_e(1), _e(2)]}))

    assert repo.inserted == []


def test_published_one_second_after_monitor_from_is_an_anchor(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo(known={_e(2).video_id: KnownVideo("UCa", CUTOFF + timedelta(seconds=1))})

    _run(FakeCatalog({"UCa": [_e(1), _e(2)]}))

    assert [i[0].video_id for i in repo.inserted] == _ids(1)


def test_a_video_published_before_monitor_from_is_not_an_anchor(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo(known={_e(2).video_id: KnownVideo("UCa", BEFORE)})

    _run(FakeCatalog({"UCa": [_e(1), _e(2)]}))

    assert repo.inserted == []


def test_a_known_video_of_another_channel_is_not_an_anchor(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo(known={_e(2).video_id: _anchor("UCother")})

    result, _, _ = _run(FakeCatalog({"UCa": [_e(1), _e(2)]}))

    assert repo.inserted == []
    assert result.videos_recovered == 0


def test_a_known_video_with_null_published_at_is_not_an_anchor(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo(known={_e(2).video_id: KnownVideo("UCa", None)})

    _run(FakeCatalog({"UCa": [_e(1), _e(2)]}))

    assert repo.inserted == []


def test_the_anchor_test_uses_the_channels_own_monitor_from(
    make_repo: Callable[..., FakeRepo],
) -> None:
    published = CUTOFF + timedelta(hours=1)
    repo = make_repo(
        [_channel("UCa", monitor_from=CUTOFF), _channel("UCb", monitor_from=CUTOFF + timedelta(days=2))],
        known={_e(2).video_id: KnownVideo("UCa", published), _e(12).video_id: KnownVideo("UCb", published)},
    )

    _run(FakeCatalog({"UCa": [_e(1), _e(2)], "UCb": [_e(11), _e(12)]}))

    assert [(i[0].video_id, i[1]) for i in repo.inserted] == [(_e(1).video_id, "UCa")]


def test_one_lookup_per_channel_covers_every_listed_id(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo([_channel("UCa"), _channel("UCb")])

    _run(FakeCatalog({"UCa": [_e(1), _e(2), _e(3)], "UCb": [_e(4), _e(5)]}))

    assert repo.lookups == [_ids(1, 2, 3), _ids(4, 5)]


def test_a_video_listed_twice_is_inserted_once(make_repo: Callable[..., FakeRepo]) -> None:
    repo = make_repo(known={_e(3).video_id: _anchor()})

    _run(FakeCatalog({"UCa": [_e(1), _e(1), _e(3)]}))

    assert [i[0].video_id for i in repo.inserted] == _ids(1)


def test_marked_channel_with_no_anchor_becomes_no_anchor(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo([_channel(gap_reason="suspected")])

    _run(FakeCatalog({"UCa": [_e(1), _e(2)]}), list_limit=3)

    assert repo.gap_changes == [("UCa", "no_anchor", "reconcile")]


def test_unmarked_channel_with_no_anchor_stays_unmarked(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo([_channel()])

    _run(FakeCatalog({"UCa": [_e(1), _e(2)]}), list_limit=3)

    assert repo.gap_changes == []


def test_full_unique_listing_with_lowest_anchor_last_marks_window_exhausted(
    make_repo: Callable[..., FakeRepo],
) -> None:
    entries = [_e(1), _e(2), _e(3)]
    repo = make_repo([_channel(gap_reason="suspected")], {entries[-1].video_id: _anchor()})

    _run(FakeCatalog({"UCa": entries}), list_limit=3)

    assert [item[0].video_id for item in repo.inserted] == _ids(1, 2)
    assert repo.gap_changes == [("UCa", "listing_window_exhausted", "reconcile")]


@pytest.mark.parametrize(
    ("entries", "anchor_index", "limit"),
    [
        ([_e(1), _e(2), _e(3)], 1, 3),
        ([_e(1), _e(2)], 1, 3),
    ],
)
def test_safe_anchor_boundaries_clear_after_candidate_writes(
    make_repo: Callable[..., FakeRepo], entries: list[CatalogEntry], anchor_index: int, limit: int
) -> None:
    repo = make_repo([_channel(gap_reason="suspected")], {entries[anchor_index].video_id: _anchor()})

    _run(FakeCatalog({"UCa": entries}), list_limit=limit)

    assert repo.gap_changes == [("UCa", None, "reconcile")]


def test_candidate_write_failure_does_not_change_gap_state(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo([_channel(gap_reason="suspected")], {_e(3).video_id: _anchor()})
    repo.fail_insert = {_e(2).video_id}

    _run(FakeCatalog({"UCa": [_e(1), _e(2), _e(3)]}), list_limit=3)

    assert repo.gap_changes == []


# -- writes ---------------------------------------------------------------------


def test_a_candidate_is_inserted_as_rss_and_queued_as_a_plain_normal_ingest(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo(known={_e(2).video_id: _anchor()})

    _, queue, _ = _run(FakeCatalog({"UCa": [_e(1), _e(2)]}))

    assert repo.inserted == [(_e(1), "UCa", "rss")]
    assert queue.calls == [("ingest", _e(1).video_id, {"priority": 0})]


def test_candidate_writes_and_gap_state_share_one_channel_transaction(
    make_repo: Callable[..., FakeRepo],
) -> None:
    make_repo([_channel(gap_reason="suspected")], known={_e(3).video_id: _anchor()})

    _, _, conn = _run(FakeCatalog({"UCa": [_e(1), _e(2), _e(3)]}))

    begins = [i for i, event in enumerate(conn.events) if event == "begin"]
    assert len(begins) == 1
    assert conn.events[begins[0] + 1] == "commit"


def test_nothing_is_left_open_across_a_catalog_call(make_repo: Callable[..., FakeRepo]) -> None:
    make_repo([_channel("UCa"), _channel("UCb")])
    conn = FakeConn()
    seen: list[list[str]] = []
    catalog = FakeCatalog(
        {"UCa": [], "UCb": []}, on_call=lambda c: seen.append(list(conn.events))
    )

    _run(catalog, conn=conn)

    # The channel list is read before the first listing; it must already be committed.
    assert seen[0] == ["commit"]
    # Between listings every write block has ended.
    assert all(e.count("begin") <= e.count("commit") for e in seen)


def test_a_video_that_polling_inserted_first_gets_no_second_job(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo(known={_e(3).video_id: _anchor()})
    repo.insert_losing_race = {_e(1).video_id}

    result, queue, _ = _run(FakeCatalog({"UCa": [_e(1), _e(2), _e(3)]}))

    assert [c[1] for c in queue.calls] == _ids(2)
    assert (result.videos_recovered, result.jobs_enqueued) == (1, 1)


def test_a_deduplicated_job_counts_as_recovered_but_not_enqueued(
    make_repo: Callable[..., FakeRepo],
) -> None:
    make_repo(known={_e(2).video_id: _anchor()})

    result, _, _ = _run(FakeCatalog({"UCa": [_e(1), _e(2)]}), queue=FakeQueue({_e(1).video_id}))

    assert (result.videos_recovered, result.jobs_enqueued) == (1, 0)


def test_a_second_pass_over_the_same_catalog_writes_nothing(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo(known={_e(3).video_id: _anchor()})
    catalog = FakeCatalog({"UCa": [_e(1), _e(2), _e(3)]})
    queue = FakeQueue()

    first, _, _ = _run(catalog, queue=queue)
    second, _, _ = _run(catalog, queue=queue)

    assert (first.videos_recovered, second.videos_recovered, second.jobs_enqueued) == (2, 0, 0)
    assert len(repo.inserted) == 2 and len(queue.calls) == 2


# -- failures -------------------------------------------------------------------


def _two_channels(
    make_repo: Callable[..., FakeRepo], first: BaseException
) -> tuple[FakeRepo, FakeCatalog]:
    repo = make_repo(
        [_channel("UCa"), _channel("UCb")], known={_e(12).video_id: _anchor("UCb")}
    )
    return repo, FakeCatalog({"UCa": first, "UCb": [_e(11), _e(12)]})


@pytest.mark.parametrize("reason", list(UnavailableReason))
def test_a_permanent_source_error_writes_nothing_logs_info_and_continues(
    make_repo: Callable[..., FakeRepo], caplog: pytest.LogCaptureFixture, reason: UnavailableReason
) -> None:
    caplog.set_level(logging.INFO, logger="services.planner.reconcile")
    repo, catalog = _two_channels(make_repo, PermanentSourceError(reason, "x"))

    result, _, _ = _run(catalog)

    assert catalog.listed == ["UCa", "UCb"]
    assert [i[1] for i in repo.inserted] == ["UCb"]
    assert (result.channels_failed, result.channels_checked) == (0, 2)
    line = next(r for r in caplog.records if r.levelno == logging.INFO and getattr(r, "channel_id", None) == "UCa")
    assert reason.value in str(line.reason)  # type: ignore[attr-defined]
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


@pytest.mark.parametrize(
    "exc",
    [
        TransientNetworkError("timeout"),
        ToolFailureError("yt-dlp exploded"),
        RuntimeError("anything else"),
        ValueError("bad listing"),
    ],
)
def test_other_errors_warn_roll_back_and_the_next_channel_still_runs(
    make_repo: Callable[..., FakeRepo], caplog: pytest.LogCaptureFixture, exc: Exception
) -> None:
    caplog.set_level(logging.WARNING, logger="services.planner.reconcile")
    repo, catalog = _two_channels(make_repo, exc)

    result, _, conn = _run(catalog)

    assert catalog.listed == ["UCa", "UCb"]
    assert [i[1] for i in repo.inserted] == ["UCb"]
    assert (result.channels_failed, result.channels_checked) == (1, 1)
    assert "rollback" in conn.events
    (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning.channel_id == "UCa"  # type: ignore[attr-defined]
    assert warning.error_class == type(exc).__name__  # type: ignore[attr-defined]


def test_a_write_error_rolls_back_the_channel_and_the_next_channel_still_runs(
    make_repo: Callable[..., FakeRepo],
) -> None:
    repo = make_repo(
        [_channel("UCa"), _channel("UCb")],
        known={_e(3).video_id: _anchor("UCa"), _e(13).video_id: _anchor("UCb")},
    )
    repo.fail_insert = {_e(2).video_id}
    catalog = FakeCatalog({"UCa": [_e(1), _e(2), _e(3)], "UCb": [_e(11), _e(13)]})

    result, queue, conn = _run(catalog)

    assert [i[0].video_id for i in repo.inserted] == [_e(1).video_id, _e(11).video_id]
    assert [c[1] for c in queue.calls] == [_e(1).video_id, _e(11).video_id]
    assert result.channels_failed == 1
    assert "rollback" in conn.events
    assert result.videos_recovered == 1 and result.jobs_enqueued == 1


def test_a_rate_limit_stops_the_pass_before_listing_the_remaining_channels(
    make_repo: Callable[..., FakeRepo], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="services.planner.reconcile")
    make_repo([_channel("UCa"), _channel("UCb"), _channel("UCc")])
    catalog = FakeCatalog({"UCa": [], "UCb": RateLimitedError("429"), "UCc": []})

    result, _, _ = _run(catalog)

    assert catalog.listed == ["UCa", "UCb"]
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert (result.channels_checked, result.channels_failed, result.channels_deferred) == (1, 1, 1)
    assert result.resume_after == "UCa"  # the rate-limited channel's predecessor


def test_a_rate_limit_on_the_first_channel_visited_leaves_the_cursor_where_it_was(
    make_repo: Callable[..., FakeRepo],
) -> None:
    make_repo([_channel("UCa"), _channel("UCb")])

    fresh, _, _ = _run(FakeCatalog({"UCa": RateLimitedError("429"), "UCb": []}))
    resumed, _, _ = _run(
        FakeCatalog({"UCa": [], "UCb": RateLimitedError("429")}), start_after="UCa"
    )

    assert fresh.resume_after is None
    assert resumed.resume_after == "UCa"  # UCb was first and rate-limited: nothing was finished


def test_the_next_run_resumes_at_the_rate_limited_channel(
    make_repo: Callable[..., FakeRepo],
) -> None:
    make_repo([_channel("UCa"), _channel("UCb"), _channel("UCc")])
    catalog = FakeCatalog({"UCa": [], "UCb": RateLimitedError("429"), "UCc": []})
    first, _, _ = _run(catalog)
    catalog.listings["UCb"] = []
    catalog.calls.clear()

    second, _, _ = _run(catalog, start_after=first.resume_after)

    assert catalog.listed == ["UCb", "UCc", "UCa"]
    assert second.channels_deferred == 0


def test_a_database_connection_error_propagates(make_repo: Callable[..., FakeRepo]) -> None:
    make_repo([_channel("UCa"), _channel("UCb")])
    catalog = FakeCatalog({"UCa": psycopg.OperationalError("server closed"), "UCb": []})

    with pytest.raises(psycopg.OperationalError):
        _run(catalog)

    assert catalog.listed == ["UCa"]


def test_no_channels_lists_nothing(make_repo: Callable[..., FakeRepo]) -> None:
    make_repo([])
    catalog = FakeCatalog({})

    result, _, _ = _run(catalog, start_after="UCz")

    assert catalog.calls == []
    assert result == ReconcileResult(0, 0, 0, 0, 0, 0, "UCz")


# -- bounded run time and round-robin ----------------------------------------------


def _five(make_repo: Callable[..., FakeRepo]) -> list[str]:
    names = ["UCe", "UCa", "UCd", "UCb", "UCc"]  # list order is the repo's, unsorted on purpose
    make_repo([_channel(n) for n in names])
    return sorted(names)


def _slow_catalog(clock: Clock, step: float) -> FakeCatalog:
    def tick(channel_id: str) -> None:
        clock.now += step

    return FakeCatalog({n: [] for n in ["UCa", "UCb", "UCc", "UCd", "UCe"]}, on_call=tick)


def test_channels_are_visited_in_channel_id_order(make_repo: Callable[..., FakeRepo]) -> None:
    names = _five(make_repo)
    clock = Clock()
    catalog = _slow_catalog(clock, 0)

    _run(catalog, clock=clock)

    assert catalog.listed == names == ["UCa", "UCb", "UCc", "UCd", "UCe"]


def test_no_new_channel_starts_once_the_budget_has_elapsed_and_the_rest_are_deferred(
    make_repo: Callable[..., FakeRepo],
) -> None:
    _five(make_repo)
    clock = Clock()
    catalog = _slow_catalog(clock, 40.0)

    result, _, _ = _run(catalog, clock=clock)

    # starts at t=0, 40, 80 (<90); at t=120 the budget is spent
    assert catalog.listed == ["UCa", "UCb", "UCc"]
    assert (result.channels_checked, result.channels_deferred) == (3, 2)
    assert result.resume_after == "UCc"


def test_deferred_channels_are_reconciled_first_in_the_next_run_and_the_order_wraps(
    make_repo: Callable[..., FakeRepo],
) -> None:
    _five(make_repo)
    clock = Clock()
    catalog = _slow_catalog(clock, 40.0)
    first, _, _ = _run(catalog, clock=clock)
    catalog.calls.clear()
    clock.now = 0.0

    second, _, _ = _run(catalog, clock=clock, start_after=first.resume_after)

    assert catalog.listed == ["UCd", "UCe", "UCa"]  # deferred ones first, then wrap around
    assert second.resume_after == "UCa"
    assert second.channels_deferred == 2


def test_the_budget_boundary_is_exclusive_of_a_channel_starting_at_exactly_the_budget(
    make_repo: Callable[..., FakeRepo],
) -> None:
    make_repo([_channel("UCa"), _channel("UCb")])
    clock = Clock()
    catalog = FakeCatalog({"UCa": [], "UCb": []}, on_call=lambda c: setattr(clock, "now", clock.now + RECONCILE_PASS_BUDGET_SEC))

    result, _, _ = _run(catalog, clock=clock)

    assert catalog.listed == ["UCa"]  # at t == budget no new channel starts
    assert result.channels_deferred == 1


def test_just_under_the_budget_still_starts_a_channel(make_repo: Callable[..., FakeRepo]) -> None:
    make_repo([_channel("UCa"), _channel("UCb")])
    clock = Clock()
    catalog = FakeCatalog(
        {"UCa": [], "UCb": []},
        on_call=lambda c: setattr(clock, "now", clock.now + RECONCILE_PASS_BUDGET_SEC - 1),
    )

    result, _, _ = _run(catalog, clock=clock)

    assert catalog.listed == ["UCa", "UCb"]
    assert result.channels_deferred == 0


def test_the_pass_budget_counts_from_the_start_of_the_pass_not_from_clock_zero(
    make_repo: Callable[..., FakeRepo],
) -> None:
    make_repo([_channel("UCa"), _channel("UCb")])
    clock = Clock()
    clock.now = 10_000.0
    catalog = FakeCatalog({"UCa": [], "UCb": []})

    result, _, _ = _run(catalog, clock=clock)

    assert catalog.listed == ["UCa", "UCb"] and result.channels_deferred == 0


def test_each_channel_is_visited_at_most_once_per_pass(
    make_repo: Callable[..., FakeRepo],
) -> None:
    make_repo([_channel("UCa"), _channel("UCb")])
    catalog = FakeCatalog({"UCa": [], "UCb": []})

    _run(catalog, start_after="UCa")

    assert catalog.listed == ["UCb", "UCa"]


@pytest.mark.parametrize(
    ("start_after", "expected"),
    [
        (None, ["UCa", "UCb", "UCc"]),
        ("UCa", ["UCb", "UCc", "UCa"]),
        ("UCb", ["UCc", "UCa", "UCb"]),
        ("UCc", ["UCa", "UCb", "UCc"]),  # a full lap: wraps to the start
        ("UCbb", ["UCc", "UCa", "UCb"]),  # a cursor that is no longer a channel
        ("UCzz", ["UCa", "UCb", "UCc"]),
        ("", ["UCa", "UCb", "UCc"]),
    ],
)
def test_the_visiting_order_for_a_cursor(
    make_repo: Callable[..., FakeRepo], start_after: str | None, expected: list[str]
) -> None:
    make_repo([_channel("UCc"), _channel("UCa"), _channel("UCb")])
    catalog = FakeCatalog({"UCa": [], "UCb": [], "UCc": []})

    result, _, _ = _run(catalog, start_after=start_after)

    assert catalog.listed == expected
    assert result.resume_after == expected[-1]


def test_a_failed_channel_still_advances_the_cursor(make_repo: Callable[..., FakeRepo]) -> None:
    make_repo([_channel("UCa"), _channel("UCb")])
    catalog = FakeCatalog({"UCa": [], "UCb": ToolFailureError("x")})

    result, _, _ = _run(catalog)

    assert result.resume_after == "UCb"


# -- result and logging --------------------------------------------------------------


def test_the_result_is_a_frozen_dataclass_with_the_documented_fields(
    make_repo: Callable[..., FakeRepo],
) -> None:
    make_repo(known={_e(2).video_id: _anchor()})

    result, _, _ = _run(FakeCatalog({"UCa": [_e(1), _e(2)]}))

    assert result == ReconcileResult(
        channels_checked=1,
        channels_failed=0,
        channels_deferred=0,
        listed=2,
        videos_recovered=1,
        jobs_enqueued=1,
        resume_after="UCa",
    )
    with pytest.raises(FrozenInstanceError):
        result.listed = 0  # type: ignore[misc]


def test_a_recovering_channel_logs_one_info_line_and_the_pass_one_summary_without_titles(
    make_repo: Callable[..., FakeRepo], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="services.planner.reconcile")
    make_repo(
        [_channel("UCa"), _channel("UCb")], known={_e(3).video_id: _anchor("UCa")}
    )
    clock = Clock()
    clock.now = 100.0  # a monotonic clock does not start at zero
    catalog = FakeCatalog(
        {"UCa": [_e(1), _e(2), _e(3)], "UCb": []},
        on_call=lambda c: setattr(clock, "now", clock.now + 0.25),
    )

    _run(catalog, clock=clock)

    recovered = [r for r in caplog.records if r.getMessage() == "planner.reconcile.recovered"]
    assert [(r.levelno, r.channel_id, r.count) for r in recovered] == [  # type: ignore[attr-defined]
        (logging.INFO, "UCa", 2)
    ]
    (summary,) = [r for r in caplog.records if r.getMessage() == "planner.reconcile"]
    assert summary.levelno == logging.INFO
    fields = {k: v for k, v in vars(summary).items() if k in {
        "channels_checked", "channels_failed", "channels_deferred", "listed",
        "videos_recovered", "jobs_enqueued", "resume_after", "duration_ms"}}
    assert fields == {
        "channels_checked": 2, "channels_failed": 0, "channels_deferred": 0, "listed": 3,
        "videos_recovered": 2, "jobs_enqueued": 2, "resume_after": "UCb", "duration_ms": 500,
    }
    assert "Title" not in " ".join(str(vars(r)) for r in caplog.records)


def test_the_pass_does_not_sleep() -> None:
    tree = ast.parse(inspect.getsource(reconcile))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}

    assert "sleep" not in names
