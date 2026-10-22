"""Tests for common/repo/channels.py (issue #14)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from common.repo import channels as channels_repo
from common.repo.channels import (
    PollHealth,
    add_channel,
    channel_exists,
    get_channel_title,
    list_active_channels,
    list_channels,
    record_poll,
    register_channel,
)

pytestmark = pytest.mark.integration


def test_add_channel_creates_an_active_channel_with_monitor_from_set(
    conn: psycopg.Connection,
) -> None:
    add_channel(conn, "UC123", "Some Channel")

    [channel] = list_active_channels(conn)
    assert channel.channel_id == "UC123"
    assert channel.title == "Some Channel"
    assert channel.active is True
    assert channel.monitor_from is not None


def test_add_channel_again_for_an_active_channel_changes_nothing(
    conn: psycopg.Connection,
) -> None:
    add_channel(conn, "UC123", "Some Channel")
    [before] = list_active_channels(conn)

    add_channel(conn, "UC123", "Renamed")

    [after] = list_active_channels(conn)
    assert after == before  # not even monitor_from changed


def test_list_active_channels_excludes_inactive_channels(conn: psycopg.Connection) -> None:
    add_channel(conn, "UC1", "One")
    conn.execute("UPDATE channels SET active = false WHERE channel_id = %s", ("UC1",))
    add_channel(conn, "UC2", "Two")

    active_ids = {c.channel_id for c in list_active_channels(conn)}

    assert active_ids == {"UC2"}


def test_list_active_channels_returns_empty_list_when_none_exist(
    conn: psycopg.Connection,
) -> None:
    assert list_active_channels(conn) == []


def test_record_poll_sets_last_polled_and_error(conn: psycopg.Connection) -> None:
    add_channel(conn, "UC1", "One")

    record_poll(conn, "UC1", error="boom")

    row = conn.execute(
        "SELECT last_polled, last_poll_err FROM channels WHERE channel_id = %s", ("UC1",)
    ).fetchone()
    assert row is not None
    last_polled, last_poll_err = row
    assert last_polled is not None
    assert last_poll_err == "boom"


def test_record_poll_with_no_error_clears_the_old_one(conn: psycopg.Connection) -> None:
    add_channel(conn, "UC1", "One")
    record_poll(conn, "UC1", error="boom")

    record_poll(conn, "UC1")

    row = conn.execute(
        "SELECT last_poll_err FROM channels WHERE channel_id = %s", ("UC1",)
    ).fetchone()
    assert row == (None,)


def test_record_poll_strips_nul_from_error(conn: psycopg.Connection) -> None:
    add_channel(conn, "UC1", "One")

    record_poll(conn, "UC1", error="bad\x00error")

    row = conn.execute(
        "SELECT last_poll_err FROM channels WHERE channel_id = %s", ("UC1",)
    ).fetchone()
    assert row == ("baderror",)


def test_add_channel_strips_nul_from_title(conn: psycopg.Connection) -> None:
    add_channel(conn, "UC1", "bad\x00title")

    [channel] = list_active_channels(conn)
    assert channel.title == "badtitle"


def _row(conn: psycopg.Connection, channel_id: str) -> tuple[object, ...] | None:
    return conn.execute(
        """
        SELECT channel_id, title, active, monitor_from, last_polled, last_poll_err, added_at
        FROM channels WHERE channel_id = %s
        """,
        (channel_id,),
    ).fetchone()


def test_register_channel_creates_an_active_channel_from_now(conn: psycopg.Connection) -> None:
    result = register_channel(conn, "UC1")

    [now] = conn.execute("SELECT now()").fetchone() or ()
    assert result.created is True
    assert result.channel.channel_id == "UC1"
    assert result.channel.active is True
    assert result.channel.monitor_from == now
    assert result.channel.added_at == now
    assert _row(conn, "UC1") == (
        "UC1", None, True, now, None, None, now,
    )


def test_register_channel_activates_an_inactive_channel_from_now(
    conn: psycopg.Connection,
) -> None:
    old = datetime(2020, 1, 1, tzinfo=UTC)
    conn.execute(
        """
        INSERT INTO channels (channel_id, title, active, monitor_from, added_at)
        VALUES ('UC1', 'Kept title', false, %s, %s)
        """,
        (old, old),
    )
    conn.commit()

    result = register_channel(conn, "UC1")

    [now] = conn.execute("SELECT now()").fetchone() or ()
    assert result.created is False
    assert _row(conn, "UC1") == ("UC1", "Kept title", True, now, None, None, old)
    assert result.channel.monitor_from == now
    assert result.channel.added_at == old


def test_register_channel_changes_nothing_for_an_active_channel(
    conn: psycopg.Connection,
) -> None:
    old = datetime(2020, 1, 1, tzinfo=UTC)
    polled = datetime(2021, 1, 1, tzinfo=UTC)
    conn.execute(
        """
        INSERT INTO channels (channel_id, title, active, monitor_from, last_polled, added_at)
        VALUES ('UC1', 'T', true, %s, %s, %s)
        """,
        (old, polled, old),
    )
    conn.commit()
    before = _row(conn, "UC1")

    result = register_channel(conn, "UC1")

    assert result.created is False
    assert _row(conn, "UC1") == before
    assert result.channel.monitor_from == old


def test_add_channel_docstring_no_longer_defers_reactivation_to_40() -> None:
    assert "moved to #40" not in (add_channel.__doc__ or "")


def test_channel_exists_for_active_and_inactive_rows_only(conn: psycopg.Connection) -> None:
    conn.execute("INSERT INTO channels (channel_id, active) VALUES ('UCon', true), ('UCoff', false)")

    assert channel_exists(conn, "UCon") is True
    assert channel_exists(conn, "UCoff") is True
    assert channel_exists(conn, "UCnone") is False


def test_get_channel_title_returns_the_stored_title(conn: psycopg.Connection) -> None:
    add_channel(conn, "UC123", "Some Channel")

    assert get_channel_title(conn, "UC123") == "Some Channel"


def test_get_channel_title_is_none_for_unknown_empty_or_untitled(
    conn: psycopg.Connection,
) -> None:
    add_channel(conn, "UC123", None)

    assert get_channel_title(conn, "UC123") is None
    assert get_channel_title(conn, "UCunknown") is None
    assert get_channel_title(conn, "") is None


# -- failure run (#90) ----------------------------------------------------------


def _run_columns(conn: psycopg.Connection) -> tuple[Any, ...]:
    conn.commit()  # a new transaction, so the next record_poll has a later now()
    row = conn.execute(
        "SELECT poll_failing_since, poll_failures, last_polled, last_poll_err, now()"
        " FROM channels WHERE channel_id = 'UC1'"
    ).fetchone()
    assert row is not None
    return tuple(row)


def test_first_failure_starts_a_run_at_the_same_now_as_last_polled(
    conn: psycopg.Connection,
) -> None:
    add_channel(conn, "UC1", "One")

    health = record_poll(conn, "UC1", error="A: x")

    since, failures, last_polled, err, db_now = _run_columns(conn)
    assert (failures, err) == (1, "A: x")
    assert since == last_polled
    assert since <= db_now
    assert health == PollHealth(since, 1)


def test_second_and_third_failure_keep_the_start_and_count_up(conn: psycopg.Connection) -> None:
    add_channel(conn, "UC1", "One")
    record_poll(conn, "UC1", error="A: first")
    start = _run_columns(conn)[0]

    for count, err in ((2, "B: second"), (3, "C: third")):
        health = record_poll(conn, "UC1", error=err)
        since, failures, last_polled, stored_err, db_now = _run_columns(conn)
        assert since == start
        assert failures == count
        assert stored_err == err  # replaced by the newest error
        assert start < last_polled <= db_now
        assert health == PollHealth(start, count)


def test_success_after_failures_resets_everything(conn: psycopg.Connection) -> None:
    add_channel(conn, "UC1", "One")
    record_poll(conn, "UC1", error="A: x")
    record_poll(conn, "UC1", error="A: x")

    health = record_poll(conn, "UC1")

    assert _run_columns(conn)[:2] == (None, 0)
    assert conn.execute("SELECT last_poll_err FROM channels").fetchone() == (None,)
    assert health == PollHealth(None, 0)


def test_failure_after_a_success_starts_a_new_run(conn: psycopg.Connection) -> None:
    add_channel(conn, "UC1", "One")
    record_poll(conn, "UC1", error="A: x")
    old_start = _run_columns(conn)[0]
    record_poll(conn, "UC1")
    conn.commit()

    record_poll(conn, "UC1", error="A: again")

    since, failures, last_polled, _, _ = _run_columns(conn)
    assert failures == 1
    assert since > old_start
    assert since == last_polled


def test_success_on_a_never_failed_channel_stays_null_and_zero(conn: psycopg.Connection) -> None:
    add_channel(conn, "UC1", "One")

    record_poll(conn, "UC1")

    assert _run_columns(conn)[:2] == (None, 0)


def test_record_poll_for_an_unknown_channel_returns_none(conn: psycopg.Connection) -> None:
    assert record_poll(conn, "UCnope", error="x") is None


def test_list_active_channels_fills_the_failure_run(conn: psycopg.Connection) -> None:
    add_channel(conn, "UC1", "One")
    record_poll(conn, "UC1", error="A: x")
    record_poll(conn, "UC1", error="A: x")

    [channel] = list_active_channels(conn)

    assert channel.poll_failures == 2
    assert channel.poll_failing_since is not None


def _gap(conn: psycopg.Connection[Any], channel_id: str = "UC1") -> tuple[Any, Any]:
    row = conn.execute(
        "SELECT feed_gap_detected_at, feed_gap_reason FROM channels WHERE channel_id = %s",
        (channel_id,),
    ).fetchone()
    assert row is not None
    return row[0], row[1]


def test_mark_suspected_sets_once_and_repeated_detection_preserves_timestamp(
    conn: psycopg.Connection[Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="common.repo.channels")
    add_channel(conn, "UC1", "One")

    first = channels_repo.mark_feed_gap_suspected(conn, "UC1", source="poll")
    conn.commit()
    second = channels_repo.mark_feed_gap_suspected(conn, "UC1", source="poll")

    assert first is not None and second is not None
    assert first.detected_at == second.detected_at
    assert _gap(conn) == (first.detected_at, "suspected")
    transitions = [r.msg for r in caplog.records if isinstance(r.msg, dict) and r.msg.get("event") == "channel.feed_gap_state_changed"]
    assert transitions == [{
        "channel_id": "UC1",
        "previous_reason": None,
        "reason": "suspected",
        "source": "poll",
        "event": "channel.feed_gap_state_changed",
        "logger": "common.repo.channels",
        "level": "info",
        "timestamp": transitions[0]["timestamp"],
    }]


def test_classify_and_clear_only_change_an_existing_marker(
    conn: psycopg.Connection[Any]
) -> None:
    add_channel(conn, "UC1", "One")
    add_channel(conn, "UC2", "Two")
    assert channels_repo.set_feed_gap_reason(conn, "UC1", "no_anchor", source="reconcile") is None
    assert channels_repo.clear_feed_gap(conn, "UC1", source="reconcile") is None

    marked = channels_repo.mark_feed_gap_suspected(conn, "UC2", source="poll")
    assert marked is not None
    classified = channels_repo.set_feed_gap_reason(conn, "UC2", "no_anchor", source="reconcile")
    assert classified is not None
    assert classified.detected_at == marked.detected_at
    cleared = channels_repo.clear_feed_gap(conn, "UC2", source="reconcile")
    assert cleared is not None
    assert _gap(conn, "UC2") == (None, None)


def test_same_reason_is_idempotent_and_emits_no_second_transition(
    conn: psycopg.Connection[Any], caplog: pytest.LogCaptureFixture
) -> None:
    add_channel(conn, "UC1", "One")
    channels_repo.mark_feed_gap_suspected(conn, "UC1", source="poll")
    caplog.clear()

    unchanged = channels_repo.set_feed_gap_reason(conn, "UC1", "suspected", source="reconcile")

    assert unchanged is not None
    assert not [r for r in caplog.records if isinstance(r.msg, dict) and r.msg.get("event") == "channel.feed_gap_state_changed"]


def test_ops_channels_filter_and_total_order(conn: psycopg.Connection[Any]) -> None:
    for channel_id, title in (("UCfail", "Zulu"), ("UCgap2", "Bravo"), ("UCgap1", "Alpha"), ("UCok", None)):
        add_channel(conn, channel_id, title)
    conn.execute(
        "UPDATE channels SET poll_failing_since = '2026-10-03T00:00:00Z', poll_failures = 2, last_poll_err = 'x'"
        " WHERE channel_id = 'UCfail'"
    )
    conn.execute(
        "UPDATE channels SET feed_gap_detected_at = '2026-10-02T00:00:00Z', feed_gap_reason = 'no_anchor'"
        " WHERE channel_id = 'UCgap2'"
    )
    conn.execute(
        "UPDATE channels SET feed_gap_detected_at = '2026-10-01T00:00:00Z', feed_gap_reason = 'suspected'"
        " WHERE channel_id = 'UCgap1'"
    )

    attention = channels_repo.list_ops_channels(conn, failing=False, attention=True)
    all_rows = channels_repo.list_ops_channels(conn, failing=False, attention=False)
    failing = channels_repo.list_ops_channels(conn, failing=True, attention=False)

    assert [row.channel_id for row in attention] == ["UCfail", "UCgap1", "UCgap2"]
    assert [row.channel_id for row in all_rows] == ["UCfail", "UCgap1", "UCgap2", "UCok"]
    assert [row.channel_id for row in failing] == ["UCfail"]


def test_feed_gap_counts_is_zero_filled_from_one_aggregate(conn: psycopg.Connection[Any]) -> None:
    assert "conn.execute" in __import__("inspect").getsource(channels_repo.feed_gap_counts)
    for channel_id, reason in (("UC1", "suspected"), ("UC2", "suspected"), ("UC3", "no_anchor")):
        add_channel(conn, channel_id, None)
        conn.execute(
            "UPDATE channels SET feed_gap_detected_at = now(), feed_gap_reason = %s WHERE channel_id = %s",
            (reason, channel_id),
        )

    assert channels_repo.feed_gap_counts(conn) == {
        "suspected": 2,
        "no_anchor": 1,
        "listing_window_exhausted": 0,
    }


# --- list_channels (issue #119) -------------------------------------------------


def _seed_channel(
    conn: psycopg.Connection[Any],
    channel_id: str,
    title: str | None,
    *,
    active: bool = True,
) -> None:
    conn.execute(
        "INSERT INTO channels (channel_id, title, active, added_at)"
        " VALUES (%s, %s, %s, %s)",
        (channel_id, title, active, datetime(2026, 1, 1, tzinfo=UTC)),
    )


def _seed_video(conn: psycopg.Connection[Any], video_id: str, channel_id: str | None) -> None:
    conn.execute(
        "INSERT INTO videos (video_id, channel_id, discovered_at) VALUES (%s, %s, %s)",
        (video_id, channel_id, datetime(2026, 1, 1, tzinfo=UTC)),
    )


def test_list_channels_on_an_empty_table(conn: psycopg.Connection) -> None:
    page = list_channels(conn, offset=0, limit=200)

    assert (page.items, page.total) == ((), 0)


def test_list_channels_lists_every_row_with_its_video_count(conn: psycopg.Connection) -> None:
    _seed_channel(conn, "UCa", "Alpha")
    _seed_channel(conn, "UCb", "Bravo", active=False)
    _seed_channel(conn, "UCc", None, active=False)  # an ad-hoc row: no title
    _seed_channel(conn, "UCd", "Delta")  # registered, no videos yet
    for n, channel_id in enumerate(["UCa", "UCa", "UCb", "UCc", None]):
        _seed_video(conn, f"v{n:010d}", channel_id)  # NULL channel_id counts nowhere

    page = list_channels(conn, offset=0, limit=200)

    assert page.total == 4
    assert [(c.channel_id, c.title, c.active, c.video_count) for c in page.items] == [
        ("UCa", "Alpha", True, 2),
        ("UCb", "Bravo", False, 1),
        ("UCd", "Delta", True, 0),
        ("UCc", None, False, 1),
    ]


def test_list_channels_orders_untitled_last_and_breaks_ties_by_id(
    conn: psycopg.Connection,
) -> None:
    _seed_channel(conn, "UC3", "Same")
    _seed_channel(conn, "UC1", "Same")
    _seed_channel(conn, "UC9", None)
    _seed_channel(conn, "UC8", None)

    page = list_channels(conn, offset=0, limit=200)

    assert [c.channel_id for c in page.items] == ["UC1", "UC3", "UC8", "UC9"]


def test_list_channels_page_boundaries(conn: psycopg.Connection) -> None:
    for n, letter in enumerate("ABCDE"):
        _seed_channel(conn, f"UC{n}", letter)

    full = list_channels(conn, offset=0, limit=5)
    first = list_channels(conn, offset=0, limit=4)
    second = list_channels(conn, offset=4, limit=4)
    beyond = list_channels(conn, offset=5, limit=4)

    assert (len(full.items), full.total) == (5, 5)
    assert ([c.title for c in first.items], first.total) == (list("ABCD"), 5)
    assert ([c.title for c in second.items], second.total) == (["E"], 5)
    assert (beyond.items, beyond.total) == ((), 5)


def test_walking_pages_returns_every_channel_exactly_once(conn: psycopg.Connection) -> None:
    for channel_id, title in [
        ("UC1", "Same"), ("UC2", "Same"), ("UC3", "Other"),
        ("UC4", None), ("UC5", None), ("UC6", None),
    ]:
        _seed_channel(conn, channel_id, title)

    walked = [
        c.channel_id
        for offset in range(0, 8, 2)
        for c in list_channels(conn, offset=offset, limit=2).items
    ]

    assert walked == ["UC3", "UC1", "UC2", "UC4", "UC5", "UC6"]
