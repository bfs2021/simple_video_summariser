"""Tests for common/repo/channels.py (issue #14)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from common.repo.channels import (
    PollHealth,
    add_channel,
    channel_exists,
    get_channel_title,
    list_active_channels,
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
