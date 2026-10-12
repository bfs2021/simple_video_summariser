"""Tests for common/repo/videos.py (issue #14)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from common.models import CatalogEntry, FeedEntry, VideoMeta
from common.repo.channels import add_channel, list_active_channels
from common.repo.videos import (
    clear_unavailable,
    insert_backfill_video,
    insert_discovered_video,
    insert_submitted_video,
    known_video_ids,
    mark_unavailable,
    record_unavailable,
    upsert_video,
    video_exists,
)

pytestmark = pytest.mark.integration


def make_meta(**overrides: object) -> VideoMeta:
    defaults: dict[str, object] = {
        "video_id": "abc12345678",
        "channel_id": "UCabc",
        "title": "Title",
        "description": "Desc",
        "duration_sec": 120,
        "published_at": datetime(2026, 1, 1, tzinfo=UTC),
        "language": "en",
        "live_status": None,
        "manual_subtitle_langs": (),
        "auto_caption_langs": (),
    }
    defaults.update(overrides)
    return VideoMeta(**defaults)  # type: ignore[arg-type]


def test_upsert_video_inserts_a_new_video(conn: psycopg.Connection) -> None:
    upsert_video(conn, make_meta(), origin="adhoc")

    row = conn.execute(
        """
        SELECT video_id, channel_id, title, duration_sec, published_at,
               description, origin, unavailable
        FROM videos WHERE video_id = %s
        """,
        ("abc12345678",),
    ).fetchone()
    assert row == (
        "abc12345678",
        "UCabc",
        "Title",
        120,
        datetime(2026, 1, 1, tzinfo=UTC),
        "Desc",
        "adhoc",
        None,
    )


def test_upsert_video_updates_title_duration_published_and_description(
    conn: psycopg.Connection,
) -> None:
    upsert_video(conn, make_meta(), origin="adhoc")

    updated = make_meta(
        title="New title",
        duration_sec=999,
        published_at=datetime(2026, 2, 2, tzinfo=UTC),
        description="New desc",
    )
    upsert_video(conn, updated, origin="rss")

    row = conn.execute(
        "SELECT title, duration_sec, published_at, description FROM videos WHERE video_id = %s",
        ("abc12345678",),
    ).fetchone()
    assert row == (
        "New title",
        999,
        datetime(2026, 2, 2, tzinfo=UTC),
        "New desc",
    )


def test_upsert_video_never_changes_origin_or_discovered_at_on_update(
    conn: psycopg.Connection,
) -> None:
    upsert_video(conn, make_meta(), origin="adhoc")
    before = conn.execute(
        "SELECT discovered_at, origin FROM videos WHERE video_id = %s", ("abc12345678",)
    ).fetchone()

    upsert_video(conn, make_meta(title="Changed"), origin="rss")

    after = conn.execute(
        "SELECT discovered_at, origin FROM videos WHERE video_id = %s", ("abc12345678",)
    ).fetchone()
    assert after == before


def test_upsert_video_creates_an_inactive_channel_placeholder_when_missing(
    conn: psycopg.Connection,
) -> None:
    upsert_video(conn, make_meta(channel_id="UCnew"), origin="adhoc")

    row = conn.execute(
        "SELECT active FROM channels WHERE channel_id = %s", ("UCnew",)
    ).fetchone()
    assert row == (False,)
    assert list_active_channels(conn) == []  # ad-hoc submission does not start monitoring


def test_upsert_video_leaves_an_existing_channel_row_untouched(conn: psycopg.Connection) -> None:
    add_channel(conn, "UCexisting", "Existing")

    upsert_video(conn, make_meta(channel_id="UCexisting"), origin="adhoc")

    [channel] = list_active_channels(conn)
    assert channel.channel_id == "UCexisting"
    assert channel.active is True


def test_upsert_video_strips_nul_from_title_and_description(conn: psycopg.Connection) -> None:
    upsert_video(
        conn, make_meta(title="bad\x00title", description="bad\x00desc"), origin="adhoc"
    )

    row = conn.execute(
        "SELECT title, description FROM videos WHERE video_id = %s", ("abc12345678",)
    ).fetchone()
    assert row == ("badtitle", "baddesc")


@pytest.mark.parametrize("reason", ["removed", "private", "geoblocked", "agegated", "short"])
def test_mark_unavailable_accepts_each_valid_reason(
    conn: psycopg.Connection, reason: str
) -> None:
    upsert_video(conn, make_meta(), origin="adhoc")

    mark_unavailable(conn, "abc12345678", reason)

    row = conn.execute(
        "SELECT unavailable FROM videos WHERE video_id = %s", ("abc12345678",)
    ).fetchone()
    assert row == (reason,)


def test_mark_unavailable_rejects_any_other_value(conn: psycopg.Connection) -> None:
    upsert_video(conn, make_meta(), origin="adhoc")

    with pytest.raises(ValueError):
        mark_unavailable(conn, "abc12345678", "banned")


# --- clear_unavailable / record_unavailable (issue #28) -------------------------


def test_clear_unavailable_resets_the_marker_to_null(conn: psycopg.Connection) -> None:
    upsert_video(conn, make_meta(), origin="adhoc")
    mark_unavailable(conn, "abc12345678", "private")

    clear_unavailable(conn, "abc12345678")

    row = conn.execute("SELECT unavailable FROM videos WHERE video_id = 'abc12345678'").fetchone()
    assert row == (None,)


def test_clear_unavailable_on_an_unknown_video_changes_nothing(conn: psycopg.Connection) -> None:
    clear_unavailable(conn, "abc12345678")

    assert conn.execute("SELECT count(*) FROM videos").fetchone() == (0,)


def test_record_unavailable_creates_a_stub_row_without_a_channel(
    conn: psycopg.Connection,
) -> None:
    record_unavailable(conn, "abc12345678", "removed", origin="rss")

    row = conn.execute(
        "SELECT channel_id, origin, unavailable, title FROM videos WHERE video_id = 'abc12345678'"
    ).fetchone()
    assert row == (None, "rss", "removed", None)


def test_record_unavailable_updates_an_existing_row_and_keeps_its_origin(
    conn: psycopg.Connection,
) -> None:
    upsert_video(conn, make_meta(), origin="backfill")

    record_unavailable(conn, "abc12345678", "geoblocked", origin="adhoc")

    row = conn.execute(
        "SELECT origin, unavailable, title FROM videos WHERE video_id = 'abc12345678'"
    ).fetchone()
    assert row == ("backfill", "geoblocked", "Title")


def test_record_unavailable_rejects_an_unknown_reason(conn: psycopg.Connection) -> None:
    with pytest.raises(ValueError, match="unavailable reason"):
        record_unavailable(conn, "abc12345678", "sad", origin="adhoc")


def test_upsert_video_fills_a_missing_channel_on_a_stub_row(conn: psycopg.Connection) -> None:
    record_unavailable(conn, "abc12345678", "private", origin="adhoc")

    upsert_video(conn, make_meta(channel_id="UCreal"), origin="adhoc")

    row = conn.execute("SELECT channel_id FROM videos WHERE video_id = 'abc12345678'").fetchone()
    assert row == ("UCreal",)


def make_entry(**overrides: object) -> FeedEntry:
    defaults: dict[str, object] = {
        "video_id": "feed1234567",
        "channel_id": "UCfeed",
        "title": "Feed title",
        "published_at": datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
    }
    defaults.update(overrides)
    return FeedEntry(**defaults)  # type: ignore[arg-type]


def test_insert_discovered_video_inserts_a_new_row(conn: psycopg.Connection) -> None:
    inserted = insert_discovered_video(conn, make_entry(), origin="rss")

    assert inserted is True
    row = conn.execute(
        """
        SELECT channel_id, title, published_at, origin, description, duration_sec
        FROM videos WHERE video_id = %s
        """,
        ("feed1234567",),
    ).fetchone()
    assert row == (
        "UCfeed",
        "Feed title",
        datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        "rss",
        None,
        None,
    )


def test_insert_discovered_video_stores_the_given_unavailable_reason(
    conn: psycopg.Connection,
) -> None:
    assert insert_discovered_video(conn, make_entry(), origin="rss", unavailable="short")

    row = conn.execute("SELECT unavailable FROM videos").fetchone()
    assert row == ("short",)


def test_insert_discovered_video_leaves_unavailable_null_by_default(
    conn: psycopg.Connection,
) -> None:
    insert_discovered_video(conn, make_entry(), origin="rss")

    row = conn.execute("SELECT unavailable FROM videos").fetchone()
    assert row == (None,)


def test_insert_discovered_video_rejects_an_unknown_unavailable_reason(
    conn: psycopg.Connection,
) -> None:
    with pytest.raises(ValueError):
        insert_discovered_video(conn, make_entry(), origin="rss", unavailable="banned")

    assert conn.execute("SELECT count(*) FROM videos").fetchone() == (0,)


def test_insert_discovered_video_leaves_an_existing_row_untouched(
    conn: psycopg.Connection,
) -> None:
    upsert_video(
        conn,
        make_meta(video_id="feed1234567", channel_id="UCfeed", title="Full title"),
        origin="adhoc",
    )
    before = conn.execute(
        "SELECT * FROM videos WHERE video_id = %s", ("feed1234567",)
    ).fetchone()

    inserted = insert_discovered_video(
        conn, make_entry(title="Other title"), origin="rss"
    )

    assert inserted is False
    after = conn.execute(
        "SELECT * FROM videos WHERE video_id = %s", ("feed1234567",)
    ).fetchone()
    assert after == before


def test_insert_discovered_video_creates_a_missing_channel_inactive(
    conn: psycopg.Connection,
) -> None:
    insert_discovered_video(conn, make_entry(channel_id="UCnew"), origin="rss")

    assert conn.execute(
        "SELECT active FROM channels WHERE channel_id = %s", ("UCnew",)
    ).fetchone() == (False,)


def test_insert_discovered_video_keeps_an_existing_channel_row(
    conn: psycopg.Connection,
) -> None:
    add_channel(conn, "UCfeed", "Kept title")

    insert_discovered_video(conn, make_entry(), origin="rss")

    assert conn.execute(
        "SELECT active, title FROM channels WHERE channel_id = %s", ("UCfeed",)
    ).fetchone() == (True, "Kept title")


def test_insert_discovered_video_strips_nul_bytes_from_the_title(
    conn: psycopg.Connection,
) -> None:
    insert_discovered_video(conn, make_entry(title="a\x00b"), origin="rss")

    assert conn.execute(
        "SELECT title FROM videos WHERE video_id = %s", ("feed1234567",)
    ).fetchone() == ("ab",)


def test_video_exists_is_true_only_for_a_stored_video(conn: psycopg.Connection) -> None:
    conn.execute("INSERT INTO videos (video_id) VALUES ('abc12345678')")

    assert video_exists(conn, "abc12345678") is True
    assert video_exists(conn, "zzz12345678") is False


def test_upsert_video_fills_the_channel_of_a_submitted_stub(conn: psycopg.Connection) -> None:
    created = insert_submitted_video(conn, "abc12345678")

    upsert_video(conn, make_meta(channel_id="UCreal"), origin="rss")

    assert created is True
    row = conn.execute(
        "SELECT channel_id, origin FROM videos WHERE video_id = 'abc12345678'"
    ).fetchone()
    assert row == ("UCreal", "adhoc")


def test_upsert_video_never_overwrites_a_known_channel(conn: psycopg.Connection) -> None:
    upsert_video(conn, make_meta(channel_id="UCfirst"), origin="adhoc")

    upsert_video(conn, make_meta(channel_id="UCsecond"), origin="adhoc")

    row = conn.execute("SELECT channel_id FROM videos WHERE video_id = 'abc12345678'").fetchone()
    assert row == ("UCfirst",)


def test_insert_submitted_video_creates_a_bare_adhoc_row(conn: psycopg.Connection) -> None:
    assert insert_submitted_video(conn, "abc12345678") is True

    row = conn.execute(
        """
        SELECT channel_id, title, duration_sec, published_at, description, origin, unavailable
        FROM videos WHERE video_id = 'abc12345678'
        """
    ).fetchone()
    assert row == (None, None, None, None, None, "adhoc", None)
    assert conn.execute("SELECT count(*) FROM channels").fetchone() == (0,)


def test_insert_submitted_video_leaves_an_existing_row_untouched(
    conn: psycopg.Connection,
) -> None:
    insert_discovered_video(conn, make_entry(), origin="rss")
    before = conn.execute("SELECT * FROM videos WHERE video_id = 'feed1234567'").fetchone()

    assert insert_submitted_video(conn, "feed1234567") is False

    after = conn.execute("SELECT * FROM videos WHERE video_id = 'feed1234567'").fetchone()
    assert after == before


def test_known_video_ids_returns_only_ids_that_have_a_row(conn: psycopg.Connection) -> None:
    insert_submitted_video(conn, "known000001")
    insert_discovered_video(conn, make_entry(video_id="known000002"), origin="rss")
    record_unavailable(conn, "known000003", "removed", origin="backfill")

    found = known_video_ids(
        conn, ["known000001", "new00000001", "known000002", "known000003"]
    )

    assert found == {"known000001", "known000002", "known000003"}


def test_known_video_ids_of_an_empty_list_is_empty(conn: psycopg.Connection) -> None:
    assert known_video_ids(conn, []) == set()


def test_known_video_ids_is_one_query(conn: psycopg.Connection) -> None:
    queries: list[str] = []

    class Spy:
        def execute(self, query: str, params: Any = None) -> Any:
            queries.append(query)
            return conn.execute(query, params)

    known_video_ids(Spy(), [f"v{i:010d}" for i in range(50)])  # type: ignore[arg-type]

    assert len(queries) == 1


def test_insert_backfill_video_inserts_catalog_fields(conn: psycopg.Connection) -> None:
    conn.execute("INSERT INTO channels (channel_id, active) VALUES ('UCb', false)")

    inserted = insert_backfill_video(
        conn, CatalogEntry("back0000001", "A\x00title", 321), channel_id="UCb"
    )

    assert inserted is True
    row = conn.execute(
        "SELECT channel_id, title, duration_sec, origin, published_at, description"
        " FROM videos WHERE video_id = 'back0000001'"
    ).fetchone()
    assert row == ("UCb", "Atitle", 321, "backfill", None, None)


def test_insert_backfill_video_leaves_an_existing_row_untouched(
    conn: psycopg.Connection,
) -> None:
    insert_discovered_video(conn, make_entry(), origin="rss")
    before = conn.execute("SELECT * FROM videos WHERE video_id = 'feed1234567'").fetchone()

    inserted = insert_backfill_video(
        conn, CatalogEntry("feed1234567", "Other", 9), channel_id="UCfeed"
    )

    assert inserted is False
    assert conn.execute("SELECT * FROM videos WHERE video_id = 'feed1234567'").fetchone() == before
