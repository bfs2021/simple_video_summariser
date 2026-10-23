"""``known_videos`` and ``insert_catalog_video`` in common/repo/videos.py (issue #93).

``insert_catalog_video`` is the origin-parameterised insert that #41's
``insert_backfill_video`` and #93's reconciliation share.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from common.models import CatalogEntry, FeedEntry
from common.repo.videos import (
    KnownVideo,
    insert_backfill_video,
    insert_catalog_video,
    insert_discovered_video,
    insert_submitted_video,
    known_video_ids,
    known_videos,
    record_unavailable,
)

pytestmark = pytest.mark.integration

PUBLISHED = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def _channel(conn: psycopg.Connection, channel_id: str, active: bool = False) -> None:
    conn.execute(
        "INSERT INTO channels (channel_id, active) VALUES (%s, %s)", (channel_id, active)
    )


def _row(conn: psycopg.Connection, video_id: str) -> tuple[Any, ...] | None:
    return conn.execute(
        "SELECT channel_id, title, duration_sec, origin, published_at, description, unavailable "
        "FROM videos WHERE video_id = %s",
        (video_id,),
    ).fetchone()


# --- known_videos ----------------------------------------------------------------


def test_known_videos_returns_channel_and_published_at_for_each_known_id(
    conn: psycopg.Connection,
) -> None:
    insert_discovered_video(
        conn, FeedEntry("feed0000001", "UCa", "t", PUBLISHED), origin="rss"
    )
    insert_submitted_video(conn, "adhoc000001")  # NULL channel, NULL published_at
    record_unavailable(conn, "gone0000001", "removed", origin="backfill")

    found = known_videos(conn, ["feed0000001", "new00000001", "adhoc000001", "gone0000001"])

    assert found == {
        "feed0000001": KnownVideo(channel_id="UCa", published_at=PUBLISHED),
        "adhoc000001": KnownVideo(channel_id=None, published_at=None),
        "gone0000001": KnownVideo(channel_id=None, published_at=None),
    }


def test_known_videos_of_an_empty_list_is_empty_and_runs_no_query(
    conn: psycopg.Connection,
) -> None:
    class Untouchable:
        def execute(self, *args: Any) -> Any:
            raise AssertionError("no query expected")

    assert known_videos(Untouchable(), []) == {}  # type: ignore[arg-type]


def test_known_videos_is_one_query_for_all_ids(conn: psycopg.Connection) -> None:
    queries: list[str] = []

    class Spy:
        def execute(self, query: str, params: Any = None) -> Any:
            queries.append(query)
            return conn.execute(query, params)

    known_videos(Spy(), [f"v{i:010d}" for i in range(50)])  # type: ignore[arg-type]

    assert len(queries) == 1


def test_known_video_ids_is_the_key_set_of_known_videos(conn: psycopg.Connection) -> None:
    insert_submitted_video(conn, "adhoc000001")
    ids = ["adhoc000001", "new00000001"]

    assert known_video_ids(conn, ids) == set(known_videos(conn, ids)) == {"adhoc000001"}


# --- insert_catalog_video ----------------------------------------------------------


def test_insert_catalog_video_writes_the_catalog_fields_with_the_given_origin(
    conn: psycopg.Connection,
) -> None:
    _channel(conn, "UCa", active=True)

    inserted = insert_catalog_video(
        conn, CatalogEntry("cat00000001", "A\x00title", 321), channel_id="UCa", origin="rss"
    )

    assert inserted is True
    assert _row(conn, "cat00000001") == ("UCa", "Atitle", 321, "rss", None, None, None)


def test_insert_catalog_video_allows_a_missing_title_and_duration(
    conn: psycopg.Connection,
) -> None:
    _channel(conn, "UCa")

    assert insert_catalog_video(
        conn, CatalogEntry("cat00000001", None, None), channel_id="UCa", origin="rss"
    )
    assert _row(conn, "cat00000001") == ("UCa", None, None, "rss", None, None, None)


def test_insert_catalog_video_leaves_an_existing_row_exactly_as_it_is(
    conn: psycopg.Connection,
) -> None:
    insert_discovered_video(
        conn, FeedEntry("feed0000001", "UCa", "Feed title", PUBLISHED), origin="rss"
    )
    before = conn.execute("SELECT * FROM videos WHERE video_id = 'feed0000001'").fetchone()

    inserted = insert_catalog_video(
        conn, CatalogEntry("feed0000001", "Other", 9), channel_id="UCb", origin="backfill"
    )

    assert inserted is False
    assert conn.execute("SELECT * FROM videos WHERE video_id = 'feed0000001'").fetchone() == before


def test_insert_catalog_video_creates_a_missing_channel_row_inactive(
    conn: psycopg.Connection,
) -> None:
    insert_catalog_video(
        conn, CatalogEntry("cat00000001", "t", 1), channel_id="UCnew", origin="rss"
    )

    assert conn.execute(
        "SELECT active FROM channels WHERE channel_id = 'UCnew'"
    ).fetchone() == (False,)


def test_insert_catalog_video_leaves_an_existing_channel_row_alone(
    conn: psycopg.Connection,
) -> None:
    _channel(conn, "UCa", active=True)
    before = conn.execute("SELECT * FROM channels WHERE channel_id = 'UCa'").fetchone()

    insert_catalog_video(conn, CatalogEntry("cat00000001", "t", 1), channel_id="UCa", origin="rss")

    assert conn.execute("SELECT * FROM channels WHERE channel_id = 'UCa'").fetchone() == before


def test_insert_backfill_video_is_insert_catalog_video_with_the_backfill_origin(
    conn: psycopg.Connection,
) -> None:
    _channel(conn, "UCb")

    assert insert_backfill_video(conn, CatalogEntry("bf000000001", "t", 5), channel_id="UCb")

    assert _row(conn, "bf000000001") == ("UCb", "t", 5, "backfill", None, None, None)


def test_tombstoned_ids_are_known_and_discovery_cannot_restore_them(
    conn: psycopg.Connection,
) -> None:
    _channel(conn, "UCa", active=True)
    ids = ["feed0000001", "cat00000001"]
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO deleted_videos (video_id) VALUES (%s)", [(value,) for value in ids])

    assert known_video_ids(conn, ids) == set(ids)
    assert not insert_discovered_video(
        conn, FeedEntry(ids[0], "UCa", "Feed", PUBLISHED), origin="rss"
    )
    assert not insert_catalog_video(
        conn, CatalogEntry(ids[1], "Catalog", 5), channel_id="UCa", origin="backfill"
    )
    assert _row(conn, ids[0]) is None
    assert _row(conn, ids[1]) is None


def test_explicit_submission_atomically_clears_a_tombstone_and_restores_the_video(
    conn: psycopg.Connection,
) -> None:
    conn.execute("INSERT INTO deleted_videos (video_id) VALUES ('adhoc000001')")

    assert insert_submitted_video(conn, "adhoc000001")

    assert _row(conn, "adhoc000001") == (None, None, None, "adhoc", None, None, None)
    assert conn.execute(
        "SELECT 1 FROM deleted_videos WHERE video_id = 'adhoc000001'"
    ).fetchone() is None
