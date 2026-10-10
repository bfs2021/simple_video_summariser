"""Ingest of premieres and live streams (issue #65). Needs Postgres, no network."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from adapters.youtube.errors import UpcomingVideoError
from common.errors import Defer, PermanentSourceError
from common.models import VideoMeta
from common.queue import PostgresQueue
from common.repo.videos import insert_submitted_video
from services.transcriber.ingest import make_ingest_handler
from tests.services.transcriber.fakes import (
    MANUAL_ID,
    FakeMetadata,
    FakeSubtitles,
    fixture_meta,
    make_ctx,
    make_job,
    make_settings,
    manual_segments,
)

pytestmark = pytest.mark.integration

PREMIERE_ID = "WI4__7z_dJw"
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
MARGIN = timedelta(minutes=10)


def _premiere(**changes: Any) -> VideoMeta:
    return replace(fixture_meta("upcoming_premiere", PREMIERE_ID), **changes)


class Rig:
    def __init__(
        self,
        conn: psycopg.Connection[Any],
        metadata: FakeMetadata,
        *,
        now: datetime = NOW,
        discovered: timedelta | None = timedelta(hours=1),
        **settings: Any,
    ) -> None:
        self.conn = conn
        self.metadata = metadata
        self.subtitles = FakeSubtitles(manual=manual_segments())
        self.settings = make_settings(**settings)
        self.queue = PostgresQueue(conn, settings=self.settings)
        self.handler = make_ingest_handler(
            conn=conn,
            queue=self.queue,
            metadata=metadata,
            subtitles=self.subtitles,
            settings=self.settings,
            now=lambda: now,
        )
        self.ctx, self.logs = make_ctx()
        self.now = now
        if discovered is not None:
            video_id = (
                metadata.result.video_id
                if isinstance(metadata.result, VideoMeta)
                else PREMIERE_ID
            )
            insert_submitted_video(conn, video_id)
            conn.execute(
                "UPDATE videos SET discovered_at = %s WHERE video_id = %s",
                (now - discovered, video_id),
            )
            conn.commit()

    def run(self, video_id: str) -> None:
        self.handler(make_job(video_id), self.ctx)

    def deferred_until(self, video_id: str) -> datetime:
        with pytest.raises(Defer) as info:
            self.run(video_id)
        return info.value.until

    def count(self, table: str) -> int:
        row = self.conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        self.conn.commit()
        assert row is not None
        return int(row[0])

    def video_row(self, video_id: str) -> tuple[Any, ...]:
        row = self.conn.execute(
            "SELECT title, unavailable FROM videos WHERE video_id = %s", (video_id,)
        ).fetchone()
        self.conn.commit()
        assert row is not None
        return tuple(row)


# --- is_upcoming ---------------------------------------------------------------


def test_an_upcoming_premiere_is_deferred_to_its_start_plus_a_margin(
    conn: psycopg.Connection[Any],
) -> None:
    start = NOW + timedelta(days=2)
    rig = Rig(conn, FakeMetadata(_premiere(release_at=start)))

    assert rig.deferred_until(PREMIERE_ID) == start + MARGIN


def test_a_premiere_whose_start_has_passed_is_rechecked_after_the_margin(
    conn: psycopg.Connection[Any],
) -> None:
    rig = Rig(conn, FakeMetadata(_premiere(release_at=NOW - timedelta(minutes=3))))

    assert rig.deferred_until(PREMIERE_ID) == NOW + MARGIN


def test_a_premiere_with_no_known_start_is_rechecked_in_an_hour(
    conn: psycopg.Connection[Any],
) -> None:
    rig = Rig(conn, FakeMetadata(_premiere(release_at=None)))

    assert rig.deferred_until(PREMIERE_ID) == NOW + timedelta(hours=1)


def test_a_deferred_premiere_changes_nothing_else(conn: psycopg.Connection[Any]) -> None:
    rig = Rig(conn, FakeMetadata(_premiere(release_at=NOW + timedelta(days=1))))

    rig.deferred_until(PREMIERE_ID)

    assert rig.count("transcripts") == 0
    assert rig.count("jobs") == 0
    assert rig.video_row(PREMIERE_ID) == (None, None)  # not unavailable, not rewritten


def test_the_upcoming_error_from_metadata_is_a_deferral_not_a_failed_attempt(
    conn: psycopg.Connection[Any],
) -> None:
    rig = Rig(conn, FakeMetadata(UpcomingVideoError("Premieres in 12 hours")))

    assert rig.deferred_until(PREMIERE_ID) == NOW + timedelta(hours=1)


# --- is_live / post_live -------------------------------------------------------


def test_a_live_stream_is_rechecked_after_two_hours(conn: psycopg.Connection[Any]) -> None:
    rig = Rig(conn, FakeMetadata(_premiere(live_status="is_live")))

    assert rig.deferred_until(PREMIERE_ID) == NOW + timedelta(hours=2)


def test_a_stream_that_ended_but_is_still_processing_is_rechecked_in_half_an_hour(
    conn: psycopg.Connection[Any],
) -> None:
    rig = Rig(conn, FakeMetadata(_premiere(live_status="post_live")))

    assert rig.deferred_until(PREMIERE_ID) == NOW + timedelta(minutes=30)


# --- finished streams and ordinary videos are untouched ------------------------


@pytest.mark.parametrize("status", ["was_live", "not_live", None])
def test_other_live_statuses_are_ingested_like_any_video(
    conn: psycopg.Connection[Any], status: str | None
) -> None:
    meta = replace(fixture_meta("manual_en", MANUAL_ID), live_status=status)
    rig = Rig(conn, FakeMetadata(meta), discovered=None)

    rig.run(MANUAL_ID)

    assert rig.count("transcripts") == 1
    assert rig.count("jobs") == 1  # the analyze job


# --- the give-up limit ---------------------------------------------------------


def test_default_limit_is_fourteen_days() -> None:
    assert make_settings().UPCOMING_MAX_WAIT_DAYS == 14


@pytest.mark.parametrize("status", ["is_upcoming", "is_live", "post_live"])
def test_exactly_at_the_limit_it_dead_letters_with_a_clear_message(
    conn: psycopg.Connection[Any], status: str
) -> None:
    rig = Rig(
        conn,
        FakeMetadata(_premiere(live_status=status)),
        discovered=timedelta(days=14),
    )

    with pytest.raises(PermanentSourceError) as info:
        rig.run(PREMIERE_ID)

    message = str(info.value)
    assert "14 days" in message and PREMIERE_ID in message
    assert status in message


def test_one_second_before_the_limit_it_still_defers(conn: psycopg.Connection[Any]) -> None:
    rig = Rig(
        conn,
        FakeMetadata(_premiere(release_at=NOW + timedelta(days=1))),
        discovered=timedelta(days=14) - timedelta(seconds=1),
    )

    rig.deferred_until(PREMIERE_ID)


def test_the_limit_is_configurable(conn: psycopg.Connection[Any]) -> None:
    rig = Rig(
        conn,
        FakeMetadata(_premiere()),
        discovered=timedelta(days=2),
        UPCOMING_MAX_WAIT_DAYS=2,
    )

    with pytest.raises(PermanentSourceError, match="2 days"):
        rig.run(PREMIERE_ID)


def test_the_upcoming_error_from_metadata_also_honours_the_limit(
    conn: psycopg.Connection[Any],
) -> None:
    rig = Rig(
        conn,
        FakeMetadata(UpcomingVideoError("Premieres in 12 hours")),
        discovered=timedelta(days=15),
    )

    with pytest.raises(PermanentSourceError, match="14 days"):
        rig.run(PREMIERE_ID)


def test_without_a_video_row_the_job_age_is_the_discovery_time(
    conn: psycopg.Connection[Any],
) -> None:
    rig = Rig(conn, FakeMetadata(_premiere()), discovered=None)
    job = replace(make_job(PREMIERE_ID), created_at=NOW - timedelta(days=20))

    with pytest.raises(PermanentSourceError):
        rig.handler(job, rig.ctx)


# --- through the real queue: what the Ops view sees ----------------------------


def test_a_deferred_job_is_pending_in_the_future_with_its_attempt_refunded(
    conn: psycopg.Connection[Any],
) -> None:
    now = datetime.now(UTC)
    start = now + timedelta(days=3)
    rig = Rig(conn, FakeMetadata(_premiere(release_at=start)), now=now)
    rig.queue.enqueue("ingest", PREMIERE_ID)

    with rig.queue.claim(["ingest"], worker="w") as job:
        assert job is not None
        rig.handler(job, rig.ctx)

    row = conn.execute(
        "SELECT state, attempts, error_class, last_error, run_after FROM jobs"
    ).fetchone()
    assert row is not None
    assert row[:4] == ("pending", 0, None, None)
    assert abs(row[4] - (start + MARGIN)) < timedelta(milliseconds=1)
    assert row[4] > now


def test_a_job_past_the_limit_is_dead_with_a_readable_last_error(
    conn: psycopg.Connection[Any],
) -> None:
    now = datetime.now(UTC)
    rig = Rig(conn, FakeMetadata(_premiere()), now=now, discovered=timedelta(days=30))
    rig.queue.enqueue("ingest", PREMIERE_ID)

    with rig.queue.claim(["ingest"], worker="w") as job:
        assert job is not None
        rig.handler(job, rig.ctx)

    row = conn.execute("SELECT state, error_class, last_error FROM jobs").fetchone()
    assert row is not None
    assert row[0] == "dead"
    assert row[1] == "PERMANENT_SOURCE"
    assert "14 days" in row[2] and "is_upcoming" in row[2]
