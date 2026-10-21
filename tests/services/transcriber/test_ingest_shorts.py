"""The Shorts policy at ingest (issue #80).

The first half runs without Postgres: a fake connection and queue, with the
repository functions ``ingest`` imports replaced by recorders. The second half
(``integration``) runs the handler against real tables.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import psycopg
import pytest

from common.models import VideoMeta
from common.queue import PostgresQueue
from services.transcriber import ingest as ingest_module
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

SHORT_ID = "VV_JW4iCni0"  # short.json: media_type == "short", 32 s, manual "en" track


def _short() -> VideoMeta:
    meta = fixture_meta("short", SHORT_ID)
    assert meta.is_short
    return meta


class FakeConn:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


class FakeQueue:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def enqueue(self, kind: str, video_id: str, **kwargs: Any) -> int | None:
        self.calls.append((kind, video_id))
        return len(self.calls)

    def handoff(self, parent_job_id: int, kind: str, video_id: str, **kwargs: Any) -> int | None:
        return self.enqueue(kind, video_id, **kwargs)


class UnitRig:
    """The handler with every database call recorded instead of made."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, meta: VideoMeta, **settings: Any) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.conn = FakeConn()
        self.queue = FakeQueue()
        self.subtitles = FakeSubtitles()
        events = self.events
        monkeypatch.setattr(
            ingest_module, "upsert_video", lambda conn, m, origin: events.append(("upsert", origin))
        )
        monkeypatch.setattr(
            ingest_module,
            "record_unavailable",
            lambda conn, vid, reason, *, origin: events.append(("record", vid, reason, origin)),
        )
        monkeypatch.setattr(
            ingest_module, "clear_unavailable", lambda conn, vid: events.append(("clear", vid))
        )
        monkeypatch.setattr(ingest_module, "get_best_transcript", lambda conn, vid: None)
        monkeypatch.setattr(ingest_module, "latest_job_state", lambda conn, vid, kind: None)
        self.handler = make_ingest_handler(
            conn=self.conn,  # type: ignore[arg-type]
            queue=self.queue,  # type: ignore[arg-type]
            metadata=FakeMetadata(meta),
            subtitles=self.subtitles,
            settings=make_settings(**settings),
        )
        self.ctx, self.logs = make_ctx()

    def run(self, video_id: str, origin: str | None = None) -> None:
        payload = {} if origin is None else {"origin": origin}
        self.handler(make_job(video_id, payload=payload), self.ctx)

    @property
    def skipped(self) -> bool:
        return any(event[0] == "record" for event in self.events)


@pytest.mark.parametrize("origin", ["rss", "backfill"])
def test_a_short_from_an_automatic_origin_is_recorded_and_not_processed(
    monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    rig = UnitRig(monkeypatch, _short(), SKIP_SHORTS=True)

    rig.run(SHORT_ID, origin)  # returns normally: the job finishes done, not dead

    assert rig.events == [("upsert", origin), ("record", SHORT_ID, "short", origin)]
    assert rig.conn.commits >= 1
    assert rig.subtitles.fetch_calls == []
    assert rig.queue.calls == []
    skipped = [r for r in rig.logs.records("info") if r["event"] == "ingest skipped: short"]
    assert len(skipped) == 1
    assert (skipped[0]["video_id"], skipped[0]["origin"]) == (SHORT_ID, origin)


def test_an_adhoc_short_is_processed_even_with_skip_shorts_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = UnitRig(monkeypatch, _short(), SKIP_SHORTS=True)

    rig.run(SHORT_ID, "adhoc")

    assert not rig.skipped
    assert ("clear", SHORT_ID) in rig.events
    assert len(rig.subtitles.fetch_calls) == 1  # the empty manual track, then transcribe
    assert rig.queue.calls == [("transcribe", SHORT_ID)]


def test_the_origin_defaults_to_adhoc_so_a_job_without_one_is_never_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = UnitRig(monkeypatch, _short(), SKIP_SHORTS=True)

    rig.run(SHORT_ID)

    assert not rig.skipped


@pytest.mark.parametrize("origin", ["rss", "backfill", "adhoc"])
def test_with_skip_shorts_off_a_short_is_processed_normally(
    monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    rig = UnitRig(monkeypatch, _short())  # the default: SKIP_SHORTS is False

    rig.run(SHORT_ID, origin)

    assert not rig.skipped
    assert rig.events[:2] == [("upsert", origin), ("clear", SHORT_ID)]
    assert rig.queue.calls == [("transcribe", SHORT_ID)]


@pytest.mark.parametrize("skip_shorts", [False, True])
@pytest.mark.parametrize("origin", ["rss", "backfill", "adhoc"])
def test_a_normal_video_is_processed_whatever_the_flag(
    monkeypatch: pytest.MonkeyPatch, skip_shorts: bool, origin: str
) -> None:
    meta = fixture_meta("no_captions", "BZiu46G4ukc")
    assert not meta.is_short
    rig = UnitRig(monkeypatch, meta, SKIP_SHORTS=skip_shorts)

    rig.run(meta.video_id, origin)

    assert not rig.skipped
    assert rig.queue.calls == [("transcribe", meta.video_id)]


def test_a_duration_under_a_minute_alone_does_not_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    meta = replace(fixture_meta("no_captions", "BZiu46G4ukc"), duration_sec=32)
    rig = UnitRig(monkeypatch, meta, SKIP_SHORTS=True)

    rig.run(meta.video_id, "rss")

    assert not rig.skipped
    assert rig.queue.calls == [("transcribe", meta.video_id)]


# -- integration -------------------------------------------------------------


class Rig:
    def __init__(self, conn: psycopg.Connection[Any], meta: VideoMeta, **settings: Any) -> None:
        self.conn = conn
        self.subtitles = FakeSubtitles(manual=manual_segments())
        self.settings = make_settings(**settings)
        self.queue = PostgresQueue(conn, settings=self.settings)
        self.handler = make_ingest_handler(
            conn=conn,
            queue=self.queue,
            metadata=FakeMetadata(meta),
            subtitles=self.subtitles,
            settings=self.settings,
        )
        self.ctx, self.logs = make_ctx()

    def run(self, video_id: str, origin: str) -> None:
        self.handler(make_job(video_id, payload={"origin": origin}), self.ctx)

    def rows(self, sql: str) -> list[tuple[Any, ...]]:
        rows = self.conn.execute(sql).fetchall()
        self.conn.commit()
        return rows


@pytest.mark.integration
@pytest.mark.parametrize("origin", ["rss", "backfill"])
def test_a_skipped_short_keeps_its_row_marked_short_and_writes_nothing_else(
    conn: psycopg.Connection[Any], origin: str
) -> None:
    rig = Rig(conn, _short(), SKIP_SHORTS=True)

    rig.run(SHORT_ID, origin)

    assert rig.rows("SELECT video_id, origin, unavailable, title IS NOT NULL FROM videos") == [
        (SHORT_ID, origin, "short", True)
    ]
    assert rig.subtitles.fetch_calls == []
    assert rig.rows("SELECT count(*) FROM jobs") == [(0,)]
    assert rig.rows("SELECT count(*) FROM transcripts") == [(0,)]


@pytest.mark.integration
def test_an_adhoc_submission_clears_an_earlier_short_skip_and_is_processed(
    conn: psycopg.Connection[Any],
) -> None:
    rig = Rig(conn, _short(), SKIP_SHORTS=True)
    rig.run(SHORT_ID, "rss")
    assert rig.rows("SELECT unavailable FROM videos") == [("short",)]

    rig.run(SHORT_ID, "adhoc")

    assert rig.rows("SELECT unavailable FROM videos") == [(None,)]
    assert rig.rows("SELECT source FROM transcripts") == [("youtube_manual",)]
    assert rig.rows("SELECT kind FROM jobs") == [("analyze",)]


@pytest.mark.integration
@pytest.mark.parametrize("origin", ["rss", "backfill", "adhoc"])
def test_with_skip_shorts_off_a_short_is_ingested_like_any_video(
    conn: psycopg.Connection[Any], origin: str
) -> None:
    rig = Rig(conn, _short())

    rig.run(SHORT_ID, origin)

    assert rig.rows("SELECT unavailable FROM videos") == [(None,)]
    assert rig.rows("SELECT source FROM transcripts") == [("youtube_manual",)]
    assert rig.rows("SELECT kind FROM jobs") == [("analyze",)]


@pytest.mark.integration
def test_a_normal_video_is_ingested_with_skip_shorts_on(conn: psycopg.Connection[Any]) -> None:
    rig = Rig(conn, fixture_meta("manual_en", MANUAL_ID), SKIP_SHORTS=True)

    rig.run(MANUAL_ID, "rss")

    assert rig.rows("SELECT unavailable FROM videos") == [(None,)]
    assert rig.rows("SELECT source FROM transcripts") == [("youtube_manual",)]
