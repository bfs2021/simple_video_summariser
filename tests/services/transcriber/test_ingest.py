"""Ingest handler against a real Postgres and the real ``PostgresQueue`` (issue #28).

Metadata and subtitles are fakes built from the recorded #16 / #17 fixtures, so
nothing touches the network or yt-dlp.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import psycopg
import pytest

from common.chunking import chunk_strategy
from common.errors import (
    Cancelled,
    PermanentSourceError,
    RateLimitedError,
    ToolFailureError,
    TransientNetworkError,
)
from common.models import Segment
from common.queue import PostgresQueue
from common.repo.transcripts import get_best_transcript, save_transcript
from services.transcriber import ingest as ingest_module
from services.transcriber import persist as persist_module
from services.transcriber.ingest import make_ingest_handler
from tests.services.transcriber.fakes import (
    AUTO_ID,
    MANUAL_ID,
    NO_CAPTIONS_ID,
    TRANSLATED_ID,
    FakeMetadata,
    FakeSubtitles,
    auto_segments,
    fixture_meta,
    make_ctx,
    make_job,
    make_settings,
    manual_segments,
)

pytestmark = pytest.mark.integration

STRATEGY = chunk_strategy(900, 60)
ANALYZE_KEY = "v1:ollama"


class Rig:
    """One handler wired to the real queue, plus readers for what it wrote."""

    def __init__(
        self,
        conn: psycopg.Connection[Any],
        metadata: FakeMetadata,
        subtitles: FakeSubtitles,
        **settings: Any,
    ) -> None:
        self.conn = conn
        self.metadata = metadata
        self.subtitles = subtitles
        self.settings = make_settings(**settings)
        self.queue = PostgresQueue(conn, settings=self.settings)
        self.handler = make_ingest_handler(
            conn=conn,
            queue=self.queue,
            metadata=metadata,
            subtitles=subtitles,
            settings=self.settings,
        )
        self.ctx, self.logs = make_ctx()

    def run(self, video_id: str, **job: Any) -> None:
        built = make_job(video_id, **job)
        parent_id = self.queue.enqueue("ingest", video_id, priority=built.priority)
        assert parent_id is not None
        self.conn.execute("UPDATE jobs SET state = 'running' WHERE id = %s", (parent_id,))
        self.conn.commit()
        try:
            self.handler(replace(built, id=parent_id), self.ctx)
        finally:
            self.conn.execute("DELETE FROM jobs WHERE id = %s", (parent_id,))
            self.conn.commit()

    def scalar(self, sql: str, *params: object) -> Any:
        row = self.conn.execute(sql, params).fetchone()
        self.conn.commit()
        return row[0] if row else None

    def rows(self, sql: str, *params: object) -> list[tuple[Any, ...]]:
        rows = self.conn.execute(sql, params).fetchall()
        self.conn.commit()
        return rows

    def jobs(self, kind: str) -> list[tuple[Any, ...]]:
        return self.rows(
            "SELECT state, dedupe_key, priority, payload FROM jobs WHERE kind = %s ORDER BY id",
            kind,
        )

    def transcripts(self) -> list[tuple[Any, ...]]:
        return self.rows(
            "SELECT source, language, speaker_source, engine_meta FROM transcripts ORDER BY id"
        )

    def chunk_count(self) -> int:
        return int(self.scalar("SELECT count(*) FROM transcript_chunks"))

    def add_transcribe_job(self, video_id: str, state: str) -> None:
        self.conn.execute(
            "INSERT INTO jobs (video_id, kind, dedupe_key, state) VALUES (%s, 'transcribe', %s, %s)",
            (video_id, "default" if state in ("pending", "running") else f"k-{state}", state),
        )
        self.conn.commit()

    def nothing_written(self) -> bool:
        return all(
            self.scalar(f"SELECT count(*) FROM {table}") == 0
            for table in ("transcripts", "transcript_chunks", "jobs")
        )


def manual_rig(conn: psycopg.Connection[Any], **settings: Any) -> Rig:
    return Rig(
        conn,
        FakeMetadata(fixture_meta("manual_en", MANUAL_ID)),
        FakeSubtitles(manual=manual_segments()),
        **settings,
    )


def auto_rig(conn: psycopg.Connection[Any], **settings: Any) -> Rig:
    return Rig(
        conn,
        FakeMetadata(fixture_meta("auto_only", AUTO_ID)),
        FakeSubtitles(auto=auto_segments()),
        **settings,
    )


# --- MANUAL path -----------------------------------------------------------------


def test_manual_subtitles_are_saved_with_chunks_and_analyze_is_enqueued(
    conn: psycopg.Connection[Any],
) -> None:
    rig = manual_rig(conn)

    rig.run(MANUAL_ID, priority=7)

    assert rig.metadata.calls == [MANUAL_ID]
    assert rig.subtitles.fetch_calls == [(MANUAL_ID, "en", "manual")]
    assert rig.transcripts() == [
        ("youtube_manual", "en", "none", {"track": "en", "kind": "manual"})
    ]
    transcript = get_best_transcript(conn, MANUAL_ID)
    assert transcript is not None
    assert len(transcript.segments) == len(manual_segments())
    chunk_rows = rig.rows("SELECT chunk_strategy, seq FROM transcript_chunks ORDER BY seq")
    assert chunk_rows and {row[0] for row in chunk_rows} == {STRATEGY}
    assert [row[1] for row in chunk_rows] == list(range(len(chunk_rows)))
    assert rig.jobs("analyze") == [("pending", ANALYZE_KEY, 7, {})]
    assert rig.jobs("transcribe") == []


def test_a_video_id_starting_with_a_dash_goes_through_the_whole_happy_path(
    conn: psycopg.Connection[Any],
) -> None:
    meta = replace(fixture_meta("manual_en", MANUAL_ID), video_id="-wNyEUrxzFU")
    rig = Rig(conn, FakeMetadata(meta), FakeSubtitles(manual=manual_segments()))

    rig.run("-wNyEUrxzFU")

    assert rig.subtitles.fetch_calls == [("-wNyEUrxzFU", "en", "manual")]
    assert [row[0] for row in rig.transcripts()] == ["youtube_manual"]
    assert rig.scalar("SELECT video_id FROM jobs WHERE kind = 'analyze'") == "-wNyEUrxzFU"


def test_the_analyze_dedupe_key_follows_the_settings(conn: psycopg.Connection[Any]) -> None:
    rig = manual_rig(conn, SUMMARIZER="anthropic", PROMPT_VERSION="v2")

    rig.run(MANUAL_ID)

    assert rig.jobs("analyze")[0][1] == "v2:anthropic"


def test_chunks_use_the_configured_window(conn: psycopg.Connection[Any]) -> None:
    rig = manual_rig(conn, CHUNK_SEC=30, OVERLAP_SEC=5)

    rig.run(MANUAL_ID)

    strategies = {row[0] for row in rig.rows("SELECT chunk_strategy FROM transcript_chunks")}
    assert strategies == {chunk_strategy(30, 5)}


def test_labelled_subtitles_give_subtitle_labels_and_plain_ones_give_none(
    conn: psycopg.Connection[Any],
) -> None:
    labelled = [Segment(0.0, 1.0, "Hello there.", "JANE DOE"), Segment(1.0, 2.0, "Hi.", None)]
    rig = Rig(
        conn,
        FakeMetadata(fixture_meta("manual_en", MANUAL_ID)),
        FakeSubtitles(manual=labelled),
    )

    rig.run(MANUAL_ID)

    assert rig.transcripts()[0][2] == "subtitle_labels"


def test_unlabelled_subtitles_give_speaker_source_none(conn: psycopg.Connection[Any]) -> None:
    rig = manual_rig(conn)

    rig.run(MANUAL_ID)

    assert rig.transcripts()[0][2] == "none"


def test_an_empty_manual_download_falls_through_to_transcribe(
    conn: psycopg.Connection[Any],
) -> None:
    rig = Rig(conn, FakeMetadata(fixture_meta("manual_en", MANUAL_ID)), FakeSubtitles(manual=[]))

    rig.run(MANUAL_ID)

    assert rig.transcripts() == []
    assert rig.chunk_count() == 0
    assert rig.jobs("analyze") == []
    assert len(rig.jobs("transcribe")) == 1
    assert rig.subtitles.fetch_calls == [(MANUAL_ID, "en", "manual")]


def test_an_empty_manual_download_then_dead_transcribe_uses_auto_captions(
    conn: psycopg.Connection[Any],
) -> None:
    rig = Rig(
        conn,
        FakeMetadata(fixture_meta("manual_en", MANUAL_ID)),
        FakeSubtitles(manual=[], auto=auto_segments()),
    )
    rig.add_transcribe_job(MANUAL_ID, "dead")

    rig.run(MANUAL_ID)

    assert [c[2] for c in rig.subtitles.fetch_calls] == ["manual", "auto"]
    assert [row[0] for row in rig.transcripts()] == ["youtube_auto"]


# --- TRANSCRIBE path -------------------------------------------------------------


def test_no_captions_enqueues_transcribe_with_priority_and_language(
    conn: psycopg.Connection[Any],
) -> None:
    meta = replace(fixture_meta("no_captions", NO_CAPTIONS_ID), language="en")
    rig = Rig(conn, FakeMetadata(meta), FakeSubtitles())

    rig.run(NO_CAPTIONS_ID, priority=3)

    assert rig.jobs("transcribe") == [
        ("pending", "default", 3, {"language": meta.language, "duration_sec": meta.duration_sec})
    ]
    assert rig.subtitles.fetch_calls == []
    assert rig.jobs("analyze") == []
    assert rig.transcripts() == []


def test_the_transcribe_payload_carries_the_duration_for_the_runtime_limit(
    conn: psycopg.Connection[Any],
) -> None:
    meta = replace(fixture_meta("no_captions", NO_CAPTIONS_ID), duration_sec=3210)
    rig = Rig(conn, FakeMetadata(meta), FakeSubtitles())

    rig.run(NO_CAPTIONS_ID)

    assert rig.jobs("transcribe")[0][3]["duration_sec"] == 3210


def test_the_language_is_omitted_from_the_transcribe_payload_when_unknown(
    conn: psycopg.Connection[Any],
) -> None:
    meta = replace(fixture_meta("no_captions", NO_CAPTIONS_ID), language=None)
    rig = Rig(conn, FakeMetadata(meta), FakeSubtitles())

    rig.run(NO_CAPTIONS_ID)

    assert rig.jobs("transcribe")[0][3] == {"duration_sec": meta.duration_sec}


def test_the_duration_is_omitted_from_the_transcribe_payload_when_unknown(
    conn: psycopg.Connection[Any],
) -> None:
    meta = replace(fixture_meta("no_captions", NO_CAPTIONS_ID), duration_sec=None)
    rig = Rig(conn, FakeMetadata(meta), FakeSubtitles())

    rig.run(NO_CAPTIONS_ID)

    assert "duration_sec" not in rig.jobs("transcribe")[0][3]


def test_prefer_whisper_skips_manual_subtitles_and_never_downloads_any(
    conn: psycopg.Connection[Any],
) -> None:
    rig = manual_rig(conn, PREFER_WHISPER=True)

    rig.run(MANUAL_ID)

    assert rig.subtitles.fetch_calls == []
    assert len(rig.jobs("transcribe")) == 1
    assert rig.transcripts() == []


def test_auto_captions_are_not_downloaded_while_transcription_is_still_possible(
    conn: psycopg.Connection[Any],
) -> None:
    rig = auto_rig(conn)

    rig.run(AUTO_ID)

    assert rig.subtitles.fetch_calls == []
    assert len(rig.jobs("transcribe")) == 1
    assert rig.transcripts() == []


@pytest.mark.parametrize("state", ["pending", "running", "done"])
def test_a_transcribe_job_that_is_not_dead_is_not_replaced_by_auto_captions(
    conn: psycopg.Connection[Any], state: str
) -> None:
    rig = auto_rig(conn)
    rig.add_transcribe_job(AUTO_ID, state)

    rig.run(AUTO_ID)

    assert rig.subtitles.fetch_calls == []
    assert rig.transcripts() == []


def test_an_already_active_transcribe_job_is_logged_at_info_and_the_run_succeeds(
    conn: psycopg.Connection[Any],
) -> None:
    rig = auto_rig(conn)
    rig.add_transcribe_job(AUTO_ID, "pending")

    rig.run(AUTO_ID)

    assert len(rig.jobs("transcribe")) == 1
    messages = " ".join(str(r["event"]) for r in rig.logs.records("info"))
    assert "already" in messages


def test_a_retry_transcribe_job_after_a_dead_one_counts_as_not_dead(
    conn: psycopg.Connection[Any],
) -> None:
    rig = auto_rig(conn)
    rig.add_transcribe_job(AUTO_ID, "dead")
    rig.add_transcribe_job(AUTO_ID, "pending")

    rig.run(AUTO_ID)

    assert rig.subtitles.fetch_calls == []
    assert rig.transcripts() == []


# --- AUTO_FALLBACK and NONE paths ------------------------------------------------


def test_auto_captions_are_used_after_transcription_dead_letters(
    conn: psycopg.Connection[Any],
) -> None:
    rig = auto_rig(conn)
    rig.add_transcribe_job(AUTO_ID, "dead")

    rig.run(AUTO_ID, priority=2)

    assert rig.subtitles.fetch_calls == [(AUTO_ID, "en", "auto")]
    assert rig.transcripts() == [
        ("youtube_auto", "en", "none", {"track": "en", "kind": "auto"})
    ]
    assert rig.chunk_count() > 0
    assert rig.jobs("analyze") == [("pending", ANALYZE_KEY, 2, {})]
    assert len(rig.jobs("transcribe")) == 1  # only the dead one
    warnings = " ".join(str(r["event"]) for r in rig.logs.records("warning"))
    assert "degraded" in warnings


def test_the_fallback_is_off_when_auto_caption_fallback_is_disabled(
    conn: psycopg.Connection[Any],
) -> None:
    rig = auto_rig(conn, AUTO_CAPTION_FALLBACK=False)
    rig.add_transcribe_job(AUTO_ID, "dead")

    rig.run(AUTO_ID)

    assert rig.subtitles.fetch_calls == []
    assert rig.transcripts() == []
    assert rig.jobs("analyze") == []
    assert [j[0] for j in rig.jobs("transcribe")] == ["dead"]  # no new transcribe
    assert rig.logs.records("warning")


def test_no_auto_track_after_a_dead_transcribe_finishes_without_enqueueing(
    conn: psycopg.Connection[Any],
) -> None:
    rig = Rig(conn, FakeMetadata(fixture_meta("no_captions", NO_CAPTIONS_ID)), FakeSubtitles())
    rig.add_transcribe_job(NO_CAPTIONS_ID, "dead")

    rig.run(NO_CAPTIONS_ID)

    assert [j[0] for j in rig.jobs("transcribe")] == ["dead"]
    assert rig.jobs("analyze") == []
    assert rig.transcripts() == []


def test_an_empty_auto_download_is_treated_as_none(conn: psycopg.Connection[Any]) -> None:
    rig = Rig(conn, FakeMetadata(fixture_meta("auto_only", AUTO_ID)), FakeSubtitles(auto=[]))
    rig.add_transcribe_job(AUTO_ID, "dead")

    rig.run(AUTO_ID)

    assert rig.transcripts() == []
    assert rig.jobs("analyze") == []
    assert [j[0] for j in rig.jobs("transcribe")] == ["dead"]
    assert rig.logs.records("info")[-1]["transcript_path"] == "none"


def test_a_translated_en_auto_track_on_a_non_english_video_is_never_used(
    conn: psycopg.Connection[Any],
) -> None:
    rig = Rig(
        conn,
        FakeMetadata(fixture_meta("non_english_translated_en", TRANSLATED_ID)),
        FakeSubtitles(auto=auto_segments()),
    )
    rig.add_transcribe_job(TRANSLATED_ID, "dead")

    rig.run(TRANSLATED_ID)

    assert rig.subtitles.fetch_calls == []
    assert rig.transcripts() == []
    assert rig.jobs("analyze") == []


# --- HAVE_TRANSCRIPT path --------------------------------------------------------


def test_an_existing_whisper_transcript_downloads_nothing_and_reenqueues_analyze(
    conn: psycopg.Connection[Any],
) -> None:
    rig = manual_rig(conn)
    rig.run(MANUAL_ID)  # creates the videos row
    conn.execute("DELETE FROM transcripts")
    conn.execute("DELETE FROM jobs")
    save_transcript(conn, MANUAL_ID, "whisper", "en", "none", [Segment(0, 1, "hi")], None)
    conn.commit()
    rig.subtitles.fetch_calls.clear()

    rig.run(MANUAL_ID, priority=4)

    assert rig.subtitles.fetch_calls == []
    assert rig.jobs("transcribe") == []
    assert rig.jobs("analyze") == [("pending", ANALYZE_KEY, 4, {})]
    assert rig.logs.records("info")[-1]["transcript_path"] == "have_transcript"


def test_an_existing_auto_transcript_does_not_block_a_manual_download(
    conn: psycopg.Connection[Any],
) -> None:
    rig = manual_rig(conn)
    rig.run(MANUAL_ID)
    conn.execute("DELETE FROM transcripts")
    save_transcript(conn, MANUAL_ID, "youtube_auto", "en", "none", [Segment(0, 1, "hi")], None)
    conn.commit()

    rig.run(MANUAL_ID)

    assert [row[0] for row in rig.transcripts()] == ["youtube_auto", "youtube_manual"]


# --- input validation ------------------------------------------------------------


def test_an_invalid_video_id_writes_no_video_row(conn: psycopg.Connection[Any]) -> None:
    rig = manual_rig(conn)

    with pytest.raises(ValueError, match="video"):
        rig.run("--exec=x")

    assert rig.metadata.calls == []
    assert rig.subtitles.fetch_calls == []
    assert rig.scalar("SELECT count(*) FROM videos") == 0


def test_origin_defaults_to_adhoc_and_other_payload_keys_are_ignored(
    conn: psycopg.Connection[Any],
) -> None:
    rig = manual_rig(conn)

    rig.run(MANUAL_ID, payload={"unrelated": [1, 2]})

    assert rig.scalar("SELECT origin FROM videos") == "adhoc"


@pytest.mark.parametrize("origin", ["adhoc", "rss", "backfill"])
def test_a_valid_origin_is_stored(conn: psycopg.Connection[Any], origin: str) -> None:
    rig = manual_rig(conn)

    rig.run(MANUAL_ID, payload={"origin": origin})

    assert rig.scalar("SELECT origin FROM videos") == origin


# --- metadata --------------------------------------------------------------------


def test_the_video_row_is_committed_before_the_subtitle_download_and_survives_its_failure(
    conn: psycopg.Connection[Any], head_dsn: str
) -> None:
    rig = Rig(
        conn,
        FakeMetadata(fixture_meta("manual_en", MANUAL_ID)),
        FakeSubtitles(manual=TransientNetworkError("boom")),
    )
    visible_during_download: list[Any] = []

    def check(_kind: str) -> None:
        with psycopg.connect(head_dsn) as other:
            visible_during_download.append(other.execute("SELECT count(*) FROM videos").fetchone())

    rig.subtitles.on_fetch = check

    with pytest.raises(TransientNetworkError):
        rig.run(MANUAL_ID)

    assert visible_during_download == [(1,)]
    assert rig.scalar("SELECT count(*) FROM videos") == 1
    assert rig.nothing_written()


def test_a_metadata_video_id_mismatch_raises_tool_failure_and_writes_nothing(
    conn: psycopg.Connection[Any],
) -> None:
    other = replace(fixture_meta("manual_en", MANUAL_ID), video_id="AAAAAAAAAAA")
    rig = Rig(conn, FakeMetadata(other), FakeSubtitles(manual=manual_segments()))

    with pytest.raises(ToolFailureError):
        rig.run(MANUAL_ID)

    assert rig.scalar("SELECT count(*) FROM videos") == 0
    assert rig.subtitles.fetch_calls == []
    assert rig.nothing_written()


def test_a_successful_metadata_fetch_clears_a_previous_unavailable_marker(
    conn: psycopg.Connection[Any],
) -> None:
    rig = manual_rig(conn)
    rig.run(MANUAL_ID)
    conn.execute("UPDATE videos SET unavailable = 'private'")
    conn.commit()

    rig.run(MANUAL_ID)

    assert rig.scalar("SELECT unavailable FROM videos") is None


@pytest.mark.parametrize("reason", ["removed", "private", "geoblocked", "agegated"])
def test_a_permanent_metadata_error_on_a_first_ingest_creates_a_stub_row_and_reraises(
    conn: psycopg.Connection[Any], reason: str
) -> None:
    rig = Rig(conn, FakeMetadata(PermanentSourceError(reason, "gone")), FakeSubtitles())

    with pytest.raises(PermanentSourceError):
        rig.run(MANUAL_ID, payload={"origin": "rss"})

    assert rig.rows("SELECT video_id, channel_id, origin, unavailable FROM videos") == [
        (MANUAL_ID, None, "rss", reason)
    ]
    assert rig.nothing_written()


def test_a_permanent_metadata_error_marks_an_existing_video(
    conn: psycopg.Connection[Any],
) -> None:
    ok = manual_rig(conn)
    ok.run(MANUAL_ID)
    rig = Rig(conn, FakeMetadata(PermanentSourceError("removed")), FakeSubtitles())

    with pytest.raises(PermanentSourceError):
        rig.run(MANUAL_ID)

    assert rig.rows("SELECT origin, unavailable FROM videos") == [("adhoc", "removed")]
    assert rig.scalar("SELECT title FROM videos") is not None


def test_a_permanent_subtitle_error_marks_the_video_after_the_upsert(
    conn: psycopg.Connection[Any],
) -> None:
    rig = Rig(
        conn,
        FakeMetadata(fixture_meta("manual_en", MANUAL_ID)),
        FakeSubtitles(manual=PermanentSourceError("geoblocked")),
    )

    with pytest.raises(PermanentSourceError):
        rig.run(MANUAL_ID)

    assert rig.rows("SELECT unavailable, title IS NOT NULL FROM videos") == [("geoblocked", True)]
    assert rig.nothing_written()


ERRORS: list[Exception] = [
    TransientNetworkError("net"),
    RateLimitedError("slow down"),
    ToolFailureError("tool"),
    RuntimeError("bug"),
]


@pytest.mark.parametrize("error", ERRORS, ids=lambda e: type(e).__name__)
def test_any_other_metadata_error_propagates_unchanged_and_writes_nothing(
    conn: psycopg.Connection[Any], error: Exception
) -> None:
    rig = Rig(conn, FakeMetadata(error), FakeSubtitles())

    with pytest.raises(type(error)) as info:
        rig.run(MANUAL_ID)

    assert info.value is error
    assert rig.scalar("SELECT count(*) FROM videos") == 0
    assert rig.nothing_written()


@pytest.mark.parametrize("error", ERRORS, ids=lambda e: type(e).__name__)
@pytest.mark.parametrize("kind", ["manual", "auto"])
def test_any_other_subtitle_error_propagates_unchanged_and_saves_no_transcript(
    conn: psycopg.Connection[Any], error: Exception, kind: str
) -> None:
    if kind == "manual":
        rig = Rig(
            conn,
            FakeMetadata(fixture_meta("manual_en", MANUAL_ID)),
            FakeSubtitles(manual=error),
        )
        video_id = MANUAL_ID
    else:
        rig = Rig(
            conn,
            FakeMetadata(fixture_meta("auto_only", AUTO_ID)),
            FakeSubtitles(auto=error),
        )
        rig.add_transcribe_job(AUTO_ID, "dead")
        video_id = AUTO_ID

    with pytest.raises(type(error)) as info:
        rig.run(video_id)

    assert info.value is error
    assert rig.scalar("SELECT count(*) FROM transcripts") == 0
    assert rig.chunk_count() == 0
    assert rig.jobs("analyze") == []
    assert len(rig.jobs("transcribe")) == (1 if kind == "auto" else 0)


# --- persist atomicity -----------------------------------------------------------


def test_a_failing_chunk_save_rolls_the_transcript_back(
    conn: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_save_chunks(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(persist_module, "save_chunks", failing_save_chunks)
    rig = manual_rig(conn)

    with pytest.raises(RuntimeError, match="disk on fire"):
        rig.run(MANUAL_ID)

    assert rig.scalar("SELECT count(*) FROM transcripts") == 0
    assert rig.chunk_count() == 0
    assert rig.jobs("analyze") == []
    assert rig.scalar("SELECT count(*) FROM videos") == 1  # the upsert committed earlier


# --- idempotence and crash recovery ----------------------------------------------


def test_running_the_same_manual_job_twice_leaves_one_of_everything(
    conn: psycopg.Connection[Any],
) -> None:
    rig = manual_rig(conn)

    rig.run(MANUAL_ID, payload={"origin": "rss"})
    first = rig.rows("SELECT origin, discovered_at FROM videos")
    chunks = rig.chunk_count()
    rig.run(MANUAL_ID, payload={"origin": "backfill"})

    assert rig.rows("SELECT origin, discovered_at FROM videos") == first
    assert first[0][0] == "rss"
    assert len(rig.transcripts()) == 1
    assert rig.chunk_count() == chunks
    assert len(rig.jobs("analyze")) == 1
    assert len(rig.subtitles.fetch_calls) == 1


def test_running_the_same_transcribe_job_twice_leaves_one_active_transcribe_job(
    conn: psycopg.Connection[Any],
) -> None:
    rig = Rig(conn, FakeMetadata(fixture_meta("no_captions", NO_CAPTIONS_ID)), FakeSubtitles())

    rig.run(NO_CAPTIONS_ID)
    rig.run(NO_CAPTIONS_ID)

    assert len(rig.jobs("transcribe")) == 1
    assert rig.scalar("SELECT count(*) FROM videos") == 1


def test_a_crash_between_the_transcript_commit_and_the_analyze_enqueue_is_recovered_on_retry(
    conn: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = manual_rig(conn)
    real_handoff = rig.queue.handoff
    crashed: list[bool] = []

    def crash_once(parent_job_id: int, kind: str, video_id: str, **kwargs: Any) -> int | None:
        if kind == "analyze" and not crashed:
            crashed.append(True)
            raise RuntimeError("power cut")
        return real_handoff(parent_job_id, kind, video_id, **kwargs)

    monkeypatch.setattr(rig.queue, "handoff", crash_once)

    with pytest.raises(RuntimeError, match="power cut"):
        rig.run(MANUAL_ID)
    assert len(rig.transcripts()) == 1
    assert rig.jobs("analyze") == []

    rig.run(MANUAL_ID)

    assert len(rig.transcripts()) == 1
    assert rig.chunk_count() > 0
    assert len(rig.jobs("analyze")) == 1
    assert len(rig.subtitles.fetch_calls) == 1  # the retry did not download again


# --- observability and cancellation ----------------------------------------------


@pytest.mark.parametrize(
    ("build", "expected_path", "expected_source"),
    [
        (manual_rig, "manual", "youtube_manual"),
        (lambda c: Rig(c, FakeMetadata(fixture_meta("no_captions", NO_CAPTIONS_ID)), FakeSubtitles()), "transcribe", None),
    ],
)
def test_every_run_logs_one_info_line_with_the_path_and_the_saved_source(
    conn: psycopg.Connection[Any],
    build: Any,
    expected_path: str,
    expected_source: str | None,
) -> None:
    rig = build(conn)
    video_id = MANUAL_ID if expected_path == "manual" else NO_CAPTIONS_ID

    rig.run(video_id)

    summaries = [r for r in rig.logs.records("info") if "transcript_path" in r]
    assert len(summaries) == 1
    assert summaries[0]["transcript_path"] == expected_path
    assert summaries[0].get("transcript_source") == expected_source
    assert "transcript_source" not in summaries[0] or expected_source is not None


def test_a_cancelled_job_raises_after_the_metadata_fetch_and_writes_no_transcript(
    conn: psycopg.Connection[Any],
) -> None:
    rig = manual_rig(conn)
    rig.ctx._cancel()

    with pytest.raises(Cancelled):
        rig.run(MANUAL_ID)

    assert rig.metadata.calls == [MANUAL_ID]
    assert rig.subtitles.fetch_calls == []
    assert rig.nothing_written()


def test_cancellation_is_checked_before_each_subtitle_download(
    conn: psycopg.Connection[Any],
) -> None:
    rig = Rig(
        conn,
        FakeMetadata(fixture_meta("manual_en", MANUAL_ID)),
        FakeSubtitles(manual=[], auto=auto_segments()),
    )
    rig.add_transcribe_job(MANUAL_ID, "dead")
    rig.subtitles.on_fetch = lambda _kind: rig.ctx._cancel()

    with pytest.raises(Cancelled):
        rig.run(MANUAL_ID)

    assert [c[2] for c in rig.subtitles.fetch_calls] == ["manual"]  # auto never started
    assert rig.scalar("SELECT count(*) FROM transcripts") == 0


def test_the_module_exposes_no_sql() -> None:
    source = ingest_module.__file__
    assert source is not None
    text = open(source, encoding="utf-8").read().upper()  # noqa: SIM115
    for keyword in ("SELECT ", "INSERT ", "UPDATE ", "DELETE "):
        assert keyword not in text
