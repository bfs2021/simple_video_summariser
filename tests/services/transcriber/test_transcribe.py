"""Transcribe handler against a real Postgres and the real ``PostgresQueue`` (issue #29).

The audio source, transcriber and (mostly) the chunker are fakes: no ffmpeg, yt-dlp,
network or model. Checks that need no database live in ``test_transcribe_unit.py``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg.pq import TransactionStatus

from common.chunking import chunk_segments
from common.errors import (
    BugError,
    Cancelled,
    JobError,
    PermanentSourceError,
    ResourceError,
    ToolFailureError,
    TransientNetworkError,
)
from common.models import AudioRef, Chunk, Segment, TranscriptResult
from common.queue import Job, PostgresQueue
from common.repo.media import delete_media as real_delete_media
from common.repo.transcripts import save_transcript
from services.transcriber.transcribe import HEARTBEAT_SEC, make_transcribe_handler
from tests.services.transcriber.fakes import make_ctx, make_job, make_settings

pytestmark = pytest.mark.integration

VID = "abcdefghijk"
REL = "ab/abcdefghijk.opus"
ANALYZE_KEY = "v1:ollama"


def segs(n: int = 3) -> tuple[Segment, ...]:
    return tuple(Segment(float(i), float(i + 1), f"word{i}") for i in range(n))


def result(
    segments: tuple[Segment, ...] | None = None, language: str = "en"
) -> TranscriptResult:
    return TranscriptResult(
        segments=segs() if segments is None else segments,
        language=language,
        engine_meta={"engine": "fake", "rtf": 0.25, "model": "tiny"},
    )


class FakeAudio:
    """Writes a real file under ``dest`` (unless told not to) and records calls."""

    def __init__(
        self,
        *,
        rel_path: str = REL,
        duration: float = 120.0,
        exc: BaseException | None = None,
        write: bool = True,
    ) -> None:
        self.rel_path = rel_path
        self.duration = duration
        self.exc = exc
        self.write = write
        self.calls: list[tuple[str, Path]] = []

    def fetch_normalized(self, video_id: str, dest: Path) -> AudioRef:
        self.calls.append((video_id, dest))
        if self.exc is not None:
            raise self.exc
        if self.write:
            target = dest / self.rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"opus")
        return AudioRef(self.rel_path, 4, self.duration)


class FakeTranscriber:
    def __init__(
        self,
        outcome: TranscriptResult | BaseException | None = None,
        *,
        on_call: Callable[[AudioRef], None] | None = None,
        progress: tuple[float, ...] = (),
    ) -> None:
        self.outcome = outcome if outcome is not None else result()
        self.on_call = on_call
        self.progress = progress
        self.calls: list[dict[str, Any]] = []

    def transcribe(
        self,
        audio: AudioRef,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> TranscriptResult:
        self.calls.append({"audio": audio, "language": language})
        if self.on_call is not None:
            self.on_call(audio)
        assert on_progress is not None
        for done in self.progress:
            on_progress(done)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class SpyQueue:
    """The real queue, recording what the handler asks of it."""

    def __init__(self, inner: PostgresQueue, *, fail_enqueue: bool = False) -> None:
        self.inner = inner
        self.fail_enqueue = fail_enqueue
        self.enqueues: list[tuple[str, str, dict[str, Any]]] = []
        self.heartbeats = 0

    def enqueue(self, kind: str, video_id: str, **kwargs: Any) -> int | None:
        self.enqueues.append((kind, video_id, kwargs))
        if self.fail_enqueue:
            raise ConnectionError("boom")
        return self.inner.enqueue(kind, video_id, **kwargs)

    def handoff(self, parent_job_id: int, kind: str, video_id: str, **kwargs: Any) -> int | None:
        self.enqueues.append((kind, video_id, kwargs))
        if self.fail_enqueue:
            raise ConnectionError("boom")
        return self.inner.handoff(parent_job_id, kind, video_id, **kwargs)

    def heartbeat(self, job_id: int, worker: str) -> bool:
        self.heartbeats += 1
        return self.inner.heartbeat(job_id, worker)

    def claim(
        self, kinds: Sequence[str], *, worker: str, dedupe_key: str | None = None
    ) -> AbstractContextManager[Job | None]:
        return self.inner.claim(kinds, worker=worker, dedupe_key=dedupe_key)

    def reap_stale(self, older_than_sec: int | None = None) -> int:
        return self.inner.reap_stale(older_than_sec)


class Rig:
    def __init__(
        self,
        conn: psycopg.Connection[Any],
        dsn: str,
        tmp_path: Path,
        *,
        audio: FakeAudio | None = None,
        transcriber: FakeTranscriber | None = None,
        chunker: Callable[..., list[Chunk]] = chunk_segments,
        with_video: bool = True,
        clock: Callable[[], float] = time.monotonic,
        **settings: Any,
    ) -> None:
        self.conn = conn
        self.dsn = dsn
        self.audio_dir = tmp_path / "audio"
        self.settings = make_settings(AUDIO_DIR=self.audio_dir, **settings)
        self.audio = audio if audio is not None else FakeAudio()
        self.transcriber = transcriber if transcriber is not None else FakeTranscriber()
        self.opened: list[psycopg.Connection[Any]] = []
        self.queue = SpyQueue(PostgresQueue(conn, settings=self.settings))
        self.handler = make_transcribe_handler(
            connect=self._connect,
            queue=self.queue,
            audio_source=self.audio,
            transcriber=self.transcriber,
            chunker=chunker,
            settings=self.settings,
            clock=clock,
        )
        self.ctx, self.logs = make_ctx()
        if with_video:
            self.add_video(VID)

    def _connect(self) -> psycopg.Connection[Any]:
        opened = psycopg.connect(self.dsn)
        self.opened.append(opened)
        return opened

    def add_video(self, video_id: str) -> None:
        self.conn.execute("INSERT INTO videos (video_id) VALUES (%s)", (video_id,))
        self.conn.commit()

    def run(self, video_id: str = VID, **job: Any) -> None:
        built = replace(make_job(video_id, **job), kind="transcribe")
        parent_id = self.queue.inner.enqueue("transcribe", video_id, priority=built.priority)
        assert parent_id is not None
        self.conn.execute("UPDATE jobs SET state = 'running', locked_by = 'w' WHERE id = %s", (parent_id,))
        self.conn.commit()
        try:
            self.handler(replace(built, id=parent_id), self.ctx)
        finally:
            self.conn.execute("DELETE FROM jobs WHERE id = %s", (parent_id,))
            self.conn.commit()

    def rows(self, sql: str, *params: object) -> list[tuple[Any, ...]]:
        rows = self.conn.execute(sql, params).fetchall()
        self.conn.commit()
        return rows

    def count(self, table: str) -> int:
        return int(self.rows(f"SELECT count(*) FROM {table}")[0][0])

    def analyze_jobs(self) -> list[tuple[Any, ...]]:
        return self.rows(
            "SELECT state, dedupe_key, priority FROM jobs WHERE kind = 'analyze' ORDER BY id"
        )

    def media(self) -> list[tuple[Any, ...]]:
        return self.rows("SELECT video_id, path, bytes, created_at, expires_at FROM media")

    def file(self, rel: str = REL) -> Path:
        return self.audio_dir / rel

    def nothing_written(self) -> bool:
        return all(self.count(t) == 0 for t in ("transcripts", "transcript_chunks", "jobs"))


@pytest.fixture
def make_rig(
    conn: psycopg.Connection[Any], head_dsn: str, tmp_path: Path
) -> Callable[..., Rig]:
    def factory(**kwargs: Any) -> Rig:
        return Rig(conn, head_dsn, tmp_path, **kwargs)

    return factory


# --- before any expensive work -------------------------------------------------------


def test_a_video_without_a_videos_row_is_a_bug_before_downloading(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(with_video=False)

    with pytest.raises(BugError):
        rig.run()

    assert rig.audio.calls == []
    assert rig.transcriber.calls == []


def test_an_existing_whisper_transcript_skips_audio_and_speech_to_text(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    save_transcript(rig.conn, VID, "whisper", "en", "none", segs(), {"rtf": 1.0})
    rig.conn.commit()

    rig.run(priority=10)

    assert rig.audio.calls == []
    assert rig.transcriber.calls == []
    assert rig.count("transcripts") == 1
    assert rig.count("transcript_chunks") == 1  # made for the current strategy
    assert rig.analyze_jobs() == [("pending", ANALYZE_KEY, 10)]
    assert rig.media() == []


def test_an_existing_whisper_transcript_with_chunks_gets_no_duplicate_chunks(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    rig.run()
    rig.run()  # a retry after the first run finished

    assert rig.count("transcripts") == 1
    assert rig.count("transcript_chunks") == 1
    assert len(rig.transcriber.calls) == 1
    assert len(rig.analyze_jobs()) == 1


def test_an_existing_silent_whisper_transcript_enqueues_nothing(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    save_transcript(rig.conn, VID, "whisper", "en", "none", (), None)
    rig.conn.commit()

    rig.run()

    assert rig.transcriber.calls == []
    assert rig.analyze_jobs() == []
    assert rig.count("transcript_chunks") == 0


@pytest.mark.parametrize("source", ["youtube_manual", "youtube_auto"])
def test_a_youtube_transcript_does_not_short_circuit_transcription(
    make_rig: Callable[..., Rig], source: str
) -> None:
    rig = make_rig()
    save_transcript(rig.conn, VID, source, "en", "none", segs(), None)
    rig.conn.commit()

    rig.run()

    assert len(rig.audio.calls) == 1
    assert len(rig.transcriber.calls) == 1
    assert rig.rows("SELECT source FROM transcripts ORDER BY source") == sorted(
        [(source,), ("whisper",)]
    )


# --- audio and media -------------------------------------------------------------------


def test_audio_is_fetched_into_audio_dir_and_registered(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(AUDIO_TTL_DAYS=12)

    rig.run()

    assert rig.audio.calls == [(VID, rig.audio_dir)]
    ((video_id, path, size, created_at, expires_at),) = rig.media()
    assert (video_id, path, size) == (VID, REL, 4)
    assert (expires_at - created_at).total_seconds() == 12 * 86400


def test_the_media_row_is_committed_before_speech_to_text_starts(
    make_rig: Callable[..., Rig],
) -> None:
    seen: list[list[tuple[Any, ...]]] = []

    def probe(_: AudioRef) -> None:
        seen.append(rig.media())  # another connection: sees committed rows only

    rig = make_rig(transcriber=FakeTranscriber(on_call=probe))

    rig.run()

    assert [[r[1] for r in rows] for rows in seen] == [[REL]]


def test_no_transaction_is_open_while_speech_to_text_runs(
    make_rig: Callable[..., Rig],
) -> None:
    statuses: list[TransactionStatus] = []

    def probe(_: AudioRef) -> None:
        statuses.extend(c.info.transaction_status for c in rig.opened if not c.closed)

    rig = make_rig(transcriber=FakeTranscriber(on_call=probe))

    rig.run()

    assert statuses and all(s == TransactionStatus.IDLE for s in statuses)


def test_a_retry_updates_the_media_row_instead_of_raising(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(transcriber=FakeTranscriber(TransientNetworkError("first try")))
    with pytest.raises(TransientNetworkError):
        rig.run()
    (first,) = rig.media()

    rig.audio.rel_path = "cd/abcdefghijk.opus"
    rig.transcriber.outcome = result()
    rig.run()

    (second,) = rig.media()
    assert second[1] == "cd/abcdefghijk.opus"
    assert second[3] > first[3]  # created_at moved


@pytest.mark.parametrize(
    "exc",
    [
        ResourceError("no space"),
        TransientNetworkError("down"),
        ToolFailureError("ffmpeg"),
        PermanentSourceError("removed"),
    ],
)
def test_a_failed_fetch_writes_no_media_row_and_propagates_unchanged(
    make_rig: Callable[..., Rig], exc: JobError
) -> None:
    rig = make_rig(audio=FakeAudio(exc=exc))

    with pytest.raises(type(exc)) as raised:
        rig.run()

    assert raised.value is exc
    assert rig.media() == []
    assert rig.transcriber.calls == []


def test_a_permanent_source_error_marks_the_video_unavailable_and_commits_first(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(audio=FakeAudio(exc=PermanentSourceError("private")))

    with pytest.raises(PermanentSourceError):
        rig.run()

    assert rig.rows("SELECT unavailable FROM videos WHERE video_id = %s", VID) == [("private",)]


def test_a_retry_always_fetches_audio_again(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(transcriber=FakeTranscriber(TransientNetworkError("x")))
    for _ in range(2):
        with pytest.raises(TransientNetworkError):
            rig.run()

    assert len(rig.audio.calls) == 2


# --- speech-to-text --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected", "warns"),
    [
        ({"language": "uk"}, "uk", False),
        ({}, None, False),
        ({"language": ""}, None, True),
        ({"language": 5}, None, True),
        ({"language": None}, None, True),
        ({"language": ["en"]}, None, True),
        ({"language": "de", "future_key": {"a": 1}}, "de", False),
    ],
)
def test_language_comes_from_the_payload_only_when_it_is_a_non_empty_string(
    make_rig: Callable[..., Rig], payload: dict[str, Any], expected: str | None, warns: bool
) -> None:
    rig = make_rig()

    rig.run(payload=payload)

    assert [c["language"] for c in rig.transcriber.calls] == [expected]
    assert bool(rig.logs.records("warning")) is warns


def test_the_transcriber_gets_the_fetched_audio_and_is_called_once(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(audio=FakeAudio(duration=77.5))

    rig.run()

    (call,) = rig.transcriber.calls
    assert call["audio"] == AudioRef(REL, 4, 77.5)


def test_cancellation_is_checked_after_the_fetch_before_speech_to_text(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    rig.ctx._cancel()

    with pytest.raises(Cancelled):
        rig.run()

    assert len(rig.audio.calls) == 1
    assert rig.transcriber.calls == []


def test_progress_reaches_the_cancellation_check_and_cancelled_writes_nothing(
    make_rig: Callable[..., Rig],
) -> None:
    def cancel_midway(_: AudioRef) -> None:
        rig.ctx._cancel()

    # cancel *after* the pre-STT check has passed: only on_progress can notice
    rig = make_rig(transcriber=FakeTranscriber(on_call=cancel_midway, progress=(1.0,)))

    with pytest.raises(Cancelled):
        rig.run()

    assert len(rig.transcriber.calls) == 1
    assert rig.count("transcripts") == 0
    assert rig.count("transcript_chunks") == 0
    assert rig.count("jobs") == 0


def test_a_thousand_progress_calls_produce_at_most_eleven_progress_log_lines(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(
        audio=FakeAudio(duration=1000.0),
        transcriber=FakeTranscriber(progress=tuple(float(i) for i in range(1000))),
    )

    rig.run()

    lines = [r for r in rig.logs.records("info") if "progress" in str(r.get("event"))]
    assert 1 < len(lines) <= 11


def test_the_handler_reports_a_quiet_stretch_using_its_injected_clock(
    make_rig: Callable[..., Rig],
) -> None:
    now = [0.0]

    def tick(done: float) -> None:
        now[0] += HEARTBEAT_SEC

    class Ticking(FakeTranscriber):
        def transcribe(self, audio: AudioRef, **kwargs: Any) -> TranscriptResult:
            on_progress = kwargs["on_progress"]
            for done in (1.0, 2.0, 3.0):
                tick(done)
                on_progress(done)
            return result()

    rig = make_rig(audio=FakeAudio(duration=1000.0), transcriber=Ticking(), clock=lambda: now[0])

    rig.run()

    lines = [r for r in rig.logs.records("info") if r.get("event") == "transcribe progress"]
    assert [r["done_sec"] for r in lines] == [1.0, 2.0, 3.0]  # step 0, then two heartbeats


def test_the_handler_never_heartbeats(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(transcriber=FakeTranscriber(progress=(1.0, 2.0, 3.0)))

    rig.run()

    assert rig.queue.heartbeats == 0


# --- persist, chunk, hand off ------------------------------------------------------------


def test_the_transcript_is_saved_as_whisper_with_engine_meta_verbatim(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(transcriber=FakeTranscriber(result(language="uk")))

    rig.run()

    assert rig.rows(
        "SELECT source, speaker_source, language, engine_meta, full_text FROM transcripts"
    ) == [
        (
            "whisper",
            "none",
            "uk",
            {"engine": "fake", "rtf": 0.25, "model": "tiny"},
            "word0 word1 word2",
        )
    ]
    assert rig.rows("SELECT DISTINCT chunk_strategy FROM transcript_chunks") == [("time:900:60",)]


def test_chunks_use_the_configured_strategy(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(CHUNK_SEC=100, OVERLAP_SEC=10)

    rig.run()

    assert rig.rows("SELECT DISTINCT chunk_strategy FROM transcript_chunks") == [("time:100:10",)]


@pytest.mark.parametrize(("audio_sec", "chunks"), [(899, 1), (900, 1), (901, 2)])
def test_real_chunker_boundaries_at_chunk_sec(
    make_rig: Callable[..., Rig], audio_sec: int, chunks: int
) -> None:
    rig = make_rig(
        audio=FakeAudio(duration=float(audio_sec)),
        transcriber=FakeTranscriber(result(segs(audio_sec))),
    )

    rig.run()

    assert rig.count("transcript_chunks") == chunks


def test_a_failure_saving_chunks_leaves_no_transcript_and_no_chunks(
    make_rig: Callable[..., Rig],
) -> None:
    def bad_chunker(segments: Any, *, chunk_sec: int, overlap_sec: int) -> list[Chunk]:
        return [Chunk(0, 0.0, 1.0, "a"), Chunk(2**40, 1.0, 2.0, "b")]  # seq overflows INTEGER

    rig = make_rig(chunker=bad_chunker)

    with pytest.raises(psycopg.Error):
        rig.run()

    assert rig.nothing_written()


def test_zero_segments_saves_an_empty_transcript_warns_and_enqueues_nothing(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(transcriber=FakeTranscriber(result(())))

    rig.run()

    assert rig.rows("SELECT source, full_text FROM transcripts") == [("whisper", "")]
    assert rig.count("transcript_chunks") == 0
    assert rig.analyze_jobs() == []
    assert len(rig.logs.records("warning")) == 1


@pytest.mark.parametrize("priority", [10, 0, -10])
def test_analyze_is_enqueued_with_the_shared_key_and_inherited_priority(
    make_rig: Callable[..., Rig], priority: int
) -> None:
    rig = make_rig()

    rig.run(priority=priority)

    assert rig.analyze_jobs() == [("pending", ANALYZE_KEY, priority)]
    (enqueue,) = rig.queue.enqueues
    assert enqueue[:2] == ("analyze", VID)


def test_an_already_active_analyze_job_is_success(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    rig.conn.execute(
        "INSERT INTO jobs (video_id, kind, dedupe_key, state) VALUES (%s, 'analyze', %s, 'pending')",
        (VID, ANALYZE_KEY),
    )
    rig.conn.commit()

    rig.run()

    assert len(rig.analyze_jobs()) == 1
    assert rig.count("transcripts") == 1


def test_a_crash_between_commit_and_enqueue_is_repaired_without_repeating_speech_to_text(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    rig.queue.fail_enqueue = True
    with pytest.raises(ConnectionError):
        rig.run()
    assert rig.count("transcripts") == 1

    rig.queue.fail_enqueue = False
    rig.run()

    assert len(rig.transcriber.calls) == 1
    assert rig.count("transcripts") == 1
    assert rig.count("transcript_chunks") == 1
    assert len(rig.analyze_jobs()) == 1


# --- audio retention flag ------------------------------------------------------------------


def test_audio_keep_true_leaves_the_file_and_the_row(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(AUDIO_KEEP=True)

    rig.run()

    assert rig.file().exists()
    assert len(rig.media()) == 1


def test_audio_keep_false_deletes_the_file_then_the_row_on_success(
    make_rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = make_rig(AUDIO_KEEP=False)
    file_existed_at_row_delete: list[bool] = []

    def spy(conn: psycopg.Connection[Any], video_id: str) -> None:
        file_existed_at_row_delete.append(rig.file().exists())
        real_delete_media(conn, video_id)

    monkeypatch.setattr("services.transcriber.transcribe.delete_media", spy)

    rig.run()

    assert file_existed_at_row_delete == [False]
    assert not rig.file().exists()
    assert rig.media() == []


@pytest.mark.parametrize("failure", ["error", "cancelled"])
def test_audio_keep_false_cleans_up_on_a_raised_error_and_on_cancel(
    make_rig: Callable[..., Rig], failure: str
) -> None:
    exc: BaseException = TransientNetworkError("x") if failure == "error" else Cancelled()
    rig = make_rig(AUDIO_KEEP=False, transcriber=FakeTranscriber(exc))

    with pytest.raises(type(exc)):
        rig.run()

    assert not rig.file().exists()
    assert rig.media() == []


def test_audio_keep_false_cleans_up_after_a_save_failure(make_rig: Callable[..., Rig]) -> None:
    def bad_chunker(segments: Any, *, chunk_sec: int, overlap_sec: int) -> list[Chunk]:
        return [Chunk(2**40, 0.0, 1.0, "b")]

    rig = make_rig(AUDIO_KEEP=False, chunker=bad_chunker)

    with pytest.raises(psycopg.Error):
        rig.run()

    assert not rig.file().exists()
    assert rig.media() == []


def test_audio_keep_true_keeps_audio_after_an_error(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(AUDIO_KEEP=True, transcriber=FakeTranscriber(TransientNetworkError("x")))

    with pytest.raises(TransientNetworkError):
        rig.run()

    assert rig.file().exists()
    assert len(rig.media()) == 1


def test_a_file_that_is_already_gone_is_not_an_error(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(AUDIO_KEEP=False, audio=FakeAudio(write=False))

    rig.run()

    assert rig.media() == []


def test_deletion_only_touches_files_under_audio_dir(
    make_rig: Callable[..., Rig], tmp_path: Path
) -> None:
    outside = tmp_path / "precious.opus"
    outside.write_bytes(b"keep me")
    rig = make_rig(AUDIO_KEEP=False, audio=FakeAudio(rel_path="../precious.opus", write=False))

    rig.run()

    assert outside.read_bytes() == b"keep me"


# --- retries through the real queue ---------------------------------------------------------


def claim_and_run(rig: Rig) -> None:
    with rig.queue.inner.claim(["transcribe"], worker="w") as job:
        assert job is not None
        rig.handler(job, rig.ctx)


def job_state(rig: Rig) -> tuple[str, int]:
    ((state, attempts),) = rig.rows("SELECT state, attempts FROM jobs WHERE kind = 'transcribe'")
    return state, attempts


@pytest.mark.parametrize("exc", [TransientNetworkError("down"), ToolFailureError("broken")])
def test_a_transcribe_job_dead_letters_after_the_second_attempt(
    make_rig: Callable[..., Rig], exc: JobError
) -> None:
    rig = make_rig(transcriber=FakeTranscriber(exc))
    rig.queue.inner.enqueue("transcribe", VID)

    claim_and_run(rig)
    assert job_state(rig) == ("pending", 1)
    assert rig.rows("SELECT run_after > now() FROM jobs") == [(True,)]  # backoff

    rig.conn.execute("UPDATE jobs SET run_after = now()")
    rig.conn.commit()
    claim_and_run(rig)

    assert job_state(rig) == ("dead", 2)


# --- progress and estimate (issue #245) ------------------------------------------------


def _stored_progress(rig: Rig) -> Any:
    """What another connection sees in ``jobs.progress`` right now (the one running job)."""
    with psycopg.connect(rig.dsn) as other:
        rows = other.execute("SELECT progress FROM jobs WHERE state = 'running'").fetchall()
    assert len(rows) == 1
    return rows[0][0]


def _add_rtf_history(rig: Rig, rtf: float, *, model: str = "tiny", n: int = 1) -> None:
    other = f"hist{n:07d}"
    rig.add_video(other)
    save_transcript(
        rig.conn, other, "whisper", "en", "none", segs(),
        {"model": model, "compute_type": "int8", "rtf": rtf},
    )
    rig.conn.commit()


def test_the_estimate_is_stored_before_speech_to_text_from_the_rtf_history(
    make_rig: Callable[..., Rig],
) -> None:
    seen: list[Any] = []
    rig = make_rig(
        audio=FakeAudio(duration=120.0),
        WHISPER_MODEL="tiny",
        WHISPER_COMPUTE="int8",
    )
    rig.transcriber.on_call = lambda _audio: seen.append(_stored_progress(rig))
    _add_rtf_history(rig, 0.5, n=1)
    _add_rtf_history(rig, 9.0, model="large-v3", n=2)  # another model: not in the median

    rig.run(payload={"duration_sec": 100})

    (progress,) = seen
    assert progress["estimated_sec"] == 60.0  # the measured 120 s x 0.5, not metadata's 100
    assert progress["estimate_source"] == "history"
    assert (progress["done_sec"], progress["duration_sec"]) == (0.0, 120.0)


def test_without_history_the_estimate_is_the_default_and_marked_so(
    make_rig: Callable[..., Rig],
) -> None:
    seen: list[Any] = []
    rig = make_rig(WHISPER_MODEL="tiny", WHISPER_COMPUTE="int8")
    rig.transcriber.on_call = lambda _audio: seen.append(_stored_progress(rig))

    rig.run()

    assert seen[0]["estimate_source"] == "default"
    assert seen[0]["estimated_sec"] > 0
