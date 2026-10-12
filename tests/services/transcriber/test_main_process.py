"""Transcriber entrypoint as a real process against a real Postgres (issue #32).

Each test starts ``tests/services/transcriber/harness.py`` in a subprocess. It
runs the real ``main()`` with fake handlers, so real signals reach the real
``Worker`` and ``PostgresQueue``, and ``kill -9`` leaves a genuinely orphaned job.
Nothing waits on a fixed sleep: the tests wait on the harness's stdout lines
(``started`` / ``finished``) and poll the database with a timeout.

The last test runs the production handler builder in-process, with fakes for the
network tools, to show that an ingest job hands off to transcribe only through
the queue, so a different worker can run the transcribe job.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest

from common.models import AudioRef, Segment, TranscriptResult
from common.queue import PostgresQueue
from services.transcriber.main import (
    LazyTranscriber,
    QueuePool,
    build_handlers,
    build_worker,
    production_queue_factory,
)
from tests.services.transcriber.fakes import (
    MANUAL_ID,
    FakeMetadata,
    FakeSubtitles,
    fixture_meta,
    make_settings,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
WAIT_SEC = 60.0


class Replica:
    """One harness subprocess, with its stdout collected by a reader thread."""

    def __init__(self, mode: str) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "tests.services.transcriber.harness", mode],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.lines: list[str] = []
        self._changed = threading.Condition()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            with self._changed:
                self.lines.append(line.rstrip("\n"))
                self._changed.notify_all()

    def wait_for(self, predicate: Callable[[str], bool], what: str) -> str:
        def found() -> bool:
            return any(predicate(line) for line in self.lines)

        with self._changed:
            if not self._changed.wait_for(found, timeout=WAIT_SEC):
                raise AssertionError(f"timed out waiting for {what}; output:\n" + "\n".join(self.lines))
            return next(line for line in self.lines if predicate(line))

    def wait_started(self) -> tuple[int, str]:
        """The job id and worker name of the first ``started`` line."""
        line = self.wait_for(lambda text: text.startswith("started "), "a handler to start")
        _, job_id, locked_by = line.split(" ", 2)
        return int(job_id), locked_by

    def started(self) -> list[tuple[int, str]]:
        return [
            (int(line.split(" ", 2)[1]), line.split(" ", 2)[2])
            for line in list(self.lines)
            if line.startswith("started ")
        ]

    @property
    def name(self) -> str:
        line = self.wait_for(lambda text: '"transcriber.started"' in text, "the startup log line")
        worker = json.loads(line)["worker"]
        assert isinstance(worker, str)
        return worker

    def release(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write("release\n")
        self.proc.stdin.flush()

    def sigterm(self) -> None:
        self.proc.send_signal(signal.SIGTERM)

    def sigkill(self) -> int:
        self.proc.send_signal(signal.SIGKILL)
        return self.proc.wait(timeout=WAIT_SEC)

    def wait_exit(self, timeout: float) -> int:
        return self.proc.wait(timeout=timeout)


@pytest.fixture
def replica_factory(
    head_dsn: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[str], Replica]]:
    monkeypatch.setenv("AUDIO_DIR", str(tmp_path / "audio"))
    monkeypatch.setenv("LOG_FORMAT", "json")
    started: list[Replica] = []

    def start(mode: str) -> Replica:
        replica = Replica(mode)
        started.append(replica)
        return replica

    yield start
    for replica in started:
        if replica.proc.poll() is None:
            replica.proc.kill()
        replica.proc.wait()
        if replica.proc.stdin is not None:
            replica.proc.stdin.close()


@pytest.fixture
def db(head_dsn: str) -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(head_dsn, autocommit=True) as connection:
        yield connection


def enqueue(db: psycopg.Connection[Any], kind: str, video_id: str) -> int:
    job_id = PostgresQueue(db, settings=make_settings()).enqueue(kind, video_id)
    assert job_id is not None
    return job_id


def job_row(db: psycopg.Connection[Any], job_id: int) -> tuple[str, int, str | None]:
    row = db.execute(
        "SELECT state, attempts, locked_by FROM jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row is not None
    return (row[0], row[1], row[2])


def wait_until(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + WAIT_SEC
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def age_heartbeat(db: psycopg.Connection[Any], job_id: int) -> None:
    db.execute("UPDATE jobs SET heartbeat_at = now() - interval '1 hour' WHERE id = %s", (job_id,))


def reap(db: psycopg.Connection[Any]) -> int:
    return PostgresQueue(db, settings=make_settings()).reap_stale()


# ---------------------------------------------------------------------------
# SIGTERM
# ---------------------------------------------------------------------------


def test_sigterm_mid_job_lets_it_finish_and_claims_nothing_more(
    replica_factory: Callable[[str], Replica], db: psycopg.Connection[Any]
) -> None:
    first = enqueue(db, "ingest", "vid00000001")
    second = enqueue(db, "ingest", "vid00000002")
    replica = replica_factory("block")
    started_id, _ = replica.wait_started()
    assert started_id == first

    replica.sigterm()
    replica.release()

    assert replica.wait_exit(WAIT_SEC) == 0
    assert job_row(db, first)[0] == "done"
    assert job_row(db, second) == ("pending", 0, None)


def test_sigterm_past_the_grace_period_hands_the_job_back_without_spending_an_attempt(
    replica_factory: Callable[[str], Replica],
    db: psycopg.Connection[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORKER_SHUTDOWN_GRACE_SEC", "1")
    job_id = enqueue(db, "ingest", "vid00000001")
    attempts_before = job_row(db, job_id)[1]
    replica = replica_factory("cooperative")
    replica.wait_started()
    assert job_row(db, job_id)[1] == attempts_before + 1

    replica.sigterm()

    assert replica.wait_exit(1 + 10) == 0
    assert job_row(db, job_id) == ("pending", attempts_before, None)


# ---------------------------------------------------------------------------
# kill -9 and the reaper
# ---------------------------------------------------------------------------


def test_a_job_left_by_kill_9_is_reaped_and_finished_by_another_worker(
    replica_factory: Callable[[str], Replica], db: psycopg.Connection[Any]
) -> None:
    job_id = enqueue(db, "ingest", "vid00000001")
    victim = replica_factory("block")
    victim.wait_started()
    victim_name = victim.name

    victim.sigkill()

    assert job_row(db, job_id) == ("running", 1, victim_name)
    age_heartbeat(db, job_id)
    assert reap(db) == 1
    assert job_row(db, job_id)[0] == "pending"
    # The reap backs the job off (#87); expire that backoff without sleeping.
    db.execute("UPDATE jobs SET run_after = now() - interval '1 second' WHERE id = %s", (job_id,))

    rescuer = replica_factory("instant")
    started_id, locked_by = rescuer.wait_started()
    wait_until(lambda: job_row(db, job_id)[0] == "done", "the reaped job to finish")
    assert started_id == job_id
    assert locked_by == rescuer.name
    assert locked_by != victim_name
    assert job_row(db, job_id)[1] == 2


def test_kill_9_on_the_final_transcribe_attempt_dead_letters_the_job(
    replica_factory: Callable[[str], Replica], db: psycopg.Connection[Any]
) -> None:
    job_id = enqueue(db, "transcribe", "vid00000001")
    db.execute("UPDATE jobs SET attempts = 1 WHERE id = %s", (job_id,))
    victim = replica_factory("block")
    victim.wait_started()
    assert job_row(db, job_id)[1] == 2  # MAX_ATTEMPTS_TRANSCRIBE

    victim.sigkill()
    age_heartbeat(db, job_id)

    assert reap(db) == 1
    assert job_row(db, job_id)[0] == "dead"


# ---------------------------------------------------------------------------
# replicas
# ---------------------------------------------------------------------------


def test_two_replicas_handle_every_job_exactly_once(
    replica_factory: Callable[[str], Replica], db: psycopg.Connection[Any]
) -> None:
    job_ids = {enqueue(db, "ingest", f"vid{n:08d}") for n in range(10)}
    one = replica_factory("gate")
    two = replica_factory("gate")

    # Each replica holds its first job until released, so both took part.
    first_one, name_one = one.wait_started()
    first_two, name_two = two.wait_started()
    assert first_one != first_two
    assert name_one != name_two
    assert {name_one, name_two} == {one.name, two.name}
    one.release()
    two.release()

    wait_until(
        lambda: db.execute("SELECT count(*) FROM jobs WHERE state = 'done'").fetchone() == (10,),
        "all ten jobs to finish",
    )
    wait_until(lambda: len(one.started()) + len(two.started()) == 10, "the started lines")
    handled = [job_id for job_id, _ in one.started() + two.started()]
    assert sorted(handled) == sorted(job_ids)
    attempts = db.execute("SELECT DISTINCT attempts FROM jobs").fetchall()
    assert attempts == [(1,)]


# ---------------------------------------------------------------------------
# ingest -> transcribe goes through the queue only (production handlers, fake tools)
# ---------------------------------------------------------------------------


class FakeAudio:
    def fetch_normalized(self, video_id: str, dest: Path) -> AudioRef:
        rel_path = f"{video_id[:2]}/{video_id}.opus"
        target = dest / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"opus")
        return AudioRef(rel_path, 4, 3.0)


class FakeWhisper:
    def transcribe(
        self,
        audio: AudioRef,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> TranscriptResult:
        segments = (Segment(0.0, 1.0, "hello"), Segment(1.0, 2.0, "world"))
        return TranscriptResult(segments=segments, language="en", engine_meta={"engine": "fake"})


def test_ingest_leaves_a_transcribe_job_that_a_different_worker_runs(
    head_dsn: str, db: psycopg.Connection[Any], tmp_path: Path
) -> None:
    settings = make_settings(PREFER_WHISPER=True, AUDIO_DIR=tmp_path / "audio")
    builds: list[int] = []

    def build_transcriber() -> FakeWhisper:
        builds.append(1)
        return FakeWhisper()

    def connect() -> psycopg.Connection[Any]:
        return psycopg.connect(head_dsn)

    handlers = build_handlers(
        settings,
        transcriber=LazyTranscriber(build_transcriber),
        connect=connect,
        audio_source=FakeAudio(),
        metadata=FakeMetadata(fixture_meta("manual_en", MANUAL_ID)),
        subtitles=FakeSubtitles(),
    )
    ingest_id = enqueue(db, "ingest", MANUAL_ID)

    def run_one(name: str) -> None:
        pool = QueuePool(production_queue_factory(settings, connect=connect))
        try:
            build_worker(
                settings, name=name, handlers=handlers, pool=pool, queue=pool.open_main()
            ).run(max_iterations=1)
        finally:
            pool.close_all()

    run_one("transcriber:host-a:1")

    assert job_row(db, ingest_id)[0] == "done"
    assert builds == []  # an ingest-only process never builds the transcriber
    rows = db.execute(
        "SELECT id, state FROM jobs WHERE video_id = %s AND kind = 'transcribe'", (MANUAL_ID,)
    ).fetchall()
    assert [state for _, state in rows] == ["pending"]

    run_one("transcriber:host-b:2")

    assert job_row(db, rows[0][0])[0] == "done"
    assert builds == [1]
    sources = db.execute(
        "SELECT source FROM transcripts WHERE video_id = %s", (MANUAL_ID,)
    ).fetchall()
    assert sources == [("whisper",)]
