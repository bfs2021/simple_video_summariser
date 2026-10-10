"""The analyzer entrypoint against a real Postgres (issue #33).

``main`` runs its real claim loop and real ``PostgresQueue`` with the real
analyze handler (#30); only the summarizer is a ``FakeSummarizer`` (injected
through ``summarizer_factory``) and the worker's sleep is a no-op, so nothing
needs Ollama, an API key or a clock.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from common.errors import LLMUnavailableError
from common.models import Segment, VideoMeta
from common.queue import PostgresQueue
from common.repo.transcripts import save_transcript
from common.repo.videos import upsert_video
from services.analyzer import main as analyzer_main
from tests.services.analyzer.fakes import FakeSummarizer
from tests.services.transcriber.fakes import make_settings

pytestmark = pytest.mark.integration

SEGMENTS = (
    Segment(0, 10, "Alpha opening."),
    Segment(100, 110, "Beta middle."),
    Segment(950, 960, "Gamma later."),
)


def seed_video(conn: psycopg.Connection[Any], video_id: str) -> None:
    meta = VideoMeta(
        video_id=video_id,
        channel_id="UC" + "a" * 22,
        title=f"Title {video_id}",
        description="d",
        duration_sec=2000,
        published_at=None,
        language=None,
        live_status=None,
        manual_subtitle_langs=(),
        auto_caption_langs=(),
    )
    upsert_video(conn, meta, "adhoc")
    save_transcript(conn, video_id, "whisper", "en", "none", SEGMENTS, None)
    conn.commit()


class Rig:
    def __init__(self, conn: psycopg.Connection[Any], dsn: str, tmp_path: Path) -> None:
        self.conn = conn
        self.dsn = dsn
        self.tmp_path = tmp_path
        self.queue = PostgresQueue(conn, settings=make_settings())

    def enqueue(self, kind: str, video_id: str, dedupe_key: str = "default") -> int:
        job_id = self.queue.enqueue(kind, video_id, dedupe_key=dedupe_key)
        assert job_id is not None
        return job_id

    def run(self, summarizer: FakeSummarizer, *, iterations: int, **settings: Any) -> int:
        return analyzer_main.main(
            settings=make_settings(**settings),
            connect=lambda: psycopg.connect(self.dsn),
            summarizer_factory=lambda s: summarizer,
            worker_options={
                "liveness_path": self.tmp_path / "heartbeat",
                "install_signal_handlers": False,
                "sleep": lambda _s: None,
            },
            max_iterations=iterations,
        )

    def job(self, job_id: int) -> dict[str, Any]:
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT *, run_after - now() AS wait FROM jobs WHERE id = %s", (job_id,)
            )
            row = cur.fetchone()
        self.conn.rollback()
        assert row is not None
        return row

    def analyses(self) -> list[dict[str, Any]]:
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT video_id, prompt_version, model FROM analyses ORDER BY id")
            rows = cur.fetchall()
        self.conn.rollback()
        return rows


@pytest.fixture
def rig(conn: psycopg.Connection[Any], head_dsn: str, tmp_path: Path) -> Rig:
    return Rig(conn, head_dsn, tmp_path)


def test_the_analyzer_claims_only_analyze_jobs(rig: Rig) -> None:
    seed_video(rig.conn, "aaaaaaaaaaa")
    ingest = rig.enqueue("ingest", "aaaaaaaaaaa")
    transcribe = rig.enqueue("transcribe", "aaaaaaaaaaa")
    analyze = rig.enqueue("analyze", "aaaaaaaaaaa", "v1:fake")

    assert rig.run(FakeSummarizer(name="fake"), iterations=3) == 0

    assert rig.job(analyze)["state"] == "done"
    for other in (ingest, transcribe):
        row = rig.job(other)
        assert (row["state"], row["attempts"], row["locked_by"]) == ("pending", 0, None)


def test_a_pending_analyze_job_ends_done_with_one_analysis_row(rig: Rig) -> None:
    seed_video(rig.conn, "aaaaaaaaaaa")
    job_id = rig.enqueue("analyze", "aaaaaaaaaaa", "v1:fake")
    fake = FakeSummarizer(name="fake", model="fake-model-7")

    rig.run(fake, iterations=2)

    assert rig.job(job_id)["state"] == "done"
    assert rig.analyses() == [
        {"video_id": "aaaaaaaaaaa", "prompt_version": "v1", "model": "fake-model-7"}
    ]


def test_a_job_for_another_configuration_is_never_claimed_nor_deferred(rig: Rig) -> None:
    # Issue #100: the analyzer claims by its own dedupe key, so a job for another prompt
    # version is left exactly as it was (it used to be claimed and pushed out 15 minutes).
    seed_video(rig.conn, "aaaaaaaaaaa")
    job_id = rig.enqueue("analyze", "aaaaaaaaaaa", "v2:fake")
    fake = FakeSummarizer(name="fake")

    rig.run(fake, iterations=2)

    row = rig.job(job_id)
    assert (row["state"], row["attempts"], row["locked_by"]) == ("pending", 0, None)
    assert row["wait"].total_seconds() <= 0  # run_after was not pushed into the future
    assert fake.calls == []
    assert rig.analyses() == []


@pytest.mark.parametrize("key", ["default", "v1:fake:extra", "fake"])
def test_a_malformed_key_is_never_claimed(rig: Rig, key: str) -> None:
    # Issue #100: no analyzer's own key can equal a malformed one, so the claim filter
    # skips it (the BugError guard in ``build_job_handler`` is unit-tested in test_main.py).
    seed_video(rig.conn, "aaaaaaaaaaa")
    job_id = rig.enqueue("analyze", "aaaaaaaaaaa", key)
    fake = FakeSummarizer(name="fake")

    rig.run(fake, iterations=2)

    row = rig.job(job_id)
    assert (row["state"], row["attempts"], row["error_class"]) == ("pending", 0, None)
    assert fake.calls == []


def test_two_analyzers_on_one_database_each_finish_only_their_own_jobs(
    rig: Rig, capsys: pytest.CaptureFixture[str]
) -> None:
    for video in ("aaaaaaaaaaa", "bbbbbbbbbbb"):
        seed_video(rig.conn, video)
    ollama = [rig.enqueue("analyze", v, "v1:ollama") for v in ("aaaaaaaaaaa", "bbbbbbbbbbb")]
    anthropic = [rig.enqueue("analyze", v, "v1:anthropic") for v in ("aaaaaaaaaaa", "bbbbbbbbbbb")]
    ollama_fake = FakeSummarizer(name="ollama", model="m-ollama")
    anthropic_fake = FakeSummarizer(name="anthropic", model="m-anthropic")

    # The ollama analyzer runs while the anthropic jobs are already pending, and the other
    # way round: neither may touch a job that is not its own.
    rig.run(ollama_fake, iterations=3)
    assert [rig.job(j)["state"] for j in ollama] == ["done", "done"]
    assert [(rig.job(j)["state"], rig.job(j)["attempts"]) for j in anthropic] == [("pending", 0)] * 2
    rig.run(anthropic_fake, iterations=3)

    assert [rig.job(j)["state"] for j in ollama + anthropic] == ["done"] * 4
    assert [c[0] for c in ollama_fake.calls] and [c[0] for c in anthropic_fake.calls]
    assert sorted((r["video_id"], r["model"]) for r in rig.analyses()) == [
        ("aaaaaaaaaaa", "m-anthropic"),
        ("aaaaaaaaaaa", "m-ollama"),
        ("bbbbbbbbbbb", "m-anthropic"),
        ("bbbbbbbbbbb", "m-ollama"),
    ]
    output = capsys.readouterr()
    assert "analyzer.job_config_mismatch" not in output.out + output.err
    for job_id in ollama + anthropic:
        assert rig.job(job_id)["wait"].total_seconds() <= 0  # nothing pushed out by MISMATCH_DEFER


def test_an_unavailable_summarizer_is_recorded_and_the_worker_goes_on(rig: Rig) -> None:
    seed_video(rig.conn, "aaaaaaaaaaa")
    seed_video(rig.conn, "bbbbbbbbbbb")
    first = rig.enqueue("analyze", "aaaaaaaaaaa", "v1:fake")
    second = rig.enqueue("analyze", "bbbbbbbbbbb", "v1:fake")
    calls = 0

    def ollama_is_down_for_the_first_call(kind: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMUnavailableError("connection refused")

    fake = FakeSummarizer(name="fake", on_call=ollama_is_down_for_the_first_call)

    assert rig.run(fake, iterations=3) == 0

    failed = rig.job(first)
    assert (failed["state"], failed["error_class"], failed["attempts"]) == (
        "pending",
        "LLM_UNAVAILABLE",
        1,
    )
    assert failed["wait"].total_seconds() > 0  # the kind's backoff
    assert rig.job(second)["state"] == "done"
    assert [row["video_id"] for row in rig.analyses()] == ["bbbbbbbbbbb"]
