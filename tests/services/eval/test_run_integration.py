"""``ytdigest eval run`` and ``eval report`` against Postgres, with a FakeSummarizer (issue #62)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import psycopg
import pytest

from common.errors import LLMInvalidOutputError
from common.models import Chunk, ChunkAnalysis, Claim, Roster, Segment, VideoMeta
from common.repo.transcripts import save_transcript
from common.repo.videos import upsert_video
from services.cli.main import Deps, main
from tests.common.repo import read_seed as seed
from tests.services.analyzer.fakes import FakeSummarizer
from tests.services.transcriber.fakes import make_settings

pytestmark = pytest.mark.integration

OK1, OK2, NOTRANS, UNKNOWN, BAD = "aaaaaaaaa01", "aaaaaaaaa02", "aaaaaaaaa03", "aaaaaaaaa04", "aaaaaaaaa05"
SEGMENTS = (Segment(0, 10, "Alpha opening."), Segment(100, 110, "Beta."), Segment(950, 960, "Gamma."))


class PickySummarizer(FakeSummarizer):
    """Fails the chunks of one video, like an LLM that returned garbage for it."""

    def __init__(self, bad_video: str = BAD, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.bad_video = bad_video
        self.videos: list[str] = []

    def analyze_chunk(self, chunk: Chunk, roster: Roster, meta: VideoMeta) -> ChunkAnalysis:
        self.videos.append(meta.video_id)
        if meta.video_id == self.bad_video:
            raise LLMInvalidOutputError("not json \x1b[31m")
        return super().analyze_chunk(chunk, roster, meta)


def add_video(conn: psycopg.Connection[Any], video_id: str, *, transcript: bool = True) -> None:
    meta = VideoMeta(
        video_id=video_id, channel_id="UC" + "a" * 22, title=f"Title {video_id}", description="d",
        duration_sec=2000, published_at=None, language=None, live_status=None,
        manual_subtitle_langs=(), auto_caption_langs=(),
    )
    upsert_video(conn, meta, "adhoc")
    if transcript:
        save_transcript(conn, video_id, "whisper", "en", "none", SEGMENTS, None)
    conn.commit()


def deps_for(dsn: str, summarizer: FakeSummarizer) -> Deps:
    return Deps(
        settings=make_settings(), connect=lambda: psycopg.connect(dsn),
        metadata=None, subtitles=None, audio=None, transcriber=None,  # type: ignore[arg-type]
        summarizer=summarizer,
    )


def count(dsn: str, table: str) -> int:
    with psycopg.connect(dsn) as conn:
        row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def write_golden(tmp_path: Path, *ids: str) -> str:
    path = tmp_path / "golden.txt"
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def seeded(head_dsn: str) -> str:
    with psycopg.connect(head_dsn) as conn:
        for vid in (OK1, OK2, BAD):
            add_video(conn, vid)
        add_video(conn, NOTRANS, transcript=False)
        # An unrelated job row: the harness must neither add to nor change it.
        seed.job(conn, OK1, "ingest", "pending", created=0)
        conn.commit()
    return head_dsn


def test_every_status_exit_code_and_stored_rows(
    seeded: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summarizer = PickySummarizer(default_chunk=ChunkAnalysis(topics=(), claims=(Claim(text="c"),), quotes=()))
    deps = deps_for(seeded, summarizer)
    # OK2 is analysed first, so the main run finds it done.
    assert main(["eval", "run", "--golden", write_golden(tmp_path, OK2)], deps=deps) == 0
    summarizer.calls.clear()
    summarizer.videos.clear()
    capsys.readouterr()
    jobs_before = count(seeded, "jobs")

    code = main(
        ["eval", "run", "--golden", write_golden(tmp_path, OK1, OK2, NOTRANS, UNKNOWN, BAD)],
        deps=deps,
    )

    err = capsys.readouterr().err
    lines = {line.split()[1]: line for line in err.splitlines() if line.startswith("[")}
    assert code == 1
    assert lines[OK1].startswith(f"[1/5] {OK1} analyzed (2 chunks, ")
    assert lines[OK2] == f"[2/5] {OK2} skipped: already analyzed"
    assert lines[NOTRANS] == f"[3/5] {NOTRANS} missing: no transcript"
    assert lines[UNKNOWN] == f"[4/5] {UNKNOWN} missing: not in database"
    assert lines[BAD].startswith(f"[5/5] {BAD} failed: LLM_INVALID_OUTPUT: ")
    assert "\x1b" not in err
    assert "summary: analyzed 1, skipped 1, missing 2, failed 1" in err
    assert OK2 not in summarizer.videos  # skipped: zero summarizer calls for it
    assert count(seeded, "jobs") == jobs_before == 1
    with psycopg.connect(seeded) as conn:
        by_video: dict[str, int] = dict(conn.execute("SELECT video_id, count(*) FROM analyses GROUP BY 1").fetchall())
        model, prompt = conn.execute("SELECT model, prompt_version FROM analyses LIMIT 1").fetchone()  # type: ignore[misc]
        in_tokens = conn.execute(
            "SELECT input_tokens, duration_ms FROM analyses WHERE video_id = %s", (OK1,)
        ).fetchone()
    assert by_video == {OK1: 1, OK2: 1}  # nothing, not even a partial row, for the failed video
    assert (model, prompt) == ("fake-model", "v1")
    assert in_tokens is not None and in_tokens[0] > 0 and in_tokens[1] is not None


def test_a_rerun_makes_no_calls_and_adds_no_rows(
    seeded: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summarizer = FakeSummarizer()
    deps = deps_for(seeded, summarizer)
    golden = write_golden(tmp_path, OK1, OK2)
    assert main(["eval", "run", "--golden", golden], deps=deps) == 0
    rows, calls = count(seeded, "analyses"), len(summarizer.calls)
    capsys.readouterr()

    assert main(["eval", "run", "--golden", golden], deps=deps) == 0

    err = capsys.readouterr().err
    assert err.count("skipped: already analyzed") == 2
    assert len(summarizer.calls) == calls
    assert count(seeded, "analyses") == rows == 2


def test_dry_run_writes_nothing_and_calls_no_llm(
    seeded: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summarizer = FakeSummarizer()
    deps = deps_for(seeded, summarizer)
    assert main(["eval", "run", "--golden", write_golden(tmp_path, OK2)], deps=deps) == 0
    summarizer.calls.clear()
    capsys.readouterr()
    before = {t: count(seeded, t) for t in ("analyses", "transcript_chunks", "jobs", "topics", "claims")}
    fresh = ["aaaaaaaaa06"]
    with psycopg.connect(seeded) as conn:
        add_video(conn, fresh[0])

    code = main(
        ["eval", "run", "--dry-run", "--golden", write_golden(tmp_path, OK2, fresh[0], NOTRANS, UNKNOWN)],
        deps=deps,
    )

    err = capsys.readouterr().err
    assert code == 1  # two videos are missing
    assert summarizer.calls == []
    assert {t: count(seeded, t) for t in before} == before
    assert f"{OK2} would skip" in err and "transcript: whisper" in err and "chunks: 2 chunks" in err
    assert f"{fresh[0]} would analyze" in err and "chunks: not chunked yet" in err
    assert "missing: no transcript" in err and "missing: not in database" in err
    assert "dry run: would analyze 1, would skip 1, missing 2" in err


def test_ctrl_c_keeps_finished_analyses_and_exits_130(
    seeded: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class Interrupting(PickySummarizer):
        def analyze_chunk(self, chunk: Chunk, roster: Roster, meta: VideoMeta) -> ChunkAnalysis:
            if meta.video_id == OK2:
                raise KeyboardInterrupt
            return super().analyze_chunk(chunk, roster, meta)

    code = main(
        ["eval", "run", "--golden", write_golden(tmp_path, OK1, OK2)],
        deps=deps_for(seeded, Interrupting()),
    )

    assert code == 130
    with psycopg.connect(seeded) as conn:
        done = [r[0] for r in conn.execute("SELECT video_id FROM analyses").fetchall()]
    assert done == [OK1]
    # Resuming finishes the rest.
    assert main(["eval", "run", "--golden", write_golden(tmp_path, OK1, OK2)], deps=deps_for(seeded, FakeSummarizer())) == 0
    assert "skipped: already analyzed" in capsys.readouterr().err


def test_golden_from_stdin(
    seeded: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{OK1}\n"))

    assert main(["eval", "run", "--golden", "-"], deps=deps_for(seeded, FakeSummarizer())) == 0
    assert f"{OK1} analyzed" in capsys.readouterr().err


def test_a_database_without_the_schema_points_at_the_migrate_command(
    postgres_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["eval", "run", "--golden", write_golden(tmp_path, OK1)], deps=deps_for(postgres_dsn, FakeSummarizer()))

    err = capsys.readouterr().err
    assert code == 1
    assert "docker compose run --rm migrate" in err


# ---------------------------------------------------------------------- report


def test_report_compares_two_model_prompt_pairs(
    seeded: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with psycopg.connect(seeded) as conn:
        t1 = seed.transcript(conn, OK1, "youtube_auto")
        seed.analysis(conn, OK1, t1, created=1, model="qwen3.5:4b", prompt_version="v1",
                      tldr="local tldr", claims=[{"text": "local claim", "start_sec": 12.0}])
        seed.analysis(conn, OK1, t1, created=2, model="claude-haiku-4-5", prompt_version="v1",
                      tldr="cloud tldr", claims=[{"text": "cloud claim", "start_sec": 640.0,
                                                   "speaker": "Ann", "confidence": "high"}])
        seed.analysis(conn, OK2, seed.transcript(conn, OK2, "youtube_auto"), created=1,
                      model="qwen3.5:4b", prompt_version="v1")
        conn.commit()
    before = {t: count(seeded, t) for t in ("analyses", "jobs", "claims")}
    out = tmp_path / "report.md"

    code = main(
        ["eval", "report", "--golden", write_golden(tmp_path, OK1, OK2, UNKNOWN, BAD),
         "--a-model", "qwen3.5:4b", "--a-prompt", "v1",
         "--b-model", "claude-haiku-4-5", "--b-prompt", "v1", "--out", str(out)],
        deps=deps_for(seeded, FakeSummarizer()),
    )

    text = out.read_text(encoding="utf-8")
    assert code == 0
    assert {t: count(seeded, t) for t in before} == before
    assert "local tldr" in text and "cloud tldr" in text
    assert "local claim" in text and "cloud claim" in text
    assert "0:00-5:00" in text and "10:00-15:00" in text
    assert "Missing A: 2" in text and "Missing B: 3" in text
    assert f"{UNKNOWN}: not in the database" in text
    assert f"{BAD}) : no analysis" not in text and "no analysis for either run" in text
    assert "No analysis for claude-haiku-4-5 / v1" in text  # OK2 has only A
    # Deterministic: the same arguments give byte-identical output.
    again = tmp_path / "again.md"
    main(
        ["eval", "report", "--golden", write_golden(tmp_path, OK1, OK2, UNKNOWN, BAD),
         "--a-model", "qwen3.5:4b", "--a-prompt", "v1",
         "--b-model", "claude-haiku-4-5", "--b-prompt", "v1", "--out", str(again)],
        deps=deps_for(seeded, FakeSummarizer()),
    )
    assert again.read_bytes() == out.read_bytes()


def test_report_goes_to_stdout_without_out(
    seeded: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["eval", "report", "--golden", write_golden(tmp_path, OK1),
         "--a-model", "m1", "--a-prompt", "v1", "--b-model", "m2", "--b-prompt", "v1"],
        deps=deps_for(seeded, FakeSummarizer()),
    )

    assert code == 0
    assert capsys.readouterr().out.startswith("# Summarizer comparison")
