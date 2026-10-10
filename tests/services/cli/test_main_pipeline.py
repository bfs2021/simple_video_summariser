"""``ytdigest run`` end to end against Postgres, with fake ports (issue #31)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import psycopg
import pytest

from common.models import ChunkAnalysis
from common.queue import PostgresQueue
from services.cli.inprocess import InProcessQueue
from services.cli.main import main
from services.cli.progress import ProgressPrinter
from tests.services.analyzer.fakes import FakeSummarizer
from tests.services.cli.fakes import (
    MANUAL_ID,
    NO_CAPTIONS_ID,
    FakeTranscriber,
    manual_rig,
    no_captions_rig,
    with_video_id,
)
from tests.services.transcriber.fakes import FakeMetadata, fixture_meta

pytestmark = pytest.mark.integration

_STARTED = re.compile(r"^\[(\w+)\] started$")
_FINISHED = re.compile(r"^\[(\w+)\] finished")


def stages(err: str, pattern: re.Pattern[str]) -> list[str]:
    return [m.group(1) for line in err.splitlines() if (m := pattern.match(line))]


def count(dsn: str, table: str, where: str = "true") -> int:
    with psycopg.connect(dsn) as conn:
        row = conn.execute(f"SELECT count(*) FROM {table} WHERE {where}").fetchone()
    assert row is not None
    return int(row[0])


def job_rows(dsn: str) -> list[tuple[Any, ...]]:
    with psycopg.connect(dsn) as conn:
        return conn.execute(
            "SELECT id, video_id, kind, state, attempts, locked_by FROM jobs ORDER BY id"
        ).fetchall()


def test_manual_subtitles_run_ingest_then_analyze_and_write_every_table(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = manual_rig(head_dsn, tmp_path)

    code = main(["run", MANUAL_ID], deps=rig.deps)

    _out, err = capsys.readouterr()
    assert code == 0
    assert stages(err, _STARTED) == ["ingest", "analyze"]
    assert stages(err, _FINISHED) == ["ingest", "analyze"]
    assert [call[:1] for call in rig.subtitles.fetch_calls] == [(MANUAL_ID,)]
    assert rig.audio.calls == [] and rig.transcriber.calls == 0
    assert "roster" in rig.summarizer.kinds and "reduce" in rig.summarizer.kinds
    assert count(head_dsn, "videos", f"video_id = '{MANUAL_ID}'") == 1
    assert count(head_dsn, "transcripts", "source = 'youtube_manual'") == 1
    assert count(head_dsn, "transcript_chunks") >= 1
    assert count(head_dsn, "analyses") == 1
    finish_lines = [line for line in err.splitlines() if _FINISHED.match(line)]
    assert any("youtube_manual" in line and re.search(r"\d+ chunk", line) for line in finish_lines)


def test_a_video_without_subtitles_runs_ingest_transcribe_analyze(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = no_captions_rig(head_dsn, tmp_path)

    code = main(["run", NO_CAPTIONS_ID], deps=rig.deps)

    _out, err = capsys.readouterr()
    assert code == 0
    assert stages(err, _STARTED) == ["ingest", "transcribe", "analyze"]
    assert stages(err, _FINISHED) == ["ingest", "transcribe", "analyze"]
    assert rig.audio.calls == [NO_CAPTIONS_ID]
    assert rig.transcriber.calls == 1
    assert count(head_dsn, "transcripts", "source = 'whisper'") == 1
    assert count(head_dsn, "analyses") == 1
    assert any(
        "whisper" in line and "chunk" in line for line in err.splitlines() if _FINISHED.match(line)
    )


def test_the_jobs_table_is_never_touched(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with psycopg.connect(head_dsn) as conn:
        conn.execute("INSERT INTO videos (video_id) VALUES ('workerjob01')")
        PostgresQueue(conn).enqueue("ingest", "workerjob01", priority=5)
        PostgresQueue(conn).enqueue("analyze", "workerjob01", dedupe_key="v1:ollama")
        conn.commit()
    before = job_rows(head_dsn)
    assert len(before) == 2

    for _ in range(2):
        assert main(["run", MANUAL_ID], deps=manual_rig(head_dsn, tmp_path).deps) == 0
    capsys.readouterr()

    assert job_rows(head_dsn) == before


def test_jobs_are_enqueued_at_interactive_priority_and_inherit_it(
    head_dsn: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, int]] = []
    real = InProcessQueue.enqueue

    def spy(self: InProcessQueue, kind: str, video_id: str, **kwargs: Any) -> int | None:
        calls.append((kind, video_id, kwargs.get("priority", 0)))
        return real(self, kind, video_id, **kwargs)

    monkeypatch.setattr(InProcessQueue, "enqueue", spy)

    assert main(["run", NO_CAPTIONS_ID], deps=no_captions_rig(head_dsn, tmp_path).deps) == 0

    capsys.readouterr()
    assert [c[0] for c in calls] == ["ingest", "transcribe", "analyze"]
    assert {c[1] for c in calls} == {NO_CAPTIONS_ID}
    assert [c[2] for c in calls] == [10, 10, 10]


def test_stdout_carries_only_the_result_and_every_log_line_goes_to_stderr(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = manual_rig(head_dsn, tmp_path)

    assert main(["run", MANUAL_ID], deps=rig.deps) == 0

    out, err = capsys.readouterr()
    assert "Do schools kill creativity?" in out
    assert MANUAL_ID in out
    assert "youtube_manual" in out
    assert "fake-model" in out and "v1" in out
    assert "the tldr" in out
    assert re.search(r"^  \[\d+:\d\d\] Opening remarks$", out, re.MULTILINE)
    assert "Claims: 1" in out and "Quotes: 0" in out
    assert "[ingest]" not in out and '"event"' not in out and "ingest finished" not in out
    assert "ingest finished" in err  # the handler's own log line, on stderr


def test_the_result_never_contains_terminal_control_bytes_from_the_llm(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hostile = FakeSummarizer(
        name="ollama",
        tldr="fine\x1b[2J\x1b]0;pwned\x07 done",
        default_chunk=ChunkAnalysis(topics=(), claims=(), quotes=()),
    )
    rig = manual_rig(head_dsn, tmp_path, summarizer=hostile)

    assert main(["run", MANUAL_ID], deps=rig.deps) == 0

    out, _err = capsys.readouterr()
    assert "fine done" in out
    assert "\x1b" not in out and "\x07" not in out and "pwned" not in out


def test_running_twice_succeeds_both_times_and_leaves_one_videos_row(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", MANUAL_ID], deps=manual_rig(head_dsn, tmp_path).deps) == 0
    assert main(["run", MANUAL_ID], deps=manual_rig(head_dsn, tmp_path).deps) == 0

    capsys.readouterr()
    assert count(head_dsn, "videos") == 1
    assert count(head_dsn, "transcripts") == 1


def test_a_url_is_accepted_and_only_the_bare_id_reaches_the_handlers(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = manual_rig(head_dsn, tmp_path)

    code = main(["run", f"https://www.youtube.com/watch?v={MANUAL_ID}&t=42s"], deps=rig.deps)

    capsys.readouterr()
    assert code == 0
    assert rig.metadata.calls == [MANUAL_ID]


def test_an_id_that_starts_with_a_dash_runs_after_double_dash(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dash_id = "-wNyEUrxzFU"
    meta = with_video_id(fixture_meta("manual_en", MANUAL_ID), dash_id)
    rig = manual_rig(head_dsn, tmp_path, meta=meta)

    code = main(["run", "--", dash_id], deps=rig.deps)

    capsys.readouterr()
    assert code == 0
    assert rig.metadata.calls == [dash_id]
    assert isinstance(rig.metadata, FakeMetadata)
    assert count(head_dsn, "analyses", f"video_id = '{dash_id}'") == 1


# ------------------------------------------------------- progress (issue #96)


class ProgressingTranscriber(FakeTranscriber):
    """Reports the audio position the way the Whisper adapter does."""

    def transcribe(self, audio: Any, *, language: Any = None, on_progress: Any = None) -> Any:
        for done in (10.0, 60.0, 120.0):
            on_progress(done)
        return super().transcribe(audio, language=language, on_progress=on_progress)


def plain_lines(err: str) -> list[str]:
    """The progress lines on stderr, i.e. the non-JSON-log lines starting with a stage tag."""
    return [
        line
        for line in err.splitlines()
        if line.startswith(("[transcribe] ", "[analyze] ")) and "started" not in line
        and "finished" not in line
    ]


def test_a_full_run_prints_transcribe_and_analyze_progress_on_stderr_only(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = no_captions_rig(head_dsn, tmp_path, transcriber=ProgressingTranscriber())

    assert main(["run", NO_CAPTIONS_ID], deps=rig.deps) == 0

    out, err = capsys.readouterr()
    lines = plain_lines(err)
    assert [re.sub(r" - elapsed .*", "", line) for line in lines] == [
        "[transcribe] 0:10 / 2:00 audio (8%)",
        "[transcribe] 1:00 / 2:00 audio (50%)",
        "[transcribe] 2:00 / 2:00 audio (100%)",
        "[analyze] roster",
        "[analyze] chunk 1/1",
        "[analyze] reduce",
    ]
    assert "[transcribe]" not in out and "[analyze]" not in out and "audio (" not in out
    assert "progress" not in err  # not also a JSON log record
    assert "\r" not in err and "\x1b" not in err


def test_a_video_with_subtitles_prints_no_transcription_progress(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = manual_rig(head_dsn, tmp_path)

    assert main(["run", MANUAL_ID], deps=rig.deps) == 0

    _, err = capsys.readouterr()
    assert not [line for line in plain_lines(err) if line.startswith("[transcribe]")]
    assert "[analyze] roster" in err


def test_a_closed_stderr_pipe_does_not_change_the_exit_code(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = manual_rig(head_dsn, tmp_path)
    real = ProgressPrinter.__call__

    class Closed:
        def write(self, text: str) -> int:
            raise BrokenPipeError

        def flush(self) -> None:
            raise BrokenPipeError

    def broken(self: ProgressPrinter, *args: Any, **kwargs: Any) -> None:
        self._stream = lambda: Closed()  # type: ignore[assignment, return-value]
        real(self, *args, **kwargs)

    monkeypatch.setattr(ProgressPrinter, "__call__", broken)

    assert main(["run", MANUAL_ID], deps=rig.deps) == 0

    out, _ = capsys.readouterr()
    assert "Do schools kill creativity?" in out
