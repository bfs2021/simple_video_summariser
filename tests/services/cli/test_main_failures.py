"""``ytdigest run`` failure paths: exit codes, one-line reports, no retries (issue #31)."""

from __future__ import annotations

import importlib.util
import io
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from adapters.transcription.faster_whisper import FasterWhisperTranscriber
from common.errors import (
    Cancelled,
    Defer,
    LLMInvalidOutputError,
    PermanentSourceError,
    ToolFailureError,
)
from services.cli.main import main
from tests.services.analyzer.fakes import FakeSummarizer
from tests.services.cli.fakes import (
    MANUAL_ID,
    NO_CAPTIONS_ID,
    REL,
    FakeAudio,
    FakeTranscriber,
    manual_rig,
    no_captions_rig,
)
from tests.services.transcriber.fakes import FakeSubtitles

pytestmark = pytest.mark.integration

_FAILED = re.compile(r"^\[(\w+)\] failed")


def failure_lines(err: str) -> list[str]:
    return [line for line in err.splitlines() if _FAILED.match(line)]


@pytest.mark.parametrize(
    ("failure", "error_class"),
    [
        (PermanentSourceError("removed", "video was removed by the uploader"), "PERMANENT_SOURCE"),
        (ToolFailureError("yt-dlp extractor broke"), "TOOL_FAILURE"),
        (RuntimeError("kaboom"), "BUG"),
    ],
)
def test_a_failing_ingest_stops_the_pipeline_with_one_line_and_exit_1(
    failure: Exception,
    error_class: str,
    head_dsn: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rig = manual_rig(head_dsn, tmp_path, meta=failure)

    code = main(["run", MANUAL_ID], deps=rig.deps)

    out, err = capsys.readouterr()
    assert code == 1
    assert out == ""
    (line,) = failure_lines(err)
    assert line.startswith("[ingest] failed")
    assert error_class in line
    assert str(failure) in line
    assert "Traceback" not in err
    assert rig.metadata.calls == [MANUAL_ID]  # not retried
    assert rig.subtitles.fetch_calls == []
    assert rig.summarizer.calls == []  # no later stage ran


def test_a_permanent_source_error_marks_the_video_unavailable(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = manual_rig(head_dsn, tmp_path, meta=PermanentSourceError("removed", "gone"))

    assert main(["run", MANUAL_ID], deps=rig.deps) == 1

    capsys.readouterr()
    with psycopg.connect(head_dsn) as conn:
        row = conn.execute(
            "SELECT unavailable FROM videos WHERE video_id = %s", (MANUAL_ID,)
        ).fetchone()
    assert row == ("removed",)


def test_an_invalid_llm_answer_fails_the_analyze_stage_once_without_a_retry(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summarizer = FakeSummarizer(
        name="ollama", fail_on={"reduce": LLMInvalidOutputError("model returned no JSON")}
    )
    rig = manual_rig(head_dsn, tmp_path, summarizer=summarizer)

    code = main(["run", MANUAL_ID], deps=rig.deps)

    _out, err = capsys.readouterr()
    assert code == 1
    (line,) = failure_lines(err)
    assert line.startswith("[analyze] failed")
    assert "LLM_INVALID_OUTPUT" in line and "model returned no JSON" in line
    assert summarizer.kinds.count("reduce") == 1
    with psycopg.connect(head_dsn) as conn:
        analyses = conn.execute("SELECT count(*) FROM analyses").fetchone()
    assert analyses == (0,)


def test_the_failure_line_is_a_single_line_free_of_control_bytes(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = manual_rig(
        head_dsn, tmp_path, meta=ToolFailureError("line one\nline two \x1b[2J\x1b]0;pwned\x07 end")
    )

    assert main(["run", MANUAL_ID], deps=rig.deps) == 1

    _out, err = capsys.readouterr()
    (line,) = failure_lines(err)
    assert "line one" in line and "line two" in line and "end" in line
    assert "\x1b" not in err and "\x07" not in err


def test_verbose_adds_the_traceback_and_the_default_omits_it(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = manual_rig(head_dsn, tmp_path, meta=RuntimeError("kaboom"))
    assert main(["run", MANUAL_ID], deps=rig.deps) == 1
    quiet = capsys.readouterr().err

    rig = manual_rig(head_dsn, tmp_path, meta=RuntimeError("kaboom"))
    assert main(["run", "--verbose", MANUAL_ID], deps=rig.deps) == 1
    loud = capsys.readouterr().err

    assert "Traceback" not in quiet
    assert "Traceback (most recent call last)" in loud
    assert "RuntimeError: kaboom" in loud
    assert len(failure_lines(loud)) == 1

    rig = manual_rig(head_dsn, tmp_path, meta=RuntimeError("kaboom"))
    assert main(["run", MANUAL_ID, "-v"], deps=rig.deps) == 1
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_a_deferred_video_exits_3_with_the_retry_time_and_does_not_wait(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    until = datetime(2026, 10, 1, 12, 30, tzinfo=UTC)
    rig = manual_rig(head_dsn, tmp_path, meta=Defer(until))

    code = main(["run", MANUAL_ID], deps=rig.deps)

    out, err = capsys.readouterr()
    assert code == 3
    assert out == ""
    assert "not available yet, retry after 2026-10-01T12:30:00+00:00" in err
    assert failure_lines(err) == []
    assert rig.metadata.calls == [MANUAL_ID]  # once: no retry


@pytest.mark.skipif(
    importlib.util.find_spec("faster_whisper") is not None,
    reason="faster-whisper is installed, so the missing-dependency path cannot be shown",
)
def test_missing_whisper_dependency_surfaces_the_requirements_file_and_exits_1(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = no_captions_rig(
        head_dsn,
        tmp_path,
        transcriber=FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=1),
    )

    code = main(["run", NO_CAPTIONS_ID], deps=rig.deps)

    _out, err = capsys.readouterr()
    assert code == 1
    (line,) = failure_lines(err)
    assert line.startswith("[transcribe] failed")
    assert "TOOL_FAILURE" in line
    assert "requirements.whisper.txt" in line
    assert rig.summarizer.calls == []


def test_missing_whisper_fails_fast_before_any_audio_work_whatever_audio_dir_is(
    head_dsn: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A None entry in sys.modules makes ``import faster_whisper`` raise
    # ImportError, so the missing-dependency path runs even where it is installed.
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    audio = FakeAudio(exc=PermissionError(13, "Permission denied", "/data"))
    rig = no_captions_rig(
        head_dsn,
        Path("/data/audio"),  # the default: not writable on a normal host
        audio=audio,
        transcriber=FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=1),
    )

    code = main(["run", NO_CAPTIONS_ID], deps=rig.deps)

    _out, err = capsys.readouterr()
    assert code == 1
    (line,) = failure_lines(err)
    assert line.startswith("[transcribe] failed")
    assert "TOOL_FAILURE" in line
    assert "requirements.whisper.txt" in line
    assert "PermissionError" not in err
    assert audio.calls == []
    assert rig.summarizer.calls == []


@pytest.mark.parametrize(
    ("failure", "exit_code", "message"),
    [
        (RuntimeError("boom"), 1, "[analyze] failed"),
        (Cancelled(), 130, "[analyze] cancelled"),
        (KeyboardInterrupt(), 130, "interrupted (Ctrl-C)"),
    ],
)
def test_tty_progress_is_closed_before_failure_output(
    failure: BaseException,
    exit_code: int,
    message: str,
    head_dsn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    err = Terminal()
    monkeypatch.setattr(sys, "stderr", err)
    summarizer = FakeSummarizer(name="ollama", fail_on={"chunk:0": failure})
    rig = manual_rig(head_dsn, tmp_path, summarizer=summarizer)

    assert main(["run", MANUAL_ID], deps=rig.deps) == exit_code

    text = err.getvalue()
    assert "[analyze] roster" in text
    assert f"\n{message}" in text


def test_ctrl_c_during_a_stage_exits_130_without_a_traceback(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summarizer = FakeSummarizer(name="ollama", fail_on={"roster": KeyboardInterrupt()})
    rig = manual_rig(head_dsn, tmp_path, summarizer=summarizer)

    code = main(["run", MANUAL_ID], deps=rig.deps)

    out, err = capsys.readouterr()
    assert code == 130
    assert out == ""
    assert "Traceback" not in err
    assert "interrupted" in err.lower()


def test_ctrl_c_during_transcription_still_removes_the_audio_when_audio_keep_is_off(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = no_captions_rig(
        head_dsn,
        tmp_path,
        transcriber=FakeTranscriber(KeyboardInterrupt()),
        AUDIO_KEEP=False,
    )

    code = main(["run", NO_CAPTIONS_ID], deps=rig.deps)

    _out, err = capsys.readouterr()
    assert code == 130
    assert "Traceback" not in err
    assert rig.audio.calls == [NO_CAPTIONS_ID]
    assert not (tmp_path / REL).exists()
    with psycopg.connect(head_dsn) as conn:
        assert conn.execute("SELECT count(*) FROM media").fetchone() == (0,)


def test_a_missing_schema_prints_one_line_pointing_at_the_migrate_command(
    postgres_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = manual_rig(postgres_dsn, tmp_path)  # a fresh database, migrations not applied

    code = main(["run", MANUAL_ID], deps=rig.deps)

    out, err = capsys.readouterr()
    assert code == 1
    assert out == ""
    assert len(err.strip().splitlines()) == 1
    assert "docker compose run --rm migrate" in err
    assert "Traceback" not in err
    assert rig.metadata.calls == []
    with psycopg.connect(postgres_dsn) as conn:
        # the CLI never runs migrations itself
        assert conn.execute("SELECT to_regclass('videos')").fetchone() == (None,)


def test_the_subtitle_fake_is_not_asked_for_anything_after_a_metadata_failure(
    head_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    subtitles = FakeSubtitles(manual=[])
    rig = manual_rig(head_dsn, tmp_path, meta=ToolFailureError("x"), subtitles=subtitles)

    assert main(["run", MANUAL_ID], deps=rig.deps) == 1

    capsys.readouterr()
    assert subtitles.fetch_calls == []
