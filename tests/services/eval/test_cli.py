"""``ytdigest eval run|report``: usage, configuration and database-setup paths (issue #62).

No database and no LLM: every ``connect`` here either fails the test when called
or raises the way an unreachable Postgres does.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from pydantic import ValidationError

from common.config import Settings, get_settings
from services.cli.main import Deps, main
from services.eval.runner import apply_overrides
from tests.services.analyzer.fakes import FakeSummarizer
from tests.services.transcriber.fakes import make_settings

VID = "dQw4w9WgXcQ"
SECRET_DSN = "postgresql://user:hunter2-secret@db.invalid/ytdigest"


def golden_file(tmp_path: Path, text: str = VID + "\n") -> str:
    path = tmp_path / "golden.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def never_connect() -> Any:
    raise AssertionError("the database must not be touched")


def unreachable() -> Any:
    raise psycopg.OperationalError(f"connection to {SECRET_DSN} failed")


def deps_with(connect: Any = never_connect, **settings: Any) -> Deps:
    summarizer = FakeSummarizer(name="ollama", model="fake-model")
    return Deps(
        settings=make_settings(**settings),
        connect=connect,
        metadata=None,  # type: ignore[arg-type]
        subtitles=None,  # type: ignore[arg-type]
        audio=None,  # type: ignore[arg-type]
        transcriber=None,  # type: ignore[arg-type]
        summarizer=summarizer,
    )


@pytest.fixture(autouse=True)
def _clean_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in ("SUMMARIZER", "OLLAMA_MODEL", "ANTHROPIC_MODEL", "ANTHROPIC_API_KEY", "PROMPT_VERSION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", SECRET_DSN)
    return monkeypatch


# ------------------------------------------------------------------ overrides


def test_overrides_apply_to_the_chosen_summarizers_model() -> None:
    base = make_settings(OLLAMA_MODEL="a", ANTHROPIC_MODEL="b")
    ollama = apply_overrides(base, summarizer=None, model="x", prompt_version=None)
    assert (ollama.OLLAMA_MODEL, ollama.ANTHROPIC_MODEL) == ("x", "b")
    cloud = apply_overrides(base, summarizer="anthropic", model="y", prompt_version="v9")
    assert cloud.SUMMARIZER == "anthropic"
    assert (cloud.OLLAMA_MODEL, cloud.ANTHROPIC_MODEL, cloud.PROMPT_VERSION) == ("a", "y", "v9")


def test_no_overrides_keeps_every_setting() -> None:
    base = make_settings(PROMPT_VERSION="v3", CHUNK_SEC=600)
    same = apply_overrides(base, summarizer=None, model=None, prompt_version=None)
    assert same == base


@pytest.mark.parametrize("bad", ["../x", "V1", "a b", "", "v1/../v2", "x\n"])
def test_a_bad_prompt_version_fails_the_settings_rule(bad: str) -> None:
    with pytest.raises(ValidationError):
        apply_overrides(make_settings(), summarizer=None, model=None, prompt_version=bad)


def test_an_unknown_summarizer_fails_validation() -> None:
    with pytest.raises(ValidationError):
        apply_overrides(make_settings(), summarizer="gpt", model=None, prompt_version=None)  # type: ignore[arg-type]


# ----------------------------------------------------------- golden-file errors


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ("", "golden set is empty"),
        ("# nothing\n", "golden set is empty"),
        ("not-an-id\n", "line 1"),
        (VID + "\nbad!\n", "line 2"),
    ],
)
@pytest.mark.parametrize("command", ["run", "report"])
def test_a_bad_golden_file_exits_2_before_any_database_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], text: str, needle: str, command: str
) -> None:
    extra = (
        ["--a-model", "m1", "--a-prompt", "v1", "--b-model", "m2", "--b-prompt", "v1"]
        if command == "report"
        else []
    )

    code = main(["eval", command, "--golden", golden_file(tmp_path, text), *extra], deps=deps_with())

    err = capsys.readouterr().err
    assert code == 2
    assert needle in err


def test_a_missing_golden_file_exits_2_and_names_the_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["eval", "run", "--golden", str(tmp_path / "nope.txt")], deps=deps_with())

    assert code == 2
    assert "nope.txt" in capsys.readouterr().err


def test_golden_from_stdin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(VID + "\n"))

    code = main(["eval", "run", "--golden", "-"], deps=deps_with(connect=unreachable))

    err = capsys.readouterr().err
    assert code == 1  # got past the golden file and failed at the database
    assert "cannot connect to the database" in err


def test_an_empty_stdin_golden_is_empty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    assert main(["eval", "run", "--golden", "-"], deps=deps_with()) == 2
    assert "golden set is empty" in capsys.readouterr().err


def test_a_duplicate_is_warned_about_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = golden_file(tmp_path, f"{VID}\nhttps://youtu.be/{VID}\n")

    main(["eval", "run", "--golden", path], deps=deps_with(connect=unreachable))

    assert "duplicate" in capsys.readouterr().err


# ------------------------------------------------------------- configuration


@pytest.mark.parametrize("bad", ["../x", "UPPER", "a b"])
def test_a_bad_prompt_version_exits_2_without_echoing_it_or_touching_the_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], bad: str
) -> None:
    code = main(
        ["eval", "run", "--golden", golden_file(tmp_path), "--prompt-version", bad], deps=deps_with()
    )

    err = capsys.readouterr().err
    assert code == 2
    assert "--prompt-version" in err
    assert bad not in err


def test_a_prompt_version_without_a_directory_exits_2_naming_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["eval", "run", "--golden", golden_file(tmp_path), "--prompt-version", "v99"], deps=deps_with()
    )

    err = capsys.readouterr().err
    assert code == 2
    assert "v99" in err


def test_anthropic_without_a_key_exits_1_like_ytdigest_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], env: pytest.MonkeyPatch
) -> None:
    code = main(["eval", "run", "--golden", golden_file(tmp_path), "--summarizer", "anthropic"])

    err = capsys.readouterr().err
    assert code == 1
    assert "invalid configuration" in err
    assert "ANTHROPIC_API_KEY" in err
    assert "hunter2" not in err


def test_the_api_key_never_appears_in_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], env: pytest.MonkeyPatch
) -> None:
    env.setenv("ANTHROPIC_API_KEY", "sk-ant-very-secret-key")
    env.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:1/none")

    main(["eval", "run", "--golden", golden_file(tmp_path), "--summarizer", "anthropic", "--model", "m"])

    captured = capsys.readouterr()
    assert "very-secret" not in captured.err + captured.out


def test_the_effective_configuration_is_printed_before_any_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], env: pytest.MonkeyPatch
) -> None:
    env.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:1/none")
    env.setenv("OLLAMA_MODEL", "from-env")

    code = main(["eval", "run", "--golden", golden_file(tmp_path), "--model", "from-flag"])

    err = capsys.readouterr().err
    assert code == 1  # the database is unreachable
    line = next(x for x in err.splitlines() if x.startswith("eval:"))
    assert "summarizer=ollama" in line and "model=from-flag" in line
    assert "prompt_version=v1" in line and "chunk_strategy=time:900:60" in line
    assert err.index(line) < err.index("cannot connect")


def test_a_missing_environment_is_an_invalid_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], env: pytest.MonkeyPatch
) -> None:
    env.delenv("DATABASE_URL")

    code = main(["eval", "run", "--golden", golden_file(tmp_path)])

    assert code == 1
    assert "invalid configuration" in capsys.readouterr().err


def test_an_unreachable_database_exits_1_without_the_dsn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["eval", "run", "--golden", golden_file(tmp_path)], deps=deps_with(connect=unreachable))

    err = capsys.readouterr().err
    assert code == 1
    assert "cannot connect to the database" in err
    assert "hunter2" not in err and "db.invalid" not in err


# ----------------------------------------------------------------- report args


def test_report_with_equal_runs_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["eval", "report", "--golden", golden_file(tmp_path), "--a-model", "m:1", "--a-prompt", "v1",
         "--b-model", "m:1", "--b-prompt", "v1"],
        deps=deps_with(),
    )

    assert code == 2
    assert "same" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["--a-model", "--a-prompt", "--b-model", "--b-prompt", "--golden"])
def test_report_requires_every_run_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    flags = {
        "--golden": golden_file(tmp_path),
        "--a-model": "m1", "--a-prompt": "v1", "--b-model": "m2", "--b-prompt": "v1",
    }
    argv = ["eval", "report"]
    for key, value in flags.items():
        if key != missing:
            argv += [key, value]

    assert main(argv, deps=deps_with()) == 2
    assert missing in capsys.readouterr().err


def test_report_model_names_with_colons_are_taken_as_one_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["eval", "report", "--golden", golden_file(tmp_path), "--a-model", "qwen3.5:4b",
         "--a-prompt", "v1", "--b-model", "qwen3.5:4b", "--b-prompt", "v2"],
        deps=deps_with(connect=unreachable),
    )

    assert code == 1
    assert "cannot connect to the database" in capsys.readouterr().err


def test_settings_type_is_unchanged_by_overrides() -> None:
    assert isinstance(apply_overrides(make_settings(), summarizer=None, model=None, prompt_version=None), Settings)
