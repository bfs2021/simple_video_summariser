"""A SIGTERM/SIGINT during ``services.transcriber.main`` startup exits 0 (issue #108).

Every case runs the real entrypoint as a subprocess; no Postgres is needed
(``DATABASE_URL`` is a closed port). Before the fix each deterministic case
died with -15, because the handler was installed only inside ``run()``.
"""

from __future__ import annotations

import ast
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.services.early_signal import (
    CLOSED_PORT_DSN,
    REPO_ROOT,
    clean_env,
    proc_status_available,
    race_once,
    run_window,
)

MODULE = "services.transcriber.main"
TRIGGERS = ["argparse", "pydantic", "psycopg", "structlog", "common.config"]


@pytest.mark.parametrize("trigger", TRIGGERS)
def test_sigterm_during_startup_imports_exits_0(trigger: str, tmp_path: Path) -> None:
    code, out = run_window(MODULE, trigger, tmp_path)

    assert code == 0, out


def test_sigint_during_startup_imports_exits_0(tmp_path: Path) -> None:
    code, out = run_window(MODULE, "psycopg", tmp_path, sig="SIGINT")

    assert code == 0, out


def test_the_handler_is_installed_before_the_heavy_imports_begin(tmp_path: Path) -> None:
    code, out = run_window(MODULE, "psycopg", tmp_path)

    assert "SIGTERM-HANDLER-AT-IMPORT:SET" in out, out
    assert code == 0, out


@pytest.mark.skipif(not proc_status_available(), reason="needs /proc")
def test_sigterm_the_moment_the_handler_exists_exits_0(tmp_path: Path) -> None:
    # The issue asks for ten rounds: the signal lands at a different point of
    # startup each time. Each round waits on the exact condition (SigCgt).
    for round_no in range(10):
        code, out = race_once(MODULE, tmp_path)

        assert code == 0, f"round {round_no}: exit {code}\n{out}"


def test_importing_the_entrypoint_does_not_change_sigterm_or_sigint() -> None:
    code = (
        "import signal\n"
        "before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))\n"
        "import services.transcriber.main\n"
        "after = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))\n"
        "assert before == after, (before, after)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=clean_env(DATABASE_URL=CLOSED_PORT_DSN),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_the_package_inits_import_nothing() -> None:
    for init in ("services/__init__.py", "services/transcriber/__init__.py"):
        tree = ast.parse((REPO_ROOT / init).read_text(encoding="utf-8"))
        assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]


def test_an_early_flag_makes_main_return_0_before_any_startup_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.transcriber import main as transcriber_main

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("startup step ran")

    monkeypatch.setattr(transcriber_main, "_early_stop", True)
    monkeypatch.setattr(transcriber_main, "get_settings", boom)
    before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))

    assert transcriber_main.main([], handlers={}, queue_factory=boom) == 0  # type: ignore[arg-type]
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == before


def test_a_signal_during_the_audio_dir_check_opens_no_queue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from services.transcriber import main as transcriber_main
    from tests.services.transcriber.fakes import make_settings

    def send_signal(_path: Path) -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    def no_queue() -> object:
        raise AssertionError("a queue was opened")

    monkeypatch.setattr(transcriber_main, "get_settings", lambda: make_settings(AUDIO_DIR=tmp_path))
    monkeypatch.setattr(transcriber_main, "configure_logging", lambda _s: None)
    monkeypatch.setattr(transcriber_main, "prepare_audio_dir", send_signal)

    rc = transcriber_main.main([], handlers={}, queue_factory=no_queue)  # type: ignore[arg-type]

    assert rc == 0


def test_a_signal_during_logging_setup_skips_the_audio_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from services.transcriber import main as transcriber_main
    from tests.services.transcriber.fakes import make_settings

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("startup step ran")

    monkeypatch.setattr(transcriber_main, "get_settings", lambda: make_settings(AUDIO_DIR=tmp_path))
    monkeypatch.setattr(
        transcriber_main, "configure_logging", lambda _s: os.kill(os.getpid(), signal.SIGTERM)
    )
    monkeypatch.setattr(transcriber_main, "prepare_audio_dir", boom)

    assert transcriber_main.main([], handlers={}, queue_factory=boom) == 0  # type: ignore[arg-type]


def test_an_early_flag_beats_a_bad_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    from services.transcriber import main as transcriber_main

    monkeypatch.setattr(transcriber_main, "_early_stop", True)

    assert transcriber_main.main(["--no-such-option"]) == 0


def _boom(*_a: object, **_k: object) -> None:
    raise AssertionError("startup step ran")


def test_a_signal_during_argument_parsing_skips_the_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import argparse

    from services.transcriber import main as transcriber_main

    real_parse: Any = argparse.ArgumentParser.parse_args

    def parse_and_signal(self: argparse.ArgumentParser, *args: Any, **kwargs: Any) -> Any:
        os.kill(os.getpid(), signal.SIGTERM)
        return real_parse(self, *args, **kwargs)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_and_signal)
    monkeypatch.setattr(transcriber_main, "get_settings", _boom)

    assert transcriber_main.main([], handlers={}) == 0


def test_a_signal_during_the_audio_dir_check_does_not_build_the_handlers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from services.transcriber import main as transcriber_main
    from tests.services.transcriber.fakes import make_settings

    monkeypatch.setattr(transcriber_main, "get_settings", lambda: make_settings(AUDIO_DIR=tmp_path))
    monkeypatch.setattr(transcriber_main, "configure_logging", lambda _s: None)
    monkeypatch.setattr(
        transcriber_main, "prepare_audio_dir", lambda _p: os.kill(os.getpid(), signal.SIGTERM)
    )
    monkeypatch.setattr(transcriber_main, "build_handlers", _boom)

    assert transcriber_main.main([], queue_factory=_boom) == 0  # type: ignore[arg-type]


def test_a_signal_while_handlers_are_built_logs_no_started_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from structlog.testing import capture_logs

    from services.transcriber import main as transcriber_main
    from tests.services.transcriber.fakes import make_settings

    def build(*_a: object, **_k: object) -> dict[str, object]:
        os.kill(os.getpid(), signal.SIGTERM)
        return {}

    monkeypatch.setattr(transcriber_main, "get_settings", lambda: make_settings(AUDIO_DIR=tmp_path))
    monkeypatch.setattr(transcriber_main, "configure_logging", lambda _s: None)
    monkeypatch.setattr(transcriber_main, "build_handlers", build)

    with capture_logs() as logs:
        rc = transcriber_main.main([], queue_factory=_boom)  # type: ignore[arg-type]

    assert rc == 0
    assert [e for e in logs if e["event"] == "transcriber.started"] == []
