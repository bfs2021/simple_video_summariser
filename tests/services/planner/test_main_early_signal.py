"""A SIGTERM/SIGINT during ``services.planner.main`` startup exits 0 (issue #108).

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

MODULE = "services.planner.main"
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


def test_once_with_an_early_sigterm_exits_0_and_runs_no_task(tmp_path: Path) -> None:
    code, out = run_window(MODULE, "psycopg", tmp_path, args=("--once",))

    assert code == 0, out
    assert "planner.task" not in out


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
        "import services.planner.main\n"
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
    for init in ("services/__init__.py", "services/planner/__init__.py"):
        tree = ast.parse((REPO_ROOT / init).read_text(encoding="utf-8"))
        assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]


class _FakeScheduler:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def request_shutdown(self) -> None:
        self.calls.append("request_shutdown")

    def run(self) -> None:
        self.calls.append("run")

    def run_once(self) -> bool:
        self.calls.append("run_once")
        return True


def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    from common.config import Settings
    from services.planner import main as planner_main

    monkeypatch.setattr(
        planner_main, "get_settings", lambda: Settings(DATABASE_URL="postgresql://u:p@h/db")  # type: ignore[arg-type]
    )
    monkeypatch.setattr(planner_main, "configure_logging", lambda _s: None)


@pytest.mark.parametrize("once", [False, True])
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_signal_while_the_scheduler_is_built_stops_it_before_it_runs(
    monkeypatch: pytest.MonkeyPatch, once: bool, sig: signal.Signals
) -> None:
    from services.planner import main as planner_main

    _quiet(monkeypatch)
    sched = _FakeScheduler()

    def factory(_settings: object) -> _FakeScheduler:
        os.kill(os.getpid(), sig)  # lands after main() took the handler over
        return sched

    before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))
    rc = planner_main.main(["--once"] if once else [], scheduler_factory=factory)  # type: ignore[arg-type]

    assert rc == 0
    assert sched.calls == ["request_shutdown"]
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == before


def test_an_early_flag_makes_main_return_0_before_any_startup_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.planner import main as planner_main

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("startup step ran")

    monkeypatch.setattr(planner_main, "_early_stop", True)
    monkeypatch.setattr(planner_main, "get_settings", boom)
    before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))

    assert planner_main.main([], scheduler_factory=boom) == 0  # type: ignore[arg-type]
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == before


def test_invalid_settings_still_exit_1_without_a_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import ValidationError

    from common.config import Settings
    from services.planner import main as planner_main

    def bad() -> Settings:
        return Settings(DATABASE_URL="postgresql://u:p@h/db", POLL_INTERVAL_SEC=-1)  # type: ignore[arg-type]

    monkeypatch.setattr(planner_main, "get_settings", bad)
    with pytest.raises(ValidationError):
        bad()

    assert planner_main.main([]) == 1


def test_an_early_flag_beats_a_bad_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    from services.planner import main as planner_main

    monkeypatch.setattr(planner_main, "_early_stop", True)

    assert planner_main.main(["--no-such-option"]) == 0


def test_a_signal_during_argument_parsing_skips_the_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import argparse

    from services.planner import main as planner_main

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("startup step ran")

    real_parse: Any = argparse.ArgumentParser.parse_args

    def parse_and_signal(self: argparse.ArgumentParser, *args: Any, **kwargs: Any) -> Any:
        os.kill(os.getpid(), signal.SIGTERM)
        return real_parse(self, *args, **kwargs)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_and_signal)
    monkeypatch.setattr(planner_main, "get_settings", boom)

    assert planner_main.main([]) == 0


def test_a_signal_during_logging_setup_skips_building_the_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.planner import main as planner_main

    _quiet(monkeypatch)
    monkeypatch.setattr(
        planner_main, "configure_logging", lambda _s: os.kill(os.getpid(), signal.SIGTERM)
    )

    def boom(_settings: object) -> None:
        raise AssertionError("scheduler was built")

    assert planner_main.main([], scheduler_factory=boom) == 0  # type: ignore[arg-type]
