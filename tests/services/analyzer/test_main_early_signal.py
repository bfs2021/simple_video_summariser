"""A SIGTERM/SIGINT during ``services.analyzer.main`` startup exits 0 (issue #108).

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
import types
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

MODULE = "services.analyzer.main"
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
        "import services.analyzer.main\n"
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
    for init in ("services/__init__.py", "services/analyzer/__init__.py"):
        tree = ast.parse((REPO_ROOT / init).read_text(encoding="utf-8"))
        assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]


def _boom(*_a: object, **_k: object) -> None:
    raise AssertionError("startup step ran")


def test_an_early_flag_makes_main_return_0_before_any_startup_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.analyzer import main as analyzer_main

    monkeypatch.setattr(analyzer_main, "_early_stop", True)
    monkeypatch.setattr(analyzer_main, "get_settings", _boom)
    before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))

    assert analyzer_main.main(summarizer_factory=_boom) == 0
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == before


def test_a_signal_during_the_ollama_probe_claims_no_job_and_opens_no_connection() -> None:
    from common.config import Settings
    from services.analyzer import main as analyzer_main

    def probing_transport(_url: str, _timeout: float) -> Any:
        os.kill(os.getpid(), signal.SIGTERM)
        raise ConnectionRefusedError("no ollama")

    settings = Settings(DATABASE_URL="postgresql://u:p@h/db", SUMMARIZER="ollama")  # type: ignore[arg-type]
    rc = analyzer_main.main(
        settings=settings,
        summarizer_factory=lambda _s: types.SimpleNamespace(name="fake"),
        connect=_boom,  # type: ignore[arg-type]
        handler_factory=_boom,  # type: ignore[arg-type]
        probe_transport=probing_transport,
    )

    assert rc == 0


def test_a_signal_between_building_the_worker_and_running_it_is_not_lost() -> None:
    from common.config import Settings
    from services.analyzer import main as analyzer_main

    def handler_factory(*_a: object, **_k: object) -> object:
        os.kill(os.getpid(), signal.SIGTERM)  # after main() took the handler over
        return lambda _job, _ctx: None

    settings = Settings(DATABASE_URL="postgresql://u:p@h/db")  # type: ignore[arg-type]
    rc = analyzer_main.main(
        settings=settings,
        summarizer_factory=lambda _s: types.SimpleNamespace(name="fake"),
        connect=_boom,  # type: ignore[arg-type]
        handler_factory=handler_factory,  # type: ignore[arg-type]
    )

    assert rc == 0


def test_a_signal_during_logging_setup_skips_the_summarizer_and_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from common.config import Settings
    from services.analyzer import main as analyzer_main

    monkeypatch.setattr(
        analyzer_main, "configure_logging", lambda _s: os.kill(os.getpid(), signal.SIGTERM)
    )
    settings = Settings(DATABASE_URL="postgresql://u:p@h/db", SUMMARIZER="ollama")  # type: ignore[arg-type]

    rc = analyzer_main.main(settings=settings, summarizer_factory=_boom, probe_transport=_boom)  # type: ignore[arg-type]

    assert rc == 0


def test_a_signal_while_the_summarizer_is_built_skips_the_probe_and_the_started_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from structlog.testing import capture_logs

    from common.config import Settings
    from services.analyzer import main as analyzer_main

    def build(_s: object) -> Any:
        os.kill(os.getpid(), signal.SIGTERM)
        return types.SimpleNamespace(name="fake")

    probes: list[str] = []

    def transport(url: str, _timeout: float) -> Any:
        probes.append(url)
        raise ConnectionRefusedError("no ollama")

    settings = Settings(DATABASE_URL="postgresql://u:p@h/db", SUMMARIZER="ollama")  # type: ignore[arg-type]
    # configure_logging would replace the processors capture_logs installed.
    monkeypatch.setattr(analyzer_main, "configure_logging", lambda _s: None)

    with capture_logs() as logs:
        rc = analyzer_main.main(settings=settings, summarizer_factory=build, probe_transport=transport)

    assert rc == 0
    assert probes == []
    assert [e for e in logs if e["event"] == "analyzer.started"] == []
