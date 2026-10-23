"""The shared yt-dlp runner (issue #16).

No test here starts yt-dlp: the runner is injected. The timeout and
not-installed tests start a plain Python child instead, through the real
``run_process``, so process-group cleanup is exercised for real.
"""

import pathlib
import subprocess
import sys
import textwrap
from collections.abc import Sequence

import pytest

from adapters.youtube.ytdlp import run_process, run_ytdlp
from common.errors import (
    PermanentSourceError,
    RateLimitedError,
    ToolFailureError,
    TransientNetworkError,
)
from tests.adapters.youtube.fakes import FakeRunner, completed

# Real stderr of `/usr/bin/python3.14 -m yt_dlp --version` on the dev machine,
# 2026-09-27, where the system interpreter has no yt-dlp installed.
NOT_INSTALLED_STDERR = "/usr/bin/python3.14: No module named yt_dlp\n"


# --- argv and output -------------------------------------------------------


def test_runs_the_venv_copy_of_yt_dlp_with_the_given_args_and_timeout() -> None:
    runner = FakeRunner(completed(stdout='{"id": "x"}\n'))

    out = run_ytdlp(
        ["--dump-json", "https://www.youtube.com/watch?v=jNQXAC9IVRw"],
        timeout=30,
        runner=runner,
    )

    assert out == '{"id": "x"}\n'
    assert runner.calls == [
        (
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--dump-json",
                "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            ],
            30,
        )
    ]


def test_stderr_of_a_successful_run_is_ignored() -> None:
    runner = FakeRunner(completed(stdout="ok", stderr="WARNING: something\n"))
    assert run_ytdlp([], timeout=5, runner=runner) == "ok"


@pytest.mark.parametrize("timeout", [0, -1])
def test_a_non_positive_timeout_is_rejected(timeout: float) -> None:
    runner = FakeRunner()
    with pytest.raises(ValueError):
        run_ytdlp([], timeout=timeout, runner=runner)
    assert runner.calls == []


# --- non-zero exit goes through from_ytdlp (#15) -------------------------


def test_a_non_zero_exit_raises_the_classified_error() -> None:
    # live run, yt-dlp 2026.08.19 (tests/fixtures/ytdlp_errors/cases.toml)
    stderr = "ERROR: [youtube] yZIXLfi8CZQ: Private video\n"
    runner = FakeRunner(completed(stderr=stderr, returncode=1))

    with pytest.raises(PermanentSourceError) as info:
        run_ytdlp([], timeout=5, runner=runner)
    assert info.value.reason == "private"


def test_a_429_raises_rate_limited() -> None:
    stderr = (
        "ERROR: [generic] 429: Unable to download webpage: HTTP Error 429: Too Many "
        "Requests (caused by <HTTPError 429: Too Many Requests>)\n"
    )
    runner = FakeRunner(completed(stderr=stderr, returncode=1))
    with pytest.raises(RateLimitedError):
        run_ytdlp([], timeout=5, runner=runner)


# --- yt-dlp missing --------------------------------------------------------


def test_yt_dlp_not_installed_is_tool_failure() -> None:
    runner = FakeRunner(completed(stderr=NOT_INSTALLED_STDERR, returncode=1))
    with pytest.raises(ToolFailureError, match="not installed"):
        run_ytdlp(["--version"], timeout=5, runner=runner)


def test_yt_dlp_not_importable_by_a_real_interpreter_is_tool_failure() -> None:
    # `-S` drops site-packages, so this real child cannot import yt_dlp.
    def no_site_packages(
        argv: Sequence[str], *, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        return run_process([argv[0], "-S", *argv[1:]], timeout=timeout)

    with pytest.raises(ToolFailureError, match="not installed"):
        run_ytdlp(["--version"], timeout=30, runner=no_site_packages)


def test_a_missing_interpreter_is_tool_failure() -> None:
    runner = FakeRunner(FileNotFoundError(2, "No such file or directory"))
    with pytest.raises(ToolFailureError):
        run_ytdlp([], timeout=5, runner=runner)


# --- timeout ---------------------------------------------------------------


def test_a_timeout_from_the_runner_is_transient_network() -> None:
    runner = FakeRunner(subprocess.TimeoutExpired(["yt-dlp"], 5))
    with pytest.raises(TransientNetworkError, match="timed out"):
        run_ytdlp([], timeout=5, runner=runner)


# A child that starts a grandchild (as yt-dlp starts ffmpeg or a JS runtime),
# records both PIDs, then hangs. The grandchild inherits the stdout pipe.
_HANGING_CHILD = textwrap.dedent(
    """
    import os, pathlib, subprocess, sys, time
    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"]
    )
    pathlib.Path(sys.argv[1]).write_text(f"{os.getpid()} {grandchild.pid}")
    time.sleep(600)
    """
)


def _is_running(pid: int) -> bool:
    """True unless ``pid`` is gone or a zombie (exited, not yet reaped by init)."""
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        # The process can exit between the open and the read; ESRCH means the
        # same thing FileNotFoundError does here: there is no such process.
        return False
    state = stat.rsplit(")", 1)[1].split()[0]
    return state != "Z"


def test_a_process_that_vanishes_mid_read_is_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def vanished(self: pathlib.Path, *args: object, **kwargs: object) -> str:
        raise ProcessLookupError

    monkeypatch.setattr(pathlib.Path, "read_text", vanished)
    assert _is_running(1234) is False


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_a_timeout_raises_transient_network_and_leaves_no_process_running(
    tmp_path: pathlib.Path,
) -> None:
    pids_file = tmp_path / "pids"

    def hanging(
        argv: Sequence[str], *, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        return run_process(
            [sys.executable, "-c", _HANGING_CHILD, str(pids_file)], timeout=timeout
        )

    with pytest.raises(TransientNetworkError):
        run_ytdlp(["--dump-json"], timeout=1.5, runner=hanging)

    assert pids_file.exists(), "the child never started; the timeout is too short"
    child, grandchild = (int(p) for p in pids_file.read_text().split())
    assert not _is_running(child)
    assert not _is_running(grandchild)


def test_run_process_returns_exit_code_stdout_and_stderr() -> None:
    script = "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"
    result = run_process([sys.executable, "-c", script], timeout=30)
    assert (result.returncode, result.stdout, result.stderr) == (3, "out\n", "err\n")


def test_run_process_does_not_hand_the_child_our_stdin() -> None:
    script = "import sys; print(repr(sys.stdin.read()))"
    result = run_process([sys.executable, "-c", script], timeout=30)
    assert result.stdout == "''\n"
