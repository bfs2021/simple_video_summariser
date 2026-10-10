"""Cookie-file support for every yt-dlp call (issue #63).

A cookie file is a credential. These tests prove that it reaches yt-dlp only
through the shared ``run_ytdlp`` argument builder, and that neither its
contents nor its path can surface in an exception message (which becomes
``jobs.last_error``) or in a log record.

yt-dlp never runs. One fact the tests encode was measured on the pinned
yt-dlp 2026.8.19: it rewrites the cookie file when it exits, so a file on a
read-only bind mount makes it crash with ``PermissionError``. The helper
therefore hands yt-dlp a private, writable, short-lived copy.
"""

from __future__ import annotations

import logging
import os
import stat
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from adapters.youtube.audio import YouTubeAudio
from adapters.youtube.catalog import YouTubeCatalog
from adapters.youtube.metadata import YouTubeMetadata
from adapters.youtube.subtitles import YouTubeSubtitles
from adapters.youtube.ytdlp import CookieFileError, check_cookie_file, run_ytdlp
from common.errors import JobError, ToolFailureError, TransientNetworkError, classify
from tests.adapters.youtube.fakes import completed

VIDEO_ID = "jNQXAC9IVRw"
CHANNEL_ID = "UCsXVk37bltHxD1rDPwtNM8Q"
SECRET_VALUE = "SENTINEL-sid-value-9f3a1c"
SECRET_NAME = "SENTINEL_cookie_name"
COOKIE_LINE = f".youtube.com\tTRUE\t/\tTRUE\t2000000000\t{SECRET_NAME}\t{SECRET_VALUE}"
COOKIE_TEXT = f"# Netscape HTTP Cookie File\n{COOKIE_LINE}\n"


@pytest.fixture
def cookie_file(tmp_path: Path) -> Path:
    path = tmp_path / "mount" / "SENTINEL-cookies-name.txt"
    path.parent.mkdir()
    path.write_text(COOKIE_TEXT, encoding="utf-8")
    path.chmod(0o444)  # as on a read-only bind mount
    return path


class SpyRunner:
    """Records argv and, at call time, what the ``--cookies`` file looked like."""

    def __init__(self, result: subprocess.CompletedProcess[str] | None = None) -> None:
        self.result = result if result is not None else completed(returncode=1, stderr="boom")
        self.argvs: list[list[str]] = []
        self.seen: list[dict[str, object]] = []

    def __call__(
        self, argv: Sequence[str], *, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        args = list(argv)
        self.argvs.append(args)
        if "--cookies" in args:
            path = Path(args[args.index("--cookies") + 1])
            self.seen.append(
                {
                    "path": path,
                    "text": path.read_text(encoding="utf-8"),
                    "mode": stat.S_IMODE(path.stat().st_mode),
                    "writable": os.access(path, os.W_OK),
                    "count": args.count("--cookies"),
                }
            )
        return self.result


# Each entry runs one adapter call against ``runner`` and swallows the
# (expected) failure: only the argv matters here.
def _metadata(runner: SpyRunner, cookie: Path | None) -> None:
    YouTubeMetadata(runner=runner, cookie_file=cookie).fetch(VIDEO_ID)


def _subtitles(runner: SpyRunner, cookie: Path | None) -> None:
    YouTubeSubtitles(runner=runner, cookie_file=cookie).fetch(VIDEO_ID, "en", "manual")


def _audio(runner: SpyRunner, cookie: Path | None, tmp: Path) -> None:
    YouTubeAudio(ytdlp_runner=runner, cookie_file=cookie).fetch_normalized(VIDEO_ID, tmp)


def _catalog(runner: SpyRunner, cookie: Path | None) -> None:
    YouTubeCatalog(runner=runner, cookie_file=cookie).list_uploads(CHANNEL_ID, limit=5)


def _drive(name: str, runner: SpyRunner, cookie: Path | None, tmp: Path) -> None:
    calls: dict[str, Callable[[], None]] = {
        "metadata": lambda: _metadata(runner, cookie),
        "subtitles": lambda: _subtitles(runner, cookie),
        "audio": lambda: _audio(runner, cookie, tmp),
        "catalog": lambda: _catalog(runner, cookie),
    }
    with pytest.raises(JobError):
        calls[name]()


ADAPTERS = ["metadata", "subtitles", "audio", "catalog"]


# --- criterion 1: unset -> no --cookies ------------------------------------


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_without_a_cookie_file_no_invocation_passes_cookies(
    adapter: str, tmp_path: Path
) -> None:
    runner = SpyRunner()
    _drive(adapter, runner, None, tmp_path)
    assert runner.argvs
    assert all("--cookies" not in argv for argv in runner.argvs)


# --- criterion 2: set -> every invocation passes --cookies ------------------


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_with_a_cookie_file_every_invocation_passes_cookies_once(
    adapter: str, tmp_path: Path, cookie_file: Path
) -> None:
    runner = SpyRunner()
    _drive(adapter, runner, cookie_file, tmp_path)
    assert runner.argvs
    assert len(runner.seen) == len(runner.argvs)
    for seen in runner.seen:
        assert seen["count"] == 1
        assert seen["text"] == COOKIE_TEXT


def test_the_metadata_upcoming_retry_also_passes_cookies(
    tmp_path: Path, cookie_file: Path
) -> None:
    upcoming = completed(returncode=1, stderr="ERROR: [youtube] x: This live event will begin in 2 hours.")
    results = [upcoming, completed(returncode=1, stderr="boom")]

    def runner(argv: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        spy(argv, timeout=timeout)
        return results.pop(0)

    spy = SpyRunner()
    with pytest.raises(JobError):
        YouTubeMetadata(runner=runner, cookie_file=cookie_file).fetch(VIDEO_ID)
    assert len(spy.argvs) == 2
    assert len(spy.seen) == 2


def test_cookies_are_added_by_the_shared_helper_only() -> None:
    adapters = Path(__file__).resolve().parents[3] / "adapters" / "youtube"
    offenders = [
        p.name
        for p in adapters.glob("*.py")
        if p.name != "ytdlp.py" and "--cookies" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_yt_dlp_gets_a_private_writable_copy_not_the_read_only_original(
    tmp_path: Path, cookie_file: Path
) -> None:
    runner = SpyRunner(completed(stdout="{}"))
    run_ytdlp(["--dump-json", "u"], timeout=5, runner=runner, cookie_file=cookie_file)

    seen = runner.seen[0]
    copy = seen["path"]
    assert isinstance(copy, Path)
    assert copy != cookie_file
    assert seen["writable"] is True  # yt-dlp saves the jar on exit
    assert seen["mode"] == 0o600
    assert not copy.exists()  # gone once the call returned
    assert cookie_file.read_text(encoding="utf-8") == COOKIE_TEXT


@pytest.mark.parametrize("outcome", ["ok", "exit", "timeout", "crash"])
def test_the_copy_is_removed_however_the_call_ends(
    outcome: str, cookie_file: Path
) -> None:
    paths: list[Path] = []

    def runner(argv: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        paths.append(Path(argv[argv.index("--cookies") + 1]))
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(list(argv), timeout)
        if outcome == "crash":
            raise KeyboardInterrupt
        if outcome == "exit":
            return completed(returncode=1, stderr="boom")
        return completed(stdout="ok")

    try:
        run_ytdlp([], timeout=5, runner=runner, cookie_file=cookie_file)
    except (JobError, KeyboardInterrupt):
        pass
    assert len(paths) == 1
    assert not paths[0].exists()
    assert not paths[0].parent.exists()


# --- criterion 4: contents never surface ------------------------------------


LEAKY_STDERRS = {
    "path": "ERROR: cannot read {path}",
    "line": "ERROR: invalid Netscape format cookies file {path}: '" + COOKIE_LINE + "'",
    "value": f"ERROR: bad cookie {SECRET_VALUE} for {SECRET_NAME}",
    "traceback": (
        "Traceback (most recent call last):\n"
        "PermissionError: [Errno 13] Permission denied: '{path}'\n"
    ),
}


@pytest.mark.parametrize("kind", sorted(LEAKY_STDERRS))
def test_cookie_path_and_contents_never_reach_an_exception_message(
    kind: str, cookie_file: Path
) -> None:
    def runner(argv: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        copy = argv[argv.index("--cookies") + 1]
        # yt-dlp would quote the path it was given; also try the original.
        stderr = LEAKY_STDERRS[kind].format(path=copy) + "\n" + LEAKY_STDERRS[kind].format(
            path=cookie_file
        )
        return completed(returncode=1, stderr=stderr)

    with pytest.raises(JobError) as info:
        run_ytdlp([], timeout=5, runner=runner, cookie_file=cookie_file)

    text = repr(info.value) + str(info.value) + "".join(
        str(a) for a in info.value.args
    )
    for secret in (SECRET_VALUE, SECRET_NAME, COOKIE_LINE, str(cookie_file), cookie_file.name):
        assert secret not in text
    assert "SENTINEL" not in text
    assert "ytdigest-cookies" not in text
    assert classify(info.value) is not None


def test_a_vanished_cookie_file_fails_the_job_without_the_path_or_os_text(
    tmp_path: Path,
) -> None:
    gone = tmp_path / "SENTINEL-gone.txt"
    runner = SpyRunner()
    with pytest.raises(ToolFailureError) as info:
        run_ytdlp([], timeout=5, runner=runner, cookie_file=gone)
    assert "YTDLP_COOKIE_FILE" in str(info.value)
    assert "SENTINEL" not in str(info.value)
    assert runner.argvs == []


def test_a_timeout_message_carries_no_cookie_data(cookie_file: Path) -> None:
    def runner(argv: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(list(argv), timeout)

    with pytest.raises(TransientNetworkError) as info:
        run_ytdlp([], timeout=5, runner=runner, cookie_file=cookie_file)
    text = repr(info.value) + repr(info.value.__cause__)
    assert "SENTINEL" not in text
    assert str(cookie_file) not in text


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_no_log_record_from_any_adapter_carries_cookie_data(
    adapter: str, tmp_path: Path, cookie_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    stderr = f"ERROR: {cookie_file}: '{COOKIE_LINE}'"
    runner = SpyRunner(completed(returncode=1, stderr=stderr))
    with caplog.at_level(logging.DEBUG):
        _drive(adapter, runner, cookie_file, tmp_path)
    rendered = "\n".join(
        f"{r.getMessage()} {r.__dict__} {r.exc_text}" for r in caplog.records
    )
    assert "SENTINEL" not in rendered
    assert str(cookie_file) not in rendered


def test_the_cookie_value_is_not_in_the_argv_only_a_path_is(cookie_file: Path) -> None:
    runner = SpyRunner(completed(stdout="{}"))
    run_ytdlp(["x"], timeout=5, runner=runner, cookie_file=cookie_file)
    assert all(SECRET_VALUE not in part for part in runner.argvs[0])


# --- criterion 3: startup check ---------------------------------------------


def test_check_accepts_unset_and_a_readable_file(cookie_file: Path) -> None:
    check_cookie_file(None)
    check_cookie_file(cookie_file)


def test_check_rejects_a_missing_file_naming_the_variable(tmp_path: Path) -> None:
    with pytest.raises(CookieFileError, match="YTDLP_COOKIE_FILE"):
        check_cookie_file(tmp_path / "nope.txt")


def test_check_rejects_a_directory(tmp_path: Path) -> None:
    with pytest.raises(CookieFileError, match="YTDLP_COOKIE_FILE"):
        check_cookie_file(tmp_path)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read anything")
def test_check_rejects_an_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "locked.txt"
    path.write_text(COOKIE_TEXT, encoding="utf-8")
    path.chmod(0)
    with pytest.raises(CookieFileError, match="YTDLP_COOKIE_FILE") as info:
        check_cookie_file(path)
    assert "SENTINEL" not in str(info.value)


def test_check_never_echoes_the_contents(tmp_path: Path) -> None:
    with pytest.raises(CookieFileError) as info:
        check_cookie_file(tmp_path / "SENTINEL-missing.txt")
    assert SECRET_VALUE not in str(info.value)
