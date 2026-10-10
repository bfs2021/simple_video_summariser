"""The one way YouTube adapters run yt-dlp.

yt-dlp runs as ``sys.executable -m yt_dlp``, so the copy pinned in this
interpreter's environment (requirements.backend.txt) is the one used, never
whatever ``yt-dlp`` happens to be on ``PATH``. It runs in its own process
group: on a timeout the whole group is killed, including anything yt-dlp
started (ffmpeg, a JS runtime).

The process call is injectable (``runner``), so tests never start yt-dlp.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from adapters.youtube.errors import from_ytdlp
from common.errors import ToolFailureError, TransientNetworkError

# How long to wait for a killed process group to close its pipes.
_REAP_TIMEOUT_SEC = 5.0

# What `python -m yt_dlp` prints when the module is missing.
_NOT_INSTALLED = "No module named yt_dlp"


class ProcessRunner(Protocol):
    """Runs ``argv`` to completion; raises ``subprocess.TimeoutExpired`` on timeout."""

    def __call__(
        self, argv: Sequence[str], *, timeout: float
    ) -> subprocess.CompletedProcess[str]: ...


def run_process(
    argv: Sequence[str], *, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` in a new process group and capture its output as text.

    On timeout (or any interruption) the whole group is killed and reaped
    before the exception propagates, so nothing keeps running.
    """
    proc = subprocess.Popen(
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except BaseException:
        _kill_group(proc)
        raise
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout, stderr)


def _kill_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        # Returns once every holder of the pipes, grandchildren too, has exited.
        proc.communicate(timeout=_REAP_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        # Something left the group and still holds a pipe; stop waiting on it.
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                pipe.close()
        proc.wait()


COOKIE_FILE_VARIABLE = "YTDLP_COOKIE_FILE"

# A cookie name or value shorter than this is not redacted on its own: it would
# mangle ordinary words in yt-dlp's messages. Whole lines always are.
_MIN_REDACTED_VALUE_CHARS = 8


class CookieFileError(ValueError):
    """``YTDLP_COOKIE_FILE`` is set but the file cannot be used.

    The message names the variable and the failure class, never the file's
    contents.
    """


def check_cookie_file(path: Path | None) -> None:
    """Fail fast, at service startup, if the configured cookie file is unusable.

    ``None`` (variable unset) is fine. Anything else must be a readable
    regular file; otherwise ``CookieFileError`` names ``YTDLP_COOKIE_FILE``.
    The message carries the path the operator configured, never contents.
    """
    if path is None:
        return
    try:
        with open(path, "rb") as handle:
            handle.read(1)
    except FileNotFoundError:
        reason = "no such file"
    except IsADirectoryError:
        reason = "it is a directory, not a file"
    except PermissionError:
        reason = "permission denied"
    except OSError as exc:
        reason = f"cannot be read (errno {exc.errno})"
    else:
        return
    raise CookieFileError(f"{COOKIE_FILE_VARIABLE} is set but unusable: {reason}")


def _cookie_secrets(text: str, *paths: Path | str) -> list[str]:
    """Everything of a cookie file that must not appear in an error message."""
    secrets = [str(p) for p in paths]
    for raw in text.splitlines():
        line = raw.strip()
        if not line or (line.startswith("#") and not line.startswith("#HttpOnly_")):
            continue
        secrets.append(line)
        # Netscape format: domain, flag, path, secure, expiry, name, value.
        fields = [f.strip() for f in raw.split("\t")]
        for field in fields[5:]:
            if len(field) >= _MIN_REDACTED_VALUE_CHARS:
                secrets.append(field)
    # Longest first, so a whole line goes before the value inside it.
    return sorted({s for s in secrets if s}, key=len, reverse=True)


def _redact(text: str, secrets: Sequence[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "<cookie-file>")
    return text


def run_ytdlp(
    args: Sequence[str],
    *,
    timeout: float,
    runner: ProcessRunner = run_process,
    cookie_file: Path | None = None,
) -> str:
    """Run yt-dlp with ``args`` and return its stdout.

    The one place that builds yt-dlp's argument list: with ``cookie_file``
    set, ``--cookies`` is added here and nowhere else (issue #63). yt-dlp
    rewrites the cookie jar when it exits, which fails on the read-only bind
    mount the file normally lives on, so it is given a private 0600 copy in
    a temporary directory that is removed when the call ends.

    Raises ``TransientNetworkError`` on timeout, ``ToolFailureError`` when
    yt-dlp is not installed or the cookie file cannot be read, and
    ``from_ytdlp(stderr, returncode)`` on any other non-zero exit. Neither
    the cookie file's contents nor its path appear in any of those messages.
    """
    if timeout <= 0:
        raise ValueError(f"timeout must be > 0, got {timeout}")
    if cookie_file is None:
        return _run(args, timeout, runner, ())
    try:
        text = Path(cookie_file).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolFailureError(
            f"{COOKIE_FILE_VARIABLE} cannot be read (errno {exc.errno})"
        ) from None
    with tempfile.TemporaryDirectory(prefix="ytdigest-cookies-") as tmp:
        copy = Path(tmp) / "cookies.txt"
        fd = os.open(copy, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        secrets = _cookie_secrets(text, cookie_file, copy, tmp)
        return _run(["--cookies", str(copy), *args], timeout, runner, secrets)


def _run(
    args: Sequence[str],
    timeout: float,
    runner: ProcessRunner,
    secrets: Sequence[str],
) -> str:
    argv = [sys.executable, "-m", "yt_dlp", *args]
    try:
        result = runner(argv, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise TransientNetworkError(f"yt-dlp timed out after {timeout}s") from None
    except FileNotFoundError as exc:
        raise ToolFailureError(f"cannot start {sys.executable}: {exc}") from None
    if result.returncode != 0:
        if _NOT_INSTALLED in result.stderr:
            raise ToolFailureError(f"yt-dlp is not installed for {sys.executable}")
        raise from_ytdlp(_redact(result.stderr, secrets), result.returncode)
    return result.stdout
