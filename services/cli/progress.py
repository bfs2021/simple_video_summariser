"""Human-readable progress lines for ``ytdigest run`` (issue #96).

``ProgressPrinter`` is the ``JobContext`` progress hook the CLI installs. It only
renders: when a line is due is decided by the handlers (transcribe's 10% steps and
heartbeat, one event per LLM call), so each event becomes exactly one line.

Non-TTY lines are plain text ending in one ``\\n``. On a TTY, the current status
is redrawn in place until ``close`` finishes the line. A stream that fails
(closed pipe) is ignored: progress is a convenience and never fails the run.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from typing import Any, TextIO

from services.cli.output import format_timestamp, sanitize_text


def _stderr() -> TextIO:
    return sys.stderr  # looked up per call, so a test's captured stderr is honoured


class ProgressPrinter:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        stream: Callable[[], TextIO] | None = None,
    ) -> None:
        self._clock = clock
        self._stream: Callable[[], TextIO] = stream if stream is not None else _stderr
        self._stage_started: float | None = None
        self._active = False

    def begin(self) -> None:
        """A stage started: elapsed time counts from here."""
        self._stage_started = self._clock()

    def __call__(self, stage: str, done: float, total: float, **fields: Any) -> None:
        line = self.render(stage, done, total, fields)
        if line is None:
            return
        try:
            stream = self._stream()
            if self._isatty(stream):
                stream.write(("\r\x1b[2K" if self._active else "") + line)
                self._active = True
            else:
                stream.write(line + "\n")
            stream.flush()
        except (OSError, ValueError):
            pass

    def close(self) -> None:
        if not self._active:
            return
        try:
            stream = self._stream()
            stream.write("\n")
            stream.flush()
        except (OSError, ValueError):
            pass
        finally:
            self._active = False

    @staticmethod
    def _isatty(stream: TextIO) -> bool:
        try:
            return stream.isatty() is True
        except (AttributeError, OSError, ValueError):
            return False

    def render(self, stage: str, done: float, total: float, fields: dict[str, Any]) -> str | None:
        name = " ".join(sanitize_text(stage).split())
        if stage == "analyze":
            return self._analyze(name, done, total, fields)
        if stage == "transcribe":
            return self._transcribe(name, done, total)
        return None

    def _analyze(self, name: str, done: float, total: float, fields: dict[str, Any]) -> str | None:
        phase = fields.get("phase")
        if phase == "roster":
            return f"[{name}] roster"
        if phase == "reduce":
            return f"[{name}] reduce"
        if phase == "chunk" and math.isfinite(done) and math.isfinite(total):
            return f"[{name}] chunk {int(done)}/{int(total)}"
        return None

    def _transcribe(self, name: str, done: float, total: float) -> str | None:
        if not math.isfinite(done):
            return None
        now = self._clock()
        if self._stage_started is None:
            self._stage_started = now
        elapsed = format_timestamp(max(now - self._stage_started, 0.0))
        done = max(done, 0.0)
        if math.isfinite(total) and total > 0:
            done = min(done, total)
            percent = int(done / total * 100 + 1e-9)
            return (
                f"[{name}] {format_timestamp(done)} / {format_timestamp(total)} audio "
                f"({percent}%) - elapsed {elapsed}"
            )
        return f"[{name}] {format_timestamp(done)} audio - elapsed {elapsed}"
