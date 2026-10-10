"""SIGTERM/SIGINT latch for the long-lived entrypoints (issue #108).

Stdlib only, so it adds nothing to an entrypoint's import cost.

A signal that arrives while an entrypoint is still starting up (importing,
reading settings, building handlers) must end the process with status 0, not
the -15 the default disposition gives. Each entrypoint therefore installs a
bare flag-setting handler as the first thing its module does when run as
``__main__`` (before its heavy imports), and its ``main()`` then takes over
with a ``SignalLatch``: the latch installs its own handlers, adopts the early
flag, remembers whether a signal came, and forwards it to whatever object
``main()`` builds (``Scheduler``/``Worker.request_shutdown``), so a signal
that lands between "object built" and "``run()`` installs its own handlers"
is not lost.

Handlers here only set a flag and call the attached callbacks; they log
nothing and take no lock of their own.

This changes nothing after startup: ``Worker.run()`` and ``Scheduler.run()``
install their own handlers over the latch's and restore the latch's on exit.
It also has no part in the hung-handler watchdog's exit 70 (#66), which
leaves through ``os._exit`` and never returns to ``main()``.
"""

from __future__ import annotations

import signal
from collections.abc import Callable
from types import FrameType
from typing import Self

_SIGNALS = (signal.SIGTERM, signal.SIGINT)


class SignalLatch:
    def __init__(self) -> None:
        self.requested = False
        self._callbacks: list[Callable[[], None]] = []
        self._previous: dict[int, object] = {}

    @classmethod
    def start(cls, early_requested: Callable[[], bool]) -> Self:
        """Install the handlers, then adopt a signal the early handler saw.

        ``early_requested`` is read only after the new handlers are in place,
        so a signal can neither fall between the two nor be missed.
        """
        latch = cls()
        latch.install()
        if early_requested():
            latch.requested = True
        return latch

    def install(self) -> None:
        for sig in _SIGNALS:
            self._previous[sig] = signal.signal(sig, self._on_signal)

    def restore(self) -> None:
        """Put back the handlers found at ``install()``; safe to call twice."""
        for sig, previous in self._previous.items():
            signal.signal(sig, previous)  # type: ignore[arg-type]
        self._previous.clear()

    def attach(self, request_shutdown: Callable[[], None]) -> None:
        """Forward the stop request to ``request_shutdown``, now if one already came."""
        self._callbacks.append(request_shutdown)
        if self.requested:
            request_shutdown()

    def _on_signal(self, signum: int, frame: FrameType | None) -> None:
        self.requested = True
        for callback in self._callbacks:
            callback()
