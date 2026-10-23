"""The 30-second progress tick of a transcribe job (issue #245).

An injected clock and an injected ``wait`` stand in for time: nothing sleeps.
"""

from __future__ import annotations

from typing import Any

from common.progress import PROGRESS_INTERVAL_SEC, Estimate
from services.transcriber.progress import ProgressTicker


class Rig:
    """A fake clock that only moves when the ticker "waits", and the writes it made."""

    def __init__(self, *, waits: int, fail_first_write: bool = False) -> None:
        self.now = 1000.0
        self.waits_left = waits
        self.waited: list[float] = []
        self.writes: list[dict[str, Any]] = []
        self.fail_first_write = fail_first_write
        self.on_wait: list[Any] = []

    def clock(self) -> float:
        return self.now

    def wait(self, interval: float) -> bool:
        """``Event.wait`` shape: True means "stop"."""
        self.waited.append(interval)
        if self.waits_left == 0:
            return True
        self.waits_left -= 1
        self.now += interval
        if self.on_wait:
            self.on_wait.pop(0)()
        return False

    def write(self, figures: dict[str, Any]) -> None:
        if self.fail_first_write and not self.writes:
            self.writes.append(figures)
            raise RuntimeError("database down")
        self.writes.append(figures)

    def ticker(self, duration: float | None = 1000.0) -> ProgressTicker:
        return ProgressTicker(
            self.write,
            duration_sec=duration,
            estimate=Estimate(500.0, "default"),
            clock=self.clock,
        )


def test_it_writes_every_interval_and_the_interval_is_the_named_constant() -> None:
    rig = Rig(waits=3)
    ticker = rig.ticker()
    ticker.begin()

    ticker.run(rig.wait)

    assert rig.waited == [PROGRESS_INTERVAL_SEC] * 4  # three ticks, then the stop
    assert [w["elapsed_sec"] for w in rig.writes] == [30.0, 60.0, 90.0]


def test_a_silent_stretch_still_writes_with_unchanged_audio_position() -> None:
    rig = Rig(waits=2)
    ticker = rig.ticker()
    ticker.begin()
    ticker.note(100.0)  # one segment, then nothing for two ticks

    ticker.run(rig.wait)

    assert [w["done_sec"] for w in rig.writes] == [100.0, 100.0]
    assert [w["elapsed_sec"] for w in rig.writes] == [30.0, 60.0]


def test_the_figures_follow_the_audio_position_noted_between_ticks() -> None:
    rig = Rig(waits=2)
    ticker = rig.ticker(duration=1000.0)
    ticker.begin()
    rig.on_wait = [lambda: ticker.note(100.0), lambda: ticker.note(400.0)]

    ticker.run(rig.wait)

    first, second = rig.writes
    assert (first["done_sec"], first["percent"]) == (100.0, 10)
    assert (second["done_sec"], second["percent"]) == (400.0, 40)
    # 400 s of audio in 60 s of work: 600 s of audio left -> 90 s.
    assert second["eta_sec"] == 90.0


def test_before_transcription_begins_elapsed_is_zero_and_eta_is_the_estimate() -> None:
    rig = Rig(waits=0)
    ticker = rig.ticker()
    rig.now += 500  # audio download and model load

    ticker.tick()

    (figures,) = rig.writes
    assert (figures["elapsed_sec"], figures["eta_sec"], figures["estimate_source"]) == (
        0.0,
        500.0,
        "default",
    )


def test_set_target_replaces_the_duration_and_estimate() -> None:
    rig = Rig(waits=0)
    ticker = rig.ticker(duration=None)

    ticker.set_target(200.0, Estimate(100.0, "history"))
    ticker.tick()

    (figures,) = rig.writes
    assert (figures["duration_sec"], figures["estimated_sec"]) == (200.0, 100.0)


def test_a_failing_write_does_not_stop_the_ticking() -> None:
    rig = Rig(waits=2, fail_first_write=True)
    ticker = rig.ticker()
    ticker.begin()

    ticker.run(rig.wait)

    assert len(rig.writes) == 2
