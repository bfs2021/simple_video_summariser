"""Dry-run subtitle probe and estimate maths (issue #117). No Docker, no network.

``_probe_videos`` and ``_estimate`` are exercised directly with a fake clock
and a fake metadata source; nothing sleeps.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
import structlog

from common import backfill as bf
from common.backfill import (
    BACKFILL_DRY_RUN_DEADLINE_SEC,
    BACKFILL_PROBE_CONCURRENCY,
    BACKFILL_PROBE_TIMEOUT_SEC,
    SubtitleProbe,
    _estimate,
    _probe_videos,
    _ProbeOutcome,
)
from common.errors import (
    PermanentSourceError,
    RateLimitedError,
    ToolFailureError,
    TransientNetworkError,
)
from common.models import CatalogEntry

CHANNEL = "UCuAXFkgsw1L7xaCfnd5JJOw"


def _vid(i: int) -> str:
    return f"pv{i:09d}"


def _entries(n: int, duration: int | None = 100) -> list[CatalogEntry]:
    return [CatalogEntry(_vid(i), f"t{i}", duration) for i in range(n)]


class Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, sec: float) -> None:
        with self._lock:
            self.now += sec


class FakeMeta:
    """Records calls; ``manual`` marks which ids have a manual track via ``meta``."""

    def __init__(self, clock: Clock | None = None, *, advance: float = 0.0) -> None:
        self.clock = clock
        self.advance = advance
        self.calls: list[str] = []
        self.timeouts: list[float] = []
        self.errors: dict[str, Exception] = {}
        self.manual: set[str] = set()
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.before: Any = None

    def factory(self, timeout: float) -> FakeMeta:
        with self._lock:
            self.timeouts.append(timeout)
        return self

    def fetch(self, video_id: str) -> Any:
        with self._lock:
            self.calls.append(video_id)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.before is not None:
                self.before(video_id)
            if self.clock is not None:
                self.clock.advance(self.advance)
            if video_id in self.errors:
                raise self.errors[video_id]
            return video_id
        finally:
            with self._lock:
                self.active -= 1

    def probe(self) -> SubtitleProbe:
        return SubtitleProbe(self.factory, lambda meta: str(meta) in self.manual)


def _run(meta: FakeMeta, entries: list[CatalogEntry], clock: Clock | None = None) -> _ProbeOutcome:
    clock = clock or Clock()
    return _probe_videos(entries, meta.probe(), clock, 0.0, CHANNEL)


def test_the_constants_are_the_documented_ones() -> None:
    assert (
        BACKFILL_DRY_RUN_DEADLINE_SEC,
        BACKFILL_PROBE_TIMEOUT_SEC,
        BACKFILL_PROBE_CONCURRENCY,
    ) == (100, 10, 4)


def test_each_video_is_classified_by_its_own_result() -> None:
    meta = FakeMeta()
    meta.manual = {_vid(1), _vid(3)}

    outcome = _run(meta, _entries(5))

    assert outcome.status == {
        _vid(0): "transcribe",
        _vid(1): "manual",
        _vid(2): "transcribe",
        _vid(3): "manual",
        _vid(4): "transcribe",
    }
    assert (outcome.probed, outcome.failed, outcome.deadline_reached) == (5, 0, False)


def test_results_are_keyed_by_video_id_whatever_order_probes_finish_in() -> None:
    meta = FakeMeta()
    meta.manual = {_vid(0)}
    gate = threading.Event()

    def before(video_id: str) -> None:
        if video_id == _vid(0):
            assert gate.wait(10), "the later probes never finished"
        elif video_id == _vid(3):
            gate.set()

    meta.before = before

    outcome = _run(meta, _entries(4))

    assert outcome.status[_vid(0)] == "manual"
    assert {v for v, s in outcome.status.items() if s == "transcribe"} == {
        _vid(1), _vid(2), _vid(3)
    }


def test_the_listing_using_the_whole_deadline_means_no_probe() -> None:
    meta = FakeMeta()

    outcome = _probe_videos(
        _entries(3), meta.probe(), Clock(BACKFILL_DRY_RUN_DEADLINE_SEC + 5), 0.0, CHANNEL
    )

    assert meta.calls == []
    assert outcome.status == {}
    assert outcome.deadline_reached is True


def test_a_probe_never_starts_when_remaining_is_exactly_zero() -> None:
    meta = FakeMeta()

    outcome = _probe_videos(
        _entries(2), meta.probe(), Clock(BACKFILL_DRY_RUN_DEADLINE_SEC), 0.0, CHANNEL
    )

    assert meta.calls == []
    assert outcome.deadline_reached is True


def test_a_probe_starts_with_one_second_remaining() -> None:
    meta = FakeMeta()

    _probe_videos(
        _entries(1), meta.probe(), Clock(BACKFILL_DRY_RUN_DEADLINE_SEC - 1), 0.0, CHANNEL
    )

    assert meta.calls == [_vid(0)]
    assert meta.timeouts == [0.5]


def test_the_deadline_passing_after_three_probes_leaves_the_rest_unchecked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bf, "BACKFILL_PROBE_CONCURRENCY", 1)
    clock = Clock()
    meta = FakeMeta(clock, advance=40)

    outcome = _run(meta, _entries(6), clock)

    assert meta.calls == [_vid(0), _vid(1), _vid(2)]
    assert outcome.probed == 3
    assert outcome.deadline_reached is True


def test_every_video_probed_before_the_deadline_is_not_deadline_reached() -> None:
    meta = FakeMeta()

    outcome = _run(meta, _entries(6))

    assert outcome.probed == 6
    assert outcome.deadline_reached is False


def test_each_probe_timeout_is_the_smaller_of_ten_and_half_the_remaining_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bf, "BACKFILL_PROBE_CONCURRENCY", 1)
    clock = Clock()
    meta = FakeMeta(clock, advance=70)

    _run(meta, _entries(3), clock)

    # remaining 100 -> 10; remaining 30 -> 10 (half is 15); then 100 - 140 < 0: stop.
    assert meta.timeouts == [10, 10]
    clock2 = Clock(88)
    meta2 = FakeMeta()
    _probe_videos(_entries(1), meta2.probe(), clock2, 0.0, CHANNEL)
    assert meta2.timeouts == [6.0]


def test_at_most_four_probes_run_at_once_and_four_do_overlap() -> None:
    meta = FakeMeta()
    barrier = threading.Barrier(BACKFILL_PROBE_CONCURRENCY, timeout=10)
    first_wave = {_vid(i) for i in range(BACKFILL_PROBE_CONCURRENCY)}

    def before(video_id: str) -> None:
        if video_id in first_wave:
            barrier.wait()

    meta.before = before

    outcome = _run(meta, _entries(12))

    assert outcome.probed == 12
    assert meta.max_active == BACKFILL_PROBE_CONCURRENCY


@pytest.mark.parametrize(
    "error",
    [
        PermanentSourceError("private", "x"),
        TransientNetworkError("timeout"),
        ToolFailureError("boom"),
        ValueError("bad"),
    ],
)
def test_a_per_video_failure_leaves_it_unchecked_and_probing_continues(error: Exception) -> None:
    meta = FakeMeta()
    meta.errors[_vid(1)] = error

    outcome = _run(meta, _entries(3))

    assert set(outcome.status) == {_vid(0), _vid(2)}
    assert (outcome.failed, outcome.probed) == (1, 2)
    assert sorted(meta.calls) == [_vid(0), _vid(1), _vid(2)]


def test_rate_limiting_stops_new_probes_and_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bf, "BACKFILL_PROBE_CONCURRENCY", 1)
    meta = FakeMeta()
    meta.errors[_vid(1)] = RateLimitedError("429")

    with structlog.testing.capture_logs() as logs:
        outcome = _run(meta, _entries(5))

    assert meta.calls == [_vid(0), _vid(1)]
    assert set(outcome.status) == {_vid(0)}
    assert outcome.deadline_reached is False
    [line] = [r for r in logs if r["event"] == "backfill.probe_rate_limited"]
    assert line["channel_id"] == CHANNEL
    assert line["probed"] == 1
    assert line["log_level"] == "warning"


def test_rate_limiting_lets_running_probes_finish() -> None:
    meta = FakeMeta()
    meta.errors[_vid(0)] = RateLimitedError("429")
    release = threading.Event()

    def before(video_id: str) -> None:
        if video_id == _vid(0):
            assert release.wait(10)
        if video_id == _vid(1):
            release.set()

    meta.before = before

    outcome = _run(meta, _entries(2))

    assert _vid(1) in outcome.status


def test_an_unexpected_exception_is_not_swallowed() -> None:
    meta = FakeMeta()
    meta.errors[_vid(0)] = RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        _run(meta, _entries(2))


# --- estimate -----------------------------------------------------------------


def _est(
    new: list[CatalogEntry],
    status: dict[str, str] | None = None,
    **kw: Any,
) -> Any:
    args: dict[str, Any] = {
        "rtf": 0.5,
        "rtf_samples": 3,
        "chars_per_sec": 10.0,
        "chars_samples": 4,
        "cost_per_char": 0.001,
        "cost_samples": 2,
        "summarizer": "anthropic",
        "model": "m",
    }
    args.update(kw)
    return _estimate(new, _ProbeOutcome(status or {}), **args)


def test_no_new_videos_give_zero_counts_and_zero_cost() -> None:
    estimate, videos = _est([])

    assert videos == ()
    assert (estimate.manual_subtitles, estimate.needs_transcription, estimate.unchecked) == (0, 0, 0)
    assert estimate.transcription_sec == 0
    assert estimate.llm.cost_usd == 0
    assert estimate.llm.est_input_tokens == 0


def test_every_video_manual_needs_no_transcription() -> None:
    new = _entries(3)

    estimate, videos = _est(new, {e.video_id: "manual" for e in new})

    assert estimate.transcription_sec == 0
    assert estimate.manual_subtitles == 3
    assert [v.transcription_sec for v in videos] == [0, 0, 0]
    assert estimate.transcribe_audio_sec == 0
    assert estimate.audio_sec == 300


def test_every_video_unchecked_equals_the_worst_case() -> None:
    new = _entries(3)

    estimate, videos = _est(new)

    assert estimate.unchecked == 3
    assert estimate.transcription_sec == round(300 * 0.5)
    assert estimate.transcribe_audio_sec == 300
    assert [v.transcription_sec for v in videos] == [50, 50, 50]
    assert [v.subtitles for v in videos] == ["unchecked"] * 3


def test_counts_add_up_to_the_new_videos() -> None:
    new = _entries(4)

    estimate, _ = _est(new, {_vid(0): "manual", _vid(1): "transcribe"})

    assert estimate.manual_subtitles + estimate.needs_transcription + estimate.unchecked == 4
    assert estimate.transcribe_audio_sec == 300  # transcribe + unchecked
    assert estimate.transcription_sec == 150


def test_llm_numbers_per_video_and_total() -> None:
    estimate, videos = _est(_entries(2, duration=100))

    # 100 s * 10 chars/s = 1000 chars; tokens = chars // 4; cost = chars * 0.001
    assert [(v.est_input_tokens, v.llm_cost_usd) for v in videos] == [(250, 1.0), (250, 1.0)]
    assert estimate.llm.est_input_tokens == 500
    assert estimate.llm.cost_usd == 2.0
    assert estimate.llm.basis == "median recorded analyses.cost_usd per transcript character"


def test_ollama_has_tokens_but_a_null_cost_never_zero() -> None:
    estimate, videos = _est(_entries(2), summarizer="ollama", cost_per_char=None, cost_samples=0)

    assert [v.llm_cost_usd for v in videos] == [None, None]
    assert estimate.llm.cost_usd is None
    assert estimate.llm.est_input_tokens == 500


def test_ollama_ignores_a_cost_per_char_even_if_one_is_passed() -> None:
    _, videos = _est(_entries(1), summarizer="ollama")

    assert videos[0].llm_cost_usd is None


def test_no_cost_samples_means_null_cost() -> None:
    estimate, videos = _est(_entries(2), cost_per_char=None, cost_samples=0)

    assert estimate.llm.cost_usd is None
    assert videos[0].est_input_tokens == 250


def test_no_chars_samples_means_everything_llm_is_null() -> None:
    estimate, videos = _est(_entries(2), chars_per_sec=None, chars_samples=0)

    assert videos[0].est_input_tokens is None and videos[0].llm_cost_usd is None
    assert estimate.llm.est_input_tokens is None and estimate.llm.cost_usd is None


def test_an_unknown_duration_nulls_that_video_and_the_totals_but_not_a_manual_transcription() -> None:
    new = [CatalogEntry(_vid(0), "t", None), CatalogEntry(_vid(1), "t", 100)]

    estimate, videos = _est(new, {_vid(0): "manual"})

    assert videos[0].transcription_sec == 0  # manual: nothing to transcribe
    assert videos[0].est_input_tokens is None
    assert videos[1].llm_cost_usd == 1.0
    assert estimate.llm.cost_usd is None  # a partial sum is never reported
    assert estimate.llm.est_input_tokens is None
    assert estimate.unknown_duration == 1


def test_an_unknown_duration_on_a_transcribe_video_has_null_transcription() -> None:
    _, videos = _est([CatalogEntry(_vid(0), "t", None)])

    assert videos[0].transcription_sec is None


def test_no_rtf_samples_give_null_transcription_but_manual_stays_zero() -> None:
    new = _entries(2)

    estimate, videos = _est(new, {_vid(0): "manual"}, rtf=None, rtf_samples=0)

    assert estimate.transcription_sec is None
    assert [v.transcription_sec for v in videos] == [0, None]


def test_a_huge_duration_or_rate_cannot_overflow_the_estimate() -> None:
    new = [CatalogEntry(_vid(0), "t", 10**18)]

    estimate, videos = _est(new, chars_per_sec=1e308, cost_per_char=1e308, rtf=1e308)

    assert videos[0].est_input_tokens is None
    assert videos[0].llm_cost_usd is None
    assert videos[0].transcription_sec is None
    assert estimate.transcription_sec is None
    assert estimate.llm.cost_usd is None


@pytest.mark.parametrize("duration", [10**308, 10**400])
def test_an_absurd_integer_duration_gives_null_estimates_not_an_error_or_a_huge_int(
    duration: int,
) -> None:
    # 10**400 raises OverflowError on int -> float; 10**308 converts, and the
    # rounded product would be a 309-digit integer
    new = [CatalogEntry(_vid(0), "t", duration)]

    estimate, videos = _est(new, rtf=0.5, chars_per_sec=0.5)

    assert videos[0].transcription_sec is None
    assert videos[0].est_input_tokens is None
    assert videos[0].llm_cost_usd is None
    assert estimate.transcription_sec is None


def test_a_zero_duration_video_does_not_divide_or_break() -> None:
    estimate, videos = _est([CatalogEntry(_vid(0), "t", 0)])

    assert videos[0].est_input_tokens == 0
    assert videos[0].llm_cost_usd == 0.0
    assert estimate.audio_sec == 0


def test_cost_is_rounded_to_five_places() -> None:
    _, videos = _est(_entries(1, duration=7), chars_per_sec=1.0, cost_per_char=1.23456789e-4)

    assert videos[0].llm_cost_usd == round(7 * 1.23456789e-4, 5)


def test_a_cost_that_overflows_to_infinity_is_null_not_inf() -> None:
    estimate, videos = _est(
        [CatalogEntry(_vid(0), "t", 10**18)], chars_per_sec=1.0, cost_per_char=1e308
    )

    assert videos[0].est_input_tokens == 10**18 // 4
    assert videos[0].llm_cost_usd is None
    assert estimate.llm.cost_usd is None


def test_a_total_that_overflows_to_infinity_is_null() -> None:
    new = [CatalogEntry(_vid(i), "t", 10**18) for i in range(2)]

    estimate, videos = _est(new, chars_per_sec=1.0, cost_per_char=1e290)

    assert [v.llm_cost_usd for v in videos] == [1e308, 1e308]  # each finite
    assert estimate.llm.cost_usd is None  # the sum is not
