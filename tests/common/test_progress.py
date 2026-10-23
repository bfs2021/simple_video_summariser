"""The transcription estimate and progress figures (issue #245)."""

from __future__ import annotations

import math

import pytest

from common.progress import (
    DEFAULT_RTF,
    Estimate,
    estimate_effort,
    progress_figures,
)


def test_empty_history_falls_back_to_the_default_rtf_and_says_so() -> None:
    assert estimate_effort(600, [], 0.5) == Estimate(300.0, "default")


def test_the_documented_default_is_used_when_none_is_passed() -> None:
    assert estimate_effort(100, []) == Estimate(100 * DEFAULT_RTF, "default")


def test_one_history_entry_is_its_own_median() -> None:
    assert estimate_effort(600, [0.25], 9.0) == Estimate(150.0, "history")


def test_odd_history_uses_the_middle_value() -> None:
    assert estimate_effort(100, [0.9, 0.1, 0.5], 9.0) == Estimate(50.0, "history")


def test_even_history_averages_the_two_middle_values() -> None:
    assert estimate_effort(100, [0.1, 0.9, 0.3, 0.5], 9.0) == Estimate(40.0, "history")


@pytest.mark.parametrize("duration", [None, 0, -5, math.nan, math.inf])
def test_no_usable_duration_means_no_estimate(duration: float | None) -> None:
    assert estimate_effort(duration, [0.5]) is None


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_non_positive_or_non_finite_history_entries_are_ignored(bad: float) -> None:
    assert estimate_effort(100, [bad, 0.5], 9.0) == Estimate(50.0, "history")


def test_history_of_only_unusable_entries_counts_as_empty() -> None:
    assert estimate_effort(100, [0.0, -2.0], 0.5) == Estimate(50.0, "default")


_EST = Estimate(200.0, "history")


def test_no_throughput_yet_falls_back_to_the_estimate() -> None:
    p = progress_figures(0, 100, 0, _EST)
    assert (p["percent"], p["eta_sec"], p["estimated_sec"], p["estimate_source"]) == (
        0,
        200.0,
        200.0,
        "history",
    )


def test_no_audio_done_after_time_has_passed_still_uses_the_estimate() -> None:
    assert progress_figures(0, 100, 40, _EST)["eta_sec"] == 200.0


def test_eta_comes_from_the_observed_throughput_not_the_estimate() -> None:
    # 25 s of audio in 50 s of work: 2 s per audio second, 75 s left -> 150 s.
    p = progress_figures(25, 100, 50, _EST)
    assert (p["percent"], p["eta_sec"], p["elapsed_sec"], p["done_sec"]) == (25, 150.0, 50.0, 25.0)


def test_eta_follows_rising_throughput() -> None:
    slow = progress_figures(10, 100, 50, _EST)["eta_sec"]
    fast = progress_figures(10, 100, 20, _EST)["eta_sec"]
    assert fast < slow


@pytest.mark.parametrize("done", [100, 150])
def test_done_at_or_above_duration_clamps_to_100_percent_and_zero_eta(done: float) -> None:
    p = progress_figures(done, 100, 80, _EST)
    assert (p["percent"], p["eta_sec"]) == (100, 0.0)


def test_unknown_duration_has_no_percent_and_no_eta() -> None:
    p = progress_figures(30, None, 60, None)
    assert (p["percent"], p["eta_sec"], p["duration_sec"], p["estimated_sec"]) == (
        None,
        None,
        None,
        None,
    )
    assert p["done_sec"] == 30.0


def test_no_estimate_and_no_throughput_gives_no_eta() -> None:
    assert progress_figures(0, 100, 0, None)["eta_sec"] is None
