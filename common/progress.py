"""Transcription effort estimate and progress figures (issue #245).

Pure functions, no I/O: the worker stores what they return in ``jobs.progress``
and the API serves it. ``rtf`` is the real-time factor, processing seconds per
audio second (``engine_meta.rtf``).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

#: Seconds between two progress writes of a running transcribe job. Time-based,
#: not tied to segment boundaries, so a long silent stretch still refreshes it.
PROGRESS_INTERVAL_SEC = 30.0

#: How many of the newest finished transcripts (same model and compute type)
#: the estimate's median is taken over.
HISTORY_SIZE = 10

#: ``rtf`` assumed when there is no history for the model and compute type:
#: about real time, which large-v3 at int8 on a typical desktop CPU is in the
#: ballpark of. Such an estimate is marked ``"default"`` and shown as "rough".
DEFAULT_RTF = 1.0

EstimateSource = Literal["history", "default"]


@dataclass(frozen=True, slots=True)
class Estimate:
    """How long transcription should take, and where that number came from."""

    estimated_sec: float
    source: EstimateSource


def _usable(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def estimate_effort(
    duration_sec: float | None, history: Sequence[float], default_rtf: float = DEFAULT_RTF
) -> Estimate | None:
    """``duration_sec x rtf``, with ``rtf`` the median of ``history``.

    Entries of ``history`` that are not positive finite numbers are ignored; with
    none left ``default_rtf`` is used and the source is ``"default"``. A missing,
    non-finite or non-positive duration gives ``None``, never an exception.
    """
    if not _usable(duration_sec):
        return None
    assert duration_sec is not None
    rtfs = [r for r in history if _usable(r)]
    if rtfs:
        return Estimate(duration_sec * statistics.median(rtfs), "history")
    return Estimate(duration_sec * default_rtf, "default")


def progress_figures(
    done_sec: float, duration_sec: float | None, elapsed_sec: float, estimate: Estimate | None
) -> dict[str, Any]:
    """The JSON stored in ``jobs.progress`` (without ``updated_at``, which the database adds).

    ``eta_sec`` comes from the throughput observed so far in this run
    (``done_sec / elapsed_sec``); until some audio is done it is the estimate.
    ``done_sec`` at or above ``duration_sec`` clamps to 100 percent and ETA 0. With
    an unknown duration there is no percent and no ETA from throughput.
    """
    done = max(done_sec, 0.0) if math.isfinite(done_sec) else 0.0
    elapsed = max(elapsed_sec, 0.0) if math.isfinite(elapsed_sec) else 0.0
    estimated = None if estimate is None else estimate.estimated_sec
    percent: int | None = None
    eta: float | None = estimated
    if _usable(duration_sec):
        assert duration_sec is not None
        percent = int(min(done / duration_sec, 1.0) * 100 + 1e-9)
        if done >= duration_sec:
            eta = 0.0
        elif done > 0 and elapsed > 0:
            eta = (duration_sec - done) * (elapsed / done)
    else:
        duration_sec = None
        eta = estimated if done <= 0 else None
    return {
        "percent": percent,
        "done_sec": round(done, 1),
        "duration_sec": None if duration_sec is None else round(duration_sec, 1),
        "elapsed_sec": round(elapsed, 1),
        "eta_sec": None if eta is None else round(eta, 1),
        "estimated_sec": None if estimated is None else round(estimated, 1),
        "estimate_source": None if estimate is None else estimate.source,
    }
