"""Prometheus text exposition (format 0.0.4) for ``GET /metrics`` (issue #60).

Pure: no I/O, no database, no clock, no third-party package (``prometheus-client``
keeps an in-process registry, which does not fit values that are derived from
Postgres at scrape time and produced by other processes). The route reads the
numbers into a ``MetricsSnapshot``; ``render`` turns that into the body.

Counters here are sums over rows in ``analyses``, not monotonic in-process
counters, so deleting a video (cascade) can lower them. Prometheus treats a
decrease as a counter reset; this is accepted (architecture.md 12, issue #60).

Output is deterministic. Families come in a fixed order, samples inside a family
are sorted by label values, label keys have a fixed order, integers are written
without a decimal point and floats with ``repr``. ``NaN`` and infinities are
never a sample value (``ValueError``); ``+Inf`` appears only as ``le``.

Label values are bounded and carry no free text except ``model``. ``model`` is
untrusted (it comes from ``OLLAMA_MODEL``/``ANTHROPIC_MODEL``): it is sanitised
(control characters replaced), truncated to ``MAX_LABEL_CHARS`` characters,
escaped, and at most ``MAX_MODELS`` distinct values are emitted, the rest
summed into ``model="other"``.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from common.repo.planner_runs import PlannerTaskRun

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

KINDS: tuple[str, ...] = ("ingest", "transcribe", "analyze")
STATES: tuple[str, ...] = ("pending", "running", "done", "dead")
OTHER = "other"
PLANNER_TASKS: tuple[str, ...] = (
    "reap",
    "auto_caption_fallback",
    "audio_retention",
    "stale_analyze",
    "reanalysis_sweep",
    "poll_channels",
    "reconcile_channels",
    "channel_backfill",
    "dead_letter_alerts",
)

#: Upper bounds in seconds, ``le`` of ``job_duration_seconds`` (``+Inf`` is implicit).
DURATION_BUCKETS: tuple[int, ...] = (1, 5, 15, 60, 300, 900, 1800, 3600, 7200, 14400)

MAX_LABEL_CHARS = 128
MAX_MODELS = 20

_REPLACEMENT = "\ufffd"
_BAD_CATEGORIES = frozenset({"Cc", "Cs", "Zl", "Zp"})


@dataclass(frozen=True, slots=True)
class DurationHistogram:
    """Durations of finished jobs of one kind.

    ``cumulative[i]`` is the number of jobs with a duration ``<= DURATION_BUCKETS[i]``
    (cumulative already); ``count`` is the total, i.e. the ``+Inf`` bucket.
    """

    cumulative: tuple[int, ...]
    count: int
    sum_seconds: float


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """Everything one scrape needs, as read from one database snapshot."""

    queue_depth: Mapping[tuple[str, str], int]
    feed_gap_channels: Mapping[str, int]
    job_durations: Mapping[str, DurationHistogram]
    whisper_rtf: float | None
    #: model name -> (input tokens, output tokens)
    token_usage: Mapping[str, tuple[int, int]]
    #: ``None`` when no analysis has a known price
    llm_cost_usd: float | None
    speaker_coercions: int
    audio_bytes: int
    reanalysis_backlog: int
    planner_task_runs: Sequence[PlannerTaskRun] = ()


def sanitize_label_value(value: str) -> str:
    """Replace control characters (newline is kept, for escaping) and truncate.

    ``\\r``, other C0/C1 controls, line/paragraph separators and lone surrogates
    become U+FFFD. The result is at most ``MAX_LABEL_CHARS`` characters.
    """
    cleaned = "".join(
        ch
        if ch == "\n" or unicodedata.category(ch) not in _BAD_CATEGORIES
        else _REPLACEMENT
        for ch in value[:MAX_LABEL_CHARS]
    )
    return cleaned


def escape_label_value(value: str) -> str:
    """Escape per the text format: ``\\`` -> ``\\\\``, ``"`` -> ``\\"``, newline -> ``\\n``."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def format_value(value: float) -> str:
    """Integers without a decimal point, floats with ``repr``; never NaN or infinite."""
    if isinstance(value, bool):
        raise TypeError("a sample value must be a number, not a bool")
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        raise ValueError(f"sample value must be finite, got {value!r}")
    return repr(float(value))


def _sample(name: str, labels: tuple[tuple[str, str], ...], value: float) -> str:
    rendered = format_value(value)
    if not labels:
        return f"{name} {rendered}"
    pairs = ",".join(f'{key}="{escape_label_value(val)}"' for key, val in labels)
    return f"{name}{{{pairs}}} {rendered}"


def _family(name: str, kind: str, help_text: str, samples: Iterable[str]) -> list[str]:
    help_escaped = help_text.replace("\\", "\\\\").replace("\n", "\\n")
    return [f"# HELP {name} {help_escaped}", f"# TYPE {name} {kind}", *samples]


def _fold(value: str, allowed: tuple[str, ...]) -> str:
    return value if value in allowed else OTHER


def _queue_depth_samples(depth: Mapping[tuple[str, str], int]) -> list[str]:
    folded: dict[tuple[str, str], int] = {(k, s): 0 for k in KINDS for s in STATES}
    for (kind, state), count in depth.items():
        key = (_fold(kind, KINDS), _fold(state, STATES))
        folded[key] = folded.get(key, 0) + count
    return [
        _sample("queue_depth", (("kind", k), ("state", s)), n)
        for (k, s), n in sorted(folded.items())
    ]


def _check_histogram(h: DurationHistogram) -> None:
    if len(h.cumulative) != len(DURATION_BUCKETS):
        raise ValueError("histogram must have one cumulative count per bucket")
    previous = 0
    for n in h.cumulative:
        if n < previous:
            raise ValueError("histogram buckets must be cumulative")
        previous = n
    if h.count < previous:
        raise ValueError("histogram count must not be below its last finite bucket")


def _duration_samples(durations: Mapping[str, DurationHistogram]) -> list[str]:
    zero = DurationHistogram((0,) * len(DURATION_BUCKETS), 0, 0.0)
    folded: dict[str, DurationHistogram] = dict.fromkeys(KINDS, zero)
    for kind, hist in durations.items():
        _check_histogram(hist)
        key = _fold(kind, KINDS)
        prev = folded.get(key, zero)
        folded[key] = DurationHistogram(
            tuple(a + b for a, b in zip(prev.cumulative, hist.cumulative, strict=True)),
            prev.count + hist.count,
            prev.sum_seconds + hist.sum_seconds,
        )
    name = "job_duration_seconds"
    lines: list[str] = []
    for kind in sorted(folded):
        hist = folded[kind]
        for bound, n in zip(DURATION_BUCKETS, hist.cumulative, strict=True):
            lines.append(_sample(f"{name}_bucket", (("kind", kind), ("le", str(bound))), n))
        lines.append(_sample(f"{name}_bucket", (("kind", kind), ("le", "+Inf")), hist.count))
        lines.append(_sample(f"{name}_sum", (("kind", kind),), hist.sum_seconds))
        lines.append(_sample(f"{name}_count", (("kind", kind),), hist.count))
    return lines


def bound_models(usage: Mapping[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """Sanitise model names, merge collisions, keep the top ``MAX_MODELS`` by total tokens.

    Ties are broken by (sanitised) name. The rest is summed into ``"other"``.
    """
    merged: dict[str, tuple[int, int]] = {}
    for model, (inp, out) in usage.items():
        key = sanitize_label_value(model)
        prev_in, prev_out = merged.get(key, (0, 0))
        merged[key] = (prev_in + inp, prev_out + out)
    ranked = sorted(merged.items(), key=lambda item: (-(item[1][0] + item[1][1]), item[0]))
    result: dict[str, tuple[int, int]] = dict(ranked[:MAX_MODELS])
    rest_in = sum(v[0] for _, v in ranked[MAX_MODELS:])
    rest_out = sum(v[1] for _, v in ranked[MAX_MODELS:])
    if ranked[MAX_MODELS:]:
        prev_in, prev_out = result.get(OTHER, (0, 0))
        result[OTHER] = (prev_in + rest_in, prev_out + rest_out)
    return result


def _token_samples(usage: Mapping[str, tuple[int, int]]) -> list[str]:
    lines: list[str] = []
    for model, (inp, out) in sorted(bound_models(usage).items()):
        lines.append(_sample("llm_tokens_total", (("model", model), ("direction", "input")), inp))
        lines.append(_sample("llm_tokens_total", (("model", model), ("direction", "output")), out))
    return lines


def _completed_planner_runs(runs: Sequence[PlannerTaskRun]) -> list[PlannerTaskRun]:
    allowed = frozenset(PLANNER_TASKS)
    return sorted(
        (
            run
            for run in runs
            if run.task_name in allowed
            and run.last_finished_at is not None
            and run.last_outcome in ("ok", "failed")
        ),
        key=lambda run: run.task_name,
    )


def render(snapshot: MetricsSnapshot) -> str:
    """The full ``/metrics`` body. Same snapshot, same bytes."""
    lines: list[str] = []
    lines += _family(
        "queue_depth",
        "gauge",
        "Jobs per kind and state. Kinds and states outside the known sets are summed into other.",
        _queue_depth_samples(snapshot.queue_depth),
    )
    lines += _family(
        "feed_gap_channels",
        "gauge",
        "Channels with a current unresolved feed gap, by bounded reason.",
        [
            _sample("feed_gap_channels", (("reason", reason),), snapshot.feed_gap_channels.get(reason, 0))
            for reason in ("suspected", "no_anchor", "listing_window_exhausted")
        ],
    )
    lines += _family(
        "job_duration_seconds",
        "histogram",
        "Seconds from the last claim to completion of done jobs that recorded a start.",
        _duration_samples(snapshot.job_durations),
    )
    lines += _family(
        "whisper_rtf",
        "gauge",
        "Real-time factor of the most recently created whisper transcript with a usable rtf.",
        []
        if snapshot.whisper_rtf is None
        else [_sample("whisper_rtf", (), float(snapshot.whisper_rtf))],
    )
    lines += _family(
        "llm_tokens_total",
        "counter",
        "Tokens recorded on analyses, by model and direction. Sum over stored rows: "
        "deleting a video can lower it.",
        _token_samples(snapshot.token_usage),
    )
    cost = snapshot.llm_cost_usd
    lines += _family(
        "llm_cost_usd_total",
        "counter",
        "Cost in USD recorded on analyses. Analyses with an unknown price are not included.",
        [_sample("llm_cost_usd_total", (), 0 if cost is None else float(cost))],
    )
    lines += _family(
        "speaker_coercions_total",
        "counter",
        "Claim and quote speakers coerced to unknown by the analyzer (speaker attribution "
        "drift canary). Analyses from before the count was recorded are not included.",
        [_sample("speaker_coercions_total", (), snapshot.speaker_coercions)],
    )
    lines += _family(
        "audio_bytes_used",
        "gauge",
        "Recorded bytes of stored audio (sum of media.bytes).",
        [_sample("audio_bytes_used", (), snapshot.audio_bytes)],
    )
    lines += _family(
        "reanalysis_backlog_videos",
        "gauge",
        "Videos with a transcript but no analysis at the configured PROMPT_VERSION.",
        [_sample("reanalysis_backlog_videos", (), snapshot.reanalysis_backlog)],
    )
    completed = _completed_planner_runs(snapshot.planner_task_runs)
    reap = next((run for run in completed if run.task_name == "reap"), None)
    lines += _family(
        "reaped_jobs_total",
        "counter",
        "Stale running jobs returned to pending or moved to dead by completed reaper runs.",
        [] if reap is None or reap.last_ok_at is None else [_sample("reaped_jobs_total", (), reap.cumulative_count)],
    )
    lines += _family(
        "planner_task_last_run_timestamp_seconds",
        "gauge",
        "Unix timestamp when the planner task most recently completed, by task.",
        [
            _sample(
                "planner_task_last_run_timestamp_seconds",
                (("task", run.task_name),),
                run.last_finished_at.timestamp(),
            )
            for run in completed
            if run.last_finished_at is not None
        ],
    )
    lines += _family(
        "planner_task_last_success",
        "gauge",
        "Whether the planner task's most recently completed run succeeded (1) or failed (0).",
        [
            _sample(
                "planner_task_last_success",
                (("task", run.task_name),),
                1 if run.last_outcome == "ok" else 0,
            )
            for run in completed
        ],
    )
    return "\n".join(lines) + "\n"
