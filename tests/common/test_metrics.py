"""``common.metrics``: the pure /metrics renderer (issue #60). No database."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from datetime import UTC, datetime
from typing import Literal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from common import metrics
from common.metrics import (
    DURATION_BUCKETS,
    MAX_LABEL_CHARS,
    MAX_MODELS,
    DurationHistogram,
    MetricsSnapshot,
    escape_label_value,
    render,
    sanitize_label_value,
)
from common.repo.planner_runs import PlannerTaskRun

N = len(DURATION_BUCKETS)
ZERO_HIST = DurationHistogram((0,) * N, 0, 0.0)

EMPTY = MetricsSnapshot(
    queue_depth={},
    feed_gap_channels={},
    job_durations={},
    whisper_rtf=None,
    token_usage={},
    llm_cost_usd=None,
    speaker_coercions=0,
    audio_bytes=0,
    reanalysis_backlog=0,
    planner_task_runs=[],
)

FAMILIES = [
    "queue_depth",
    "feed_gap_channels",
    "job_duration_seconds",
    "whisper_rtf",
    "llm_tokens_total",
    "llm_cost_usd_total",
    "speaker_coercions_total",
    "audio_bytes_used",
    "reanalysis_backlog_videos",
    "reaped_jobs_total",
    "planner_task_last_run_timestamp_seconds",
    "planner_task_last_success",
]

_LABEL = r'[a-z]+="(?:[^"\\\n]|\\["\\n])*"'
_VALUE = r"(?:-?\d+|-?\d+\.\d+(?:e[+-]?\d+)?|-?\d+e[+-]?\d+)"
SAMPLE = re.compile(rf"^[a-z_]+(?:\{{{_LABEL}(?:,{_LABEL})*\}})? {_VALUE}$")
LABEL = re.compile(r'([a-z]+)="((?:[^"\\\n]|\\["\\n])*)"')


def lines(snapshot: MetricsSnapshot) -> list[str]:
    return render(snapshot).splitlines()


def samples(snapshot: MetricsSnapshot, name: str) -> list[str]:
    return [ln for ln in lines(snapshot) if re.match(rf"{name}(\{{| )", ln)]


def unescape(value: str) -> str:
    return re.sub(r"\\(.)", lambda m: "\n" if m.group(1) == "n" else m.group(1), value)


def hist(*durations: float) -> DurationHistogram:
    return DurationHistogram(
        tuple(sum(1 for d in durations if d <= b) for b in DURATION_BUCKETS),
        len(durations),
        float(sum(durations)),
    )


# -- structure -----------------------------------------------------------------


def test_every_family_has_exactly_one_help_and_type_in_the_documented_order() -> None:
    out = lines(EMPTY)

    helps = [ln.split()[2] for ln in out if ln.startswith("# HELP ")]
    types = [ln.split()[2] for ln in out if ln.startswith("# TYPE ")]
    assert helps == FAMILIES
    assert types == FAMILIES
    for name in FAMILIES:
        assert out.index(next(ln for ln in out if ln.startswith(f"# HELP {name} "))) + 1 == (
            out.index(next(ln for ln in out if ln.startswith(f"# TYPE {name} ")))
        )


def test_metric_types_follow_the_spec() -> None:
    types = dict(ln.split()[2:4] for ln in lines(EMPTY) if ln.startswith("# TYPE "))
    assert types == {
        "queue_depth": "gauge",
        "feed_gap_channels": "gauge",
        "job_duration_seconds": "histogram",
        "whisper_rtf": "gauge",
        "llm_tokens_total": "counter",
        "llm_cost_usd_total": "counter",
        "speaker_coercions_total": "counter",
        "audio_bytes_used": "gauge",
        "reanalysis_backlog_videos": "gauge",
        "reaped_jobs_total": "counter",
        "planner_task_last_run_timestamp_seconds": "gauge",
        "planner_task_last_success": "gauge",
    }


def test_the_body_ends_with_one_newline_and_has_no_timestamps() -> None:
    body = render(EMPTY)
    assert body.endswith("\n")
    assert not body.endswith("\n\n")
    for ln in body.splitlines():
        if not ln.startswith("#"):
            assert SAMPLE.match(ln), ln  # name{labels} value: nothing after the value


def test_content_type_is_text_format_0_0_4() -> None:
    assert metrics.CONTENT_TYPE == "text/plain; version=0.0.4; charset=utf-8"


# -- empty inputs and zero-fill ----------------------------------------------------


def test_empty_tables_give_twelve_zero_queue_series() -> None:
    out = samples(EMPTY, "queue_depth")
    assert len(out) == 12
    assert out[0] == 'queue_depth{kind="analyze",state="dead"} 0'
    assert all(ln.endswith(" 0") for ln in out)
    assert not any("other" in ln for ln in out)


def test_empty_tables_give_zero_histograms_for_the_three_kinds() -> None:
    out = lines(EMPTY)
    assert 'job_duration_seconds_bucket{kind="ingest",le="+Inf"} 0' in out
    assert 'job_duration_seconds_count{kind="transcribe"} 0' in out
    assert 'job_duration_seconds_sum{kind="analyze"} 0.0' in out
    assert not any('kind="other"' in ln for ln in out)


def test_no_whisper_row_means_help_and_type_but_no_sample() -> None:
    assert samples(EMPTY, "whisper_rtf") == []
    assert "# TYPE whisper_rtf gauge" in lines(EMPTY)


def test_no_analyses_means_no_token_samples_and_zero_cost_and_coercions() -> None:
    assert samples(EMPTY, "llm_tokens_total") == []
    assert samples(EMPTY, "llm_cost_usd_total") == ["llm_cost_usd_total 0"]
    assert samples(EMPTY, "speaker_coercions_total") == ["speaker_coercions_total 0"]
    assert samples(EMPTY, "audio_bytes_used") == ["audio_bytes_used 0"]
    assert samples(EMPTY, "reanalysis_backlog_videos") == ["reanalysis_backlog_videos 0"]


def _planner_run(
    task: str,
    *,
    finished: datetime | None,
    outcome: Literal["ok", "failed"] | None,
    cumulative: int = 0,
    error: str | None = None,
    last_ok_at: datetime | None = None,
) -> PlannerTaskRun:
    return PlannerTaskRun(
        task, 60.0, None, finished, outcome, error, None, last_ok_at, None, cumulative
    )


def test_empty_or_incomplete_planner_rows_emit_families_without_samples() -> None:
    snapshot = replace(
        EMPTY,
        planner_task_runs=[_planner_run("reap", finished=None, outcome=None)],
    )

    assert samples(snapshot, "reaped_jobs_total") == []
    assert samples(snapshot, "planner_task_last_run_timestamp_seconds") == []
    assert samples(snapshot, "planner_task_last_success") == []


def test_current_planner_task_labels_are_the_exact_fixed_registry_names() -> None:
    assert metrics.PLANNER_TASKS == (
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


def test_completed_known_tasks_emit_fixed_labels_in_lexical_order() -> None:
    finished = datetime(2026, 10, 22, 12, 30, tzinfo=UTC)
    snapshot = replace(
        EMPTY,
        planner_task_runs=[
            _planner_run("retired_secret", finished=finished, outcome="ok", error="password"),
            _planner_run(
                "reap",
                finished=finished,
                outcome="failed",
                cumulative=7,
                last_ok_at=finished,
            ),
            _planner_run("audio_retention", finished=finished, outcome="ok", cumulative=99),
        ],
    )

    assert samples(snapshot, "reaped_jobs_total") == ["reaped_jobs_total 7"]
    assert samples(snapshot, "planner_task_last_run_timestamp_seconds") == [
        'planner_task_last_run_timestamp_seconds{task="audio_retention"} 1792672200.0',
        'planner_task_last_run_timestamp_seconds{task="reap"} 1792672200.0',
    ]
    assert samples(snapshot, "planner_task_last_success") == [
        'planner_task_last_success{task="audio_retention"} 1',
        'planner_task_last_success{task="reap"} 0',
    ]
    body = render(snapshot)
    assert "retired_secret" not in body
    assert "password" not in body
    assert render(snapshot) == body


def test_successful_zero_reaper_run_emits_zero_counter() -> None:
    finished = datetime(2026, 10, 22, 12, 30, tzinfo=UTC)
    snapshot = replace(
        EMPTY,
        planner_task_runs=[
            _planner_run("reap", finished=finished, outcome="ok", last_ok_at=finished)
        ],
    )

    assert samples(snapshot, "reaped_jobs_total") == ["reaped_jobs_total 0"]


def test_planner_metric_help_text_is_exact() -> None:
    out = lines(EMPTY)
    assert "# HELP reaped_jobs_total Stale running jobs returned to pending or moved to dead by completed reaper runs." in out
    assert "# HELP planner_task_last_run_timestamp_seconds Unix timestamp when the planner task most recently completed, by task." in out
    assert "# HELP planner_task_last_success Whether the planner task's most recently completed run succeeded (1) or failed (0)." in out


# -- number formatting -----------------------------------------------------------------


def test_integers_have_no_decimal_point_and_floats_use_repr() -> None:
    snap = replace(
        EMPTY,
        whisper_rtf=0.1,
        llm_cost_usd=1.5e-05,
        audio_bytes=10**15,
        speaker_coercions=7,
        reanalysis_backlog=3,
    )
    assert samples(snap, "whisper_rtf") == ["whisper_rtf 0.1"]
    assert samples(snap, "llm_cost_usd_total") == [f"llm_cost_usd_total {1.5e-05!r}"]
    assert samples(snap, "audio_bytes_used") == ["audio_bytes_used 1000000000000000"]
    assert samples(snap, "speaker_coercions_total") == ["speaker_coercions_total 7"]
    assert samples(snap, "reanalysis_backlog_videos") == ["reanalysis_backlog_videos 3"]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nan_and_infinity_are_never_rendered(bad: float) -> None:
    for snap in (
        replace(EMPTY, whisper_rtf=bad),
        replace(EMPTY, llm_cost_usd=bad),
        replace(EMPTY, job_durations={"ingest": DurationHistogram((0,) * N, 0, bad)}),
    ):
        with pytest.raises(ValueError, match="finite"):
            render(snap)


def test_inf_appears_only_as_the_le_label() -> None:
    body = render(replace(EMPTY, job_durations={"ingest": hist(2.0)}))
    assert re.findall(r"[Ii]nf", body).count("Inf") == 3  # one +Inf bucket per kind
    assert all(
        'le="+Inf"' in ln for ln in body.splitlines() if "Inf" in ln and not ln.startswith("#")
    )
    assert "nan" not in body.lower()


# -- queue_depth ------------------------------------------------------------------------


def test_unknown_kinds_and_states_are_summed_into_other_not_emitted() -> None:
    snap = replace(
        EMPTY,
        queue_depth={
            ("ingest", "pending"): 2,
            ("notify", "pending"): 3,
            ("weird", "pending"): 4,
            ("ingest", "limbo"): 5,
            ("notify", "limbo"): 6,
        },
    )
    out = samples(snap, "queue_depth")
    assert 'queue_depth{kind="ingest",state="pending"} 2' in out
    assert 'queue_depth{kind="other",state="pending"} 7' in out
    assert 'queue_depth{kind="ingest",state="other"} 5' in out
    assert 'queue_depth{kind="other",state="other"} 6' in out
    assert len(out) == 12 + 3
    assert not any("notify" in ln or "weird" in ln or "limbo" in ln for ln in out)


def test_queue_depth_samples_are_sorted_by_label_values() -> None:
    out = samples(EMPTY, "queue_depth")
    assert out == sorted(out)


# -- job_duration_seconds -----------------------------------------------------------------


def test_histogram_is_cumulative_and_inf_bucket_equals_count() -> None:
    snap = replace(EMPTY, job_durations={"transcribe": hist(0.5, 1.0, 1.0001, 60.0, 60.001, 99999)})
    out = [ln for ln in lines(snap) if 'kind="transcribe"' in ln and ln.startswith("job_")]

    buckets = {
        re.search(r'le="([^"]+)"', ln).group(1): int(ln.rsplit(" ", 1)[1])  # type: ignore[union-attr]
        for ln in out
        if "_bucket" in ln
    }
    assert list(buckets) == [str(b) for b in DURATION_BUCKETS] + ["+Inf"]
    values = list(buckets.values())
    assert values == sorted(values)
    assert buckets["1"] == 2  # 0.5 and exactly 1.0
    assert buckets["60"] == 4  # exactly 60 lands in le="60"
    assert buckets["300"] == 5  # 60.001 does not
    assert buckets["+Inf"] == 6
    assert out[-1] == 'job_duration_seconds_count{kind="transcribe"} 6'
    assert out[-2].startswith('job_duration_seconds_sum{kind="transcribe"} ')


def test_histogram_unknown_kinds_fold_into_other_and_keep_cumulativity() -> None:
    snap = replace(EMPTY, job_durations={"a": hist(2.0), "b": hist(20.0, 5000.0)})
    out = lines(snap)
    assert 'job_duration_seconds_bucket{kind="other",le="5"} 1' in out
    assert 'job_duration_seconds_bucket{kind="other",le="60"} 2' in out
    assert 'job_duration_seconds_bucket{kind="other",le="+Inf"} 3' in out
    assert 'job_duration_seconds_count{kind="other"} 3' in out
    assert 'job_duration_seconds_sum{kind="other"} 5022.0' in out


def test_histogram_kinds_are_sorted() -> None:
    snap = replace(EMPTY, job_durations={"zzz": hist(1.0)})
    kinds = [
        m.group(1)
        for ln in lines(snap)
        if (m := re.match(r'job_duration_seconds_count\{kind="(\w+)"\}', ln))
    ]
    assert kinds == ["analyze", "ingest", "other", "transcribe"]


@pytest.mark.parametrize(
    "bad",
    [
        DurationHistogram((2,) + (1,) * (N - 1), 2, 1.0),  # not cumulative
        DurationHistogram((1,) * (N - 1), 1, 1.0),  # wrong number of buckets
        DurationHistogram((3,) * N, 2, 1.0),  # finite bucket above the count
    ],
)
def test_an_inconsistent_histogram_is_rejected(bad: DurationHistogram) -> None:
    with pytest.raises(ValueError):
        render(replace(EMPTY, job_durations={"ingest": bad}))


# -- llm_tokens_total and the model label ------------------------------------------------------


def test_tokens_have_exactly_two_directions_sorted_by_model() -> None:
    snap = replace(EMPTY, token_usage={"zeta": (1, 2), "alpha": (3, 4)})
    assert samples(snap, "llm_tokens_total") == [
        'llm_tokens_total{model="alpha",direction="input"} 3',
        'llm_tokens_total{model="alpha",direction="output"} 4',
        'llm_tokens_total{model="zeta",direction="input"} 1',
        'llm_tokens_total{model="zeta",direction="output"} 2',
    ]


def _models(n: int) -> dict[str, tuple[int, int]]:
    # model00 has the most tokens, model(n-1) the fewest.
    return {f"model{i:02d}": (1000 - i, 0) for i in range(n)}


def _model_labels(snap: MetricsSnapshot) -> set[str]:
    return {m.group(1) for ln in samples(snap, "llm_tokens_total") if (m := re.search(r'model="([^"]*)"', ln))}


def test_twenty_models_are_all_emitted() -> None:
    labels = _model_labels(replace(EMPTY, token_usage=_models(MAX_MODELS)))
    assert len(labels) == 20
    assert "other" not in labels


def test_the_twenty_first_model_is_folded_into_other() -> None:
    snap = replace(EMPTY, token_usage=_models(MAX_MODELS + 1))
    labels = _model_labels(snap)
    assert len(labels) == 21
    assert "model20" not in labels
    assert 'llm_tokens_total{model="other",direction="input"} 980' in samples(
        snap, "llm_tokens_total"
    )


def test_with_twenty_five_models_the_top_twenty_by_tokens_survive_and_totals_are_conserved() -> None:
    usage = _models(25)
    snap = replace(EMPTY, token_usage=usage)
    labels = _model_labels(snap)
    assert labels == {f"model{i:02d}" for i in range(20)} | {"other"}
    total = sum(
        int(ln.rsplit(" ", 1)[1]) for ln in samples(snap, "llm_tokens_total")
    )
    assert total == sum(i + o for i, o in usage.values())


def test_ties_are_broken_by_name() -> None:
    usage = {f"m{i:02d}": (5, 5) for i in range(25)}
    labels = _model_labels(replace(EMPTY, token_usage=usage))
    assert labels == {f"m{i:02d}" for i in range(20)} | {"other"}


def test_total_tokens_not_just_input_decide_the_ranking() -> None:
    usage = {f"m{i:02d}": (10, 0) for i in range(20)} | {"outputheavy": (0, 11)}
    labels = _model_labels(replace(EMPTY, token_usage=usage))
    assert "outputheavy" in labels
    assert "m19" not in labels  # ties on tokens lose by name, last name first
    assert "m00" in labels


def test_models_that_collide_after_truncation_are_one_series() -> None:
    long = "x" * MAX_LABEL_CHARS
    snap = replace(EMPTY, token_usage={long + "a": (1, 2), long + "b": (10, 20)})
    out = samples(snap, "llm_tokens_total")
    assert out == [
        f'llm_tokens_total{{model="{long}",direction="input"}} 11',
        f'llm_tokens_total{{model="{long}",direction="output"}} 22',
    ]


def test_a_real_model_named_other_is_summed_with_the_overflow() -> None:
    usage = _models(MAX_MODELS + 1) | {"other": (1, 0)}
    out = samples(replace(EMPTY, token_usage=usage), "llm_tokens_total")
    assert len([ln for ln in out if 'model="other",direction="input"' in ln]) == 1


@pytest.mark.parametrize("length,expected", [(127, 127), (128, 128), (129, 128)])
def test_model_labels_are_truncated_at_128_characters(length: int, expected: int) -> None:
    snap = replace(EMPTY, token_usage={"m" * length: (1, 1)})
    (label,) = _model_labels(snap)
    assert len(label) == expected


HOSTILE = ['a"b', "a\\b", "a\nb", "a}b", "a\r\nb", "# TYPE x counter", 'x"} 1\nevil 9', "\x00\x1b\x7f", "é€😀"]


@pytest.mark.parametrize("model", HOSTILE)
def test_hostile_model_names_cannot_break_a_line_or_inject_a_sample(model: str) -> None:
    out = lines(replace(EMPTY, token_usage={model: (1, 2)}))

    for ln in out:
        assert ln.startswith("# ") or SAMPLE.match(ln), ln
    assert len(samples(replace(EMPTY, token_usage={model: (1, 2)}), "llm_tokens_total")) == 2
    assert not any(ln.startswith("evil") for ln in out)
    assert "\r" not in render(replace(EMPTY, token_usage={model: (1, 2)}))


def test_escaping_follows_the_text_format() -> None:
    assert escape_label_value('a\\b"c\nd') == 'a\\\\b\\"c\\nd'


def test_control_characters_other_than_newline_are_replaced() -> None:
    assert sanitize_label_value("a\rb\x00c\x1fd\x7fe\x85f g") == "a�b�c�d�e�f�g"
    assert sanitize_label_value("a\nb") == "a\nb"
    assert sanitize_label_value("\ud800") == "�"


@given(st.text(max_size=300))
def test_every_output_line_is_a_comment_or_a_sample_and_labels_round_trip(model: str) -> None:
    snap = replace(EMPTY, token_usage={model: (3, 4)})

    out = render(snap).split("\n")

    assert out[-1] == ""
    for ln in out[:-1]:
        assert ln.startswith("# ") or SAMPLE.match(ln), repr(ln)
    got = [m.group(2) for ln in samples(snap, "llm_tokens_total") for m in LABEL.finditer(ln) if m.group(1) == "model"]
    expected = sanitize_label_value(model)
    assert got == [escape_label_value(expected)] * 2
    assert [unescape(g) for g in got] == [expected] * 2
    assert len(expected) <= MAX_LABEL_CHARS


@given(st.dictionaries(st.text(max_size=40), st.tuples(st.integers(0, 10**9), st.integers(0, 10**9)), max_size=40))
def test_at_most_twenty_one_models_and_tokens_are_conserved(usage: dict[str, tuple[int, int]]) -> None:
    bounded = metrics.bound_models(usage)

    assert len(bounded) <= MAX_MODELS + 1
    assert sum(i + o for i, o in bounded.values()) == sum(i + o for i, o in usage.values())


# -- determinism and PII ---------------------------------------------------------------------


def test_equal_input_gives_byte_identical_output_regardless_of_insertion_order() -> None:
    a = replace(
        EMPTY,
        queue_depth={("ingest", "done"): 1, ("analyze", "dead"): 2},
        token_usage={"b": (1, 1), "a": (2, 2)},
        job_durations={"ingest": hist(3.0), "analyze": hist(70.0)},
    )
    b = replace(
        EMPTY,
        queue_depth={("analyze", "dead"): 2, ("ingest", "done"): 1},
        token_usage={"a": (2, 2), "b": (1, 1)},
        job_durations={"analyze": hist(70.0), "ingest": hist(3.0)},
    )
    assert render(a).encode() == render(b).encode() == render(a).encode()


def test_only_the_documented_label_keys_exist() -> None:
    snap = replace(
        EMPTY,
        queue_depth={("x", "y"): 1},
        token_usage={"m": (1, 1)},
        job_durations={"ingest": hist(1.0)},
    )
    keys = {m.group(1) for ln in lines(snap) if not ln.startswith("#") for m in LABEL.finditer(ln)}
    assert keys == {"kind", "state", "model", "direction", "le", "reason"}


def test_help_text_is_a_single_line() -> None:
    for ln in lines(EMPTY):
        assert "\n" not in ln
