"""``analyses.cost_usd`` from summarizer usage (issue #102). Needs Postgres."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from common.errors import LLMUnavailableError
from common.models import Segment
from tests.services.analyzer.fakes import FakeSummarizer
from tests.services.analyzer.test_handler import (
    ROSTER,
    Rig,
    rig_factory,  # noqa: F401 - the fixture
)

pytestmark = pytest.mark.integration

# Under 900/60 chunking, 901 one-second segments are two chunks; 2000 are three.
THREE_CHUNKS = {"whisper": tuple(Segment(i, i + 1, f"w{i}") for i in range(2000))}


def _done_line(rig: Rig) -> dict[str, object]:
    (line,) = [r for r in rig.logs.records("info") if "analysis_id" in r]
    return dict(line)


def test_cost_is_the_rounded_sum_of_roster_every_chunk_and_reduce(
    rig_factory: Callable[..., Rig],  # noqa: F811
) -> None:
    summarizer = FakeSummarizer(
        roster=ROSTER,
        costs={"roster": 0.000011, "chunk": 0.000123, "reduce": 0.0000049},
    )
    rig = rig_factory(summarizer, segments=THREE_CHUNKS)

    rig.run()

    assert rig.summarizer.kinds.count("chunk") == 3
    # 0.000011 + 3 * 0.000123 + 0.0000049 = 0.0003849, stored with 5 decimals.
    assert rig.latest().cost_usd == 0.00038
    assert _done_line(rig)["cost_usd"] == 0.00038


def test_distinct_costs_per_call_are_all_counted(
    rig_factory: Callable[..., Rig],  # noqa: F811
) -> None:
    # Different value for every kind, so a missed or doubled call changes the total.
    summarizer = FakeSummarizer(roster=ROSTER, costs={"roster": 0.1, "chunk": 0.02, "reduce": 0.003})
    rig = rig_factory(summarizer, segments=THREE_CHUNKS)

    rig.run()

    assert rig.latest().cost_usd == 0.163  # 0.1 + 3 * 0.02 + 0.003


def test_a_summarizer_that_reports_no_price_stores_null_not_zero(
    rig_factory: Callable[..., Rig],  # noqa: F811
) -> None:
    summarizer = FakeSummarizer(costs={"roster": None, "chunk": None, "reduce": None})
    rig = rig_factory(summarizer)

    rig.run()

    assert rig.latest().cost_usd is None
    assert _done_line(rig)["cost_usd"] is None


def test_one_unpriced_call_among_priced_ones_stores_null_never_a_partial_sum(
    rig_factory: Callable[..., Rig],  # noqa: F811
) -> None:
    summarizer = FakeSummarizer(costs={"roster": 0.5, "chunk": None, "reduce": 0.25})
    rig = rig_factory(summarizer)

    rig.run()

    assert rig.latest().cost_usd is None


def test_an_unpriced_first_call_followed_by_priced_ones_stores_null(
    rig_factory: Callable[..., Rig],  # noqa: F811
) -> None:
    summarizer = FakeSummarizer(costs={"roster": None, "chunk": 0.5, "reduce": 0.25})
    rig = rig_factory(summarizer)

    rig.run()

    assert rig.latest().cost_usd is None


def test_usage_with_no_calls_neither_adds_cost_nor_forces_null(
    rig_factory: Callable[..., Rig],  # noqa: F811
) -> None:
    # The roster and reduce takes report no calls (and no price); only chunks are priced.
    summarizer = FakeSummarizer(costs={"chunk": 0.01})
    rig = rig_factory(summarizer)

    rig.run()

    assert rig.summarizer.kinds.count("chunk") == 2
    assert rig.latest().cost_usd == 0.02


def test_no_reported_usage_at_all_stores_null(
    rig_factory: Callable[..., Rig],  # noqa: F811
) -> None:
    rig = rig_factory(FakeSummarizer())  # the fake's default: calls == 0, cost None

    rig.run()

    assert rig.latest().cost_usd is None


def test_an_empty_transcript_stores_null_cost_and_logs_none(
    rig_factory: Callable[..., Rig],  # noqa: F811
) -> None:
    rig = rig_factory(FakeSummarizer(costs={"roster": 1.0}), segments={"whisper": ()})

    rig.run()

    assert rig.summarizer.calls == []
    assert rig.latest().cost_usd is None
    assert _done_line(rig)["cost_usd"] is None


def test_a_second_job_on_a_reused_summarizer_stores_only_its_own_cost(
    rig_factory: Callable[..., Rig],  # noqa: F811
) -> None:
    summarizer = FakeSummarizer(
        costs={"roster": 1.0, "chunk": 10.0, "reduce": 100.0},
        fail_on={"chunk:1": LLMUnavailableError("down")},
    )
    rig = rig_factory(summarizer)
    with pytest.raises(LLMUnavailableError):
        rig.run()
    # The failed chunk call left its 10.0 pending on the summarizer: only the
    # job-start take_usage() discard keeps it out of the next job's total.
    assert summarizer._calls == 1 and summarizer._cost == 10.0
    summarizer.fail_on.clear()

    rig.run()

    assert rig.latest().cost_usd == 1.0 + 2 * 10.0 + 100.0
