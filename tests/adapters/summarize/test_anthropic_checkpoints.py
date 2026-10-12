"""AnthropicSummarizer inside an analyze job: batch checkpoints, resume, cancel (task #89).

A ``FakeScope`` stands in for the handler's job scope. No database, no network,
no real key, no real sleep.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from adapters.summarize.anthropic import AnthropicSummarizer
from common.errors import Cancelled, LLMUnavailableError
from common.models import BatchCheckpoint
from tests.adapters.summarize.test_anthropic import (
    API,
    BATCH_ID,
    CHUNK,
    KEY,
    META,
    ROSTER,
    ROSTER_JSON,
    FakeTime,
    FakeTransport,
    Reply,
    batch_flow,
    fixture_response,
    raw,
    settings,
)
from tests.adapters.summarize.test_anthropic_chunks import every, lines, succeeded

BASE = datetime(2026, 10, 12, tzinfo=UTC)
WINNER = "msgbatch_WINNER0000000000000000"


class FakeScope:
    """An in-memory ``JobScope``: rows per step, and a clock tied to the fake time."""

    def __init__(self, time: FakeTime) -> None:
        self.job_id = 1
        self.time = time
        self.rows: dict[str, BatchCheckpoint] = {}
        self.cancel = False

    def cancelled(self) -> bool:
        return self.cancel

    def now(self) -> datetime:
        return BASE + timedelta(seconds=self.time.now)

    def checkpoint(self, step_key: str) -> BatchCheckpoint | None:
        return self.rows.get(step_key)

    def record(self, step_key: str, batch_id: str, request_sha256: str) -> BatchCheckpoint:
        if step_key not in self.rows:
            self.rows[step_key] = BatchCheckpoint(step_key, batch_id, request_sha256, self.now())
        return self.rows[step_key]

    def forget(self, step_key: str) -> None:
        self.rows.pop(step_key, None)


class RacingScope(FakeScope):
    """Another worker recorded its own batch for the step first."""

    def record(self, step_key: str, batch_id: str, request_sha256: str) -> BatchCheckpoint:
        row = BatchCheckpoint(step_key, WINNER, request_sha256, self.now())
        self.rows[step_key] = row
        return row


def run_job(
    transport: FakeTransport, time: FakeTime, scope: FakeScope, **overrides: Any
) -> AnthropicSummarizer:
    summarizer = AnthropicSummarizer(
        settings(**overrides), batch=True, transport=transport, clock=time.clock, sleep=time.sleep
    )
    summarizer.begin_job(scope)
    return summarizer


def cancels(transport: FakeTransport) -> list[str]:
    return [r.url for r in transport.requests if r.url.endswith("/cancel")]


def creates(transport: FakeTransport) -> int:
    return len(
        [r for r in transport.requests if r.method == "POST" and r.url.endswith("/batches")]
    )


def roster_result() -> Reply:
    return lines(lambda _r: succeeded(ROSTER_JSON))


def ended() -> Reply:
    return fixture_response("batch_ended.json")


# --- recording before the first poll -------------------------------------------


def test_the_row_is_recorded_before_the_first_poll() -> None:
    time = FakeTime()
    scope = FakeScope(time)
    seen: list[str] = []

    def in_progress(t: FakeTransport) -> Any:
        seen.append(scope.rows["roster"].batch_id)
        return fixture_response("batch_in_progress.json")

    transport = FakeTransport(
        fixture_response("batch_create.json"), in_progress, ended(), roster_result()
    )
    run_job(transport, time, scope).derive_roster(META, "opening")

    assert seen == [BATCH_ID]


def test_the_stored_checkpoint_has_no_secret_in_it() -> None:
    time = FakeTime()
    scope = FakeScope(time)
    transport = FakeTransport(
        fixture_response("batch_create.json"), ended(), roster_result()
    )
    run_job(transport, time, scope).derive_roster(META, "opening")

    assert KEY not in repr(scope.rows)


def test_each_step_has_its_own_key() -> None:
    time = FakeTime()
    scope = FakeScope(time)
    transport = FakeTransport(*batch_flow(roster_result()), *batch_flow(every()))
    summarizer = run_job(transport, time, scope)

    summarizer.derive_roster(META, "opening")
    summarizer.analyze_chunks([CHUNK], ROSTER, META)

    assert set(scope.rows) == {"roster", "chunks"}


# --- resume -----------------------------------------------------------------------


def test_a_restarted_job_resumes_the_recorded_batch_and_submits_nothing() -> None:
    time = FakeTime()
    scope = FakeScope(time)
    transport = FakeTransport(*batch_flow(roster_result()), ended(), roster_result())

    first = run_job(transport, time, scope).derive_roster(META, "opening")
    second = run_job(transport, time, scope).derive_roster(META, "opening")

    assert second == first
    assert creates(transport) == 1


def test_a_changed_request_cancels_the_recorded_batch_and_submits_again() -> None:
    time = FakeTime()
    scope = FakeScope(time)
    transport = FakeTransport(
        *batch_flow(roster_result()),
        raw(200, "{}"),  # cancel of the old batch
        *batch_flow(roster_result()),
    )
    run_job(transport, time, scope).derive_roster(META, "opening")

    run_job(transport, time, scope).derive_roster(replace(META, title="Changed"), "opening")

    assert creates(transport) == 2
    assert len(cancels(transport)) == 1
    assert cancels(transport)[0].startswith(f"{API}/v1/messages/batches/{BATCH_ID}")


def test_a_recorded_batch_the_api_no_longer_knows_is_submitted_again() -> None:
    time = FakeTime()
    scope = FakeScope(time)
    transport = FakeTransport(
        *batch_flow(roster_result()),
        raw(404, '{"error": {"type": "not_found_error", "message": "m"}}'),
        *batch_flow(roster_result()),
    )
    run_job(transport, time, scope).derive_roster(META, "opening")

    result = run_job(transport, time, scope).derive_roster(META, "opening")

    assert result.speakers
    assert creates(transport) == 2


def test_an_expired_result_line_on_resume_is_submitted_again() -> None:
    time = FakeTime()
    scope = FakeScope(time)
    expired = lines(lambda _r: {"type": "expired"})
    transport = FakeTransport(
        *batch_flow(roster_result()), ended(), expired, *batch_flow(roster_result())
    )
    run_job(transport, time, scope).derive_roster(META, "opening")

    run_job(transport, time, scope).derive_roster(META, "opening")

    assert creates(transport) == 2


# --- races, age, cancellation ------------------------------------------------------


def test_a_lost_insert_race_cancels_its_own_batch_and_polls_the_winner() -> None:
    time = FakeTime()
    scope = RacingScope(time)
    transport = FakeTransport(
        fixture_response("batch_create.json"),
        raw(200, "{}"),  # cancel our own batch
        ended(),
        roster_result(),
    )

    run_job(transport, time, scope).derive_roster(META, "opening")

    assert len(cancels(transport)) == 1
    assert any(r.url.endswith(f"/batches/{WINNER}") for r in transport.requests)


def test_max_wait_counts_from_the_recorded_submission_not_from_this_process() -> None:
    time = FakeTime()
    scope = FakeScope(time)
    transport = FakeTransport(*batch_flow(roster_result()), raw(200, "{}"))
    run_job(transport, time, scope).derive_roster(META, "opening")
    time.now += 10_000  # far past ANTHROPIC_BATCH_MAX_WAIT_SEC

    with pytest.raises(LLMUnavailableError, match="did not end"):
        run_job(transport, time, scope).derive_roster(META, "opening")

    assert creates(transport) == 1
    assert "roster" not in scope.rows
    assert len(cancels(transport)) == 1


def test_cancellation_stops_the_wait_without_cancelling_the_batch() -> None:
    time = FakeTime()
    scope = FakeScope(time)

    def in_progress_then_cancelled(t: FakeTransport) -> Any:
        scope.cancel = True
        return fixture_response("batch_in_progress.json")

    transport = FakeTransport(fixture_response("batch_create.json"), in_progress_then_cancelled)

    with pytest.raises(Cancelled):
        run_job(transport, time, scope).derive_roster(META, "opening")

    assert cancels(transport) == []
    assert scope.rows["roster"].batch_id == BATCH_ID
    assert all(seconds <= 1.0 for seconds in time.sleeps)


def test_a_retryable_poll_failure_keeps_the_row_for_the_retry() -> None:
    time = FakeTime()
    scope = FakeScope(time)
    transport = FakeTransport(
        fixture_response("batch_create.json"),
        raw(503, "{}"),
        raw(503, "{}"),
        raw(503, "{}"),
        raw(503, "{}"),
        raw(503, "{}"),
    )

    with pytest.raises(LLMUnavailableError):
        run_job(transport, time, scope).derive_roster(META, "opening")

    assert scope.rows["roster"].batch_id == BATCH_ID


def test_without_a_job_scope_nothing_is_recorded(caplog: pytest.LogCaptureFixture) -> None:
    time = FakeTime()
    scope = FakeScope(time)
    transport = FakeTransport(*batch_flow(roster_result()))
    summarizer = AnthropicSummarizer(
        settings(), batch=True, transport=transport, clock=time.clock, sleep=time.sleep
    )

    with caplog.at_level(logging.INFO):
        summarizer.derive_roster(META, "opening")

    assert scope.rows == {}
    assert creates(transport) == 1
