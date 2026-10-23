"""``AnthropicSummarizer.cancel_orphan_batch`` (issue #160).

``FakeTransport`` and the recorded fixtures: no network, no real key.
"""

from __future__ import annotations

import json
import logging

import pytest

from common.models import OrphanBatchCanceller, OrphanOutcome
from tests.adapters.summarize.test_anthropic import (
    API,
    BATCH_ID,
    KEY,
    FakeTime,
    FakeTransport,
    Reply,
    fixture_response,
    make,
    raw,
)
from tests.adapters.summarize.test_anthropic_checkpoints import FakeScope

CANCEL_URL = f"{API}/v1/messages/batches/{BATCH_ID}/cancel"


def cancel(*replies: Reply, batch_id: str = BATCH_ID) -> tuple[OrphanOutcome, FakeTransport]:
    summarizer, transport = make(*replies, batch=True)
    return summarizer.cancel_orphan_batch(batch_id), transport


def test_the_summarizer_implements_the_optional_protocol() -> None:
    summarizer, _ = make()

    assert isinstance(summarizer, OrphanBatchCanceller)


def test_a_2xx_cancels_with_one_post_to_the_cancel_url() -> None:
    outcome, transport = cancel(fixture_response("batch_in_progress.json"))

    assert outcome == "cancelled"
    assert [(r.method, r.url) for r in transport.requests] == [("POST", CANCEL_URL)]
    assert transport.requests[0].headers["x-api-key"] == KEY


def test_a_batch_that_already_ended_still_counts_as_cancelled() -> None:
    outcome, _ = cancel(fixture_response("batch_ended.json"))

    assert outcome == "cancelled"


@pytest.mark.parametrize(
    ("status", "fixture_name"),
    [
        (404, "error_not_found.json"),
        (400, "error_invalid_request.json"),
        (413, "error_request_too_large.json"),
    ],
)
def test_a_batch_the_api_does_not_know_or_refuses_is_not_cancellable(
    status: int, fixture_name: str
) -> None:
    outcome, _ = cancel(fixture_response(fixture_name, status))

    assert outcome == "not_cancellable"


@pytest.mark.parametrize(
    ("status", "fixture_name"),
    [
        (401, "error_authentication.json"),
        (403, "error_permission.json"),
        (429, "error_rate_limit.json"),
        (500, "error_api.json"),
        (529, "error_overloaded.json"),
    ],
)
def test_auth_rate_limit_and_server_errors_retry_later(status: int, fixture_name: str) -> None:
    outcome, _ = cancel(fixture_response(fixture_name, status))

    assert outcome == "retry_later"


def test_a_transport_exception_retries_later_and_does_not_raise() -> None:
    outcome, _ = cancel(OSError("connection refused"))

    assert outcome == "retry_later"


def test_a_non_json_error_body_does_not_raise() -> None:
    assert cancel(raw(404, "<html>nope</html>"))[0] == "not_cancellable"
    assert cancel(raw(502, "<html>bad gateway</html>"))[0] == "retry_later"


@pytest.mark.parametrize("bad", ["", "../x", "msgbatch/../../v1/messages", "a b", "x" * 129, "id?x=1"])
def test_a_malformed_batch_id_is_not_cancellable_without_any_request(bad: str) -> None:
    outcome, transport = cancel(batch_id=bad)

    assert outcome == "not_cancellable"
    assert transport.requests == []


def test_it_touches_no_per_job_state() -> None:
    time = FakeTime()
    scope = FakeScope(time)
    summarizer, _ = make(fixture_response("batch_in_progress.json"), batch=True, ftime=time)
    summarizer.begin_job(scope)
    before = summarizer.take_usage()

    outcome = summarizer.cancel_orphan_batch(BATCH_ID)

    assert outcome == "cancelled"
    assert summarizer._scope is scope
    assert summarizer.take_usage() == before
    assert scope.rows == {}
    summarizer.end_job()


def test_the_key_never_leaks_through_the_retry_later_path(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    leaky = raw(
        500, json.dumps({"type": "error", "error": {"type": "api_error", "message": f"key {KEY} bad"}})
    )
    for reply in (leaky, RuntimeError(f"boom {KEY}"), OSError(f"cannot connect with {KEY}")):
        outcome, _ = cancel(reply)
        assert outcome == "retry_later"

    assert KEY not in caplog.text
    for record in caplog.records:
        assert KEY not in json.dumps(record.__dict__, default=str)
