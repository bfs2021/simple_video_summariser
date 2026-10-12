"""AnthropicSummarizer.analyze_chunks: every chunk of an episode in one batch (task #88).

Uses the fake transport from test_anthropic.py. No network, no real key, no sleep.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from adapters.summarize.ollama import OllamaSummarizer
from common.errors import (
    JobError,
    LLMInvalidOutputError,
    LLMUnavailableError,
    RateLimitedError,
)
from common.models import Chunk, ChunkBatchSummarizer
from tests.adapters.summarize.test_anthropic import (
    CHUNK,
    CHUNK_JSON,
    META,
    ROSTER,
    FakeTransport,
    Reply,
    batch_flow,
    make,
    message,
    ok,
    raw,
)
from tests.services.analyzer.fakes import FakeSummarizer

BLANK = Chunk(seq=2, start_sec=200.0, end_sec=300.0, text="   \n")
SECOND = Chunk(seq=3, start_sec=300.0, end_sec=400.0, text="[300] the second chunk")
THIRD = Chunk(seq=4, start_sec=400.0, end_sec=500.0, text="[400] the third chunk")
INVALID = "not json at all"
RATE_LIMITED = {
    "type": "errored",
    "error": {"type": "error", "error": {"type": "rate_limit_error", "message": "m"}},
}


def submitted(t: FakeTransport) -> list[dict[str, Any]]:
    """The requests of the most recent batch create."""
    for req in reversed(t.requests):
        if req.method == "POST" and req.url.endswith("/v1/messages/batches"):
            return list(req.body["requests"])
    raise AssertionError("no batch was submitted")


def posts(t: FakeTransport) -> int:
    return len([r for r in t.requests if r.method == "POST"])


def lines(make_result: Callable[[dict[str, Any]], dict[str, Any] | None]) -> Reply:
    """One results line per submitted request; ``None`` leaves that request without one."""

    def reply(t: FakeTransport) -> Any:
        out = []
        for request in submitted(t):
            result = make_result(request)
            if result is not None:
                out.append({"custom_id": request["custom_id"], "result": result})
        return raw(200, "".join(json.dumps(line) + "\n" for line in out))

    return reply


def succeeded(text: str, **overrides: Any) -> dict[str, Any]:
    return {"type": "succeeded", "message": message(text, **overrides)}


def every(text: str = CHUNK_JSON) -> Reply:
    return lines(lambda _request: succeeded(text))


def topic_json(title: str) -> str:
    return json.dumps(
        {"topics": [{"title": title, "summary": "s", "start_sec": 100}], "claims": [], "quotes": []}
    )


# --- the optional protocol -----------------------------------------------------


def test_only_the_anthropic_summarizer_satisfies_the_batch_protocol() -> None:
    anthropic, _ = make(batch=True)
    assert isinstance(anthropic, ChunkBatchSummarizer)
    assert not isinstance(FakeSummarizer(), ChunkBatchSummarizer)
    assert not isinstance(OllamaSummarizer.__new__(OllamaSummarizer), ChunkBatchSummarizer)


# --- batch mode: one batch for all chunks -----------------------------------------


def test_no_chunks_makes_no_request_and_returns_nothing() -> None:
    summarizer, transport = make(batch=True)

    assert summarizer.analyze_chunks([], ROSTER, META) == []
    assert transport.requests == []


def test_all_blank_chunks_make_no_request_and_get_empty_analyses() -> None:
    summarizer, transport = make(batch=True)

    result = summarizer.analyze_chunks([BLANK], ROSTER, META)

    assert transport.requests == []
    assert result[0].topics == () and result[0].claims == () and result[0].quotes == ()


def test_one_chunk_is_one_batch_with_one_request() -> None:
    summarizer, transport = make(*batch_flow(every()), batch=True)

    summarizer.analyze_chunks([CHUNK], ROSTER, META)

    assert posts(transport) == 1
    assert len(submitted(transport)) == 1


def test_n_chunks_are_one_batch_and_blank_chunks_are_not_sent() -> None:
    summarizer, transport = make(*batch_flow(every()), batch=True)

    result = summarizer.analyze_chunks([CHUNK, BLANK, SECOND], ROSTER, META)

    assert posts(transport) == 1
    assert len(submitted(transport)) == 2
    assert len(result) == 3
    assert result[1].topics == ()


def test_each_batch_request_carries_the_params_a_sync_call_would_send() -> None:
    sync, sync_t = make(ok(CHUNK_JSON), ok(CHUNK_JSON), batch=False)
    sync.analyze_chunk(CHUNK, ROSTER, META)
    sync.analyze_chunk(SECOND, ROSTER, META)
    expected = [r.body for r in sync_t.requests]

    batch, batch_t = make(*batch_flow(every()), batch=True)
    batch.analyze_chunks([CHUNK, SECOND], ROSTER, META)

    assert [r["params"] for r in submitted(batch_t)] == expected


def test_custom_ids_are_unique_and_valid_for_a_hostile_video_id() -> None:
    hostile = replace(META, video_id="-" + "x" * 40)
    summarizer, transport = make(*batch_flow(every()), batch=True)

    summarizer.analyze_chunks([CHUNK, SECOND, THIRD], ROSTER, hostile)

    ids = [r["custom_id"] for r in submitted(transport)]
    assert len(set(ids)) == len(ids) == 3
    for custom_id in ids:
        assert len(custom_id) <= 64
        assert all(c.isalnum() or c in "_-" for c in custom_id)


@given(st.permutations([0, 1, 2]))
def test_for_any_order_of_result_lines_the_list_is_in_input_order(order: list[int]) -> None:
    titles = {CHUNK.text: "Alpha", SECOND.text: "Bravo", THIRD.text: "Charlie"}

    def reply(t: FakeTransport) -> Any:
        requests = submitted(t)
        by_id: dict[str, dict[str, Any]] = {}
        for request in requests:
            prompt = request["params"]["messages"][0]["content"]
            text = next(known for known in titles if known in prompt)
            by_id[request["custom_id"]] = {
                "custom_id": request["custom_id"],
                "result": succeeded(topic_json(titles[text])),
            }
        body = "".join(json.dumps(by_id[requests[i]["custom_id"]]) + "\n" for i in order)
        return raw(200, body)

    summarizer, _ = make(*batch_flow(reply), batch=True)
    result = summarizer.analyze_chunks([CHUNK, SECOND, THIRD], ROSTER, META)

    assert [r.topics[0].title for r in result] == ["Alpha", "Bravo", "Charlie"]


# --- repair --------------------------------------------------------------------


def test_one_invalid_chunk_gets_exactly_one_repair_batch_with_one_request() -> None:
    def first(request: dict[str, Any]) -> dict[str, Any]:
        text = INVALID if request["custom_id"].startswith("chunk-1-") else CHUNK_JSON
        return succeeded(text)

    summarizer, transport = make(
        *batch_flow(lines(first)), *batch_flow(every()), batch=True
    )

    summarizer.analyze_chunks([CHUNK, SECOND], ROSTER, META)

    assert posts(transport) == 2
    repair = submitted(transport)
    assert len(repair) == 1
    assert repair[0]["custom_id"].startswith("repair-1-")
    messages = repair[0]["params"]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == INVALID


def test_no_invalid_chunk_means_no_second_batch() -> None:
    summarizer, transport = make(*batch_flow(every()), batch=True)

    summarizer.analyze_chunks([CHUNK, SECOND], ROSTER, META)

    assert posts(transport) == 1


def test_a_chunk_still_invalid_after_repair_raises_and_there_is_no_third_round() -> None:
    summarizer, transport = make(
        *batch_flow(lines(lambda _r: succeeded(INVALID))),
        *batch_flow(lines(lambda _r: succeeded(INVALID))),
        batch=True,
    )

    with pytest.raises(LLMInvalidOutputError):
        summarizer.analyze_chunks([CHUNK], ROSTER, META)

    assert posts(transport) == 2


def test_refusal_raises_without_repair() -> None:
    summarizer, transport = make(
        *batch_flow(lines(lambda _r: succeeded(CHUNK_JSON, stop_reason="refusal"))),
        batch=True,
    )

    with pytest.raises(LLMInvalidOutputError, match="refused"):
        summarizer.analyze_chunks([CHUNK], ROSTER, META)

    assert posts(transport) == 1


def test_a_truncated_chunk_is_repaired_like_any_invalid_output() -> None:
    summarizer, transport = make(
        *batch_flow(lines(lambda _r: succeeded(CHUNK_JSON, stop_reason="max_tokens"))),
        *batch_flow(every()),
        batch=True,
    )

    summarizer.analyze_chunks([CHUNK], ROSTER, META)

    assert posts(transport) == 2


# --- failures ------------------------------------------------------------------


def test_a_missing_result_line_raises_and_returns_no_partial_list() -> None:
    def make_result(request: dict[str, Any]) -> dict[str, Any] | None:
        return None if request["custom_id"].startswith("chunk-1-") else succeeded(CHUNK_JSON)

    summarizer, _ = make(*batch_flow(lines(make_result)), batch=True)

    with pytest.raises(LLMUnavailableError, match="no line"):
        summarizer.analyze_chunks([CHUNK, SECOND], ROSTER, META)


def test_several_failures_raise_the_lowest_index_error() -> None:
    def make_result(request: dict[str, Any]) -> dict[str, Any]:
        if request["custom_id"].startswith("chunk-2-"):
            return RATE_LIMITED
        if request["custom_id"].startswith("chunk-1-"):
            return {"type": "expired"}
        return succeeded(CHUNK_JSON)

    summarizer, _ = make(*batch_flow(lines(make_result)), batch=True)

    with pytest.raises(LLMUnavailableError, match="expired"):
        summarizer.analyze_chunks([CHUNK, SECOND, THIRD], ROSTER, META)


def test_an_errored_request_is_classified_as_in_a_single_call() -> None:
    summarizer, _ = make(*batch_flow(lines(lambda _r: RATE_LIMITED)), batch=True)

    with pytest.raises(RateLimitedError):
        summarizer.analyze_chunks([CHUNK], ROSTER, META)


def test_usage_counts_succeeded_lines_even_when_the_call_then_raises() -> None:
    def make_result(request: dict[str, Any]) -> dict[str, Any]:
        if request["custom_id"].startswith("chunk-0-"):
            return {"type": "expired"}
        return succeeded(CHUNK_JSON)

    summarizer, _ = make(*batch_flow(lines(make_result)), batch=True)

    with pytest.raises(LLMUnavailableError):
        summarizer.analyze_chunks([CHUNK, SECOND], ROSTER, META)

    assert summarizer.take_usage().calls == 1


# --- sync mode -----------------------------------------------------------------


def test_sync_mode_makes_the_same_calls_in_order_as_per_chunk_calls() -> None:
    per_chunk, per_chunk_t = make(ok(CHUNK_JSON), ok(CHUNK_JSON), batch=False)
    per_chunk.analyze_chunk(CHUNK, ROSTER, META)
    per_chunk.analyze_chunk(SECOND, ROSTER, META)

    grouped, grouped_t = make(ok(CHUNK_JSON), ok(CHUNK_JSON), batch=False)
    grouped.analyze_chunks([CHUNK, SECOND], ROSTER, META)

    assert [r.body for r in grouped_t.requests] == [r.body for r in per_chunk_t.requests]


def test_sync_mode_stops_at_the_first_chunk_that_raises() -> None:
    summarizer, transport = make(
        ok(CHUNK_JSON),
        raw(401, '{"error": {"type": "authentication_error", "message": "m"}}'),
        batch=False,
    )

    with pytest.raises(JobError):
        summarizer.analyze_chunks([CHUNK, SECOND, THIRD], ROSTER, META)

    assert len(transport.requests) == 2
