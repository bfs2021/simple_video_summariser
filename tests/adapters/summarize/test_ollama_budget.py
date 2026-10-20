"""Context-window budget for OllamaSummarizer: chunk splitting, roster trimming (#85).

A prompt that would not fit ``OLLAMA_NUM_CTX - OLLAMA_OUTPUT_RESERVE_TOKENS`` is
detected before the call and made to fit. All HTTP goes through the injectable
``Transport`` fake; nothing here touches the network.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import SecretStr

from adapters.summarize.ollama import HttpResponse, OllamaSummarizer, estimate_tokens
from adapters.summarize.prompt_loader import PromptSet, format_roster, load_prompt_set
from adapters.summarize.schema import MAX_CLAIMS, MAX_QUOTES, MAX_TOPICS
from common.config import Settings
from common.errors import LLMInvalidOutputError, LLMUnavailableError
from common.models import Chunk, ChunkAnalysis, Claim, Roster, Topic, Usage, VideoMeta
from tests.adapters.summarize.test_ollama import (
    META,
    ROSTER,
    FakeClock,
    FakeTransport,
    Recorded,
    chat_body,
    ok,
)

RESERVE = 100
EMPTY = '{"topics": [], "claims": [], "quotes": []}'
LOGGER = "adapters.summarize.ollama"


@pytest.fixture(scope="module")
def prompts() -> PromptSet:
    return load_prompt_set("v1")


def reply(content: str = EMPTY) -> HttpResponse:
    """A reply that cannot trip the post-call overflow check."""
    return ok(content, prompt_eval_count=0, eval_count=0)


class AlwaysReply:
    """Transport that answers every request with ``content`` and records them."""

    def __init__(self, content: str = EMPTY) -> None:
        self.body = chat_body(content, prompt_eval_count=0, eval_count=0)
        self.requests: list[Recorded] = []

    def __call__(self, url: str, body: bytes, timeout: float) -> HttpResponse:
        self.requests.append(Recorded(url, json.loads(body), timeout))
        return HttpResponse(200, self.body)


def build(
    prompts: PromptSet,
    transport: Callable[[str, bytes, float], HttpResponse],
    *,
    budget: int,
    reserve: int = RESERVE,
) -> OllamaSummarizer:
    return OllamaSummarizer(
        host="http://h:11434",
        model="m",
        num_ctx=budget + reserve,
        output_reserve_tokens=reserve,
        timeout_sec=1,
        prompts=prompts,
        transport=transport,
        clock=FakeClock(),
    )


def chunk_cost(prompts: PromptSet, chunk: Chunk, roster: Roster = ROSTER) -> int:
    p = prompts.render_chunk(
        title=META.title,
        roster=format_roster([(s.name, s.role) for s in roster.speakers]),
        chunk_start=chunk.start_sec,
        chunk_end=chunk.end_sec,
        transcript=chunk.text,
    )
    return estimate_tokens(p.system) + estimate_tokens(p.user)


def roster_cost(prompts: PromptSet, description: str, opening: str, meta: VideoMeta = META) -> int:
    p = prompts.render_roster(title=meta.title, description=description, opening=opening)
    return estimate_tokens(p.system) + estimate_tokens(p.user)


def make_chunk(lines: list[str], seq: int = 3, start: float = 0.0, end: float = 1000.0) -> Chunk:
    return Chunk(seq=seq, start_sec=start, end_sec=end, text="\n".join(lines))


def dense_lines(n: int, width: int = 80, step: int = 10) -> list[str]:
    return [f"[{i * step}] " + chr(97 + i % 26) * width for i in range(n)]


def sent_transcript(req: Recorded) -> str:
    user = req.body["messages"][1]["content"]
    match = re.search(r"<transcript>\n(.*)\n</transcript>", user, re.DOTALL)
    assert match, user
    return match.group(1)


def sent_span(req: Recorded) -> tuple[int, int]:
    match = re.search(r"seconds (\d+) to (\d+)", req.body["messages"][1]["content"])
    assert match
    return int(match.group(1)), int(match.group(2))


def sent_cost(req: Recorded) -> int:
    system, user = req.body["messages"][:2]
    return estimate_tokens(system["content"]) + estimate_tokens(user["content"])


def reduce_cost(prompts: PromptSet, partials: list[ChunkAnalysis]) -> int:
    rendered = prompts.render_reduce(
        title=META.title,
        partials=json.dumps(
            [
                {
                    "topics": [
                        {"title": t.title, "summary": t.summary, "start_sec": t.start_sec}
                        for t in partial.topics
                    ],
                    "claims": [
                        {
                            "text": claim.text,
                            "speaker": claim.speaker,
                            "start_sec": claim.start_sec,
                            "confidence": claim.confidence,
                        }
                        for claim in partial.claims
                    ],
                    "quotes": [],
                }
                for partial in partials
            ],
            ensure_ascii=False,
            indent=1,
        ),
    )
    return estimate_tokens(rendered.system) + estimate_tokens(rendered.user)


def reduce_partial(marker: str, width: int = 80) -> ChunkAnalysis:
    return ChunkAnalysis(
        topics=(Topic(seq=0, title=marker, summary=marker + "s" * width, start_sec=1.0),),
        claims=(Claim(text=marker + "c" * width, speaker="unknown", start_sec=2.0),),
        quotes=(),
    )


def sent_reduce_partials(req: Recorded) -> list[dict[str, Any]]:
    user = req.body["messages"][1]["content"]
    match = re.search(r"<partials>\n(.*)\n</partials>", user, re.DOTALL)
    assert match, user
    value = json.loads(match.group(1))
    assert isinstance(value, list)
    return value


# --- reduce ------------------------------------------------------------------


def test_reduce_exactly_at_budget_keeps_the_one_shot_path(
    prompts: PromptSet, caplog: pytest.LogCaptureFixture
) -> None:
    partials = [reduce_partial("A"), reduce_partial("B")]
    cost = reduce_cost(prompts, partials)
    transport = FakeTransport(reply("final"))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert build(prompts, transport, budget=cost).reduce(partials, META) == "final"
    assert len(transport.requests) == 1
    assert not [r for r in caplog.records if r.getMessage() == "ollama_reduce_hierarchical"]


def test_reduce_one_token_over_budget_uses_fewest_consecutive_groups(prompts: PromptSet) -> None:
    partials = [reduce_partial(marker) for marker in "ABCD"]
    budget = reduce_cost(prompts, partials) - 1
    transport = FakeTransport(reply("AB"), reply("CD"), reply("final"))
    assert build(prompts, transport, budget=budget).reduce(partials, META) == "final"
    sent = [sent_reduce_partials(req) for req in transport.requests]
    assert [[p["topics"][0]["title"] for p in group] for group in sent[:2]] == [
        ["A", "B", "C"],
        ["D"],
    ]
    assert [p["topics"][0]["title"] for p in sent[2]] == [
        "Original partials 1-3",
        "Original partials 4-4",
    ]
    assert [p["topics"][0]["summary"] for p in sent[2]] == ["AB", "CD"]
    assert all(p["topics"][0]["start_sec"] is None for p in sent[2])
    assert all(p["claims"] == [] and p["quotes"] == [] for p in sent[2])
    assert all(sent_cost(req) <= budget for req in transport.requests)


def test_reduce_repeats_intermediate_levels_in_order_and_accounts_for_every_call(
    prompts: PromptSet,
) -> None:
    partials = [reduce_partial(marker, 170) for marker in "ABCDEFGH"]
    pair_cost = max(reduce_cost(prompts, partials[i : i + 2]) for i in range(0, 8, 2))
    summaries = ["summary " + marker * 300 for marker in "abcd"]
    transport = FakeTransport(
        *(ok(text, prompt_eval_count=10, eval_count=1) for text in summaries),
        ok("left", prompt_eval_count=20, eval_count=2),
        ok("right", prompt_eval_count=30, eval_count=3),
        ok("final", prompt_eval_count=40, eval_count=4),
    )
    summarizer = build(prompts, transport, budget=pair_cost)
    assert summarizer.reduce(partials, META) == "final"
    assert len(transport.requests) == 7
    assert all(sent_cost(req) <= pair_cost for req in transport.requests)
    assert summarizer.take_usage() == Usage(input_tokens=130, output_tokens=13, calls=7)
    assert summarizer.take_usage() == Usage()


def test_reduce_repairs_an_intermediate_and_stops_after_a_later_failure(
    prompts: PromptSet,
) -> None:
    partials = [reduce_partial(marker) for marker in "ABCD"]
    budget = reduce_cost(prompts, partials) - 1
    transport = FakeTransport(reply(""), reply("AB"), ConnectionRefusedError("down"), reply("later"))
    summarizer = build(prompts, transport, budget=budget)
    with pytest.raises(LLMUnavailableError):
        summarizer.reduce(partials, META)
    assert len(transport.requests) == 3


def test_reduce_invalid_after_repair_stops_before_later_groups(prompts: PromptSet) -> None:
    partials = [reduce_partial(marker) for marker in "ABCD"]
    budget = reduce_cost(prompts, partials) - 1
    transport = FakeTransport(reply(""), reply(""), reply("later"))
    with pytest.raises(LLMInvalidOutputError):
        build(prompts, transport, budget=budget).reduce(partials, META)
    assert len(transport.requests) == 2


def test_reduce_post_call_overflow_stops_before_later_groups(prompts: PromptSet) -> None:
    partials = [reduce_partial(marker) for marker in "ABCD"]
    budget = reduce_cost(prompts, partials) - 1
    transport = FakeTransport(
        ok("AB", prompt_eval_count=budget + RESERVE - 1, eval_count=1), reply("later")
    )
    with pytest.raises(LLMInvalidOutputError, match="ran out of context"):
        build(prompts, transport, budget=budget).reduce(partials, META)
    assert len(transport.requests) == 1


def test_reduce_level_that_cannot_shrink_names_the_blocking_range_without_calls(
    prompts: PromptSet,
) -> None:
    partials = [reduce_partial(marker) for marker in "ABC"]
    budget = max(reduce_cost(prompts, [partial]) for partial in partials)
    level_estimate = reduce_cost(prompts, partials)
    assert level_estimate > budget
    transport = FakeTransport()
    with pytest.raises(LLMInvalidOutputError) as excinfo:
        build(prompts, transport, budget=budget).reduce(partials, META)
    message = str(excinfo.value)
    assert transport.requests == []
    assert "original partials 1-3" in message
    assert f"{level_estimate} tokens" in message


def test_reduce_impossible_singleton_names_range_and_settings_without_call(
    prompts: PromptSet,
) -> None:
    partial = reduce_partial("SECRET", 500)
    estimated = reduce_cost(prompts, [partial])
    budget = estimated - 1
    transport = FakeTransport()
    with pytest.raises(LLMInvalidOutputError) as excinfo:
        build(prompts, transport, budget=budget).reduce([partial], META)
    message = str(excinfo.value)
    assert transport.requests == []
    assert "reduce" in message
    assert "original partial 1" in message
    assert f"{estimated} tokens" in message
    assert f"OLLAMA_NUM_CTX={budget + RESERVE}" in message
    assert f"OLLAMA_OUTPUT_RESERVE_TOKENS={RESERVE}" in message


def test_reduce_hierarchical_log_is_once_and_contains_no_content(
    prompts: PromptSet, caplog: pytest.LogCaptureFixture
) -> None:
    partials = [reduce_partial(marker) for marker in ("SECRET-A", "SECRET-B", "SECRET-C", "SECRET-D")]
    initial = reduce_cost(prompts, partials)
    transport = FakeTransport(reply("SECRET-SUMMARY-1"), reply("SECRET-SUMMARY-2"), reply("final"))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        build(prompts, transport, budget=initial - 1).reduce(partials, META)
    [record] = [r for r in caplog.records if r.getMessage() == "ollama_reduce_hierarchical"]
    assert record.__dict__["original_partials"] == 4
    assert record.__dict__["initial_estimated_tokens"] == initial
    assert record.__dict__["budget"] == initial - 1
    assert record.__dict__["levels"] == 1
    assert record.__dict__["reduce_calls"] == 3
    assert not any("SECRET" in record.getMessage() for record in caplog.records)


# --- estimate_tokens ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", 0), ("a", 1), ("abc", 1), ("abcd", 2), ("abcdef", 2), ("abcdefg", 3), ("é", 1), ("éé", 2), ("日", 1), ("日本", 2)],
)
def test_estimate_tokens_is_ceil_of_utf8_bytes_over_three(text: str, expected: int) -> None:
    assert estimate_tokens(text) == expected


@given(st.text(max_size=200))
def test_property_estimate_tokens_is_ceil_of_utf8_bytes_over_three(text: str) -> None:
    n = len(text.encode("utf-8"))
    assert estimate_tokens(text) == -(-n // 3)


# --- settings wiring ---------------------------------------------------------


def test_from_settings_uses_num_ctx_minus_the_output_reserve_as_budget(prompts: PromptSet) -> None:
    chunk = make_chunk(dense_lines(6))
    cost = chunk_cost(prompts, chunk)
    reserve = 700
    config = Settings(
        DATABASE_URL=SecretStr("postgresql://u:p@h/d"),
        OLLAMA_NUM_CTX=cost + reserve,
        OLLAMA_OUTPUT_RESERVE_TOKENS=reserve,
    )
    at = FakeTransport(reply())
    OllamaSummarizer.from_settings(config, transport=at).analyze_chunk(chunk, ROSTER, META)
    assert len(at.requests) == 1

    over = FakeTransport(reply(), reply())
    tighter = Settings(
        DATABASE_URL=SecretStr("postgresql://u:p@h/d"),
        OLLAMA_NUM_CTX=cost + reserve - 1,
        OLLAMA_OUTPUT_RESERVE_TOKENS=reserve,
    )
    OllamaSummarizer.from_settings(tighter, transport=over).analyze_chunk(chunk, ROSTER, META)
    assert len(over.requests) == 2


# --- analyze_chunk: at / over budget ------------------------------------------


def test_chunk_exactly_at_budget_makes_one_call(prompts: PromptSet) -> None:
    chunk = make_chunk(dense_lines(6))
    cost = chunk_cost(prompts, chunk)
    transport = FakeTransport(reply())
    build(prompts, transport, budget=cost).analyze_chunk(chunk, ROSTER, META)
    assert len(transport.requests) == 1
    assert sent_transcript(transport.requests[0]) == chunk.text


def test_chunk_one_token_over_budget_makes_two_calls(prompts: PromptSet) -> None:
    chunk = make_chunk(dense_lines(6))
    cost = chunk_cost(prompts, chunk)
    transport = FakeTransport(reply(), reply())
    build(prompts, transport, budget=cost - 1).analyze_chunk(chunk, ROSTER, META)
    assert len(transport.requests) == 2


def test_chunk_one_byte_over_budget_is_split(prompts: PromptSet) -> None:
    lines = dense_lines(6)
    # Pad the last line until the rendered prompt is a whole number of tokens, so
    # one more byte is a whole token more.
    while True:
        chunk = make_chunk(lines)
        p = prompts.render_chunk(
            title=META.title,
            roster=format_roster([(s.name, s.role) for s in ROSTER.speakers]),
            chunk_start=0.0,
            chunk_end=1000.0,
            transcript=chunk.text,
        )
        if len(p.user.encode()) % 3 == 0:
            break
        lines[-1] += "x"
    cost = chunk_cost(prompts, chunk)
    at = FakeTransport(reply())
    build(prompts, at, budget=cost).analyze_chunk(chunk, ROSTER, META)
    assert len(at.requests) == 1

    longer = make_chunk([*lines[:-1], lines[-1] + "x"])
    assert chunk_cost(prompts, longer) == cost + 1
    over = FakeTransport(reply(), reply())
    build(prompts, over, budget=cost).analyze_chunk(longer, ROSTER, META)
    assert len(over.requests) == 2


def test_split_pieces_keep_every_line_in_order_and_show_their_own_span(prompts: PromptSet) -> None:
    lines = dense_lines(9)
    chunk = make_chunk(lines, start=0.0, end=95.0)
    budget = chunk_cost(prompts, chunk) - 1
    transport = AlwaysReply()
    build(prompts, transport, budget=budget).analyze_chunk(chunk, ROSTER, META)

    sent = [sent_transcript(r).split("\n") for r in transport.requests]
    assert [line for piece in sent for line in piece] == lines
    spans = [sent_span(r) for r in transport.requests]
    firsts = [int(re.match(r"\[(\d+)\]", piece[0]).group(1)) for piece in sent]  # type: ignore[union-attr]
    assert [a for a, _ in spans] == firsts
    assert [b for _, b in spans[:-1]] == firsts[1:]
    assert spans[-1][1] == 95
    assert all(sent_cost(r) <= budget for r in transport.requests)


def test_a_chunk_over_budget_is_split_into_the_fewest_pieces(prompts: PromptSet) -> None:
    lines = dense_lines(12)
    chunk = make_chunk(lines, start=0.0, end=120.0)
    # a budget that fits exactly four lines, even with the widest span shown
    one_piece = make_chunk(lines[8:], start=80.0, end=120.0)
    budget = chunk_cost(prompts, one_piece)
    transport = AlwaysReply()
    build(prompts, transport, budget=budget).analyze_chunk(chunk, ROSTER, META)
    assert len(transport.requests) == 3
    assert [len(sent_transcript(r).split("\n")) for r in transport.requests] == [4, 4, 4]


def test_each_piece_prompt_names_the_original_roster(prompts: PromptSet) -> None:
    chunk = make_chunk(dense_lines(6))
    transport = AlwaysReply()
    build(prompts, transport, budget=chunk_cost(prompts, chunk) - 1).analyze_chunk(
        chunk, ROSTER, META
    )
    for req in transport.requests:
        assert "Lex Fridman (host)" in req.body["messages"][1]["content"]


# --- analyze_chunk: validation against the original chunk ------------------------


def claim_json(text: str, start: float, speaker: str = "unknown") -> dict[str, Any]:
    return {"text": text, "speaker": speaker, "start_sec": start, "confidence": "high"}


def analysis_json(topics: Any = (), claims: Any = (), quotes: Any = ()) -> str:
    return json.dumps({"topics": list(topics), "claims": list(claims), "quotes": list(quotes)})


def test_pieces_are_validated_against_the_original_chunk_span_and_roster(
    prompts: PromptSet,
) -> None:
    chunk = make_chunk(dense_lines(6), start=0.0, end=100.0)
    second = analysis_json(
        claims=[
            claim_json("early", 5.0, "Lex Fridman"),  # outside piece 2, inside the chunk
            claim_json("late", 5000.0),  # outside the chunk: clamped to its end
            claim_json("invented", 60.0, "Elon Musk"),
        ]
    )
    transport = FakeTransport(reply(), reply(second))
    summarizer = build(prompts, transport, budget=chunk_cost(prompts, chunk) - 1)
    result = summarizer.analyze_chunk(chunk, ROSTER, META)
    assert len(transport.requests) == 2
    assert [(c.text, c.start_sec, c.speaker) for c in result.claims] == [
        ("early", 5.0, "Lex Fridman"),
        ("late", 100.0, "unknown"),
        ("invented", 60.0, "unknown"),
    ]
    assert all(c.source_chunk_seq == chunk.seq for c in result.claims)
    assert result.start_sec_clamped == 1
    assert result.speaker_coercions == 1


def test_split_result_concatenates_in_piece_order_and_renumbers_topics(prompts: PromptSet) -> None:
    chunk = make_chunk(dense_lines(6), start=0.0, end=100.0)
    first = analysis_json(
        topics=[{"title": "A", "summary": "s", "start_sec": 1}, {"title": "B", "start_sec": 2}],
        claims=[claim_json("c1", 1.0)],
        quotes=[{"text": "q1", "speaker": "unknown", "start_sec": 1}],
    )
    second = analysis_json(
        topics=[{"title": "C", "start_sec": 60}],
        claims=[claim_json("c2", 61.0), claim_json("c3", 62.0)],
        quotes=[{"text": "q2", "speaker": "unknown", "start_sec": 61}],
    )
    transport = FakeTransport(reply(first), reply(second))
    result = build(prompts, transport, budget=chunk_cost(prompts, chunk) - 1).analyze_chunk(
        chunk, ROSTER, META
    )
    assert [(t.seq, t.title) for t in result.topics] == [(0, "A"), (1, "B"), (2, "C")]
    assert [c.text for c in result.claims] == ["c1", "c2", "c3"]
    assert [q.text for q in result.quotes] == ["q1", "q2"]


def test_split_result_is_capped_dropping_items_from_the_end(prompts: PromptSet) -> None:
    chunk = make_chunk(dense_lines(6), start=0.0, end=100.0)
    half = MAX_TOPICS // 2 + 5

    def piece(tag: str, n_topics: int, n_claims: int, n_quotes: int) -> str:
        return analysis_json(
            topics=[{"title": f"{tag}{i}", "start_sec": 1} for i in range(n_topics)],
            claims=[claim_json(f"{tag}{i}", 1.0) for i in range(n_claims)],
            quotes=[
                {"text": f"{tag}{i}", "speaker": "unknown", "start_sec": 1}
                for i in range(n_quotes)
            ],
        )

    transport = FakeTransport(
        reply(piece("a", half, MAX_CLAIMS // 2 + 5, MAX_QUOTES // 2 + 5)),
        reply(piece("b", half, MAX_CLAIMS // 2 + 5, MAX_QUOTES // 2 + 5)),
    )
    result = build(prompts, transport, budget=chunk_cost(prompts, chunk) - 1).analyze_chunk(
        chunk, ROSTER, META
    )
    assert len(result.topics) == MAX_TOPICS
    assert len(result.claims) == MAX_CLAIMS
    assert len(result.quotes) == MAX_QUOTES
    assert result.topics[-1].title == f"b{MAX_TOPICS - half - 1}"
    assert result.claims[-1].text == f"b{MAX_CLAIMS - (MAX_CLAIMS // 2 + 5) - 1}"
    assert result.quotes[-1].text == f"b{MAX_QUOTES - (MAX_QUOTES // 2 + 5) - 1}"
    assert [t.seq for t in result.topics] == list(range(MAX_TOPICS))
    assert result.items_dropped == 10 + 10 + 10


def test_split_counters_are_summed(prompts: PromptSet) -> None:
    chunk = make_chunk(dense_lines(6), start=0.0, end=100.0)
    bad = analysis_json(claims=[claim_json("x", 999.0, "Nobody"), claim_json("", 1.0)])
    transport = FakeTransport(reply(bad), reply(bad))
    result = build(prompts, transport, budget=chunk_cost(prompts, chunk) - 1).analyze_chunk(
        chunk, ROSTER, META
    )
    assert (result.speaker_coercions, result.start_sec_clamped, result.items_dropped) == (2, 2, 2)


def test_unsplit_chunk_result_is_not_reshaped(prompts: PromptSet) -> None:
    chunk = make_chunk(dense_lines(3), start=0.0, end=100.0)
    body = analysis_json(topics=[{"title": "A", "start_sec": 1}])
    result = build(prompts, FakeTransport(reply(body)), budget=10_000).analyze_chunk(
        chunk, ROSTER, META
    )
    assert [t.title for t in result.topics] == ["A"]


# --- analyze_chunk: logging, repair, failures, usage ------------------------------


def test_split_logs_one_info_record(
    prompts: PromptSet, caplog: pytest.LogCaptureFixture
) -> None:
    chunk = make_chunk(dense_lines(6))
    cost = chunk_cost(prompts, chunk)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        build(prompts, FakeTransport(reply(), reply()), budget=cost - 1).analyze_chunk(
            chunk, ROSTER, META
        )
    [record] = [r for r in caplog.records if r.getMessage() == "ollama_chunk_split"]
    assert record.levelno == logging.INFO
    assert record.__dict__["chunk_seq"] == chunk.seq
    assert record.__dict__["pieces"] == 2
    assert record.__dict__["estimated_tokens"] == cost
    assert record.__dict__["budget"] == cost - 1


def test_unsplit_chunk_logs_no_split(prompts: PromptSet, caplog: pytest.LogCaptureFixture) -> None:
    chunk = make_chunk(dense_lines(6))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        build(prompts, FakeTransport(reply()), budget=chunk_cost(prompts, chunk)).analyze_chunk(
            chunk, ROSTER, META
        )
    assert not [r for r in caplog.records if r.getMessage() == "ollama_chunk_split"]


def test_each_piece_gets_its_own_single_repair_attempt(prompts: PromptSet) -> None:
    chunk = make_chunk(dense_lines(6))
    transport = FakeTransport(reply("garbage"), reply(), reply("garbage"), reply())
    build(prompts, transport, budget=chunk_cost(prompts, chunk) - 1).analyze_chunk(
        chunk, ROSTER, META
    )
    assert len(transport.requests) == 4


def test_a_piece_that_stays_invalid_after_repair_fails_the_chunk(prompts: PromptSet) -> None:
    chunk = make_chunk(dense_lines(6))
    transport = FakeTransport(reply(), reply("garbage"), reply("still garbage"), reply())
    summarizer = build(prompts, transport, budget=chunk_cost(prompts, chunk) - 1)
    with pytest.raises(LLMInvalidOutputError):
        summarizer.analyze_chunk(chunk, ROSTER, META)
    assert len(transport.requests) == 3


def test_a_piece_that_is_unavailable_fails_the_chunk_without_a_partial_result(
    prompts: PromptSet,
) -> None:
    chunk = make_chunk(dense_lines(6))
    transport = FakeTransport(reply(), ConnectionRefusedError("down"))
    summarizer = build(prompts, transport, budget=chunk_cost(prompts, chunk) - 1)
    with pytest.raises(LLMUnavailableError):
        summarizer.analyze_chunk(chunk, ROSTER, META)


def test_the_post_call_overflow_check_still_runs_on_every_piece(prompts: PromptSet) -> None:
    chunk = make_chunk(dense_lines(6))
    budget = chunk_cost(prompts, chunk) - 1
    transport = FakeTransport(
        reply(), ok(EMPTY, prompt_eval_count=budget + RESERVE - 5, eval_count=5), reply()
    )
    with pytest.raises(LLMInvalidOutputError, match="OLLAMA_NUM_CTX"):
        build(prompts, transport, budget=budget).analyze_chunk(chunk, ROSTER, META)
    assert len(transport.requests) == 2


def test_usage_after_a_split_chunk_is_the_sum_of_every_call_including_repairs(
    prompts: PromptSet,
) -> None:
    chunk = make_chunk(dense_lines(6))
    transport = FakeTransport(
        ok("garbage", prompt_eval_count=10, eval_count=1),
        ok(EMPTY, prompt_eval_count=20, eval_count=2),
        ok(EMPTY, prompt_eval_count=40, eval_count=4),
    )
    summarizer = build(prompts, transport, budget=chunk_cost(prompts, chunk) - 1)
    summarizer.analyze_chunk(chunk, ROSTER, META)
    usage = summarizer.take_usage()
    assert (usage.input_tokens, usage.output_tokens, usage.calls) == (70, 7, 3)


def test_a_single_over_long_line_raises_without_any_call(prompts: PromptSet) -> None:
    lines = [*dense_lines(2), "[120] " + "z" * 3000, "[130] tail"]
    chunk = make_chunk(lines, seq=7)
    budget = chunk_cost(prompts, make_chunk(lines[:2], end=20.0)) + 5
    transport = FakeTransport()
    summarizer = build(prompts, transport, budget=budget)
    line_alone = make_chunk([lines[2]], seq=7, start=120.0, end=130.0)
    with pytest.raises(LLMInvalidOutputError) as excinfo:
        summarizer.analyze_chunk(chunk, ROSTER, META)
    message = str(excinfo.value)
    assert transport.requests == []
    assert "chunk 7" in message
    assert "second 120" in message
    assert f"{chunk_cost(prompts, line_alone)} tokens" in message
    assert f"OLLAMA_NUM_CTX={budget + RESERVE}" in message


def test_the_line_that_only_just_fits_alone_is_sent(prompts: PromptSet) -> None:
    lines = ["[0] aaaa", "[10] " + "z" * 600, "[20] bbbb"]
    chunk = make_chunk(lines, start=0.0, end=30.0)
    budget = chunk_cost(prompts, make_chunk([lines[1]], start=10.0, end=20.0))
    transport = AlwaysReply()
    build(prompts, transport, budget=budget).analyze_chunk(chunk, ROSTER, META)
    assert lines[1] in [sent_transcript(r) for r in transport.requests]
    assert all(sent_cost(r) <= budget for r in transport.requests)
    assert "\n".join(sent_transcript(r) for r in transport.requests) == "\n".join(lines)


def test_blank_chunk_still_returns_empty_without_a_call(prompts: PromptSet) -> None:
    transport = FakeTransport()
    result = build(prompts, transport, budget=1).analyze_chunk(
        Chunk(seq=1, start_sec=0.0, end_sec=1.0, text="  \n "), ROSTER, META
    )
    assert (result.topics, result.claims, result.quotes) == ((), (), ())
    assert transport.requests == []


line_text_st = st.text(alphabet="abcdéあ xyz", min_size=0, max_size=60)


@st.composite
def chunk_lines(draw: st.DrawFn) -> list[str]:
    n = draw(st.integers(1, 12))
    gaps = draw(st.lists(st.integers(0, 40), min_size=n, max_size=n))
    seconds = [sum(gaps[: i + 1]) for i in range(n)]
    return [f"[{sec}] {draw(line_text_st)}" for sec in seconds]


def min_pieces(prompts: PromptSet, lines: list[str], chunk: Chunk, budget: int) -> int | None:
    """Independent brute force: fewest consecutive pieces whose prompts fit."""
    secs = [int(re.match(r"\[(\d+)\]", line).group(1)) for line in lines]  # type: ignore[union-attr]

    def fits(i: int, j: int) -> bool:  # lines[i:j]
        end = secs[j] if j < len(lines) else chunk.end_sec
        piece = Chunk(seq=chunk.seq, start_sec=secs[i], end_sec=end, text="\n".join(lines[i:j]))
        return chunk_cost(prompts, piece) <= budget

    best: list[float] = [float("inf")] * (len(lines) + 1)
    best[0] = 0
    for j in range(1, len(lines) + 1):
        for i in range(j):
            if best[i] + 1 < best[j] and fits(i, j):
                best[j] = best[i] + 1
    return None if best[-1] == float("inf") else int(best[-1])


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(lines=chunk_lines(), shrink=st.integers(0, 400))
def test_property_split_is_lossless_ordered_fits_and_fewest(
    prompts: PromptSet, lines: list[str], shrink: int
) -> None:
    chunk = make_chunk(lines, start=0.0, end=float(int(lines[-1][1 : lines[-1].index("]")]) + 50))
    full = chunk_cost(prompts, chunk)
    budget = max(1, full - shrink)
    transport = AlwaysReply()
    summarizer = build(prompts, transport, budget=budget)
    expected = 1 if full <= budget else min_pieces(prompts, lines, chunk, budget)
    if expected is None:
        with pytest.raises(LLMInvalidOutputError):
            summarizer.analyze_chunk(chunk, ROSTER, META)
        assert transport.requests == []
        return
    summarizer.analyze_chunk(chunk, ROSTER, META)
    sent = [sent_transcript(r).split("\n") for r in transport.requests]
    assert [line for piece in sent for line in piece] == lines
    assert all(sent_cost(r) <= budget for r in transport.requests)
    assert len(sent) == expected


# --- derive_roster -------------------------------------------------------------


def roster_meta(description: str) -> VideoMeta:
    return replace(META, description=description)


def user_description(req: Recorded) -> str:
    user = req.body["messages"][1]["content"]
    match = re.search(r"<description>\n(.*?)\n</description>", user, re.DOTALL)
    assert match, user
    return match.group(1)


def user_opening(req: Recorded) -> str:
    user = req.body["messages"][1]["content"]
    match = re.search(r"<opening>\n(.*?)\n</opening>", user, re.DOTALL)
    assert match, user
    return match.group(1)


ROSTER_REPLY = '{"speakers": []}'


def test_roster_prompt_exactly_at_budget_is_sent_untouched(
    prompts: PromptSet, caplog: pytest.LogCaptureFixture
) -> None:
    description, opening = "d" * 500, "o" * 500
    budget = roster_cost(prompts, description, opening)
    transport = FakeTransport(reply(ROSTER_REPLY))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        build(prompts, transport, budget=budget).derive_roster(roster_meta(description), opening)
    [req] = transport.requests
    assert user_description(req) == description
    assert user_opening(req) == opening
    assert not [r for r in caplog.records if r.getMessage() == "ollama_roster_trimmed"]


def test_roster_description_is_cut_first_keeping_the_start_and_ending_in_an_ellipsis(
    prompts: PromptSet, caplog: pytest.LogCaptureFixture
) -> None:
    description, opening = "".join(chr(97 + i % 26) for i in range(900)), "o" * 200
    full = roster_cost(prompts, description, opening)
    budget = full - 1
    transport = FakeTransport(reply(ROSTER_REPLY))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        build(prompts, transport, budget=budget).derive_roster(roster_meta(description), opening)
    [req] = transport.requests
    sent = user_description(req)
    assert sent.endswith("…")
    assert description.startswith(sent[:-1])
    assert len(sent) < len(description)
    assert user_opening(req) == opening
    assert sent_cost(req) <= budget
    # the longest cut that fits: one more character would not
    longer = description[: len(sent)] + "…"
    assert roster_cost(prompts, longer, opening) > budget
    [record] = [r for r in caplog.records if r.getMessage() == "ollama_roster_trimmed"]
    assert record.levelno == logging.INFO
    extra = record.__dict__
    assert (extra["description_chars_original"], extra["description_chars_final"]) == (
        len(description),
        len(sent),
    )
    assert (extra["opening_chars_original"], extra["opening_chars_final"]) == (
        len(opening),
        len(opening),
    )


def test_roster_description_cut_never_splits_a_character(prompts: PromptSet) -> None:
    description = "日本語" * 300
    budget = roster_cost(prompts, description, "") - 7
    transport = FakeTransport(reply(ROSTER_REPLY))
    build(prompts, transport, budget=budget).derive_roster(roster_meta(description), "")
    sent = user_description(transport.requests[0])
    assert description.startswith(sent[:-1])
    sent.encode("utf-8")  # whole characters only


def test_roster_opening_is_cut_only_once_the_description_is_gone(
    prompts: PromptSet, caplog: pytest.LogCaptureFixture
) -> None:
    description, opening = "d" * 400, "".join(chr(97 + i % 26) for i in range(900))
    budget = roster_cost(prompts, "", opening) - 1
    transport = FakeTransport(reply(ROSTER_REPLY))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        build(prompts, transport, budget=budget).derive_roster(roster_meta(description), opening)
    [req] = transport.requests
    assert user_description(req) == ""
    sent = user_opening(req)
    assert sent.endswith("…")
    assert opening.startswith(sent[:-1])
    assert sent_cost(req) <= budget
    assert roster_cost(prompts, "", opening[: len(sent)] + "…") > budget
    [record] = [r for r in caplog.records if r.getMessage() == "ollama_roster_trimmed"]
    extra = record.__dict__
    assert (extra["description_chars_original"], extra["description_chars_final"]) == (400, 0)
    assert (extra["opening_chars_original"], extra["opening_chars_final"]) == (900, len(sent))


def test_roster_with_empty_description_and_opening_over_budget_raises_without_a_call(
    prompts: PromptSet,
) -> None:
    budget = roster_cost(prompts, "", "") - 1
    transport = FakeTransport()
    with pytest.raises(LLMInvalidOutputError, match="OLLAMA_NUM_CTX"):
        build(prompts, transport, budget=budget).derive_roster(roster_meta("desc"), "open")
    assert transport.requests == []


def test_roster_with_both_empty_exactly_at_budget_is_sent(prompts: PromptSet) -> None:
    budget = roster_cost(prompts, "", "")
    transport = FakeTransport(reply(ROSTER_REPLY))
    build(prompts, transport, budget=budget).derive_roster(roster_meta("desc"), "open")
    [req] = transport.requests
    assert (user_description(req), user_opening(req)) == ("", "")


roster_text_st = st.text(alphabet="abcdef xyé日", max_size=300)


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(description=roster_text_st, opening=roster_text_st, cut=st.integers(0, 400))
def test_property_roster_trim_fits_keeps_prefixes_and_is_maximal(
    prompts: PromptSet, description: str, opening: str, cut: int
) -> None:
    budget = max(1, roster_cost(prompts, description, opening) - cut)
    transport = FakeTransport(reply(ROSTER_REPLY))
    summarizer = build(prompts, transport, budget=budget)
    if roster_cost(prompts, "", "") > budget:
        with pytest.raises(LLMInvalidOutputError):
            summarizer.derive_roster(roster_meta(description), opening)
        assert transport.requests == []
        return
    summarizer.derive_roster(roster_meta(description), opening)
    [req] = transport.requests
    assert sent_cost(req) <= budget
    sent_d, sent_o = user_description(req), user_opening(req)
    for original, sent in ((description, sent_d), (opening, sent_o)):
        assert sent == original or sent == "" or (
            sent.endswith("…") and original.startswith(sent[:-1])
        )
    if roster_cost(prompts, description, opening) <= budget:
        assert (sent_d, sent_o) == (description, opening)
    elif roster_cost(prompts, "", opening) <= budget:
        # only the description was cut, to the longest prefix that fits
        assert sent_o == opening
        if sent_d.endswith("…"):
            assert roster_cost(prompts, description[: len(sent_d)] + "…", opening) > budget
        else:
            assert sent_d == ""
            assert roster_cost(prompts, "…", opening) > budget
    else:
        assert sent_d == ""


def test_roster_description_cut_can_drop_just_the_last_character(prompts: PromptSet) -> None:
    # A 4-byte last character replaced by the 3-byte ellipsis saves one byte;
    # pick the length so that byte is a whole token.
    for k in range(3, 12):
        description = "a" * k + "😀"
        if roster_cost(prompts, "a" * k + "…", "") < roster_cost(prompts, description, ""):
            break
    else:
        pytest.fail("no suitable length found")
    budget = roster_cost(prompts, "a" * k + "…", "")
    transport = FakeTransport(reply(ROSTER_REPLY))
    build(prompts, transport, budget=budget).derive_roster(roster_meta(description), "")
    assert user_description(transport.requests[0]) == "a" * k + "…"
