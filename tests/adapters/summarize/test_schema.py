"""Tests for adapters/summarize/schema.py: untrusted LLM output in, typed values out."""

import json
import logging
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from adapters.summarize import schema
from adapters.summarize.schema import (
    MAX_RAW_CHARS,
    MAX_SPEAKERS,
    NONTRIVIAL_CHUNK_WORDS,
    SchemaError,
    parse_chunk_analysis,
    parse_roster,
    parse_tldr,
    parse_with_repair,
)
from common.errors import LLMInvalidOutputError, LLMUnavailableError
from common.models import (
    Chunk,
    ChunkAnalysis,
    Claim,
    Quote,
    Roster,
    RosterSpeaker,
    Topic,
)

FIXTURES = Path(__file__).parent / "fixtures"

ROSTER = Roster(
    (
        RosterSpeaker("Lex Fridman", "host"),
        RosterSpeaker("Guest One", "guest"),
    )
)
CHUNK = Chunk(seq=3, start_sec=100.0, end_sec=200.0, text="some words here")
EMPTY_ANALYSIS = '{"topics": [], "claims": [], "quotes": []}'


def analysis(**overrides: object) -> str:
    body: dict[str, object] = {"topics": [], "claims": [], "quotes": []}
    body.update(overrides)
    return json.dumps(body)


def claim_analysis(**claim: object) -> str:
    return analysis(claims=[{"text": "x", **claim}])


def parse(raw: str, roster: Roster = ROSTER, chunk: Chunk = CHUNK) -> ChunkAnalysis:
    return parse_chunk_analysis(raw, roster=roster, chunk=chunk)


# --- types ------------------------------------------------------------------


def test_roster_to_json_shape() -> None:
    roster = Roster((RosterSpeaker("A", "host"), RosterSpeaker("B", "unknown")))
    assert roster.to_json() == {
        "speakers": [{"name": "A", "role": "host"}, {"name": "B", "role": "unknown"}]
    }
    assert Roster(()).to_json() == {"speakers": []}


def test_public_returns_are_plain_dataclasses_not_pydantic() -> None:
    result = parse(EMPTY_ANALYSIS)
    for value in (result, parse_roster('{"speakers": []}'), *result.topics):
        assert not hasattr(value, "model_dump")
        assert type(value).__module__ == "common.models"


# --- JSON extraction --------------------------------------------------------

BARE = '{"speakers": [{"name": "Ann", "role": "host"}]}'
EXPECTED = Roster((RosterSpeaker("Ann", "host"),))


@pytest.mark.parametrize(
    "raw",
    [
        BARE,
        f"```json\n{BARE}\n```",
        f"```JSON\n{BARE}\n```",
        f"```\n{BARE}\n```",
        f"Here you go:\n{BARE}\nHope that helps!",
        f"﻿{BARE}",
        f"  \n\t{BARE}\n\n ",
        f"```json\n{BARE}",
        f"```json\n{BARE}\n``",
    ],
    ids=[
        "bare",
        "json_fence",
        "JSON_fence",
        "bare_fence",
        "prose_around",
        "bom",
        "whitespace",
        "unclosed_fence",
        "half_closed_fence",
    ],
)
def test_extraction_forms_give_same_result_as_bare_json(raw: str) -> None:
    assert parse_roster(raw) == EXPECTED


def test_braces_and_backticks_inside_strings_do_not_confuse_extraction() -> None:
    raw = analysis(
        quotes=[
            {"text": "a { b", "speaker": "Guest One"},
            {"text": "``` and } and {\"x\": 1}", "speaker": "Guest One"},
        ]
    )
    result = parse(f"```json\n{raw}\n```")
    assert [q.text for q in result.quotes] == ["a { b", '``` and } and {"x": 1}']


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   \n\t ",
        "I could not do that.",
        "[]",
        '[{"speakers": []}]',
        "```json\n[{\"speakers\": []}]\n```",
        "42",
        '"speakers"',
        "null",
        '{"speakers": [',
        '{"speakers": [}',
        '{"speakers": [], "x": NaN}',
        '{"speakers": [], "x": Infinity}',
        '{"speakers": [], "x": -Infinity}',
        '{"a": {"b": 1}, "c": NaN}',
        "[" * 100_000,
        '{"a":' * 100_000,
        "{" * 100_000,
    ],
    ids=[
        "empty",
        "whitespace_only",
        "no_json",
        "empty_array",
        "array_of_object",
        "fenced_array",
        "scalar_number",
        "scalar_string",
        "scalar_null",
        "truncated",
        "invalid",
        "nan",
        "infinity",
        "neg_infinity",
        "nan_after_valid_inner_object",
        "deep_array",
        "deep_object",
        "many_open_braces",
    ],
)
def test_unextractable_output_raises_schema_error(raw: str) -> None:
    with pytest.raises(SchemaError):
        parse_roster(raw)


def test_input_over_max_raw_chars_is_rejected_before_parsing() -> None:
    padding = " " * (MAX_RAW_CHARS - len(BARE))
    assert parse_roster(BARE + padding) == EXPECTED  # exactly at the limit
    with pytest.raises(SchemaError):
        parse_roster(BARE + padding + " ")  # one past


def test_max_raw_chars_is_256_kib() -> None:
    assert MAX_RAW_CHARS == 256 * 1024


def test_two_top_level_objects_are_ambiguous() -> None:
    with pytest.raises(SchemaError, match="(?i)ambiguous|more than one|two"):
        parse_roster((FIXTURES / "echoed_fragment_then_answer.txt").read_text())
    with pytest.raises(SchemaError):
        parse_roster(f"{BARE}\n{BARE}")


def test_trailing_non_json_braces_are_not_a_second_object() -> None:
    assert parse_roster(f"{BARE}\nNote: {{this is not json}}") == EXPECTED


def test_extraction_errors_are_schema_errors_for_every_parser() -> None:
    with pytest.raises(SchemaError):
        parse("no json here")


# --- roster -----------------------------------------------------------------


def test_roster_missing_or_non_list_speakers_raises() -> None:
    for raw in ('{}', '{"speakers": null}', '{"speakers": "Ann"}', '{"speakers": {"name": "A"}}'):
        with pytest.raises(SchemaError):
            parse_roster(raw)


def test_empty_roster_is_valid() -> None:
    assert parse_roster('{"speakers": []}') == Roster(())


def test_roster_names_are_trimmed_and_whitespace_collapsed() -> None:
    raw = json.dumps({"speakers": [{"name": "  Lex \t\n  Fridman  ", "role": "host"}]})
    assert parse_roster(raw).speakers[0].name == "Lex Fridman"


def test_roster_drops_empty_overlong_and_unknown_names() -> None:
    ok = "x" * 100
    too_long = "x" * 101
    entries = [
        {"name": ""},
        {"name": "   "},
        {"name": too_long},
        {"name": "unknown"},
        {"name": "UnKnown"},
        {"name": " Unknown "},
        {"name": ok},
        {"role": "host"},
        {"name": 7},
        "Ann",
        None,
    ]
    roster = parse_roster(json.dumps({"speakers": entries}))
    assert [s.name for s in roster.speakers] == [ok]


def test_roster_duplicates_keep_first_occurrence_case_insensitively() -> None:
    raw = json.dumps(
        {
            "speakers": [
                {"name": "Ann Lee", "role": "host"},
                {"name": "ann   LEE", "role": "guest"},
                {"name": "Bob", "role": "guest"},
                {"name": "ANN LEE"},
            ]
        }
    )
    assert parse_roster(raw) == Roster(
        (RosterSpeaker("Ann Lee", "host"), RosterSpeaker("Bob", "guest"))
    )


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("host", "host"),
        ("GUEST", "guest"),
        ("Panelist", "panelist"),
        ("unknown", "unknown"),
        ("moderator", "unknown"),
        ("", "unknown"),
        (None, "unknown"),
        (5, "unknown"),
        (["host"], "unknown"),
    ],
)
def test_roster_role_is_matched_case_insensitively_else_unknown(
    role: object, expected: str
) -> None:
    raw = json.dumps({"speakers": [{"name": "Ann", "role": role}]})
    assert parse_roster(raw).speakers[0].role == expected


def test_roster_missing_role_is_unknown() -> None:
    assert parse_roster('{"speakers": [{"name": "Ann"}]}').speakers[0].role == "unknown"


def test_roster_is_capped_in_order_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    def names(n: int) -> str:
        return json.dumps({"speakers": [{"name": f"S{i}"} for i in range(n)]})

    with caplog.at_level(logging.WARNING, logger=schema.__name__):
        exact = parse_roster(names(MAX_SPEAKERS))
    assert len(exact.speakers) == MAX_SPEAKERS
    assert not caplog.records

    with caplog.at_level(logging.WARNING, logger=schema.__name__):
        over = parse_roster(names(MAX_SPEAKERS + 1))
    assert [s.name for s in over.speakers] == [f"S{i}" for i in range(MAX_SPEAKERS)]
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_max_speakers_is_20() -> None:
    assert MAX_SPEAKERS == 20


# --- chunk analysis: structure ---------------------------------------------


@pytest.mark.parametrize("missing", ["topics", "claims", "quotes"])
def test_missing_list_key_raises(missing: str) -> None:
    body = json.loads(EMPTY_ANALYSIS)
    del body[missing]
    with pytest.raises(SchemaError, match=missing):
        parse(json.dumps(body))


@pytest.mark.parametrize("key", ["topics", "claims", "quotes"])
@pytest.mark.parametrize("value", [None, "text", {"a": 1}, 3])
def test_non_list_value_raises(key: str, value: object) -> None:
    with pytest.raises(SchemaError, match=key):
        parse(analysis(**{key: value}))


def test_empty_lists_and_extra_keys_are_fine() -> None:
    result = parse(analysis(extra={"a": 1}, notes="hi"))
    assert (result.topics, result.claims, result.quotes) == ((), (), ())
    assert (result.speaker_coercions, result.start_sec_clamped, result.items_dropped) == (0, 0, 0)


@pytest.mark.parametrize(
    "raw",
    [
        analysis(claims=["just a string"]),
        analysis(claims=[None]),
        analysis(claims=[["text"]]),
        analysis(claims=[{"speaker": "Guest One"}]),
        analysis(claims=[{"text": None}]),
        analysis(claims=[{"text": 12}]),
        analysis(quotes=[{"text": ["a"]}]),
        analysis(quotes=[{}]),
        analysis(topics=[{"summary": "no title"}]),
        analysis(topics=[{"title": 5}]),
        analysis(topics=["x"]),
    ],
)
def test_malformed_item_raises_and_names_the_location(raw: str) -> None:
    with pytest.raises(SchemaError) as exc:
        parse(raw)
    assert any(k in str(exc.value) for k in ("topics.0", "claims.0", "quotes.0"))


def test_blank_and_overlong_items_are_dropped_and_counted() -> None:
    long_text = "x" * 2001
    ok_text = "y" * 2000
    raw = analysis(
        topics=[
            {"title": "  "},
            {"title": "t" * 201},
            {"title": "t" * 200},
            {"title": "ok", "summary": "s" * 2001},
            {"title": "ok2", "summary": "s" * 2000},
        ],
        claims=[{"text": "\n "}, {"text": long_text}, {"text": ok_text}],
        quotes=[{"text": ""}, {"text": long_text}, {"text": "fine"}],
    )
    result = parse(raw)
    assert [t.title for t in result.topics] == ["t" * 200, "ok2"]
    assert [c.text for c in result.claims] == [ok_text]
    assert [q.text for q in result.quotes] == ["fine"]
    assert result.items_dropped == 3 + 2 + 2  # topics, claims, quotes


def test_items_dropped_counts_exact_numbers() -> None:
    raw = analysis(
        topics=[{"title": " "}, {"title": "a"}],
        claims=[{"text": ""}, {"text": "z" * 2001}, {"text": "kept"}],
        quotes=[{"text": "  "}],
    )
    result = parse(raw)
    assert result.items_dropped == 1 + 2 + 1


@pytest.mark.parametrize(
    ("key", "field", "cap"),
    [("topics", "title", 50), ("claims", "text", 200), ("quotes", "text", 200)],
)
def test_lists_are_truncated_in_order_and_tail_is_counted(key: str, field: str, cap: int) -> None:
    def build(n: int) -> str:
        return analysis(**{key: [{field: f"item {i}"} for i in range(n)]})

    exact = getattr(parse(build(cap)), key)
    assert len(exact) == cap
    assert parse(build(cap)).items_dropped == 0

    over = parse(build(cap + 3))
    items = getattr(over, key)
    assert [getattr(i, field) for i in items] == [f"item {i}" for i in range(cap)]
    assert over.items_dropped == 3


def test_malformed_items_past_the_cap_do_not_raise() -> None:
    items: list[object] = [{"text": f"c{i}"} for i in range(200)] + [None, 5]
    result = parse(analysis(claims=items))
    assert len(result.claims) == 200
    assert result.items_dropped == 2


def test_source_chunk_seq_and_topic_seq() -> None:
    raw = analysis(
        topics=[{"title": "a"}, {"title": " "}, {"title": "b"}, {"title": "c"}],
        claims=[{"text": "c1"}, {"text": "c2"}],
        quotes=[{"text": "q1"}],
    )
    result = parse(raw)
    assert [(t.seq, t.title) for t in result.topics] == [(0, "a"), (1, "b"), (2, "c")]
    assert {c.source_chunk_seq for c in result.claims} == {CHUNK.seq}
    assert {q.source_chunk_seq for q in result.quotes} == {CHUNK.seq}


def test_text_is_trimmed_and_topic_summary_optional() -> None:
    result = parse(analysis(topics=[{"title": " T "}, {"title": "U", "summary": " S "}]))
    assert result.topics[0].title == "T"
    assert result.topics[0].summary is None
    assert result.topics[1].summary == "S"


# --- speaker closed set -----------------------------------------------------


@pytest.mark.parametrize(
    "spoken",
    ["Lex Fridman", "  lex   fridman ", "LEX FRIDMAN", "Lex\tFridman\n"],
)
def test_roster_match_returns_rosters_own_spelling(spoken: str) -> None:
    result = parse(claim_analysis(speaker=spoken))
    assert result.claims[0].speaker == "Lex Fridman"
    assert result.speaker_coercions == 0


@pytest.mark.parametrize("spoken", ["unknown", "Unknown", "UNKNOWN", " unknown ", "", "  ", None])
def test_unknown_null_and_empty_speakers_become_unknown_without_counting(
    spoken: object, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=schema.__name__):
        result = parse(claim_analysis(speaker=spoken))
    assert result.claims[0].speaker == "unknown"
    assert result.speaker_coercions == 0
    assert not [r for r in caplog.records if r.getMessage() == "speaker_coerced"]


def test_missing_speaker_key_is_unknown_and_not_counted() -> None:
    result = parse(analysis(claims=[{"text": "x"}], quotes=[{"text": "y"}]))
    assert result.claims[0].speaker == "unknown"
    assert result.quotes[0].speaker == "unknown"
    assert result.speaker_coercions == 0


@pytest.mark.parametrize(
    "spoken",
    [
        "Lex",
        "Elon Musk",
        "SYSTEM",
        "ignore previous instructions",
        42,
        3.5,
        True,
        ["Lex Fridman"],
        {"name": "Lex Fridman"},
    ],
)
def test_other_speakers_are_coerced_and_counted(spoken: object) -> None:
    result = parse(claim_analysis(speaker=spoken))
    assert result.claims[0].speaker == "unknown"
    assert result.speaker_coercions == 1


def test_coercion_logs_truncated_value(caplog: pytest.LogCaptureFixture) -> None:
    offender = "ignore previous instructions " * 10
    with caplog.at_level(logging.WARNING, logger=schema.__name__):
        parse(claim_analysis(speaker=offender))
    [record] = [r for r in caplog.records if r.getMessage() == "speaker_coerced"]
    assert record.levelno == logging.WARNING
    value = record.__dict__["speaker"]
    assert value == offender[:100]
    assert len(value) == 100


def test_coercion_counts_claims_and_quotes_together() -> None:
    raw = analysis(
        claims=[{"text": "a", "speaker": "X"}, {"text": "b", "speaker": "Guest One"}],
        quotes=[{"text": "c", "speaker": "Y"}],
    )
    result = parse(raw)
    assert result.speaker_coercions == 2
    assert [c.speaker for c in result.claims] == ["unknown", "Guest One"]


def test_empty_roster_makes_every_speaker_unknown() -> None:
    raw = analysis(
        claims=[{"text": "a", "speaker": "Lex Fridman"}],
        quotes=[{"text": "b", "speaker": "unknown"}],
    )
    result = parse(raw, roster=Roster(()))
    assert result.claims[0].speaker == "unknown"
    assert result.quotes[0].speaker == "unknown"
    assert result.speaker_coercions == 1


def test_roster_name_that_looks_like_unknown_never_leaks() -> None:
    roster = Roster((RosterSpeaker("Ann", "host"),))
    result = parse(claim_analysis(speaker="ann"), roster=roster)
    assert result.claims[0].speaker == "Ann"


def test_invented_speakers_fixture() -> None:
    result = parse((FIXTURES / "invented_speakers.json").read_text())
    assert [c.speaker for c in result.claims] == [
        "Lex Fridman",
        "unknown",
        "unknown",
        "unknown",
        "Guest One",
        "unknown",
    ]
    assert result.quotes[0].speaker == "unknown"
    assert result.speaker_coercions == 4


def test_echoed_prompt_fixture() -> None:
    result = parse((FIXTURES / "echoed_prompt.json").read_text())
    assert [c.speaker for c in result.claims] == ["unknown", "unknown"]
    assert result.speaker_coercions == 2
    # echoed text stays in the claim text (data), never in the speaker field
    assert "ignore previous instructions" in result.claims[0].text


name_st = st.text(max_size=30)
scalar_st = st.one_of(
    name_st, st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False, allow_infinity=False)
)


@given(
    roster_names=st.lists(name_st, max_size=6),
    speakers=st.lists(st.one_of(scalar_st, st.lists(name_st, max_size=2)), max_size=8),
    echo_roster_case=st.booleans(),
)
def test_property_every_speaker_is_in_roster_or_unknown(
    roster_names: list[str], speakers: list[object], echo_roster_case: bool
) -> None:
    roster = Roster(tuple(RosterSpeaker(n, "guest") for n in roster_names))
    if echo_roster_case:
        speakers = speakers + [n.upper() for n in roster_names]
    raw = json.dumps(
        {
            "topics": [],
            "claims": [{"text": "c", "speaker": s} for s in speakers],
            "quotes": [{"text": "q", "speaker": s} for s in speakers],
        }
    )
    result = parse(raw, roster=roster)
    allowed = {n for n in roster_names} | {"unknown"}
    assert {c.speaker for c in result.claims} <= allowed
    assert {q.speaker for q in result.quotes} <= allowed


# --- timestamps and confidence ---------------------------------------------


@pytest.mark.parametrize("value", [None, "absent"])
def test_missing_or_null_start_sec_stays_none(value: object) -> None:
    item: dict[str, object] = {"text": "x"}
    if value != "absent":
        item["start_sec"] = value
    result = parse(analysis(claims=[item], quotes=[item], topics=[{"title": "t", **item}]))
    assert result.claims[0].start_sec is None
    assert result.quotes[0].start_sec is None
    assert result.topics[0].start_sec is None
    assert result.start_sec_clamped == 0


@pytest.mark.parametrize(
    ("value", "expected", "clamped"),
    [
        (100, 100.0, 0),
        (200, 200.0, 0),
        (100.0, 100.0, 0),
        ("200", 200.0, 0),
        (150.5, 150.5, 0),
        ("150.5", 150.5, 0),
        (99.999, 100.0, 1),
        (200.001, 200.0, 1),
        (0, 100.0, 1),
        (-1, 100.0, 1),
        (-1e9, 100.0, 1),
        (1e9, 200.0, 1),
        ("12.5", 100.0, 1),
        (" 9999 ", 200.0, 1),
    ],
)
def test_start_sec_is_clamped_into_chunk_and_counted(
    value: object, expected: float, clamped: int
) -> None:
    result = parse(claim_analysis(start_sec=value))
    assert result.claims[0].start_sec == expected
    assert result.start_sec_clamped == clamped


def test_start_sec_clamping_applies_to_topics_and_quotes() -> None:
    raw = analysis(
        topics=[{"title": "t", "start_sec": 0}],
        claims=[{"text": "c", "start_sec": 500}],
        quotes=[{"text": "q", "start_sec": -3}],
    )
    result = parse(raw)
    assert result.topics[0].start_sec == 100.0
    assert result.claims[0].start_sec == 200.0
    assert result.quotes[0].start_sec == 100.0
    assert result.start_sec_clamped == 3


@pytest.mark.parametrize(
    "value",
    [True, False, "soon", "", "12s", "nan", "inf", "-inf", [1], {"a": 1}, 1e999],
)
def test_boolean_or_non_numeric_start_sec_raises(value: object) -> None:
    raw = claim_analysis(start_sec=value).replace("Infinity", "1e999")
    with pytest.raises(SchemaError, match="start_sec"):
        parse(raw)


def test_huge_integer_start_sec_is_clamped_or_schema_error_never_other() -> None:
    raw = '{"topics": [], "claims": [{"text": "x", "start_sec": ' + "9" * 400 + '}], "quotes": []}'
    try:
        result = parse(raw)
    except SchemaError:
        return
    assert result.claims[0].start_sec == 200.0


def test_out_of_range_fixture() -> None:
    result = parse((FIXTURES / "out_of_range_timestamps.json").read_text())
    assert result.topics[0].start_sec == 100.0
    assert [c.start_sec for c in result.claims] == [100.0, 200.0, 150.0, 100.0, 200.0, None]
    assert result.start_sec_clamped == 3


@given(
    values=st.lists(
        st.one_of(
            st.none(),
            st.integers(min_value=-(10**12), max_value=10**12),
            st.floats(allow_nan=False, allow_infinity=False),
            st.floats(allow_nan=False, allow_infinity=False).map(repr),
        ),
        max_size=10,
    ),
    start=st.floats(min_value=0, max_value=1e6),
    length=st.floats(min_value=0, max_value=1e6),
)
def test_property_start_sec_is_none_or_inside_chunk(
    values: list[object], start: float, length: float
) -> None:
    chunk = Chunk(seq=0, start_sec=start, end_sec=start + length, text="w")
    raw = json.dumps(
        {
            "topics": [{"title": "t", "start_sec": v} for v in values],
            "claims": [{"text": "c", "start_sec": v} for v in values],
            "quotes": [{"text": "q", "start_sec": v} for v in values],
        }
    )
    result = parse(raw, chunk=chunk)
    items: list[Topic | Claim | Quote] = [*result.topics, *result.claims, *result.quotes]
    for item in items:
        assert item.start_sec is None or chunk.start_sec <= item.start_sec <= chunk.end_sec


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("high", "high"),
        ("HIGH", "high"),
        (" Medium ", "medium"),
        ("low", "low"),
        ("certain", None),
        ("", None),
        (None, None),
        (3, None),
        (["high"], None),
    ],
)
def test_confidence_is_matched_case_insensitively_else_none(
    value: object, expected: str | None
) -> None:
    assert parse(claim_analysis(confidence=value)).claims[0].confidence == expected


def test_missing_confidence_is_none() -> None:
    assert parse(claim_analysis()).claims[0].confidence is None


# --- empty-claims signal ----------------------------------------------------


def _chunk_of(words: int) -> Chunk:
    return Chunk(seq=1, start_sec=0.0, end_sec=60.0, text=" ".join(["w"] * words))


def test_empty_claims_warns_on_nontrivial_chunk(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=schema.__name__):
        result = parse(EMPTY_ANALYSIS, chunk=_chunk_of(NONTRIVIAL_CHUNK_WORDS))
    assert result.claims == ()
    assert [r for r in caplog.records if r.getMessage() == "empty_claims"]


def test_empty_claims_is_silent_on_short_chunk(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger=schema.__name__):
        parse(EMPTY_ANALYSIS, chunk=_chunk_of(NONTRIVIAL_CHUNK_WORDS - 1))
    assert not caplog.records


def test_claims_present_on_long_chunk_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger=schema.__name__):
        parse(claim_analysis(), chunk=_chunk_of(NONTRIVIAL_CHUNK_WORDS + 50))
    assert not caplog.records


def test_nontrivial_chunk_words_is_150() -> None:
    assert NONTRIVIAL_CHUNK_WORDS == 150


# --- fixtures: fenced / prose / truncated ----------------------------------


def test_fenced_prose_wrapped_fixture() -> None:
    result = parse((FIXTURES / "fenced_analysis.txt").read_text())
    assert [t.title for t in result.topics] == ["Intro"]
    assert result.claims[0].speaker == "Lex Fridman"
    assert result.claims[0].confidence == "high"
    assert result.quotes[0].text == "Use {braces} and ``` fences"


def test_truncated_fixture_raises_schema_error() -> None:
    with pytest.raises(SchemaError):
        parse((FIXTURES / "truncated_analysis.txt").read_text())


# --- tl;dr ------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "A summary.",
        "  A summary.\n\n",
        "```\nA summary.\n```",
        "```text\nA summary.\n```",
        "```markdown\nA summary.",
        "﻿A summary.",
    ],
)
def test_tldr_strips_fences_and_whitespace(raw: str) -> None:
    assert parse_tldr(raw) == "A summary."


@pytest.mark.parametrize("raw", ["", "   \n", "``````", "```\n\n```", "```json\n```"])
def test_tldr_empty_raises(raw: str) -> None:
    with pytest.raises(SchemaError):
        parse_tldr(raw)


def test_tldr_length_boundary() -> None:
    assert parse_tldr("a" * 2000) == "a" * 2000
    with pytest.raises(SchemaError):
        parse_tldr("a" * 2001)
    with pytest.raises(SchemaError):
        parse_tldr("a" * (MAX_RAW_CHARS + 1))


def test_tldr_keeps_inner_text_intact() -> None:
    assert parse_tldr("```\nLine one\n\nLine `two`\n```") == "Line one\n\nLine `two`"


# --- repair -----------------------------------------------------------------


def test_repair_not_called_when_first_parse_succeeds() -> None:
    calls: list[str] = []
    def repair(fb: str) -> str:
        calls.append(fb)
        return ""

    result = parse_with_repair(BARE, parse_roster, repair)
    assert result == EXPECTED
    assert calls == []


def test_repair_called_once_and_second_output_is_parsed() -> None:
    feedback: list[str] = []

    def repair(fb: str) -> str:
        feedback.append(fb)
        return BARE

    result = parse_with_repair('{"speakers": "oops"}', parse_roster, repair)
    assert result == EXPECTED
    assert len(feedback) == 1
    assert isinstance(feedback[0], str)
    assert "speakers" in feedback[0]
    assert len(feedback[0]) <= 2000


def test_repair_feedback_names_failing_field_locations() -> None:
    feedback: list[str] = []

    def repair(fb: str) -> str:
        feedback.append(fb)
        return EMPTY_ANALYSIS

    raw = analysis(claims=[{"text": "ok"}, {"speaker": "x"}], quotes=[{"text": 1}])
    parse_with_repair(raw, parse, repair)
    assert "claims.1.text" in feedback[0]
    assert "quotes.0.text" in feedback[0]


def test_repair_feedback_is_capped_at_2000_chars() -> None:
    feedback: list[str] = []

    def repair(fb: str) -> str:
        feedback.append(fb)
        return EMPTY_ANALYSIS

    raw = analysis(claims=[{"speaker": "x"}] * 200)
    parse_with_repair(raw, parse, repair)
    assert 0 < len(feedback[0]) <= 2000


def test_second_failure_raises_llm_invalid_output_chained() -> None:
    secret = "SECRET-RAW-" + "z" * 5000
    calls = 0

    def repair(fb: str) -> str:
        nonlocal calls
        calls += 1
        return '{"speakers": "' + secret + '"}'

    with pytest.raises(LLMInvalidOutputError) as exc:
        parse_with_repair("not json", parse_roster, repair)
    assert calls == 1
    assert isinstance(exc.value.__cause__, SchemaError)
    assert len(str(exc.value)) <= 2000
    assert secret not in str(exc.value)


def test_invalid_output_message_never_contains_full_raw_output() -> None:
    raw = "garbage " * 2000
    with pytest.raises(LLMInvalidOutputError) as exc:
        parse_with_repair(raw, parse_roster, lambda fb: raw)
    assert raw not in str(exc.value)
    assert len(str(exc.value)) <= 2000


def test_repair_exception_propagates_unchanged() -> None:
    boom = LLMUnavailableError("down")

    def repair(fb: str) -> str:
        raise boom

    with pytest.raises(LLMUnavailableError) as exc:
        parse_with_repair("not json", parse_roster, repair)
    assert exc.value is boom


def test_non_schema_errors_from_parse_are_not_swallowed() -> None:
    def parse_bug(raw: str) -> str:
        raise KeyError("bug")

    with pytest.raises(KeyError):
        parse_with_repair("x", parse_bug, lambda fb: "y")


def test_parse_with_repair_works_for_tldr() -> None:
    assert parse_with_repair("", parse_tldr, lambda fb: "Fixed.") == "Fixed."
