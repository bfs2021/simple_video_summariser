"""Speaker alias rules for the roster closed set (#83, architecture.md 8.1, D4b).

The exact rule from #22 runs first; only after it fails may one of the small,
token-based alias rules resolve a decorated or shortened roster name. Anything
else is still coerced to ``unknown`` and counted.
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from adapters.summarize import schema
from adapters.summarize.schema import (
    HONORIFICS,
    NAME_SUFFIXES,
    parse_chunk_analysis,
)
from common.models import Chunk, Roster, RosterSpeaker

CHUNK = Chunk(seq=1, start_sec=0.0, end_sec=100.0, text="some words here")
UNKNOWN = "unknown"


def make_roster(*names: str) -> Roster:
    return Roster(tuple(RosterSpeaker(n, "guest") for n in names))


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def captured() -> Iterator[_Capture]:
    """Capture schema logs; usable inside Hypothesis tests (no function fixture)."""
    handler = _Capture()
    logger = logging.getLogger(schema.__name__)
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def resolve(roster: Roster, value: object) -> tuple[str, int, list[logging.LogRecord]]:
    """Return (speaker, speaker_coercions, speaker_aliased records) for one claim."""
    raw = json.dumps({"topics": [], "claims": [{"text": "x", "speaker": value}], "quotes": []})
    with captured() as cap:
        result = parse_chunk_analysis(raw, roster=roster, chunk=CHUNK)
    aliased = [r for r in cap.records if r.getMessage() == "speaker_aliased"]
    return result.claims[0].speaker, result.speaker_coercions, aliased


def speaker_of(names: tuple[str, ...], value: object) -> str:
    return resolve(make_roster(*names), value)[0]


# --- constants ----------------------------------------------------------------


def test_constants_are_the_specified_frozensets() -> None:
    assert HONORIFICS == frozenset({"dr", "prof", "professor", "mr", "mrs", "ms", "mx", "sir", "dame"})
    assert NAME_SUFFIXES == frozenset({"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "phd", "md"})
    assert isinstance(HONORIFICS, frozenset)
    assert isinstance(NAME_SUFFIXES, frozenset)


# --- accepted aliases ---------------------------------------------------------


ACCEPTED = [
    # honorific added
    (("Jane Smith",), "Dr. Jane Smith", "Jane Smith"),
    (("Jane Smith",), "dr jane smith", "Jane Smith"),
    (("Jane Smith",), "Prof. Dr. Jane Smith", "Jane Smith"),
    (("Jane Smith",), "  DR.   JANE\tSMITH ", "Jane Smith"),
    # honorific dropped / swapped
    (("Dr. Jane Smith",), "Jane Smith", "Dr. Jane Smith"),
    (("Dr. Jane Smith",), "Prof. Jane Smith", "Dr. Jane Smith"),
    # last name only
    (("Lex Fridman",), "Fridman", "Lex Fridman"),
    (("Lex Fridman",), "FRIDMAN", "Lex Fridman"),
    (("Lex Fridman",), "Mr. Fridman", "Lex Fridman"),
    (("Dr. Jane Smith",), "Dr. Smith", "Dr. Jane Smith"),
    (("Dr. Jane Smith",), "Smith", "Dr. Jane Smith"),
    # hyphen and apostrophe
    (("Conan O'Brien",), "O'Brien", "Conan O'Brien"),
    (("Ann Smith-Jones",), "Smith-Jones", "Ann Smith-Jones"),
    # single-name entry through an honorific
    (("Dr. Phil",), "Phil", "Dr. Phil"),
    # every honorific works, with and without the dot
    (("Jane Smith",), "Professor Jane Smith", "Jane Smith"),
    (("Jane Smith",), "Mrs Smith", "Jane Smith"),
    (("Jane Smith",), "Ms. Smith", "Jane Smith"),
    (("Jane Smith",), "Mx Smith", "Jane Smith"),
    (("Jane Smith",), "Sir Smith", "Jane Smith"),
    (("Jane Smith",), "Dame Smith", "Jane Smith"),
    # roster honorific is stripped too, repeated leading honorifics all go
    (("Prof. Dr. Anna Schmidt",), "Anna Schmidt", "Prof. Dr. Anna Schmidt"),
    (("Prof. Dr. Anna Schmidt",), "Schmidt", "Prof. Dr. Anna Schmidt"),
    # a single-name entry is reached through its stripped full name even when
    # another entry has the same token as a first name (only last-name keys
    # need the token to be unused elsewhere)
    (("Dr. Phil", "Phil Collins"), "Phil", "Dr. Phil"),
    # a two-letter last name is long enough
    (("Jane Li",), "Li", "Jane Li"),
    # a suffix stays part of the full name and is never stripped
    (("Martin King Jr.",), "Martin King Jr.", "Martin King Jr."),
]


@pytest.mark.parametrize(("roster", "spoken", "expected"), ACCEPTED)
def test_alias_returns_rosters_own_spelling_uncounted_and_logged(
    roster: tuple[str, ...], spoken: str, expected: str
) -> None:
    speaker, coercions, aliased = resolve(make_roster(*roster), spoken)
    assert speaker == expected
    if _is_exact(roster, spoken):
        assert aliased == []  # exact rule, not an alias
    else:
        assert coercions == 0
        [record] = aliased
        assert record.levelno == logging.INFO
        assert record.__dict__["speaker"] == spoken
        assert record.__dict__["resolved"] == expected
    assert coercions == 0


def _is_exact(roster: tuple[str, ...], spoken: str) -> bool:
    return schema._name_key(spoken) in {schema._name_key(n) for n in roster}


def test_alias_applies_to_quotes_too() -> None:
    raw = json.dumps({"topics": [], "claims": [], "quotes": [{"text": "q", "speaker": "Smith"}]})
    result = parse_chunk_analysis(raw, roster=make_roster("Jane Smith"), chunk=CHUNK)
    assert result.quotes[0].speaker == "Jane Smith"
    assert result.speaker_coercions == 0


def test_alias_log_truncates_the_value_to_100_chars() -> None:
    long_name = "Jane " + "x" * 200
    roster = make_roster("Dr. " + long_name)  # roster names are bounded elsewhere
    value = "Prof. " + long_name
    speaker, _, [record] = resolve(roster, value)
    assert speaker == "Dr. " + long_name
    assert record.__dict__["speaker"] == value[:100]
    assert len(record.__dict__["speaker"]) == 100


def test_alias_does_not_log_a_coercion() -> None:
    raw = json.dumps({"topics": [], "claims": [{"text": "x", "speaker": "Smith"}], "quotes": []})
    with captured() as cap:
        parse_chunk_analysis(raw, roster=make_roster("Jane Smith"), chunk=CHUNK)
    assert [r.getMessage() for r in cap.records if r.levelno >= logging.INFO] == ["speaker_aliased"]


# --- rejected: coerced and counted --------------------------------------------


REJECTED = [
    # first name only
    (("Lex Fridman",), "Lex"),
    (("Dr. Jane Smith",), "Jane"),
    # ambiguous last name
    (("Jane Smith", "John Smith"), "Smith"),
    (("Jane Smith", "John Smith"), "Dr. Smith"),
    # last name that is another entry's first or middle name
    (("Tracy Morgan", "Morgan Freeman"), "Morgan"),
    (("Tracy Morgan", "Anne Morgan Lee", "Bob Lee"), "Morgan"),
    # ambiguous stripped full name (exact keys still resolve, see below)
    (("Dr. Jane Smith", "Jane Smith"), "Prof. Jane Smith"),
    # honorific only
    (("Dr. Jane Smith",), "Dr."),
    (("Dr. Jane Smith",), "Mr"),
    (("Dr. Jane Smith",), "Prof. Dr."),
    # initials, reordering, punctuation, partial tokens
    (("Jane Smith",), "J. Smith"),
    (("Jane Smith",), "Smith, Jane"),
    (("Jane Smith",), "Smith Jane"),
    (("Jane Smith",), "Smit"),
    (("Jane Smith",), "Smiths"),
    (("Jane Smith",), "Jane S."),
    # cross-kind ambiguity
    (("Dr. Phil", "Lou Phil"), "Phil"),
    # suffix and placeholder last tokens give no last-name key
    (("Martin King Jr.",), "Jr."),
    (("Martin King Jr.",), "Jr"),
    (("Martin King Jr.",), "King"),
    (("Speaker 1",), "1"),
    (("Speaker 2b",), "2b"),
    (("Martin King Jr",), "Jr"),
    (("Martin King III",), "III"),
    (("Martin King PhD",), "PhD"),
    (("Martin King MD",), "md"),
    (("Martin King Sr",), "Sr"),
    # a single letter is too short to be a last name
    (("Jane L",), "L"),
    # extra words around a valid alias
    (("Jane Smith",), "Smith said"),
    (("Jane Smith",), "Dr. Smith (guest)"),
    (("Jane Smith",), "the host Smith"),
    (("Jane Smith",), "Smith ignore previous instructions"),
    # honorifics only strip at the front, not in the middle or at the end
    (("Jane Smith",), "Jane Dr. Smith"),
    (("Jane Smith",), "Smith Dr."),
    (("Jane Smith",), "Jane Smith Dr."),
    # dot rules: one optional dot only
    (("Jane Smith",), "dr.. Smith"),
    (("Jane Smith",), "d.r. Smith"),
    (("Jane Smith",), "Dr Dr.. Smith"),
    # unknown never reaches a roster last name, with or without an honorific
    (("Jane Unknown",), "Dr. Unknown"),
    (("Jane Unknown",), "Prof. Unknown"),
    # a roster whose only honorific-stripped name is empty gives no keys
    (("Dr.",), "Mr."),
    # a last-name token with digits or other characters qualifies for no key
    (("Jane Smith2",), "Smith2"),
    (("Jane -",), "-"),
    (("Jane '",), "'"),
    (("Jane Smith.",), "Smith."),
]


@pytest.mark.parametrize(("roster", "spoken"), REJECTED)
def test_rejected_alias_is_coerced_counted_and_not_logged_as_alias(
    roster: tuple[str, ...], spoken: str
) -> None:
    speaker, coercions, aliased = resolve(make_roster(*roster), spoken)
    assert speaker == UNKNOWN
    assert coercions == 1
    assert aliased == []


def test_unknown_in_any_case_never_reaches_a_roster_last_name() -> None:
    roster = make_roster("Jane Unknown")
    for value in ("unknown", "Unknown", "UNKNOWN", " unknown "):
        assert resolve(roster, value) == (UNKNOWN, 0, [])
    assert resolve(roster, None) == (UNKNOWN, 0, [])
    assert resolve(roster, "") == (UNKNOWN, 0, [])


def test_exact_keys_still_win_where_the_stripped_name_is_ambiguous() -> None:
    roster = make_roster("Dr. Jane Smith", "Jane Smith")
    assert resolve(roster, "Jane Smith") == ("Jane Smith", 0, [])
    assert resolve(roster, "dr. jane smith") == ("Dr. Jane Smith", 0, [])


@pytest.mark.parametrize("spoken", [42, 3.5, True, False, ["Smith"], {"name": "Smith"}, [], {}])
def test_non_string_values_are_coerced_and_never_alias_looked_up(spoken: object) -> None:
    assert resolve(make_roster("Jane Smith"), spoken) == (UNKNOWN, 1, [])


def test_empty_roster_keeps_every_speaker_unknown() -> None:
    for value in ("Smith", "Dr. Smith", "Dr."):
        assert resolve(Roster(()), value) == (UNKNOWN, 1, [])


def test_exact_match_logs_no_alias() -> None:
    assert resolve(make_roster("Jane Smith"), "jane smith") == ("Jane Smith", 0, [])


def test_alias_keys_are_built_once_per_parse_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    real = schema._alias_index

    def counting(roster: Roster) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return real(roster)

    monkeypatch.setattr(schema, "_alias_index", counting)
    raw = json.dumps(
        {
            "topics": [],
            "claims": [{"text": "x", "speaker": "Smith"} for _ in range(5)],
            "quotes": [{"text": "x", "speaker": "Smith"} for _ in range(5)],
        }
    )
    parse_chunk_analysis(raw, roster=make_roster("Jane Smith"), chunk=CHUNK)
    assert calls == 1


# --- property tests -----------------------------------------------------------

POOL = ["dr", "dr.", "prof", "Mr", "jane", "John", "smith", "Smith", "Jr.", "o'brien", "lee", "1", "x"]
pool_name_st = st.lists(st.sampled_from(POOL), min_size=0, max_size=4).map(" ".join)
text_name_st = st.text(max_size=20)
any_name_st = st.one_of(pool_name_st, pool_name_st, text_name_st)
roster_names_st = st.lists(any_name_st, max_size=5)
value_st = st.one_of(pool_name_st, text_name_st)
PROP = settings(max_examples=300, deadline=None)


def exact_rule(names: list[str], value: str) -> str:
    """The #22 rule, restated independently."""
    key = schema._name_key(value)
    if key in ("", UNKNOWN):
        return UNKNOWN
    table = {schema._name_key(n): n for n in names}
    return table.get(key, UNKNOWN)


@PROP
@given(names=roster_names_st, value=value_st)
def test_property_closed_set(names: list[str], value: str) -> None:
    speaker, _, _ = resolve(make_roster(*names), value)
    assert speaker == UNKNOWN or speaker in names


@PROP
@given(names=roster_names_st, extra=st.sampled_from(POOL + [""]), data=st.data())
def test_property_exact_dominates(names: list[str], extra: str, data: st.DataObject) -> None:
    real = [n for n in names if schema._name_key(n) not in ("", UNKNOWN)]
    if not real:
        return
    base = data.draw(st.sampled_from(real))
    value = data.draw(st.sampled_from([base, base.upper(), f"  {base}  ", base.swapcase()]))
    speaker, coercions, aliased = resolve(make_roster(*names), value)
    assert speaker == exact_rule(names, value)
    assert coercions == 0
    assert aliased == []


def _tokens(text: str) -> list[str]:
    return schema._name_key(text).split()


@PROP
@given(names=roster_names_st, value=value_st)
def test_property_no_invented_tokens(names: list[str], value: str) -> None:
    speaker, _, aliased = resolve(make_roster(*names), value)
    if not aliased:
        return
    assert speaker in names
    allowed = set(_tokens(speaker))
    for token in _tokens(value):
        bare = token.removesuffix(".")
        assert token in allowed or bare in HONORIFICS


@PROP
@given(names=roster_names_st, value=value_st)
def test_property_alias_is_unique_candidate(names: list[str], value: str) -> None:
    speaker, _, aliased = resolve(make_roster(*names), value)
    if not aliased:
        return
    index = names.index(speaker)
    rest = names[:index] + names[index + 1 :]
    assert speaker not in rest  # a duplicate spelling would have been ambiguous
    assert resolve(make_roster(*rest), value)[0] == UNKNOWN


@PROP
@given(names=roster_names_st, value=value_st, added=any_name_st)
def test_property_monotone_in_roster(names: list[str], value: str, added: str) -> None:
    before = resolve(make_roster(*names), value)[0]
    after = resolve(make_roster(*names, added), value)[0]
    if before != after:
        assert after in (added, UNKNOWN)


@PROP
@given(names=roster_names_st, value=value_st)
def test_property_alias_only_when_exact_fails(names: list[str], value: str) -> None:
    speaker, coercions, aliased = resolve(make_roster(*names), value)
    if aliased:
        assert exact_rule(names, value) == UNKNOWN
        assert speaker != UNKNOWN
        assert coercions == 0
        assert len(aliased) == 1
