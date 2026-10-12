"""The analyze handler (issue #30).

Pure logic (normalization, dedupe, sanitizing, opening text) runs under
``-m "not integration"``; every test that runs the handler against Postgres
carries the ``integration`` marker.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import psycopg
import pytest
from hypothesis import given
from hypothesis import strategies as st

from common.chunking import chunk_strategy
from common.errors import (
    BugError,
    Cancelled,
    LLMInvalidOutputError,
    LLMUnavailableError,
    RateLimitedError,
    TransientNetworkError,
)
from common.models import (
    Chunk,
    ChunkAnalysis,
    Claim,
    Quote,
    Roster,
    RosterSpeaker,
    Segment,
    Topic,
    VideoMeta,
)
from common.queue import PostgresQueue
from common.repo.analyses import latest_analysis
from common.repo.transcripts import get_chunks, save_chunks, save_transcript
from common.repo.videos import upsert_video
from common.worker import JobContext
from services.analyzer import handler as handler_module
from services.analyzer.handler import (
    dedupe,
    make_analyze_handler,
    normalize_key,
    opening_text,
    sanitize_chunk,
    timestamped_chunk,
)
from tests.services.analyzer.fakes import (
    BatchSummarizer,
    CheckpointingSummarizer,
    FakeClock,
    FakeSummarizer,
    make_job,
)
from tests.services.transcriber.fakes import make_ctx, make_settings

integration = pytest.mark.integration

VIDEO = "abcdefghijk"
STRATEGY = chunk_strategy(900, 60)
ALICE = RosterSpeaker(name="Alice", role="host")
BOB = RosterSpeaker(name="Bob", role="guest")
ROSTER = Roster((ALICE, BOB))

# Two chunks under 900/60: window 0 holds s0-s2 (0..860); window 1 holds the
# overlap segment s2 again, then s3 and s4 (850..1710).
S0 = Segment(0, 10, "Alpha opening.")
S1 = Segment(100, 110, "Beta middle.")
S2 = Segment(850, 860, "Gamma late.")
S3 = Segment(950, 960, "Delta second.")
S4 = Segment(1700, 1710, "Epsilon end.")
TWO_CHUNKS = (S0, S1, S2, S3, S4)


def claim(text: str, speaker: str = "unknown", start: float | None = None, **kw: Any) -> Claim:
    return Claim(text=text, speaker=speaker, start_sec=start, **kw)


def quote(text: str, speaker: str = "unknown", start: float | None = None, **kw: Any) -> Quote:
    return Quote(text=text, speaker=speaker, start_sec=start, **kw)


def analysis(
    topics: tuple[Topic, ...] = (),
    claims: tuple[Claim, ...] = (),
    quotes: tuple[Quote, ...] = (),
    **kw: Any,
) -> ChunkAnalysis:
    return ChunkAnalysis(topics=topics, claims=claims, quotes=quotes, **kw)


CHUNK = Chunk(seq=3, start_sec=100.0, end_sec=200.0, text="x")


# --------------------------------------------------------------- normalize_key


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("The GDP grew 3%.", "  the gdp  grew 3%"),
        ("ﬁne  print", "FINE print"),  # NFKC: the fi ligature
        ("ＡＢＣ", "abc"),  # NFKC: fullwidth letters
        ("Straße", "STRASSE"),  # casefold, not lower
        ("“Hello there!”", "hello there"),
        ("a\tb\nc", "a b c"),
    ],
)
def test_normalize_key_collides_for_equivalent_texts(a: str, b: str) -> None:
    assert normalize_key(a) == normalize_key(b)


def test_normalize_key_keeps_inner_punctuation_and_distinguishes_words() -> None:
    assert normalize_key("well, no") != normalize_key("well no")
    assert normalize_key("cats") != normalize_key("dogs")


@pytest.mark.parametrize("text", ["...", "  ", "— !? ", ""])
def test_normalize_key_is_empty_for_punctuation_and_whitespace_only(text: str) -> None:
    assert normalize_key(text) == ""


# ---------------------------------------------------------------------- dedupe


def test_dedupe_collapses_normalized_duplicates_and_keeps_original_text() -> None:
    kept = dedupe([claim("The GDP grew 3%.", "Alice", 5, source_chunk_seq=0),
                   claim("  the gdp  grew 3%", "Alice", 9, source_chunk_seq=1)])

    assert [c.text for c in kept] == ["The GDP grew 3%."]


def test_dedupe_prefers_a_named_speaker_over_unknown() -> None:
    kept = dedupe([
        claim("same", "unknown", 1, source_chunk_seq=0),
        claim("same", "Bob", 50, source_chunk_seq=1),
    ])

    assert [(c.speaker, c.source_chunk_seq) for c in kept] == [("Bob", 1)]


def test_dedupe_then_prefers_the_lowest_chunk_seq() -> None:
    kept = dedupe([
        claim("same", "Bob", 1, source_chunk_seq=2),
        claim("same", "Alice", 50, source_chunk_seq=1),
    ])

    assert [(c.speaker, c.source_chunk_seq) for c in kept] == [("Alice", 1)]


def test_dedupe_then_prefers_the_earliest_start_and_start_none_loses() -> None:
    kept = dedupe([
        claim("same", "Bob", None, source_chunk_seq=1),
        claim("same", "Bob", 90, source_chunk_seq=1),
        claim("same", "Bob", 40, source_chunk_seq=1),
    ])

    assert [c.start_sec for c in kept] == [40]


def test_dedupe_drops_items_whose_normalized_text_is_empty() -> None:
    kept = dedupe([claim("..."), claim("real claim")])

    assert [c.text for c in kept] == ["real claim"]


def test_dedupe_treats_a_claim_and_a_quote_with_the_same_text_separately() -> None:
    claims = dedupe([claim("same text", "Alice")])
    quotes = dedupe([quote("same text", "Alice")])

    assert len(claims) == 1
    assert len(quotes) == 1


def test_dedupe_of_nothing_is_nothing() -> None:
    assert dedupe([]) == []


_texts = st.sampled_from(
    ["a", "A.", " a ", "b", "B!", "b  b", "B B", "...", "", "ﬁ", "fi", "x y", "x\ty"]
)
_items = st.builds(
    Claim,
    text=_texts,
    speaker=st.sampled_from(["unknown", "Alice", "Bob"]),
    start_sec=st.one_of(st.none(), st.floats(0, 1000, allow_nan=False)),
    source_chunk_seq=st.integers(0, 5),
)


@given(st.lists(_items, max_size=30))
def test_dedupe_output_is_a_subset_with_unique_nonempty_keys_and_is_idempotent(
    items: list[Claim],
) -> None:
    once = dedupe(items)

    assert all(item in items for item in once)
    keys = [normalize_key(item.text) for item in once]
    assert len(keys) == len(set(keys))
    assert "" not in keys
    assert dedupe(once) == once


# -------------------------------------------------------------------- sanitize


def test_sanitize_coerces_an_invented_speaker_and_counts_it() -> None:
    result = analysis(
        claims=(claim("c1", "Alice"), claim("c2", "Mallory"), claim("c3", "unknown")),
        quotes=(quote("q1", "Mallory"), quote("q2", "alice")),
    )

    _, claims, quotes, coerced = sanitize_chunk(CHUNK, result, ROSTER)

    assert [c.speaker for c in claims] == ["Alice", "unknown", "unknown"]
    assert [q.speaker for q in quotes] == ["unknown", "unknown"]  # exact match only
    assert coerced == 3


def test_sanitize_adds_the_adapters_own_coercion_count() -> None:
    result = analysis(claims=(claim("c", "Mallory"),), speaker_coercions=2)

    assert sanitize_chunk(CHUNK, result, ROSTER)[3] == 3


@pytest.mark.parametrize(
    ("given_start", "expected"),
    [
        (150.0, 150.0),
        (100.0, 100.0),
        (200.0, 200.0),
        (99.0, 100.0),
        (-5.0, 100.0),
        (201.0, 200.0),
        (1e9, 200.0),
        (None, None),
        (math.nan, None),
        (math.inf, None),
        (-math.inf, None),
    ],
)
def test_sanitize_clamps_start_sec_into_the_chunk_span(
    given_start: float | None, expected: float | None
) -> None:
    result = analysis(claims=(claim("c", start=given_start),), quotes=(quote("q", start=given_start),))

    _, claims, quotes, _ = sanitize_chunk(CHUNK, result, ROSTER)

    assert claims[0].start_sec == expected
    assert quotes[0].start_sec == expected


@pytest.mark.parametrize(
    ("given", "expected"),
    [("high", "high"), ("medium", "medium"), ("low", "low"), ("HIGH", None), ("sure", None), ("", None), (None, None)],
)
def test_sanitize_nulls_a_confidence_outside_high_medium_low(
    given: str | None, expected: str | None
) -> None:
    result = analysis(claims=(claim("c", confidence=given),))

    assert sanitize_chunk(CHUNK, result, ROSTER)[1][0].confidence == expected


def test_sanitize_stamps_the_source_chunk_seq() -> None:
    result = analysis(claims=(claim("c", source_chunk_seq=99),), quotes=(quote("q"),))

    _, claims, quotes, _ = sanitize_chunk(CHUNK, result, ROSTER)

    assert claims[0].source_chunk_seq == 3
    assert quotes[0].source_chunk_seq == 3


def test_sanitize_keeps_topics_in_the_order_the_model_emitted_them() -> None:
    result = analysis(topics=(Topic(0, "B", start_sec=120.0), Topic(1, "A", start_sec=110.0)))

    topics, _, _, _ = sanitize_chunk(CHUNK, result, ROSTER)

    assert [t.title for t in topics] == ["B", "A"]


# ----------------------------------------------------------- opening_text etc.


def test_opening_text_takes_segments_starting_within_600_seconds_of_the_first() -> None:
    segments = [
        Segment(0, 5, "one"),
        Segment(300, 305, "two"),
        Segment(599.9, 601, "three"),
        Segment(600, 605, "four"),
        Segment(900, 905, "five"),
    ]

    assert opening_text(segments) == "one two three"


def test_opening_text_of_a_short_transcript_is_all_of_it() -> None:
    assert opening_text([Segment(0, 1, "  a "), Segment(2, 3, "b")]) == "a b"


def test_opening_text_is_relative_to_when_speech_starts() -> None:
    segments = [Segment(700, 705, "late"), Segment(1000, 1005, "later"), Segment(1400, 1401, "far")]

    assert opening_text(segments) == "late later"


def test_opening_text_of_nothing_is_empty() -> None:
    assert opening_text([]) == ""


def test_timestamped_chunk_formats_overlapping_segments_and_keeps_the_span() -> None:
    chunk = Chunk(seq=1, start_sec=850.0, end_sec=1710.0, text="plain", chunk_strategy=STRATEGY)

    got = timestamped_chunk(chunk, TWO_CHUNKS)

    assert got == replace(chunk, text="[850] Gamma late.\n[950] Delta second.\n[1700] Epsilon end.")


def test_timestamped_chunk_excludes_segments_outside_the_half_open_span() -> None:
    chunk = Chunk(seq=0, start_sec=100.0, end_sec=850.0, text="plain")

    got = timestamped_chunk(chunk, TWO_CHUNKS)

    assert got.text == "[100] Beta middle."  # S0 ends before, S2 starts at the end


# ------------------------------------------------------------ integration rig


class Rig:
    """A handler wired to a real Postgres, with readers for what it wrote."""

    def __init__(
        self,
        conn: psycopg.Connection[Any],
        dsn: str,
        summarizer: FakeSummarizer,
        *,
        segments: dict[str, tuple[Segment, ...]] | None = None,
        clock: Callable[[], float] | None = None,
        connect_cls: type[psycopg.Connection[Any]] | None = None,
        **settings: Any,
    ) -> None:
        self.conn = conn
        self.dsn = dsn
        self.summarizer = summarizer
        self.settings = make_settings(**settings)
        self.transcripts: dict[str, int] = {}
        self.meta = VideoMeta(
            video_id=VIDEO,
            channel_id="UC" + "a" * 22,
            title="The title",
            description="The description",
            duration_sec=2000,
            published_at=None,
            language=None,
            live_status=None,
            manual_subtitle_langs=(),
            auto_caption_langs=(),
        )
        upsert_video(conn, self.meta, "adhoc")
        for source, segs in (segments if segments is not None else {"whisper": TWO_CHUNKS}).items():
            self.transcripts[source] = save_transcript(conn, VIDEO, source, "en", "none", segs, None)
        conn.commit()
        self.clock = clock or FakeClock()
        cls = connect_cls or psycopg.Connection
        self.handler = make_analyze_handler(
            connect=lambda: cls.connect(dsn, application_name="analyze-handler"),
            summarizer=summarizer,
            settings=self.settings,
            clock=self.clock,
        )
        self.ctx, self.logs = make_ctx()

    def run(self, video_id: str = VIDEO, **job: Any) -> None:
        self.handler(make_job(video_id, **job), self.ctx)

    def scalar(self, sql: str, *params: object) -> Any:
        row = self.conn.execute(sql, params).fetchone()
        self.conn.rollback()
        return row[0] if row else None

    def count(self, table: str) -> int:
        return int(self.scalar(f"SELECT count(*) FROM {table}"))

    def latest(self) -> Any:
        found = latest_analysis(self.conn, VIDEO)
        self.conn.rollback()
        return found

    def chunks(self, source: str = "whisper", strategy: str = STRATEGY) -> list[Chunk]:
        found = get_chunks(self.conn, self.transcripts[source], strategy)
        self.conn.rollback()
        return found

    def rows(self) -> tuple[int, int, int, int]:
        return (
            self.count("analyses"),
            self.count("topics"),
            self.count("claims"),
            self.count("quotes"),
        )


@pytest.fixture
def rig_factory(
    conn: psycopg.Connection[Any], head_dsn: str
) -> Callable[..., Rig]:
    def build(summarizer: FakeSummarizer | None = None, **kwargs: Any) -> Rig:
        return Rig(conn, head_dsn, summarizer or FakeSummarizer(), **kwargs)

    return build


# ----------------------------------------------------------- happy path, shape


@integration
def test_a_two_chunk_video_is_analyzed_and_persisted(rig_factory: Callable[..., Rig]) -> None:
    summarizer = FakeSummarizer(
        model="qwen-test",
        roster=ROSTER,
        chunks={
            0: analysis(
                topics=(Topic(0, "T0a"), Topic(1, "T0b")),
                claims=(claim("claim one", "Alice", 10),),
                quotes=(quote("quote one", "Bob", 20),),
            ),
            1: analysis(
                topics=(Topic(0, "T1a", "sum", 900.0),),
                claims=(claim("claim two", "Bob", 1000),),
            ),
        },
        tldr="  The whole story.  ",
    )
    rig = rig_factory(summarizer, clock=FakeClock(start=100.0, step=2.5))

    rig.run()

    got = rig.latest()
    assert got.tldr == "The whole story."
    assert (got.model, got.prompt_version, got.chunk_strategy) == ("qwen-test", "v1", STRATEGY)
    assert got.transcript_id == rig.transcripts["whisper"]
    assert got.speaker_roster == {
        "speakers": [{"name": "Alice", "role": "host"}, {"name": "Bob", "role": "guest"}]
    }
    assert [(t.seq, t.title) for t in got.topics] == [(0, "T0a"), (1, "T0b"), (2, "T1a")]
    assert sorted((c.text, c.speaker, c.source_chunk_seq) for c in got.claims) == [
        ("claim one", "Alice", 0),
        ("claim two", "Bob", 1),
    ]
    assert [(q.text, q.source_chunk_seq) for q in got.quotes] == [("quote one", 0)]
    assert got.cost_usd is None
    assert got.duration_ms == 2500  # two clock reads: start, just before persisting


@integration
def test_summarizer_calls_are_roster_then_chunks_in_order_then_one_reduce(
    rig_factory: Callable[..., Rig],
) -> None:
    summarizer = FakeSummarizer(
        roster=ROSTER,
        chunks={0: analysis(topics=(Topic(0, "A"),)), 1: analysis(topics=(Topic(0, "B"),))},
    )
    rig = rig_factory(summarizer)

    rig.run()

    assert summarizer.kinds == ["roster", "chunk", "chunk", "reduce"]
    _, meta, opening = summarizer.calls[0]
    assert (meta.title, meta.description, meta.duration_sec) == ("The title", "The description", 2000)
    assert opening == "Alpha opening. Beta middle."
    chunk0, roster0 = summarizer.calls[1][1:3]
    chunk1, roster1 = summarizer.calls[2][1:3]
    assert roster0 is roster1
    assert roster0 is summarizer.roster
    assert (chunk0.seq, chunk1.seq) == (0, 1)
    assert chunk0.text == "[0] Alpha opening.\n[100] Beta middle.\n[850] Gamma late."
    assert chunk1.text == "[850] Gamma late.\n[950] Delta second.\n[1700] Epsilon end."
    assert (chunk1.start_sec, chunk1.end_sec) == (850.0, 1710.0)
    partials = summarizer.calls[3][1]
    assert partials == [summarizer.chunks[0], summarizer.chunks[1]]


@integration
def test_a_single_chunk_video_still_calls_reduce_once_with_one_partial(
    rig_factory: Callable[..., Rig],
) -> None:
    summarizer = FakeSummarizer()
    rig = rig_factory(summarizer, segments={"whisper": (S0, S1)})

    rig.run()

    assert summarizer.kinds == ["roster", "chunk", "reduce"]
    assert len(summarizer.calls[2][1]) == 1


@integration
def test_a_dedupe_key_that_differs_from_the_configured_one_warns_and_still_runs(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = rig_factory(FakeSummarizer(name="ollama"))

    rig.run(dedupe_key="v0:ollama")

    warnings = rig.logs.records("warning")
    assert any("v0:ollama" in str(w) and "v1:ollama" in str(w) for w in warnings)
    assert rig.latest().prompt_version == "v1"


@integration
def test_a_matching_dedupe_key_logs_no_warning(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()

    rig.run(dedupe_key="v1:ollama")

    assert rig.logs.records("warning") == []


@integration
def test_a_configured_prompt_version_and_chunk_size_are_recorded(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = rig_factory(PROMPT_VERSION="v2", CHUNK_SEC=600, OVERLAP_SEC=30)

    rig.run(dedupe_key="v2:ollama")

    got = rig.latest()
    assert (got.prompt_version, got.chunk_strategy) == ("v2", "time:600:30")


# ------------------------------------------------------ transcript and chunks


@integration
def test_the_best_transcript_is_used_whisper_over_auto(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(segments={"youtube_auto": (S0,), "whisper": (S1,)})

    rig.run()

    assert rig.latest().transcript_id == rig.transcripts["whisper"]
    assert rig.summarizer.calls[0][2] == "Beta middle."


@integration
def test_a_missing_video_raises_a_bug_error_naming_it_and_writes_nothing(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = rig_factory()

    with pytest.raises(BugError, match="zzzzzzzzzzz"):
        rig.run("zzzzzzzzzzz")

    assert rig.summarizer.calls == []
    assert rig.rows() == (0, 0, 0, 0)


@integration
def test_a_video_without_a_transcript_raises_a_bug_error_naming_it(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = rig_factory(segments={})

    with pytest.raises(BugError, match=VIDEO):
        rig.run()

    assert rig.summarizer.calls == []
    assert rig.rows() == (0, 0, 0, 0)


@integration
def test_stored_chunks_are_used_as_stored_and_the_chunker_is_not_called(
    rig_factory: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = rig_factory()
    stored = [
        Chunk(seq=0, start_sec=0.0, end_sec=200.0, text="stored zero"),
        Chunk(seq=1, start_sec=850.0, end_sec=1710.0, text="stored one"),
    ]
    save_chunks(rig.conn, rig.transcripts["whisper"], STRATEGY, stored)
    rig.conn.commit()

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("the chunker must not run")

    monkeypatch.setattr(handler_module, "chunk_segments", boom)

    rig.run()

    seen = [(c[1].seq, c[1].start_sec, c[1].end_sec) for c in rig.summarizer.calls if c[0] == "chunk"]
    assert seen == [(0, 0.0, 200.0), (1, 850.0, 1710.0)]


@integration
def test_missing_chunks_are_generated_and_saved_leaving_other_strategies_alone(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = rig_factory()
    other = Chunk(seq=0, start_sec=0.0, end_sec=5.0, text="other strategy")
    save_chunks(rig.conn, rig.transcripts["whisper"], "time:300:30", [other])
    rig.conn.commit()

    rig.run()

    assert [(c.seq, c.start_sec, c.end_sec) for c in rig.chunks()] == [
        (0, 0.0, 860.0),
        (1, 850.0, 1710.0),
    ]
    assert [c.text for c in rig.chunks(strategy="time:300:30")] == ["other strategy"]


@integration
def test_generated_chunks_stay_committed_when_a_later_call_fails(
    rig_factory: Callable[..., Rig],
) -> None:
    summarizer = FakeSummarizer(fail_on={"chunk:0": LLMUnavailableError("down")})
    rig = rig_factory(summarizer)

    with pytest.raises(LLMUnavailableError):
        rig.run()

    assert len(rig.chunks()) == 2
    assert rig.rows() == (0, 0, 0, 0)


def _one_second_segments(count: int) -> tuple[Segment, ...]:
    return tuple(Segment(i, i + 1, f"w{i}") for i in range(count))


@integration
def test_a_transcript_exactly_chunk_sec_long_is_one_chunk(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(segments={"whisper": _one_second_segments(900)})

    rig.run()

    assert rig.summarizer.kinds.count("chunk") == 1


@integration
def test_a_transcript_one_second_longer_is_two_chunks(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(segments={"whisper": _one_second_segments(901)})

    rig.run()

    assert rig.summarizer.kinds.count("chunk") == 2


@integration
@pytest.mark.parametrize("segments", [(), (Segment(0, 1, "   "), Segment(1, 2, "\t"))])
def test_an_empty_transcript_persists_an_empty_analysis_without_llm_calls(
    rig_factory: Callable[..., Rig], segments: tuple[Segment, ...]
) -> None:
    rig = rig_factory(segments={"whisper": segments})

    rig.run()

    assert rig.summarizer.calls == []
    got = rig.latest()
    assert got.tldr == ""
    assert (got.topics, got.claims, got.quotes) == ((), (), ())
    assert (got.input_tokens, got.output_tokens) == (0, 0)
    assert got.speaker_roster == {"speakers": []}
    assert got.transcript_id == rig.transcripts["whisper"]
    assert got.speakers_coerced == 0  # the no-chunks path stores 0, not NULL (#60)


@integration
def test_the_coerced_speaker_total_is_stored_on_the_analysis(
    rig_factory: Callable[..., Rig],
) -> None:
    summarizer = FakeSummarizer(
        roster=ROSTER,
        chunks={
            0: analysis(
                claims=(claim("c", "Mallory"),),
                quotes=(quote("q", "Eve"),),
                speaker_coercions=1,
            ),
            1: analysis(),
        },
    )
    rig = rig_factory(summarizer)

    rig.run()

    assert rig.latest().speakers_coerced == 3  # same total the completion log line reports


@integration
def test_an_analysis_with_only_known_speakers_stores_zero_coercions(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = rig_factory(FakeSummarizer(roster=ROSTER, chunks={0: analysis(claims=(claim("c", "Alice"),))}))

    rig.run()

    assert rig.latest().speakers_coerced == 0


# ---------------------------------------------------------------- idempotency


@integration
def test_a_replayed_job_makes_no_llm_calls_writes_nothing_and_logs_info(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = rig_factory()
    rig.run()
    calls_before = len(rig.summarizer.calls)
    rows_before = rig.rows()

    rig.run()

    assert len(rig.summarizer.calls) == calls_before
    assert rig.rows() == rows_before
    assert any("already analyzed" in str(r) for r in rig.logs.records("info"))


@integration
def test_a_forced_job_runs_again_and_keeps_the_earlier_analysis(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = rig_factory()
    rig.run()

    rig.run(payload={"force": True})

    assert rig.count("analyses") == 2
    assert rig.summarizer.kinds.count("roster") == 2


@integration
def test_a_force_value_other_than_true_does_not_force(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    rig.run()

    rig.run(payload={"force": "yes"})

    assert rig.count("analyses") == 1


@integration
def test_a_different_model_is_not_a_replay(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(FakeSummarizer(model="model-a"))
    rig.run()
    rig.summarizer.model = "model-b"

    rig.run()

    assert rig.count("analyses") == 2


# --------------------------------------------------------------------- roster


@integration
def test_an_invalid_roster_falls_back_to_an_empty_one_with_a_warning(
    rig_factory: Callable[..., Rig],
) -> None:
    summarizer = FakeSummarizer(
        fail_on={"roster": LLMInvalidOutputError("bad json")},
        usage_on_fail=(9, 9),
        chunks={0: analysis(claims=(claim("c", "Alice"),))},
    )
    rig = rig_factory(summarizer, segments={"whisper": (S0, S1)})

    rig.run()

    assert rig.logs.records("warning")
    got = rig.latest()
    assert got.speaker_roster == {"speakers": []}
    assert [c.speaker for c in got.claims] == ["unknown"]
    assert summarizer.calls[1][2].speakers == ()
    # roster tokens (failed) are not counted; chunk 100/50 + reduce 20/10 are
    assert (got.input_tokens, got.output_tokens) == (120, 60)


@integration
@pytest.mark.parametrize(
    "error",
    [
        LLMUnavailableError("down"),
        RateLimitedError("slow down"),
        TransientNetworkError("blip"),
        RuntimeError("bug"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_any_other_roster_error_propagates_unchanged(
    rig_factory: Callable[..., Rig], error: BaseException
) -> None:
    summarizer = FakeSummarizer(fail_on={"roster": error})
    rig = rig_factory(summarizer)

    with pytest.raises(type(error)) as caught:
        rig.run()

    assert caught.value is error
    assert summarizer.kinds == ["roster"]
    assert rig.rows() == (0, 0, 0, 0)


# ----------------------------------------------------------------- map, reduce


@integration
@pytest.mark.parametrize(
    ("key", "error"),
    [
        ("chunk:0", LLMInvalidOutputError("bad")),
        ("chunk:1", LLMUnavailableError("down")),
        ("chunk:1", RuntimeError("bug")),
        ("reduce", LLMInvalidOutputError("bad")),
        ("reduce", RateLimitedError("slow")),
    ],
)
def test_a_chunk_or_reduce_error_propagates_and_nothing_is_written(
    rig_factory: Callable[..., Rig], key: str, error: BaseException
) -> None:
    rig = rig_factory(FakeSummarizer(roster=ROSTER, fail_on={key: error}))

    with pytest.raises(type(error)) as caught:
        rig.run()

    assert caught.value is error
    assert rig.rows() == (0, 0, 0, 0)


@integration
@pytest.mark.parametrize("tldr", ["", "   \n\t "])
def test_a_blank_tldr_is_an_invalid_output_error(
    rig_factory: Callable[..., Rig], tldr: str
) -> None:
    rig = rig_factory(FakeSummarizer(tldr=tldr))

    with pytest.raises(LLMInvalidOutputError):
        rig.run()

    assert rig.rows() == (0, 0, 0, 0)


@integration
def test_a_job_cancelled_mid_map_stops_calling_the_summarizer_and_writes_nothing(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = rig_factory()

    def cancel_during_first_chunk(kind: str) -> None:
        if kind == "chunk":
            rig.ctx._cancel()

    rig.summarizer.on_call = cancel_during_first_chunk

    with pytest.raises(Cancelled):
        rig.run()

    assert rig.summarizer.kinds == ["roster", "chunk"]
    assert rig.rows() == (0, 0, 0, 0)


@integration
def test_a_job_cancelled_before_the_first_call_makes_no_calls(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = rig_factory()
    rig.ctx._cancel()

    with pytest.raises(Cancelled):
        rig.run()

    assert rig.summarizer.calls == []


@integration
def test_cancelling_during_the_last_chunk_stops_before_reduce(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = rig_factory()

    def cancel_on_second_chunk(kind: str) -> None:
        if kind == "chunk" and rig.summarizer.kinds.count("chunk") == 2:
            rig.ctx._cancel()

    rig.summarizer.on_call = cancel_on_second_chunk

    with pytest.raises(Cancelled):
        rig.run()

    assert "reduce" not in rig.summarizer.kinds


# ---------------------------------------------------------- untrusted LLM output


@integration
def test_untrusted_output_is_coerced_clamped_and_stamped_before_persisting(
    rig_factory: Callable[..., Rig],
) -> None:
    summarizer = FakeSummarizer(
        roster=ROSTER,
        chunks={
            0: analysis(
                claims=(
                    claim("invented", "Mallory", 50.0, confidence="high"),
                    claim("early", "Alice", -100.0, confidence="certain"),
                    claim("late", "Alice", 99999.0),
                    claim("nan", "Alice", math.nan),
                ),
                quotes=(quote("q", "Mallory", 1e9),),
            ),
            1: analysis(),
        },
    )
    rig = rig_factory(summarizer)

    rig.run()

    got = rig.latest()
    by_text = {c.text: c for c in got.claims}
    assert by_text["invented"].speaker == "unknown"
    assert by_text["early"].start_sec == 0.0
    assert by_text["early"].confidence is None
    assert by_text["late"].start_sec == 860.0
    assert by_text["nan"].start_sec is None
    assert {c.source_chunk_seq for c in got.claims} == {0}
    assert (got.quotes[0].speaker, got.quotes[0].start_sec) == ("unknown", 860.0)


@integration
def test_topics_are_not_deduplicated_and_are_renumbered_across_chunks(
    rig_factory: Callable[..., Rig],
) -> None:
    summarizer = FakeSummarizer(
        chunks={
            0: analysis(topics=(Topic(0, "same"), Topic(1, "second"))),
            1: analysis(topics=(Topic(0, "same"), Topic(1, "last"))),
        },
    )
    rig = rig_factory(summarizer)

    rig.run()

    assert [(t.seq, t.title) for t in rig.latest().topics] == [
        (0, "same"),
        (1, "second"),
        (2, "same"),
        (3, "last"),
    ]


# ---------------------------------------------------------- dedupe via handler


@integration
def test_duplicates_across_the_seam_and_within_a_chunk_are_stored_once(
    rig_factory: Callable[..., Rig],
) -> None:
    summarizer = FakeSummarizer(
        roster=ROSTER,
        chunks={
            0: analysis(
                claims=(
                    claim("The GDP grew 3%.", "unknown", 850.0),
                    claim("solo claim", "Alice", 20.0),
                    claim("solo claim", "Alice", 30.0),
                ),
                quotes=(quote("shared text", "Alice", 855.0),),
            ),
            1: analysis(
                claims=(claim("  the gdp  grew 3%", "Bob", 851.0),),
                quotes=(quote("...", "Bob", 900.0),),
            ),
        },
    )
    rig = rig_factory(summarizer)

    rig.run()

    got = rig.latest()
    assert sorted((c.text, c.speaker, c.source_chunk_seq) for c in got.claims) == [
        ("  the gdp  grew 3%", "Bob", 1),  # the named speaker beats unknown
        ("solo claim", "Alice", 0),
    ]
    assert [q.text for q in got.quotes] == ["shared text"]
    done = [r for r in rig.logs.records("info") if "analysis_id" in r]
    assert done[0]["claims_deduped"] == 2
    assert done[0]["quotes_deduped"] == 1


@integration
def test_a_claim_and_a_quote_with_the_same_text_are_both_kept(
    rig_factory: Callable[..., Rig],
) -> None:
    summarizer = FakeSummarizer(
        chunks={0: analysis(claims=(claim("same"),), quotes=(quote("same"),))}
    )
    rig = rig_factory(summarizer, segments={"whisper": (S0,)})

    rig.run()

    assert rig.rows()[2:] == (1, 1)


# -------------------------------------------------------- persistence, resources


@integration
def test_tokens_are_the_sum_of_every_successful_call(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(FakeSummarizer(roster=ROSTER))

    rig.run()

    got = rig.latest()
    assert (got.input_tokens, got.output_tokens) == (7 + 2 * 100 + 20, 3 + 2 * 50 + 10)


@integration
def test_a_second_job_on_the_same_summarizer_counts_only_its_own_tokens(
    rig_factory: Callable[..., Rig],
) -> None:
    summarizer = FakeSummarizer(fail_on={"chunk:1": LLMUnavailableError("down")})
    rig = rig_factory(summarizer)
    with pytest.raises(LLMUnavailableError):
        rig.run()  # leaves the roster and chunk-0 usage behind
    summarizer.fail_on.clear()

    rig.run()

    got = rig.latest()
    assert (got.input_tokens, got.output_tokens) == (7 + 2 * 100 + 20, 3 + 2 * 50 + 10)


@integration
def test_the_clock_is_read_once_at_the_start_and_once_before_persisting(
    rig_factory: Callable[..., Rig],
) -> None:
    clock = FakeClock(start=10.0, step=0.75)
    rig = rig_factory(clock=clock)

    rig.run()

    assert clock.calls == 2
    assert rig.latest().duration_ms == 750


class _FailingInsertConnection(psycopg.Connection[Any]):
    """Raises when a ``claims`` row is inserted, after ``analyses`` and ``topics`` were."""

    def execute(self, query: Any, params: Any = None, **kwargs: Any) -> Any:
        if isinstance(query, str) and "INSERT INTO claims" in query:
            raise RuntimeError("injected failure")
        return super().execute(query, params, **kwargs)


@integration
def test_a_failure_after_the_analysis_row_is_inserted_leaves_no_rows(
    rig_factory: Callable[..., Rig],
) -> None:
    summarizer = FakeSummarizer(
        chunks={0: analysis(topics=(Topic(0, "t"),), claims=(claim("c"),))}
    )
    rig = rig_factory(summarizer, connect_cls=_FailingInsertConnection)

    with pytest.raises(RuntimeError, match="injected"):
        rig.run()

    assert rig.rows() == (0, 0, 0, 0)


def _idle_in_transaction(dsn: str, app: str) -> int:
    with psycopg.connect(dsn, autocommit=True) as watcher:
        row = watcher.execute(
            "SELECT count(*) FROM pg_stat_activity "
            "WHERE application_name = %s AND state LIKE 'idle in transaction%%'",
            (app,),
        ).fetchone()
    assert row is not None
    return int(row[0])


@integration
def test_the_idle_in_transaction_probe_can_see_an_open_transaction(head_dsn: str) -> None:
    with psycopg.connect(head_dsn, application_name="probe-check") as held:
        held.execute("SELECT 1")

        assert _idle_in_transaction(head_dsn, "probe-check") == 1


@integration
def test_no_transaction_is_open_while_a_summarizer_call_runs(
    rig_factory: Callable[..., Rig],
) -> None:
    seen: list[int] = []
    rig = rig_factory()
    rig.summarizer.on_call = lambda kind: seen.append(_idle_in_transaction(rig.dsn, "analyze-handler"))

    rig.run()

    assert len(seen) == 4  # roster, two chunks, reduce
    assert seen == [0, 0, 0, 0]


@integration
def test_the_completion_log_line_reports_the_run(rig_factory: Callable[..., Rig]) -> None:
    summarizer = FakeSummarizer(
        roster=ROSTER,
        chunks={0: analysis(claims=(claim("c", "Mallory"),), speaker_coercions=1)},
    )
    rig = rig_factory(summarizer, clock=FakeClock(step=1.5))

    rig.run()

    done = [r for r in rig.logs.records("info") if "analysis_id" in r]
    assert len(done) == 1
    line = done[0]
    assert line["analysis_id"] == rig.latest().id
    assert line["chunks"] == 2
    assert (line["input_tokens"], line["output_tokens"]) == (227, 113)
    assert line["duration_ms"] == 1500
    assert line["speakers_coerced"] == 2
    assert (line["claims_deduped"], line["quotes_deduped"]) == (0, 0)


@integration
def test_the_connection_is_closed_after_the_job(
    conn: psycopg.Connection[Any], head_dsn: str
) -> None:
    opened: list[psycopg.Connection[Any]] = []

    def connect() -> psycopg.Connection[Any]:
        opened.append(psycopg.connect(head_dsn))
        return opened[-1]

    rig = Rig(conn, head_dsn, FakeSummarizer())
    rig.handler = make_analyze_handler(
        connect=connect, summarizer=rig.summarizer, settings=rig.settings, clock=FakeClock()
    )

    rig.run()

    assert opened
    assert all(c.closed for c in opened)


# ------------------------------------------------------- progress (issue #96)


def progress_rig(
    rig_factory: Callable[..., Rig], summarizer: FakeSummarizer, **kwargs: Any
) -> tuple[Rig, list[str]]:
    """A rig whose ctx records each progress event and each summarizer call in order."""
    events: list[str] = []
    summarizer.on_call = lambda kind: events.append(f"call {kind}")
    rig = rig_factory(summarizer, **kwargs)

    def hook(stage: str, done: float, total: float, **fields: Any) -> None:
        assert stage == "analyze"
        phase = fields["phase"]
        events.append(f"{phase} {int(done)}/{int(total)}" if phase == "chunk" else str(phase))

    rig.ctx = JobContext(rig.ctx.logger, hook)
    return rig, events


@integration
def test_each_llm_call_is_announced_just_before_it_starts(
    rig_factory: Callable[..., Rig],
) -> None:
    rig, events = progress_rig(rig_factory, FakeSummarizer(roster=ROSTER))

    rig.run()

    assert events == [
        "roster",
        "call roster",
        "chunk 1/2",
        "call chunk",
        "chunk 2/2",
        "call chunk",
        "reduce",
        "call reduce",
    ]


@integration
def test_an_invalid_roster_still_announces_roster_and_the_chunk_lines_follow(
    rig_factory: Callable[..., Rig],
) -> None:
    summarizer = FakeSummarizer(fail_on={"roster": LLMInvalidOutputError("bad json")})
    rig, events = progress_rig(rig_factory, summarizer)

    rig.run()

    assert [e for e in events if not e.startswith("call")] == [
        "roster",
        "chunk 1/2",
        "chunk 2/2",
        "reduce",
    ]


@integration
def test_a_single_chunk_transcript_announces_chunk_1_of_1(
    rig_factory: Callable[..., Rig],
) -> None:
    rig, events = progress_rig(rig_factory, FakeSummarizer(), segments={"whisper": (S0, S1)})

    rig.run()

    assert [e for e in events if not e.startswith("call")] == [
        "roster",
        "chunk 1/1",
        "reduce",
    ]


@integration
@pytest.mark.parametrize("segments", [(), (Segment(0, 1, "   "),)])
def test_a_transcript_without_chunks_announces_nothing(
    rig_factory: Callable[..., Rig], segments: tuple[Segment, ...]
) -> None:
    rig, events = progress_rig(rig_factory, FakeSummarizer(), segments={"whisper": segments})

    rig.run()

    assert events == []


@integration
def test_an_already_analyzed_video_announces_nothing(rig_factory: Callable[..., Rig]) -> None:
    rig, events = progress_rig(rig_factory, FakeSummarizer())
    rig.run()
    events.clear()

    rig.run()

    assert events == []


@integration
def test_chunk_lines_match_chunk_calls_one_for_one(
    rig_factory: Callable[..., Rig],
) -> None:
    # Announced per handler-level call, so a retry inside the summarizer adds no line.
    summarizer = FakeSummarizer()
    rig, events = progress_rig(rig_factory, summarizer, segments={"whisper": (S0, S1)})

    rig.run()

    assert sum(e.startswith("chunk") for e in events) == len(summarizer.kinds) - 2


@integration
def test_the_worker_default_logs_one_structured_line_per_llm_call(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = rig_factory(FakeSummarizer())

    rig.run()

    lines = [r for r in rig.logs.records("info") if r["event"] == "analyze progress"]
    assert [(r["phase"], r["done"], r["total"]) for r in lines] == [
        ("roster", 0, 2),
        ("chunk", 1, 2),
        ("chunk", 2, 2),
        ("reduce", 2, 2),
    ]
    assert all(r["stage"] == "analyze" for r in lines)


# ------------------------------------------------- batch-capable summarizers (#88)

BATCH_CHUNKS = {
    0: analysis(
        topics=(Topic(0, "T0a"), Topic(1, "T0b")),
        claims=(claim("claim one", "Alice", 10),),
        quotes=(quote("quote one", "Bob", 20),),
    ),
    1: analysis(
        topics=(Topic(0, "T1a", "sum", 900.0),),
        claims=(claim("claim two", "Bob", 1000),),
    ),
}


@integration
def test_a_batch_capable_summarizer_gets_every_chunk_in_one_call(
    rig_factory: Callable[..., Rig],
) -> None:
    summarizer = BatchSummarizer(roster=ROSTER, chunks=BATCH_CHUNKS, tldr="done")
    rig = rig_factory(summarizer)

    rig.run()

    assert summarizer.batch_calls == 1
    assert summarizer.kinds == ["roster", "chunks", "reduce"]
    batch = summarizer.calls[1][1]
    assert [c.seq for c in batch] == [0, 1]
    assert all("[" in c.text for c in batch)  # timestamped, as per-chunk calls are


@integration
@pytest.mark.parametrize("batch", [False, True])
def test_both_paths_persist_the_same_analysis(
    rig_factory: Callable[..., Rig], batch: bool
) -> None:
    kwargs: dict[str, Any] = {"roster": ROSTER, "chunks": BATCH_CHUNKS, "tldr": "  The story.  "}
    summarizer = BatchSummarizer(**kwargs) if batch else FakeSummarizer(**kwargs)
    rig = rig_factory(summarizer, clock=FakeClock(start=100.0, step=2.5))

    rig.run()

    got = rig.latest()
    assert got.tldr == "The story."
    assert got.speaker_roster == {
        "speakers": [{"name": "Alice", "role": "host"}, {"name": "Bob", "role": "guest"}]
    }
    assert [(t.seq, t.title) for t in got.topics] == [(0, "T0a"), (1, "T0b"), (2, "T1a")]
    assert sorted((c.text, c.speaker, c.source_chunk_seq) for c in got.claims) == [
        ("claim one", "Alice", 0),
        ("claim two", "Bob", 1),
    ]
    assert [(q.text, q.source_chunk_seq) for q in got.quotes] == [("quote one", 0)]


@integration
def test_a_batch_returning_the_wrong_number_of_analyses_is_a_bug_and_persists_nothing(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = rig_factory(BatchSummarizer(roster=ROSTER, chunks=BATCH_CHUNKS, result_delta=1))
    before = rig.rows()

    with pytest.raises(BugError, match="returned 3 analyses for 2 chunks"):
        rig.run()

    assert rig.rows() == before


@integration
def test_an_error_from_the_batch_call_propagates_and_persists_nothing(
    rig_factory: Callable[..., Rig],
) -> None:
    summarizer = BatchSummarizer(
        roster=ROSTER, chunks=BATCH_CHUNKS, fail_on={"chunks": LLMUnavailableError("down")}
    )
    rig = rig_factory(summarizer)
    before = rig.rows()

    with pytest.raises(LLMUnavailableError, match="down"):
        rig.run()

    assert rig.rows() == before


# ------------------------------------------------------ batch checkpoints (#89)


def _enqueue_analyze(conn: psycopg.Connection[Any]) -> int:
    job_id = PostgresQueue(conn).enqueue("analyze", VIDEO, dedupe_key="v1:fake")
    conn.commit()
    assert job_id is not None
    return job_id


def _checkpoint_steps(rig: Rig, job_id: int) -> set[str]:
    rows = rig.conn.execute(
        "SELECT step_key FROM llm_batches WHERE job_id = %s", (job_id,)
    ).fetchall()
    rig.conn.rollback()
    return {str(row[0]) for row in rows}


@integration
def test_a_saved_analysis_clears_the_jobs_batch_checkpoints(
    rig_factory: Callable[..., Rig], conn: psycopg.Connection[Any]
) -> None:
    summarizer = CheckpointingSummarizer(roster=ROSTER, chunks=BATCH_CHUNKS, tldr="done")
    rig = rig_factory(summarizer)
    job_id = _enqueue_analyze(conn)

    rig.handler(replace(make_job(VIDEO), id=job_id), rig.ctx)

    assert _checkpoint_steps(rig, job_id) == set()
    assert (summarizer.begun, summarizer.ended) == (1, 1)


@integration
def test_a_failed_job_keeps_its_checkpoints_for_the_retry(
    rig_factory: Callable[..., Rig], conn: psycopg.Connection[Any]
) -> None:
    summarizer = CheckpointingSummarizer(
        roster=ROSTER,
        chunks=BATCH_CHUNKS,
        fail_on={"reduce": LLMUnavailableError("down")},
    )
    rig = rig_factory(summarizer)
    job_id = _enqueue_analyze(conn)

    with pytest.raises(LLMUnavailableError):
        rig.handler(replace(make_job(VIDEO), id=job_id), rig.ctx)

    assert _checkpoint_steps(rig, job_id) == {"roster", "reduce"}
    assert (summarizer.begun, summarizer.ended) == (1, 1)
