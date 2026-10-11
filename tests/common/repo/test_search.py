"""``search_transcripts`` against Postgres (issue #43).

Seeded through ``common.repo`` (``upsert_video``, ``save_transcript``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from common.models import Analysis, Claim, FeedEntry, Segment, Topic, VideoMeta
from common.repo.analyses import save_analysis
from common.repo.search import SEARCH_SQL, SearchPage, Span, search_transcripts
from common.repo.transcripts import save_transcript
from common.repo.videos import insert_discovered_video, record_unavailable, upsert_video

pytestmark = pytest.mark.integration

CH = "UC" + "s" * 22


def _video(
    conn: psycopg.Connection[Any],
    video_id: str,
    *,
    published: datetime | None = datetime(2026, 1, 1, tzinfo=UTC),
    title: str = "T",
    channel_id: str = CH,
) -> None:
    upsert_video(
        conn,
        VideoMeta(
            video_id=video_id,
            channel_id=channel_id,
            title=title,
            description="",
            duration_sec=60,
            published_at=published,
            language="en",
            live_status=None,
            manual_subtitle_langs=(),
            auto_caption_langs=(),
        ),
        "adhoc",
    )


def _transcript(
    conn: psycopg.Connection[Any], video_id: str, text: str, source: str = "whisper"
) -> int:
    segments = (Segment(0.0, 1.0, text),) if text else ()
    return save_transcript(conn, video_id, source, "en", "none", segments, None)


def _seed(conn: psycopg.Connection[Any], video_id: str, text: str, **kw: Any) -> None:
    _video(conn, video_id, **kw)
    _transcript(conn, video_id, text)


def _ids(page: SearchPage) -> list[str]:
    return [hit.video_id for hit in page.results]


def _search(conn: psycopg.Connection[Any], q: str, limit: int = 50, offset: int = 0) -> SearchPage:
    return search_transcripts(conn, q, limit, offset)


# --- matching -----------------------------------------------------------------


def test_stemming_and_case(conn: psycopg.Connection[Any]) -> None:
    _seed(conn, "stem0000001", "she was running late")
    _seed(conn, "case0000001", "nasa launched a rocket")

    assert _ids(_search(conn, "runs")) == ["stem0000001"]
    assert _ids(_search(conn, "NASA")) == ["case0000001"]


def test_websearch_syntax(conn: psycopg.Connection[Any]) -> None:
    _seed(conn, "phrase00001", "we study machine learning daily")
    _seed(conn, "phrase00002", "learning about a machine is fun")
    _seed(conn, "cats0000001", "cats are great")
    _seed(conn, "catsdogs001", "cats and dogs together")
    _seed(conn, "dogs0000001", "dogs bark")

    assert _ids(_search(conn, '"machine learning"')) == ["phrase00001"]
    assert sorted(_ids(_search(conn, "cats -dogs"))) == ["cats0000001"]
    assert sorted(_ids(_search(conn, "cats or dogs"))) == [
        "cats0000001", "catsdogs001", "dogs0000001",
    ]


@pytest.mark.parametrize("q", ["the", "and of", "-dogs", "-dogs -cats", "or", "'", "😀"])
def test_a_query_without_a_positive_term_returns_nothing(
    conn: psycopg.Connection[Any], q: str
) -> None:
    _seed(conn, "cats0000001", "cats are great")
    _seed(conn, "the00000001", "the and of")

    page = _search(conn, q)

    assert page == SearchPage((), False)


def test_a_query_without_a_positive_term_never_runs_the_search(
    conn: psycopg.Connection[Any],
) -> None:
    queries: list[str] = []

    class Spy:
        def execute(self, query: Any, params: Any = None) -> Any:
            queries.append(str(query))
            return conn.execute(query, params)

    search_transcripts(Spy(), "-dogs", 10, 0)  # type: ignore[arg-type]

    assert SEARCH_SQL not in queries
    assert not any("ts_headline" in q or "FROM transcripts" in q for q in queries)


def test_only_the_preferred_transcript_is_searched(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "pref0000001")
    _transcript(conn, "pref0000001", "a zebra appears", "youtube_auto")
    _transcript(conn, "pref0000001", "a horse appears", "whisper")
    _video(conn, "pref0000002")
    _transcript(conn, "pref0000002", "a zebra appears", "youtube_auto")

    page = _search(conn, "zebra")

    assert _ids(page) == ["pref0000002"]
    assert page.results[0].transcript_source == "youtube_auto"
    assert _ids(_search(conn, "horse")) == ["pref0000001"]
    assert _search(conn, "horse").results[0].transcript_source == "whisper"


def test_manual_beats_whisper(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "pref0000003")
    _transcript(conn, "pref0000003", "zebra", "whisper")
    _transcript(conn, "pref0000003", "zebra too", "youtube_manual")

    [hit] = _search(conn, "zebra").results

    assert hit.transcript_source == "youtube_manual"


def test_a_video_appears_once_even_if_several_transcripts_match(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "once0000001")
    for source in ("youtube_auto", "whisper", "youtube_manual"):
        _transcript(conn, "once0000001", "zebra zebra", source)

    assert _ids(_search(conn, "zebra")) == ["once0000001"]


def test_empty_transcripts_and_untitled_videos_without_content_never_match(
    conn: psycopg.Connection[Any],
) -> None:
    _seed(conn, "empty000001", "")
    conn.execute("INSERT INTO videos (video_id) VALUES ('bare0000001')")
    conn.execute("INSERT INTO videos (video_id, title) VALUES ('null0000001', NULL)")

    assert _search(conn, "zebra").results == ()


def test_unavailable_videos_are_returned_with_the_field_set(
    conn: psycopg.Connection[Any],
) -> None:
    _seed(conn, "unav0000001", "zebra")
    record_unavailable(conn, "unav0000001", "removed", origin="adhoc")

    [hit] = _search(conn, "zebra").results

    assert hit.unavailable == "removed"


def test_result_metadata_and_nullable_columns(conn: psycopg.Connection[Any]) -> None:
    _seed(conn, "meta0000001", "zebra", published=datetime(2026, 3, 1, tzinfo=UTC))
    conn.execute("UPDATE channels SET title = 'Chan' WHERE channel_id = %s", (CH,))
    conn.execute("INSERT INTO videos (video_id) VALUES ('stub0000001')")
    _transcript(conn, "stub0000001", "zebra")

    hits = {h.video_id: h for h in _search(conn, "zebra").results}

    meta = hits["meta0000001"]
    assert (meta.title, meta.channel_id, meta.channel_title, meta.published_at,
            meta.duration_sec, meta.unavailable, meta.transcript_source) == (
        "T", CH, "Chan", datetime(2026, 3, 1, tzinfo=UTC), 60, None, "whisper",
    )
    stub = hits["stub0000001"]
    assert (stub.title, stub.channel_id, stub.channel_title, stub.published_at,
            stub.duration_sec) == (None, None, None, None, None)


@pytest.mark.parametrize(
    "q",
    [
        "'", '"', '"unbalanced', "' OR 1=1 --", "foo & bar | !(baz:*) <-> qux", "\\", "%", "_",
        "-", "--", "or", "<script>alert(1)</script>", "café", "🦓", "!@#$%^&*()-+=[]{};:,.<>/?"
        * 8,
    ],
)
def test_hostile_queries_do_not_raise(conn: psycopg.Connection[Any], q: str) -> None:
    _seed(conn, "host0000001", "café zebra foo bar baz qux script alert")

    page = _search(conn, q[:200])

    assert isinstance(page, SearchPage)


def test_the_search_uses_the_gin_index(conn: psycopg.Connection[Any]) -> None:
    _seed(conn, "idx00000001", "zebra")
    conn.execute("SET enable_seqscan = off")

    plan = conn.execute(
        "EXPLAIN " + SEARCH_SQL, {"q": "zebra", "limit": 21, "offset": 0}
    ).fetchall()

    assert "search_docs_fts_idx" in "\n".join(row[0] for row in plan)
    assert "to_tsvector" not in SEARCH_SQL
    assert "websearch_to_tsquery('english'" in SEARCH_SQL


def test_a_nul_byte_in_the_query_is_rejected_without_touching_sql(
    conn: psycopg.Connection[Any],
) -> None:
    class Spy:
        def execute(self, query: Any, params: Any = None) -> Any:
            raise AssertionError("must not run SQL")

    with pytest.raises(ValueError, match="NUL"):
        search_transcripts(Spy(), "zebra\x00", 10, 0)  # type: ignore[arg-type]


def test_two_transcripts_of_the_same_unrecognised_rank_still_give_one_result(
    conn: psycopg.Connection[Any],
) -> None:
    # transcript_rank() falls back to the same rank (3) for any source it
    # does not recognise; save_transcript's own _SOURCES check keeps this
    # from happening through the application, but the SQL still has to
    # produce one row per video, not two, if it ever does.
    _video(conn, "tie0000001")
    conn.execute(
        "INSERT INTO transcripts (video_id, source, segments, full_text) VALUES"
        " (%s, 'bogus_a', '[]', 'zebra one'), (%s, 'bogus_b', '[]', 'zebra two')",
        ("tie0000001", "tie0000001"),
    )

    page = _search(conn, "zebra")

    assert len(page.results) == 1


def test_the_search_writes_nothing(conn: psycopg.Connection[Any]) -> None:
    _seed(conn, "rw000000001", "zebra")

    def counts() -> Any:
        return conn.execute(
            "SELECT (SELECT count(*) FROM videos), (SELECT count(*) FROM transcripts),"
            " (SELECT count(*) FROM channels), (SELECT count(*) FROM jobs)"
        ).fetchone()

    before = counts()
    _search(conn, "zebra")
    assert counts() == before


# --- ranking and paging -------------------------------------------------------


def test_relevance_then_published_then_video_id(conn: psycopg.Connection[Any]) -> None:
    _seed(conn, "rank_low001", "zebra " + "filler " * 40, published=datetime(2026, 9, 1,
                                                                            tzinfo=UTC))
    _seed(conn, "rank_hi0001", "zebra zebra zebra", published=datetime(2020, 1, 1, tzinfo=UTC))
    _seed(conn, "tie_b000001", "zebra horse", published=datetime(2025, 1, 1, tzinfo=UTC))
    _seed(conn, "tie_a000001", "zebra horse", published=datetime(2025, 1, 1, tzinfo=UTC))
    _seed(conn, "tie_new0001", "zebra horse", published=datetime(2025, 6, 1, tzinfo=UTC))
    _seed(conn, "tie_null001", "zebra horse", published=None)

    assert _ids(_search(conn, "zebra")) == [
        "rank_hi0001", "tie_new0001", "tie_a000001", "tie_b000001", "tie_null001",
        "rank_low001",
    ]


def test_a_short_transcript_outranks_a_long_one_with_the_same_hits(
    conn: psycopg.Connection[Any],
) -> None:
    short = "zebra " + "word " * 48 + "zebra " + "word " * 49
    long = "zebra " + "word " * 2498 + "zebra " + "word " * 2499
    _seed(conn, "short000001", short, published=datetime(2020, 1, 1, tzinfo=UTC))
    _seed(conn, "long0000001", long, published=datetime(2026, 1, 1, tzinfo=UTC))

    assert _ids(_search(conn, "zebra")) == ["short000001", "long0000001"]


def test_has_more_boundaries(conn: psycopg.Connection[Any]) -> None:
    for i in range(3):
        _seed(conn, f"more{i:07d}", "zebra")

    assert _search(conn, "zebra", limit=3).has_more is False
    assert _search(conn, "zebra", limit=2).has_more is True
    assert _search(conn, "zebra", limit=2, offset=1).has_more is False
    last = _search(conn, "zebra", limit=2, offset=2)
    assert (len(last.results), last.has_more) == (1, False)
    beyond = _search(conn, "zebra", limit=2, offset=3)
    assert beyond == SearchPage((), False)


def test_walking_pages_of_one_matches_a_single_page(conn: psycopg.Connection[Any]) -> None:
    for i in range(7):
        _seed(conn, f"walk{i:07d}", "zebra " * (i % 3 + 1) + "x " * i,
              published=datetime(2026, 1, 1 + i % 2, tzinfo=UTC))

    whole = _ids(_search(conn, "zebra", limit=50))
    walked: list[str] = []
    for offset in range(20):
        page = _search(conn, "zebra", limit=1, offset=offset)
        walked.extend(_ids(page))
        if not page.has_more:
            break
    else:
        raise AssertionError("has_more never became false")

    assert walked == whole
    assert len(set(walked)) == 7


# --- excerpts -----------------------------------------------------------------


def test_matched_words_are_marked(conn: psycopg.Connection[Any]) -> None:
    _seed(conn, "exc00000001", "the zebra ran")

    [hit] = _search(conn, "zebra").results

    assert 1 <= len(hit.excerpts) <= 3
    spans = [s for fragment in hit.excerpts for s in fragment]
    assert Span("zebra", True) in spans
    assert all(s.text for s in spans)
    assert "".join(s.text for s in hit.excerpts[0]) == "the zebra ran"


def test_html_in_a_transcript_is_literal_unmarked_text(conn: psycopg.Connection[Any]) -> None:
    _seed(conn, "html0000001", "zebra <script>alert(1)</script> and <b>fake</b> & more")

    [hit] = _search(conn, "zebra").results

    spans = [s for fragment in hit.excerpts for s in fragment]
    text = "".join(s.text for s in spans)
    assert "<script>alert(1)</script>" in text
    assert "<b>fake</b>" in text
    assert [s.text for s in spans if s.match] == ["zebra"]


def test_marker_characters_in_a_transcript_cannot_forge_matches(
    conn: psycopg.Connection[Any],
) -> None:
    forged = "zebra \x01fake\x02 \x03 split \ue000x\ue001 \x01dangling"
    _seed(conn, "forge000001", forged)

    [hit] = _search(conn, "zebra").results

    spans = [s for fragment in hit.excerpts for s in fragment]
    assert [s.text for s in spans if s.match] == ["zebra"]
    assert len(hit.excerpts) == 1
    for span in spans:
        assert not any(c in span.text for c in "\x01\x02\x03\ue000\ue001")


def test_long_transcripts_give_at_most_three_short_fragments(
    conn: psycopg.Connection[Any],
) -> None:
    filler = " ".join(f"w{i}" for i in range(200))
    _seed(conn, "frag0000001", f"zebra {filler} zebra {filler} zebra {filler} zebra {filler}")

    [hit] = _search(conn, "zebra").results

    assert 1 <= len(hit.excerpts) <= 3
    for fragment in hit.excerpts:
        words = "".join(s.text for s in fragment).split()
        assert len(words) <= 36
        assert any(s.match for s in fragment)


def test_headline_is_computed_only_for_the_page(conn: psycopg.Connection[Any]) -> None:
    head, _, tail = SEARCH_SQL.partition("ts_headline")
    assert tail, "SEARCH_SQL must call ts_headline"
    assert "LIMIT" in head and "OFFSET" in head
    # Past the page CTE the only LIMIT is the per-row "best transcript" lookup.
    assert "OFFSET" not in tail
    assert tail.count("LIMIT") == 1 and "FROM page" in tail


# --- title and analysis text (#111) --------------------------------------------


def _analysis(
    conn: psycopg.Connection[Any],
    video_id: str,
    transcript_id: int,
    tldr: str,
    *,
    topic: tuple[str, str] = ("", ""),
    claim: str = "nothing",
) -> int:
    analysis = Analysis(
        video_id=video_id,
        transcript_id=transcript_id,
        chunk_strategy="x",
        model="m",
        prompt_version="p",
        tldr=tldr,
        topics=(Topic(0, topic[0] or "plain", topic[1] or None),),
        claims=(Claim(claim),),
    )
    saved = save_analysis(conn, analysis)
    conn.commit()  # the analyses trigger is deferred: it fires at commit
    return saved


def _full(
    conn: psycopg.Connection[Any],
    video_id: str,
    *,
    title: str = "plain title words",
    tldr: str = "plain tldr words",
    topic: tuple[str, str] = ("plain topic", "words"),
    claim: str = "plain claim words",
    transcript: str = "plain transcript words",
    published: datetime | None = datetime(2026, 1, 1, tzinfo=UTC),
) -> None:
    _video(conn, video_id, title=title, published=published)
    tid = _transcript(conn, video_id, transcript)
    _analysis(conn, video_id, tid, tldr, topic=topic, claim=claim)


def _doc_text(conn: psycopg.Connection[Any], video_id: str) -> str:
    row = conn.execute("SELECT fts::text FROM search_docs WHERE video_id = %s", (video_id,)).fetchone()
    assert row is not None
    return str(row[0])


def test_a_title_only_match_surfaces_the_video(conn: psycopg.Connection[Any]) -> None:
    _full(conn, "title000001", title="The Zebra Report")

    [hit] = _search(conn, "zebra").results

    assert (hit.video_id, hit.matched_in) == ("title000001", ("title",))
    assert hit.excerpts == ()


def test_a_titled_video_without_transcript_or_analysis_matches_its_title(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "notrans0001", title="zebra crossing")

    [hit] = _search(conn, "zebra").results

    assert hit.transcript_source is None
    assert hit.matched_in == ("title",)
    assert hit.excerpts == ()


def test_matching_works_across_sources(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "cross000001", title="Fridman interview")
    _transcript(conn, "cross000001", "we discussed nuclear power")
    _seed(conn, "cross000002", "nuclear power only", title="other")

    [hit] = _search(conn, "fridman nuclear").results

    assert (hit.video_id, hit.matched_in) == ("cross000001", ("title", "transcript"))
    assert any(s.match for fragment in hit.excerpts for s in fragment)


def test_exclusion_applies_to_the_whole_document(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "excl0000001", title="cats are great")
    _transcript(conn, "excl0000001", "dogs bark")
    _video(conn, "excl0000002", title="cats are great")
    _transcript(conn, "excl0000002", "birds sing")

    assert _ids(_search(conn, "cats -dogs")) == ["excl0000002"]


def test_matched_in_lists_every_matching_source_in_fixed_order(
    conn: psycopg.Connection[Any],
) -> None:
    _full(conn, "mi000000001", title="zebra", tldr="zebra", claim="zebra", transcript="zebra")
    _full(conn, "mi000000002", claim="a zebra claim")
    _full(conn, "mi000000003", topic=("zebra", "x"), transcript="zebra here")

    hits = {h.video_id: h for h in _search(conn, "zebra").results}

    assert hits["mi000000001"].matched_in == ("title", "analysis", "transcript")
    assert hits["mi000000002"].matched_in == ("analysis",)
    assert hits["mi000000003"].matched_in == ("analysis", "transcript")


def test_excerpts_stay_transcript_only_and_are_empty_without_a_transcript_match(
    conn: psycopg.Connection[Any],
) -> None:
    _full(conn, "ex000000001", tldr="a zebra summary", transcript="plain talk about horses")

    [hit] = _search(conn, "zebra").results

    assert hit.matched_in == ("analysis",)
    assert hit.transcript_source == "whisper"
    assert hit.excerpts == ()


def test_a_negated_word_does_not_count_as_a_match_source(conn: psycopg.Connection[Any]) -> None:
    _full(conn, "neg00000001", title="plain", tldr="plain", claim="plain", transcript="zebra here")

    [hit] = _search(conn, "zebra -dogs").results

    assert hit.matched_in == ("transcript",)


def test_only_the_latest_analysis_counts(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "latest00001")
    tid = _transcript(conn, "latest00001", "plain")
    _analysis(conn, "latest00001", tid, "all about giraffes", claim="okapi claim")
    _analysis(conn, "latest00001", tid, "all about lions", claim="zebra claim")

    assert _search(conn, "giraffes").results == ()
    assert _search(conn, "okapi").results == ()
    assert _ids(_search(conn, "lions")) == ["latest00001"]
    assert _ids(_search(conn, "zebra")) == ["latest00001"]


def test_analyses_created_in_one_transaction_tie_on_created_at_and_the_higher_id_wins(
    conn: psycopg.Connection[Any],
) -> None:
    # now() is the transaction start, so both rows get the same created_at.
    _video(conn, "sametx00001")
    tid = _transcript(conn, "sametx00001", "plain")
    for tldr in ("about giraffes", "about lions"):
        save_analysis(conn, Analysis("sametx00001", tid, "x", "m", "p", tldr))
    conn.commit()
    created = conn.execute(
        "SELECT count(DISTINCT created_at) FROM analyses WHERE video_id = 'sametx00001'"
    ).fetchone()
    assert created == (1,)

    assert _search(conn, "giraffes").results == ()
    assert _ids(_search(conn, "lions")) == ["sametx00001"]


def test_a_non_preferred_transcript_does_not_match(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "nonpref0001")
    _transcript(conn, "nonpref0001", "a zebra appears", "youtube_auto")
    _transcript(conn, "nonpref0001", "a horse appears", "whisper")

    assert _search(conn, "zebra").results == ()


# --- ranking by source ----------------------------------------------------------


def test_title_outranks_tldr_outranks_claim_outranks_transcript(
    conn: psycopg.Connection[Any],
) -> None:
    # Equal length, one "zebra" each; the transcript hit is the newest.
    fields = {"title": "plain title words", "tldr": "plain tldr words",
              "claim": "plain claim words", "transcript": "plain transcript words"}
    for day, (field, base) in enumerate(fields.items(), start=1):
        _full(conn, f"src_{field}"[:11].ljust(11, "_"),
              published=datetime(2026, 1, day, tzinfo=UTC),
              **{field: base.replace("plain", "zebra")})  # type: ignore[arg-type]

    assert _ids(_search(conn, "zebra")) == [
        "src_title__", "src_tldr___", "src_claim__", "src_transcr",
    ]


def test_ties_in_rank_fall_back_to_published_then_video_id(
    conn: psycopg.Connection[Any],
) -> None:
    # Identical documents: genuinely equal rank, so only the tie-breakers order them.
    _full(conn, "tie_b000001", claim="zebra claim words", published=datetime(2025, 1, 1, tzinfo=UTC))
    _full(conn, "tie_a000001", claim="zebra claim words", published=datetime(2025, 1, 1, tzinfo=UTC))
    _full(conn, "tie_new0001", claim="zebra claim words", published=datetime(2025, 6, 1, tzinfo=UTC))
    _full(conn, "tie_null001", claim="zebra claim words", published=None)
    ranks = conn.execute(
        "SELECT DISTINCT ts_rank_cd('{0.1, 0.2, 0.4, 1.0}', fts, websearch_to_tsquery('english', 'zebra'), 1)"
        " FROM search_docs WHERE video_id LIKE 'tie\\_%'"
    ).fetchall()
    assert len(ranks) == 1

    assert _ids(_search(conn, "zebra")) == [
        "tie_new0001", "tie_a000001", "tie_b000001", "tie_null001",
    ]


# --- search_docs stays in sync (#111) ---------------------------------------------


def test_changing_a_title_moves_the_match(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "retitle0001", title="giraffe safari")
    assert _ids(_search(conn, "giraffe")) == ["retitle0001"]

    _video(conn, "retitle0001", title="zebra safari")

    assert _search(conn, "giraffe").results == ()
    assert _ids(_search(conn, "zebra")) == ["retitle0001"]


def test_an_rss_stub_is_searchable_by_title_before_any_transcript(
    conn: psycopg.Connection[Any],
) -> None:
    conn.execute("INSERT INTO channels (channel_id) VALUES (%s)", (CH,))
    entry = FeedEntry("rss00000001", CH, "Zebra Weekly", datetime(2026, 2, 1, tzinfo=UTC))
    assert insert_discovered_video(conn, entry, "monitor")

    [hit] = _search(conn, "zebra").results

    assert (hit.video_id, hit.transcript_source) == ("rss00000001", None)


def test_a_whisper_transcript_replaces_the_auto_one(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "swap0000001")
    _transcript(conn, "swap0000001", "only giraffes here", "youtube_auto")
    assert _ids(_search(conn, "giraffes")) == ["swap0000001"]

    _transcript(conn, "swap0000001", "only zebras here", "whisper")

    assert _search(conn, "giraffes").results == ()
    assert _ids(_search(conn, "zebras")) == ["swap0000001"]


def test_updating_a_transcript_updates_the_document(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "resave000001")
    _transcript(conn, "resave000001", "giraffes")
    conn.execute("UPDATE transcripts SET full_text = 'zebras' WHERE video_id = 'resave000001'")

    assert _search(conn, "giraffes").results == ()
    assert _ids(_search(conn, "zebras")) == ["resave000001"]


def test_deleting_a_transcript_removes_its_words(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "deltr0000001", title="plain")
    _transcript(conn, "deltr0000001", "zebras")
    conn.execute("DELETE FROM transcripts WHERE video_id = 'deltr0000001'")

    assert _search(conn, "zebras").results == ()


def test_deleting_an_analysis_removes_its_words(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "delan0000001")
    tid = _transcript(conn, "delan0000001", "plain")
    _analysis(conn, "delan0000001", tid, "about zebras")
    assert _ids(_search(conn, "zebras")) == ["delan0000001"]

    conn.execute("DELETE FROM analyses WHERE video_id = 'delan0000001'")
    conn.commit()

    assert _search(conn, "zebras").results == ()


def test_a_rolled_back_analysis_leaves_the_document_unchanged(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "rollback0001")
    tid = _transcript(conn, "rollback0001", "plain")
    conn.commit()
    before = _doc_text(conn, "rollback0001")

    save_analysis(
        conn,
        Analysis("rollback0001", tid, "x", "m", "p", "about zebras", claims=(Claim("zebra claim"),)),
    )
    conn.rollback()

    assert _doc_text(conn, "rollback0001") == before
    assert _search(conn, "zebras").results == ()


def test_the_analysis_trigger_fires_after_children_are_inserted(
    conn: psycopg.Connection[Any],
) -> None:
    # A word that only exists in a claim, saved by save_analysis in one
    # transaction: the deferred trigger must see the claim row.
    _video(conn, "deferred0001")
    tid = _transcript(conn, "deferred0001", "plain")
    _analysis(conn, "deferred0001", tid, "plain", claim="the okapi claim")

    assert _ids(_search(conn, "okapi")) == ["deferred0001"]


def test_deleting_a_video_removes_its_document(conn: psycopg.Connection[Any]) -> None:
    _seed(conn, "gone0000001", "zebra")
    conn.commit()

    conn.execute("DELETE FROM videos WHERE video_id = 'gone0000001'")
    conn.commit()

    row = conn.execute("SELECT count(*) FROM search_docs WHERE video_id = 'gone0000001'").fetchone()
    assert row == (0,)


def test_deleting_a_video_with_an_analysis_commits_cleanly(
    conn: psycopg.Connection[Any],
) -> None:
    _full(conn, "gone0000002", title="zebra")
    conn.execute("DELETE FROM analyses WHERE video_id = 'gone0000002'")
    conn.execute("DELETE FROM transcripts WHERE video_id = 'gone0000002'")
    conn.execute("DELETE FROM videos WHERE video_id = 'gone0000002'")
    conn.commit()

    assert _search(conn, "zebra").results == ()


def test_a_four_hour_transcript_saves_and_stays_searchable(
    conn: psycopg.Connection[Any],
) -> None:
    words = [f"word{i % 6000}" for i in range(40_000)]
    segments = tuple(Segment(float(i), i + 1.0, " ".join(words[i : i + 20]))
                     for i in range(0, 40_000, 20))
    _video(conn, "longpod00001", title="podcast zebra")
    save_transcript(conn, "longpod00001", "whisper", "en", "none", segments, None)

    assert _ids(_search(conn, "word5999")) == ["longpod00001"]
    assert _ids(_search(conn, "zebra")) == ["longpod00001"]
