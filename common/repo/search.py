"""Full-text search over videos' titles, analyses and transcripts (#43, #111).

``search_transcripts`` matches ``search_docs.fts`` (GIN index
``search_docs_fts_idx``). That
vector is materialised per video by the database (migration 0010): title (weight
A), latest analysis TL;DR (B), topics and claims (C) and the preferred
transcript (D). Only a video's preferred transcript counts: the one with the
lowest ``transcript_rank(source)`` (D2), the same one ``get_best_transcript``
returns. Results are ranked by ``ts_rank_cd`` with explicit weights and length
normalisation, then ``published_at`` (newest first, nulls last), then
``video_id``. Language (#112): the transcript part of the vector is stemmed with the
configuration ``transcript_ts_config(language)`` picks. The index condition is
one row-independent tsquery, ``transcript_search_tsquery(q)`` (the OR of
``websearch_to_tsquery(cfg, q)`` over every configuration the mapping can
return). Each candidate is then checked per source (#112): the transcript's own
configuration query against the transcript only (weight D) and the English query
against the title and analysis only (weights A, B, C, which are always English, so
they stay searchable on a non-English video). A row is never admitted because the
English stem of a word matches another language's transcript. A row is admitted on
a lexeme in a source exactly when that source is in ``matched_in`` and its
excerpts are highlighted: ``matched_in`` and the headlines use the same per-source
queries as the filter. ``matched_in`` says which sources contain a query lexeme,
derived from the weights stored in the vector (``ts_filter``), for the page's
rows only. The rank (``ts_rank_cd``) uses both queries over the whole vector, for
admitted rows only.

``q`` is only ever a bound parameter, and ``websearch_to_tsquery`` never
raises a syntax error. A query with no positive term (only stop words, or
only exclusions such as ``-dogs``, which would otherwise match nearly every
transcript) returns no results without running the search.

Excerpts (#110) are windows over the preferred transcript's ``segments``: one
segment whose text matches, or two adjacent segments that match only when
joined, each with the ``start`` of its first segment (at most 3, best
``ts_rank`` first, then in start order; see ``WINDOWS_SQL``). When no window
matches, the excerpts come from ``ts_headline`` over ``full_text`` with no
timestamp. Either way the text is highlighted by ``ts_headline``, evaluated in the outer query over the
already-limited page only. Matches are delimited with control characters,
never HTML, and split into ``Span``s here, so the client never has markup to
inject. The marker characters are stripped from the transcript text first,
so a transcript cannot forge a match. ``<``, ``>`` and ``&`` are swapped for
private-use characters around ``ts_headline`` (whose parser would otherwise
drop anything that looks like an HTML tag) and swapped back afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import psycopg

_START, _STOP, _DELIM = "\x01", "\x02", "\x03"
_PROTECT = {"<": "\ue000", ">": "\ue001", "&": "\ue002"}
_RESTORE = str.maketrans({v: k for k, v in _PROTECT.items()})

#: The ts_headline options: at most 3 fragments of about 35 words.
HEADLINE_OPTIONS = "MaxFragments=3, MaxWords=35, MinWords=15, ShortWord=0"
#: A segment window is one short text: a single fragment of the same size.
WINDOW_OPTIONS = "MaxFragments=0, MaxWords=35, MinWords=15, ShortWord=0"

# chr(1)..chr(3) are the markers; chr(57344)..chr(57346) are U+E000..U+E002.
_STRIPPED = "chr(1) || chr(2) || chr(3) || chr(57344) || chr(57345) || chr(57346)"
_PROTECTED = "chr(57344) || chr(57345) || chr(57346)"

#: ``ts_rank_cd`` weights in the order D, C, B, A (title A > TL;DR B > topics
#: and claims C > transcript D).
RANK_WEIGHTS = "{0.1, 0.2, 0.4, 1.0}"

#: Segment windows for one page row (#110), a scalar subquery over ``t``,
#: ``page.cfg`` and ``r.own`` of the outer query: a JSON array of
#: ``[start_sec, headline]``, at most 3, ascending by start. A window is one
#: segment whose own text matches, or two adjacent segments that match only
#: when joined (a phrase split at a boundary). Windows never share a segment:
#: of a run of consecutive qualifying pairs, every other one (starting with the
#: first) is kept, which is what a left-to-right greedy pass keeps. Segments
#: without text, and windows whose first segment has no valid ``start``
#: (missing, not a number, negative, above 1e9), are skipped. Evaluated only
#: for the rows of the returned page that matched in the transcript.
WINDOWS_SQL = f"""
    SELECT coalesce(
        jsonb_agg(jsonb_build_array(top.start, top.headline) ORDER BY top.start, top.ord),
        '[]'::jsonb) AS windows
    FROM (
        WITH segs AS (
            SELECT e.ord,
                   CASE WHEN jsonb_typeof(e.seg -> 'text') = 'string'
                        THEN e.seg ->> 'text' ELSE '' END AS text,
                   CASE WHEN jsonb_typeof(e.seg -> 'start') = 'number'
                        THEN CASE WHEN (e.seg ->> 'start')::numeric BETWEEN 0 AND 1000000000
                                  THEN (e.seg ->> 'start')::numeric END END AS start
            FROM jsonb_array_elements(
                     CASE WHEN m.in_transcript AND jsonb_typeof(t.segments) = 'array'
                          THEN t.segments ELSE '[]'::jsonb END
                 ) WITH ORDINALITY AS e(seg, ord)
        ),
        hits AS (
            SELECT segs.*, to_tsvector(page.cfg, segs.text) @@ r.own AS hit FROM segs
        ),
        adj AS (
            SELECT hits.*, lead(hit) OVER o AS nhit, lead(text) OVER o AS ntext
            FROM hits WINDOW o AS (ORDER BY ord)
        ),
        singles AS (
            SELECT ord, start, text AS wtext FROM adj WHERE hit AND start IS NOT NULL
        ),
        pairs AS (
            SELECT ord, start, text || ' ' || ntext AS wtext,
                   ord - row_number() OVER (ORDER BY ord) AS run
            FROM adj
            WHERE NOT hit AND nhit IS NOT NULL AND NOT nhit
              AND text <> '' AND ntext <> '' AND start IS NOT NULL
              AND to_tsvector(page.cfg, text || ' ' || ntext) @@ r.own
        ),
        windows AS (
            SELECT ord, start, wtext FROM singles
            UNION ALL
            SELECT ord, start, wtext FROM (
                SELECT pairs.*, row_number() OVER (PARTITION BY run ORDER BY ord) AS k FROM pairs
            ) AS kept WHERE k %% 2 = 1
        ),
        ranked AS (
            SELECT windows.*, row_number() OVER (
                       ORDER BY ts_rank(to_tsvector(page.cfg, wtext), r.own) DESC, start, ord
                   ) AS pick
            FROM windows
        )
        SELECT ord, start,
               ts_headline(
                   page.cfg,
                   translate(translate(wtext, {_STRIPPED}, ''), '<>&', {_PROTECTED}),
                   r.own,
                   'StartSel=' || chr(1) || ', StopSel=' || chr(2) || ', {WINDOW_OPTIONS}'
               ) AS headline
        FROM ranked WHERE pick <= 3
    ) AS top
"""

#: The queries one row is checked against (#112). ``own`` is the query in the
#: row's transcript configuration and is only ever applied to the transcript
#: (weight D); ``en`` is the English query and is only applied to the title and
#: analysis (weights A, B, C), which are always English. ``rq`` is their OR,
#: used only to rank a row that has already been admitted.
ROW_QUERY_SQL = """
    SELECT CASE WHEN k.cfg = 'pg_catalog.english'::regconfig THEN q.en
                ELSE websearch_to_tsquery(k.cfg, %(q)s) END AS own,
           q.en AS en,
           CASE WHEN k.cfg = 'pg_catalog.english'::regconfig THEN q.en
                ELSE websearch_to_tsquery(k.cfg, %(q)s) || q.en END AS rq
"""

SEARCH_SQL = f"""
WITH q AS (
    SELECT websearch_to_tsquery('english', %(q)s) AS en,
           transcript_search_tsquery(%(q)s) AS any_config
),
page AS (
    SELECT v.video_id, v.title, v.channel_id, c.title AS channel_title,
           v.published_at, v.duration_sec, v.unavailable, sd.fts,
           bt.id AS transcript_id, k.cfg,
           ts_rank_cd('{RANK_WEIGHTS}', sd.fts, r.rq, 1) AS rank
    FROM q
    JOIN search_docs sd ON sd.fts @@ q.any_config
    JOIN videos v ON v.video_id = sd.video_id
    LEFT JOIN channels c ON c.channel_id = v.channel_id
    LEFT JOIN LATERAL (
        SELECT best.id, best.language FROM transcripts best
        WHERE best.video_id = sd.video_id
        ORDER BY transcript_rank(best.source), best.id
        LIMIT 1
    ) bt ON true
    CROSS JOIN LATERAL (SELECT transcript_ts_config(bt.language) AS cfg) k
    CROSS JOIN LATERAL ({ROW_QUERY_SQL}) r
    WHERE CASE WHEN k.cfg = 'pg_catalog.english'::regconfig THEN sd.fts @@ r.en
               ELSE (sd.fts @@ r.own AND ts_filter(sd.fts, '{{d}}') @@ r.own)
                    OR (sd.fts @@ r.en AND ts_filter(sd.fts, '{{a,b,c}}') @@ r.en) END
    ORDER BY rank DESC, v.published_at DESC NULLS LAST, v.video_id ASC
    OFFSET %(offset)s LIMIT %(limit)s
)
SELECT page.video_id, page.title, page.channel_id, page.channel_title,
       page.published_at, page.duration_sec, page.unavailable, t.source,
       m.in_title, m.in_analysis, m.in_transcript, w.windows,
       CASE WHEN m.in_transcript AND jsonb_array_length(w.windows) = 0 THEN ts_headline(
           page.cfg,
           translate(translate(t.full_text, {_STRIPPED}, ''), '<>&', {_PROTECTED}),
           r.own,
           'StartSel=' || chr(1) || ', StopSel=' || chr(2)
               || ', FragmentDelimiter=' || chr(3) || ', {HEADLINE_OPTIONS}'
       ) END
FROM page
CROSS JOIN q
CROSS JOIN LATERAL ({ROW_QUERY_SQL.replace('k.cfg', 'page.cfg')}) r
CROSS JOIN LATERAL (
    SELECT tsquery_positive_lexemes(r.en) AS en, tsquery_positive_lexemes(r.own) AS own
) lx
CROSS JOIN LATERAL (
    SELECT tsvector_to_array(ts_filter(page.fts, '{{a}}')) && lx.en AS in_title,
           tsvector_to_array(ts_filter(page.fts, '{{b,c}}')) && lx.en AS in_analysis,
           tsvector_to_array(ts_filter(page.fts, '{{d}}')) && lx.own AS in_transcript
) m
LEFT JOIN transcripts t ON t.id = page.transcript_id
CROSS JOIN LATERAL ({WINDOWS_SQL}) w
ORDER BY page.rank DESC, page.published_at DESC NULLS LAST, page.video_id ASC
"""


@dataclass(frozen=True, slots=True)
class Span:
    """A piece of excerpt text; ``match`` is true for a highlighted word."""

    text: str
    match: bool


@dataclass(frozen=True, slots=True)
class Excerpt:
    """One excerpt: its spans and where in the video it was said (#110).

    ``start_sec`` is the start of the transcript segment the match is in, or
    ``None`` for a ``full_text`` fallback excerpt.
    """

    spans: tuple[Span, ...]
    start_sec: float | None = None


@dataclass(frozen=True, slots=True)
class SearchHit:
    video_id: str
    title: str | None
    channel_id: str | None
    channel_title: str | None
    published_at: datetime | None
    duration_sec: int | None
    unavailable: str | None
    transcript_source: str | None
    #: Where the query matched, in the fixed order title, analysis, transcript.
    matched_in: tuple[Literal["title", "analysis", "transcript"], ...]
    excerpts: tuple[Excerpt, ...]


@dataclass(frozen=True, slots=True)
class SearchPage:
    results: tuple[SearchHit, ...]
    has_more: bool


def search_transcripts(
    conn: psycopg.Connection[Any], query: str, limit: int, offset: int
) -> SearchPage:
    """One page of videos whose preferred transcript matches ``query``.

    Fetches ``limit + 1`` rows to compute ``has_more``; never counts every
    match. ``query`` must not contain NUL (``ValueError``).
    """
    if "\x00" in query:
        raise ValueError("query must not contain NUL")
    if not _has_positive_term(conn, query):
        return SearchPage((), False)
    rows = conn.execute(
        SEARCH_SQL, {"q": query, "offset": offset, "limit": limit + 1}
    ).fetchall()
    hits = tuple(_hit(row) for row in rows[:limit])
    return SearchPage(hits, has_more=len(rows) > limit)


def _has_positive_term(conn: psycopg.Connection[Any], query: str) -> bool:
    # querytree() is '' for no lexemes and 'T' when nothing indexable is left
    # (only exclusions): both would otherwise scan or list the corpus.
    row = conn.execute(
        "SELECT querytree(transcript_search_tsquery(%s))", (query,)
    ).fetchone()
    return row is not None and row[0] not in ("", "T")


def _hit(row: tuple[Any, ...]) -> SearchHit:
    (video_id, title, channel_id, channel_title, published_at, duration_sec, unavailable,
     source, in_title, in_analysis, in_transcript, windows, headline) = row
    matched_in: list[Literal["title", "analysis", "transcript"]] = []
    if in_title:
        matched_in.append("title")
    if in_analysis:
        matched_in.append("analysis")
    if in_transcript:
        matched_in.append("transcript")
    return SearchHit(
        video_id=video_id,
        title=title,
        channel_id=channel_id,
        channel_title=channel_title,
        published_at=published_at,
        duration_sec=duration_sec,
        unavailable=unavailable,
        transcript_source=source,
        matched_in=tuple(matched_in),
        excerpts=_window_excerpts(windows) or _excerpts(headline or ""),
    )


def _window_excerpts(windows: Any) -> tuple[Excerpt, ...]:
    excerpts = []
    for start, headline in windows or ():
        spans = next(
            (spans for part in (headline or "").split(_DELIM) if (spans := _spans(part))), ()
        )
        if spans:
            excerpts.append(Excerpt(spans, float(start)))
    return tuple(excerpts[:3])


def _excerpts(headline: str) -> tuple[Excerpt, ...]:
    fragments = []
    for fragment in headline.split(_DELIM):
        spans = _spans(fragment)
        if spans:
            fragments.append(Excerpt(spans))
    return tuple(fragments[:3])


def _spans(fragment: str) -> tuple[Span, ...]:
    spans: list[Span] = []
    match = False
    text = ""
    for char in fragment:
        if char in (_START, _STOP):
            if text:
                spans.append(Span(text.translate(_RESTORE), match))
            text = ""
            match = char == _START
        else:
            text += char
    if text:
        spans.append(Span(text.translate(_RESTORE), match))
    return tuple(spans)
