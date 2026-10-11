"""Full-text search over videos' titles, analyses and transcripts (#43, #111).

``search_transcripts`` matches ``search_docs.fts`` (GIN index
``search_docs_fts_idx``) with ``websearch_to_tsquery('english', q)``. That
vector is materialised per video by the database (migration 0010): title (weight
A), latest analysis TL;DR (B), topics and claims (C) and the preferred
transcript (D). Only a video's preferred transcript counts: the one with the
lowest ``transcript_rank(source)`` (D2), the same one ``get_best_transcript``
returns. Results are ranked by ``ts_rank_cd`` with explicit weights and length
normalisation, then ``published_at`` (newest first, nulls last), then
``video_id``. ``matched_in`` says which sources contain a query lexeme,
derived from the weights stored in the vector (``ts_filter``), for the page's
rows only.

``q`` is only ever a bound parameter, and ``websearch_to_tsquery`` never
raises a syntax error. A query with no positive term (only stop words, or
only exclusions such as ``-dogs``, which would otherwise match nearly every
transcript) returns no results without running the search.

Excerpts come from ``ts_headline``, evaluated in the outer query over the
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

# chr(1)..chr(3) are the markers; chr(57344)..chr(57346) are U+E000..U+E002.
_STRIPPED = "chr(1) || chr(2) || chr(3) || chr(57344) || chr(57345) || chr(57346)"
_PROTECTED = "chr(57344) || chr(57345) || chr(57346)"

#: ``ts_rank_cd`` weights in the order D, C, B, A (title A > TL;DR B > topics
#: and claims C > transcript D).
RANK_WEIGHTS = "{0.1, 0.2, 0.4, 1.0}"

SEARCH_SQL = f"""
WITH q AS (
    SELECT websearch_to_tsquery('english', %(q)s) AS query,
           tsquery_positive_lexemes(websearch_to_tsquery('english', %(q)s)) AS lexemes
),
page AS (
    SELECT v.video_id, v.title, v.channel_id, c.title AS channel_title,
           v.published_at, v.duration_sec, v.unavailable, sd.fts,
           ts_rank_cd('{RANK_WEIGHTS}', sd.fts, q.query, 1) AS rank
    FROM q
    JOIN search_docs sd ON sd.fts @@ q.query
    JOIN videos v ON v.video_id = sd.video_id
    LEFT JOIN channels c ON c.channel_id = v.channel_id
    ORDER BY rank DESC, v.published_at DESC NULLS LAST, v.video_id ASC
    OFFSET %(offset)s LIMIT %(limit)s
)
SELECT page.video_id, page.title, page.channel_id, page.channel_title,
       page.published_at, page.duration_sec, page.unavailable, t.source,
       m.in_title, m.in_analysis, m.in_transcript,
       CASE WHEN m.in_transcript THEN ts_headline(
           'english',
           translate(translate(t.full_text, {_STRIPPED}, ''), '<>&', {_PROTECTED}),
           q.query,
           'StartSel=' || chr(1) || ', StopSel=' || chr(2)
               || ', FragmentDelimiter=' || chr(3) || ', {HEADLINE_OPTIONS}'
       ) END
FROM page
CROSS JOIN q
CROSS JOIN LATERAL (
    SELECT tsvector_to_array(ts_filter(page.fts, '{{a}}')) && q.lexemes AS in_title,
           tsvector_to_array(ts_filter(page.fts, '{{b,c}}')) && q.lexemes AS in_analysis,
           tsvector_to_array(ts_filter(page.fts, '{{d}}')) && q.lexemes AS in_transcript
) m
LEFT JOIN LATERAL (
    SELECT best.source, best.full_text FROM transcripts best
    WHERE best.video_id = page.video_id
    ORDER BY transcript_rank(best.source), best.id
    LIMIT 1
) t ON true
ORDER BY page.rank DESC, page.published_at DESC NULLS LAST, page.video_id ASC
"""


@dataclass(frozen=True, slots=True)
class Span:
    """A piece of excerpt text; ``match`` is true for a highlighted word."""

    text: str
    match: bool


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
    excerpts: tuple[tuple[Span, ...], ...]


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
        "SELECT querytree(websearch_to_tsquery('english', %s))", (query,)
    ).fetchone()
    return row is not None and row[0] not in ("", "T")


def _hit(row: tuple[Any, ...]) -> SearchHit:
    (video_id, title, channel_id, channel_title, published_at, duration_sec, unavailable,
     source, in_title, in_analysis, in_transcript, headline) = row
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
        excerpts=_excerpts(headline or ""),
    )


def _excerpts(headline: str) -> tuple[tuple[Span, ...], ...]:
    fragments = []
    for fragment in headline.split(_DELIM):
        spans = _spans(fragment)
        if spans:
            fragments.append(spans)
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
