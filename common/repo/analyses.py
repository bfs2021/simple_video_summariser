"""Analysis repository functions (issue #14, architecture.md §6, §7.4).

Unlike the rest of ``common/repo/``, this module's contract is called out
explicitly by issue #14: ``save_analysis`` never commits. The caller's
transaction decides, so #30 can persist a whole analysis - plus everything
else in that job - atomically.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from common.models import Analysis, AnalysisRun, Claim, Quote, Topic
from common.repo._hygiene import clean_json, clean_text


def save_analysis(conn: psycopg.Connection[Any], analysis: Analysis) -> int:
    """Insert ``analysis`` plus its topics, claims and quotes; return the id.

    Does not commit (see module docstring).
    """
    row = conn.execute(
        """
        INSERT INTO analyses
            (video_id, transcript_id, chunk_strategy, model, prompt_version,
             tldr, speaker_roster, input_tokens, output_tokens, cost_usd,
             duration_ms, speakers_coerced)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            analysis.video_id,
            analysis.transcript_id,
            analysis.chunk_strategy,
            analysis.model,
            analysis.prompt_version,
            clean_text(analysis.tldr),
            _to_jsonb(analysis.speaker_roster),
            analysis.input_tokens,
            analysis.output_tokens,
            analysis.cost_usd,
            analysis.duration_ms,
            analysis.speakers_coerced,
        ),
    ).fetchone()
    assert row is not None
    analysis_id = int(row[0])

    for topic in analysis.topics:
        conn.execute(
            """
            INSERT INTO topics (analysis_id, seq, title, summary, start_sec)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                analysis_id,
                topic.seq,
                clean_text(topic.title),
                clean_text(topic.summary),
                topic.start_sec,
            ),
        )

    for claim in analysis.claims:
        conn.execute(
            """
            INSERT INTO claims
                (analysis_id, text, speaker, start_sec, confidence, source_chunk_seq)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                analysis_id,
                clean_text(claim.text),
                clean_text(claim.speaker),
                claim.start_sec,
                claim.confidence,
                claim.source_chunk_seq,
            ),
        )

    for quote in analysis.quotes:
        conn.execute(
            """
            INSERT INTO quotes (analysis_id, text, speaker, start_sec, source_chunk_seq)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                analysis_id,
                clean_text(quote.text),
                clean_text(quote.speaker),
                quote.start_sec,
                quote.source_chunk_seq,
            ),
        )

    return analysis_id


def analysis_exists(
    conn: psycopg.Connection[Any],
    video_id: str,
    transcript_id: int,
    chunk_strategy: str,
    model: str,
    prompt_version: str,
) -> bool:
    """Whether an analysis with exactly this five-part key is already stored."""
    row = conn.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM analyses
            WHERE video_id = %s AND transcript_id = %s AND chunk_strategy = %s
              AND model = %s AND prompt_version = %s
        )
        """,
        (video_id, transcript_id, chunk_strategy, model, prompt_version),
    ).fetchone()
    assert row is not None
    return bool(row[0])


_ANALYSIS_COLUMNS = """
    id, video_id, transcript_id, chunk_strategy, model, prompt_version, tldr,
    speaker_roster, input_tokens, output_tokens, cost_usd, duration_ms,
    created_at, speakers_coerced
"""


def latest_analysis(conn: psycopg.Connection[Any], video_id: str) -> Analysis | None:
    """The newest analysis for ``video_id``, with its children populated.

    Topics ordered by ``seq``, claims and quotes by ``start_sec`` (nulls
    last, then ``id``); ties on ``created_at`` broken by ``id DESC``. ``None`` when the video has no
    analysis.
    """
    row = conn.execute(
        f"""
        SELECT {_ANALYSIS_COLUMNS}
        FROM analyses
        WHERE video_id = %s
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (video_id,),
    ).fetchone()
    return None if row is None else _with_children(conn, row)


def get_analysis(
    conn: psycopg.Connection[Any], video_id: str, analysis_id: int
) -> Analysis | None:
    """The analysis ``analysis_id``, only if it belongs to ``video_id`` (#109).

    Same shape and child ordering as ``latest_analysis`` (one shared loader).
    ``None`` for an unknown id and for another video's id. Read-only.
    """
    row = conn.execute(
        f"SELECT {_ANALYSIS_COLUMNS} FROM analyses WHERE id = %s AND video_id = %s",
        (analysis_id, video_id),
    ).fetchone()
    return None if row is None else _with_children(conn, row)


def _with_children(conn: psycopg.Connection[Any], row: tuple[Any, ...]) -> Analysis:
    """An ``Analysis`` from an ``_ANALYSIS_COLUMNS`` row, children loaded and ordered."""
    (
        analysis_id,
        video_id,
        transcript_id,
        chunk_strategy,
        model,
        prompt_version,
        tldr,
        speaker_roster,
        input_tokens,
        output_tokens,
        cost_usd,
        duration_ms,
        created_at,
        speakers_coerced,
    ) = row

    topics = conn.execute(
        """
        SELECT id, seq, title, summary, start_sec FROM topics
        WHERE analysis_id = %s ORDER BY seq
        """,
        (analysis_id,),
    ).fetchall()
    claims = conn.execute(
        """
        SELECT id, text, speaker, start_sec, confidence, source_chunk_seq FROM claims
        WHERE analysis_id = %s ORDER BY start_sec ASC NULLS LAST, id ASC
        """,
        (analysis_id,),
    ).fetchall()
    quotes = conn.execute(
        """
        SELECT id, text, speaker, start_sec, source_chunk_seq FROM quotes
        WHERE analysis_id = %s ORDER BY start_sec ASC NULLS LAST, id ASC
        """,
        (analysis_id,),
    ).fetchall()

    return Analysis(
        id=int(analysis_id),
        video_id=video_id,
        transcript_id=int(transcript_id),
        chunk_strategy=chunk_strategy,
        model=model,
        prompt_version=prompt_version,
        tldr=tldr,
        speaker_roster=speaker_roster,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=float(cost_usd) if cost_usd is not None else None,
        duration_ms=duration_ms,
        speakers_coerced=speakers_coerced,
        created_at=created_at,
        topics=tuple(_topic(item) for item in topics),
        claims=tuple(_claim(item) for item in claims),
        quotes=tuple(_quote(item) for item in quotes),
    )


def _to_jsonb(value: dict[str, object] | None) -> Jsonb | None:
    return None if value is None else Jsonb(clean_json(value))


def _topic(row: tuple[Any, ...]) -> Topic:
    topic_id, seq, title, summary, start_sec = row
    return Topic(
        seq=seq,
        title=title,
        summary=summary,
        start_sec=float(start_sec) if start_sec is not None else None,
        id=int(topic_id),
    )


def _claim(row: tuple[Any, ...]) -> Claim:
    claim_id, text, speaker, start_sec, confidence, source_chunk_seq = row
    return Claim(
        text=text,
        speaker=speaker,
        start_sec=float(start_sec) if start_sec is not None else None,
        confidence=confidence,
        source_chunk_seq=source_chunk_seq,
        id=int(claim_id),
    )


def _quote(row: tuple[Any, ...]) -> Quote:
    quote_id, text, speaker, start_sec, source_chunk_seq = row
    return Quote(
        text=text,
        speaker=speaker,
        start_sec=float(start_sec) if start_sec is not None else None,
        source_chunk_seq=source_chunk_seq,
        id=int(quote_id),
    )


_RUN_COLUMNS = """
    a.id, a.video_id, a.transcript_id, a.chunk_strategy, a.model, a.prompt_version,
    a.tldr, a.speaker_roster, a.input_tokens, a.output_tokens, a.cost_usd,
    a.duration_ms, a.created_at, a.speakers_coerced, t.source
"""


def latest_analysis_run(conn: psycopg.Connection[Any], video_id: str) -> AnalysisRun | None:
    """The newest analysis of ``video_id`` with its children and transcript source (#42)."""
    runs = _runs(conn, video_id, offset=0, limit=1)
    return runs[0] if runs else None


def analysis_run_for(
    conn: psycopg.Connection[Any], video_id: str, model: str, prompt_version: str
) -> AnalysisRun | None:
    """The newest analysis of ``video_id`` made with ``model`` and ``prompt_version`` (#62).

    Whatever transcript or chunk strategy it used. Ordered by ``created_at DESC,
    id DESC``; children and their order are exactly those of ``latest_analysis_run``,
    and the number of statements is fixed (one per table, none per child).
    """
    runs = _runs(conn, video_id, offset=0, limit=1, model=model, prompt_version=prompt_version)
    return runs[0] if runs else None


def list_analysis_runs(
    conn: psycopg.Connection[Any], video_id: str, *, offset: int, limit: int
) -> tuple[tuple[AnalysisRun, ...], int]:
    """A page of ``video_id``'s analyses, newest first, and the total count (#42).

    Ordered by ``created_at DESC, id DESC``. Children are loaded with one
    statement per child table for the whole page, never one per analysis;
    topics by ``seq``, claims and quotes by ``start_sec`` (nulls last, then ``id``).
    """
    total_row = conn.execute(
        "SELECT count(*) FROM analyses WHERE video_id = %s", (video_id,)
    ).fetchone()
    total = int(total_row[0]) if total_row else 0
    return _runs(conn, video_id, offset=offset, limit=limit), total


def _runs(
    conn: psycopg.Connection[Any],
    video_id: str,
    *,
    offset: int,
    limit: int,
    model: str | None = None,
    prompt_version: str | None = None,
) -> tuple[AnalysisRun, ...]:
    # ``model`` and ``prompt_version`` are optional filters; ``NULL`` means "any".
    rows = conn.execute(
        f"""
        SELECT {_RUN_COLUMNS}
        FROM analyses a JOIN transcripts t ON t.id = a.transcript_id
        WHERE a.video_id = %s
          AND (%s::text IS NULL OR a.model = %s)
          AND (%s::text IS NULL OR a.prompt_version = %s)
        ORDER BY a.created_at DESC, a.id DESC
        OFFSET %s LIMIT %s
        """,
        (video_id, model, model, prompt_version, prompt_version, offset, limit),
    ).fetchall()
    ids = [int(row[0]) for row in rows]
    topics: dict[int, list[Topic]] = {i: [] for i in ids}
    claims: dict[int, list[Claim]] = {i: [] for i in ids}
    quotes: dict[int, list[Quote]] = {i: [] for i in ids}
    if ids:
        for row in conn.execute(
            """
            SELECT analysis_id, id, seq, title, summary, start_sec FROM topics
            WHERE analysis_id = ANY(%s) ORDER BY analysis_id, seq, id
            """,
            (ids,),
        ).fetchall():
            topics[int(row[0])].append(_topic(row[1:]))
        for row in conn.execute(
            """
            SELECT analysis_id, id, text, speaker, start_sec, confidence, source_chunk_seq
            FROM claims WHERE analysis_id = ANY(%s)
            ORDER BY analysis_id, start_sec ASC NULLS LAST, id ASC
            """,
            (ids,),
        ).fetchall():
            claims[int(row[0])].append(_claim(row[1:]))
        for row in conn.execute(
            """
            SELECT analysis_id, id, text, speaker, start_sec, source_chunk_seq
            FROM quotes WHERE analysis_id = ANY(%s)
            ORDER BY analysis_id, start_sec ASC NULLS LAST, id ASC
            """,
            (ids,),
        ).fetchall():
            quotes[int(row[0])].append(_quote(row[1:]))
    return tuple(
        AnalysisRun(
            analysis=_analysis(row[:14], topics[int(row[0])], claims[int(row[0])],
                               quotes[int(row[0])]),
            transcript_source=row[14],
        )
        for row in rows
    )


def _analysis(
    row: tuple[Any, ...], topics: list[Topic], claims: list[Claim], quotes: list[Quote]
) -> Analysis:
    (
        analysis_id, video_id, transcript_id, chunk_strategy, model, prompt_version, tldr,
        speaker_roster, input_tokens, output_tokens, cost_usd, duration_ms, created_at,
        speakers_coerced,
    ) = row
    return Analysis(
        id=int(analysis_id),
        video_id=video_id,
        transcript_id=int(transcript_id),
        chunk_strategy=chunk_strategy,
        model=model,
        prompt_version=prompt_version,
        tldr=tldr,
        speaker_roster=speaker_roster,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=float(cost_usd) if cost_usd is not None else None,
        duration_ms=duration_ms,
        speakers_coerced=speakers_coerced,
        created_at=created_at,
        topics=tuple(topics),
        claims=tuple(claims),
        quotes=tuple(quotes),
    )


def token_usage_by_model(conn: psycopg.Connection[Any]) -> dict[str, tuple[int, int]]:
    """``model -> (SUM(input_tokens), SUM(output_tokens))`` over all analyses (#60)."""
    rows = conn.execute(
        "SELECT model, SUM(input_tokens), SUM(output_tokens) FROM analyses GROUP BY model"
    ).fetchall()
    return {str(model): (int(inp), int(out)) for model, inp, out in rows}


def total_cost_usd(conn: psycopg.Connection[Any]) -> float | None:
    """``SUM(cost_usd)`` over analyses with a known price; ``None`` when there are none (#60)."""
    row = conn.execute("SELECT SUM(cost_usd) FROM analyses WHERE cost_usd IS NOT NULL").fetchone()
    assert row is not None
    return None if row[0] is None else float(row[0])


def total_speaker_coercions(conn: psycopg.Connection[Any]) -> int:
    """``SUM(speakers_coerced)`` over analyses that recorded it; 0 when none (#60)."""
    row = conn.execute(
        "SELECT COALESCE(SUM(speakers_coerced), 0) FROM analyses WHERE speakers_coerced IS NOT NULL"
    ).fetchone()
    assert row is not None
    return int(row[0])


def reanalysis_backlog_count(conn: psycopg.Connection[Any], prompt_version: str) -> int:
    """Videos with a non-blank transcript and no analysis at ``prompt_version`` (#60).

    The first two predicates of the planner sweep's candidate query
    (``services/planner/sweep.py``), as a count; jobs are not considered.
    """
    row = conn.execute(
        r"""
        SELECT count(*) FROM videos v
        WHERE EXISTS (
                SELECT 1 FROM transcripts t
                WHERE t.video_id = v.video_id AND t.full_text ~ '\S')
          AND NOT EXISTS (
                SELECT 1 FROM analyses a
                WHERE a.video_id = v.video_id AND a.prompt_version = %s)
        """,
        (prompt_version,),
    ).fetchone()
    assert row is not None
    return int(row[0])


COST_SAMPLE_SIZE = 20


def median_cost_per_char(conn: psycopg.Connection[Any], model: str) -> tuple[float | None, int]:
    """Median ``cost_usd`` per transcript character, newest 20 analyses of ``model`` (#117).

    Only rows with ``cost_usd > 0`` (``NULL`` and zero are ignored, like #102's
    unknown price) whose transcript text is not empty count. Returns
    ``(median, samples)``; with none ``(None, 0)``. A read only: no lock.
    """
    rows = conn.execute(
        """
        SELECT a.cost_usd::float8 / length(t.full_text)::float8
        FROM analyses a JOIN transcripts t ON t.id = a.transcript_id
        WHERE a.cost_usd IS NOT NULL AND a.cost_usd > 0
          AND a.model = %s AND length(t.full_text) > 0
        ORDER BY a.created_at DESC, a.id DESC
        LIMIT %s
        """,
        (model, COST_SAMPLE_SIZE),
    ).fetchall()
    values = [float(row[0]) for row in rows if math.isfinite(float(row[0])) and row[0] > 0]
    if not values:
        return None, 0
    return statistics.median(values), len(values)
