"""The ``analyze`` job handler (issue #30, architecture.md 7.4).

Turns a video's best stored transcript into one persisted analysis: a roster
pass, a map pass over the chunks, a reduce pass, dedupe across the overlap
seams, and one transaction that writes it all. Everything the summarizer
returns is untrusted and is re-checked here before it reaches the database,
whatever the adapter already did (the port may be a fake, or a future adapter
that skips the #22 parser).

The handler writes no SQL: the database is reached through ``common.repo``.
No transaction is open while a summarizer call runs - each phase commits
before the next slow call - so an LLM call that takes minutes never holds a
snapshot or a lock.
"""

from __future__ import annotations

import math
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import psycopg

from adapters.summarize.prompt_loader import format_timestamped
from common.chunking import chunk_segments, chunk_strategy
from common.config import Settings
from common.errors import BugError, LLMInvalidOutputError
from common.models import (
    Analysis,
    Chunk,
    ChunkAnalysis,
    Claim,
    Quote,
    Roster,
    Segment,
    Summarizer,
    Topic,
)
from common.queue import Job, analyze_dedupe_key
from common.repo.analyses import analysis_exists, save_analysis
from common.repo.transcripts import get_best_transcript, get_chunks, save_chunks
from common.repo.videos import get_video_meta
from common.worker import JobContext

#: Segments starting within this many seconds of the first one feed the roster pass (D4b).
OPENING_SEC = 600

UNKNOWN = "unknown"
_CONFIDENCES = frozenset({"high", "medium", "low"})


# ------------------------------------------------------------------ pure logic


def normalize_key(text: str) -> str:
    """The dedupe key: NFKC, casefold, whitespace collapsed, edge punctuation stripped."""
    collapsed = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    start, end = 0, len(collapsed)
    while start < end and _is_edge(collapsed[start]):
        start += 1
    while end > start and _is_edge(collapsed[end - 1]):
        end -= 1
    return collapsed[start:end]


def _is_edge(char: str) -> bool:
    return char.isspace() or unicodedata.category(char).startswith("P")


def _finite(value: float | None) -> float:
    return value if value is not None and math.isfinite(value) else math.inf


def _preference(item: Claim | Quote) -> tuple[bool, float, float]:
    """Sorts the item to keep first: a named speaker, lowest chunk, earliest start."""
    seq = item.source_chunk_seq
    return (
        item.speaker == UNKNOWN,
        math.inf if seq is None else float(seq),
        _finite(item.start_sec),
    )


def dedupe[T: (Claim, Quote)](items: Sequence[T]) -> list[T]:
    """One item per normalized text; items whose normalized text is empty are dropped.

    Among duplicates the kept one has a named speaker over ``unknown``, then the
    lowest ``source_chunk_seq``, then the earliest ``start_sec``; ties keep the
    first seen. The kept items stay in their input order.
    """
    best: dict[str, int] = {}
    for index, item in enumerate(items):
        key = normalize_key(item.text)
        if not key:
            continue
        current = best.get(key)
        if current is None or _preference(item) < _preference(items[current]):
            best[key] = index
    return [items[index] for index in sorted(best.values())]


def opening_text(segments: Sequence[Segment]) -> str:
    """Text of the segments starting within ``OPENING_SEC`` of the first one.

    Measured from when speech starts, so a transcript whose first words come at
    700 s still has an opening.
    """
    spoken = sorted((s for s in segments if s.text.strip()), key=lambda s: s.start)
    if not spoken:
        return ""
    limit = spoken[0].start + OPENING_SEC
    return " ".join(s.text.strip() for s in spoken if s.start < limit)


def timestamped_chunk(chunk: Chunk, segments: Sequence[Segment]) -> Chunk:
    """``chunk`` with its text replaced by the timestamped lines of the segments in its span.

    A segment is in the span when it overlaps ``[start_sec, end_sec)``. The stored
    ``Chunk.text`` stays plain; only the copy given to the summarizer changes. If no
    segment matches (not expected for stored chunks), the plain text is kept.
    """
    inside = [
        s
        for s in segments
        if s.start < chunk.end_sec and (s.end > chunk.start_sec or s.start >= chunk.start_sec)
    ]
    text = format_timestamped(inside)
    return replace(chunk, text=text) if text else chunk


def _clamp(value: float | None, low: float, high: float) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return min(max(float(value), low), high)


def sanitize_chunk(
    chunk: Chunk, analysis: ChunkAnalysis, roster: Roster
) -> tuple[list[Topic], list[Claim], list[Quote], int]:
    """Re-check one chunk's output: speakers, ``start_sec``, confidence, source chunk.

    Returns the topics, claims and quotes, plus the number of speakers coerced to
    ``unknown`` (the adapter's own count included).
    """
    names = {speaker.name for speaker in roster.speakers}
    coerced = analysis.speaker_coercions

    def speaker_of(speaker: str) -> str:
        nonlocal coerced
        if speaker == UNKNOWN or speaker in names:
            return speaker
        coerced += 1
        return UNKNOWN

    def start_of(value: float | None) -> float | None:
        return _clamp(value, chunk.start_sec, chunk.end_sec)

    topics = [replace(t, start_sec=start_of(t.start_sec)) for t in analysis.topics]
    claims = [
        Claim(
            text=c.text,
            speaker=speaker_of(c.speaker),
            start_sec=start_of(c.start_sec),
            confidence=c.confidence if c.confidence in _CONFIDENCES else None,
            source_chunk_seq=chunk.seq,
        )
        for c in analysis.claims
    ]
    quotes = [
        Quote(
            text=q.text,
            speaker=speaker_of(q.speaker),
            start_sec=start_of(q.start_sec),
            source_chunk_seq=chunk.seq,
        )
        for q in analysis.quotes
    ]
    return topics, claims, quotes, coerced


# --------------------------------------------------------------------- handler


@dataclass(slots=True)
class _Tally:
    """Tokens and spend of the successful calls of one job, from ``take_usage()``.

    ``cost_usd`` follows one rule (#102): ``NULL`` means "no price reported".
    ``total_cost`` is ``None`` when any tallied ``Usage`` with ``calls > 0`` has no
    ``cost_usd`` (a partial sum would understate the spend), and when no ``Usage``
    reported a cost at all. A ``Usage`` with ``calls == 0`` contributes nothing and
    does not force ``None``. So an Ollama run stores ``NULL``, never ``0``; the
    Compare view tells "local" from "unknown price" by ``analyses.model``.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    priced: bool = False
    unpriced: bool = False

    def add(self, summarizer: Summarizer) -> None:
        usage = summarizer.take_usage()
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        if usage.calls > 0:
            if usage.cost_usd is None:
                self.unpriced = True
            else:
                self.cost += usage.cost_usd
                self.priced = True

    @property
    def total_cost(self) -> float | None:
        """The job's spend rounded to ``NUMERIC(10,5)``, or ``None`` (see the class docstring)."""
        if self.unpriced or not self.priced:
            return None
        return round(self.cost, 5)


def make_analyze_handler(
    *,
    connect: Callable[[], psycopg.Connection[Any]],
    summarizer: Summarizer,
    settings: Settings,
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[Job, JobContext], None]:
    """Build the ``analyze`` handler. ``connect`` opens one connection per job."""
    strategy = chunk_strategy(settings.CHUNK_SEC, settings.OVERLAP_SEC)
    expected_key = analyze_dedupe_key(settings.PROMPT_VERSION, summarizer.name)

    def handler(job: Job, ctx: JobContext) -> None:
        started = clock()
        summarizer.take_usage()  # a reused summarizer may hold a failed job's usage
        if job.dedupe_key != expected_key:
            ctx.logger.warning(
                "analyze job dedupe key differs from the configured one",
                job_dedupe_key=job.dedupe_key,
                configured_dedupe_key=expected_key,
            )
        conn = connect()
        try:
            _analyze(conn, job, ctx, started)
        finally:
            conn.close()

    def _analyze(
        conn: psycopg.Connection[Any], job: Job, ctx: JobContext, started: float
    ) -> None:
        video_id = job.video_id
        meta = get_video_meta(conn, video_id)
        if meta is None:
            raise BugError(f"analyze job for video {video_id!r}: no such video")
        transcript = get_best_transcript(conn, video_id)
        if transcript is None:
            raise BugError(f"analyze job for video {video_id!r}: the video has no transcript")

        model = summarizer.model
        if job.payload.get("force") is not True and analysis_exists(
            conn, video_id, transcript.id, strategy, model, settings.PROMPT_VERSION
        ):
            conn.rollback()
            ctx.logger.info(
                "already analyzed", video_id=video_id, transcript_id=transcript.id, model=model
            )
            return

        segments = transcript.segments
        chunks: list[Chunk] = []
        if segments and transcript.full_text.strip():
            chunks = get_chunks(conn, transcript.id, strategy)
            if not chunks:
                generated = chunk_segments(
                    segments, chunk_sec=settings.CHUNK_SEC, overlap_sec=settings.OVERLAP_SEC
                )
                conn.commit()  # so the write below is a transaction of its own
                with conn.transaction():
                    save_chunks(conn, transcript.id, strategy, generated)
                chunks = get_chunks(conn, transcript.id, strategy)
        conn.commit()  # no transaction stays open across the summarizer calls

        def persist(analysis: Analysis, log: dict[str, Any]) -> None:
            with conn.transaction():
                analysis_id = save_analysis(conn, analysis)
            ctx.logger.info("analysis complete", analysis_id=analysis_id, **log)

        def elapsed_ms() -> int:
            return round((clock() - started) * 1000)

        base = Analysis(
            video_id=video_id,
            transcript_id=transcript.id,
            chunk_strategy=strategy,
            model=model,
            prompt_version=settings.PROMPT_VERSION,
            tldr="",
            speaker_roster=Roster(()).to_json(),
            speakers_coerced=0,
        )
        if not chunks:
            empty_ms = elapsed_ms()
            persist(
                replace(base, duration_ms=empty_ms),
                {
                    "chunks": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": None,
                    "duration_ms": empty_ms,
                    "speakers_coerced": 0,
                    "claims_deduped": 0,
                    "quotes_deduped": 0,
                },
            )
            return

        tally = _Tally()

        ctx.check_cancelled()
        try:
            roster = summarizer.derive_roster(meta, opening_text(segments))
            tally.add(summarizer)
        except LLMInvalidOutputError as exc:
            ctx.logger.warning(
                "roster pass gave invalid output; continuing with an empty roster",
                video_id=video_id,
                error=str(exc),
            )
            roster = Roster(())
            summarizer.take_usage()  # the failed call adds nothing

        partials: list[ChunkAnalysis] = []
        topics: list[Topic] = []
        claims: list[Claim] = []
        quotes: list[Quote] = []
        coerced = 0
        for chunk in sorted(chunks, key=lambda c: c.seq):
            ctx.check_cancelled()
            partial = summarizer.analyze_chunk(timestamped_chunk(chunk, segments), roster, meta)
            tally.add(summarizer)
            partials.append(partial)
            chunk_topics, chunk_claims, chunk_quotes, chunk_coerced = sanitize_chunk(
                chunk, partial, roster
            )
            topics += chunk_topics
            claims += chunk_claims
            quotes += chunk_quotes
            coerced += chunk_coerced

        ctx.check_cancelled()
        tldr = summarizer.reduce(partials, meta).strip()
        tally.add(summarizer)
        if not tldr:
            raise LLMInvalidOutputError(f"reduce returned an empty summary for video {video_id!r}")

        kept_claims = dedupe(claims)
        kept_quotes = dedupe(quotes)
        duration_ms = elapsed_ms()
        persist(
            replace(
                base,
                tldr=tldr,
                speaker_roster=roster.to_json(),
                input_tokens=tally.input_tokens,
                output_tokens=tally.output_tokens,
                cost_usd=tally.total_cost,
                duration_ms=duration_ms,
                speakers_coerced=coerced,
                topics=tuple(replace(t, seq=i) for i, t in enumerate(topics)),
                claims=tuple(kept_claims),
                quotes=tuple(kept_quotes),
            ),
            {
                "chunks": len(chunks),
                "input_tokens": tally.input_tokens,
                "output_tokens": tally.output_tokens,
                "cost_usd": tally.total_cost,
                "duration_ms": duration_ms,
                "speakers_coerced": coerced,
                "claims_deduped": len(claims) - len(kept_claims),
                "quotes_deduped": len(quotes) - len(kept_quotes),
            },
        )

    return handler
