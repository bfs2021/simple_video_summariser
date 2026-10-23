"""Response models for the read and ops routes (issues #42, #44).

They are the public contract #46 generates TypeScript types from, so every
field is declared explicitly. Datetimes are always serialised in UTC.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, cast

from pydantic import AfterValidator, BaseModel, Field, JsonValue, ValidationError

from common.models import (
    Analysis,
    AnalysisRun,
    ChannelPage,
    TranscriptInfo,
    TranscriptPage,
    VideoSummary,
)
from common.progress import estimate_effort
from common.repo.backfill_requests import BackfillRequestRow
from common.repo.jobs import JobRow
from common.repo.search import SearchPage as RepoSearchPage

UtcDatetime = Annotated[datetime, AfterValidator(lambda value: value.astimezone(UTC))]
VideoStatus = Literal["done", "processing", "unavailable", "failed", "idle"]


class HealthOut(BaseModel):
    """``GET /healthz`` 200 body: liveness plus queue depth by kind and state."""

    status: str
    queue: dict[str, dict[str, int]]


class ActiveJobOut(BaseModel):
    id: int
    kind: str
    state: str


class LastFailureOut(BaseModel):
    job_id: int
    kind: str
    error_class: str | None
    finished_at: UtcDatetime | None


class ProgressOut(BaseModel):
    """Progress of a transcribe job (#245).

    A running job carries every figure. A pending one carries only the estimate
    (``estimated_sec``, ``eta_sec``, ``duration_sec``, ``estimate_source``); the
    rest are ``None``. ``estimate_source`` is ``"default"`` when there was no RTF
    history, so the UI can call the figure rough.
    """

    percent: int | None = None
    done_sec: float | None = None
    duration_sec: float | None = None
    elapsed_sec: float | None = None
    eta_sec: float | None = None
    estimated_sec: float | None = None
    estimate_source: Literal["history", "default"] | None = None
    updated_at: UtcDatetime | None = None


def progress_out(row: JobRow, history: Sequence[float] | None = None) -> ProgressOut | None:
    """The progress to show for ``row``, or ``None``.

    Only ``transcribe`` jobs have any. A ``running`` one returns what the worker
    stored (``None`` before its first write, or when the stored JSON is not
    the expected shape). A ``pending`` one is estimated from its payload duration
    and ``history`` (the recent RTFs); without ``history`` or a duration there is
    nothing to show.
    """
    if row.kind != "transcribe":
        return None
    if row.state == "running":
        if row.progress is None:
            return None
        try:
            return ProgressOut.model_validate(row.progress)
        except ValidationError:
            return None
    if row.state == "pending" and history is not None:
        estimate = estimate_effort(row.payload_duration_sec, history)
        if estimate is None:
            return None
        return ProgressOut(
            duration_sec=row.payload_duration_sec,
            eta_sec=round(estimate.estimated_sec, 1),
            estimated_sec=round(estimate.estimated_sec, 1),
            estimate_source=estimate.source,
        )
    return None


class VideoFields(BaseModel):
    """Fields shared by a library item and a search result (#42, #43)."""

    video_id: str
    title: str | None
    channel_id: str | None
    channel_title: str | None
    published_at: UtcDatetime | None
    duration_sec: int | None
    unavailable: str | None


class VideoItem(VideoFields):
    origin: str
    status: VideoStatus
    active_job: ActiveJobOut | None
    last_failure: LastFailureOut | None
    latest_analysis_at: UtcDatetime | None


class VideoList(BaseModel):
    items: list[VideoItem]
    total: int
    offset: int
    limit: int


class ChannelItem(BaseModel):
    channel_id: str
    title: str | None
    active: bool
    video_count: int


class ChannelList(BaseModel):
    items: list[ChannelItem]
    total: int
    offset: int
    limit: int


def channel_list(page: ChannelPage, *, offset: int, limit: int) -> ChannelList:
    return ChannelList(
        items=[
            ChannelItem(
                channel_id=c.channel_id,
                title=c.title,
                active=c.active,
                video_count=c.video_count,
            )
            for c in page.items
        ],
        total=page.total,
        offset=offset,
        limit=limit,
    )


class TopicOut(BaseModel):
    seq: int
    title: str
    summary: str | None
    start_sec: float | None


class ClaimOut(BaseModel):
    text: str
    speaker: str
    start_sec: float | None
    confidence: str | None
    source_chunk_seq: int | None


class QuoteOut(BaseModel):
    text: str
    speaker: str
    start_sec: float | None
    source_chunk_seq: int | None


class AnalysisOut(BaseModel):
    id: int
    model: str
    prompt_version: str
    chunk_strategy: str
    transcript_id: int
    transcript_source: str
    created_at: UtcDatetime
    tldr: str
    speaker_roster: JsonValue | None
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    duration_ms: int | None
    topics: list[TopicOut]
    claims: list[ClaimOut]
    quotes: list[QuoteOut]


class TranscriptSummary(BaseModel):
    id: int
    source: str
    language: str | None
    speaker_source: str
    segment_count: int


class VideoDetail(VideoItem):
    transcript: TranscriptSummary | None
    analysis: AnalysisOut | None
    #: The active transcribe job's figures, else ``None`` (#245).
    progress: ProgressOut | None


class AnalysisList(BaseModel):
    items: list[AnalysisOut]
    total: int
    offset: int
    limit: int


class SegmentOut(BaseModel):
    index: int
    start: float
    end: float
    text: str
    speaker: str | None


class TranscriptOut(BaseModel):
    video_id: str
    transcript_id: int
    source: str
    language: str | None
    speaker_source: str
    total_segments: int
    offset: int
    limit: int
    segments: list[SegmentOut]


class SpanOut(BaseModel):
    text: str
    match: bool


class ExcerptOut(BaseModel):
    spans: list[SpanOut]
    #: Seconds from the start of the video of the transcript segment the match
    #: is in; ``None`` for a ``full_text`` fallback excerpt (#110).
    start_sec: Annotated[float, Field(ge=0)] | None


class SearchResult(VideoFields):
    #: ``None`` when the video matched on its title or analysis and has no transcript.
    transcript_source: str | None
    matched_in: list[Literal["title", "analysis", "transcript"]]
    excerpts: list[ExcerptOut]


class SearchPage(BaseModel):
    results: list[SearchResult]
    limit: int
    offset: int
    has_more: bool


def search_page(page: RepoSearchPage, *, limit: int, offset: int) -> SearchPage:
    return SearchPage(
        results=[
            SearchResult(
                video_id=hit.video_id,
                title=hit.title,
                channel_id=hit.channel_id,
                channel_title=hit.channel_title,
                published_at=hit.published_at,
                duration_sec=hit.duration_sec,
                unavailable=hit.unavailable,
                transcript_source=hit.transcript_source,
                matched_in=list(hit.matched_in),
                excerpts=[
                    ExcerptOut(
                        spans=[SpanOut(text=span.text, match=span.match) for span in e.spans],
                        start_sec=e.start_sec,
                    )
                    for e in hit.excerpts
                ],
            )
            for hit in page.results
        ],
        limit=limit,
        offset=offset,
        has_more=page.has_more,
    )


def video_item(video: VideoSummary) -> VideoItem:
    return VideoItem(**_video_fields(video))


def video_detail(
    video: VideoSummary,
    transcript: TranscriptInfo | None,
    run: AnalysisRun | None,
    progress: ProgressOut | None = None,
) -> VideoDetail:
    return VideoDetail(
        **_video_fields(video),
        progress=progress,
        transcript=None if transcript is None else transcript_summary(transcript),
        analysis=None if run is None else analysis_out(run),
    )


def _video_fields(video: VideoSummary) -> dict[str, Any]:
    active = video.active_job
    failure = video.last_failure
    return {
        "video_id": video.video_id,
        "title": video.title,
        "channel_id": video.channel_id,
        "channel_title": video.channel_title,
        "published_at": video.published_at,
        "duration_sec": video.duration_sec,
        "origin": video.origin,
        "unavailable": video.unavailable,
        "status": video.status,
        "active_job": (
            None if active is None
            else ActiveJobOut(id=active.id, kind=active.kind, state=active.state)
        ),
        "last_failure": (
            None if failure is None
            else LastFailureOut(
                job_id=failure.job_id,
                kind=failure.kind,
                error_class=failure.error_class,
                finished_at=failure.finished_at,
            )
        ),
        "latest_analysis_at": video.latest_analysis_at,
    }


def transcript_summary(info: TranscriptInfo) -> TranscriptSummary:
    return TranscriptSummary(
        id=info.id,
        source=info.source,
        language=info.language,
        speaker_source=info.speaker_source,
        segment_count=info.segment_count,
    )


def analysis_out(run: AnalysisRun) -> AnalysisOut:
    a: Analysis = run.analysis
    assert a.id is not None and a.created_at is not None
    return AnalysisOut(
        id=a.id,
        model=a.model,
        prompt_version=a.prompt_version,
        chunk_strategy=a.chunk_strategy,
        transcript_id=a.transcript_id,
        transcript_source=run.transcript_source,
        created_at=a.created_at,
        tldr=a.tldr,
        speaker_roster=cast(JsonValue, a.speaker_roster),
        input_tokens=a.input_tokens,
        output_tokens=a.output_tokens,
        cost_usd=a.cost_usd,
        duration_ms=a.duration_ms,
        topics=[
            TopicOut(seq=t.seq, title=t.title, summary=t.summary, start_sec=t.start_sec)
            for t in a.topics
        ],
        claims=[
            ClaimOut(
                text=c.text,
                speaker=c.speaker,
                start_sec=c.start_sec,
                confidence=c.confidence,
                source_chunk_seq=c.source_chunk_seq,
            )
            for c in a.claims
        ],
        quotes=[
            QuoteOut(
                text=q.text,
                speaker=q.speaker,
                start_sec=q.start_sec,
                source_chunk_seq=q.source_chunk_seq,
            )
            for q in a.quotes
        ],
    )


class OpsChannelOut(BaseModel):
    channel_id: str
    title: str | None
    poll_failing_since: UtcDatetime | None
    poll_failures: int
    last_poll_err: str | None
    feed_gap_detected_at: UtcDatetime | None
    feed_gap_reason: Literal["suspected", "no_anchor", "listing_window_exhausted"] | None


class OpsChannelListOut(BaseModel):
    items: list[OpsChannelOut]


class JobOut(BaseModel):
    """One ``jobs`` row for the Ops view (#44). ``payload`` is never exposed."""

    id: int
    video_id: str
    video_title: str | None
    kind: str
    dedupe_key: str
    state: str
    priority: int
    attempts: int
    error_class: str | None
    last_error: str | None
    run_after: UtcDatetime
    locked_by: str | None
    heartbeat_at: UtcDatetime | None
    finished_at: UtcDatetime | None
    created_at: UtcDatetime
    progress: ProgressOut | None


class JobListOut(BaseModel):
    items: list[JobOut]
    next_before_id: int | None


def job_out(row: JobRow, history: Sequence[float] | None = None) -> JobOut:
    return JobOut(
        id=row.id,
        video_id=row.video_id,
        video_title=row.video_title,
        kind=row.kind,
        dedupe_key=row.dedupe_key,
        state=row.state,
        priority=row.priority,
        attempts=row.attempts,
        error_class=row.error_class,
        last_error=row.last_error,
        run_after=row.run_after,
        locked_by=row.locked_by,
        heartbeat_at=row.heartbeat_at,
        finished_at=row.finished_at,
        created_at=row.created_at,
        progress=progress_out(row, history),
    )


def transcript_out(page: TranscriptPage, *, offset: int, limit: int) -> TranscriptOut:
    info = page.info
    return TranscriptOut(
        video_id=page.video_id,
        transcript_id=info.id,
        source=info.source,
        language=info.language,
        speaker_source=info.speaker_source,
        total_segments=info.segment_count,
        offset=offset,
        limit=limit,
        segments=[
            SegmentOut(index=s.index, start=s.start, end=s.end, text=s.text, speaker=s.speaker)
            for s in page.segments
        ],
    )


class BackfillRequestOut(BaseModel):
    """``GET /channels/{id}/backfill-requests/{request_id}`` (#115).

    ``result`` is ``null`` until the first listing succeeds; then it has the
    keys of ``POST /channels/{id}/backfill``'s response.
    """

    request_id: int
    channel_id: str
    limit: int
    dry_run: bool
    state: Literal["pending", "running", "done", "failed"]
    attempts: int
    enqueued: int
    error_class: str | None
    created_at: UtcDatetime
    updated_at: UtcDatetime
    finished_at: UtcDatetime | None
    result: dict[str, JsonValue] | None


def backfill_request_out(row: BackfillRequestRow) -> BackfillRequestOut:
    return BackfillRequestOut(
        request_id=row.id,
        channel_id=row.channel_id,
        limit=row.limit_n,
        dry_run=row.dry_run,
        state=cast(Any, row.state),
        attempts=row.attempts,
        enqueued=row.enqueued,
        error_class=row.error_class,
        created_at=row.created_at,
        updated_at=row.updated_at,
        finished_at=row.finished_at,
        result=cast(Any, row.result),
    )
