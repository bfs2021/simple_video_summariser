"""Write routes (issues #39, #40; architecture.md 8.3; D13). Rate-limited as a group.

- ``POST /videos`` turns a pasted video URL or bare ID into a ``videos`` row
  and one ``ingest`` job at ``PRIORITY_INTERACTIVE``.
- ``POST /channels`` starts forward-only monitoring from ``monitor_from = now()``.
- ``POST /channels/{channel_id}/backfill`` lists a registered channel's newest
  uploads and, unless it is a dry run, enqueues the unknown ones at
  ``PRIORITY_BACKFILL`` (#41; logic in ``services.api.backfill``).
- ``POST /channels/{channel_id}/backfill/cancel`` deletes one real backfill's
  still-pending ``ingest`` jobs (#116).

Input is untrusted. It is parsed by ``common.youtube_refs`` before anything
reaches the database, and a rejection returns a fixed message that never
echoes the submitted text. Both routes are idempotent: repeating a submission
never creates a second job or row, and never moves ``monitor_from``.
Route bodies hold no SQL; they call ``common.repo`` and ``PostgresQueue``.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any

import psycopg
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from psycopg.pq import TransactionStatus
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.convertors import Convertor, register_url_convertor

from common.config import get_settings
from common.errors import RateLimitedError, ToolFailureError, TransientNetworkError
from common.models import CatalogSource
from common.queue import (
    PRIORITY_INTERACTIVE,
    PostgresQueue,
    analyze_dedupe_key,
)
from common.repo.channels import register_channel
from common.repo.jobs import active_job, latest_job
from common.repo.transcripts import get_best_transcript
from common.repo.videos import insert_submitted_video, video_exists
from common.youtube_refs import (
    UnsupportedChannelRef,
    is_channel_id,
    is_video_id,
    parse_channel_ref,
    parse_video_ref,
)
from services.api.backfill import (
    MAX_BACKFILL_LIMIT,
    BackfillResult,
    CancelResult,
    ChannelNotRegistered,
    SubtitleProbe,
    backfill,
    cancel_backfill,
)
from services.api.deps import get_catalog, get_conn, rate_limit

_log = structlog.get_logger(__name__)


class _UntrustedSegment(Convertor[str]):
    """Matches anything, newlines and decoded ``/`` included.

    The default ``str`` converter stops at ``/`` and ``path`` stops at a
    newline, so a malformed ID would get a bare 404 instead of the route's
    own 422. The route validates the value itself. Only routes that name
    ``:untrusted`` use it; any other ``/channels/.../backfill``-suffixed route
    added later would be shadowed by this one.
    """

    regex = "(?s:.+?)"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        return value


register_url_convertor("untrusted", _UntrustedSegment())

router = APIRouter(dependencies=[Depends(rate_limit)])

_BACKFILL_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
BAD_VIDEO_REF = "url must be a YouTube video ID or video URL"
BAD_CHANNEL_REF = "url must be a YouTube channel ID or youtube.com/channel/ URL"
BAD_BACKFILL_CHANNEL = "channel_id must be a canonical YouTube channel ID (UC + 22 characters)"
CHANNEL_NOT_REGISTERED = "channel is not registered; register it with POST /channels first"
LISTING_UNAVAILABLE = "YouTube listing unavailable, try again later"
LISTING_FAILED = "YouTube listing failed"
VIDEO_NOT_FOUND = "video not found"
NO_TRANSCRIPT = "no transcript yet"
ANALYSIS_RUNNING = "an analysis is already running for this video"
REANALYSIS_CONTENDED = "the analysis job kept changing, try again"
BAD_REANALYZE_VIDEO = "video id must be 11 characters of A-Z, a-z, 0-9, '_' or '-'"


class SubmitRef(BaseModel):
    model_config = ConfigDict(strict=True)

    url: str


class BackfillRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    limit: int = Field(ge=1, le=MAX_BACKFILL_LIMIT)
    dry_run: bool = True


class BackfillLlmEstimate(BaseModel):
    summarizer: str
    model: str
    chars_per_audio_sec: float | None
    chars_samples: int
    cost_per_char: float | None
    cost_samples: int
    est_input_tokens: int | None
    cost_usd: float | None
    basis: str


class BackfillEstimate(BaseModel):
    audio_sec: int
    unknown_duration: int
    rtf: float | None
    rtf_samples: int
    transcription_sec: int | None
    assumes: str
    probed: int
    manual_subtitles: int
    needs_transcription: int
    unchecked: int
    probe_failed: int
    deadline_reached: bool
    transcribe_audio_sec: int
    llm: BackfillLlmEstimate


class BackfillVideo(BaseModel):
    video_id: str
    duration_sec: int | None
    subtitles: str
    transcription_sec: int | None
    est_input_tokens: int | None
    llm_cost_usd: float | None


class BackfillResponse(BaseModel):
    channel_id: str
    dry_run: bool
    limit: int
    listed: int
    total_count: int | None
    already_known: int
    new_videos: int
    enqueued: int
    video_ids: list[str]
    videos: list[BackfillVideo]
    estimate: BackfillEstimate
    backfill_id: str | None


class CancelBackfillRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    backfill_id: str

    @field_validator("backfill_id")
    @classmethod
    def _canonical_uuid(cls, value: str) -> str:
        # A lowercase canonical UUID (what ``backfill`` generates), nothing else.
        # ``fullmatch`` so a trailing newline cannot slip through ``$``.
        if _BACKFILL_ID_RE.fullmatch(value) is None:
            raise ValueError("backfill_id must be a lowercase canonical UUID")
        return value


class CancelBackfillResponse(BaseModel):
    channel_id: str
    backfill_id: str
    cancelled: int
    videos_removed: int
    videos_kept: int
    video_ids: list[str]


class VideoSubmission(BaseModel):
    video_id: str
    job_id: int
    state: str
    created: bool


class ChannelRegistration(BaseModel):
    channel_id: str
    active: bool
    monitor_from: datetime
    added_at: datetime
    created: bool


@dataclass(frozen=True, slots=True)
class SubmitResult:
    video_id: str
    job_id: int
    state: str
    created: bool


def _require_idle(conn: psycopg.Connection[Any]) -> None:
    """The write must be the outermost transaction, or it would only be a savepoint.

    ``get_conn`` closes without committing, so a savepoint inside a
    transaction some earlier statement opened would be silently lost.
    """
    if conn.info.transaction_status != TransactionStatus.IDLE:
        raise RuntimeError("write route expects a connection with no open transaction")


def submit_video(
    conn: psycopg.Connection[Any], video_id: str, *, queue: PostgresQueue
) -> SubmitResult:
    """Record ``video_id`` and give it one ``ingest`` job, unless it already has one.

    In one transaction: a bare ``adhoc`` row (an existing row is kept as it
    is), then the latest ``ingest`` job is returned if there is one, whatever
    its state; a finished job is not resubmitted (retry is #44). Then every
    ``pending`` job of the video below ``PRIORITY_INTERACTIVE`` is raised to
    it (#103), whatever its kind; running jobs are never touched. Otherwise
    one job is enqueued at ``PRIORITY_INTERACTIVE``. A concurrent submission that loses the enqueue
    race returns the winner's job.
    """
    with conn.transaction():
        insert_submitted_video(conn, video_id)
        existing = latest_job(conn, video_id, "ingest")
        if existing is None:
            job_id = queue.enqueue("ingest", video_id, priority=PRIORITY_INTERACTIVE)
            if job_id is not None:
                return SubmitResult(video_id, job_id, "pending", created=True)
            existing = latest_job(conn, video_id, "ingest")
        if existing is None:
            raise RuntimeError("enqueue conflicted but no ingest job is visible")
        queue.raise_priority(video_id, PRIORITY_INTERACTIVE)
        job_id, state = existing
        return SubmitResult(video_id, job_id, state, created=False)


@router.post(
    "/videos",
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"model": VideoSubmission, "description": "Already submitted"}},
)
def post_videos(
    body: SubmitRef,
    response: Response,
    conn: Annotated[psycopg.Connection[Any], Depends(get_conn)],
) -> VideoSubmission:
    try:
        video_id = parse_video_ref(body.url)
    except ValueError:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, BAD_VIDEO_REF
        ) from None
    _require_idle(conn)
    result = submit_video(conn, video_id, queue=PostgresQueue(conn))
    if not result.created:
        response.status_code = status.HTTP_200_OK
    return VideoSubmission(
        video_id=result.video_id,
        job_id=result.job_id,
        state=result.state,
        created=result.created,
    )


@dataclass(frozen=True, slots=True)
class Reanalysis:
    """The one active forced ``analyze`` job the request resolved to (#101)."""

    job_id: int
    state: str
    created: bool


class ReanalysisConflict(Exception):
    """An unforced ``analyze`` job is ``running`` at the key: 409 (#101).

    A running job cannot be changed safely, and it may skip as "already analyzed".
    """

    def __init__(self, job_id: int) -> None:
        super().__init__(job_id)
        self.job_id = job_id


class ReanalysisContended(Exception):
    """The active job at the key changed under every attempt: 409, retry (#101)."""


#: Re-reads after losing a race. Each loss means another connection changed the
#: job in between; a handful of consecutive losses is not a realistic race.
_REANALYSIS_ATTEMPTS = 5


def request_reanalysis(
    conn: psycopg.Connection[Any],
    video_id: str,
    *,
    queue: PostgresQueue,
    dedupe_key: str,
    after_lookup: Callable[[], None] | None = None,
) -> Reanalysis:
    """Make exactly one active, forced, interactive ``analyze`` job at ``dedupe_key`` (#101).

    One key is shared with the sweep and the post-transcribe analyze (#33's
    guard dead-letters any other), so an active job there is reused:

    - none: a new ``{"force": true}`` job at ``PRIORITY_INTERACTIVE`` (created);
    - ``pending`` and forced: as it is;
    - ``pending`` and not forced: upgraded in place by ``force_pending``;
    - ``running`` and forced: as it is;
    - ``running`` and not forced: ``ReanalysisConflict``.

    A lost race (the upgrade matched no row because the job was claimed or
    finished, or the enqueue conflicted with a job that is gone again) re-reads
    and applies the rules once more. ``after_lookup`` is a test seam, called
    between the lookup of a ``pending`` unforced job and its upgrade. The
    caller owns the transaction (the route wraps this in ``conn.transaction()``).
    """
    for _ in range(_REANALYSIS_ATTEMPTS):
        found = active_job(conn, video_id, "analyze", dedupe_key)
        if found is None:
            job_id = queue.enqueue(
                "analyze",
                video_id,
                dedupe_key=dedupe_key,
                payload={"force": True},
                priority=PRIORITY_INTERACTIVE,
            )
            if job_id is not None:
                return Reanalysis(job_id, "pending", created=True)
            continue
        if found.forced:
            return Reanalysis(found.id, found.state, created=False)
        if found.state == "running":
            raise ReanalysisConflict(found.id)
        if after_lookup is not None:
            after_lookup()
        if queue.force_pending(found.id, PRIORITY_INTERACTIVE):
            return Reanalysis(found.id, "pending", created=False)
    raise ReanalysisContended


@router.post(
    "/videos/{video_id:untrusted}/reanalyze",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=VideoSubmission,
    responses={
        200: {"model": VideoSubmission, "description": "An active forced job was reused"},
        404: {"description": "No such video"},
        409: {"description": "No transcript yet, or an analysis is already running"},
        422: {"description": "Malformed video ID"},
    },
)
def post_video_reanalyze(
    video_id: Annotated[str, Depends(_reanalyze_video_id)],
    response: Response,
    conn: Annotated[psycopg.Connection[Any], Depends(get_conn)],
) -> VideoSubmission | JSONResponse:
    """Force a fresh analysis of a stored transcript at the current version (#101).

    Only enqueues: the analyzer (#33) does the work. The request has no body
    and nothing in it changes the payload, priority or key.
    """
    settings = get_settings()
    dedupe_key = analyze_dedupe_key(settings.PROMPT_VERSION, settings.SUMMARIZER)
    _require_idle(conn)
    with conn.transaction():
        if not video_exists(conn, video_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, VIDEO_NOT_FOUND)
        if get_best_transcript(conn, video_id) is None:
            raise HTTPException(status.HTTP_409_CONFLICT, NO_TRANSCRIPT)
        try:
            result = request_reanalysis(
                conn, video_id, queue=PostgresQueue(conn), dedupe_key=dedupe_key
            )
        except ReanalysisConflict as conflict:
            return JSONResponse(
                {"detail": ANALYSIS_RUNNING, "job_id": conflict.job_id},
                status_code=status.HTTP_409_CONFLICT,
            )
        except ReanalysisContended:
            raise HTTPException(status.HTTP_409_CONFLICT, REANALYSIS_CONTENDED) from None
    if not result.created:
        response.status_code = status.HTTP_200_OK
    return VideoSubmission(
        video_id=video_id, job_id=result.job_id, state=result.state, created=result.created
    )


@router.post(
    "/channels",
    status_code=status.HTTP_201_CREATED,
    responses={
        200: {"model": ChannelRegistration, "description": "Already registered"}
    },
)
def post_channels(
    body: SubmitRef,
    response: Response,
    conn: Annotated[psycopg.Connection[Any], Depends(get_conn)],
) -> ChannelRegistration:
    try:
        channel_id = parse_channel_ref(body.url)
    except UnsupportedChannelRef as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from None
    except ValueError:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, BAD_CHANNEL_REF
        ) from None
    _require_idle(conn)
    with conn.transaction():
        registered = register_channel(conn, channel_id)
    if not registered.created:
        response.status_code = status.HTTP_200_OK
    channel = registered.channel
    return ChannelRegistration(
        channel_id=channel.channel_id,
        active=channel.active,
        monitor_from=channel.monitor_from,
        added_at=channel.added_at,
        created=registered.created,
    )


def _reanalyze_video_id(request: Request, video_id: str) -> str:
    """Validate the path before ``get_conn`` resolves.

    The id must also appear verbatim as a raw path segment: a valid id never
    contains ``%``, so a percent-encoded one that decodes into validity is
    rejected. The message is fixed; the submitted value is never echoed.
    """
    raw_path = request.scope.get("raw_path")
    raw_ok = not isinstance(raw_path, bytes) or video_id.encode() in raw_path.split(b"/")
    if not raw_ok or not is_video_id(video_id):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, BAD_REANALYZE_VIDEO)
    return video_id


def _backfill_channel_id(channel_id: str) -> str:
    """Validate the path before ``get_conn`` or ``get_catalog`` resolve."""
    if not is_channel_id(channel_id):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, BAD_BACKFILL_CHANNEL)
    return channel_id


def get_subtitle_probe() -> SubtitleProbe:
    """The dry run's subtitle probe (#117), built per request.

    The adapters are imported here, not at module level, so importing the app
    loads no ``adapters.*`` or yt-dlp module. Tests override this dependency.
    Lives here rather than in ``deps`` next to ``get_catalog`` because that
    module was owned by another change when this was written.
    """
    from adapters.youtube.metadata import YouTubeMetadata
    from adapters.youtube.subtitles import YouTubeSubtitles

    cookie_file = get_settings().YTDLP_COOKIE_FILE
    available = YouTubeSubtitles().available
    return SubtitleProbe(
        metadata_factory=lambda timeout: YouTubeMetadata(timeout=timeout, cookie_file=cookie_file),
        has_manual=lambda meta: available(meta).manual is not None,
    )


@router.post(
    "/channels/{channel_id:untrusted}/backfill",
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"model": BackfillResponse, "description": "Dry run, or nothing new"}},
)
def post_channel_backfill(
    channel_id: Annotated[str, Depends(_backfill_channel_id)],
    body: BackfillRequest,
    response: Response,
    conn: Annotated[psycopg.Connection[Any], Depends(get_conn)],
    catalog: Annotated[CatalogSource, Depends(get_catalog)],
    probe: Annotated[SubtitleProbe, Depends(get_subtitle_probe)],
) -> BackfillResponse:
    """Backfill a registered channel's newest ``limit`` uploads (#41, D9b).

    This is the one API route that runs a subprocess: the catalog is a
    yt-dlp flat listing, bounded by ``deps.BACKFILL_LIST_TIMEOUT_SEC`` and
    ``MAX_BACKFILL_LIMIT``. It departs from "the API performs no processing"
    because architecture.md 8.3 wants a synchronous dry-run answer, and a
    flat listing is network-bound, not CPU work (owner decision; moving it
    to a worker is #115). A dry run (the default) writes nothing; a real run
    answers 202 when it enqueued at least one job, else 200.

    A dry run also checks each new video's subtitles, newest first, within a
    100 s budget (#117); the rest are reported as ``unchecked``. It writes
    nothing.
    """
    _require_idle(conn)
    settings = get_settings()
    started = time.perf_counter()
    try:
        result = backfill(
            conn,
            PostgresQueue(conn),
            catalog,
            channel_id,
            limit=body.limit,
            dry_run=body.dry_run,
            probe=probe,
            prefer_whisper=settings.PREFER_WHISPER,
            summarizer=settings.SUMMARIZER,
            model=(
                settings.ANTHROPIC_MODEL
                if settings.SUMMARIZER == "anthropic"
                else settings.OLLAMA_MODEL
            ),
        )
    except ChannelNotRegistered:
        raise HTTPException(status.HTTP_404_NOT_FOUND, CHANNEL_NOT_REGISTERED) from None
    except (TransientNetworkError, RateLimitedError, ToolFailureError) as exc:
        _log.warning(
            "backfill listing failed",
            channel_id=channel_id,
            error_class=type(exc).__name__,
        )
        if isinstance(exc, ToolFailureError):
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, LISTING_FAILED) from None
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, LISTING_UNAVAILABLE
        ) from None
    _log.info(
        "backfill",
        channel_id=channel_id,
        limit=result.limit,
        dry_run=result.dry_run,
        listed=result.listed,
        new_videos=result.new_videos,
        enqueued=result.enqueued,
        backfill_id=result.backfill_id,
        probed=result.estimate.probed,
        manual_subtitles=result.estimate.manual_subtitles,
        unchecked=result.estimate.unchecked,
        probe_failed=result.estimate.probe_failed,
        deadline_reached=result.estimate.deadline_reached,
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    if result.enqueued == 0:
        response.status_code = status.HTTP_200_OK
    return _backfill_response(result)


_LLM_FIELDS = tuple(BackfillLlmEstimate.model_fields)


def _backfill_response(result: BackfillResult) -> BackfillResponse:
    estimate = result.estimate
    llm = estimate.llm
    return BackfillResponse(
        channel_id=result.channel_id,
        dry_run=result.dry_run,
        limit=result.limit,
        listed=result.listed,
        total_count=result.total_count,
        already_known=result.already_known,
        new_videos=result.new_videos,
        enqueued=result.enqueued,
        video_ids=list(result.video_ids),
        videos=[
            BackfillVideo(
                video_id=v.video_id,
                duration_sec=v.duration_sec,
                subtitles=v.subtitles,
                transcription_sec=v.transcription_sec,
                est_input_tokens=v.est_input_tokens,
                llm_cost_usd=v.llm_cost_usd,
            )
            for v in result.videos
        ],
        estimate=BackfillEstimate(
            audio_sec=estimate.audio_sec,
            unknown_duration=estimate.unknown_duration,
            rtf=estimate.rtf,
            rtf_samples=estimate.rtf_samples,
            transcription_sec=estimate.transcription_sec,
            assumes=estimate.assumes,
            probed=estimate.probed,
            manual_subtitles=estimate.manual_subtitles,
            needs_transcription=estimate.needs_transcription,
            unchecked=estimate.unchecked,
            probe_failed=estimate.probe_failed,
            deadline_reached=estimate.deadline_reached,
            transcribe_audio_sec=estimate.transcribe_audio_sec,
            llm=BackfillLlmEstimate(**{f: getattr(llm, f) for f in _LLM_FIELDS}),
        ),
        backfill_id=result.backfill_id,
    )


def _require_json_content_type(request: Request) -> None:
    """Refuse a non-JSON body (415) so a cross-site form post cannot trigger a cancel.

    Same rule as ``routes_ops``' check (#44), which is private to that module.
    """
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json"
        )


@router.post(
    "/channels/{channel_id:untrusted}/backfill/cancel",
    dependencies=[Depends(_require_json_content_type)],
)
def post_channel_backfill_cancel(
    channel_id: Annotated[str, Depends(_backfill_channel_id)],
    body: CancelBackfillRequest,
    conn: Annotated[psycopg.Connection[Any], Depends(get_conn)],
) -> CancelBackfillResponse:
    """Cancel one real backfill's still-pending ``ingest`` jobs (#116).

    Deletes only jobs that are ``pending``, ``ingest``, at ``PRIORITY_BACKFILL``,
    tagged with ``backfill_id`` and belonging to this channel; running jobs,
    follow-up jobs and jobs from RSS, ``POST /videos`` or another backfill are
    untouched. An unknown ``backfill_id`` cancels nothing (200, ``cancelled: 0``).
    Each cancelled job's ``videos`` row is removed too, unless something else
    depends on it (a transcript, another job); those are ``videos_kept``. A kept
    video is no longer picked up by backfill, so re-submit it with
    ``POST /videos``.
    """
    _require_idle(conn)
    try:
        result = cancel_backfill(conn, channel_id, body.backfill_id)
    except ChannelNotRegistered:
        raise HTTPException(status.HTTP_404_NOT_FOUND, CHANNEL_NOT_REGISTERED) from None
    _log.warning(
        "backfill.cancelled",
        channel_id=result.channel_id,
        backfill_id=result.backfill_id,
        cancelled=result.cancelled,
        videos_removed=result.videos_removed,
        videos_kept=result.videos_kept,
    )
    return _cancel_response(result)


def _cancel_response(result: CancelResult) -> CancelBackfillResponse:
    return CancelBackfillResponse(
        channel_id=result.channel_id,
        backfill_id=result.backfill_id,
        cancelled=result.cancelled,
        videos_removed=result.videos_removed,
        videos_kept=result.videos_kept,
        video_ids=list(result.video_ids),
    )
