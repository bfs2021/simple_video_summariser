"""Read routes (issues #39, #42; architecture.md 8.3, 9; D2, D7, D13).

- ``GET /videos``: the library, filtered and paged.
- ``GET /channels``: every known channel with its video count, paged (#119).
- ``GET /videos/{video_id}``: one video, its best transcript and latest analysis.
- ``GET /videos/{video_id}/analyses``: every analysis run, newest first.
- ``GET /videos/{video_id}/transcript``: one page of the best transcript's segments.
- ``GET /videos/{video_id}/render``: the latest analysis as one HTML page (#45).
- ``GET /search``: full-text search over each video's preferred transcript (#43).
- ``GET /channels/{id}/backfill-requests/{request_id}``: one backfill request (#115).

Every path and query parameter is untrusted. It is validated by a gate
dependency that runs before ``get_read_conn``, so a 422 never opens a
database connection (FastAPI would otherwise resolve the connection even
when parameter parsing fails). A rejection never echoes the submitted value.
The routes hold no SQL, run on a read-only connection, and write nothing.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Annotated, Any, Literal

import psycopg
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from starlette.convertors import Convertor, register_url_convertor

from common.repo.analyses import (
    get_analysis,
    latest_analysis,
    latest_analysis_run,
    list_analysis_runs,
)
from common.repo.backfill_requests import get_request
from common.repo.channels import get_channel_title, list_channels
from common.repo.search import search_transcripts
from common.repo.transcripts import best_transcript_info, best_transcript_page
from common.repo.videos import get_video, get_video_meta, list_videos, video_exists
from common.youtube_refs import is_channel_id, is_video_id
from services.api import schemas
from services.api.deps import ReadDatabaseUnavailable, get_read_conn
from services.api.render import render_analysis_html


class _AnySegment(Convertor[str]):
    """Matches anything, newlines and a decoded ``/`` included.

    The default ``str`` converter stops at ``/``, so ``/videos/a%2Fb/render``
    would be a bare 404 instead of the route's own 422. ``VideoId`` validates
    the value. Only the render route names ``:any_segment``.
    """

    regex = "(?s:.+?)"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        return value


register_url_convertor("any_segment", _AnySegment())


def _mark_read_request(request: Request) -> None:
    request.state.read_route = True


def is_read_request(request: Request) -> bool:
    """Whether ``request`` is being served by this router (for the 503 body)."""
    return bool(getattr(request.state, "read_route", False))


router = APIRouter(dependencies=[Depends(_mark_read_request)])
_log = structlog.get_logger(__name__)

MAX_OFFSET = 1_000_000
MAX_SEARCH_CHARS = 200
MAX_SEARCH_OFFSET = 1000
BAD_VIDEO_ID = "video id must be 11 characters of A-Z, a-z, 0-9, '_' or '-'"
VIDEO_NOT_FOUND = "video not found"
BAD_BACKFILL_CHANNEL = "channel_id must be a canonical YouTube channel ID (UC + 22 characters)"
BAD_REQUEST_ID = "request id must be a positive integer"
BACKFILL_REQUEST_NOT_FOUND = "backfill request not found"
TRANSCRIPT_NOT_FOUND = "transcript not found"
NO_ANALYSIS_YET = "no analysis yet"
ANALYSIS_NOT_FOUND = "analysis not found"
#: The render page loads nothing and runs nothing (#45).
RENDER_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",
}

_DIGITS = re.compile(r"[0-9]{1,7}")
_MAX_DATE_CHARS = 64


def _query_int(value: object) -> object:
    """Plain ASCII digits only: no sign, exponent, fraction, underscore or space.

    At most 7 digits (every bound here is at most 1_000_000), so a huge value
    is rejected before it is parsed; a zero-padded value longer than that is
    rejected too.
    """
    if isinstance(value, str):
        if _DIGITS.fullmatch(value) is None:
            raise ValueError("must be a non-negative integer within range")
        return int(value)
    return value


def _query_datetime(value: object) -> object:
    """ISO 8601 date (midnight UTC) or datetime; a naive datetime is UTC."""
    if not isinstance(value, str):
        return value
    if len(value) > _MAX_DATE_CHARS:
        raise ValueError("must be an ISO 8601 date or datetime")
    try:
        if len(value) <= 10:
            parsed = datetime.combine(date.fromisoformat(value), datetime.min.time())
        else:
            parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("must be an ISO 8601 date or datetime") from None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _channel(value: object) -> object:
    if value is not None and not is_channel_id(value):
        raise ValueError("must be a canonical channel ID (UC + 22 characters)")
    return value


QueryInt = Annotated[int, BeforeValidator(_query_int)]
QueryDatetime = Annotated[datetime, BeforeValidator(_query_datetime)]
ChannelId = Annotated[str, BeforeValidator(_channel)]

_BIGINT_MAX = 9_223_372_036_854_775_807
_ID_DIGITS = re.compile(r"[0-9]{1,19}")


def _query_id(value: object) -> object:
    """Plain ASCII digits only, at most 19 of them (a BIGINT), parsed after the check."""
    if isinstance(value, str):
        if _ID_DIGITS.fullmatch(value) is None:
            raise ValueError("must be a positive integer within range")
        return int(value)
    return value


QueryId = Annotated[int, Field(ge=1, le=_BIGINT_MAX), BeforeValidator(_query_id)]


class RenderQuery(BaseModel):
    """``analysis_id`` is optional, but present-and-empty is invalid (#109)."""

    model_config = ConfigDict(extra="ignore")

    analysis_id: QueryId | None = None


class VideoListQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    channel: ChannelId | None = None
    status: Literal["done", "processing", "unavailable", "failed", "idle"] | None = None
    published_after: QueryDatetime | None = None
    published_before: QueryDatetime | None = None
    offset: QueryInt = Field(0, ge=0, le=MAX_OFFSET)
    limit: QueryInt = Field(50, ge=1, le=200)

    @model_validator(mode="after")
    def _ordered_dates(self) -> VideoListQuery:
        after, before = self.published_after, self.published_before
        if after is not None and before is not None and after > before:
            raise ValueError("published_after must not be later than published_before")
        return self


def _search_q(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if "\x00" in stripped:
        raise ValueError("q must not contain NUL")
    if not 1 <= len(stripped) <= MAX_SEARCH_CHARS:
        raise ValueError(f"q must be 1-{MAX_SEARCH_CHARS} characters after trimming")
    return stripped


class ChannelListQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    offset: QueryInt = Field(0, ge=0, le=MAX_OFFSET)
    limit: QueryInt = Field(200, ge=1, le=200)


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    q: Annotated[str, BeforeValidator(_search_q)] = Field(
        min_length=1, max_length=MAX_SEARCH_CHARS
    )
    offset: QueryInt = Field(0, ge=0, le=MAX_SEARCH_OFFSET)
    limit: QueryInt = Field(20, ge=1, le=50)


class AnalysesQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    offset: QueryInt = Field(0, ge=0, le=MAX_OFFSET)
    limit: QueryInt = Field(10, ge=1, le=50)


class TranscriptQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    offset: QueryInt = Field(0, ge=0, le=MAX_OFFSET)
    limit: QueryInt = Field(200, ge=1, le=1000)


def _gate(model: type[BaseModel]) -> Callable[[Request], None]:
    """A dependency that validates the query string and *raises* on failure.

    Raising (rather than letting FastAPI collect the error) stops dependency
    resolution before ``get_read_conn``. The route still declares ``model``
    as its ``Query()`` parameter, for the typed value and for ``/openapi.json``.
    """

    def gate(request: Request) -> None:
        try:
            model.model_validate(dict(request.query_params))
        except ValidationError as exc:
            errors = exc.errors(include_input=False, include_context=False, include_url=False)
            raise RequestValidationError(
                [{**error, "loc": ("query", *error["loc"])} for error in errors]
            ) from None

    return gate


def _valid_video_id(request: Request, video_id: str) -> str:
    """The path id, also required verbatim in the raw path.

    A valid id never contains ``%``, so requiring it as a whole raw segment
    rejects a percent-encoded id (``%41...``) that would otherwise decode into
    validity, whatever prefix (``root_path``) the app is mounted under.
    """
    raw_path = request.scope.get("raw_path")
    raw_ok = not isinstance(raw_path, bytes) or video_id.encode() in raw_path.split(b"/")
    if not raw_ok or not is_video_id(video_id):
        raise RequestValidationError(
            [{"loc": ("path", "video_id"), "msg": BAD_VIDEO_ID, "type": "value_error"}]
        )
    return video_id


def _valid_backfill_channel(channel_id: str) -> str:
    if not is_channel_id(channel_id):
        raise RequestValidationError(
            [{"loc": ("path", "channel_id"), "msg": BAD_BACKFILL_CHANNEL, "type": "value_error"}]
        )
    return channel_id


def _valid_request_id(request_id: str) -> int:
    """Plain ASCII digits, 1..BIGINT max; the message never echoes the input."""
    if _ID_DIGITS.fullmatch(request_id) is None or not 1 <= int(request_id) <= _BIGINT_MAX:
        raise RequestValidationError(
            [{"loc": ("path", "request_id"), "msg": BAD_REQUEST_ID, "type": "value_error"}]
        )
    return int(request_id)


ReadConn = Annotated[psycopg.Connection[Any], Depends(get_read_conn)]
BackfillChannelId = Annotated[str, Depends(_valid_backfill_channel)]
BackfillRequestId = Annotated[int, Depends(_valid_request_id)]
VideoId = Annotated[str, Depends(_valid_video_id)]


@router.get("/videos", response_model=schemas.VideoList)
def get_videos(
    _: Annotated[None, Depends(_gate(VideoListQuery))],
    params: Annotated[VideoListQuery, Query()],
    conn: ReadConn,
) -> schemas.VideoList:
    """The library, newest published first; ``total`` counts every match."""
    page = list_videos(
        conn,
        channel_id=params.channel,
        status=params.status,
        published_after=params.published_after,
        published_before=params.published_before,
        offset=params.offset,
        limit=params.limit,
    )
    return schemas.VideoList(
        items=[schemas.video_item(v) for v in page.items],
        total=page.total,
        offset=params.offset,
        limit=params.limit,
    )


@router.get("/channels", response_model=schemas.ChannelList)
def get_channels(
    _: Annotated[None, Depends(_gate(ChannelListQuery))],
    params: Annotated[ChannelListQuery, Query()],
    conn: ReadConn,
) -> schemas.ChannelList:
    """Every channel, active or not, by title; ``total`` counts them all (#119)."""
    page = list_channels(conn, offset=params.offset, limit=params.limit)
    return schemas.channel_list(page, offset=params.offset, limit=params.limit)


@router.get(
    "/channels/{channel_id}/backfill-requests/{request_id}",
    response_model=schemas.BackfillRequestOut,
)
def get_backfill_request(
    channel_id: BackfillChannelId, request_id: BackfillRequestId, conn: ReadConn
) -> schemas.BackfillRequestOut:
    """One backfill request's progress (#115); ``result`` is null until a listing succeeded."""
    row = get_request(conn, channel_id, request_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, BACKFILL_REQUEST_NOT_FOUND)
    return schemas.backfill_request_out(row)


@router.get("/videos/{video_id}", response_model=schemas.VideoDetail)
def get_video_detail(video_id: VideoId, conn: ReadConn) -> schemas.VideoDetail:
    """One video, its status, best transcript (D2) and latest analysis."""
    video = get_video(conn, video_id)
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, VIDEO_NOT_FOUND)
    return schemas.video_detail(
        video, best_transcript_info(conn, video_id), latest_analysis_run(conn, video_id)
    )


@router.get("/videos/{video_id}/analyses", response_model=schemas.AnalysisList)
def get_video_analyses(
    video_id: VideoId,
    _: Annotated[None, Depends(_gate(AnalysesQuery))],
    params: Annotated[AnalysesQuery, Query()],
    conn: ReadConn,
) -> schemas.AnalysisList:
    """Every analysis run of the video, newest first (D7)."""
    if not video_exists(conn, video_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, VIDEO_NOT_FOUND)
    runs, total = list_analysis_runs(conn, video_id, offset=params.offset, limit=params.limit)
    return schemas.AnalysisList(
        items=[schemas.analysis_out(run) for run in runs],
        total=total,
        offset=params.offset,
        limit=params.limit,
    )


@router.get("/videos/{video_id}/transcript", response_model=schemas.TranscriptOut)
def get_video_transcript(
    video_id: VideoId,
    _: Annotated[None, Depends(_gate(TranscriptQuery))],
    params: Annotated[TranscriptQuery, Query()],
    conn: ReadConn,
) -> schemas.TranscriptOut:
    """One page of the best transcript's segments (D2, architecture.md 9)."""
    if not video_exists(conn, video_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, VIDEO_NOT_FOUND)
    page = best_transcript_page(conn, video_id, offset=params.offset, limit=params.limit)
    if page is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, TRANSCRIPT_NOT_FOUND)
    return schemas.transcript_out(page, offset=params.offset, limit=params.limit)


@router.get("/videos/{video_id:any_segment}/render", response_class=HTMLResponse)
def get_video_render(
    video_id: VideoId,
    _: Annotated[None, Depends(_gate(RenderQuery))],
    params: Annotated[RenderQuery, Query()],
    conn: ReadConn,
) -> HTMLResponse:
    """One analysis of the video as a self-contained page (D12c).

    The latest by default; ``analysis_id`` picks one run of this video (#109).
    The page comes from ``render.render_analysis_html``; this handler only
    reads and sets the security headers. No 404 echoes the ids, and another
    video's analysis id answers like an unknown one.
    """
    video = get_video_meta(conn, video_id)
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, VIDEO_NOT_FOUND)
    if params.analysis_id is None:
        analysis = latest_analysis(conn, video_id)
        if analysis is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, NO_ANALYSIS_YET)
    else:
        analysis = get_analysis(conn, video_id, params.analysis_id)
        if analysis is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, ANALYSIS_NOT_FOUND)
    page = render_analysis_html(video, get_channel_title(conn, video.channel_id), analysis)
    return HTMLResponse(page, headers=RENDER_HEADERS)


@router.get("/search", response_model=schemas.SearchPage)
def get_search(
    _: Annotated[None, Depends(_gate(SearchQuery))],
    params: Annotated[SearchQuery, Query()],
    conn: ReadConn,
) -> schemas.SearchPage:
    """Videos whose preferred transcript matches ``q``, most relevant first.

    The body never echoes ``q``, and the log line records only its length.
    Any database error is a 503 (``deps.get_read_conn``).
    """
    started = time.perf_counter()
    try:
        page = search_transcripts(conn, params.q, params.limit, params.offset)
    except psycopg.Error as exc:
        raise ReadDatabaseUnavailable from exc
    _log.info(
        "search",
        q_len=len(params.q),
        results=len(page.results),
        offset=params.offset,
        limit=params.limit,
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return schemas.search_page(page, limit=params.limit, offset=params.offset)
