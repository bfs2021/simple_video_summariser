"""Operational routes (issues #39, #44; architecture.md 8.3, 11.6).

- ``GET /ops/jobs`` is a bounded, paginated list of jobs, filterable by
  state, kind and error class, for ``curl`` or the Ops view (#54) to inspect
  the queue without a database client.
- ``POST /ops/jobs/{id}/retry`` resurrects one ``dead`` job: same id,
  ``attempts`` reset to 0, straight back into ``pending``. It refuses
  anything that would create a duplicate active job or redo work a later
  job already covers.

Every path and query parameter is untrusted. A gate dependency validates the
query string before ``get_conn`` resolves, so a 422 never opens a database
connection; a rejection never echoes the submitted value. The retry route
also requires ``Content-Type: application/json`` (``deps.rate_limit`` is
attached per-route here, since this is not the write router - #39's own
default-deny CORS then makes a cross-site browser POST need a preflight,
which it refuses). Route bodies hold no SQL; they call ``common.repo.jobs``
and ``PostgresQueue.retry_dead``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

import psycopg
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from common import metrics
from common.config import get_settings
from common.errors import ErrorClass
from common.queue import MAX_BULK_RETRY, VALID_KINDS, PostgresQueue
from common.repo.analyses import (
    reanalysis_backlog_count,
    token_usage_by_model,
    total_cost_usd,
    total_speaker_coercions,
)
from common.repo.jobs import UNSET, duration_histograms, get_job, list_jobs, queue_depth
from common.repo.media import total_bytes
from common.repo.planner_runs import list_task_runs
from common.repo.transcripts import latest_whisper_rtf
from services.api import schemas
from services.api.deps import UNAVAILABLE_BODY, get_conn, rate_limit

router = APIRouter()
_log = structlog.get_logger(__name__)

KINDS = ("ingest", "transcribe", "analyze")
STATES = ("pending", "running", "done", "dead")

type QueueDepth = dict[str, dict[str, int]]

JOB_NOT_FOUND = {"detail": "job not found"}
MAX_JOBS_LIMIT = 200
_MAX_JOB_ID = 9_223_372_036_854_775_807

_ERROR_CLASS_VALUES = tuple(cls.value for cls in ErrorClass)
_KIND_VALUES = tuple(sorted(VALID_KINDS))


def _zero_filled(depth: dict[tuple[str, str], int]) -> QueueDepth:
    queue: QueueDepth = {kind: dict.fromkeys(STATES, 0) for kind in KINDS}
    for (kind, state), count in depth.items():
        queue.setdefault(kind, {})[state] = count
    return queue


@router.get("/healthz", response_model=schemas.HealthOut)
def healthz(
    conn: Annotated[psycopg.Connection[Any], Depends(get_conn)],
) -> dict[str, object] | JSONResponse:
    """Liveness plus queue depth by kind and state; 503 when the database is not usable."""
    try:
        conn.read_only = True
        depth = queue_depth(conn)
    except Exception:  # noqa: BLE001 - any database failure is "unavailable", never a 500
        _log.exception("healthz database check failed")
        return JSONResponse(status_code=503, content=UNAVAILABLE_BODY)
    return {"status": "ok", "queue": _zero_filled(depth)}


METRICS_PATH = "/metrics"
#: The fixed 503 body of ``/metrics``: no exception text, DSN or host (#60).
METRICS_UNAVAILABLE_BODY = "database unavailable\n"


def metrics_unavailable() -> PlainTextResponse:
    return PlainTextResponse(METRICS_UNAVAILABLE_BODY, status_code=503)


def _read_snapshot(conn: psycopg.Connection[Any]) -> metrics.MetricsSnapshot:
    """One statement per family, all inside one ``REPEATABLE READ, READ ONLY`` transaction."""
    conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
    conn.read_only = True
    try:
        return metrics.MetricsSnapshot(
            queue_depth=queue_depth(conn),
            job_durations=duration_histograms(conn),
            whisper_rtf=latest_whisper_rtf(conn),
            token_usage=token_usage_by_model(conn),
            llm_cost_usd=total_cost_usd(conn),
            speaker_coercions=total_speaker_coercions(conn),
            audio_bytes=total_bytes(conn),
            reanalysis_backlog=reanalysis_backlog_count(conn, get_settings().PROMPT_VERSION),
            planner_task_runs=list_task_runs(conn),
        )
    finally:
        conn.rollback()


# Not in the OpenAPI schema: an operational endpoint, not part of the generated
# client contract (#46). That also hides it from the all-routes 401 test in
# tests/services/api/test_app.py, so its 401 is covered by
# test_metrics_returns_401_when_require_auth_denies in test_routes_metrics.py.
@router.get(METRICS_PATH, response_class=PlainTextResponse, include_in_schema=False)
def get_metrics(
    conn: Annotated[psycopg.Connection[Any], Depends(get_conn)],
) -> Response:
    """Prometheus text format 0.0.4, derived from Postgres at scrape time (#60)."""
    try:
        snapshot = _read_snapshot(conn)
    except Exception:  # noqa: BLE001 - any database failure is "unavailable", never a 500
        _log.exception("metrics database query failed")
        return metrics_unavailable()
    return PlainTextResponse(metrics.render(snapshot), media_type=metrics.CONTENT_TYPE)


class JobsQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: Literal[STATES] | None = None  # type: ignore[valid-type]
    kind: Literal[_KIND_VALUES] | None = None  # type: ignore[valid-type]
    error_class: Literal[(*_ERROR_CLASS_VALUES, "none")] | None = None  # type: ignore[valid-type]
    before_id: int | None = Field(None, ge=1, le=_MAX_JOB_ID)
    limit: int = Field(50, ge=1, le=MAX_JOBS_LIMIT)


def _gate(model: type[BaseModel]) -> Callable[[Request], None]:
    """Validate the query string and *raise* on failure, before ``get_conn`` resolves.

    A repeated query param (``?state=dead&state=done``) is rejected here: a
    bare dict of ``request.query_params`` keeps only the last value, which
    this method checks for and turns into a 422 rather than silently
    filtering on it.
    """

    def gate(request: Request) -> None:
        repeated = [
            key
            for key in model.model_fields
            if len(request.query_params.getlist(key)) > 1
        ]
        if repeated:
            raise RequestValidationError(
                [
                    {"loc": ("query", key), "msg": "duplicated", "type": "value_error"}
                    for key in repeated
                ]
            )
        try:
            model.model_validate(dict(request.query_params))
        except ValidationError as exc:
            errors = exc.errors(include_input=False, include_context=False, include_url=False)
            raise RequestValidationError(
                [{**error, "loc": ("query", *error["loc"])} for error in errors]
            ) from None

    return gate


@router.get("/ops/jobs", response_model=schemas.JobListOut)
def get_ops_jobs(
    _: Annotated[None, Depends(_gate(JobsQuery))],
    params: Annotated[JobsQuery, Query()],
    conn: Annotated[psycopg.Connection[Any], Depends(get_conn)],
) -> schemas.JobListOut | JSONResponse:
    """A page of ``jobs``, newest first (#44). ``payload`` is never exposed."""
    error_class = UNSET if params.error_class is None else params.error_class
    if params.error_class == "none":
        error_class = None
    try:
        conn.read_only = True
        rows = list_jobs(
            conn,
            state=params.state,
            kind=params.kind,
            error_class=error_class,
            before_id=params.before_id,
            limit=params.limit + 1,
        )
    except Exception:  # noqa: BLE001 - any database failure is "unavailable", never a 500
        _log.exception("ops.jobs database query failed")
        return JSONResponse(status_code=503, content=UNAVAILABLE_BODY)
    has_more = len(rows) > params.limit
    page = rows[: params.limit]
    return schemas.JobListOut(
        items=[schemas.job_out(row) for row in page],
        next_before_id=page[-1].id if has_more else None,
    )


def _retry_job_id(id: int) -> int:
    """The path id, validated before any dependency opens a connection."""
    if not 1 <= id <= _MAX_JOB_ID:
        raise RequestValidationError(
            [
                {
                    "loc": ("path", "id"),
                    "msg": f"id must be between 1 and {_MAX_JOB_ID}",
                    "type": "value_error",
                }
            ]
        )
    return id


def _require_json_content_type(request: Request) -> None:
    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json")


@router.post(
    "/ops/jobs/{id}/retry",
    response_model=schemas.JobOut,
    dependencies=[Depends(rate_limit), Depends(_require_json_content_type)],
)
def post_ops_job_retry(
    id: Annotated[int, Depends(_retry_job_id)],
    conn: Annotated[psycopg.Connection[Any], Depends(get_conn)],
) -> schemas.JobOut | JSONResponse:
    """Revive one ``dead`` job in place: same id, ``attempts`` reset to 0 (#44).

    One conditional ``UPDATE`` (``PostgresQueue.retry_dead``) decides and
    writes the new state together, so a concurrent change can only ever be
    caught, never missed. The one log line, ``ops.job_retried``, is emitted
    only on success; a refusal writes nothing and leaves no trace beyond the
    response.
    """
    outcome = PostgresQueue(conn).retry_dead(id)
    if outcome.status == "not_found":
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=JOB_NOT_FOUND)
    if outcome.status == "not_dead":
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": "job is not dead", "state": outcome.state},
        )
    if outcome.status == "superseded":
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "detail": "superseded by a newer job",
                "job_id": outcome.newer_job_id,
            },
        )
    assert outcome.job is not None
    job = outcome.job
    row = get_job(conn, job.id)
    assert row is not None
    _log.warning(
        "ops.job_retried",
        job_id=job.id,
        video_id=job.video_id,
        kind=job.kind,
        prev_error_class=job.error_class,
        prev_attempts=outcome.prev_attempts,
    )
    return schemas.job_out(row)


class BulkRetryBody(BaseModel):
    """``POST /ops/jobs/retry`` body (#113): strict types, no unknown keys.

    A typo such as ``error_clas`` must be a 422, never a silently wider
    filter. ``state`` exists only so a client can send the listing's filter
    unchanged; ``None`` is the same as omitting a key.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    limit: int = Field(ge=1, le=MAX_BULK_RETRY)
    state: Literal["dead"] | None = None
    kind: Literal[_KIND_VALUES] | None = None  # type: ignore[valid-type]
    error_class: Literal[(*_ERROR_CLASS_VALUES, "none")] | None = None  # type: ignore[valid-type]
    dry_run: bool = False


class BulkRetryOut(BaseModel):
    retried: int
    job_ids: list[int]
    has_more: bool
    dry_run: bool


BULK_CONFLICT = {"detail": "concurrent change, try again"}


async def _bulk_retry_gate(request: Request) -> None:
    """Validate the body and *raise* before ``get_conn`` resolves: a 422 never
    checks out a connection, and its message never echoes a submitted value."""
    try:
        raw = await request.json()
    except ValueError:
        raise RequestValidationError(
            [{"loc": ("body",), "msg": "invalid JSON", "type": "json_invalid"}]
        ) from None
    try:
        BulkRetryBody.model_validate(raw)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_context=False, include_url=False)
        raise RequestValidationError(
            [{**error, "loc": ("body", *error["loc"])} for error in errors]
        ) from None


@router.post(
    "/ops/jobs/retry",
    response_model=BulkRetryOut,
    dependencies=[
        Depends(rate_limit),
        Depends(_require_json_content_type),
        Depends(_bulk_retry_gate),
    ],
)
def post_ops_jobs_retry(
    body: BulkRetryBody,
    conn: Annotated[psycopg.Connection[Any], Depends(get_conn)],
) -> BulkRetryOut | JSONResponse:
    """Re-queue up to ``limit`` ``dead`` jobs matching the filters (#113).

    One statement in ``PostgresQueue.retry_dead_matching`` selects and
    updates; ``UniqueViolation`` (a concurrent ``enqueue`` won a key) retries
    nothing and answers 409.
    """
    error_class = UNSET if body.error_class is None else body.error_class
    if body.error_class == "none":
        error_class = None
    try:
        result = PostgresQueue(conn).retry_dead_matching(
            kind=body.kind, error_class=error_class, limit=body.limit, dry_run=body.dry_run
        )
    except psycopg.errors.UniqueViolation:
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content=BULK_CONFLICT)
    filters = {"kind": body.kind, "error_class": body.error_class, "limit": body.limit}
    if body.dry_run:
        _log.info(
            "ops.jobs_bulk_retry_dry_run",
            would_retry=len(result.jobs),
            has_more=result.has_more,
            **filters,
        )
    else:
        for job in result.jobs:
            _log.warning(
                "ops.job_retried",
                job_id=job.job_id,
                video_id=job.video_id,
                kind=job.kind,
                prev_error_class=job.error_class,
                prev_attempts=job.prev_attempts,
                bulk=True,
            )
        _log.warning(
            "ops.jobs_bulk_retried",
            retried=len(result.jobs),
            has_more=result.has_more,
            **filters,
        )
    return BulkRetryOut(
        retried=len(result.jobs),
        job_ids=result.job_ids,
        has_more=result.has_more,
        dry_run=body.dry_run,
    )
