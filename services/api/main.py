"""HTTP API entrypoint (issue #39; architecture.md 2, 8.3; D13).

``uvicorn services.api.main:app`` serves the app built by ``create_app()``.

The API does no processing of its own: it imports only ``common.*`` (never
``adapters.*``, yt-dlp or faster-whisper), opens no database connection at
import (its pool opens in the lifespan without waiting, #95), and never runs
migrations. Routes that touch the database are sync ``def``, so their blocking psycopg calls run in the threadpool.

Every route passes through ``deps.require_auth`` (attached once, here); write
routes also pass through ``deps.rate_limit``. Errors never reflect input: an
unhandled exception is a bare ``500 {"detail": "internal error"}``, a
validation error lists only ``loc``/``msg``/``type`` (an unknown key's
name is dropped from ``loc``), and there is no CORS
middleware (same-origin through nginx, §9).

``RequestContextMiddleware`` gives each request a ``request_id`` (the
incoming ``X-Request-Id`` if it is safe, else a fresh one), binds it to every
log line emitted while the request is handled, echoes it in the response,
and logs one ``request`` line per request. It also turns uncaught exceptions
into the generic 500 itself: Starlette's own ``Exception`` handler runs
outside every user middleware, so its response would carry no request id.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any, cast

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from common.config import get_settings
from common.db import create_pool
from common.logging import bound_context, configure_logging
from services.api import routes_ops, routes_read, routes_write
from services.api.deps import (
    CONNECT_TIMEOUT_SEC,
    READ_UNAVAILABLE_BODY,
    STATEMENT_TIMEOUT_MS,
    UNAVAILABLE_BODY,
    DatabaseUnavailable,
    ReadDatabaseUnavailable,
    require_auth,
)

_log = structlog.get_logger(__name__)

INTERNAL_ERROR_BODY = {"detail": "internal error"}
_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")
_HEADER = b"x-request-id"


def _request_id(scope: Scope) -> str:
    for name, value in scope.get("headers", []):
        if name == _HEADER:
            try:
                candidate = value.decode("ascii")
            except UnicodeDecodeError:
                break
            if _REQUEST_ID.fullmatch(candidate):
                return str(candidate)
            break
    return uuid.uuid4().hex


class RequestContextMiddleware:
    """Pure ASGI, so the context also covers requests whose handler raised."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id(scope)
        status = 500
        started_response = False
        started = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status, started_response
            if message["type"] == "http.response.start":
                started_response = True
                status = message["status"]
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() != _HEADER
                ]
                headers.append((_HEADER, request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        with bound_context(request_id=request_id):
            try:
                await self.app(scope, receive, send_with_id)
            except Exception:
                _log.exception("unhandled error")
                if started_response:
                    raise
                response = JSONResponse(status_code=500, content=INTERNAL_ERROR_BODY)
                await response(scope, receive, send_with_id)
            finally:
                _log.info(
                    "request",
                    method=scope["method"],
                    path=scope["path"],
                    status=status,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(get_settings())
    pool = create_pool(
        connect_timeout=CONNECT_TIMEOUT_SEC, statement_timeout_ms=STATEMENT_TIMEOUT_MS
    )
    # wait=False: the app starts, and /healthz answers 503, while Postgres is
    # unreachable; the pool keeps retrying in the background.
    pool.open(wait=False)
    app.state.pool = pool
    try:
        yield
    finally:
        pool.close()


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    errors = cast(RequestValidationError, exc).errors()
    detail = [
        {
            "loc": _safe_loc(error),
            "msg": error.get("msg", ""),
            "type": error.get("type", ""),
        }
        for error in errors
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


def _safe_loc(error: Any) -> list[Any]:
    """An unknown key's name is submitted text, so it is dropped from ``loc``."""
    loc = list(error.get("loc", ()))
    if error.get("type") == "extra_forbidden" and loc:
        loc = loc[:-1]
    return loc


async def _database_unavailable(request: Request, exc: Exception) -> Response:
    _log.error("database unavailable", exc_info=exc)
    if request.url.path == routes_ops.METRICS_PATH:
        return routes_ops.metrics_unavailable()
    read = isinstance(exc, ReadDatabaseUnavailable) or routes_read.is_read_request(request)
    return JSONResponse(
        status_code=503, content=READ_UNAVAILABLE_BODY if read else UNAVAILABLE_BODY
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="ytdigest",
        docs_url=None,
        redoc_url=None,
        lifespan=_lifespan,
        dependencies=[Depends(require_auth)],
    )
    app.include_router(routes_read.router)
    app.include_router(routes_write.router)
    app.include_router(routes_ops.router)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(DatabaseUnavailable, _database_unavailable)
    app.add_middleware(RequestContextMiddleware)
    return app


app = create_app()

# uvicorn logs "Started server process" right after importing this module and
# before the lifespan runs, and its own config has just given its loggers
# plain-text handlers. Configure logging at import so that line, and every
# later uvicorn line, is JSON; the lifespan call stays for in-process servers.
# Without settings (tooling that merely imports the module) there is nothing
# to configure; a real server fails loudly in the lifespan instead.
with suppress(ValidationError):
    configure_logging(get_settings())
