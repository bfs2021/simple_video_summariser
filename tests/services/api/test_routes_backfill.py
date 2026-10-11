"""``POST /channels/{channel_id}/backfill``: HTTP layer (issue #41).

No Postgres: ``get_conn`` yields a sentinel, ``get_catalog`` a fake, and
``routes_write.backfill`` is replaced so validation, status codes, error
mapping and the response shape are tested without a database.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from psycopg.pq import TransactionStatus

from common.errors import (
    RateLimitedError,
    ToolFailureError,
    TransientNetworkError,
)
from common.models import CatalogSource, ChannelCatalog
from services.api import deps, routes_write
from services.api.backfill import (
    MAX_BACKFILL_LIMIT,
    BackfillResult,
    ChannelNotRegistered,
    Estimate,
    LlmEstimate,
    SubtitleProbe,
    VideoEstimate,
)
from tests.services.api.conftest import LogSink

CHANNEL = "UCuAXFkgsw1L7xaCfnd5JJOw"
MARKER = "ZZMARKER42"
SECRET = "yt-dlp stderr https://www.youtube.com/playlist?list=UUsecret"


class _Info:
    transaction_status = TransactionStatus.IDLE


class _Conn:
    info = _Info()


_PROBE = SubtitleProbe(
    metadata_factory=lambda timeout: (_ for _ in ()).throw(AssertionError("no real probe")),
    has_manual=lambda meta: False,
)


class _Catalog:
    def list_uploads(self, channel_id: str, *, limit: int) -> ChannelCatalog:
        raise AssertionError("the route must not call the catalog itself")


def _result(**overrides: Any) -> BackfillResult:
    values: dict[str, Any] = {
        "channel_id": CHANNEL,
        "dry_run": True,
        "limit": 25,
        "listed": 3,
        "total_count": 120,
        "already_known": 1,
        "new_videos": 2,
        "enqueued": 0,
        "video_ids": ("aaaaaaaaaaa", "bbbbbbbbbbb"),
        "estimate": Estimate(
            600, 1, 0.25, 4, 150, probed=1, manual_subtitles=1, unchecked=1,
            transcribe_audio_sec=300,
            llm=LlmEstimate("anthropic", "m", 13.3, 5, 3e-06, 2, 1000, 0.012),
        ),
        "videos": (VideoEstimate("aaaaaaaaaaa", 300, "manual", 0, 1000, 0.012),),
        "backfill_id": None,
    }
    values.update(overrides)
    return BackfillResult(**values)


class Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = _result()
        self.error: Exception | None = None

    def __call__(
        self,
        conn: Any,
        queue: Any,
        catalog: CatalogSource,
        channel_id: str,
        *,
        limit: int,
        dry_run: bool,
        **extra: Any,
    ) -> BackfillResult:
        self.calls.append(
            {"conn": conn, "queue": queue, "catalog": catalog, "channel_id": channel_id,
             "limit": limit, "dry_run": dry_run, **extra}
        )
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(routes_write, "backfill", rec)
    return rec


@pytest.fixture
def catalog() -> _Catalog:
    return _Catalog()


@pytest.fixture
def conn() -> _Conn:
    return _Conn()


@pytest.fixture
def api(
    app: FastAPI, recorder: Recorder, catalog: _Catalog, conn: _Conn
) -> Iterator[TestClient]:
    def fake_conn() -> Iterator[_Conn]:
        yield conn

    app.dependency_overrides[deps.get_conn] = fake_conn
    app.dependency_overrides[deps.get_catalog] = lambda: catalog
    app.dependency_overrides[routes_write.get_subtitle_probe] = lambda: _PROBE
    with TestClient(app) as client:
        yield client


def _url(channel: str = CHANNEL) -> str:
    return f"/channels/{channel}/backfill"


# --- path ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "channel",
    [
        "UC" + "a" * 21,
        "UC" + "a" * 23,
        "UU" + "a" * 22,
        "@" + MARKER,
        CHANNEL + "%0A",
        CHANNEL[:-1] + "%0A",
        "UC" + "a" * 10 + "%2F" + "a" * 11,
        "UC" + "a" * 10 + "%20" + "a" * 11,
        "UC" + "a" * 21 + "%C3%A9",
        "UC" + MARKER + "a" * 11,
        "UC" + MARKER + "a" * 13,
        "UC" + "a" * 21 + "%00",
        "UC" + "a" * 10 + "%2F" + "a" * 11 + "/x",
    ],
)
def test_a_non_canonical_channel_id_is_rejected_without_echo(
    api: TestClient, recorder: Recorder, channel: str
) -> None:
    response = api.post(_url(channel), json={"limit": 5})

    assert response.status_code == 422
    assert response.json() == {"detail": routes_write.BAD_BACKFILL_CHANNEL}
    assert MARKER not in response.text
    assert recorder.calls == []


def test_a_channel_id_whose_suffix_starts_with_a_dash_is_accepted(
    api: TestClient, recorder: Recorder
) -> None:
    channel = "UC-" + "a" * 21

    assert api.post(_url(channel), json={"limit": 5}).status_code == 200
    assert recorder.calls[0]["channel_id"] == channel


# --- body ---------------------------------------------------------------------


@pytest.mark.parametrize("limit", [1, MAX_BACKFILL_LIMIT])
def test_limit_boundaries_are_accepted(
    api: TestClient, recorder: Recorder, limit: int
) -> None:
    assert MAX_BACKFILL_LIMIT == 500

    response = api.post(_url(), json={"limit": limit})

    assert response.status_code == 200
    assert recorder.calls[0]["limit"] == limit


@pytest.mark.parametrize(
    "body",
    [
        {"limit": 0},
        {"limit": -1},
        {"limit": 501},
        {"limit": True},
        {"limit": 2.0},
        {"limit": "25"},
        {"limit": None},
        {},
        {"dry_run": False},
        {"limit": 5, "dry_run": "false"},
        {"limit": 5, "dry_run": 0},
        {"limit": 5, "dry_run": None},
        {"limit": 5, "dryrun": False},
        {"limit": 5, MARKER: 1},
    ],
)
def test_an_invalid_body_is_422_without_echo(
    api: TestClient, recorder: Recorder, body: dict[str, Any]
) -> None:
    response = api.post(_url(), json=body)

    assert response.status_code == 422
    assert MARKER not in response.text
    for item in response.json()["detail"]:
        assert set(item) == {"loc", "msg", "type"}
    assert recorder.calls == []


@pytest.mark.parametrize(
    "raw",
    [None, b"not json " + MARKER.encode(), b'[{"limit": 5}]', b'"' + MARKER.encode() + b'"'],
)
def test_a_missing_or_non_object_body_is_422_without_echo(
    api: TestClient, recorder: Recorder, raw: bytes | None
) -> None:
    response = api.post(
        _url(), content=raw, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert MARKER not in response.text
    assert recorder.calls == []


def test_dry_run_defaults_to_true(api: TestClient, recorder: Recorder) -> None:
    api.post(_url(), json={"limit": 5})

    assert recorder.calls[0]["dry_run"] is True


def test_dry_run_false_must_be_explicit(api: TestClient, recorder: Recorder) -> None:
    recorder.result = _result(dry_run=False, enqueued=2, backfill_id="b-id")

    response = api.post(_url(), json={"limit": 5, "dry_run": False})

    assert response.status_code == 202
    assert recorder.calls[0]["dry_run"] is False


def test_the_route_passes_the_injected_connection_catalog_and_a_queue_on_it(
    api: TestClient, recorder: Recorder, conn: _Conn, catalog: _Catalog
) -> None:
    api.post(_url(), json={"limit": 9})

    [call] = recorder.calls
    assert call["conn"] is conn
    assert call["catalog"] is catalog
    assert call["channel_id"] == CHANNEL
    assert call["queue"]._conn is conn


# --- response -----------------------------------------------------------------


def test_the_response_has_exactly_the_documented_keys(
    api: TestClient, recorder: Recorder
) -> None:
    response = api.post(_url(), json={"limit": 25})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "channel_id": CHANNEL,
        "dry_run": True,
        "limit": 25,
        "listed": 3,
        "total_count": 120,
        "already_known": 1,
        "new_videos": 2,
        "enqueued": 0,
        "video_ids": ["aaaaaaaaaaa", "bbbbbbbbbbb"],
        "backfill_id": None,
        "videos": [
            {
                "video_id": "aaaaaaaaaaa",
                "duration_sec": 300,
                "subtitles": "manual",
                "transcription_sec": 0,
                "est_input_tokens": 1000,
                "llm_cost_usd": 0.012,
            }
        ],
        "estimate": {
            "audio_sec": 600,
            "unknown_duration": 1,
            "rtf": 0.25,
            "rtf_samples": 4,
            "transcription_sec": 150,
            "assumes": (
                "manual-subtitle videos skip speech-to-text;"
                " unchecked videos are assumed to need it"
            ),
            "probed": 1,
            "manual_subtitles": 1,
            "needs_transcription": 0,
            "unchecked": 1,
            "probe_failed": 0,
            "deadline_reached": False,
            "transcribe_audio_sec": 300,
            "llm": {
                "summarizer": "anthropic",
                "model": "m",
                "chars_per_audio_sec": 13.3,
                "chars_samples": 5,
                "cost_per_char": 3e-06,
                "cost_samples": 2,
                "est_input_tokens": 1000,
                "cost_usd": 0.012,
                "basis": "median recorded analyses.cost_usd per transcript character",
            },
        },
    }


def test_null_estimate_fields_and_total_count_are_serialized_as_null(
    api: TestClient, recorder: Recorder
) -> None:
    recorder.result = _result(total_count=None, estimate=Estimate(0, 0, None, 0, None))

    body = api.post(_url(), json={"limit": 25}).json()

    assert body["total_count"] is None
    assert body["estimate"]["rtf"] is None
    assert body["estimate"]["transcription_sec"] is None


def test_a_real_run_that_found_nothing_new_is_200(
    api: TestClient, recorder: Recorder
) -> None:
    recorder.result = _result(dry_run=False, new_videos=0, enqueued=0, backfill_id="b-id")

    response = api.post(_url(), json={"limit": 5, "dry_run": False})

    assert response.status_code == 200
    assert response.json()["backfill_id"] == "b-id"


# --- errors -------------------------------------------------------------------


def test_an_unregistered_channel_is_404_with_a_fixed_message(
    api: TestClient, recorder: Recorder
) -> None:
    recorder.error = ChannelNotRegistered(CHANNEL)

    response = api.post(_url(), json={"limit": 5, "dry_run": False})

    assert response.status_code == 404
    assert response.json() == {"detail": routes_write.CHANNEL_NOT_REGISTERED}
    assert "POST /channels" in routes_write.CHANNEL_NOT_REGISTERED
    assert CHANNEL not in response.text


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (TransientNetworkError(SECRET), 503, "LISTING_UNAVAILABLE"),
        (TransientNetworkError("yt-dlp timed out after 90s"), 503, "LISTING_UNAVAILABLE"),
        (RateLimitedError(SECRET, retry_after_sec=30), 503, "LISTING_UNAVAILABLE"),
        (ToolFailureError(SECRET), 502, "LISTING_FAILED"),
    ],
)
def test_catalog_failures_map_to_fixed_bodies_and_are_logged_by_class(
    api: TestClient,
    recorder: Recorder,
    logs: LogSink,
    error: Exception,
    status: int,
    detail: str,
) -> None:
    recorder.error = error

    response = api.post(
        _url(), json={"limit": 5}, headers={"X-Request-Id": "req-41"}
    )

    assert response.status_code == status
    assert response.json() == {"detail": getattr(routes_write, detail)}
    assert "yt-dlp" not in response.text
    assert "youtube.com" not in response.text
    [line] = [r for r in logs.records() if r.get("event") == "backfill listing failed"]
    assert line["error_class"] == type(error).__name__
    assert line["request_id"] == "req-41"
    assert line["channel_id"] == CHANNEL
    assert SECRET not in logs.text()


def test_the_rate_limit_hook_covers_the_route(
    app: FastAPI, api: TestClient, recorder: Recorder
) -> None:
    def too_many() -> None:
        raise HTTPException(429, "slow down")

    app.dependency_overrides[deps.rate_limit] = too_many

    assert api.post(_url(), json={"limit": 5}).status_code == 429
    assert recorder.calls == []


def test_the_route_is_registered_under_the_documented_path(app: FastAPI) -> None:
    assert "post" in app.openapi()["paths"]["/channels/{channel_id}/backfill"]


# --- catalog dependency -------------------------------------------------------


def test_get_catalog_builds_a_youtube_catalog_with_the_backfill_timeout() -> None:
    from adapters.youtube.catalog import YouTubeCatalog

    catalog = deps.get_catalog()

    assert isinstance(catalog, YouTubeCatalog)
    assert deps.BACKFILL_LIST_TIMEOUT_SEC == 90
    assert catalog._timeout == 90


def test_a_malformed_channel_id_is_rejected_before_any_connection_or_catalog(
    app: FastAPI, recorder: Recorder
) -> None:
    opened: list[str] = []

    def tracking_conn() -> Iterator[_Conn]:
        opened.append("conn")
        yield _Conn()

    def tracking_catalog() -> _Catalog:
        opened.append("catalog")
        return _Catalog()

    app.dependency_overrides[deps.get_conn] = tracking_conn
    app.dependency_overrides[deps.get_catalog] = tracking_catalog
    app.dependency_overrides[routes_write.get_subtitle_probe] = lambda: _PROBE
    with TestClient(app) as client:
        response = client.post(_url("UU" + "a" * 22), json={"limit": 5})

    assert response.status_code == 422
    assert opened == []
