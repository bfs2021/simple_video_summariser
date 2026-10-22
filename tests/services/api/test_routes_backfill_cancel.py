"""``POST /channels/{channel_id}/backfill/cancel``: HTTP layer (issue #116).

No Postgres: ``get_conn`` yields a sentinel and ``routes_write.cancel_backfill``
is replaced, so validation, status codes, error mapping and the response shape
are tested without a database.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from psycopg.pq import TransactionStatus

from services.api import deps, routes_write
from services.api.backfill import CancelResult, ChannelNotRegistered
from tests.services.api.conftest import LogSink

CHANNEL = "UCuAXFkgsw1L7xaCfnd5JJOw"
BID = "0f8fad5b-d9cb-469f-a165-70867728950e"
MARKER = "ZZMARKER42"


class _Info:
    transaction_status = TransactionStatus.IDLE


class _Conn:
    info = _Info()


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, str, str]] = []
        self.error: Exception | None = None
        self.result = CancelResult(CHANNEL, BID, 3, 2, 1, ("aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"))

    def __call__(self, conn: Any, channel_id: str, backfill_id: str) -> CancelResult:
        self.calls.append((conn, channel_id, backfill_id))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(routes_write, "cancel_backfill", rec)
    return rec


@pytest.fixture
def conn() -> _Conn:
    return _Conn()


@pytest.fixture
def api(app: FastAPI, recorder: Recorder, conn: _Conn) -> Iterator[TestClient]:
    def fake_conn() -> Iterator[_Conn]:
        yield conn

    app.dependency_overrides[deps.get_conn] = fake_conn
    with TestClient(app) as client:
        yield client


def _url(channel: str = CHANNEL) -> str:
    return f"/channels/{channel}/backfill/cancel"


def test_a_valid_request_calls_cancel_with_the_path_and_body(
    api: TestClient, recorder: Recorder, conn: _Conn
) -> None:
    response = api.post(_url(), json={"backfill_id": BID})

    assert response.status_code == 200
    assert recorder.calls == [(conn, CHANNEL, BID)]


def test_the_response_has_exactly_the_documented_keys(api: TestClient) -> None:
    body = api.post(_url(), json={"backfill_id": BID}).json()

    assert body == {
        "channel_id": CHANNEL,
        "backfill_id": BID,
        "cancelled": 3,
        "videos_removed": 2,
        "videos_kept": 1,
        "video_ids": ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"],
    }


def test_the_route_logs_one_warning_without_titles(api: TestClient, logs: LogSink) -> None:
    api.post(_url(), json={"backfill_id": BID})

    [line] = [r for r in logs.records() if r.get("event") == "backfill.cancelled"]
    assert line["level"] == "warning"
    assert line["channel_id"] == CHANNEL
    assert line["backfill_id"] == BID
    assert (line["cancelled"], line["videos_removed"], line["videos_kept"]) == (3, 2, 1)
    assert line["request_id"]


@pytest.mark.parametrize(
    "backfill_id",
    [
        None,
        "",
        BID.upper(),
        BID.replace("-", ""),
        "{" + BID + "}",
        BID[:-1],
        BID + "0",
        BID + "\n",
        " " + BID,
        12345,
        [BID],
        {"x": BID},
        True,
        MARKER + BID[len(MARKER):],
    ],
)
def test_a_malformed_backfill_id_is_422_without_echo(
    api: TestClient, recorder: Recorder, backfill_id: Any
) -> None:
    response = api.post(_url(), json={"backfill_id": backfill_id})

    assert response.status_code == 422
    assert MARKER not in response.text
    assert recorder.calls == []


@pytest.mark.parametrize(
    "body",
    [{}, {"backfill_id": BID, "extra": 1}, {"backfill_id": BID, MARKER: 1}, {"id": BID}],
)
def test_a_missing_or_extra_key_is_422_without_echo(
    api: TestClient, recorder: Recorder, body: dict[str, Any]
) -> None:
    response = api.post(_url(), json=body)

    assert response.status_code == 422
    assert MARKER not in response.text
    assert recorder.calls == []


@pytest.mark.parametrize(
    "raw",
    [None, b"not json " + MARKER.encode(), b'["' + BID.encode() + b'"]', b'"' + MARKER.encode() + b'"'],
)
def test_a_missing_non_json_or_non_object_body_is_422_without_echo(
    api: TestClient, recorder: Recorder, raw: bytes | None
) -> None:
    response = api.post(_url(), content=raw, headers={"Content-Type": "application/json"})

    assert response.status_code == 422
    assert MARKER not in response.text
    assert recorder.calls == []


@pytest.mark.parametrize(
    "channel",
    ["UC" + "a" * 21, "UC" + "a" * 23, "@" + MARKER, CHANNEL + "%0A", "UC" + MARKER + "a" * 11],
)
def test_a_non_canonical_channel_id_is_rejected_without_echo(
    api: TestClient, recorder: Recorder, channel: str
) -> None:
    response = api.post(_url(channel), json={"backfill_id": BID})

    assert response.status_code == 422
    assert response.json() == {"detail": routes_write.BAD_BACKFILL_CHANNEL}
    assert MARKER not in response.text
    assert recorder.calls == []


@pytest.mark.parametrize(
    "content_type", [None, "text/plain", "application/x-www-form-urlencoded", "multipart/form-data"]
)
def test_a_non_json_content_type_is_415_and_cancels_nothing(
    api: TestClient, recorder: Recorder, content_type: str | None
) -> None:
    headers = {} if content_type is None else {"Content-Type": content_type}
    response = api.post(_url(), content=f'{{"backfill_id": "{BID}"}}'.encode(), headers=headers)

    assert response.status_code == 415
    assert recorder.calls == []


def test_a_json_content_type_with_a_charset_is_accepted(api: TestClient) -> None:
    response = api.post(
        _url(),
        content=f'{{"backfill_id": "{BID}"}}'.encode(),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200


def test_an_unregistered_channel_is_404_with_a_fixed_message(
    api: TestClient, recorder: Recorder
) -> None:
    recorder.error = ChannelNotRegistered(CHANNEL)

    response = api.post(_url(), json={"backfill_id": BID})

    assert response.status_code == 404
    assert response.json() == {"detail": routes_write.CHANNEL_NOT_REGISTERED}
    assert CHANNEL not in response.text


def test_the_rate_limit_hook_covers_the_route(
    app: FastAPI, api: TestClient, recorder: Recorder
) -> None:
    def too_many() -> None:
        raise HTTPException(429, "slow down")

    app.dependency_overrides[deps.rate_limit] = too_many

    assert api.post(_url(), json={"backfill_id": BID}).status_code == 429
    assert recorder.calls == []


def test_openapi_documents_pending_only_cancellation(app: FastAPI) -> None:
    operation = app.openapi()["paths"]["/channels/{channel_id}/backfill/cancel"]["post"]

    description = operation["description"]
    assert "active backfill request" in description
    assert "already running" in description
    assert "not included in ``cancelled``" in description
    assert "``transcribe`` and ``analyze`` jobs are not cancelled" in description


def test_the_backfill_route_is_not_shadowed_by_the_cancel_route(
    api: TestClient, recorder: Recorder
) -> None:
    # /backfill with a body for cancel must not reach cancel, and vice versa.
    response = api.post(f"/channels/{CHANNEL}/backfill", json={"backfill_id": BID})

    assert response.status_code == 422
    assert recorder.calls == []
