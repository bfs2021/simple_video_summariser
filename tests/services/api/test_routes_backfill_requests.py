"""``POST/GET /channels/{id}/backfill-requests``: HTTP layer, no Postgres (issue #115).

``get_conn`` yields a fake and the two repo calls the POST route makes are
replaced, so validation, the 404/409/422 mapping and the response shapes are
tested without a database. ``tests/services/api/test_routes_backfill_requests_db.py``
runs the same routes on the real app and Postgres.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg.pq import TransactionStatus

from adapters.youtube.catalog import MAX_CATALOG_LIMIT
from common.repo.backfill_requests import CreateOutcome
from services.api import deps, routes_write
from services.api.backfill import MAX_BACKFILL_REQUEST_LIMIT

CHANNEL = "UCuAXFkgsw1L7xaCfnd5JJOw"
MARKER = "ZZMARKER42"


class _Info:
    transaction_status = TransactionStatus.IDLE


class _Conn:
    info = _Info()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        yield


class Repo:
    def __init__(self) -> None:
        self.registered = True
        self.outcome = CreateOutcome(41, True)
        self.created: list[dict[str, Any]] = []

    def channel_exists(self, conn: Any, channel_id: str) -> bool:
        return self.registered

    def create_request(
        self, conn: Any, channel_id: str, *, limit_n: int, dry_run: bool
    ) -> CreateOutcome:
        self.created.append({"channel_id": channel_id, "limit_n": limit_n, "dry_run": dry_run})
        return self.outcome


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> Repo:
    fake = Repo()
    monkeypatch.setattr(routes_write, "channel_exists", fake.channel_exists)
    monkeypatch.setattr(routes_write, "create_request", fake.create_request)
    return fake


@pytest.fixture
def api(app: FastAPI, repo: Repo) -> Iterator[TestClient]:
    def fake_conn() -> Iterator[_Conn]:
        yield _Conn()

    app.dependency_overrides[deps.get_conn] = fake_conn
    with TestClient(app) as client:
        yield client


def _url(channel: str = CHANNEL) -> str:
    return f"/channels/{channel}/backfill-requests"


def test_the_cap_is_the_catalog_adapters_cap() -> None:
    assert MAX_BACKFILL_REQUEST_LIMIT == MAX_CATALOG_LIMIT == 5000


@pytest.mark.parametrize("limit", [1, 5000])
def test_limit_boundaries_are_accepted_and_dry_run_defaults_to_true(
    api: TestClient, repo: Repo, limit: int
) -> None:
    response = api.post(_url(), json={"limit": limit})

    assert response.status_code == 202
    assert response.json() == {
        "request_id": 41,
        "channel_id": CHANNEL,
        "limit": limit,
        "dry_run": True,
        "state": "pending",
    }
    assert repo.created == [{"channel_id": CHANNEL, "limit_n": limit, "dry_run": True}]


def test_an_explicit_real_run_is_recorded_as_such(api: TestClient, repo: Repo) -> None:
    response = api.post(_url(), json={"limit": 9, "dry_run": False})

    assert response.status_code == 202 and response.json()["dry_run"] is False
    assert repo.created[0]["dry_run"] is False


@pytest.mark.parametrize(
    "body",
    [
        {"limit": 0},
        {"limit": 5001},
        {"limit": -1},
        {"limit": True},
        {"limit": "25"},
        {"limit": 2.0},
        {"limit": None},
        {},
        {"limit": 5, "dry_run": "false"},
        {"limit": 5, "dry_run": 0},
        {"limit": 5, "dry_run": None},
        {"limit": 5, "extra": MARKER},
    ],
)
def test_invalid_bodies_are_422_without_echo_and_write_nothing(
    api: TestClient, repo: Repo, body: dict[str, Any]
) -> None:
    response = api.post(_url(), json=body)

    assert response.status_code == 422
    assert MARKER not in response.text
    assert repo.created == []


def test_an_invalid_json_body_is_422(api: TestClient, repo: Repo) -> None:
    response = api.post(_url(), content=f"{{{MARKER}", headers={"content-type": "application/json"})

    assert response.status_code == 422 and MARKER not in response.text
    assert repo.created == []


@pytest.mark.parametrize(
    "channel",
    ["UC" + "a" * 21, "UC" + "a" * 23, "@" + MARKER, CHANNEL + "%0A", "UC" + MARKER + "a" * 11],
)
def test_a_non_canonical_channel_id_is_422_with_the_backfill_message(
    api: TestClient, repo: Repo, channel: str
) -> None:
    response = api.post(_url(channel), json={"limit": 5})

    assert response.status_code == 422
    assert response.json() == {"detail": routes_write.BAD_BACKFILL_CHANNEL}
    assert MARKER not in response.text
    assert repo.created == []


def test_an_unknown_channel_is_404_with_the_register_first_message(
    api: TestClient, repo: Repo
) -> None:
    repo.registered = False

    response = api.post(_url(), json={"limit": 5})

    assert response.status_code == 404
    assert response.json() == {"detail": routes_write.CHANNEL_NOT_REGISTERED}
    assert repo.created == []


def test_a_channel_with_an_active_request_is_409_naming_it(
    api: TestClient, repo: Repo
) -> None:
    repo.outcome = CreateOutcome(7, False)

    response = api.post(_url(), json={"limit": 5})

    assert response.status_code == 409
    assert response.json() == {"detail": routes_write.BACKFILL_REQUEST_ACTIVE, "request_id": 7}


def test_the_route_does_not_import_the_catalog_or_yt_dlp() -> None:
    import subprocess
    import sys

    code = (
        "import sys; import services.api.main; "
        "bad = [m for m in sys.modules if m.startswith(('adapters.', 'yt_dlp'))]; "
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# --- GET: validation happens before any connection is taken ---------------------------


@pytest.fixture
def no_db_api(app: FastAPI) -> Iterator[TestClient]:
    def forbidden() -> Iterator[None]:
        pytest.fail("a rejected request must not take a connection")
        yield None

    app.dependency_overrides[deps.get_conn] = forbidden
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize(
    "request_id",
    ["0", "-1", "abc", "1.5", "1e3", "+1", "0x10", "9" * 20, "9223372036854775808", MARKER, "%20", "٣"],
)
def test_a_bad_request_id_is_422_without_echo_or_a_connection(
    no_db_api: TestClient, request_id: str
) -> None:
    response = no_db_api.get(f"{_url()}/{request_id}")

    assert response.status_code == 422
    assert MARKER not in response.text and request_id not in response.text.replace("\\u", "")


@pytest.mark.parametrize("channel", ["UC" + "a" * 21, "@" + MARKER, "UC" + MARKER + "a" * 11])
def test_a_bad_channel_id_on_get_is_422_without_a_connection(
    no_db_api: TestClient, channel: str
) -> None:
    response = no_db_api.get(f"{_url(channel)}/1")

    assert response.status_code == 422
    assert MARKER not in response.text
