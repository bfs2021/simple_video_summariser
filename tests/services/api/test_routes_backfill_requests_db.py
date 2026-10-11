"""Backfill requests end to end on the real app and Postgres (issue #115).

Real ``create_app`` (real route table, real pool); only the catalog is faked.
The 409 race is forced with a held transaction and ``pg_stat_activity``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from common.config import Settings, get_settings
from common.models import CatalogEntry, ChannelCatalog
from common.queue import PostgresQueue
from common.repo.backfill_requests import create_request
from services.api import routes_write
from services.api.main import create_app
from services.planner.backfill import run_channel_backfill

pytestmark = pytest.mark.integration

CHANNEL = "UCuAXFkgsw1L7xaCfnd5JJOw"
OTHER = "UCxxxxxxxxxxxxxxxxxxxxxx"


class Catalog:
    def __init__(self, n: int) -> None:
        self.entries = tuple(CatalogEntry(f"rq{i:09d}", f"t{i}", 60) for i in range(n))

    def list_uploads(self, channel_id: str, *, limit: int) -> ChannelCatalog:
        return ChannelCatalog(channel_id, self.entries[:limit], len(self.entries))


@pytest.fixture
def db(head_dsn: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[psycopg.Connection[Any]]:
    monkeypatch.setenv("DATABASE_URL", head_dsn)
    get_settings.cache_clear()
    with psycopg.connect(head_dsn, autocommit=True) as conn:
        conn.execute("INSERT INTO channels (channel_id) VALUES (%s)", (CHANNEL,))
        yield conn


@pytest.fixture
def api(db: psycopg.Connection[Any]) -> Iterator[TestClient]:
    with TestClient(create_app()) as client:
        yield client


def _post(api: TestClient, channel: str = CHANNEL, **body: Any) -> Any:
    return api.post(f"/channels/{channel}/backfill-requests", json={"limit": 5, **body})


def test_post_then_get_through_the_real_app(api: TestClient, db: psycopg.Connection[Any]) -> None:
    created = _post(api)
    assert created.status_code == 202
    rid = created.json()["request_id"]

    got = api.get(f"/channels/{CHANNEL}/backfill-requests/{rid}")

    assert got.status_code == 200
    body = got.json()
    assert (body["state"], body["attempts"], body["enqueued"], body["result"]) == (
        "pending", 0, 0, None
    )
    assert body["error_class"] is None and body["finished_at"] is None
    assert api.get(f"/channels/{CHANNEL}/backfill-requests/{rid + 1}").status_code == 404
    assert api.get(f"/channels/{OTHER}/backfill-requests/{rid}").status_code == 404


def test_unknown_channel_404_and_second_post_409_with_one_row(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    assert _post(api, OTHER).status_code == 404
    first = _post(api)

    second = _post(api, limit=9)

    assert second.status_code == 409
    assert second.json()["request_id"] == first.json()["request_id"]
    assert db.execute("SELECT count(*) FROM backfill_requests").fetchone() == (1,)


def test_a_post_racing_an_uncommitted_request_waits_then_gets_409(
    api: TestClient, db: psycopg.Connection[Any], head_dsn: str
) -> None:
    out: dict[str, Any] = {}
    with psycopg.connect(head_dsn) as holder:
        held = create_request(holder, CHANNEL, limit_n=3, dry_run=True)  # uncommitted

        thread = threading.Thread(target=lambda: out.update(r=_post(api)))
        thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            row = db.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock'"
                " AND query ILIKE '%INSERT INTO backfill_requests%' AND pid <> pg_backend_pid()"
            ).fetchone()
            if row is not None and row[0] == 1:
                break
        else:
            pytest.fail("the POST never blocked on the uncommitted request")
        holder.commit()
        thread.join(10)
    assert out["r"].status_code == 409 and out["r"].json()["request_id"] == held.request_id
    assert db.execute("SELECT count(*) FROM backfill_requests").fetchone() == (1,)


def test_planner_result_is_served_and_cancel_ends_the_request(
    api: TestClient, db: psycopg.Connection[Any], head_dsn: str
) -> None:
    rid = _post(api, limit=5000, dry_run=False).json()["request_id"]
    with psycopg.connect(head_dsn) as conn:
        run_channel_backfill(
            conn, PostgresQueue(conn), Catalog(600), settings=Settings(DATABASE_URL=head_dsn)  # type: ignore[arg-type]
        )
    body = api.get(f"/channels/{CHANNEL}/backfill-requests/{rid}").json()
    assert (body["state"], body["enqueued"]) == ("running", 500)
    assert body["result"]["enqueued"] == 500 and body["result"]["new_videos"] == 600

    cancelled = api.post(
        f"/channels/{CHANNEL}/backfill/cancel",
        json={"backfill_id": body["result"]["backfill_id"]},
    )

    assert cancelled.status_code == 200 and cancelled.json()["cancelled"] == 500
    after = api.get(f"/channels/{CHANNEL}/backfill-requests/{rid}").json()
    assert (after["state"], after["error_class"]) == ("failed", "CANCELLED")
    assert db.execute("SELECT count(*) FROM jobs").fetchone() == (0,)
    assert _post(api).status_code == 202  # the channel is free again
    assert routes_write.BACKFILL_REQUEST_ACTIVE  # message constant exists
