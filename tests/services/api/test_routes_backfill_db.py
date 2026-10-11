"""``POST /channels/{channel_id}/backfill`` end to end on Postgres (issue #41).

The real ``get_conn`` is used; only the catalog is faked.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from psycopg.pq import TransactionStatus

from common.config import get_settings
from common.models import CatalogEntry, ChannelCatalog
from common.queue import PRIORITY_BACKFILL
from services.api import deps, routes_write
from services.api.backfill import SubtitleProbe
from services.api.main import create_app
from tests.services.api.conftest import LogSink

pytestmark = pytest.mark.integration

CHANNEL = "UCuAXFkgsw1L7xaCfnd5JJOw"
class _FakeMetadata:
    def fetch(self, video_id: str) -> Any:
        return video_id


NO_PROBE = SubtitleProbe(  # a fake: never yt-dlp
    metadata_factory=lambda timeout: _FakeMetadata(),
    has_manual=lambda meta: False,
)
ENTRIES = tuple(CatalogEntry(f"rt{i:09d}", f"secret title {i}", 60) for i in range(3))


class Catalog:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def list_uploads(self, channel_id: str, *, limit: int) -> ChannelCatalog:
        self.calls.append((channel_id, limit))
        return ChannelCatalog(channel_id, ENTRIES[:limit], 3)


@pytest.fixture
def db(head_dsn: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[psycopg.Connection[Any]]:
    monkeypatch.setenv("DATABASE_URL", head_dsn)
    get_settings.cache_clear()
    with psycopg.connect(head_dsn, autocommit=True) as conn:
        yield conn


@pytest.fixture
def catalog() -> Catalog:
    return Catalog()


@pytest.fixture
def api(db: psycopg.Connection[Any], catalog: Catalog, logs: LogSink) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[deps.get_catalog] = lambda: catalog
    app.dependency_overrides[routes_write.get_subtitle_probe] = lambda: NO_PROBE
    with TestClient(app) as client:
        yield client


def _counts(db: psycopg.Connection[Any]) -> tuple[int, int, int]:
    row = db.execute(
        "SELECT (SELECT count(*) FROM channels), (SELECT count(*) FROM videos),"
        " (SELECT count(*) FROM jobs)"
    ).fetchone()
    assert row is not None
    return (int(row[0]), int(row[1]), int(row[2]))


def _register(db: psycopg.Connection[Any]) -> None:
    db.execute("INSERT INTO channels (channel_id) VALUES (%s)", (CHANNEL,))


def test_an_unregistered_channel_is_404_and_the_catalog_is_not_called(
    api: TestClient, db: psycopg.Connection[Any], catalog: Catalog
) -> None:
    response = api.post(f"/channels/{CHANNEL}/backfill", json={"limit": 5, "dry_run": False})

    assert response.status_code == 404
    assert catalog.calls == []
    assert _counts(db) == (0, 0, 0)


def test_a_dry_run_is_200_and_writes_nothing(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    _register(db)
    before = _counts(db)

    response = api.post(f"/channels/{CHANNEL}/backfill", json={"limit": 5})

    assert response.status_code == 200
    body = response.json()
    assert (body["dry_run"], body["new_videos"], body["enqueued"]) == (True, 3, 0)
    assert body["backfill_id"] is None
    assert _counts(db) == before


def test_a_real_run_is_202_then_200_on_repeat(
    api: TestClient, db: psycopg.Connection[Any], logs: LogSink
) -> None:
    _register(db)

    first = api.post(f"/channels/{CHANNEL}/backfill", json={"limit": 5, "dry_run": False})
    second = api.post(f"/channels/{CHANNEL}/backfill", json={"limit": 5, "dry_run": False})

    assert first.status_code == 202
    assert first.json()["enqueued"] == 3
    assert second.status_code == 200
    assert (second.json()["already_known"], second.json()["enqueued"]) == (3, 0)
    jobs = db.execute("SELECT priority, payload FROM jobs ORDER BY id").fetchall()
    assert jobs == [
        (PRIORITY_BACKFILL, {"origin": "backfill", "backfill_id": first.json()["backfill_id"]})
    ] * 3

    lines = [r for r in logs.records() if r.get("event") == "backfill"]
    assert len(lines) == 2
    for line, response in zip(lines, (first, second), strict=True):
        body = response.json()
        assert line["channel_id"] == CHANNEL
        assert line["limit"] == 5
        assert line["dry_run"] is False
        assert line["listed"] == body["listed"]
        assert line["new_videos"] == body["new_videos"]
        assert line["enqueued"] == body["enqueued"]
        assert line["backfill_id"] == body["backfill_id"]
        assert isinstance(line["duration_ms"], int | float)
        assert line["request_id"]
    assert "secret title" not in logs.text()


def test_the_catalog_runs_on_an_idle_request_connection(
    db: psycopg.Connection[Any], logs: LogSink
) -> None:
    _register(db)
    seen: list[TransactionStatus] = []
    holder: list[psycopg.Connection[Any]] = []

    def capture_conn(request: Request) -> Iterator[psycopg.Connection[Any]]:
        for conn in deps.get_conn(request):
            holder.append(conn)
            yield conn

    class Watching(Catalog):
        def list_uploads(self, channel_id: str, *, limit: int) -> ChannelCatalog:
            seen.append(holder[0].info.transaction_status)
            return super().list_uploads(channel_id, limit=limit)

    app = create_app()
    app.dependency_overrides[deps.get_conn] = capture_conn
    app.dependency_overrides[deps.get_catalog] = Watching
    app.dependency_overrides[routes_write.get_subtitle_probe] = lambda: NO_PROBE
    with TestClient(app) as client:
        response = client.post(f"/channels/{CHANNEL}/backfill", json={"limit": 5, "dry_run": False})

    assert response.status_code == 202
    assert seen == [TransactionStatus.IDLE]


def test_a_permanent_source_error_is_an_empty_200_and_changes_nothing(
    db: psycopg.Connection[Any], logs: LogSink
) -> None:
    from common.errors import PermanentSourceError

    _register(db)
    before = (_counts(db), db.execute("SELECT * FROM channels").fetchall())

    class Gone:
        def list_uploads(self, channel_id: str, *, limit: int) -> ChannelCatalog:
            raise PermanentSourceError("removed", "The playlist does not exist")

    app = create_app()
    app.dependency_overrides[deps.get_catalog] = Gone
    app.dependency_overrides[routes_write.get_subtitle_probe] = lambda: NO_PROBE
    with TestClient(app) as client:
        response = client.post(
            f"/channels/{CHANNEL}/backfill", json={"limit": 5, "dry_run": False}
        )

    assert response.status_code == 200
    body = response.json()
    assert (body["listed"], body["new_videos"], body["enqueued"]) == (0, 0, 0)
    assert "removed" not in response.text
    assert (_counts(db), db.execute("SELECT * FROM channels").fetchall()) == before
