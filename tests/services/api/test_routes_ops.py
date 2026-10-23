"""``GET /ops/jobs`` and ``POST /ops/jobs/{id}/retry`` (issue #44).

Unit tests (no Docker) fake ``deps.get_conn`` and check parameter validation,
the ``415``/``429`` seams and status-code mapping; the integration tests at
the bottom run both routes end to end against a seeded Postgres.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, Self

import psycopg
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from common.config import get_settings
from common.errors import ErrorClass
from common.queue import PostgresQueue
from common.repo.jobs import write_progress
from services.api import deps
from services.api.main import create_app
from tests.services.api.conftest import LogSink

MARKER = "ZZMARKER42"


class _Info:
    from psycopg.pq import TransactionStatus

    transaction_status = TransactionStatus.IDLE


class FakeConn:
    def __init__(
        self,
        rows: list[tuple[Any, ...]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.rows = rows or []
        self.error = error
        self.read_only = False
        self.queries: list[str] = []
        self.info = _Info()

    def execute(self, query: str, *args: Any, **kwargs: Any) -> FakeConn:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self

    def cursor(self, *args: Any, **kwargs: Any) -> _FakeCursor:
        return _FakeCursor(self)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


class _FakeCursor:
    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, query: str, params: Any = None) -> None:
        self._conn.queries.append(query)
        if self._conn.error is not None:
            raise self._conn.error

    def fetchall(self) -> list[dict[str, Any]]:
        return [_as_dict(row) for row in self._conn.rows]

    def fetchone(self) -> dict[str, Any] | None:
        rows = self.fetchall()
        return rows[0] if rows else None


_COLUMNS = (
    "id",
    "video_id",
    "video_title",
    "kind",
    "dedupe_key",
    "state",
    "priority",
    "attempts",
    "error_class",
    "last_error",
    "run_after",
    "locked_by",
    "heartbeat_at",
    "finished_at",
    "created_at",
    "progress",
    "payload_duration_sec",
)


def _as_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(_COLUMNS, row, strict=True))


def _job_row(**overrides: Any) -> tuple[Any, ...]:
    values: dict[str, Any] = {
        "id": 1,
        "video_id": "dQw4w9WgXcQ",
        "video_title": "Title",
        "kind": "ingest",
        "dedupe_key": "default",
        "state": "dead",
        "priority": 0,
        "attempts": 3,
        "error_class": "TOOL_FAILURE",
        "last_error": "boom",
        "run_after": datetime(2026, 9, 1, 12, tzinfo=UTC),
        "locked_by": None,
        "heartbeat_at": None,
        "finished_at": None,
        "created_at": datetime(2026, 9, 1, 11, tzinfo=UTC),
        "progress": None,
        "payload_duration_sec": None,
    }
    values.update(overrides)
    return tuple(values[c] for c in _COLUMNS)


def _use(app: FastAPI, conn: FakeConn) -> None:
    def override() -> Iterator[FakeConn]:
        yield conn

    app.dependency_overrides[deps.get_conn] = override


def _assert_422_without_echo(response: Any) -> None:
    assert response.status_code == 422, response.text
    assert MARKER not in response.text
    for item in response.json()["detail"]:
        assert set(item) == {"loc", "msg", "type"}


# ---------------------------------------------------------------------------
# GET /ops/jobs: parameter validation (unit, no database)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"state": "DEAD"},
        {"state": ""},
        {"kind": "Ingest"},
        {"kind": ""},
        {"error_class": "tool_failure"},
        {"error_class": ""},
        {"error_class": "NOT_A_CLASS"},
        {"limit": "0"},
        {"limit": "201"},
        {"limit": "-1"},
        {"limit": "abc"},
        {"limit": "1.5"},
        {"before_id": "0"},
        {"before_id": "-1"},
        {"before_id": "9223372036854775808"},
        {"before_id": "abc"},
        {"state": MARKER},
        {"kind": MARKER},
        {"error_class": MARKER},
        {"before_id": MARKER},
    ],
)
def test_bad_parameters_are_422_without_echo(
    app: FastAPI, params: dict[str, str]
) -> None:
    conn = FakeConn()
    _use(app, conn)
    with TestClient(app) as client:
        response = client.get("/ops/jobs", params=params)
    _assert_422_without_echo(response)
    assert conn.queries == []


def test_a_repeated_param_is_422_and_never_filters_silently(app: FastAPI) -> None:
    conn = FakeConn()
    _use(app, conn)
    with TestClient(app) as client:
        response = client.get("/ops/jobs?state=dead&state=done")
    _assert_422_without_echo(response)
    assert conn.queries == []


@pytest.mark.parametrize("limit", [1, 200])
def test_boundary_limits_are_accepted(app: FastAPI, limit: int) -> None:
    conn = FakeConn()
    _use(app, conn)
    with TestClient(app) as client:
        response = client.get("/ops/jobs", params={"limit": limit})
    assert response.status_code == 200


def test_unknown_query_params_are_ignored(app: FastAPI) -> None:
    conn = FakeConn([_job_row()])
    _use(app, conn)
    with TestClient(app) as client:
        response = client.get("/ops/jobs", params={"bogus": "x"})
    assert response.status_code == 200


@pytest.mark.parametrize("value", [cls.value for cls in ErrorClass])
def test_every_error_class_value_is_accepted(app: FastAPI, value: str) -> None:
    conn = FakeConn([])
    _use(app, conn)
    with TestClient(app) as client:
        response = client.get("/ops/jobs", params={"error_class": value})
    assert response.status_code == 200


def test_error_class_none_is_accepted(app: FastAPI) -> None:
    conn = FakeConn([])
    _use(app, conn)
    with TestClient(app) as client:
        response = client.get("/ops/jobs", params={"error_class": "none"})
    assert response.status_code == 200


@pytest.mark.parametrize("kind", ["ingest", "transcribe", "analyze"])
def test_every_kind_value_is_accepted(app: FastAPI, kind: str) -> None:
    conn = FakeConn([])
    _use(app, conn)
    with TestClient(app) as client:
        response = client.get("/ops/jobs", params={"kind": kind})
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# GET /ops/jobs: shape, pagination and database-failure mapping (unit)
# ---------------------------------------------------------------------------


def test_default_response_shape_with_no_rows(app: FastAPI) -> None:
    conn = FakeConn([])
    _use(app, conn)
    with TestClient(app) as client:
        response = client.get("/ops/jobs")
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_before_id": None}


def test_item_shape_omits_payload_and_includes_every_declared_field(
    app: FastAPI,
) -> None:
    conn = FakeConn([_job_row(id=7)])
    _use(app, conn)
    with TestClient(app) as client:
        body = client.get("/ops/jobs").json()
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert set(item) == {
        "id",
        "video_id",
        "video_title",
        "kind",
        "dedupe_key",
        "state",
        "priority",
        "attempts",
        "error_class",
        "last_error",
        "run_after",
        "locked_by",
        "heartbeat_at",
        "finished_at",
        "created_at",
        "progress",
    }
    assert item["id"] == 7
    assert item["progress"] is None


def test_next_before_id_is_none_when_the_page_is_not_full(app: FastAPI) -> None:
    conn = FakeConn([_job_row(id=1)])
    _use(app, conn)
    with TestClient(app) as client:
        body = client.get("/ops/jobs", params={"limit": 5}).json()
    assert body["next_before_id"] is None


def test_next_before_id_is_the_last_items_id_when_more_rows_exist(app: FastAPI) -> None:
    # limit=2 means the route asks the repo layer for 3; the fake returns 3.
    conn = FakeConn([_job_row(id=3), _job_row(id=2), _job_row(id=1)])
    _use(app, conn)
    with TestClient(app) as client:
        body = client.get("/ops/jobs", params={"limit": 2}).json()
    assert len(body["items"]) == 2
    assert body["items"][-1]["id"] == 2
    assert body["next_before_id"] == 2


@pytest.mark.parametrize(
    "error",
    [
        psycopg.OperationalError("connection to 127.0.0.1 failed"),
        psycopg.errors.QueryCanceled("canceling statement due to statement timeout"),
        RuntimeError("boom 127.0.0.1"),
    ],
)
def test_a_database_error_returns_503_without_details_and_is_logged(
    app: FastAPI, logs: LogSink, error: Exception
) -> None:
    conn = FakeConn(error=error)
    _use(app, conn)
    with TestClient(app) as client:
        response = client.get("/ops/jobs")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
    for secret in ("127.0.0.1", "Traceback", "boom"):
        assert secret not in response.text
    failures = [r for r in logs.records() if r.get("level") == "error"]
    assert failures and "exception" in failures[0]


# ---------------------------------------------------------------------------
# POST /ops/jobs/{id}/retry: path validation and content type (unit)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("job_id", ["0", "-1", "abc", "9223372036854775808"])
def test_bad_id_is_422_without_echo(app: FastAPI, job_id: str) -> None:
    conn = FakeConn()
    _use(app, conn)
    with TestClient(app) as client:
        response = client.post(
            f"/ops/jobs/{job_id}/retry", json={}, headers={"Content-Type": "application/json"}
        )
    _assert_422_without_echo(response)


@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Type": "text/plain"},
        {"Content-Type": "application/x-www-form-urlencoded"},
        {},
    ],
)
def test_wrong_or_missing_content_type_is_415(
    app: FastAPI, headers: dict[str, str]
) -> None:
    conn = FakeConn()
    _use(app, conn)
    with TestClient(app) as client:
        response = client.post("/ops/jobs/1/retry", content=b"", headers=headers)
    assert response.status_code == 415


@pytest.mark.parametrize("body", [b"", b"{}"])
def test_an_empty_or_empty_object_json_body_is_accepted_content_type(
    app: FastAPI, body: bytes
) -> None:
    conn = FakeConn(error=RuntimeError("stop before touching the database"))
    _use(app, conn)
    with TestClient(app) as client:
        response = client.post(
            "/ops/jobs/1/retry", content=body, headers={"Content-Type": "application/json"}
        )
    # Content-Type passed the gate; the 500 comes from the fake raising once
    # retry_dead touches the connection, proving the gate let it through.
    assert response.status_code != 415


def test_the_rate_limit_hook_covers_the_retry_route_but_not_the_list_route(
    app: FastAPI,
) -> None:
    def too_many() -> None:
        raise HTTPException(429, "slow down")

    app.dependency_overrides[deps.rate_limit] = too_many
    conn = FakeConn([])
    _use(app, conn)
    with TestClient(app) as client:
        retry = client.post(
            "/ops/jobs/1/retry", json={}, headers={"Content-Type": "application/json"}
        )
        listing = client.get("/ops/jobs")
    assert retry.status_code == 429
    assert listing.status_code == 200


# ---------------------------------------------------------------------------
# Integration: both routes end to end against seeded Postgres
# ---------------------------------------------------------------------------


@pytest.fixture
def dsn(head_dsn: str, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("DATABASE_URL", head_dsn)
    get_settings.cache_clear()
    return head_dsn


@pytest.fixture
def db(dsn: str) -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(dsn, autocommit=True) as conn:
        yield conn


@pytest.fixture
def api(dsn: str, logs: LogSink) -> Iterator[TestClient]:
    with TestClient(create_app()) as client:
        yield client


def _seed_job(
    db: psycopg.Connection[Any],
    *,
    video_id: str = "v1",
    kind: str = "ingest",
    dedupe_key: str = "default",
    state: str = "dead",
    attempts: int = 3,
    error_class: str | None = "TOOL_FAILURE",
    last_error: str | None = "boom",
    priority: int = 0,
) -> int:
    row = db.execute(
        """
        INSERT INTO jobs (video_id, kind, dedupe_key, state, attempts,
                           error_class, last_error, priority, finished_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                CASE WHEN %s = 'dead' THEN now() ELSE NULL END)
        RETURNING id
        """,
        (video_id, kind, dedupe_key, state, attempts, error_class, last_error, priority, state),
    ).fetchone()
    assert row is not None
    return int(row[0])


@pytest.mark.integration
def test_list_jobs_end_to_end(api: TestClient, db: psycopg.Connection[Any]) -> None:
    db.execute("INSERT INTO videos (video_id, title) VALUES ('v1', 'My Video')")
    dead_id = _seed_job(db, video_id="v1", kind="transcribe", state="dead")
    _seed_job(db, video_id="v2", kind="ingest", state="pending", error_class=None)

    response = api.get("/ops/jobs", params={"state": "dead", "kind": "transcribe"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["id"] == dead_id
    assert item["video_id"] == "v1"
    assert item["video_title"] == "My Video"
    assert item["state"] == "dead"
    assert "payload" not in item
    assert body["next_before_id"] is None


@pytest.mark.integration
def test_list_jobs_error_class_none_matches_only_null_rows_end_to_end(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    reaped = _seed_job(db, video_id="v1", state="dead", error_class=None)
    _seed_job(db, video_id="v2", state="dead", error_class="BUG")

    response = api.get("/ops/jobs", params={"error_class": "none"})

    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["items"]]
    assert ids == [reaped]


@pytest.mark.integration
def test_list_jobs_pagination_boundaries_end_to_end(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    """``next_before_id`` and ``has_more`` at the exact-limit / limit+1 boundaries.

    ``list_jobs`` itself is asked for ``limit`` and ``limit + 1`` rows in
    ``tests/common/repo/test_jobs.py``; this test instead proves the route
    is the one passing ``params.limit + 1`` to it (a route that passed a
    bare ``limit`` would still make every unit test in this module pass,
    since those construct ``FakeConn`` rows directly).
    """
    assert api.get("/ops/jobs", params={"limit": 3}).json() == {
        "items": [],
        "next_before_id": None,
    }

    ids = [_seed_job(db, video_id=f"v{i}") for i in range(3)]
    exact = api.get("/ops/jobs", params={"limit": 3}).json()
    assert [item["id"] for item in exact["items"]] == list(reversed(ids))
    assert exact["next_before_id"] is None

    ids.append(_seed_job(db, video_id="v3"))
    full_page = api.get("/ops/jobs", params={"limit": 3}).json()
    assert len(full_page["items"]) == 3
    assert [item["id"] for item in full_page["items"]] == list(reversed(ids))[:3]
    cursor = full_page["next_before_id"]
    assert cursor == full_page["items"][-1]["id"]
    assert cursor is not None

    second_page = api.get("/ops/jobs", params={"limit": 3, "before_id": cursor}).json()
    assert [item["id"] for item in second_page["items"]] == list(reversed(ids))[3:]
    assert second_page["next_before_id"] is None


@pytest.mark.integration
def test_retry_success_end_to_end(
    api: TestClient, db: psycopg.Connection[Any], logs: LogSink
) -> None:
    job_id = _seed_job(db, video_id="v1", kind="ingest", attempts=4, error_class="BUG")

    response = api.post(
        f"/ops/jobs/{job_id}/retry", json={}, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == job_id
    assert body["state"] == "pending"
    assert body["attempts"] == 0
    assert body["error_class"] == "BUG"  # kept

    row = db.execute(
        "SELECT state, attempts, finished_at, locked_by, heartbeat_at,"
        " error_class, last_error FROM jobs WHERE id = %s",
        (job_id,),
    ).fetchone()
    assert row == ("pending", 0, None, None, None, "BUG", "boom")

    log_line = next(
        r for r in logs.records() if r.get("event") == "ops.job_retried"
    )
    assert log_line["job_id"] == job_id
    assert log_line["video_id"] == "v1"
    assert log_line["kind"] == "ingest"
    assert log_line["prev_attempts"] == 4
    assert log_line["prev_error_class"] == "BUG"
    assert log_line["level"] == "warning"
    assert log_line["request_id"]


@pytest.mark.integration
def test_retried_job_is_claimable_at_once_end_to_end(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    job_id = _seed_job(db, video_id="v1", kind="ingest")

    response = api.post(
        f"/ops/jobs/{job_id}/retry", json={}, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200

    queue = PostgresQueue(db)
    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        assert job.id == job_id
        assert job.attempts == 1


@pytest.mark.integration
def test_retry_does_not_clear_videos_unavailable_end_to_end(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    db.execute(
        "INSERT INTO videos (video_id, unavailable) VALUES ('v1', 'private')"
    )
    job_id = _seed_job(db, video_id="v1", error_class="PERMANENT_SOURCE")

    response = api.post(
        f"/ops/jobs/{job_id}/retry", json={}, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    row = db.execute(
        "SELECT unavailable FROM videos WHERE video_id = 'v1'"
    ).fetchone()
    assert row == ("private",)


@pytest.mark.integration
def test_retry_missing_job_is_404_end_to_end(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    response = api.post(
        "/ops/jobs/999999999/retry", json={}, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "job not found"}


@pytest.mark.integration
@pytest.mark.parametrize("state", ["pending", "running", "done"])
def test_retry_non_dead_job_is_409_with_state_end_to_end(
    api: TestClient, db: psycopg.Connection[Any], state: str
) -> None:
    job_id = _seed_job(db, state=state, error_class=None, attempts=2)

    response = api.post(
        f"/ops/jobs/{job_id}/retry", json={}, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "job is not dead", "state": state}
    row = db.execute("SELECT attempts FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert row == (2,)  # untouched - the refusal writes nothing


@pytest.mark.integration
def test_retry_superseded_by_a_newer_active_job_is_409_end_to_end(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    dead_id = _seed_job(db, video_id="v1", kind="ingest", dedupe_key="default")
    newer = db.execute(
        "INSERT INTO jobs (video_id, kind, dedupe_key, state)"
        " VALUES ('v1', 'ingest', 'default', 'pending') RETURNING id"
    ).fetchone()
    assert newer is not None
    newer_id = int(newer[0])

    response = api.post(
        f"/ops/jobs/{dead_id}/retry", json={}, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "superseded by a newer job", "job_id": newer_id}
    row = db.execute("SELECT state FROM jobs WHERE id = %s", (dead_id,)).fetchone()
    assert row == ("dead",)


@pytest.mark.integration
def test_retry_concurrency_two_requests_give_one_200_and_one_409(
    api: TestClient, db: psycopg.Connection[Any], dsn: str
) -> None:
    job_id = _seed_job(db, video_id="v1", kind="ingest")

    barrier = threading.Barrier(2)
    results: list[int] = []
    results_lock = threading.Lock()

    def call() -> None:
        barrier.wait(timeout=10)
        with TestClient(create_app()) as c:
            r = c.post(
                f"/ops/jobs/{job_id}/retry",
                json={},
                headers={"Content-Type": "application/json"},
            )
        with results_lock:
            results.append(r.status_code)

    threads = [threading.Thread(target=call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert all(not t.is_alive() for t in threads)

    assert sorted(results) == [200, 409]
    row = db.execute(
        "SELECT state, attempts FROM jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row == ("pending", 0)


@pytest.mark.integration
def test_retry_racing_an_enqueue_leaves_exactly_one_active_job(
    api: TestClient, db: psycopg.Connection[Any], dsn: str
) -> None:
    job_id = _seed_job(db, video_id="v1", kind="ingest", dedupe_key="default")

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def retry_call() -> None:
        try:
            barrier.wait(timeout=10)
            with TestClient(create_app()) as c:
                c.post(
                    f"/ops/jobs/{job_id}/retry",
                    json={},
                    headers={"Content-Type": "application/json"},
                )
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            errors.append(exc)

    def enqueue_call() -> None:
        try:
            barrier.wait(timeout=10)
            with psycopg.connect(dsn, autocommit=True) as c:
                PostgresQueue(c).enqueue("ingest", "v1", dedupe_key="default")
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            errors.append(exc)

    threads = [threading.Thread(target=retry_call), threading.Thread(target=enqueue_call)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert all(not t.is_alive() for t in threads)
    assert not errors, f"unexpected exception: {errors!r}"

    active = db.execute(
        "SELECT count(*) FROM jobs WHERE video_id = 'v1' AND kind = 'ingest'"
        " AND dedupe_key = 'default' AND state IN ('pending', 'running')"
    ).fetchone()
    assert active == (1,)


_FIGURES: dict[str, Any] = {
    "percent": 40,
    "done_sec": 40.0,
    "duration_sec": 100.0,
    "elapsed_sec": 80.0,
    "eta_sec": 120.0,
    "estimated_sec": 100.0,
    "estimate_source": "history",
}


@pytest.mark.integration
def test_a_running_transcribe_job_carries_its_progress_end_to_end(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    job_id = _seed_job(db, video_id="v1", kind="transcribe", state="running", error_class=None)
    db.execute("UPDATE jobs SET locked_by = 'w1' WHERE id = %s", (job_id,))
    write_progress(db, job_id, "w1", _FIGURES)

    (item,) = api.get("/ops/jobs").json()["items"]

    progress = item["progress"]
    assert {k: progress[k] for k in _FIGURES} == _FIGURES
    assert progress["updated_at"].endswith("Z") or "+00:00" in progress["updated_at"]


@pytest.mark.integration
def test_a_pending_transcribe_job_shows_the_estimate_from_the_rtf_history(
    api: TestClient, db: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    monkeypatch.setenv("WHISPER_COMPUTE", "int8")
    get_settings.cache_clear()
    db.execute("INSERT INTO videos (video_id) VALUES ('v1'), ('hist')")
    db.execute(
        "INSERT INTO transcripts (video_id, source, segments, full_text, engine_meta)"
        " VALUES ('hist', 'whisper', '[]', '', %s::jsonb)",
        (json.dumps({"model": "tiny", "compute_type": "int8", "rtf": 0.5}),),
    )
    job_id = _seed_job(db, video_id="v1", kind="transcribe", state="pending", error_class=None)
    db.execute("UPDATE jobs SET payload = %s::jsonb WHERE id = %s", (json.dumps({"duration_sec": 200}), job_id))

    (item,) = api.get("/ops/jobs").json()["items"]

    assert item["progress"] == {
        "percent": None,
        "done_sec": None,
        "duration_sec": 200.0,
        "elapsed_sec": None,
        "eta_sec": 100.0,
        "estimated_sec": 100.0,
        "estimate_source": "history",
        "updated_at": None,
    }


@pytest.mark.integration
def test_a_pending_transcribe_job_without_a_duration_has_no_progress(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    _seed_job(db, video_id="v1", kind="transcribe", state="pending", error_class=None)

    (item,) = api.get("/ops/jobs").json()["items"]

    assert item["progress"] is None


@pytest.mark.integration
@pytest.mark.parametrize("kind,state", [("ingest", "running"), ("analyze", "pending")])
def test_other_kinds_never_carry_progress(
    api: TestClient, db: psycopg.Connection[Any], kind: str, state: str
) -> None:
    job_id = _seed_job(db, video_id="v1", kind=kind, state=state, error_class=None)
    db.execute(
        "UPDATE jobs SET progress = %s::jsonb, payload = %s::jsonb WHERE id = %s",
        (json.dumps(_FIGURES), json.dumps({"duration_sec": 200}), job_id),
    )

    (item,) = api.get("/ops/jobs").json()["items"]

    assert item["progress"] is None


@pytest.mark.integration
def test_a_dead_transcribe_job_never_shows_progress(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    job_id = _seed_job(db, video_id="v1", kind="transcribe", state="dead")
    db.execute("UPDATE jobs SET progress = %s::jsonb WHERE id = %s", (json.dumps(_FIGURES), job_id))

    (item,) = api.get("/ops/jobs").json()["items"]

    assert item["progress"] is None
