"""Read routes end to end on Postgres (issue #42).

The real ``get_conn`` is used. Rows are seeded with explicit timestamps.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from common.config import get_settings
from common.db import create_pool
from services.api import deps
from services.api.main import create_app
from tests.common.repo import read_seed as seed
from tests.services.api.conftest import LogSink, request_with

pytestmark = pytest.mark.integration

CH = "UC" + "c" * 22
VID = "-wNyEUrxzFU"


@pytest.fixture
def db(head_dsn: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[psycopg.Connection[Any]]:
    monkeypatch.setenv("DATABASE_URL", head_dsn)
    get_settings.cache_clear()
    with psycopg.connect(head_dsn, autocommit=True) as conn:
        yield conn


@pytest.fixture
def api(db: psycopg.Connection[Any], logs: LogSink) -> Iterator[TestClient]:
    with TestClient(create_app()) as client:
        yield client


def _counts(db: psycopg.Connection[Any]) -> tuple[int, ...]:
    row = db.execute(
        "SELECT (SELECT count(*) FROM videos), (SELECT count(*) FROM jobs),"
        " (SELECT count(*) FROM analyses), (SELECT count(*) FROM transcripts)"
    ).fetchone()
    assert row is not None
    return tuple(int(v) for v in row)


# --- GET /videos ----------------------------------------------------------------


def test_an_empty_library(api: TestClient) -> None:
    response = api.get("/videos")

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "offset": 0, "limit": 50}


def test_a_library_item_has_exactly_the_documented_fields(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    seed.channel(db, CH, "Chan")
    seed.video(db, VID, channel_id=CH, title="<script>alert(1)</script>",
               published_at=datetime(2026, 9, 1, 12, tzinfo=UTC), duration_sec=61, origin="rss")
    t = seed.transcript(db, VID)
    seed.analysis(db, VID, t, created=7)
    job_id = seed.job(db, VID, "analyze", "pending", created=8, dedupe_key="v2")
    dead = seed.job(db, VID, "ingest", "dead", created=1, error_class="BUG",
                    last_error="Traceback SECRET", finished_at=seed.at(2))

    response = api.get("/videos")

    assert response.headers["content-type"] == "application/json"
    assert response.json()["items"] == [
        {
            "video_id": VID,
            "title": "<script>alert(1)</script>",
            "channel_id": CH,
            "channel_title": "Chan",
            "published_at": "2026-09-01T12:00:00Z",
            "duration_sec": 61,
            "origin": "rss",
            "unavailable": None,
            "status": "done",
            "active_job": {"id": job_id, "kind": "analyze", "state": "pending"},
            "last_failure": {"job_id": dead, "kind": "ingest", "error_class": "BUG",
                             "finished_at": "2026-01-01T00:02:00Z"},
            "latest_analysis_at": "2026-01-01T00:07:00Z",
            "has_transcript": True,
        }
    ]
    assert "SECRET" not in response.text


def test_list_and_detail_agree_on_every_status(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    expected = seed.status_matrix(db)

    items = api.get("/videos", params={"limit": 200}).json()["items"]

    assert {i["video_id"]: i["status"] for i in items} == expected
    for item in items:
        detail = api.get(f"/videos/{item['video_id']}").json()
        for key in item:
            assert detail[key] == item[key], (item["video_id"], key)
    for status in ("done", "processing", "unavailable", "failed", "idle"):
        got = api.get("/videos", params={"status": status, "limit": 200}).json()
        assert {i["video_id"] for i in got["items"]} == {
            v for v, s in expected.items() if s == status
        }
        assert got["total"] == sum(1 for s in expected.values() if s == status)


def test_paging_boundaries(api: TestClient, db: psycopg.Connection[Any]) -> None:
    for i in range(5):
        seed.video(db, f"pg{i:09d}", discovered_at=seed.at(i))

    walked: list[str] = []
    for offset in range(0, 6, 2):
        body = api.get("/videos", params={"offset": offset, "limit": 2}).json()
        assert (body["total"], body["offset"], body["limit"]) == (5, offset, 2)
        walked.extend(i["video_id"] for i in body["items"])
    beyond = api.get("/videos", params={"offset": 5}).json()

    assert walked == [f"pg{i:09d}" for i in reversed(range(5))]
    assert (beyond["items"], beyond["total"]) == ([], 5)
    assert api.get("/videos", params={"limit": 200}).status_code == 200
    assert api.get("/videos", params={"offset": 1_000_000}).json()["items"] == []


@pytest.mark.parametrize(
    ("after", "before", "expected"),
    [
        ("2026-09-01", None, ["d_sep01noon", "d_sep01mid0"]),
        ("2026-09-01T12:00:00+00:00", None, ["d_sep01noon"]),
        ("2026-09-01T14:00:00+02:00", None, ["d_sep01noon"]),
        ("2026-09-01T12:00:00", None, ["d_sep01noon"]),
        ("2026-09-01T12:00:01Z", None, []),
        (None, "2026-09-01", ["d_aug31eve0"]),
        (None, "2026-09-01T12:00:00Z", ["d_sep01mid0", "d_aug31eve0"]),
        ("2026-09-01", "2026-09-01", []),
    ],
)
def test_date_filters(
    api: TestClient, db: psycopg.Connection[Any], after: str | None, before: str | None,
    expected: list[str],
) -> None:
    seed.video(db, "d_sep01noon", published_at=datetime(2026, 9, 1, 12, tzinfo=UTC))
    seed.video(db, "d_sep01mid0", published_at=datetime(2026, 9, 1, tzinfo=UTC))
    seed.video(db, "d_aug31eve0", published_at=datetime(2026, 8, 31, 23, 59, tzinfo=UTC))
    seed.video(db, "d_nopub0000", published_at=None)
    params = {k: v for k, v in (("published_after", after), ("published_before", before)) if v}

    body = api.get("/videos", params=params).json()

    assert [i["video_id"] for i in body["items"]] == expected


def test_channel_filter_and_an_unknown_channel(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    seed.channel(db, CH)
    seed.video(db, "ch000000001", channel_id=CH)
    seed.video(db, "ch000000002")

    assert [i["video_id"] for i in api.get("/videos", params={"channel": CH}).json()["items"]] == [
        "ch000000001"
    ]
    unknown = api.get("/videos", params={"channel": "UC" + "z" * 22})
    assert (unknown.status_code, unknown.json()["total"]) == (200, 0)


# --- GET /channels (issue #119) -------------------------------------------------


def _channel_ids(prefix: str, n: int) -> list[str]:
    return [f"UC{prefix}{i:0{22 - len(prefix)}d}" for i in range(n)]


def test_an_empty_channel_list(api: TestClient) -> None:
    response = api.get("/channels")

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "offset": 0, "limit": 200}


def test_channel_list_items_counts_and_order(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    html = "<script>alert(1)</script>"
    seed.channel(db, "UCb", "Bravo")
    seed.channel(db, "UCa", html)
    seed.channel(db, "UCz", None)  # untitled: last
    seed.channel(db, "UCe", "Echo")  # no videos yet
    db.execute("UPDATE channels SET active = false WHERE channel_id = 'UCz'")
    seed.video(db, "chl00000001", channel_id="UCb")
    seed.video(db, "chl00000002", channel_id="UCb")
    seed.video(db, "chl00000003", channel_id="UCz")
    seed.video(db, "chl00000004")  # NULL channel: counted nowhere

    response = api.get("/channels")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    assert (body["total"], body["offset"], body["limit"]) == (4, 0, 200)
    assert body["items"] == [
        {"channel_id": "UCa", "title": html, "active": True, "video_count": 0},
        {"channel_id": "UCb", "title": "Bravo", "active": True, "video_count": 2},
        {"channel_id": "UCe", "title": "Echo", "active": True, "video_count": 0},
        {"channel_id": "UCz", "title": None, "active": False, "video_count": 1},
    ]


def test_channel_video_count_equals_the_library_total_for_that_channel(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    [one, two] = _channel_ids("m", 2)
    seed.channel(db, one, "One")
    seed.channel(db, two, "Two")
    for n, channel_id in enumerate([one, one, one, two]):
        seed.video(db, f"cnt{n:08d}", channel_id=channel_id)

    items = api.get("/channels").json()["items"]

    for item in items:
        library = api.get("/videos", params={"channel": item["channel_id"]}).json()
        assert item["video_count"] == library["total"]
    assert [i["video_count"] for i in items] == [3, 1]


def test_channel_list_paging_boundaries_and_unknown_params(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    for n, channel_id in enumerate(_channel_ids("p", 3)):
        seed.channel(db, channel_id, "ABC"[n])

    first = api.get("/channels", params={"limit": 2, "ignored": "x"}).json()
    second = api.get("/channels", params={"limit": 2, "offset": 2}).json()
    beyond = api.get("/channels", params={"offset": 3}).json()

    assert ([i["title"] for i in first["items"]], first["total"], first["limit"]) == (
        ["A", "B"], 3, 2,
    )
    assert ([i["title"] for i in second["items"]], second["offset"]) == (["C"], 2)
    assert (beyond["items"], beyond["total"], beyond["offset"]) == ([], 3, 3)


# --- GET /videos/{id} -----------------------------------------------------------


def test_an_unknown_video_is_404(api: TestClient) -> None:
    for suffix in ("", "/analyses", "/transcript"):
        response = api.get(f"/videos/{VID}{suffix}")
        assert response.status_code == 404
        assert response.json() == {"detail": "video not found"}


def test_a_processing_video_detail(api: TestClient, db: psycopg.Connection[Any]) -> None:
    seed.video(db, VID)
    job_id = seed.job(db, VID, "ingest", "running", created=1)

    body = api.get(f"/videos/{VID}").json()

    assert body["status"] == "processing"
    assert body["active_job"] == {"id": job_id, "kind": "ingest", "state": "running"}
    assert (body["analysis"], body["transcript"]) == (None, None)


def test_a_done_video_detail_has_the_transcript_and_latest_analysis(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    seed.video(db, VID, title="T")
    seed.transcript(db, VID, "youtube_auto", segments=[{"start": 0, "end": 1, "text": "a"}])
    whisper = seed.transcript(db, VID, "whisper", segments=[
        {"start": 0, "end": 1, "text": "a"}, {"start": 1, "end": 2, "text": "b"}
    ], speaker_source="none")
    seed.analysis(db, VID, whisper, created=1, tldr="old")
    newest = seed.analysis(
        db, VID, whisper, created=2, tldr="</p><b>x", cost_usd="0.5",
        speaker_roster={"speakers": []},
        topics=[(1, "B"), (0, "A")],
        claims=[{"text": "c-null", "start_sec": None},
                {"text": "c-1", "start_sec": 1.5, "speaker": "Host", "confidence": "low",
                 "source_chunk_seq": 3}],
        quotes=[{"text": "q", "start_sec": None}],
    )

    body = api.get(f"/videos/{VID}").json()

    assert body["transcript"] == {"id": whisper, "source": "whisper", "language": "en",
                                  "speaker_source": "none", "segment_count": 2}
    analysis = body["analysis"]
    assert set(analysis) == {
        "id", "model", "prompt_version", "chunk_strategy", "transcript_id",
        "transcript_source", "created_at", "tldr", "speaker_roster", "input_tokens",
        "output_tokens", "cost_usd", "duration_ms", "topics", "claims", "quotes",
    }
    assert (analysis["id"], analysis["tldr"], analysis["transcript_source"]) == (
        newest, "</p><b>x", "whisper",
    )
    assert analysis["cost_usd"] == 0.5
    assert analysis["speaker_roster"] == {"speakers": []}
    assert analysis["created_at"] == "2026-01-01T00:02:00Z"
    assert [t["title"] for t in analysis["topics"]] == ["A", "B"]
    assert analysis["claims"] == [
        {"text": "c-1", "speaker": "Host", "start_sec": 1.5, "confidence": "low",
         "source_chunk_seq": 3},
        {"text": "c-null", "speaker": "unknown", "start_sec": None, "confidence": None,
         "source_chunk_seq": None},
    ]
    assert analysis["quotes"] == [
        {"text": "q", "speaker": "unknown", "start_sec": None, "source_chunk_seq": None}
    ]


@pytest.mark.parametrize("roster", [["A", "B"], "plain", 3, {"speakers": [{"name": "A"}]}])
def test_speaker_roster_is_passed_through_as_stored(
    api: TestClient, db: psycopg.Connection[Any], roster: Any
) -> None:
    seed.video(db, VID)
    t = seed.transcript(db, VID)
    seed.analysis(db, VID, t, created=1, speaker_roster=roster)

    response = api.get(f"/videos/{VID}")

    assert response.status_code == 200
    assert response.json()["analysis"]["speaker_roster"] == roster


def test_an_analysis_with_no_children_and_no_cost(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    seed.video(db, VID)
    t = seed.transcript(db, VID)
    seed.analysis(db, VID, t, created=1)

    analysis = api.get(f"/videos/{VID}").json()["analysis"]

    assert (analysis["topics"], analysis["claims"], analysis["quotes"]) == ([], [], [])
    assert (analysis["cost_usd"], analysis["speaker_roster"]) == (None, None)


# --- GET /videos/{id}/analyses --------------------------------------------------


def test_analyses_of_a_video_without_any(api: TestClient, db: psycopg.Connection[Any]) -> None:
    seed.video(db, VID)

    response = api.get(f"/videos/{VID}/analyses")

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "offset": 0, "limit": 10}


def test_analyses_are_newest_first_with_their_own_children(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    seed.video(db, VID)
    t = seed.transcript(db, VID)
    ids = [
        seed.analysis(db, VID, t, created=i, model=f"model-{i}", prompt_version=f"p{i}",
                      claims=[{"text": f"claim-{i}"}], quotes=[{"text": f"quote-{i}"}],
                      topics=[(0, f"topic-{i}")])
        for i in range(3)
    ]

    body = api.get(f"/videos/{VID}/analyses", params={"limit": 2}).json()
    rest = api.get(f"/videos/{VID}/analyses", params={"offset": 2, "limit": 50}).json()
    detail = api.get(f"/videos/{VID}").json()["analysis"]

    assert (body["total"], body["offset"], body["limit"]) == (3, 0, 2)
    items = body["items"] + rest["items"]
    assert [a["id"] for a in items] == list(reversed(ids))
    for item in items:
        i = item["model"].split("-")[1]
        assert item["prompt_version"] == f"p{i}"
        assert [c["text"] for c in item["claims"]] == [f"claim-{i}"]
        assert [q["text"] for q in item["quotes"]] == [f"quote-{i}"]
        assert [x["title"] for x in item["topics"]] == [f"topic-{i}"]
    assert items[0] == detail


# --- GET /videos/{id}/transcript ------------------------------------------------


def test_a_known_video_without_a_transcript(api: TestClient, db: psycopg.Connection[Any]) -> None:
    seed.video(db, VID)

    response = api.get(f"/videos/{VID}/transcript")

    assert response.status_code == 404
    assert response.json() == {"detail": "transcript not found"}


def test_transcript_pages(api: TestClient, db: psycopg.Connection[Any]) -> None:
    seed.video(db, VID)
    seed.transcript(db, VID, "youtube_auto", segments=[{"start": 0, "end": 1, "text": "auto"}])
    t = seed.transcript(db, VID, "whisper", segments=[
        {"start": i, "end": i + 1, "text": f"s{i}", **({"speaker": "H"} if i == 1 else {})}
        for i in range(2500)
    ])

    first = api.get(f"/videos/{VID}/transcript").json()
    pages = [
        api.get(f"/videos/{VID}/transcript", params={"offset": o, "limit": 1000}).json()
        for o in (0, 1000, 2000)
    ]
    last = api.get(f"/videos/{VID}/transcript", params={"offset": 2499}).json()
    beyond = api.get(f"/videos/{VID}/transcript", params={"offset": 2500}).json()

    assert {k: v for k, v in first.items() if k != "segments"} == {
        "video_id": VID, "transcript_id": t, "source": "whisper", "language": "en",
        "speaker_source": "none", "total_segments": 2500, "offset": 0, "limit": 200,
    }
    assert first["segments"][:2] == [
        {"index": 0, "start": 0.0, "end": 1.0, "text": "s0", "speaker": None},
        {"index": 1, "start": 1.0, "end": 2.0, "text": "s1", "speaker": "H"},
    ]
    assert [len(p["segments"]) for p in pages] == [1000, 1000, 500]
    assert [s["index"] for p in pages for s in p["segments"]] == list(range(2500))
    assert [s["index"] for s in last["segments"]] == [2499]
    assert (beyond["segments"], beyond["total_segments"]) == ([], 2500)


def test_an_empty_transcript(api: TestClient, db: psycopg.Connection[Any]) -> None:
    seed.video(db, VID)
    seed.transcript(db, VID, segments=[])

    body = api.get(f"/videos/{VID}/transcript").json()

    assert (body["total_segments"], body["segments"]) == (0, [])


# --- shared -------------------------------------------------------------------


def test_read_routes_write_nothing(api: TestClient, db: psycopg.Connection[Any]) -> None:
    seed.status_matrix(db)
    before = _counts(db)

    for path in ("/videos", "/videos/st_done0001", "/videos/st_done0001/analyses",
                 "/videos/st_done0001/transcript"):
        assert api.get(path).status_code == 200

    assert _counts(db) == before


def _statement_timeout_ms(conn: psycopg.Connection[Any]) -> int:
    row = conn.execute("SHOW statement_timeout").fetchone()
    assert row is not None
    value = str(row[0])
    return int(value[:-2]) if value.endswith("ms") else int(value[:-1]) * 1000


def test_the_read_connection_is_read_only_with_a_bounded_statement_timeout(
    db: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through a real one-connection pool, so the *reused* connection is what is checked.

    The first borrow marks the connection read-only; the second borrow must get
    the same backend with that state cleared (``reset=``), still bounded by the
    pool-startup ``statement_timeout``, and read-only again only via
    ``get_read_conn``.
    """
    monkeypatch.setenv("DB_POOL_MIN_SIZE", "1")
    monkeypatch.setenv("DB_POOL_MAX_SIZE", "1")
    get_settings.cache_clear()
    pool = create_pool(
        connect_timeout=deps.CONNECT_TIMEOUT_SEC, statement_timeout_ms=deps.STATEMENT_TIMEOUT_MS
    )
    pool.open(wait=True, timeout=30)
    request = request_with(pool)
    try:
        for raw in deps.get_conn(request):
            first_pid = raw.info.backend_pid
            for conn in deps.get_read_conn(raw):
                assert 0 < _statement_timeout_ms(conn) <= 2000
                conn.rollback()
                with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
                    conn.execute("INSERT INTO videos (video_id) VALUES ('ro000000001')")
                conn.rollback()
            assert raw.read_only is True  # set for this request...

        for raw in deps.get_conn(request):  # ...and gone for the next one
            assert raw.info.backend_pid == first_pid  # the pooled connection was reused
            assert raw.read_only is None
            assert 0 < _statement_timeout_ms(raw) <= 2000
            raw.execute("INSERT INTO videos (video_id) VALUES ('rw000000001')")  # writable again
            raw.rollback()

        for raw in deps.get_conn(request):
            for conn in deps.get_read_conn(raw):
                assert conn.info.backend_pid == first_pid
                conn.rollback()
                with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
                    conn.execute("INSERT INTO videos (video_id) VALUES ('ro000000002')")
                conn.rollback()
    finally:
        pool.close()
    assert db.execute("SELECT count(*) FROM videos").fetchone() == (0,)


def test_a_statement_timeout_is_503(
    db: psycopg.Connection[Any], logs: LogSink, head_dsn: str
) -> None:
    """A held ACCESS EXCLUSIVE lock makes the read block until its timeout fires."""

    def short_timeout_conn() -> Iterator[psycopg.Connection[Any]]:
        conn = psycopg.connect(head_dsn, options="-c statement_timeout=200")
        try:
            yield conn
        finally:
            conn.close()

    holder = psycopg.connect(head_dsn)
    try:
        holder.execute("LOCK TABLE videos IN ACCESS EXCLUSIVE MODE")
        app = create_app()
        app.dependency_overrides[deps.get_conn] = short_timeout_conn
        with TestClient(app) as client:
            response = client.get("/videos", headers={"X-Request-Id": "req-timeout"})
    finally:
        holder.rollback()
        holder.close()

    assert response.status_code == 503
    assert response.json() == {"detail": "database unavailable"}
    [line] = [r for r in logs.records() if r.get("event") == "database unavailable"]
    assert line["request_id"] == "req-timeout"
    assert "QueryCanceled" in line["exception"]
