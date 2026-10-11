"""``GET /search`` (issue #43).

Unit tests (no Docker) replace ``routes_read.search_transcripts`` and the
connection; integration tests at the bottom run against Postgres.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from psycopg.pq import TransactionStatus

from common.config import get_settings
from common.repo.search import Excerpt, SearchHit, SearchPage, Span
from services.api import deps, routes_read
from services.api.main import create_app
from tests.services.api.conftest import UNREACHABLE_DSN, FakePool, LogSink

MARKER = "ZZMARKER42"


class _Info:
    transaction_status = TransactionStatus.IDLE


class _Conn:
    read_only = False
    info = _Info()


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.page = SearchPage((), False)
        self.error: Exception | None = None

    def __call__(self, conn: Any, query: str, limit: int, offset: int) -> SearchPage:
        self.calls.append((query, limit, offset))
        if self.error is not None:
            raise self.error
        return self.page


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(routes_read, "search_transcripts", rec)
    return rec


@pytest.fixture
def api(app: FastAPI, recorder: Recorder) -> Iterator[TestClient]:
    def fake_conn() -> Iterator[_Conn]:
        yield _Conn()

    app.dependency_overrides[deps.get_conn] = fake_conn
    with TestClient(app) as client:
        yield client


def _hit(**overrides: Any) -> SearchHit:
    values: dict[str, Any] = {
        "video_id": "dQw4w9WgXcQ",
        "title": "Title",
        "channel_id": "UC" + "c" * 22,
        "channel_title": "Chan",
        "published_at": datetime(2026, 9, 1, 12, tzinfo=UTC),
        "duration_sec": 61,
        "unavailable": None,
        "transcript_source": "whisper",
        "matched_in": ("transcript",),
        "excerpts": (
            Excerpt((Span("the ", False), Span("zebra", True), Span(" ran", False)), 12.5),
        ),
    }
    values.update(overrides)
    return SearchHit(**values)


def _server_log(logs: LogSink) -> str:
    """Server log lines only; the test client's own ``httpx`` logger records its URL."""
    return "\n".join(
        str(r) for r in logs.records() if not str(r.get("logger", "")).startswith("httpx")
    )


def _assert_422_without_echo(response: Any) -> None:
    assert response.status_code == 422, response.text
    assert MARKER not in response.text
    for item in response.json()["detail"]:
        assert set(item) == {"loc", "msg", "type"}


# --- parameters ---------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "   \t "},
        {"q": "a" * 201},
        {"q": MARKER + "a" * 191},
        {"q": "  " + "a" * 201 + "  "},
        {"q": "zebra\x00" + MARKER},
        {"q": "zebra", "limit": "0"},
        {"q": "zebra", "limit": "51"},
        {"q": "zebra", "limit": "abc"},
        {"q": "zebra", "limit": "1.5"},
        {"q": "zebra", "limit": ""},
        {"q": "zebra", "offset": "-1"},
        {"q": "zebra", "offset": "1001"},
        {"q": "zebra", "offset": "abc"},
        {"q": "zebra", "offset": "1.5"},
        {"q": "zebra", "offset": ""},
        {"q": "zebra", "offset": MARKER},
    ],
)
def test_bad_parameters_are_422_without_echo_and_never_search(
    api: TestClient, recorder: Recorder, params: dict[str, str]
) -> None:
    _assert_422_without_echo(api.get("/search", params=params))
    assert recorder.calls == []


def test_a_201_character_q_with_a_marker_is_not_echoed(api: TestClient) -> None:
    q = (MARKER * 21)[:201]

    _assert_422_without_echo(api.get("/search", params={"q": q}))


@pytest.mark.parametrize(
    ("q", "expected"),
    [("a" * 200, "a" * 200), ("  zebra  ", "zebra"), ("\t" + "b" * 200 + "\n", "b" * 200)],
)
def test_q_is_stripped_and_200_characters_are_accepted(
    api: TestClient, recorder: Recorder, q: str, expected: str
) -> None:
    assert api.get("/search", params={"q": q}).status_code == 200
    assert recorder.calls == [(expected, 20, 0)]


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, (20, 0)),
        ({"limit": "1"}, (1, 0)),
        ({"limit": "50"}, (50, 0)),
        ({"offset": "1000"}, (20, 1000)),
    ],
)
def test_paging_bounds_and_defaults(
    api: TestClient, recorder: Recorder, params: dict[str, str], expected: tuple[int, int]
) -> None:
    response = api.get("/search", params={"q": "zebra", **params})

    assert response.status_code == 200
    assert recorder.calls == [("zebra", *expected)]
    body = response.json()
    assert (body["limit"], body["offset"]) == expected


# --- response -----------------------------------------------------------------


def test_the_response_shape(api: TestClient, recorder: Recorder) -> None:
    recorder.page = SearchPage((_hit(),), True)

    response = api.get("/search", params={"q": "zebra " + MARKER, "limit": "1"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert MARKER not in response.text
    assert response.json() == {
        "results": [
            {
                "video_id": "dQw4w9WgXcQ",
                "title": "Title",
                "channel_id": "UC" + "c" * 22,
                "channel_title": "Chan",
                "published_at": "2026-09-01T12:00:00Z",
                "duration_sec": 61,
                "unavailable": None,
                "transcript_source": "whisper",
                "matched_in": ["transcript"],
                "excerpts": [
                    {
                        "spans": [
                            {"text": "the ", "match": False},
                            {"text": "zebra", "match": True},
                            {"text": " ran", "match": False},
                        ],
                        "start_sec": 12.5,
                    }
                ],
            }
        ],
        "limit": 1,
        "offset": 0,
        "has_more": True,
    }


def test_nullable_columns_are_null(api: TestClient, recorder: Recorder) -> None:
    recorder.page = SearchPage(
        (_hit(title=None, channel_id=None, channel_title=None, published_at=None,
              duration_sec=None, unavailable="removed"),),
        False,
    )

    [result] = api.get("/search", params={"q": "zebra"}).json()["results"]

    assert (result["title"], result["channel_id"], result["channel_title"],
            result["published_at"], result["duration_sec"], result["unavailable"]) == (
        None, None, None, None, None, "removed",
    )


def test_start_sec_is_a_float_or_null(api: TestClient, recorder: Recorder) -> None:
    spans = (Span("zebra", True),)
    recorder.page = SearchPage(
        (_hit(excerpts=(Excerpt(spans, 0.0), Excerpt(spans, 3723.0), Excerpt(spans, None))),),
        False,
    )

    [result] = api.get("/search", params={"q": "zebra"}).json()["results"]

    assert [e["start_sec"] for e in result["excerpts"]] == [0.0, 3723.0, None]


def test_the_response_model_rejects_a_negative_start_sec() -> None:
    from pydantic import ValidationError

    from services.api.schemas import ExcerptOut

    assert ExcerptOut(spans=[], start_sec=1.5).start_sec == 1.5
    assert ExcerptOut(spans=[], start_sec=None).start_sec is None
    with pytest.raises(ValidationError):
        ExcerptOut(spans=[], start_sec=-1)


def test_a_title_only_match_has_no_transcript_and_no_excerpts(
    api: TestClient, recorder: Recorder
) -> None:
    recorder.page = SearchPage(
        (_hit(transcript_source=None, matched_in=("title", "analysis"), excerpts=()),), False
    )

    [result] = api.get("/search", params={"q": "zebra"}).json()["results"]

    assert result["transcript_source"] is None
    assert result["matched_in"] == ["title", "analysis"]
    assert result["excerpts"] == []


def test_search_result_fields_share_names_and_types_with_the_library_item(
    app: FastAPI,
) -> None:
    schemas = app.openapi()["components"]["schemas"]
    result = schemas["SearchResult"]["properties"]
    item = schemas["VideoItem"]["properties"]
    for name in ("video_id", "title", "channel_id", "channel_title", "published_at",
                 "duration_sec", "unavailable"):
        assert result[name] == item[name], name
    assert app.openapi()["paths"]["/search"]["get"]["responses"]["200"]["content"][
        "application/json"]["schema"] == {"$ref": "#/components/schemas/SearchPage"}


def test_one_log_line_per_search_without_the_query(
    api: TestClient, recorder: Recorder, logs: LogSink
) -> None:
    recorder.page = SearchPage((_hit(), _hit(video_id="b" * 11)), False)

    api.get("/search", params={"q": "zebra " + MARKER, "limit": "5", "offset": "3"})

    [line] = [r for r in logs.records() if r.get("event") == "search"]
    assert (line["q_len"], line["results"], line["offset"], line["limit"]) == (
        len("zebra " + MARKER), 2, 3, 5,
    )
    assert isinstance(line["duration_ms"], int | float)
    assert MARKER not in _server_log(logs)


# --- failures -----------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        psycopg.errors.QueryCanceled("canceling statement due to statement timeout"),
        psycopg.OperationalError(f"connection to {UNREACHABLE_DSN} lost"),
        psycopg.errors.SyntaxError("syntax error at or near SELECT websearch_to_tsquery"),
        psycopg.errors.InternalError_("XX000 boom"),
    ],
)
def test_database_errors_are_503_with_a_fixed_body_and_logged(
    api: TestClient, recorder: Recorder, logs: LogSink, error: Exception
) -> None:
    recorder.error = error

    response = api.get(
        "/search", params={"q": "zebra " + MARKER}, headers={"X-Request-Id": "req-43"}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "database unavailable"}
    for secret in (MARKER, "SECRETPW", "127.0.0.1", "SELECT", "websearch", "timeout"):
        assert secret not in response.text
    [line] = [r for r in logs.records() if r.get("event") == "database unavailable"]
    assert line["request_id"] == "req-43"
    assert type(error).__name__ in line["exception"]
    assert MARKER not in _server_log(logs)


def test_an_unreachable_database_is_503(
    app: FastAPI, logs: LogSink, use_pool: Any
) -> None:
    use_pool(FakePool(getconn_error=psycopg.OperationalError(f"could not connect to {UNREACHABLE_DSN}")))
    with TestClient(app) as client:
        response = client.get("/search", params={"q": "zebra"})

    assert response.status_code == 503
    assert response.json() == {"detail": "database unavailable"}


def test_the_route_is_a_sync_read_route() -> None:
    import inspect

    [route] = [r for r in routes_read.router.routes if getattr(r, "path", "") == "/search"]
    assert not inspect.iscoroutinefunction(route.endpoint)  # type: ignore[attr-defined]


# --- integration ----------------------------------------------------------------


@pytest.fixture
def db(head_dsn: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[psycopg.Connection[Any]]:
    monkeypatch.setenv("DATABASE_URL", head_dsn)
    get_settings.cache_clear()
    with psycopg.connect(head_dsn, autocommit=True) as conn:
        conn.execute("INSERT INTO videos (video_id, title) VALUES ('dbsearch001', 'T')")
        conn.execute(
            "INSERT INTO transcripts (video_id, source, segments, full_text)"
            " VALUES ('dbsearch001', 'whisper', '[]',"
            " 'the zebra ran past café <b>tags</b> and 🦓 with \\ % _ quotes '' \"')"
        )
        yield conn


@pytest.fixture
def live(db: psycopg.Connection[Any], logs: LogSink) -> Iterator[TestClient]:
    with TestClient(create_app()) as client:
        yield client


HOSTILE = [
    "'", '"', '"unbalanced', "' OR 1=1 --", "foo & bar | !(baz:*) <-> qux", "\\", "%", "_",
    "-", "--", "or", "<script>alert(1)</script>", "café", "🦓", "!@#$%^&*()_+-=[]{}|;:,.<>?/~`" * 7,
]


@pytest.mark.integration
@pytest.mark.parametrize("q", HOSTILE)
def test_hostile_queries_are_200_end_to_end(live: TestClient, q: str) -> None:
    response = live.get("/search", params={"q": q[:200]})

    assert response.status_code == 200, response.text
    assert set(response.json()) == {"results", "limit", "offset", "has_more"}


@pytest.mark.integration
def test_a_real_search_end_to_end_writes_nothing(
    live: TestClient, db: psycopg.Connection[Any]
) -> None:
    def counts() -> Any:
        return db.execute(
            "SELECT (SELECT count(*) FROM videos), (SELECT count(*) FROM transcripts),"
            " (SELECT count(*) FROM jobs), (SELECT count(*) FROM channels)"
        ).fetchone()

    before = counts()
    body = live.get("/search", params={"q": "zebras"}).json()

    assert counts() == before
    [result] = body["results"]
    assert result["video_id"] == "dbsearch001"
    spans = [s for e in result["excerpts"] for s in e["spans"]]
    assert {"text": "zebra", "match": True} in spans
    assert "<b>tags</b>" in "".join(s["text"] for s in spans)


@pytest.mark.integration
@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    q=st.text(
        st.characters(exclude_characters="\x00", exclude_categories=["Cs"]),
        min_size=1, max_size=200,
    )
)
def test_any_query_is_200_or_422_never_500(live: TestClient, q: str) -> None:
    response = live.get("/search", params={"q": q})

    assert response.status_code in (200, 422), (q, response.text)
