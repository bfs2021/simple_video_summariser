"""``GET /videos/{id}/render?analysis_id=`` without a database (issue #109).

The repository functions are fakes and ``get_conn`` is overridden. The 422
tests use a ``get_conn`` that fails the test if it is ever resolved.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from common.models import Analysis, Claim, VideoMeta
from services.api import deps, routes_read
from services.api.render import render_analysis_html

VID = "dQw4w9WgXcQ"
OTHER = "-wNyEUrxzFU"
CH = "UC" + "a" * 22
MARKER = "ZZMARKER109"
MAX_ID = 9223372036854775807

OLDER = Analysis(
    id=11, video_id=VID, transcript_id=1, chunk_strategy="c", model="old-model",
    prompt_version="v1", tldr="OLDER tldr", claims=(Claim(text="OLDER claim", start_sec=5),),
    created_at=datetime(2026, 1, 3, tzinfo=UTC),
)  # fmt: skip
NEWER = Analysis(
    id=12, video_id=VID, transcript_id=1, chunk_strategy="c", model="new-model",
    prompt_version="v2", tldr="NEWER tldr", claims=(Claim(text="NEWER claim", start_sec=6),),
    created_at=datetime(2026, 2, 3, tzinfo=UTC),
)  # fmt: skip
FOREIGN = Analysis(
    id=99, video_id=OTHER, transcript_id=2, chunk_strategy="c", model="m",
    prompt_version="v1", tldr="FOREIGN tldr", created_at=datetime(2026, 1, 3, tzinfo=UTC),
)  # fmt: skip


def _no_db() -> Iterator[Any]:
    raise AssertionError("the database must not be touched")
    yield  # pragma: no cover


class _Conn:
    read_only = False


def _fake_conn() -> Iterator[Any]:
    yield _Conn()


class Fakes:
    def __init__(self) -> None:
        self.videos = {
            VID: VideoMeta(
                video_id=VID, channel_id=CH, title="Fake title", description="",
                duration_sec=61, published_at=datetime(2026, 1, 2, tzinfo=UTC), language=None,
                live_status=None, manual_subtitle_langs=(), auto_caption_langs=(),
            ),
        }  # fmt: skip
        self.analyses: list[Analysis] = [OLDER, NEWER, FOREIGN]
        self.calls: list[tuple[Any, ...]] = []

    def get_video_meta(self, conn: Any, video_id: str) -> VideoMeta | None:
        self.calls.append(("meta", video_id))
        return self.videos.get(video_id)

    def latest_analysis(self, conn: Any, video_id: str) -> Analysis | None:
        self.calls.append(("latest", video_id))
        mine = [a for a in self.analyses if a.video_id == video_id]
        return max(mine, key=lambda a: a.created_at or datetime.min.replace(tzinfo=UTC), default=None)

    def get_analysis(self, conn: Any, video_id: str, analysis_id: int) -> Analysis | None:
        self.calls.append(("get", video_id, analysis_id))
        for a in self.analyses:
            if a.id == analysis_id and a.video_id == video_id:
                return a
        return None

    def get_channel_title(self, conn: Any, channel_id: str) -> str | None:
        return "Fake channel"


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    fake = Fakes()
    monkeypatch.setattr(routes_read, "get_video_meta", fake.get_video_meta)
    monkeypatch.setattr(routes_read, "latest_analysis", fake.latest_analysis)
    monkeypatch.setattr(routes_read, "get_analysis", fake.get_analysis, raising=False)
    monkeypatch.setattr(routes_read, "get_channel_title", fake.get_channel_title)
    return fake


@pytest.fixture
def api(app: FastAPI, fakes: Fakes) -> Iterator[TestClient]:
    app.dependency_overrides[deps.get_conn] = _fake_conn
    with TestClient(app) as client:
        yield client


@pytest.fixture
def no_db_api(app: FastAPI) -> Iterator[TestClient]:
    app.dependency_overrides[deps.get_conn] = _no_db
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize(
    "value",
    ["0", "-1", "1.5", "abc", "1e3", str(MAX_ID + 1), "", MARKER, "+1", " 1", "1 ", "0x1", "١"],
)
def test_a_bad_analysis_id_is_422_before_the_database_and_never_echoed(
    no_db_api: TestClient, value: str
) -> None:
    response = no_db_api.get(f"/videos/{VID}/render", params={"analysis_id": value})

    assert response.status_code == 422
    assert MARKER not in response.text
    if value:
        assert value not in response.text


def test_an_empty_analysis_id_is_not_treated_as_omitted(no_db_api: TestClient) -> None:
    assert no_db_api.get(f"/videos/{VID}/render?analysis_id=").status_code == 422


def test_a_malformed_video_id_is_422_even_with_a_valid_analysis_id(
    no_db_api: TestClient,
) -> None:
    response = no_db_api.get(f"/videos/{MARKER}x/render?analysis_id=1")

    assert response.status_code == 422
    assert MARKER not in response.text


@pytest.mark.parametrize("value", [1, MAX_ID])
def test_the_bigint_boundaries_are_accepted_and_reach_the_lookup(
    api: TestClient, fakes: Fakes, value: int
) -> None:
    response = api.get(f"/videos/{VID}/render", params={"analysis_id": value})

    assert response.status_code == 404
    assert ("get", VID, value) in fakes.calls


@pytest.mark.parametrize("query", ["", "?analysis_id=11", "?analysis_id=99", "?analysis_id=5"])
def test_an_unknown_video_is_404_with_or_without_an_analysis_id(
    api: TestClient, fakes: Fakes, query: str
) -> None:
    response = api.get(f"/videos/{OTHER[:-1]}x/render{query}")

    assert response.status_code == 404
    assert response.json() == {"detail": "video not found"}
    assert all(call[0] == "meta" for call in fakes.calls)


def test_an_unused_id_and_another_videos_id_give_identical_404s(
    api: TestClient, fakes: Fakes
) -> None:
    fakes.videos[OTHER] = fakes.videos[VID]
    unused = api.get(f"/videos/{VID}/render?analysis_id=5000")
    foreign = api.get(f"/videos/{VID}/render?analysis_id=99")

    assert unused.status_code == foreign.status_code == 404
    assert unused.json() == {"detail": "analysis not found"}
    assert foreign.content == unused.content
    ignore = {"x-request-id"}  # unique per request by design
    assert {k: v for k, v in foreign.headers.items() if k not in ignore} == {
        k: v for k, v in unused.headers.items() if k not in ignore
    }
    assert VID not in foreign.text and "99" not in foreign.text and "5000" not in unused.text


def test_a_video_with_no_analyses_is_analysis_not_found_with_an_id_and_no_analysis_yet_without(
    api: TestClient, fakes: Fakes
) -> None:
    fakes.analyses = []

    with_id = api.get(f"/videos/{VID}/render?analysis_id=11")
    without = api.get(f"/videos/{VID}/render")

    assert (with_id.status_code, with_id.json()) == (404, {"detail": "analysis not found"})
    assert (without.status_code, without.json()) == (404, {"detail": "no analysis yet"})


def test_omitting_the_parameter_still_renders_the_latest_without_the_by_id_lookup(
    api: TestClient, fakes: Fakes
) -> None:
    response = api.get(f"/videos/{VID}/render")

    assert response.status_code == 200
    assert "NEWER tldr" in response.text
    assert not any(call[0] == "get" for call in fakes.calls)


def test_an_older_runs_id_renders_that_run_and_none_of_the_newer_content(
    api: TestClient, fakes: Fakes
) -> None:
    response = api.get(f"/videos/{VID}/render?analysis_id=11")

    assert response.status_code == 200
    for part in ("OLDER tldr", "OLDER claim", "old-model", "v1"):
        assert part in response.text
    for part in ("NEWER", "new-model"):
        assert part not in response.text
    assert not any(call[0] == "latest" for call in fakes.calls)


def test_the_latest_runs_id_is_byte_identical_to_no_parameter(api: TestClient) -> None:
    by_id = api.get(f"/videos/{VID}/render?analysis_id=12")
    plain = api.get(f"/videos/{VID}/render")

    assert by_id.status_code == plain.status_code == 200
    assert by_id.content == plain.content


def test_the_body_is_exactly_the_renderer_output_for_the_looked_up_run(
    api: TestClient, fakes: Fakes
) -> None:
    response = api.get(f"/videos/{VID}/render?analysis_id=11")

    expected = render_analysis_html(fakes.videos[VID], "Fake channel", OLDER)
    assert response.content == expected.encode("utf-8")


def test_the_200_carries_the_hardening_headers(api: TestClient) -> None:
    response = api.get(f"/videos/{VID}/render?analysis_id=11")

    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.headers["content-security-policy"] == (
        "default-src 'none'; style-src 'unsafe-inline'"
    )
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-cache"


def test_openapi_lists_analysis_id_as_an_optional_integer_with_minimum_1(
    api: TestClient,
) -> None:
    schema = api.get("/openapi.json").json()
    get = schema["paths"]["/videos/{video_id}/render"]["get"]

    [param] = [p for p in get["parameters"] if p["name"] == "analysis_id"]
    assert param["in"] == "query"
    assert param["required"] is False
    options = param["schema"].get("anyOf", [param["schema"]])
    [integer] = [o for o in options if o.get("type") == "integer"]
    assert integer["minimum"] == 1
    assert list(get["responses"]["200"]["content"]) == ["text/html"]


def test_the_handler_is_sync_and_has_no_sql() -> None:
    import inspect

    handler = routes_read.get_video_render
    assert not inspect.iscoroutinefunction(handler)
    assert "execute" not in inspect.getsource(handler)
