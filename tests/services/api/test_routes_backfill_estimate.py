"""Backfill route wiring for the #117 estimate: settings, probe dependency, log line."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg.pq import TransactionStatus

from common.models import VideoMeta
from services.api import deps, routes_write
from services.api.backfill import BackfillResult, Estimate, SubtitleProbe
from tests.services.api.conftest import LogSink

CHANNEL = "UCuAXFkgsw1L7xaCfnd5JJOw"


class _Info:
    transaction_status = TransactionStatus.IDLE


class _Conn:
    info = _Info()


PROBE = SubtitleProbe(lambda timeout: None, lambda meta: False)  # type: ignore[arg-type,return-value]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def fake(conn: Any, queue: Any, catalog: Any, channel_id: str, **kw: Any) -> BackfillResult:
        seen.append(kw)
        return BackfillResult(
            CHANNEL, True, kw["limit"], 0, None, 0, 0, 0, (),
            Estimate(0, 0, None, 0, None, probed=3, manual_subtitles=1, unchecked=2,
                     probe_failed=1, deadline_reached=True),
            None,
        )

    monkeypatch.setattr(routes_write, "backfill", fake)
    return seen


@pytest.fixture
def api(app: FastAPI, calls: list[dict[str, Any]]) -> Iterator[TestClient]:
    def fake_conn() -> Iterator[_Conn]:
        yield _Conn()

    app.dependency_overrides[deps.get_conn] = fake_conn
    app.dependency_overrides[deps.get_catalog] = lambda: object()
    app.dependency_overrides[routes_write.get_subtitle_probe] = lambda: PROBE
    with TestClient(app) as client:
        yield client


def test_settings_probe_and_dry_run_reach_backfill(
    api: TestClient, calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PREFER_WHISPER", "1")
    monkeypatch.setenv("SUMMARIZER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")
    from common.config import get_settings

    get_settings.cache_clear()

    api.post(f"/channels/{CHANNEL}/backfill", json={"limit": 5})

    [kw] = calls
    assert kw["probe"] is PROBE
    assert kw["prefer_whisper"] is True
    assert (kw["summarizer"], kw["model"]) == ("anthropic", "claude-test")


def test_ollama_reports_the_ollama_model(
    api: TestClient, calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUMMARIZER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "local-m")
    from common.config import get_settings

    get_settings.cache_clear()

    api.post(f"/channels/{CHANNEL}/backfill", json={"limit": 5})

    assert (calls[0]["summarizer"], calls[0]["model"]) == ("ollama", "local-m")


def test_the_info_line_gains_the_probe_counts_and_no_titles(
    api: TestClient, logs: LogSink
) -> None:
    api.post(f"/channels/{CHANNEL}/backfill", json={"limit": 5})

    [line] = [r for r in logs.records() if r.get("event") == "backfill"]
    assert (line["probed"], line["manual_subtitles"], line["unchecked"]) == (3, 1, 2)
    assert (line["probe_failed"], line["deadline_reached"]) == (1, True)


def _meta(manual: tuple[str, ...], auto: tuple[str, ...]) -> VideoMeta:
    return VideoMeta("aaaaaaaaaaa", "UC", "t", "", 60, None, "en", None, manual, auto)


def test_the_default_probe_wraps_the_adapters_and_only_a_manual_track_counts() -> None:
    probe = routes_write.get_subtitle_probe()

    assert type(probe.metadata_factory(7.0)).__name__ == "YouTubeMetadata"
    assert probe.has_manual(_meta(("en",), ())) is True
    assert probe.has_manual(_meta((), ("en",))) is False  # auto-captions are only a fallback
    assert probe.has_manual(_meta((), ())) is False
