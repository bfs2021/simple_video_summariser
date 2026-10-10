"""The Ollama startup probe in the analyzer entrypoint (issue #84).

``main`` gets an injected probe transport, a ``FakeDatabase`` and a spy handler:
no socket is opened and nothing sleeps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from adapters.summarize.ollama import PROBE_TIMEOUT_SEC, HttpResponse
from common.config import Settings
from common.queue import analyze_dedupe_key
from services.analyzer import main as analyzer_main
from tests.services.analyzer.fake_db import FakeDatabase
from tests.services.analyzer.fakes import make_job
from tests.services.analyzer.test_main import (
    _isolated_process_state,  # noqa: F401 - autouse fixture shared with test_main
    fail_if_called,
    fake_summarizer_factory,
    log_lines,
    make_settings,
    spy_handler_factory,
    worker_options,
)

MODEL = "qwen3.5:4b"
EVENTS = ("analyzer.ollama_ready", "analyzer.ollama_model_missing", "analyzer.ollama_unreachable")


class ProbeTransport:
    def __init__(self, result: HttpResponse | BaseException) -> None:
        self.result = result
        self.calls: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout: float) -> HttpResponse:
        self.calls.append((url, timeout))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def tags(*names: str) -> HttpResponse:
    return HttpResponse(200, json.dumps({"models": [{"name": n} for n in names]}).encode())


def run(
    tmp_path: Path, transport: Any, settings: Settings | None = None, **kwargs: Any
) -> tuple[int, list[Any]]:
    handler_factory, seen = spy_handler_factory()
    db = FakeDatabase()
    code = analyzer_main.main(
        settings=settings or make_settings(OLLAMA_MODEL=MODEL),
        connect=db.connect,
        summarizer_factory=fake_summarizer_factory(),
        handler_factory=handler_factory,
        worker_options=worker_options(tmp_path),
        max_iterations=1,
        probe_transport=transport,
        **kwargs,
    )
    return code, seen


def probe_lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return [line for line in log_lines(capsys) if str(line["event"]).startswith("analyzer.ollama_")]


def test_ready_logs_one_info_line_and_the_worker_runs(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    transport = ProbeTransport(tags(MODEL))

    code, _ = run(tmp_path, transport, make_settings(OLLAMA_MODEL=MODEL, OLLAMA_HOST="http://oh:1/"))

    assert code == 0
    assert transport.calls == [("http://oh:1/api/tags", PROBE_TIMEOUT_SEC)]
    [line] = probe_lines(capsys)
    assert (line["event"], line["level"]) == ("analyzer.ollama_ready", "info")
    assert (line["host"], line["model"]) == ("http://oh:1/", MODEL)


@pytest.mark.parametrize("response", [tags("other:1b"), tags()])
def test_model_missing_warns_with_the_exact_pull_command(
    response: HttpResponse, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, _ = run(tmp_path, ProbeTransport(response))

    assert code == 0
    [line] = probe_lines(capsys)
    assert (line["event"], line["level"]) == ("analyzer.ollama_model_missing", "warning")
    assert line["hint"] == f"ollama pull {MODEL}"
    assert (line["host"], line["model"]) == ("http://host.docker.internal:11434", MODEL)


@pytest.mark.parametrize(
    ("result", "needle"),
    [
        (ConnectionRefusedError("refused"), "ConnectionRefusedError: refused"),
        (TimeoutError("timed out"), "TimeoutError: timed out"),
        (HttpResponse(503, b"loading"), "HTTP 503: loading"),
        (HttpResponse(200, b"<html>"), "not JSON"),
        (HttpResponse(200, b'["x"]'), "not a JSON object"),
        (HttpResponse(200, b"{}"), "no models list"),
    ],
)
def test_unreachable_and_unexpected_answers_warn_and_startup_continues(
    result: HttpResponse | BaseException,
    needle: str,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    code, _ = run(tmp_path, ProbeTransport(result))

    assert code == 0
    [line] = probe_lines(capsys)
    assert (line["event"], line["level"]) == ("analyzer.ollama_unreachable", "warning")
    assert needle in line["error"]
    assert "retried when Ollama is up" in line["note"]
    assert line["model"] == MODEL


@pytest.mark.parametrize(
    "result",
    [tags(MODEL), tags(), OSError("down"), TimeoutError("slow"), HttpResponse(500, b"")],
    ids=["ready", "missing", "unreachable", "timeout", "non-2xx"],
)
def test_every_outcome_still_reaches_the_handler_and_returns_0(
    result: HttpResponse | BaseException, tmp_path: Path
) -> None:
    job = make_job("dQw4w9WgXcQ", dedupe_key=analyze_dedupe_key(make_settings().PROMPT_VERSION, "ollama"))
    handler_factory, seen = spy_handler_factory()

    code = analyzer_main.main(
        settings=make_settings(OLLAMA_MODEL=MODEL),
        connect=FakeDatabase(jobs=[job]).connect,
        summarizer_factory=fake_summarizer_factory(),
        handler_factory=handler_factory,
        worker_options=worker_options(tmp_path),
        max_iterations=1,
        probe_transport=ProbeTransport(result),
    )

    assert code == 0
    assert [j.id for j in seen] == [job.id]


def test_the_probe_line_comes_after_started_and_before_the_worker(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    run(tmp_path, ProbeTransport(OSError("down")))

    events = [line["event"] for line in log_lines(capsys)]
    assert events.index("analyzer.started") < events.index("analyzer.ollama_unreachable")


def test_credentials_in_ollama_host_never_reach_a_log_line(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    settings = make_settings(OLLAMA_MODEL=MODEL, OLLAMA_HOST="http://bob:Hunter2-PW@oh:11434")

    for result in (tags(MODEL), tags(), ConnectionError("cannot reach http://bob:Hunter2-PW@oh:11434")):
        run(tmp_path, ProbeTransport(result), settings)

    output = capsys.readouterr()
    text = output.out + output.err
    assert "Hunter2-PW" not in text
    assert "bob:" not in text
    assert "http://oh:11434" in text


def test_anthropic_sends_nothing_to_ollama_and_logs_no_probe_line(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    settings = make_settings(SUMMARIZER="anthropic", ANTHROPIC_API_KEY="sk-ant-x")

    code, _ = run(tmp_path, fail_if_called, settings)

    assert code == 0
    assert probe_lines(capsys) == []


@pytest.mark.parametrize("case", ["invalid_settings", "no_key", "unbuildable"])
def test_exit_2_paths_send_no_probe_request(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    summarizer_factory: Any = fail_if_called
    settings: Settings | None
    if case == "invalid_settings":
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.invalid/v")
        monkeypatch.setenv("SUMMARIZER", "bogus")
        settings = None
    elif case == "no_key":
        settings = make_settings(SUMMARIZER="anthropic", ANTHROPIC_API_KEY=None)
    else:
        settings = make_settings()

        def broken(s: Settings) -> Any:
            raise FileNotFoundError("prompt version 'v9' not found")

        summarizer_factory = broken

    code = analyzer_main.main(
        settings=settings,
        connect=fail_if_called,
        handler_factory=spy_handler_factory()[0],
        worker_options=worker_options(tmp_path),
        max_iterations=1,
        summarizer_factory=summarizer_factory,
        probe_transport=fail_if_called,
    )

    assert code == 2
    assert probe_lines(capsys) == []


def test_the_probe_needs_no_database_connection(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    order: list[str] = []

    def transport(url: str, timeout: float) -> HttpResponse:
        order.append("probe")
        return tags(MODEL)

    def connect() -> Any:
        order.append("connect")
        raise OSError("db down")

    analyzer_main.main(
        settings=make_settings(OLLAMA_MODEL=MODEL),
        connect=connect,
        summarizer_factory=fake_summarizer_factory(),
        handler_factory=spy_handler_factory()[0],
        worker_options=worker_options(tmp_path),
        max_iterations=1,
        probe_transport=transport,
    )

    assert order[0] == "probe"
