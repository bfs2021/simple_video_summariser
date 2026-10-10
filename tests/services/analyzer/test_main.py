"""The analyzer entrypoint (issue #33): startup, wiring, the config guard.

Everything here runs under ``-m "not integration"``: ``main`` gets a
``FakeDatabase`` for ``connect``, a ``FakeSummarizer`` factory, a spy handler
and worker options that replace the clock, so there is no Postgres, no
network and no real sleep. The three subprocess tests at the end prove what
only a real process can: the import graph, the exit code of a bad
configuration, and SIGTERM during a database outage.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import structlog

from common.config import Settings, get_settings
from common.errors import BugError, Defer
from common.logging import configure_logging
from common.models import Summarizer, VideoMeta
from common.queue import Job
from common.worker import JobContext
from services.analyzer import main as analyzer_main
from tests.services.analyzer.fake_db import CLAIM_MARKER, HEARTBEAT_MARKER, FakeDatabase
from tests.services.analyzer.fakes import FakeSummarizer, make_job
from tests.services.transcriber.fakes import make_ctx

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PASSWORD = "PW-SENTINEL-4f1c"
API_KEY = "sk-ant-SENTINEL-77ab"
DB_URL = f"postgresql://user:{DB_PASSWORD}@db.invalid/videos"
NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def make_settings(**values: Any) -> Settings:
    # ``Any`` lets plain strings stand in for the SecretStr fields.
    return Settings(**{"DATABASE_URL": DB_URL, **values})


@pytest.fixture(autouse=True)
def _isolated_process_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        "SUMMARIZER",
        "ANTHROPIC_API_KEY",
        "DATABASE_URL",
        "OVERLAP_SEC",
        "CHUNK_SEC",
        "PROMPT_VERSION",
        "LOG_FORMAT",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    structlog.reset_defaults()
    get_settings.cache_clear()


def worker_options(tmp_path: Path, sleeps: list[float] | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "liveness_path": tmp_path / "heartbeat",
        "install_signal_handlers": False,
        "sleep": sleeps.append if sleeps is not None else (lambda _s: None),
        **extra,
    }


def log_lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    captured = capsys.readouterr()
    lines = (captured.out + captured.err).splitlines()
    return [json.loads(line) for line in lines if line.startswith("{")]


def spy_handler_factory() -> tuple[Callable[..., Callable[[Job, JobContext], None]], list[Job]]:
    seen: list[Job] = []

    def factory(*, connect: Any, summarizer: Summarizer, settings: Settings) -> Callable[..., None]:
        def handler(job: Job, ctx: JobContext) -> None:
            seen.append(job)

        return handler

    return factory, seen


def fail_if_called(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("must not be called")


def fake_summarizer_factory(name: str = "ollama") -> Callable[[Settings], Summarizer]:
    return lambda settings: FakeSummarizer(name=name)


# ---------------------------------------------------------------- startup logging


def test_configure_logging_runs_before_anything_is_logged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    order: list[str] = []
    real = configure_logging

    def spy(settings: Settings, **kwargs: Any) -> None:
        # Unconfigured structlog prints to stdout: nothing may be there yet.
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""
        order.append("configure_logging")
        real(settings, **kwargs)

    monkeypatch.setattr(analyzer_main, "configure_logging", spy)
    db = FakeDatabase()

    code = analyzer_main.main(
        settings=make_settings(),
        connect=db.connect,
        summarizer_factory=fake_summarizer_factory(),
        handler_factory=spy_handler_factory()[0],
        worker_options=worker_options(tmp_path),
        max_iterations=1,
    )

    assert code == 0
    assert order == ["configure_logging"]


def test_started_line_records_the_ollama_configuration(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    settings = make_settings(
        SUMMARIZER="ollama", OLLAMA_MODEL="qwen-x", CHUNK_SEC=600, OVERLAP_SEC=30, PROMPT_VERSION="v1"
    )

    analyzer_main.main(
        settings=settings,
        connect=FakeDatabase().connect,
        handler_factory=spy_handler_factory()[0],
        worker_options=worker_options(tmp_path),
        max_iterations=1,
    )

    started = [line for line in log_lines(capsys) if line["event"] == "analyzer.started"]
    assert len(started) == 1
    line = started[0]
    assert line["level"] == "info"
    assert line["worker"] == f"analyzer-{socket.gethostname()}-{os.getpid()}"
    assert line["summarizer"] == "ollama"
    assert line["model"] == "qwen-x"
    assert line["prompt_version"] == "v1"
    assert line["chunk_sec"] == 600
    assert line["overlap_sec"] == 30
    assert "anthropic_batch" not in line


@pytest.mark.parametrize("batch", [True, False])
def test_started_line_records_model_and_batch_flag_for_anthropic(
    batch: bool, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    settings = make_settings(
        SUMMARIZER="anthropic",
        ANTHROPIC_API_KEY=API_KEY,
        ANTHROPIC_MODEL="claude-x",
        ANTHROPIC_BATCH=batch,
    )

    analyzer_main.main(
        settings=settings,
        connect=FakeDatabase().connect,
        handler_factory=spy_handler_factory()[0],
        worker_options=worker_options(tmp_path),
        max_iterations=1,
    )

    (line,) = [entry for entry in log_lines(capsys) if entry["event"] == "analyzer.started"]
    assert line["summarizer"] == "anthropic"
    assert line["model"] == "claude-x"
    assert line["anthropic_batch"] is batch


# ------------------------------------------------------- configuration errors


@pytest.mark.parametrize(
    ("env", "named"),
    [
        ({"SUMMARIZER": "openai"}, "SUMMARIZER"),
        ({"SUMMARIZER": ""}, "SUMMARIZER"),
        ({"SUMMARIZER": "Anthropic"}, "SUMMARIZER"),
        ({"DATABASE_URL": None}, "DATABASE_URL"),
        ({"CHUNK_SEC": "not-a-number"}, "CHUNK_SEC"),
        ({"CHUNK_SEC": "100", "OVERLAP_SEC": "100"}, "OVERLAP_SEC"),
    ],
)
def test_invalid_settings_exit_2_naming_the_variable_and_touch_nothing(
    env: dict[str, str | None],
    named: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name, value in {"DATABASE_URL": DB_URL, **env}.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    code = analyzer_main.main(connect=fail_if_called, summarizer_factory=fail_if_called)

    assert code == 2
    lines = log_lines(capsys)
    errors = [line for line in lines if line["level"] == "error"]
    assert len(errors) == 1
    assert named in json.dumps(errors[0])
    assert all(line["event"] != "analyzer.started" for line in lines)


def test_invalid_settings_still_configure_logging_first(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("SUMMARIZER", "bogus")
    calls: list[str] = []
    real = configure_logging

    def spy(settings: Settings, **kwargs: Any) -> None:
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""
        calls.append("configure_logging")
        real(settings, **kwargs)

    monkeypatch.setattr(analyzer_main, "configure_logging", spy)

    code = analyzer_main.main(connect=fail_if_called, summarizer_factory=fail_if_called)

    assert code == 2
    assert calls == ["configure_logging"]


@pytest.mark.parametrize("key", [None, "", "   "])
def test_anthropic_without_a_key_exits_2_before_any_database_connection(
    key: str | None, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    settings = make_settings(SUMMARIZER="anthropic", ANTHROPIC_API_KEY=key)

    code = analyzer_main.main(
        settings=settings,
        connect=fail_if_called,
        handler_factory=spy_handler_factory()[0],
        worker_options=worker_options(tmp_path),
        max_iterations=1,
    )

    assert code == 2
    output = json.dumps(log_lines(capsys))
    assert "ANTHROPIC_API_KEY" in output
    assert "analyzer.started" not in output


def test_a_summarizer_that_cannot_be_built_exits_2_before_any_database_connection(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    def broken(settings: Settings) -> Summarizer:
        raise FileNotFoundError("prompt version 'v9' not found")

    code = analyzer_main.main(
        settings=make_settings(),
        connect=fail_if_called,
        summarizer_factory=broken,
        handler_factory=spy_handler_factory()[0],
        worker_options=worker_options(tmp_path),
        max_iterations=1,
    )

    assert code == 2
    assert "prompt version" in json.dumps(log_lines(capsys))


def test_no_output_contains_the_api_key_or_the_database_password(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # 1. invalid settings, secrets present in the environment
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    monkeypatch.setenv("SUMMARIZER", "bogus")
    analyzer_main.main(connect=fail_if_called, summarizer_factory=fail_if_called)
    monkeypatch.delenv("SUMMARIZER")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("DATABASE_URL")
    # 2. a full healthy start with the real summarizer factory and the anthropic key
    analyzer_main.main(
        settings=make_settings(SUMMARIZER="anthropic", ANTHROPIC_API_KEY=API_KEY),
        connect=FakeDatabase().connect,
        handler_factory=spy_handler_factory()[0],
        worker_options=worker_options(tmp_path),
        max_iterations=2,
    )
    # 3. a build failure
    analyzer_main.main(
        settings=make_settings(),
        connect=fail_if_called,
        summarizer_factory=lambda s: (_ for _ in ()).throw(RuntimeError("boom")),
        worker_options=worker_options(tmp_path),
        max_iterations=1,
    )

    captured = capsys.readouterr()
    everything = captured.out + captured.err
    assert "analyzer.started" in everything  # the run above really logged
    assert API_KEY not in everything
    assert DB_PASSWORD not in everything


# ------------------------------------------------------------------ worker wiring


def test_the_worker_claims_only_analyze_jobs_under_a_per_process_name(tmp_path: Path) -> None:
    db = FakeDatabase()

    analyzer_main.main(
        settings=make_settings(),
        connect=db.connect,
        summarizer_factory=fake_summarizer_factory(),
        handler_factory=spy_handler_factory()[0],
        worker_options=worker_options(tmp_path),
        max_iterations=2,
    )

    claims = [p for conn in db.connections for p in conn.statements(CLAIM_MARKER)]
    assert len(claims) == 2
    for params in claims:
        assert params["kinds"] == ["analyze"]
        assert params["worker"] == f"analyzer-{socket.gethostname()}-{os.getpid()}"


def test_the_heartbeat_uses_its_own_connection(tmp_path: Path) -> None:
    db = FakeDatabase(jobs=[make_job("abcdefghijk", dedupe_key="v1:ollama")])
    fired: list[Callable[[], None]] = []

    class Handle:
        def cancel(self) -> None:
            return None

    def timer_factory(interval: float, function: Callable[[], None]) -> Handle:
        if not fired:  # the first heartbeat tick fires at once, later ones never
            fired.append(function)
            function()
        return Handle()

    factory, seen = spy_handler_factory()
    analyzer_main.main(
        settings=make_settings(),
        connect=db.connect,
        summarizer_factory=fake_summarizer_factory(),
        handler_factory=factory,
        worker_options=worker_options(tmp_path, timer_factory=timer_factory),
        max_iterations=1,
    )

    assert len(seen) == 1
    claim_conn, heartbeat_conn = db.connections[0], db.connections[1]
    assert claim_conn is not heartbeat_conn
    assert claim_conn.statements(CLAIM_MARKER) and not claim_conn.statements(HEARTBEAT_MARKER)
    assert heartbeat_conn.statements(HEARTBEAT_MARKER) and not heartbeat_conn.statements(
        CLAIM_MARKER
    )


def test_the_default_liveness_path_is_the_worker_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class SpyWorker:
        def __init__(self, name: str, kinds: Any, handler: Any, queue: Any, **kwargs: Any) -> None:
            captured.update(kwargs, kinds=list(kinds))

        def run(self, max_iterations: int | None = None) -> None:
            return None

    monkeypatch.setattr(analyzer_main, "Worker", SpyWorker)

    analyzer_main.main(
        settings=make_settings(),
        connect=fail_if_called,
        summarizer_factory=fake_summarizer_factory(),
        handler_factory=spy_handler_factory()[0],
    )

    assert captured["kinds"] == ["analyze"]
    assert "liveness_path" not in captured  # so common.worker's /tmp/heartbeat applies


def test_a_database_down_at_startup_backs_off_capped_and_retries_without_exiting(
    tmp_path: Path,
) -> None:
    db = FakeDatabase(failures=10)
    sleeps: list[float] = []

    code = analyzer_main.main(
        settings=make_settings(),
        connect=db.connect,
        summarizer_factory=fake_summarizer_factory(),
        handler_factory=spy_handler_factory()[0],
        worker_options=worker_options(tmp_path, sleeps),
        max_iterations=14,
    )

    assert code == 0
    assert sleeps[:10] == [1, 2, 4, 8, 16, 32, 60, 60, 60, 60]
    assert db.connect_calls == 11  # ten refusals, then the first connection
    assert len(db.connections) == 1
    assert db.connections[0].statements(CLAIM_MARKER)  # and it went on to claim


def test_the_summarizer_and_the_handler_are_built_once_per_process(tmp_path: Path) -> None:
    db = FakeDatabase(jobs=[make_job("aaaaaaaaaaa", dedupe_key="v1:ollama")])
    built: list[str] = []
    handlers: list[str] = []
    seen: list[Job] = []

    def summarizers(settings: Settings) -> Summarizer:
        built.append("summarizer")
        return FakeSummarizer()

    def handler_factory(*, connect: Any, summarizer: Summarizer, settings: Settings) -> Any:
        handlers.append("handler")
        return lambda job, ctx: seen.append(job)

    analyzer_main.main(
        settings=make_settings(),
        connect=db.connect,
        summarizer_factory=summarizers,
        handler_factory=handler_factory,
        worker_options=worker_options(tmp_path),
        max_iterations=5,
    )

    assert built == ["summarizer"]
    assert handlers == ["handler"]
    assert len(seen) == 1


def test_the_handler_gets_the_summarizer_that_was_built(tmp_path: Path) -> None:
    summarizer = FakeSummarizer(name="ollama")
    given: list[Summarizer] = []

    def handler_factory(*, connect: Any, summarizer: Summarizer, settings: Settings) -> Any:
        given.append(summarizer)
        return lambda job, ctx: None

    analyzer_main.main(
        settings=make_settings(),
        connect=FakeDatabase().connect,
        summarizer_factory=lambda s: summarizer,
        handler_factory=handler_factory,
        worker_options=worker_options(tmp_path),
        max_iterations=1,
    )

    assert given == [summarizer]


# --------------------------------------------------- job / configuration mismatch


def guarded(
    summarizer: FakeSummarizer, **settings: Any
) -> tuple[Callable[[Job, JobContext], None], list[Job]]:
    seen: list[Job] = []

    def handler_factory(
        *, connect: Any, summarizer: Summarizer, settings: Settings
    ) -> Callable[[Job, JobContext], None]:
        def handler(job: Job, ctx: JobContext) -> None:
            seen.append(job)
            summarizer.derive_roster(replace(_META, video_id=job.video_id), "opening")

        return handler

    handler = analyzer_main.build_job_handler(
        make_settings(**settings),
        summarizer,
        fail_if_called,
        handler_factory=handler_factory,
        now=lambda: NOW,
    )
    return handler, seen


_META = VideoMeta(
    video_id="abcdefghijk",
    channel_id="UC" + "a" * 22,
    title="t",
    description="d",
    duration_sec=1,
    published_at=None,
    language=None,
    live_status=None,
    manual_subtitle_langs=(),
    auto_caption_langs=(),
)


def test_a_matching_key_runs_the_handler() -> None:
    summarizer = FakeSummarizer(name="ollama")
    handler, seen = guarded(summarizer, PROMPT_VERSION="v1")
    ctx, _logs = make_ctx()

    handler(make_job("abcdefghijk", dedupe_key="v1:ollama"), ctx)

    assert [job.video_id for job in seen] == ["abcdefghijk"]
    assert len(summarizer.calls) == 1


@pytest.mark.parametrize("key", ["v2:ollama", "v1:anthropic", "v2:anthropic", "V1:ollama"])
def test_a_well_formed_key_that_does_not_match_defers_15_minutes_without_calling_the_summarizer(
    key: str,
) -> None:
    summarizer = FakeSummarizer(name="ollama")
    handler, seen = guarded(summarizer, PROMPT_VERSION="v1")
    ctx, logs = make_ctx()

    with pytest.raises(Defer) as deferred:
        handler(make_job("abcdefghijk", dedupe_key=key), ctx)

    assert deferred.value.until == NOW + timedelta(minutes=15)
    assert seen == []
    assert summarizer.calls == []
    (warning,) = logs.records("warning")
    assert warning["event"] == "analyzer.job_config_mismatch"
    assert key in json.dumps(warning, default=str)
    assert "v1:ollama" in json.dumps(warning, default=str)


@pytest.mark.parametrize("key", ["default", "", "v1", "v1:a:b", ":ollama", "v1:", "v1: ollama", "v1:ollama\n"])
def test_a_malformed_key_is_a_bug_and_does_not_call_the_summarizer(key: str) -> None:
    summarizer = FakeSummarizer(name="ollama")
    handler, seen = guarded(summarizer, PROMPT_VERSION="v1")
    ctx, logs = make_ctx()

    with pytest.raises(BugError):
        handler(make_job("abcdefghijk", dedupe_key=key), ctx)

    assert seen == []
    assert summarizer.calls == []
    assert logs.records("warning") == []


def test_the_expected_key_follows_the_configured_prompt_version_and_backend() -> None:
    summarizer = FakeSummarizer(name="anthropic")
    handler, seen = guarded(summarizer, PROMPT_VERSION="v2")
    ctx, _logs = make_ctx()

    handler(make_job("abcdefghijk", dedupe_key="v2:anthropic"), ctx)
    with pytest.raises(Defer):
        handler(make_job("abcdefghijk", dedupe_key="v1:anthropic"), ctx)

    assert len(seen) == 1


def test_two_jobs_for_different_videos_give_the_same_results_as_each_alone() -> None:
    def run(*video_ids: str) -> list[tuple[Any, ...]]:
        summarizer = FakeSummarizer(name="ollama")
        handler, seen = guarded(summarizer, PROMPT_VERSION="v1")
        ctx, _logs = make_ctx()
        for video_id in video_ids:
            handler(make_job(video_id, dedupe_key="v1:ollama"), ctx)
        return [(job.video_id, job.dedupe_key) for job in seen] + [
            (call[0], call[1].video_id) for call in summarizer.calls
        ]

    together = run("aaaaaaaaaaa", "bbbbbbbbbbb")
    alone_a = run("aaaaaaaaaaa")
    alone_b = run("bbbbbbbbbbb")

    assert together == [
        alone_a[0],
        alone_b[0],
        alone_a[1],
        alone_b[1],
    ]


# -------------------------------------------------------------------- isolation


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_analyzer_sources_import_nothing_from_the_transcriber_side() -> None:
    files = [*(REPO_ROOT / "services" / "analyzer").glob("*.py")]
    files.append(REPO_ROOT / "adapters" / "summarize" / "factory.py")
    assert len(files) >= 3

    for path in files:
        for module in _imports(path):
            assert not module.startswith("services.transcriber"), (path, module)
            assert not module.startswith("adapters.transcription"), (path, module)
            assert module.split(".")[0] != "faster_whisper", (path, module)


# ------------------------------------------------------------- subprocess tests


def _clean_env(**values: str) -> dict[str, str]:
    """A from-scratch environment: nothing of the developer's leaks into the child."""
    return {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", **values}


def test_importing_the_entrypoint_loads_no_whisper_modules() -> None:
    code = (
        "import json, sys\n"
        "import services.analyzer.main\n"
        "bad = [m for m in sys.modules if m == 'faster_whisper' or m.startswith('faster_whisper.')"
        " or m == 'adapters.transcription' or m.startswith('adapters.transcription.')"
        " or m == 'services.transcriber' or m.startswith('services.transcriber.')]\n"
        "print(json.dumps(bad))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def _run_module(env: dict[str, str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "services.analyzer.main"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def test_a_bogus_summarizer_exits_2_and_stderr_names_it() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "services.analyzer.main"],
        cwd=REPO_ROOT,
        env=_clean_env(DATABASE_URL=DB_URL, SUMMARIZER="bogus"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert "SUMMARIZER" in result.stderr
    assert DB_PASSWORD not in result.stdout + result.stderr


def _reader(stream: Any, lines: queue.Queue[str | None]) -> None:
    for line in stream:
        lines.put(line)
    lines.put(None)


def _wait_for_events(lines: queue.Queue[str | None], event: str, count: int, seen: list[str]) -> None:
    """Block until ``count`` log lines with ``event`` arrived (a hard timeout, no fixed sleep)."""
    found = 0
    while found < count:
        line = lines.get(timeout=30)
        assert line is not None, "process exited early:\n" + "".join(seen)
        seen.append(line)
        if re.search(rf'"event": "{re.escape(event)}"', line):
            found += 1


def _closed_loopback_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_sigterm_during_a_database_outage_exits_0_within_the_grace_period() -> None:
    port = _closed_loopback_port()
    grace = 5
    env = _clean_env(
        DATABASE_URL=f"postgresql://user:{DB_PASSWORD}@127.0.0.1:{port}/db?connect_timeout=3",
        OLLAMA_HOST=f"http://127.0.0.1:{port}",  # startup probe (#84): refused at once, no outside call
        WORKER_SHUTDOWN_GRACE_SEC=str(grace),
        LOG_FORMAT="json",
    )
    proc = _run_module(env)
    lines: queue.Queue[str | None] = queue.Queue()
    assert proc.stdout is not None
    threading.Thread(target=_reader, args=(proc.stdout, lines), daemon=True).start()
    seen: list[str] = []
    try:
        # Two failed iterations: the process survived a backoff, so it is not crash-looping.
        _wait_for_events(lines, "worker.iteration_failed", 2, seen)
        assert proc.poll() is None

        proc.send_signal(signal.SIGTERM)
        returncode = proc.wait(timeout=grace)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    assert returncode == 0, "".join(seen)
    assert DB_PASSWORD not in "".join(seen)
