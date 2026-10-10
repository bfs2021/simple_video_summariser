"""Transcriber service entrypoint, everything that needs no database (issue #32).

``main()`` and the module's builders are driven in-process with fake handlers, a
fake transcriber and a fake queue. Loops end through a real ``SIGTERM`` raised
from inside a handler (``signal.raise_signal``) or through ``run(max_iterations=)``,
so nothing here waits on a clock. The subprocess tests (real signals, ``kill -9``,
two replicas) are in ``test_main_process.py``.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
import pytest
import structlog

import services.transcriber.main as main_module
from common.config import get_settings
from common.errors import (
    BugError,
    Cancelled,
    JobError,
    ToolFailureError,
    TransientNetworkError,
    classify,
)
from common.logging import configure_logging
from common.models import AudioRef, TranscriptResult
from common.queue import Job, JobQueue
from services.transcriber.main import (
    LazyTranscriber,
    QueuePool,
    build_worker,
    default_transcriber_factory,
    dispatcher,
    main,
    open_queue_with_backoff,
    prepare_audio_dir,
    production_queue_factory,
    worker_name,
)
from tests.services.transcriber.fakes import make_job, make_settings

REPO_ROOT = Path(__file__).resolve().parents[3]
VID = "abcdefghijk"
AUDIO = AudioRef("ab/abcdefghijk.opus", 4, 1.0)
RESULT = TranscriptResult(segments=(), language="en", engine_meta={})
SENTINEL_PASSWORD = "SENTINEL-db-password"
SENTINEL_KEY = "sk-SENTINEL-api-key"
DSN = f"postgresql://user:{SENTINEL_PASSWORD}@localhost:5432/db"


@pytest.fixture(autouse=True)
def isolated_process_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """A valid environment, and no logging or settings state leaking between tests."""
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    monkeypatch.setenv("DATABASE_URL", DSN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    monkeypatch.setenv("AUDIO_DIR", str(tmp_path / "audio"))
    monkeypatch.setattr(main_module, "LIVENESS_PATH", tmp_path / "heartbeat")
    get_settings.cache_clear()
    yield
    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------


def job(kind: str = "ingest", job_id: int = 1) -> Job:
    return dataclasses.replace(make_job(VID), kind=kind, id=job_id)


class FakeQueues:
    """A queue factory: every call opens a new ``FakeQueue`` sharing this state.

    ``claim`` mimics ``PostgresQueue.claim``: a plain exception from the handler is
    classified and recorded, ``Cancelled``/``SystemExit`` are recorded and re-raised.
    """

    def __init__(
        self,
        jobs: Sequence[Job] = (),
        *,
        open_errors: Sequence[BaseException] = (),
        claim_errors: Sequence[BaseException] = (),
    ) -> None:
        self.jobs = list(jobs)
        self.open_errors = list(open_errors)
        self.claim_errors = list(claim_errors)
        self.made: list[FakeQueue] = []
        self.calls = 0
        self.outcomes: list[str] = []
        self.claimed_by: list[str] = []
        self.kinds: list[tuple[str, ...]] = []

    def __call__(self) -> FakeQueue:
        self.calls += 1
        if self.open_errors:
            raise self.open_errors.pop(0)
        queue = FakeQueue(self)
        self.made.append(queue)
        return queue


class FakeQueue:
    def __init__(self, shared: FakeQueues) -> None:
        self.shared = shared
        self.claims = 0
        self.closed = False

    @contextmanager
    def claim(
        self, kinds: Sequence[str], *, worker: str, dedupe_key: str | None = None
    ) -> Iterator[Job | None]:
        shared = self.shared
        self.claims += 1
        shared.kinds.append(tuple(kinds))
        shared.claimed_by.append(worker)
        if shared.claim_errors:
            raise shared.claim_errors.pop(0)
        if not shared.jobs:
            yield None
            return
        claimed = shared.jobs.pop(0)
        try:
            yield claimed
        except (Cancelled, KeyboardInterrupt, SystemExit):
            shared.outcomes.append("pending")
            raise
        except JobError as exc:
            shared.outcomes.append(exc.error_class.value)
        except Exception as exc:  # noqa: BLE001 - mirrors PostgresQueue.claim's safety net
            shared.outcomes.append(classify(exc).value)
        else:
            shared.outcomes.append("done")

    def heartbeat(self, job_id: int, worker: str) -> bool:
        return True

    def enqueue(self, kind: str, video_id: str, **kwargs: Any) -> int | None:
        raise NotImplementedError

    def reap_stale(self, older_than_sec: int | None = None) -> int:
        raise NotImplementedError

    def close(self) -> None:
        self.closed = True


class Calls:
    """Handlers that record ``(kind, job id)`` and raise SIGTERM after ``stop_after`` calls."""

    def __init__(self, stop_after: int) -> None:
        self.stop_after = stop_after
        self.seen: list[tuple[str, int]] = []

    def handler(self, kind: str) -> Callable[..., None]:
        def run(claimed: Job, ctx: object) -> None:
            self.seen.append((kind, claimed.id))
            if len(self.seen) >= self.stop_after:
                signal.raise_signal(signal.SIGTERM)

        return run

    def handlers(self, *kinds: str) -> dict[str, Callable[..., None]]:
        return {kind: self.handler(kind) for kind in (kinds or ("ingest", "transcribe"))}


def log_events(out: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def events_named(out: str, event: str) -> list[dict[str, Any]]:
    return [e for e in log_events(out) if e.get("event") == event]


# ---------------------------------------------------------------------------
# module contract
# ---------------------------------------------------------------------------


def test_importing_the_module_has_no_side_effects(tmp_path: Path) -> None:
    script = tmp_path / "probe.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(REPO_ROOT)!r})
            import psycopg
            import common.config

            def refuse(*args, **kwargs):
                raise AssertionError("touched the database or the settings at import")

            psycopg.connect = refuse
            common.config.get_settings = refuse
            import services.transcriber.main
            assert "faster_whisper" not in sys.modules
            assert "alembic" not in sys.modules
            print("OK")
            """
        )
    )
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.stdout.strip() == "OK", result.stderr


def test_module_has_a_main_guard_that_exits_with_the_return_value() -> None:
    tree = ast.parse(Path(main_module.__file__ or "").read_text(encoding="utf-8"))
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
    ]
    assert len(guards) == 1
    assert "sys.exit(main())" in ast.unparse(guards[0])


def test_invoking_the_module_with_invalid_settings_exits_2(tmp_path: Path) -> None:
    bare_environment = {"PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [sys.executable, "-m", "services.transcriber.main"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=REPO_ROOT,
        env=bare_environment,
    )
    assert result.returncode == 2
    assert "DATABASE_URL" in result.stderr


def test_module_never_touches_migrations() -> None:
    tree = ast.parse(Path(main_module.__file__ or "").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any("alembic" in name for name in imported)
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "upgrade" not in called


def test_module_holds_no_mutable_state_at_module_level() -> None:
    tree = ast.parse(Path(main_module.__file__ or "").read_text(encoding="utf-8"))
    mutable = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)
    offenders = [
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, mutable)
    ]
    assert offenders == []


def test_liveness_file_is_tmp_heartbeat() -> None:
    source = Path(main_module.__file__ or "").read_text(encoding="utf-8")
    assert 'Path("/tmp/heartbeat")' in source


# ---------------------------------------------------------------------------
# kinds, dispatch, worker name
# ---------------------------------------------------------------------------


def test_worker_name_is_transcriber_host_pid() -> None:
    assert worker_name() == f"transcriber:{socket.gethostname()}:{os.getpid()}"


def test_the_worker_claims_exactly_the_keys_of_the_handler_map() -> None:
    queues = FakeQueues([job("ingest")])
    calls = Calls(stop_after=1)

    rc = main([], handlers=calls.handlers("ingest", "transcribe"), queue_factory=queues)

    assert rc == 0
    assert queues.kinds == [("ingest", "transcribe")]


def test_the_claimed_kinds_follow_the_map_so_they_cannot_drift() -> None:
    queues = FakeQueues([job("beta")])
    calls = Calls(stop_after=1)

    main([], handlers=calls.handlers("beta", "alpha"), queue_factory=queues)

    assert queues.kinds == [("beta", "alpha")]


def test_each_job_goes_to_the_handler_for_its_kind() -> None:
    queues = FakeQueues([job("ingest", 1), job("transcribe", 2), job("ingest", 3)])
    calls = Calls(stop_after=3)

    rc = main([], handlers=calls.handlers(), queue_factory=queues)

    assert rc == 0
    assert calls.seen == [("ingest", 1), ("transcribe", 2), ("ingest", 3)]
    assert queues.outcomes == ["done", "done", "done"]


def test_a_kind_missing_from_the_map_raises_a_bug_error() -> None:
    dispatch = dispatcher({"ingest": lambda claimed, ctx: None})

    with pytest.raises(BugError, match="analyze"):
        dispatch(job("analyze"), None)  # type: ignore[arg-type]


def test_an_unknown_kind_is_recorded_and_the_loop_carries_on() -> None:
    queues = FakeQueues([job("analyze", 1), job("ingest", 2)])
    calls = Calls(stop_after=1)

    main([], handlers=calls.handlers("ingest", "transcribe"), queue_factory=queues)

    assert queues.outcomes == ["BUG", "done"]
    assert calls.seen == [("ingest", 2)]


def test_the_worker_name_reaches_the_queue_and_the_startup_log(
    capsys: pytest.CaptureFixture[str],
) -> None:
    queues = FakeQueues([job("ingest")])

    main([], handlers=Calls(1).handlers(), queue_factory=queues)

    assert queues.claimed_by[0] == worker_name()
    (started,) = events_named(capsys.readouterr().out, "transcriber.started")
    assert started["worker"] == worker_name()


# ---------------------------------------------------------------------------
# connections: heartbeat isolation, reconnect, closing
# ---------------------------------------------------------------------------


def test_the_heartbeat_gets_its_own_queue_and_every_queue_is_closed_on_exit() -> None:
    queues = FakeQueues([job("ingest")])

    rc = main([], handlers=Calls(1).handlers(), queue_factory=queues)

    assert rc == 0
    claim_queue, heartbeat_queue = queues.made
    assert claim_queue.claims >= 1
    assert heartbeat_queue.claims == 0
    assert all(queue.closed for queue in queues.made)


def test_a_claim_failure_reconnects_and_the_dead_queue_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_build_worker = main_module.build_worker
    monkeypatch.setattr(
        main_module,
        "build_worker",
        lambda *args, **kwargs: real_build_worker(*args, sleep=lambda seconds: None, **kwargs),
    )
    queues = FakeQueues([job("ingest")], claim_errors=[psycopg.OperationalError("db went away")])

    rc = main([], handlers=Calls(1).handlers(), queue_factory=queues)

    assert rc == 0
    assert queues.outcomes == ["done"]
    dead, fresh, heartbeat = queues.made
    assert dead.closed
    assert fresh.claims >= 1
    assert heartbeat.claims == 0
    assert all(queue.closed for queue in queues.made)


def test_queue_pool_reconnect_closes_only_the_replaced_claim_queue() -> None:
    queues = FakeQueues()
    pool = QueuePool(queues)

    first = pool.open_main()
    heartbeat = pool.open_heartbeat()
    second = pool.reconnect()

    assert second is not first
    assert queues.made[0].closed
    assert not queues.made[1].closed
    assert not queues.made[2].closed
    pool.close_all()
    assert all(queue.closed for queue in queues.made)
    assert heartbeat is queues.made[1]


def test_queue_pool_close_all_survives_a_queue_that_cannot_close() -> None:
    queues = FakeQueues()
    pool = QueuePool(queues)
    pool.open_main()
    pool.open_heartbeat()

    def explode() -> None:
        raise psycopg.OperationalError("already gone")

    queues.made[0].close = explode  # type: ignore[method-assign]
    pool.close_all()

    assert queues.made[1].closed


def test_queue_pool_accepts_queues_that_have_no_close_method() -> None:
    class Bare:
        pass

    pool = QueuePool(lambda: Bare())  # type: ignore[arg-type,return-value]
    pool.open_main()
    pool.close_all()


def test_production_queue_factory_opens_one_connection_per_queue_and_closes_it() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    opened: list[FakeConnection] = []

    def connect() -> Any:
        opened.append(FakeConnection())
        return opened[-1]

    pool = QueuePool(production_queue_factory(make_settings(), connect=connect))

    pool.open_main()
    pool.open_heartbeat()
    assert len(opened) == 2
    assert not any(conn.closed for conn in opened)
    pool.close_all()
    assert all(conn.closed for conn in opened)


def test_liveness_file_is_touched_by_the_loop(tmp_path: Path) -> None:
    queues = FakeQueues([job("ingest")])

    main([], handlers=Calls(1).handlers(), queue_factory=queues)

    assert (tmp_path / "heartbeat").exists()


# ---------------------------------------------------------------------------
# startup order, settings and AUDIO_DIR failures
# ---------------------------------------------------------------------------


def test_startup_validates_settings_then_logging_then_creates_audio_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    order: list[str] = []
    real_settings = get_settings
    real_logging = configure_logging

    def spy_settings() -> Any:
        order.append("settings")
        return real_settings()

    def spy_logging(*args: Any, **kwargs: Any) -> None:
        order.append(f"logging(audio_dir_exists={(tmp_path / 'audio').exists()})")
        real_logging(*args, **kwargs)

    monkeypatch.setattr(main_module, "get_settings", spy_settings)
    monkeypatch.setattr(main_module, "configure_logging", spy_logging)

    queues = FakeQueues([job("ingest")])

    def queue_factory() -> JobQueue:
        order.append(f"queue(audio_dir_exists={(tmp_path / 'audio').is_dir()})")
        return queues()

    main([], handlers=Calls(1).handlers(), queue_factory=queue_factory)

    assert order[:3] == [
        "settings",
        "logging(audio_dir_exists=False)",
        "queue(audio_dir_exists=True)",
    ]
    assert capsys.readouterr().out.count("transcriber.started") == 1


def test_started_line_carries_the_worker_and_the_relevant_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    monkeypatch.setenv("WHISPER_COMPUTE", "int8_float32")
    monkeypatch.setenv("WHISPER_THREADS", "3")
    monkeypatch.setenv("AUDIO_KEEP", "false")

    main([], handlers=Calls(1).handlers(), queue_factory=FakeQueues([job()]))

    out = capsys.readouterr().out
    (started,) = events_named(out, "transcriber.started")
    assert started["level"] == "info"
    assert started["worker"] == worker_name()
    assert started["kinds"] == ["ingest", "transcribe"]
    assert started["whisper_model"] == "tiny"
    assert started["whisper_compute"] == "int8_float32"
    assert started["whisper_threads"] == 3
    assert started["audio_dir"] == str(tmp_path / "audio")
    assert started["audio_keep"] is False
    assert SENTINEL_PASSWORD not in out
    assert SENTINEL_KEY not in out


def test_missing_database_url_exits_2_naming_the_field(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DATABASE_URL")
    queues = FakeQueues()

    rc = main([], handlers=Calls(1).handlers(), queue_factory=queues)

    captured = capsys.readouterr()
    assert rc == 2
    (line,) = captured.err.strip().splitlines()
    assert "DATABASE_URL" in line
    assert queues.calls == 0


def test_invalid_settings_never_leak_secrets_to_stderr_or_the_logs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HEARTBEAT_SEC", "not-a-number")

    rc = main([], handlers=Calls(1).handlers(), queue_factory=FakeQueues())

    captured = capsys.readouterr()
    assert rc == 2
    (line,) = captured.err.strip().splitlines()
    assert "HEARTBEAT_SEC" in line
    for text in (captured.err, captured.out):
        assert SENTINEL_PASSWORD not in text
        assert SENTINEL_KEY not in text
        assert DSN not in text


def _audio_dir_error_lines(out: str) -> list[dict[str, Any]]:
    return [e for e in log_events(out) if e.get("level") == "error"]


def test_audio_dir_that_is_a_file_exits_2_and_names_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "audio-is-a-file"
    blocker.write_text("x")
    monkeypatch.setenv("AUDIO_DIR", str(blocker))
    queues = FakeQueues([job()])

    rc = main([], handlers=Calls(1).handlers(), queue_factory=queues)

    errors = _audio_dir_error_lines(capsys.readouterr().out)
    assert rc == 2
    assert any(str(blocker) in json.dumps(e) for e in errors)
    assert queues.calls == 0
    assert queues.claimed_by == []


def test_audio_dir_that_cannot_be_created_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "parent-is-a-file"
    blocker.write_text("x")
    target = blocker / "audio"
    monkeypatch.setenv("AUDIO_DIR", str(target))
    queues = FakeQueues()

    rc = main([], handlers=Calls(1).handlers(), queue_factory=queues)

    errors = _audio_dir_error_lines(capsys.readouterr().out)
    assert rc == 2
    assert any(str(target) in json.dumps(e) for e in errors)
    assert queues.calls == 0


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_audio_dir_that_is_not_writable_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o500)
    monkeypatch.setenv("AUDIO_DIR", str(readonly))
    queues = FakeQueues()
    try:
        rc = main([], handlers=Calls(1).handlers(), queue_factory=queues)
    finally:
        readonly.chmod(0o700)

    errors = _audio_dir_error_lines(capsys.readouterr().out)
    assert rc == 2
    assert any(str(readonly) in json.dumps(e) for e in errors)
    assert queues.calls == 0


def test_prepare_audio_dir_creates_nested_directories_and_leaves_no_probe_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "a" / "b"

    prepare_audio_dir(target)

    assert target.is_dir()
    assert list(target.iterdir()) == []


# ---------------------------------------------------------------------------
# database unreachable at startup
# ---------------------------------------------------------------------------


class Stopped:
    """A ``wait`` double: records the delays and reports 'stop requested' on demand."""

    def __init__(self, stop_on_call: int | None = None) -> None:
        self.delays: list[float] = []
        self.stop_on_call = stop_on_call

    def __call__(self, delay: float) -> bool:
        self.delays.append(delay)
        return self.stop_on_call is not None and len(self.delays) >= self.stop_on_call


def test_startup_retries_with_a_backoff_capped_at_60_seconds() -> None:
    failures = [psycopg.OperationalError("down")] * 8
    queues = FakeQueues(open_errors=failures)
    wait = Stopped()

    queue = open_queue_with_backoff(queues, threading.Event(), wait=wait)

    assert queue is queues.made[0]
    assert wait.delays == [1, 2, 4, 8, 16, 32, 60, 60]


def test_a_stop_request_during_the_startup_wait_returns_no_queue() -> None:
    queues = FakeQueues(open_errors=[psycopg.OperationalError("down")] * 5)

    queue = open_queue_with_backoff(queues, threading.Event(), wait=Stopped(stop_on_call=2))

    assert queue is None
    assert queues.calls == 2


def test_sigterm_while_the_database_is_down_exits_0_without_claiming(
    capsys: pytest.CaptureFixture[str],
) -> None:
    queues = FakeQueues(
        [job()],
        open_errors=[psycopg.OperationalError(f"could not connect using {DSN}")],
    )

    def factory() -> JobQueue:
        try:
            return queues()
        finally:
            signal.raise_signal(signal.SIGTERM)

    calls = Calls(stop_after=1)
    rc = main([], handlers=calls.handlers(), queue_factory=factory)

    out = capsys.readouterr().out
    assert rc == 0
    assert calls.seen == []
    assert queues.claimed_by == []
    errors = [e for e in log_events(out) if e.get("level") == "error"]
    assert errors, "the unreachable database is logged at ERROR"
    assert SENTINEL_PASSWORD not in out
    assert DSN not in out


# ---------------------------------------------------------------------------
# no state between jobs: the lazily loaded model
# ---------------------------------------------------------------------------


class FakeWhisper:
    """A ``Transcriber`` whose 'model load' happens inside the first ``transcribe``."""

    def __init__(self, load_errors: Sequence[BaseException] = ()) -> None:
        self.load_errors = list(load_errors)
        self.loads = 0
        self.load_attempts = 0
        self.calls = 0

    def transcribe(
        self,
        audio: AudioRef,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> TranscriptResult:
        self.calls += 1
        if self.loads == 0:
            self.load_attempts += 1
            if self.load_errors:
                raise self.load_errors.pop(0)
            self.loads += 1
        return RESULT


class Factory:
    def __init__(self, whisper: FakeWhisper, errors: Sequence[BaseException] = ()) -> None:
        self.whisper = whisper
        self.errors = list(errors)
        self.calls = 0

    def __call__(self) -> FakeWhisper:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.whisper


def use_transcriber_in_handlers(
    monkeypatch: pytest.MonkeyPatch, *, total_jobs: int
) -> None:
    """Replace the production handler builder with one that only uses the transcriber."""
    done = 0

    def fake_build_handlers(settings: Any, *, transcriber: Any, **kwargs: Any) -> dict[str, Any]:
        def count_and_stop() -> None:
            nonlocal done
            done += 1
            if done >= total_jobs:
                signal.raise_signal(signal.SIGTERM)

        def ingest(claimed: Job, ctx: object) -> None:
            count_and_stop()

        def transcribe(claimed: Job, ctx: object) -> None:
            try:
                transcriber.transcribe(AUDIO, language=None, on_progress=None)
            finally:
                count_and_stop()

        return {"ingest": ingest, "transcribe": transcribe}

    monkeypatch.setattr(main_module, "build_handlers", fake_build_handlers)


def test_the_model_loads_once_across_three_transcribe_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    whisper = FakeWhisper()
    factory = Factory(whisper)
    queues = FakeQueues([job("transcribe", n) for n in (1, 2, 3)])
    use_transcriber_in_handlers(monkeypatch, total_jobs=3)

    rc = main([], transcriber_factory=factory, queue_factory=queues)

    assert rc == 0
    assert queues.outcomes == ["done", "done", "done"]
    assert factory.calls == 1
    assert whisper.loads == 1
    assert whisper.calls == 3


def test_a_process_that_only_ingests_never_builds_the_transcriber(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = Factory(FakeWhisper())
    queues = FakeQueues([job("ingest", n) for n in (1, 2, 3)])
    use_transcriber_in_handlers(monkeypatch, total_jobs=3)

    rc = main([], transcriber_factory=factory, queue_factory=queues)

    assert rc == 0
    assert queues.outcomes == ["done", "done", "done"]
    assert factory.calls == 0
    assert "faster_whisper" not in sys.modules


@pytest.mark.parametrize(
    ("error", "error_class"),
    [
        (TransientNetworkError("hub unreachable"), "TRANSIENT_NETWORK"),
        (ToolFailureError("model file corrupt"), "TOOL_FAILURE"),
    ],
)
def test_a_factory_that_fails_records_the_job_and_is_tried_again_next_time(
    monkeypatch: pytest.MonkeyPatch, error: JobError, error_class: str
) -> None:
    whisper = FakeWhisper()
    factory = Factory(whisper, errors=[error])
    queues = FakeQueues([job("transcribe", 1), job("ingest", 2), job("transcribe", 3)])
    use_transcriber_in_handlers(monkeypatch, total_jobs=3)

    rc = main([], transcriber_factory=factory, queue_factory=queues)

    assert rc == 0
    assert queues.outcomes == [error_class, "done", "done"]
    assert factory.calls == 2
    assert whisper.loads == 1


@pytest.mark.parametrize(
    ("error", "error_class"),
    [
        (TransientNetworkError("hub unreachable"), "TRANSIENT_NETWORK"),
        (ToolFailureError("model file corrupt"), "TOOL_FAILURE"),
    ],
)
def test_a_model_load_that_fails_inside_the_transcriber_is_retried_on_the_next_job(
    monkeypatch: pytest.MonkeyPatch, error: JobError, error_class: str
) -> None:
    whisper = FakeWhisper(load_errors=[error])
    factory = Factory(whisper)
    queues = FakeQueues([job("transcribe", 1), job("ingest", 2), job("transcribe", 3)])
    use_transcriber_in_handlers(monkeypatch, total_jobs=3)

    rc = main([], transcriber_factory=factory, queue_factory=queues)

    assert rc == 0
    assert queues.outcomes == [error_class, "done", "done"]
    assert factory.calls == 1
    assert whisper.load_attempts == 2
    assert whisper.loads == 1


def test_lazy_transcriber_builds_nothing_until_first_use() -> None:
    factory = Factory(FakeWhisper())

    lazy = LazyTranscriber(factory)

    assert factory.calls == 0
    lazy.transcribe(AUDIO)
    lazy.transcribe(AUDIO)
    assert factory.calls == 1


def test_default_transcriber_factory_does_not_import_faster_whisper() -> None:
    already_imported = "faster_whisper" in sys.modules
    factory = default_transcriber_factory(make_settings(WHISPER_MODEL="tiny"))

    transcriber = LazyTranscriber(factory)
    built = factory()

    assert built is not None
    assert transcriber is not None
    if not already_imported:
        assert "faster_whisper" not in sys.modules


# ---------------------------------------------------------------------------
# build_worker
# ---------------------------------------------------------------------------


def test_build_worker_records_the_outcome_of_each_job_through_run_max_iterations() -> None:
    queues = FakeQueues([job("ingest", 1), job("transcribe", 2)])
    calls = Calls(stop_after=99)
    pool = QueuePool(queues)
    worker = build_worker(
        make_settings(),
        name="transcriber:h:1",
        handlers=calls.handlers(),
        pool=pool,
        queue=pool.open_main(),
        sleep=lambda seconds: None,
    )

    worker.run(max_iterations=2)

    assert calls.seen == [("ingest", 1), ("transcribe", 2)]
    assert queues.claimed_by == ["transcriber:h:1", "transcriber:h:1"]
    assert queues.outcomes == ["done", "done"]
    pool.close_all()
