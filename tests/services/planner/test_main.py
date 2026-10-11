"""Tests for the planner entrypoint wiring (issue #38): registry, lock keys, CLI."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from psycopg.pq import TransactionStatus

from common.config import Settings, get_settings
from services.planner import main as planner_main
from services.planner.reconcile import ReconcileResult
from services.planner.retention import RetentionReport
from services.planner.scheduler import Scheduler, Task

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_URL = "postgresql://u:p@h/db"
TASK_NAMES = [
    "reap",
    "auto_caption_fallback",
    "audio_retention",
    "stale_analyze",
    "reanalysis_sweep",
    "poll_channels",
    "reconcile_channels",
    "channel_backfill",
    "dead_letter_alerts",
]


def _settings(**values: Any) -> Settings:
    # ``Any`` lets a plain str stand in for the SecretStr DATABASE_URL.
    return Settings(**{"DATABASE_URL": DB_URL, **values})


def _no_connect() -> Any:
    raise AssertionError("building the registry must not open a connection")


def test_registry_order_is_reap_retention_sweep_poll() -> None:
    tasks = planner_main.build_registry(_settings(), _no_connect)

    assert [t.name for t in tasks] == TASK_NAMES


def test_reap_uses_its_own_interval_and_the_others_use_the_poll_interval() -> None:
    settings = _settings(REAP_INTERVAL_SEC=7, POLL_INTERVAL_SEC=99)

    tasks = planner_main.build_registry(settings, _no_connect)

    assert {t.name: t.interval_sec for t in tasks} == {
        "reap": 7,
        "auto_caption_fallback": 99,
        "audio_retention": 99,
        "stale_analyze": 99,
        "reanalysis_sweep": 99,
        "poll_channels": 99,
        "reconcile_channels": 21600,
        "channel_backfill": 60,
        "dead_letter_alerts": 7,
    }


def test_default_intervals_are_one_minute_and_one_hour() -> None:
    tasks = planner_main.build_registry(_settings(), _no_connect)

    assert {t.name: t.interval_sec for t in tasks} == {
        "reap": 60,
        "auto_caption_fallback": 3600,
        "audio_retention": 3600,
        "stale_analyze": 3600,
        "reanalysis_sweep": 3600,
        "poll_channels": 3600,
        "reconcile_channels": 21600,
        "channel_backfill": 60,
        "dead_letter_alerts": 60,
    }


def test_alert_task_without_a_channel_returns_without_opening_a_connection() -> None:
    tasks = {t.name: t for t in planner_main.build_registry(_settings(), _no_connect)}

    assert tasks["dead_letter_alerts"].fn() == 0  # _no_connect would raise


def test_alert_task_with_a_channel_opens_a_connection() -> None:
    settings = _settings(ALERT_WEBHOOK_URL="https://hooks.example.test/s3cr3t")
    tasks = {t.name: t for t in planner_main.build_registry(settings, _no_connect)}

    with pytest.raises(AssertionError, match="must not open a connection"):
        tasks["dead_letter_alerts"].fn()


class _FakeConn:
    info = SimpleNamespace(transaction_status=TransactionStatus.IDLE)  # run_on_fresh_connection

    def close(self) -> None:
        pass


def test_audio_retention_passes_the_settings_and_counts_evicted_for_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_retention(conn: Any, audio_dir: Any, **kwargs: Any) -> RetentionReport:
        seen["audio_dir"] = audio_dir
        seen.update(kwargs)
        return RetentionReport(
            expired_deleted=1, evicted=2, orphans_deleted=4, bytes_freed=0, bytes_retained=0,
            skipped_running=0, errors=0, evicted_for_space=8, free_bytes=None,
        )

    monkeypatch.setattr(planner_main, "enforce_audio_retention", fake_retention)
    settings = _settings(AUDIO_MAX_GB=3, AUDIO_MIN_FREE_GB=1.5, AUDIO_MIN_FREE_PCT=7)
    tasks = {t.name: t for t in planner_main.build_registry(settings, lambda: _FakeConn())}  # type: ignore[arg-type,return-value]

    count = tasks["audio_retention"].fn()

    assert seen["min_free_gb"] == 1.5
    assert seen["min_free_pct"] == 7
    assert seen["max_bytes"] == 3_000_000_000
    assert count == 1 + 2 + 4 + 8


def test_every_task_has_one_fixed_distinct_integer_lock_key() -> None:
    keys = planner_main.LOCK_KEYS

    assert set(keys) == set(TASK_NAMES)
    assert len(set(keys.values())) == len(TASK_NAMES)
    for namespace, task_id in keys.values():
        assert isinstance(namespace, int) and isinstance(task_id, int)
        assert -(2**31) <= namespace < 2**31
        assert -(2**31) <= task_id < 2**31


def test_auto_caption_fallback_runs_right_after_reap_with_its_own_fixed_lock_key() -> None:
    names = [t.name for t in planner_main.build_registry(_settings(), _no_connect)]

    assert names.index("auto_caption_fallback") == names.index("reap") + 1
    assert planner_main.LOCK_KEYS["auto_caption_fallback"] == (planner_main.LOCK_NAMESPACE, 6)


def test_stale_analyze_runs_before_the_sweep_with_its_own_fixed_lock_key() -> None:
    names = [t.name for t in planner_main.build_registry(_settings(), _no_connect)]

    assert names.index("stale_analyze") == names.index("reanalysis_sweep") - 1
    assert planner_main.LOCK_KEYS["stale_analyze"] == (planner_main.LOCK_NAMESPACE, 8)


def test_stale_analyze_task_resolves_with_the_configured_prompt_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def fake_resolve(queue: Any, settings: Settings) -> int:
        seen.append(settings.PROMPT_VERSION)
        return 5

    monkeypatch.setattr(planner_main, "resolve_stale_analyze_jobs", fake_resolve)
    tasks = {
        t.name: t
        for t in planner_main.build_registry(_settings(PROMPT_VERSION="v9"), lambda: cast(Any, _Conn()))
    }

    assert tasks["stale_analyze"].fn() == 5
    assert seen == ["v9"]


def test_reconcile_runs_after_poll_on_its_own_interval_with_lock_key_7() -> None:
    names = [t.name for t in planner_main.build_registry(_settings(), _no_connect)]

    assert names.index("reconcile_channels") == names.index("poll_channels") + 1
    assert planner_main.LOCK_KEYS["reconcile_channels"] == (planner_main.LOCK_NAMESPACE, 7)
    tasks = {
        t.name: t.interval_sec
        for t in planner_main.build_registry(_settings(RECONCILE_INTERVAL_SEC=123), _no_connect)
    }
    assert tasks["reconcile_channels"] == 123


class _Conn:
    info = SimpleNamespace(transaction_status=TransactionStatus.IDLE)  # run_on_fresh_connection

    def close(self) -> None:
        pass


def test_reconcile_builds_the_catalog_with_the_list_timeout_and_keeps_its_cursor_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[float] = []
    cursors: list[str | None] = []
    results = iter([("UCb", 2), ("UCz", 0), (None, 1)])

    class FakeCatalog:
        def __init__(self, *, timeout: float) -> None:
            built.append(timeout)

    def fake_reconcile(conn: Any, catalog: Any, queue: Any, *, clock: Any, start_after: Any) -> Any:
        cursors.append(start_after)
        assert isinstance(catalog, FakeCatalog)
        assert clock is time.monotonic
        resume, enqueued = next(results)
        return ReconcileResult(1, 0, 0, 0, 0, enqueued, resume)

    monkeypatch.setattr(planner_main, "YouTubeCatalog", FakeCatalog)
    monkeypatch.setattr(planner_main, "reconcile_channels", fake_reconcile)
    tasks = {
        t.name: t for t in planner_main.build_registry(_settings(), lambda: cast(Any, _Conn()))
    }

    returned = [tasks["reconcile_channels"].fn() for _ in range(3)]

    assert built == [60]
    assert cursors == [None, "UCb", "UCz"]  # each run starts where the last one stopped
    assert returned == [2, 0, 1]  # the task returns jobs_enqueued


def test_the_module_documents_the_single_replica_constraint() -> None:
    doc = planner_main.__doc__ or ""

    assert "exactly one" in doc.lower()
    assert "defence in depth" in doc.lower()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class StubLocks:
    def try_lock(self, name: str) -> bool:
        return True

    def release(self, name: str) -> None:
        pass

    def reset(self) -> None:
        pass


def _factory(tmp_path: Path, tasks: list[Task]) -> Any:
    def make(settings: Settings) -> Scheduler:
        return Scheduler(
            tasks,
            StubLocks(),
            liveness_path=tmp_path / "hb",
            install_signal_handlers=False,
            sleep=lambda _s: None,
        )

    return make


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setattr(planner_main, "configure_logging", lambda settings: None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_once_exits_0_when_every_task_is_ok(tmp_path: Path) -> None:
    calls: list[str] = []

    def a() -> int:
        calls.append("a")
        return 1

    tasks = [Task("a", 60, a), Task("b", 60, lambda: None)]

    code = planner_main.main(["--once"], scheduler_factory=_factory(tmp_path, tasks))

    assert code == 0
    assert calls == ["a"]


def test_once_exits_1_when_a_task_raised(tmp_path: Path) -> None:
    def boom() -> int:
        raise RuntimeError("x")

    ran: list[str] = []

    def b() -> None:
        ran.append("b")

    tasks = [Task("a", 60, boom), Task("b", 60, b)]

    code = planner_main.main(["--once"], scheduler_factory=_factory(tmp_path, tasks))

    assert code == 1
    assert ran == ["b"]  # the failure did not stop the rest


def test_without_once_the_loop_runs_until_shutdown_and_exits_0(tmp_path: Path) -> None:
    holder: dict[str, Scheduler] = {}

    def stop_after_first_run() -> None:
        holder["s"].request_shutdown()

    def make(settings: Settings) -> Scheduler:
        holder["s"] = _factory(tmp_path, [Task("a", 60, stop_after_first_run)])(settings)
        return holder["s"]

    assert planner_main.main([], scheduler_factory=make) == 0


def test_an_unknown_argument_exits_2_with_usage(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        planner_main.main(["--bogus"])

    assert exc_info.value.code == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_invalid_settings_exit_nonzero_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("REAP_INTERVAL_SEC", "0")

    code = planner_main.main(["--once"])

    assert code != 0
    assert "REAP_INTERVAL_SEC" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Subprocess: a real SIGTERM
# ---------------------------------------------------------------------------


def test_invalid_environment_makes_the_process_exit_nonzero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "services.planner.main", "--once"],
        cwd=REPO_ROOT,
        env={"DATABASE_URL": DB_URL, "POLL_INTERVAL_SEC": "abc", "PATH": "/usr/bin"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "POLL_INTERVAL_SEC" in result.stderr


def test_real_sigterm_stops_an_idle_planner(tmp_path: Path) -> None:
    script = tmp_path / "run_planner.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(REPO_ROOT)!r})
            from pathlib import Path
            from services.planner.scheduler import Scheduler, Task

            class Locks:
                def try_lock(self, name): return True
                def release(self, name): pass
                def reset(self): pass

            def first_run():
                print("READY", flush=True)  # printed once run() has installed its handlers

            sched = Scheduler(
                [Task("a", 3600, first_run)], Locks(),
                liveness_path=Path({str(tmp_path / "hb")!r}),
            )
            sched.run()
            print("DONE", flush=True)
            """
        )
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "READY"

        proc.terminate()  # SIGTERM
        returncode = proc.wait(timeout=10)
        remaining = proc.stdout.read()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    assert returncode == 0, remaining
    assert "DONE" in remaining


def test_the_auto_caption_fallback_task_invokes_the_sweep_and_returns_its_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    def fake_sweep(conn: Any, queue: Any, *, settings: Settings, **kwargs: Any) -> int:
        calls.append((conn, queue, settings, kwargs))
        return 7

    monkeypatch.setattr(planner_main, "run_auto_caption_fallback_sweep", fake_sweep)
    conn = _FakeConn()
    settings = _settings(AUTO_CAPTION_FALLBACK=True)
    tasks = {t.name: t for t in planner_main.build_registry(settings, lambda: conn)}  # type: ignore[arg-type,return-value]

    assert tasks["auto_caption_fallback"].fn() == 7  # the count is propagated, not discarded

    assert len(calls) == 1
    assert calls[0][0] is conn
    assert calls[0][2] is settings  # the real flag reaches the sweep, not a copy


def _dead_transcribe(conn: Any) -> None:
    conn.execute("INSERT INTO videos (video_id) VALUES ('deadvideo01')")
    conn.execute(
        "INSERT INTO jobs (video_id, kind, dedupe_key, state, error_class, finished_at) "
        "VALUES ('deadvideo01', 'transcribe', 'k', 'dead', 'TOOL_FAILURE', now())"
    )
    conn.commit()


@pytest.mark.integration
@pytest.mark.parametrize(("enabled", "expected"), [(True, 1), (False, 0)])
def test_the_auto_caption_fallback_task_enqueues_only_when_the_flag_is_on(
    conn: Any, head_dsn: str, enabled: bool, expected: int
) -> None:
    import psycopg

    _dead_transcribe(conn)
    settings = _settings(DATABASE_URL=head_dsn, AUTO_CAPTION_FALLBACK=enabled)
    tasks = {
        t.name: t for t in planner_main.build_registry(settings, lambda: psycopg.connect(head_dsn))
    }

    assert tasks["auto_caption_fallback"].fn() == expected

    # Read from another connection: the task's own is closed without a commit.
    with psycopg.connect(head_dsn) as fresh:
        assert fresh.execute("SELECT count(*) FROM jobs WHERE kind = 'ingest'").fetchone() == (
            expected,
        )
