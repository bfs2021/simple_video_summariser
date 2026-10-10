"""Spend cap and dry run of the re-analysis sweep (issue #99). Needs Postgres."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, cast

import psycopg
import pytest

from common.config import Settings, get_settings
from common.queue import analyze_dedupe_key
from services.planner import main as planner_main
from services.planner.sweep import plan_reanalysis_sweep, run_reanalysis_sweep
from tests.services.planner.test_sweep import VERSION, _queue, _ready

pytestmark = pytest.mark.integration

KEY = analyze_dedupe_key(VERSION, "anthropic")
OLLAMA_KEY = analyze_dedupe_key(VERSION, "ollama")


def _sweep(
    conn: psycopg.Connection[Any], *, cap: int, summarizer: str = "anthropic", **kwargs: Any
) -> int:
    return run_reanalysis_sweep(
        conn,
        _queue(conn),
        prompt_version=VERSION,
        summarizer_name=summarizer,
        max_jobs_anthropic=cap,
        **kwargs,
    )


def _candidates(conn: psycopg.Connection[Any], n: int) -> None:
    for i in range(n):
        _ready(conn, f"c{i:03d}", age_days=i)
    conn.commit()


def _already(
    conn: psycopg.Connection[Any],
    n: int,
    *,
    priority: int = -5,
    state: str = "done",
    key: str = KEY,
) -> None:
    """``n`` jobs at this key, on videos that are not candidates (no transcript)."""
    for i in range(n):
        vid = f"x{priority}{state}{i:03d}"
        conn.execute("INSERT INTO videos (video_id) VALUES (%s)", (vid,))
        conn.execute(
            "INSERT INTO jobs (video_id, kind, dedupe_key, priority, state) "
            "VALUES (%s, 'analyze', %s, %s, %s)",
            (vid, key, priority, state),
        )
    conn.commit()


def _count_jobs(conn: psycopg.Connection[Any]) -> int:
    row = conn.execute("SELECT count(*) FROM jobs").fetchone()
    assert row is not None
    return int(row[0])


@pytest.mark.parametrize(
    ("cap", "already", "candidates", "expected"),
    [
        (10, 0, 25, 10),  # plenty of candidates: the cap binds
        (10, 9, 25, 1),  # one below
        (10, 10, 25, 0),  # exactly at the cap
        (10, 12, 25, 0),  # cap lowered after the fact: never negative
        (0, 0, 5, 0),  # zero enqueues nothing
        (10, 0, 3, 3),  # fewer candidates than room
    ],
)
def test_a_run_enqueues_at_most_the_remaining_room(
    conn: psycopg.Connection[Any], cap: int, already: int, candidates: int, expected: int
) -> None:
    _already(conn, already)
    _candidates(conn, candidates)

    assert _sweep(conn, cap=cap) == expected


def test_the_per_run_limit_still_applies_below_the_cap(conn: psycopg.Connection[Any]) -> None:
    _candidates(conn, 6)

    assert _sweep(conn, cap=10, limit=4) == 4


def test_a_second_run_after_the_cap_is_reached_enqueues_nothing_even_in_a_new_connection(
    conn: psycopg.Connection[Any], head_dsn: str
) -> None:
    _candidates(conn, 5)
    assert _sweep(conn, cap=3) == 3
    conn.close()

    # A restarted planner holds no memory of the first run.
    with psycopg.connect(head_dsn) as fresh:
        assert _sweep(fresh, cap=3) == 0
        fresh.commit()


@pytest.mark.parametrize("state", ["pending", "running", "done", "dead"])
def test_sweep_jobs_in_any_state_count_toward_the_cap(
    conn: psycopg.Connection[Any], state: str
) -> None:
    _already(conn, 2, state=state)
    _candidates(conn, 5)

    assert _sweep(conn, cap=3) == 1


@pytest.mark.parametrize("priority", [0, 10])
def test_jobs_at_other_priorities_neither_count_nor_are_blocked(
    conn: psycopg.Connection[Any], priority: int
) -> None:
    _already(conn, 5, priority=priority)
    _candidates(conn, 5)

    assert _sweep(conn, cap=3) == 3  # the five other-priority jobs are not counted


def test_the_cap_does_not_block_the_queue_from_accepting_other_analyze_jobs(
    conn: psycopg.Connection[Any],
) -> None:
    _already(conn, 3)
    _ready(conn, "fresh")
    conn.commit()
    assert _sweep(conn, cap=3) == 0

    assert _queue(conn).enqueue("analyze", "fresh", dedupe_key=KEY, priority=10) is not None


def test_ollama_ignores_the_cap(conn: psycopg.Connection[Any]) -> None:
    _already(conn, 5, key=OLLAMA_KEY)
    _candidates(conn, 4)

    assert _sweep(conn, cap=0, summarizer="ollama") == 4


def test_a_deduplicated_enqueue_is_not_counted(conn: psycopg.Connection[Any]) -> None:
    _candidates(conn, 3)

    class Losing:
        calls = 0

        def enqueue(self, *args: Any, **kwargs: Any) -> int | None:
            self.calls += 1
            return None

    queue = Losing()
    result = run_reanalysis_sweep(
        conn,
        cast(Any, queue),
        prompt_version=VERSION,
        summarizer_name="anthropic",
        max_jobs_anthropic=10,
    )

    assert (result, queue.calls) == (0, 3)


def _records(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records if r.name == "services.planner.sweep" and r.levelno == level
    ]


def test_one_warning_names_the_cap_and_what_was_left_when_the_cap_stops_the_run(
    conn: psycopg.Connection[Any], caplog: pytest.LogCaptureFixture
) -> None:
    _already(conn, 1)
    _candidates(conn, 9)

    with caplog.at_level(logging.INFO, logger="services.planner.sweep"):
        _sweep(conn, cap=4)

    (warning,) = _records(caplog, logging.WARNING)
    assert {
        k: warning.__dict__[k]
        for k in ("prompt_version", "summarizer", "cap", "already_enqueued", "unenqueued")
    } == {
        "prompt_version": VERSION,
        "summarizer": "anthropic",
        "cap": 4,
        "already_enqueued": 1,
        "unenqueued": 6,
    }


@pytest.mark.parametrize(("cap", "candidates"), [(10, 3), (3, 3), (10, 0)])
def test_no_warning_when_the_cap_is_not_what_stopped_the_run(
    conn: psycopg.Connection[Any],
    caplog: pytest.LogCaptureFixture,
    cap: int,
    candidates: int,
) -> None:
    _candidates(conn, candidates)

    with caplog.at_level(logging.INFO, logger="services.planner.sweep"):
        _sweep(conn, cap=cap)

    assert _records(caplog, logging.WARNING) == []


def test_no_warning_when_the_per_run_limit_stopped_it_before_the_cap(
    conn: psycopg.Connection[Any], caplog: pytest.LogCaptureFixture
) -> None:
    _candidates(conn, 6)

    with caplog.at_level(logging.INFO, logger="services.planner.sweep"):
        _sweep(conn, cap=10, limit=2)

    assert _records(caplog, logging.WARNING) == []


def test_the_info_line_carries_cap_and_already_enqueued(
    conn: psycopg.Connection[Any], caplog: pytest.LogCaptureFixture
) -> None:
    _already(conn, 2)
    _candidates(conn, 1)

    with caplog.at_level(logging.INFO, logger="services.planner.sweep"):
        _sweep(conn, cap=5)
        _sweep(conn, cap=5, summarizer="ollama")

    anthropic, ollama = _records(caplog, logging.INFO)
    assert (anthropic.__dict__["cap"], anthropic.__dict__["already_enqueued"]) == (5, 2)
    assert (ollama.__dict__["cap"], ollama.__dict__["already_enqueued"]) == (None, 0)


# --- planning / dry run --------------------------------------------------------------------


def test_the_plan_counts_all_candidates_not_just_the_per_run_limit(
    conn: psycopg.Connection[Any],
) -> None:
    _already(conn, 1)
    _candidates(conn, 12)

    plan = plan_reanalysis_sweep(
        conn, prompt_version=VERSION, summarizer_name="anthropic", limit=2, max_jobs_anthropic=5
    )

    assert (
        plan.candidates,
        plan.already_enqueued,
        plan.cap,
        plan.would_enqueue_this_run,
        plan.would_remain_after_cap,
    ) == (12, 1, 5, 2, 8)


def test_the_plan_for_ollama_has_no_cap_and_nothing_remains(
    conn: psycopg.Connection[Any],
) -> None:
    _candidates(conn, 3)

    plan = plan_reanalysis_sweep(
        conn, prompt_version=VERSION, summarizer_name="ollama", max_jobs_anthropic=0
    )

    assert (plan.cap, plan.would_enqueue_this_run, plan.would_remain_after_cap) == (None, 3, 0)


@pytest.mark.parametrize(("cap", "already"), [(10, 0), (10, 9), (10, 10), (0, 0), (50, 2)])
def test_the_dry_run_prediction_equals_what_the_live_run_enqueues(
    conn: psycopg.Connection[Any], cap: int, already: int
) -> None:
    _already(conn, already)
    _candidates(conn, 20)
    before = _count_jobs(conn)

    plan = plan_reanalysis_sweep(
        conn, prompt_version=VERSION, summarizer_name="anthropic", max_jobs_anthropic=cap
    )
    assert _count_jobs(conn) == before  # planning wrote nothing

    assert _sweep(conn, cap=cap) == plan.would_enqueue_this_run


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, head_dsn: str) -> Iterator[None]:
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", head_dsn)
    monkeypatch.setenv("SUMMARIZER", "anthropic")
    monkeypatch.setenv("PROMPT_VERSION", VERSION)
    monkeypatch.setenv("SWEEP_MAX_JOBS_ANTHROPIC", "5")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _forbidden_factory(settings: Settings) -> Any:
    raise AssertionError("the dry run must not build a scheduler (no tasks, no advisory lock)")


def test_sweep_dry_run_prints_one_summary_line_logs_it_and_changes_no_row(
    conn: psycopg.Connection[Any],
    cli_env: None,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _already(conn, 2)
    _candidates(conn, 7)
    snapshot = conn.execute("SELECT * FROM jobs ORDER BY id").fetchall()
    conn.commit()

    with caplog.at_level(logging.INFO, logger="services.planner.sweep"):
        code = planner_main.main(["--sweep-dry-run"], scheduler_factory=_forbidden_factory)

    out = capsys.readouterr().out
    assert code == 0
    assert out.splitlines() == [
        (
            f"prompt_version={VERSION} summarizer=anthropic candidates=7 already_enqueued=2 "
            "cap=5 would_enqueue_this_run=3 would_remain_after_cap=4"
        )
    ]
    (event,) = [r for r in caplog.records if r.getMessage() == "planner.reanalysis_sweep.dry_run"]
    assert {
        k: event.__dict__[k]
        for k in (
            "prompt_version",
            "summarizer",
            "candidates",
            "already_enqueued",
            "cap",
            "would_enqueue_this_run",
            "would_remain_after_cap",
        )
    } == {
        "prompt_version": VERSION,
        "summarizer": "anthropic",
        "candidates": 7,
        "already_enqueued": 2,
        "cap": 5,
        "would_enqueue_this_run": 3,
        "would_remain_after_cap": 4,
    }
    assert conn.execute("SELECT * FROM jobs ORDER BY id").fetchall() == snapshot


def test_sweep_dry_run_says_cap_none_for_ollama(
    conn: psycopg.Connection[Any],
    cli_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SUMMARIZER", "ollama")
    get_settings.cache_clear()
    _candidates(conn, 2)

    assert planner_main.main(["--sweep-dry-run"]) == 0

    assert "cap=none would_enqueue_this_run=2 would_remain_after_cap=0" in capsys.readouterr().out


def test_sweep_dry_run_cannot_be_combined_with_once(
    cli_env: None, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        planner_main.main(["--once", "--sweep-dry-run"])

    assert exc_info.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_the_plan_with_a_cap_lowered_below_what_was_enqueued_leaves_every_candidate(
    conn: psycopg.Connection[Any],
) -> None:
    _already(conn, 12)
    _candidates(conn, 5)

    plan = plan_reanalysis_sweep(
        conn, prompt_version=VERSION, summarizer_name="anthropic", max_jobs_anthropic=10
    )

    assert (plan.would_enqueue_this_run, plan.would_remain_after_cap) == (0, 5)
