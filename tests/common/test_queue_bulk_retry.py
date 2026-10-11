"""``PostgresQueue.retry_dead_matching`` against real Postgres (issue #113)."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from common.errors import ErrorClass
from common.queue import MAX_BULK_RETRY, PostgresQueue
from common.repo.jobs import UNSET
from tests.common.repo.conftest import head_dsn

pytestmark = pytest.mark.integration
__all__ = ["head_dsn"]


@pytest.fixture
def conn(head_dsn: str) -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(head_dsn, autocommit=True) as connection:
        yield connection


@pytest.fixture
def queue(conn: psycopg.Connection[Any]) -> PostgresQueue:
    return PostgresQueue(conn)


def seed(
    conn: psycopg.Connection[Any],
    *,
    video: str,
    kind: str = "ingest",
    state: str = "dead",
    error_class: str | None = "TOOL_FAILURE",
    dedupe_key: str = "default",
    attempts: int = 3,
    priority: int = 5,
) -> int:
    row = conn.execute(
        """
        INSERT INTO jobs (video_id, kind, dedupe_key, state, attempts, error_class,
                          last_error, priority, finished_at, locked_by, locked_at,
                          heartbeat_at, payload)
        VALUES (%s, %s, %s, %s, %s, %s, 'boom', %s,
                CASE WHEN %s IN ('dead', 'done') THEN now() END,
                'w1', now(), now(), '{"x": 1}')
        RETURNING id
        """,
        (video, kind, dedupe_key, state, attempts, error_class, priority, state),
    ).fetchone()
    assert row is not None
    return int(row[0])


def table(conn: psycopg.Connection[Any]) -> dict[int, dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM jobs ORDER BY id")
        return {row["id"]: row for row in cur.fetchall()}


def changed(before: dict[int, dict[str, Any]], after: dict[int, dict[str, Any]]) -> set[int]:
    assert before.keys() == after.keys()
    return {i for i in before if before[i] != after[i]}


# --- which jobs, and the transition -------------------------------------------


def test_a_seeded_mix_retries_exactly_the_matching_non_superseded_dead_rows(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    want_a = seed(conn, video="a", kind="ingest", error_class="TOOL_FAILURE")
    want_b = seed(conn, video="b", kind="ingest", error_class="TOOL_FAILURE")
    seed(conn, video="c", kind="ingest", error_class="BUG")  # other class
    seed(conn, video="d", kind="transcribe", error_class="TOOL_FAILURE")  # other kind
    seed(conn, video="e", kind="ingest", error_class=None)  # reaped, no class
    superseded = seed(conn, video="f", kind="ingest", error_class="TOOL_FAILURE")
    seed(conn, video="f", kind="ingest", state="pending", error_class=None)  # newer, same key
    for state in ("pending", "running", "done"):  # same kind and class, not dead
        seed(conn, video=f"s-{state}", kind="ingest", state=state, error_class="TOOL_FAILURE",
             dedupe_key=state)
    before = table(conn)

    result = queue.retry_dead_matching(kind="ingest", error_class="TOOL_FAILURE", limit=50)

    after = table(conn)
    assert result.job_ids == [want_b, want_a]
    assert changed(before, after) == {want_a, want_b}
    assert superseded not in result.job_ids
    assert result.has_more is False


def test_the_transition_is_retry_deads_and_keeps_everything_else(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    job_id = seed(conn, video="a", attempts=4, priority=7)
    before = table(conn)[job_id]

    queue.retry_dead_matching(limit=1)

    after = table(conn)[job_id]
    assert (after["state"], after["attempts"]) == ("pending", 0)
    assert after["finished_at"] is None
    assert after["locked_by"] is after["locked_at"] is after["heartbeat_at"] is None
    assert after["run_after"] >= before["run_after"]
    for column in ("priority", "payload", "dedupe_key", "created_at", "error_class", "last_error"):
        assert after[column] == before[column], column
    with PostgresQueue(conn).claim(["ingest"], worker="w2") as claimed:
        assert claimed is not None and claimed.id == job_id


def test_the_result_carries_the_previous_attempts_and_class(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    job_id = seed(conn, video="a", attempts=4, error_class="BUG")

    [job] = queue.retry_dead_matching(limit=5).jobs

    assert (job.job_id, job.video_id, job.kind, job.error_class, job.prev_attempts) == (
        job_id, "a", "ingest", "BUG", 4,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"kind": "transcribe"}, {"t_tool", "t_bug"}),
        ({"error_class": "BUG"}, {"i_bug", "t_bug"}),
        ({"error_class": None}, {"i_none"}),
        ({"kind": "ingest", "error_class": "BUG"}, {"i_bug"}),
        ({"kind": "analyze", "error_class": "BUG"}, set()),
        ({}, {"i_tool", "i_bug", "i_none", "t_tool", "t_bug"}),
    ],
)
def test_each_filter_and_their_combination(
    conn: psycopg.Connection[Any], queue: PostgresQueue, kwargs: dict[str, Any], expected: set[str]
) -> None:
    ids = {
        "i_tool": seed(conn, video="1", kind="ingest", error_class="TOOL_FAILURE"),
        "i_bug": seed(conn, video="2", kind="ingest", error_class="BUG"),
        "i_none": seed(conn, video="3", kind="ingest", error_class=None),
        "t_tool": seed(conn, video="4", kind="transcribe", error_class="TOOL_FAILURE"),
        "t_bug": seed(conn, video="5", kind="transcribe", error_class="BUG"),
    }

    result = queue.retry_dead_matching(limit=50, **kwargs)

    assert set(result.job_ids) == {ids[name] for name in expected}


def test_no_filters_retries_only_the_newest_limit_eligible_jobs(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    ids = [seed(conn, video=f"v{i}", kind=("ingest", "transcribe", "analyze")[i % 3]) for i in range(6)]

    result = queue.retry_dead_matching(limit=3)

    assert result.job_ids == ids[:2:-1]
    assert result.has_more is True
    assert {i for i, r in table(conn).items() if r["state"] == "pending"} == set(ids[3:])


def test_superseded_dead_jobs_are_not_retried_and_do_not_count_against_the_limit(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    older = seed(conn, video="a")
    seed(conn, video="a", state="done", error_class=None)  # newer job of the same key
    plain = [seed(conn, video=f"p{i}") for i in range(2)]
    # the superseded row has the *lowest* id; if it counted it would eat the limit
    result = queue.retry_dead_matching(limit=2)

    assert result.job_ids == plain[::-1]
    assert result.has_more is False
    assert table(conn)[older]["state"] == "dead"


def test_has_more_boundaries(conn: psycopg.Connection[Any], queue: PostgresQueue) -> None:
    empty = queue.retry_dead_matching(limit=2)
    assert (empty.job_ids, empty.has_more) == ([], False)

    first = seed(conn, video="a")
    second = seed(conn, video="b")
    exact = queue.retry_dead_matching(limit=2)
    assert (exact.job_ids, exact.has_more) == ([second, first], False)

    third = seed(conn, video="c")
    fourth = seed(conn, video="d")
    fifth = seed(conn, video="e")
    # mark the earlier two dead again so three are eligible
    conn.execute("UPDATE jobs SET state='dead', finished_at=now() WHERE id IN (%s, %s)", (first, second))
    over = queue.retry_dead_matching(limit=4)
    assert over.job_ids == [fifth, fourth, third, second]
    assert over.has_more is True

    rest = queue.retry_dead_matching(limit=4)
    assert (rest.job_ids, rest.has_more) == ([first], False)


def test_dry_run_reports_what_a_real_call_would_and_writes_nothing(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    ids = [seed(conn, video=f"v{i}") for i in range(4)]
    before = table(conn)

    dry = queue.retry_dead_matching(limit=3, dry_run=True)

    assert table(conn) == before
    real = queue.retry_dead_matching(limit=3)
    assert dry.job_ids == real.job_ids == ids[:0:-1]
    assert dry.has_more is real.has_more is True
    assert [j.prev_attempts for j in dry.jobs] == [j.prev_attempts for j in real.jobs]


def test_a_dry_run_uses_the_same_filters(conn: psycopg.Connection[Any], queue: PostgresQueue) -> None:
    keep = seed(conn, video="a", kind="ingest", error_class="BUG")
    seed(conn, video="b", kind="transcribe", error_class="BUG")
    seed(conn, video="c", kind="ingest", error_class="TOOL_FAILURE")

    assert queue.retry_dead_matching(kind="ingest", error_class="BUG", limit=9, dry_run=True).job_ids == [keep]


# --- bounds and crafted input --------------------------------------------------


class _NoSql:
    """A connection that fails the test if any SQL is attempted."""

    def transaction(self) -> Any:
        raise AssertionError("SQL must not run for invalid input")

    cursor = execute = transaction


@pytest.mark.parametrize("limit", [0, -1, MAX_BULK_RETRY + 1, 10**9, True, False, "5", 1.5, None])
def test_an_out_of_range_limit_is_refused_before_any_sql(limit: Any) -> None:
    with pytest.raises(ValueError, match="limit"):
        PostgresQueue(_NoSql()).retry_dead_matching(limit=limit)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [1, MAX_BULK_RETRY])
def test_the_limit_boundaries_are_accepted(queue: PostgresQueue, limit: int) -> None:
    assert queue.retry_dead_matching(limit=limit).job_ids == []


def test_the_cap_holds_with_many_more_eligible_jobs(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    conn.execute(
        """
        INSERT INTO jobs (video_id, kind, dedupe_key, state, attempts, error_class, finished_at)
        SELECT 'bulk' || g, 'ingest', 'default', 'dead', 3, 'TOOL_FAILURE', now()
        FROM generate_series(1, %s) g
        """,
        (MAX_BULK_RETRY * 3,),
    )

    result = queue.retry_dead_matching(limit=MAX_BULK_RETRY)

    assert len(result.job_ids) == MAX_BULK_RETRY and result.has_more is True
    pending = conn.execute("SELECT count(*) FROM jobs WHERE state = 'pending'").fetchone()
    assert pending == (MAX_BULK_RETRY,)


CRAFTED = [
    "ingest' OR '1'='1",
    "ingest'; UPDATE jobs SET state='pending'; --",
    "INGEST",
    "ingest ",
    "",
    "%",
    "notify",
]


@pytest.mark.parametrize("value", CRAFTED)
def test_a_crafted_kind_is_refused_and_nothing_changes(
    conn: psycopg.Connection[Any], queue: PostgresQueue, value: str
) -> None:
    seed(conn, video="a")
    before = table(conn)

    with pytest.raises(ValueError, match="kind"):
        queue.retry_dead_matching(kind=value, limit=5)

    assert table(conn) == before


@pytest.mark.parametrize("value", [*CRAFTED, "tool_failure", "none", "UNSET"])
def test_a_crafted_error_class_is_refused_and_nothing_changes(
    conn: psycopg.Connection[Any], queue: PostgresQueue, value: str
) -> None:
    seed(conn, video="a")
    before = table(conn)

    with pytest.raises(ValueError, match="error class"):
        queue.retry_dead_matching(error_class=value, limit=5)

    assert table(conn) == before


def test_every_real_error_class_is_accepted(queue: PostgresQueue) -> None:
    for cls in ErrorClass:
        assert queue.retry_dead_matching(error_class=cls.value, limit=1).job_ids == []
    assert queue.retry_dead_matching(error_class=UNSET, limit=1).job_ids == []


# --- failure, atomicity and concurrency ----------------------------------------


def wait_until_blocked(dsn_conn: psycopg.Connection[Any], deadline_sec: float = 15) -> None:
    """Wait on the exact condition: some backend is waiting on a lock."""
    deadline = time.monotonic() + deadline_sec
    gate = threading.Event()
    while time.monotonic() < deadline:
        row = dsn_conn.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock'"
        ).fetchone()
        if row and row[0]:
            return
        gate.wait(0.01)
    raise AssertionError("no backend became blocked on a lock")


def test_a_unique_violation_retries_nothing_and_the_enqueue_wins(
    head_dsn: str, conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    collide = seed(conn, video="a", dedupe_key="k")
    other = seed(conn, video="b", dedupe_key="k")
    before = table(conn)
    outcome: list[BaseException | None] = []

    with psycopg.connect(head_dsn) as enqueuer:
        # An uncommitted pending job for key (a, ingest, k): invisible to the
        # NOT EXISTS check, but the unique index makes the retry wait for it.
        enqueuer.execute(
            "INSERT INTO jobs (video_id, kind, dedupe_key, state) VALUES ('a', 'ingest', 'k', 'pending')"
        )

        def run() -> None:
            with psycopg.connect(head_dsn, autocommit=True) as c:
                try:
                    PostgresQueue(c).retry_dead_matching(limit=10)
                    outcome.append(None)
                except BaseException as exc:  # noqa: BLE001 - asserted below
                    outcome.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        wait_until_blocked(conn)
        enqueuer.commit()
        thread.join(timeout=20)

    assert not thread.is_alive()
    assert isinstance(outcome[0], psycopg.errors.UniqueViolation)
    after = table(conn)
    assert after[other] == before[other] and after[collide] == before[collide]  # all or nothing
    active = conn.execute(
        "SELECT count(*) FROM jobs WHERE video_id='a' AND kind='ingest' AND dedupe_key='k' "
        "AND state IN ('pending','running')"
    ).fetchone()
    assert active == (1,)


def test_two_bulk_retries_over_one_set_retry_each_job_exactly_once(
    head_dsn: str, conn: psycopg.Connection[Any]
) -> None:
    ids = {seed(conn, video=f"v{i}") for i in range(4)}
    with psycopg.connect(head_dsn) as first, psycopg.connect(head_dsn, autocommit=True) as second:
        with first.transaction():  # holds its row locks until it commits
            a = PostgresQueue(first).retry_dead_matching(limit=50)
            b = PostgresQueue(second).retry_dead_matching(limit=50)  # skips locked rows
        assert set(a.job_ids) | set(b.job_ids) == ids
        assert set(a.job_ids).isdisjoint(b.job_ids)
        assert b.job_ids == [] and b.has_more is False

    rows = table(conn)
    assert all(rows[i]["state"] == "pending" and rows[i]["attempts"] == 0 for i in ids)


def test_a_bulk_retry_racing_a_single_retry_gives_the_job_exactly_one_winner(
    head_dsn: str, conn: psycopg.Connection[Any]
) -> None:
    job = seed(conn, video="a")
    outcome: list[Any] = []
    with psycopg.connect(head_dsn) as bulk_conn:
        with bulk_conn.transaction():
            bulk = PostgresQueue(bulk_conn).retry_dead_matching(limit=5)

            def single() -> None:
                with psycopg.connect(head_dsn, autocommit=True) as c:
                    outcome.append(PostgresQueue(c).retry_dead(job))

            thread = threading.Thread(target=single)
            thread.start()
            wait_until_blocked(conn)
        thread.join(timeout=20)

    assert not thread.is_alive()
    assert bulk.job_ids == [job]
    assert outcome[0].status == "not_dead" and outcome[0].state == "pending"


def test_the_connection_is_left_usable_after_a_failure(
    conn: psycopg.Connection[Any], queue: PostgresQueue
) -> None:
    with pytest.raises(ValueError):
        queue.retry_dead_matching(limit=0)
    seed(conn, video="a")

    assert len(queue.retry_dead_matching(limit=1).job_ids) == 1


