"""Adversarial concurrency and crash suite for ``common.queue`` (issue #12).

#11's own tests (``tests/common/test_queue.py``) cover ``PostgresQueue``'s
single-worker behaviour, one acceptance-criterion bullet at a time, against a
single connection. This module is the adversarial half named by architecture.md
§13 ("Queue semantics | Real Postgres (testcontainers); concurrent claim,
dedupe (C1), reap (C2)"): multiple real connections racing each other, a real
subprocess killed with ``SIGKILL``, and the split-brain sequence where a
reaped worker's stale handler finishes after its job has already been
reclaimed.

Kept deliberately self-contained (its own fixtures, duplicated in spirit from
#11's, per #12's constraint to stay inside this one file plus helpers) so it
does not depend on #11's test module while #11 is in QA.

Determinism, per ``_docs/testing-guidelines.md`` and #12's own constraints:

- No ``sleep`` anywhere. Staleness is written directly to ``heartbeat_at``.
- Races start every thread at once with ``threading.Barrier``, not timing
  luck.
- Every "must not block" assertion is backed by a bounded ``join``/``wait``
  timeout, so a real hang fails the test instead of hanging the suite.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from psycopg.rows import dict_row

from common.config import Settings, get_settings
from common.errors import PermanentSourceError, TransientNetworkError
from common.queue import Job, PostgresQueue

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# fixtures (self-contained; see module docstring)
# ---------------------------------------------------------------------------


def _config_for(dsn: str, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("DATABASE_URL", dsn)
    get_settings.cache_clear()
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    return config


@pytest.fixture
def head_dsn(postgres_dsn: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """A DSN for a database migrated all the way to head."""
    command.upgrade(_config_for(postgres_dsn, monkeypatch), "head")
    return postgres_dsn


@pytest.fixture
def conn(head_dsn: str) -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(head_dsn, autocommit=True) as connection:
        yield connection


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"DATABASE_URL": "postgresql://u:p@h/db"}
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


@pytest.fixture
def queue(conn: psycopg.Connection[Any]) -> PostgresQueue:
    return PostgresQueue(conn, settings=_settings(), rng=random.Random(1234))


def _connect(dsn: str) -> psycopg.Connection[Any]:
    return psycopg.connect(dsn, autocommit=True)


def _row(conn: psycopg.Connection[Any], job_id: int) -> dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
        row = cur.fetchone()
    assert row is not None
    return row


def _set(conn: psycopg.Connection[Any], job_id: int, **columns: object) -> None:
    assignments = ", ".join(f"{name} = %({name})s" for name in columns)
    params = {**columns, "id": job_id}
    with conn.cursor() as cur:
        cur.execute(f"UPDATE jobs SET {assignments} WHERE id = %(id)s", params)


def _make_stale(
    conn: psycopg.Connection[Any],
    job_id: int,
    *,
    seconds_ago: float,
    locked_by: str,
    attempts: int,
) -> None:
    """Put ``job_id`` into ``running`` with a heartbeat ``seconds_ago`` old.

    Raw SQL, not ``claim()`` + a sleep, per testing-guidelines.md.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs SET state='running', locked_by=%(locked_by)s, locked_at=now(),
                   heartbeat_at = now() - (%(seconds_ago)s * interval '1 second'),
                   attempts=%(attempts)s
            WHERE id = %(id)s
            """,
            {
                "locked_by": locked_by,
                "seconds_ago": seconds_ago,
                "attempts": attempts,
                "id": job_id,
            },
        )


def _delta_seconds(conn: psycopg.Connection[Any], job_id: int) -> float:
    with conn.cursor() as cur:
        cur.execute("SELECT run_after - now() FROM jobs WHERE id = %s", (job_id,))
        row = cur.fetchone()
    assert row is not None
    return row[0].total_seconds()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# No double claim (8 threads, 200 jobs)
# ---------------------------------------------------------------------------


def test_no_double_claim_across_eight_threads(head_dsn: str) -> None:
    n_jobs = 200
    n_threads = 8

    with _connect(head_dsn) as seed_conn:
        seed_queue = PostgresQueue(seed_conn, settings=_settings())
        enqueued_ids = {seed_queue.enqueue("ingest", f"v{i}") for i in range(n_jobs)}
    assert None not in enqueued_ids
    assert len(enqueued_ids) == n_jobs

    barrier = threading.Barrier(n_threads)
    claimed: list[int] = []
    claimed_lock = threading.Lock()

    def worker(name: str) -> None:
        conn = _connect(head_dsn)
        try:
            worker_queue = PostgresQueue(conn, settings=_settings())
            barrier.wait()
            misses = 0
            while True:
                with claimed_lock:
                    if len(claimed) >= n_jobs:
                        return
                with worker_queue.claim(["ingest"], worker=name) as job:
                    if job is None:
                        misses += 1
                        # A safety valve against a real infinite loop, not a
                        # timing assumption: under correct SKIP LOCKED
                        # behaviour a None here only happens because every
                        # remaining pending row is momentarily locked by
                        # another one of these 8 threads, which resolves in
                        # microseconds.
                        if misses > 5000:
                            return
                        continue
                    misses = 0
                    with claimed_lock:
                        claimed.append(job.id)
        finally:
            conn.close()

    threads = [
        threading.Thread(target=worker, args=(f"w{i}",)) for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert all(not t.is_alive() for t in threads), "a claiming thread hung"

    assert len(claimed) == n_jobs, "not every job was claimed"
    assert len(set(claimed)) == n_jobs, "some job was claimed more than once"
    assert set(claimed) == enqueued_ids


# ---------------------------------------------------------------------------
# Skip, don't block
# ---------------------------------------------------------------------------


def test_second_claim_skips_a_locked_row_and_claims_a_different_one(
    head_dsn: str,
) -> None:
    with _connect(head_dsn) as seed_conn:
        seed_queue = PostgresQueue(seed_conn, settings=_settings())
        locked_id = seed_queue.enqueue("ingest", "v-locked")
        free_id = seed_queue.enqueue("ingest", "v-free")
    assert locked_id is not None
    assert free_id is not None

    lock_conn = psycopg.connect(head_dsn)  # autocommit=False: transaction stays open
    try:
        with lock_conn.cursor() as cur:
            cur.execute("SELECT id FROM jobs WHERE id = %s FOR UPDATE", (locked_id,))
            cur.fetchone()

        result: list[Job | None] = []

        def attempt_claim() -> None:
            with _connect(head_dsn) as conn2:
                queue2 = PostgresQueue(conn2, settings=_settings())
                with queue2.claim(["ingest"], worker="w2") as job:
                    result.append(job)

        t = threading.Thread(target=attempt_claim)
        t.start()
        t.join(timeout=2)
        assert not t.is_alive(), (
            "the second claim waited for the locked row instead of skipping it"
        )
    finally:
        lock_conn.rollback()
        lock_conn.close()

    assert len(result) == 1
    job = result[0]
    assert job is not None
    assert job.id == free_id


def test_second_claim_returns_none_without_blocking_when_the_only_job_is_locked(
    head_dsn: str,
) -> None:
    with _connect(head_dsn) as seed_conn:
        seed_queue = PostgresQueue(seed_conn, settings=_settings())
        locked_id = seed_queue.enqueue("ingest", "v-locked")
    assert locked_id is not None

    lock_conn = psycopg.connect(head_dsn)
    try:
        with lock_conn.cursor() as cur:
            cur.execute("SELECT id FROM jobs WHERE id = %s FOR UPDATE", (locked_id,))
            cur.fetchone()

        result: list[Job | None] = []

        def attempt_claim() -> None:
            with _connect(head_dsn) as conn2:
                queue2 = PostgresQueue(conn2, settings=_settings())
                with queue2.claim(["ingest"], worker="w2") as job:
                    result.append(job)

        t = threading.Thread(target=attempt_claim)
        t.start()
        t.join(timeout=2)
        assert not t.is_alive(), (
            "the second claim waited for the locked row instead of returning None"
        )
    finally:
        lock_conn.rollback()
        lock_conn.close()

    assert result == [None]


# ---------------------------------------------------------------------------
# Concurrent enqueue
# ---------------------------------------------------------------------------


def test_concurrent_enqueue_of_the_same_key_produces_exactly_one_row(
    head_dsn: str,
) -> None:
    n_threads = 8
    barrier = threading.Barrier(n_threads)
    results: list[int | None] = [None] * n_threads
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(i: int) -> None:
        try:
            with _connect(head_dsn) as conn:
                worker_queue = PostgresQueue(conn, settings=_settings())
                barrier.wait()
                results[i] = worker_queue.enqueue(
                    "analyze", "v1", dedupe_key="same-key"
                )
        except BaseException as exc:  # noqa: BLE001
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert all(not t.is_alive() for t in threads)

    # The losing 7 calls must return None per the contract (architecture.md
    # §0 C1) - not raise.
    assert not errors, f"enqueue() raised under the race: {errors!r}"

    non_none = [r for r in results if r is not None]
    assert len(non_none) == 1

    with _connect(head_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM jobs WHERE video_id = 'v1' AND kind = 'analyze' "
            "AND dedupe_key = 'same-key'"
        )
        row = cur.fetchone()
        assert row is not None
        assert row[0] == 1


def test_concurrent_enqueue_on_transactional_connections_commits_one_row_and_leaves_all_idle(
    head_dsn: str,
) -> None:
    """#75: the 8-thread race on default (non-autocommit) connections, as ``connect()`` gives."""
    n_threads = 8
    barrier = threading.Barrier(n_threads)
    results: list[int | None] = [None] * n_threads
    statuses: list[object] = [None] * n_threads
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(i: int) -> None:
        conn = psycopg.connect(head_dsn)
        try:
            worker_queue = PostgresQueue(conn, settings=_settings())
            barrier.wait(timeout=10)
            results[i] = worker_queue.enqueue("analyze", "v1", dedupe_key="same-key")
            statuses[i] = conn.info.transaction_status
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            with errors_lock:
                errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert all(not t.is_alive() for t in threads)

    assert not errors, f"enqueue() raised under the race: {errors!r}"
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    assert statuses == [psycopg.pq.TransactionStatus.IDLE] * n_threads

    with _connect(head_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM jobs WHERE video_id = 'v1' AND kind = 'analyze'"
            " AND dedupe_key = 'same-key'"
        )
        assert cur.fetchall() == [(winners[0],)]


# ---------------------------------------------------------------------------
# Duplicate while running / re-run after completion (C1)
# ---------------------------------------------------------------------------


def test_enqueue_returns_none_while_the_matching_job_is_running(
    queue: PostgresQueue,
) -> None:
    job_id = queue.enqueue("ingest", "v1", dedupe_key="k1")
    assert job_id is not None

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        assert queue.enqueue("ingest", "v1", dedupe_key="k1") is None


@pytest.mark.parametrize(
    "raise_exc", [None, PermanentSourceError], ids=["done", "dead"]
)
def test_enqueue_after_the_prior_job_finishes_creates_a_new_row(
    queue: PostgresQueue,
    conn: psycopg.Connection[Any],
    raise_exc: type[BaseException] | None,
) -> None:
    first_id = queue.enqueue("ingest", "v1", dedupe_key="k1")
    assert first_id is not None

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        if raise_exc is not None:
            raise raise_exc("removed")

    expected_state = "dead" if raise_exc is not None else "done"
    assert _row(conn, first_id)["state"] == expected_state

    second_id = queue.enqueue("ingest", "v1", dedupe_key="k1")

    assert second_id is not None
    assert second_id != first_id
    assert _row(conn, second_id)["state"] == "pending"


# ---------------------------------------------------------------------------
# Retry path
# ---------------------------------------------------------------------------


def test_retry_path_leaves_the_job_pending_with_backoff_and_error_fields(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        raise TransientNetworkError("dns lookup failed")

    row = _row(conn, job_id)
    assert row["state"] == "pending"
    assert row["attempts"] == 1
    assert _delta_seconds(conn, job_id) > 0
    assert row["error_class"] == "TRANSIENT_NETWORK"
    assert "dns lookup failed" in row["last_error"]


# ---------------------------------------------------------------------------
# Dead-letter at max
# ---------------------------------------------------------------------------


def test_ingest_dead_letters_on_its_fourth_attempt(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")  # kind max is 4
    assert job_id is not None
    _set(conn, job_id, attempts=3)

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        assert job.attempts == 4
        raise TransientNetworkError("boom")

    row = _row(conn, job_id)
    assert row["state"] == "dead"
    assert row["attempts"] == 4


def test_transcribe_dead_letters_on_its_second_attempt(
    conn: psycopg.Connection[Any],
) -> None:
    q = PostgresQueue(
        conn, settings=_settings(MAX_ATTEMPTS_TRANSCRIBE=2), rng=random.Random(1)
    )
    job_id = q.enqueue("transcribe", "v1")
    assert job_id is not None
    _set(conn, job_id, attempts=1)

    with q.claim(["transcribe"], worker="w1") as job:
        assert job is not None
        assert job.attempts == 2
        raise TransientNetworkError("boom")

    row = _row(conn, job_id)
    assert row["state"] == "dead"
    assert row["attempts"] == 2


def test_permanent_source_error_dead_letters_on_its_first_attempt(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        assert job.attempts == 1
        raise PermanentSourceError("removed")

    row = _row(conn, job_id)
    assert row["state"] == "dead"
    assert row["attempts"] == 1
    assert row["error_class"] == "PERMANENT_SOURCE"


# ---------------------------------------------------------------------------
# Reap stale, spare live / reap at max attempts
# ---------------------------------------------------------------------------


def test_reap_stale_requeues_only_the_stale_of_two_running_jobs(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    stale_id = queue.enqueue("ingest", "v-stale")
    fresh_id = queue.enqueue("ingest", "v-fresh")
    assert stale_id is not None
    assert fresh_id is not None
    _make_stale(conn, stale_id, seconds_ago=600, locked_by="wA", attempts=1)
    _make_stale(conn, fresh_id, seconds_ago=10, locked_by="wB", attempts=1)

    count = queue.reap_stale(older_than_sec=300)

    assert count == 1
    stale_row = _row(conn, stale_id)
    assert stale_row["state"] == "pending"
    assert stale_row["locked_by"] is None
    fresh_row = _row(conn, fresh_id)
    assert fresh_row["state"] == "running"
    assert fresh_row["locked_by"] == "wB"


def test_reap_stale_moves_a_job_at_max_attempts_to_dead_instead_of_pending(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")  # kind max is 4
    assert job_id is not None
    _make_stale(conn, job_id, seconds_ago=600, locked_by="w1", attempts=4)
    _set(conn, job_id, error_class="TRANSIENT_NETWORK")

    count = queue.reap_stale(older_than_sec=300)

    assert count == 1
    row = _row(conn, job_id)
    assert row["state"] == "dead"
    assert row["locked_by"] is None
    assert row["error_class"] is None


# ---------------------------------------------------------------------------
# Split brain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["done", "failed"])
def test_split_brain_leaves_b_as_sole_owner_after_a_resolves(
    head_dsn: str, outcome: str
) -> None:
    with _connect(head_dsn) as seed_conn:
        seed_queue = PostgresQueue(seed_conn, settings=_settings())
        job_id = seed_queue.enqueue("ingest", "v1")
    assert job_id is not None

    conn_a = _connect(head_dsn)
    conn_b = _connect(head_dsn)
    try:
        queue_a = PostgresQueue(conn_a, settings=_settings())
        queue_b = PostgresQueue(conn_b, settings=_settings())

        # Worker A claims the job.
        cm_a = queue_a.claim(["ingest"], worker="A")
        job_a = cm_a.__enter__()
        assert job_a is not None
        assert job_a.id == job_id
        assert job_a.attempts == 1

        # A goes silent long enough to be reaped, while it's still "inside"
        # its (never-exited) handler - the point of the split-brain scenario.
        with conn_a.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET heartbeat_at = now() - interval '10 minutes' "
                "WHERE id = %s",
                (job_id,),
            )
        reaped = queue_a.reap_stale(older_than_sec=300)
        assert reaped == 1
        # The reap backs the job off (#87); expire that backoff without sleeping.
        with conn_a.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET run_after = now() - interval '1 second' WHERE id = %s",
                (job_id,),
            )

        # Worker B claims the now-pending job.
        cm_b = queue_b.claim(["ingest"], worker="B")
        job_b = cm_b.__enter__()
        assert job_b is not None
        assert job_b.id == job_id
        assert job_b.attempts == 2

        # A checks in, unaware it lost the job.
        assert queue_a.heartbeat(job_id, "A") is False

        # A's handler finally finishes - clean exit or a failure, either way
        # its outcome update must be a no-op: it no longer owns the row.
        if outcome == "failed":
            try:
                raise TransientNetworkError("A finished late")
            except TransientNetworkError:
                cm_a.__exit__(*sys.exc_info())
        else:
            cm_a.__exit__(None, None, None)

        row = _row(conn_a, job_id)
        assert row["state"] == "running"
        assert row["locked_by"] == "B"
        assert row["attempts"] == 2

        cm_b.__exit__(None, None, None)
    finally:
        conn_a.close()
        conn_b.close()


# ---------------------------------------------------------------------------
# Crash mid-job
# ---------------------------------------------------------------------------

# A real worker process: claims a job, announces the claim over stdout once
# the claim has committed, then blocks (no busy-wait, no sleep) until the
# parent test kills it with SIGKILL - simulating a worker that crashes mid
# handler, never releasing the job.
_CRASH_WORKER_SRC = textwrap.dedent(
    """
    import sys

    from common.db import connect
    from common.queue import PostgresQueue

    conn = connect()
    queue = PostgresQueue(conn)
    with queue.claim(["ingest"], worker=sys.argv[1]) as job:
        assert job is not None
        print("claimed", flush=True)
        sys.stdin.readline()  # blocks until the parent SIGKILLs this process
    """
)


def test_crash_mid_job_is_reaped_and_reclaimed(
    head_dsn: str, queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None

    # Not "reading settings" (the banned-api rule's concern): building the
    # subprocess's environment so the child's own get_settings() sees the
    # test database.
    env = {**os.environ, "DATABASE_URL": head_dsn}  # noqa: TID251
    proc = subprocess.Popen(
        [sys.executable, "-c", _CRASH_WORKER_SRC, "crashy"],
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        announced: list[str] = []
        reader = threading.Thread(
            target=lambda: announced.append(proc.stdout.readline())  # type: ignore[union-attr]
        )
        reader.start()
        reader.join(timeout=15)
        assert not reader.is_alive(), "the crash worker never announced its claim"
        assert announced and announced[0].strip() == "claimed"

        row = _row(conn, job_id)
        assert row["state"] == "running"
        assert row["locked_by"] == "crashy"

        proc.kill()  # SIGKILL
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    # The kill left the job stuck 'running' - nothing released it.
    row = _row(conn, job_id)
    assert row["state"] == "running"
    assert row["locked_by"] == "crashy"

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET heartbeat_at = now() - interval '10 minutes' "
            "WHERE id = %s",
            (job_id,),
        )
    count = queue.reap_stale(older_than_sec=300)
    assert count == 1
    assert _row(conn, job_id)["state"] == "pending"
    # The reap backs the job off (#87); expire that backoff without sleeping.
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET run_after = now() - interval '1 second' WHERE id = %s",
            (job_id,),
        )

    with queue.claim(["ingest"], worker="w2") as job2:
        assert job2 is not None
        assert job2.id == job_id


# ---------------------------------------------------------------------------
# Graceful release
# ---------------------------------------------------------------------------


def test_graceful_release_on_keyboard_interrupt_leaves_the_job_claimable_at_once(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None

    with pytest.raises(KeyboardInterrupt), queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        assert job.attempts == 1
        raise KeyboardInterrupt()

    released = _row(conn, job_id)
    assert released["state"] == "pending"
    assert released["attempts"] == 0  # unchanged from before the interrupted claim
    assert released["locked_by"] is None

    with queue.claim(["ingest"], worker="w2") as job2:
        assert job2 is not None
        assert job2.id == job_id
        assert job2.attempts == 1


# ---------------------------------------------------------------------------
# Forced interleavings (testing-guidelines.md, "Flaky or racy behaviour")
#
# The Barrier tests above make a race likely; these make the dangerous order
# happen on every run: connection A holds its transaction open, B's call is
# started and confirmed blocked on A through pg_blocking_pids, then A commits
# or rolls back. One run is the proof - no repeat loops.
# ---------------------------------------------------------------------------


def _wait_until_blocked_by(
    observer: psycopg.Connection[Any], blocked_pid: int, blocker_pid: int
) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with observer.cursor() as cur:
            cur.execute(
                "SELECT %s = ANY(pg_blocking_pids(%s))", (blocker_pid, blocked_pid)
            )
            row = cur.fetchone()
        if row is not None and row[0]:
            return
    raise AssertionError(f"backend {blocked_pid} was never blocked by {blocker_pid}")


class _Call(threading.Thread):
    """Runs ``fn(conn)`` on its own connection; keeps the result or the error."""

    def __init__(self, dsn: str, fn: Any, *, autocommit: bool = True) -> None:
        super().__init__(daemon=True)
        self.conn = psycopg.connect(dsn, autocommit=autocommit)
        self.pid = self.conn.info.backend_pid
        self._fn = fn
        self.result: Any = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.result = self._fn(self.conn)
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            self.error = exc

    def finish(self) -> Any:
        self.join(timeout=10)
        assert not self.is_alive(), "the blocked call never finished"
        if self.error is not None:
            raise self.error
        return self.result


def _insert_pending(conn: psycopg.Connection[Any], key: str = "k") -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs (video_id, kind, dedupe_key) VALUES ('v1', 'analyze', %s)"
            " RETURNING id",
            (key,),
        )
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _rows_for(conn: psycopg.Connection[Any], key: str = "k") -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, state FROM jobs WHERE video_id = 'v1' AND kind = 'analyze'"
            " AND dedupe_key = %s",
            (key,),
        )
        return cur.fetchall()


@pytest.mark.parametrize(
    "autocommit", [True, False], ids=["autocommit", "transactional"]
)
def test_enqueue_blocked_by_an_uncommitted_duplicate_returns_none_once_it_commits(
    head_dsn: str, autocommit: bool
) -> None:
    holder = psycopg.connect(head_dsn)
    racer: _Call | None = None
    try:
        winner_id = _insert_pending(holder)
        racer = _Call(
            head_dsn,
            lambda c: PostgresQueue(c, settings=_settings()).enqueue(
                "analyze", "v1", dedupe_key="k"
            ),
            autocommit=autocommit,
        )
        racer.start()
        with _connect(head_dsn) as observer:
            _wait_until_blocked_by(observer, racer.pid, holder.info.backend_pid)
            assert racer.is_alive()
            holder.commit()
            assert racer.finish() is None
            assert (
                racer.conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
            )
            assert _rows_for(observer) == [(winner_id, "pending")]
    finally:
        holder.rollback()
        holder.close()
        if racer is not None:
            racer.join(timeout=10)
            racer.conn.close()


@pytest.mark.parametrize(
    "autocommit", [True, False], ids=["autocommit", "transactional"]
)
def test_enqueue_blocked_by_an_uncommitted_duplicate_inserts_once_it_rolls_back(
    head_dsn: str, autocommit: bool
) -> None:
    holder = psycopg.connect(head_dsn)
    racer: _Call | None = None
    try:
        _insert_pending(holder)
        racer = _Call(
            head_dsn,
            lambda c: PostgresQueue(c, settings=_settings()).enqueue(
                "analyze", "v1", dedupe_key="k"
            ),
            autocommit=autocommit,
        )
        racer.start()
        with _connect(head_dsn) as observer:
            _wait_until_blocked_by(observer, racer.pid, holder.info.backend_pid)
            assert racer.is_alive()
            holder.rollback()
            new_id = racer.finish()
            assert isinstance(new_id, int)
            assert (
                racer.conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
            )
            assert _rows_for(observer) == [(new_id, "pending")]
    finally:
        holder.rollback()
        holder.close()
        if racer is not None:
            racer.join(timeout=10)
            racer.conn.close()


def test_a_claim_racing_a_claim_mid_commit_takes_the_other_job(head_dsn: str) -> None:
    """A claims job 1 and holds the row lock; B's claim must skip it, never wait."""
    with _connect(head_dsn) as seed:
        q = PostgresQueue(seed, settings=_settings())
        first = q.enqueue("ingest", "v1")
        second = q.enqueue("ingest", "v2")
    holder = psycopg.connect(head_dsn)
    try:
        with holder.cursor() as cur:
            cur.execute(
                "SELECT id FROM jobs WHERE state='pending' ORDER BY priority DESC, run_after"
                " LIMIT 1 FOR UPDATE"
            )
            row = cur.fetchone()
        assert row is not None and row[0] == first

        def claim(c: psycopg.Connection[Any]) -> int | None:
            with PostgresQueue(c, settings=_settings()).claim(
                ["ingest"], worker="B"
            ) as job:
                return None if job is None else job.id

        racer = _Call(head_dsn, claim)
        racer.start()
        try:
            assert racer.finish() == second
        finally:
            racer.conn.close()
    finally:
        holder.rollback()
        holder.close()
