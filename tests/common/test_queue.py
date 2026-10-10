"""Tests for ``common.queue`` (issue #11, architecture.md §0 C1, §2, §4-§6, §8.4).

Every test drives a real ``PostgresQueue`` against a Postgres database
migrated to head via the ``postgres_dsn`` fixture from ``tests/conftest.py``
(task #7), so the whole module is marked ``integration``. The concurrency and
crash suite (multiple workers, killed processes) is #12 - these tests cover
single-worker behaviour only, one AC bullet at a time.

Jitter is deterministic because a fixed ``random.Random`` seed is injected;
timestamps are asserted with generous tolerances since they come from the
database's own ``now()``, not the test process's clock.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from psycopg import pq
from psycopg.rows import dict_row

from common.config import Settings, get_settings
from common.errors import (
    Cancelled,
    Defer,
    PermanentSourceError,
    RateLimitedError,
    ToolFailureError,
    TransientNetworkError,
)
from common.queue import Job, PostgresQueue, RetryOutcome

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config_for(dsn: str, monkeypatch: pytest.MonkeyPatch) -> Config:
    # Same pattern as tests/migrations/test_jobs.py (#11): env.py reads
    # DATABASE_URL via common.config.get_settings(), which is lru_cache'd per
    # process, so the cache has to be cleared after pointing it at the test
    # database.
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
    # autocommit=True is a test-only convenience so the raw setup/assertion
    # SQL sprinkled through these tests (updating attempts, heartbeat_at, ...
    # directly to control time per AGENTS.md/testing-guidelines) doesn't need
    # its own explicit commits. PostgresQueue itself must work the same way
    # regardless: its methods always wrap their own work in
    # ``conn.transaction()``, which psycopg supports on an autocommit
    # connection by suspending autocommit for the block.
    with psycopg.connect(head_dsn, autocommit=True) as connection:
        yield connection


@pytest.fixture
def commit_conn(head_dsn: str) -> Iterator[psycopg.Connection[Any]]:
    """A connection with default settings (``autocommit=False``) - the kind
    ``common.db.connect()`` returns in production (issue #73). Unlike
    ``conn`` above, nothing suspends autocommit here: ``enqueue`` must commit
    its own work through ``self._conn.transaction()`` alone.
    """
    with psycopg.connect(head_dsn) as connection:
        yield connection


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"DATABASE_URL": "postgresql://u:p@h/db"}
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


@pytest.fixture
def queue(conn: psycopg.Connection[Any]) -> PostgresQueue:
    return PostgresQueue(conn, settings=_settings(), rng=random.Random(1234))


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


def _delta_seconds(conn: psycopg.Connection[Any], job_id: int) -> float:
    """``run_after`` minus the database's own ``now()``, in seconds."""
    with conn.cursor() as cur:
        cur.execute("SELECT run_after - now() FROM jobs WHERE id = %s", (job_id,))
        row = cur.fetchone()
    assert row is not None
    interval: timedelta = row[0]
    return interval.total_seconds()


def _count(conn: psycopg.Connection[Any]) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM jobs")
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


# ---------------------------------------------------------------------------
# enqueue
# ---------------------------------------------------------------------------


def test_enqueue_returns_the_new_job_id(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")

    assert isinstance(job_id, int)
    row = _row(conn, job_id)
    assert row["video_id"] == "v1"
    assert row["kind"] == "ingest"
    assert row["dedupe_key"] == "default"
    assert row["state"] == "pending"
    assert row["priority"] == 0
    assert row["payload"] == {}


@pytest.mark.parametrize("blocking_state", ["pending", "running"])
def test_enqueue_returns_none_and_touches_no_row_when_active_job_exists(
    queue: PostgresQueue, conn: psycopg.Connection[Any], blocking_state: str
) -> None:
    first_id = queue.enqueue("ingest", "v1")
    assert first_id is not None
    _set(conn, first_id, state=blocking_state)

    result = queue.enqueue("ingest", "v1")

    assert result is None
    assert _count(conn) == 1
    assert _row(conn, first_id)["state"] == blocking_state


@pytest.mark.parametrize("finished_state", ["done", "dead"])
def test_enqueue_succeeds_again_once_prior_job_is_finished(
    queue: PostgresQueue, conn: psycopg.Connection[Any], finished_state: str
) -> None:
    first_id = queue.enqueue("ingest", "v1")
    assert first_id is not None
    _set(conn, first_id, state=finished_state)

    second_id = queue.enqueue("ingest", "v1")

    assert second_id is not None
    assert second_id != first_id
    assert _count(conn) == 2
    assert _row(conn, second_id)["state"] == "pending"


def test_enqueue_unknown_kind_raises_value_error_without_touching_the_database(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    with pytest.raises(ValueError):
        queue.enqueue("bogus", "v1")

    assert _count(conn) == 0


def test_enqueue_distinct_dedupe_keys_do_not_block_each_other(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    first_id = queue.enqueue("analyze", "v1", dedupe_key="v1:ollama")
    second_id = queue.enqueue("analyze", "v1", dedupe_key="v2:ollama")

    assert first_id is not None
    assert second_id is not None
    assert first_id != second_id


# ---------------------------------------------------------------------------
# enqueue: non-conflict errors still raise (issue #73)
# ---------------------------------------------------------------------------


def test_unique_violation_on_a_different_index_still_raises(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    first_id = queue.enqueue("ingest", "v1")
    assert first_id is not None
    # Move jobs_id_seq back so the next nextval() reproduces first_id,
    # forcing the next insert to collide on jobs_pkey rather than on
    # jobs_active_uniq. A bare `ON CONFLICT DO NOTHING` (no target) would
    # swallow this and return None instead of raising.
    with conn.cursor() as cur:
        cur.execute("SELECT setval('jobs_id_seq', %s, false)", (first_id,))

    with pytest.raises(psycopg.errors.UniqueViolation):
        queue.enqueue("ingest", "v2")

    assert _count(conn) == 1

    # The connection is usable again, and a normal enqueue succeeds (the
    # failed nextval() call already advanced the sequence past first_id).
    assert conn.info.transaction_status == pq.TransactionStatus.IDLE
    second_id = queue.enqueue("ingest", "v3")
    assert second_id is not None
    assert second_id != first_id


def test_not_null_violation_on_a_different_constraint_still_raises(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    with pytest.raises(psycopg.errors.NotNullViolation):
        queue.enqueue("ingest", None)  # type: ignore[arg-type]

    assert _count(conn) == 0

    # The connection is usable again.
    assert conn.info.transaction_status == pq.TransactionStatus.IDLE
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None


# ---------------------------------------------------------------------------
# enqueue: commit behaviour on a non-autocommit connection (issue #73)
# ---------------------------------------------------------------------------


def test_enqueue_commits_an_insert_at_once_on_a_non_autocommit_connection(
    commit_conn: psycopg.Connection[Any], head_dsn: str
) -> None:
    q = PostgresQueue(commit_conn, settings=_settings(), rng=random.Random(1))

    job_id = q.enqueue("ingest", "v1")

    assert job_id is not None
    assert commit_conn.info.transaction_status == pq.TransactionStatus.IDLE
    # A second, independent connection sees the row without commit_conn
    # doing anything further - enqueue committed it on its own.
    with psycopg.connect(head_dsn) as other, other.cursor() as cur:
        cur.execute("SELECT count(*) FROM jobs WHERE id = %s", (job_id,))
        row = cur.fetchone()
    assert row == (1,)

    duplicate_id = q.enqueue("ingest", "v1")

    assert duplicate_id is None
    assert commit_conn.info.transaction_status == pq.TransactionStatus.IDLE


# ---------------------------------------------------------------------------
# claim
# ---------------------------------------------------------------------------


def test_claim_yields_none_when_nothing_is_claimable(queue: PostgresQueue) -> None:
    with queue.claim(["ingest"], worker="w1") as job:
        assert job is None


def test_claim_yields_a_running_job_committed_with_ownership_fields(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        assert isinstance(job, Job)
        assert job.id == job_id
        assert job.state == "running"
        assert job.locked_by == "w1"
        assert job.locked_at is not None
        assert job.heartbeat_at is not None
        assert job.attempts == 1

    row = _row(conn, job_id)
    assert row["state"] == "done"  # clean exit of the with-block


def test_handler_runs_outside_any_open_transaction(
    queue: PostgresQueue, head_dsn: str
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        # A second, independent connection must see the claim as committed
        # already. Under READ COMMITTED, it would still see 'pending' if the
        # claiming transaction were still open.
        with psycopg.connect(head_dsn) as other, other.cursor() as cur:
            cur.execute(
                "SELECT state, locked_by FROM jobs WHERE id = %s", (job_id,)
            )
            row = cur.fetchone()
        assert row == ("running", "w1")


def test_claim_order_is_priority_desc_then_run_after_asc(
    queue: PostgresQueue,
) -> None:
    now = datetime.now(UTC)
    low_priority = queue.enqueue("ingest", "v-low", priority=-5, run_after=now)
    high_priority_later = queue.enqueue(
        "ingest",
        "v-high-later",
        priority=10,
        run_after=now - timedelta(seconds=30),
    )
    high_priority_earlier = queue.enqueue(
        "ingest",
        "v-high-earlier",
        priority=10,
        run_after=now - timedelta(seconds=60),
    )
    assert low_priority and high_priority_later and high_priority_earlier

    claimed_ids = []
    for _ in range(3):
        with queue.claim(["ingest"], worker="w1") as job:
            assert job is not None
            claimed_ids.append(job.id)

    assert claimed_ids == [high_priority_earlier, high_priority_later, low_priority]


def test_claim_does_not_claim_a_job_whose_run_after_is_in_the_future(
    queue: PostgresQueue,
) -> None:
    future = datetime.now(UTC) + timedelta(hours=1)
    queue.enqueue("ingest", "v1", run_after=future)

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is None


def test_claim_does_not_claim_a_job_of_a_different_kind(queue: PostgresQueue) -> None:
    queue.enqueue("analyze", "v1")

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is None


# ---------------------------------------------------------------------------
# outcome: clean exit
# ---------------------------------------------------------------------------


def test_clean_exit_marks_the_job_done_and_clears_ownership(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None

    row = _row(conn, job_id)
    assert row["state"] == "done"
    assert row["finished_at"] is not None
    assert row["locked_by"] is None


# ---------------------------------------------------------------------------
# outcome: exception -> classify decides
# ---------------------------------------------------------------------------


def test_retryable_exception_records_error_fields_and_goes_pending_with_backoff(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None

    # Unlike Cancelled/KeyboardInterrupt/SystemExit, a plain classified
    # failure is fully handled by the queue and does not propagate: the
    # worker loop must be able to move on to the next job.
    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        raise TransientNetworkError("dns lookup failed")

    row = _row(conn, job_id)
    assert row["state"] == "pending"
    assert row["error_class"] == "TRANSIENT_NETWORK"
    assert "TransientNetworkError" in row["last_error"]
    assert "dns lookup failed" in row["last_error"]
    assert row["locked_by"] is None
    # base backoff for attempts=1 is 5 min, ±10% jitter -> roughly 270-330s.
    assert 200 <= _delta_seconds(conn, job_id) <= 400


def test_non_retryable_exception_dead_letters_immediately(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        raise PermanentSourceError("removed")

    row = _row(conn, job_id)
    assert row["state"] == "dead"
    assert row["error_class"] == "PERMANENT_SOURCE"
    assert row["locked_by"] is None


def test_unclassified_exception_is_a_bug_and_dead_letters(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        raise RuntimeError("boom")

    row = _row(conn, job_id)
    assert row["state"] == "dead"
    assert row["error_class"] == "BUG"
    assert "RuntimeError" in row["last_error"]
    assert "boom" in row["last_error"]


# ---------------------------------------------------------------------------
# outcome: actual delay = max(computed backoff, policy.min_backoff, retry_after_sec)
# ---------------------------------------------------------------------------


def test_rate_limited_waits_at_least_the_policy_min_backoff(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        raise RateLimitedError("429")

    # min_backoff for RATE_LIMITED is 30 min; computed exponential backoff at
    # attempts=1 (5 min) is well below it, so the floor wins.
    delta = _delta_seconds(conn, job_id)
    assert 1750 <= delta <= 1900


def test_rate_limited_retry_after_overrides_the_min_backoff_when_longer(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        raise RateLimitedError("429", retry_after_sec=5400)

    delta = _delta_seconds(conn, job_id)
    assert 5350 <= delta <= 5500


# ---------------------------------------------------------------------------
# outcome: dead once attempts reaches the smaller of the two limits
# ---------------------------------------------------------------------------


def test_dead_letters_once_the_kind_max_attempts_is_reached(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")  # kind max is 4
    assert job_id is not None
    _set(conn, job_id, attempts=3)

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        assert job.attempts == 4
        raise TransientNetworkError()

    row = _row(conn, job_id)
    assert row["state"] == "dead"
    assert row["attempts"] == 4


def test_dead_letters_once_the_kind_max_is_not_reached_but_the_class_max_is(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")  # kind max is 4
    assert job_id is not None
    _set(conn, job_id, attempts=2)

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        assert job.attempts == 3
        raise ToolFailureError()  # TOOL_FAILURE class max is 3

    row = _row(conn, job_id)
    assert row["state"] == "dead"
    assert row["error_class"] == "TOOL_FAILURE"
    assert row["attempts"] == 3


def test_retries_when_below_both_the_kind_and_class_max(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None
    _set(conn, job_id, attempts=1)  # below both kind max (4) and class max (3)

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        raise ToolFailureError()

    row = _row(conn, job_id)
    assert row["state"] == "pending"


def test_transcribe_kind_dead_letters_after_its_lower_max_attempts(
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
        raise TransientNetworkError()

    row = _row(conn, job_id)
    assert row["state"] == "dead"


# ---------------------------------------------------------------------------
# outcome: Defer
# ---------------------------------------------------------------------------


def test_defer_returns_to_pending_at_until_without_spending_an_attempt(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None
    until = datetime.now(UTC) + timedelta(hours=2)

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        assert job.attempts == 1
        raise Defer(until=until)

    row = _row(conn, job_id)
    assert row["state"] == "pending"
    assert row["attempts"] == 0  # not counted
    assert abs((row["run_after"] - until).total_seconds()) < 1
    assert row["locked_by"] is None


# ---------------------------------------------------------------------------
# outcome: Cancelled / KeyboardInterrupt / SystemExit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc_type", [Cancelled, KeyboardInterrupt, SystemExit])
def test_cancellation_returns_to_pending_now_without_spending_an_attempt_and_reraises(
    queue: PostgresQueue, conn: psycopg.Connection[Any], exc_type: type[BaseException]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None

    with pytest.raises(exc_type), queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        assert job.attempts == 1
        raise exc_type()

    row = _row(conn, job_id)
    assert row["state"] == "pending"
    assert row["attempts"] == 0
    assert row["locked_by"] is None
    assert _delta_seconds(conn, job_id) <= 1


# ---------------------------------------------------------------------------
# outcome: lost ownership -> zero rows, warning, no raise, no overwrite
# ---------------------------------------------------------------------------


def test_outcome_update_is_a_noop_when_the_job_was_reclaimed_by_another_worker(
    queue: PostgresQueue, conn: psycopg.Connection[Any], caplog: pytest.LogCaptureFixture
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None

    with caplog.at_level("WARNING"), queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        # Simulate a reap-and-reclaim by a second worker while we still
        # think we own the job.
        _set(conn, job_id, locked_by="w2", attempts=2)

    row = _row(conn, job_id)
    # The new owner's row is untouched by our (no-op) outcome update.
    assert row["locked_by"] == "w2"
    assert row["attempts"] == 2
    assert row["state"] == "running"
    assert caplog.records, "expected a warning to be logged"
    assert any(r.levelname == "WARNING" for r in caplog.records)


# ---------------------------------------------------------------------------
# heartbeat
# ---------------------------------------------------------------------------


def test_heartbeat_returns_true_and_advances_heartbeat_at_when_owned(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None
    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        assert job.heartbeat_at is not None
        before = job.heartbeat_at
        # Push the existing heartbeat into the past so a fresh now() is
        # provably later, without sleeping.
        _set(conn, job_id, heartbeat_at=before - timedelta(seconds=30))

        result = queue.heartbeat(job_id, "w1")

        assert result is True
        after = _row(conn, job_id)["heartbeat_at"]
        assert after > before - timedelta(seconds=30)


def test_heartbeat_returns_false_when_not_owned_by_that_worker(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None
    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        before = _row(conn, job_id)["heartbeat_at"]

        result = queue.heartbeat(job_id, "w2")

        assert result is False
        assert _row(conn, job_id)["heartbeat_at"] == before


def test_heartbeat_returns_false_for_a_job_that_is_not_running(
    queue: PostgresQueue,
) -> None:
    job_id = queue.enqueue("ingest", "v1")  # still 'pending', never claimed
    assert job_id is not None

    assert queue.heartbeat(job_id, "w1") is False


# ---------------------------------------------------------------------------
# reap_stale
# ---------------------------------------------------------------------------


def test_reap_stale_returns_stale_running_jobs_to_pending_and_counts_them(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None
    # Set up a 'running' job with a stale heartbeat directly, rather than
    # going through claim() then waiting - no sleeps, per
    # testing-guidelines.md.
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs SET state='running', locked_by='w1', locked_at=now(),
                   heartbeat_at = now() - interval '10 minutes', attempts=1
            WHERE id = %s
            """,
            (job_id,),
        )

    count = queue.reap_stale(older_than_sec=300)

    assert count == 1
    row = _row(conn, job_id)
    assert row["state"] == "pending"
    assert row["locked_by"] is None


def test_reap_stale_ignores_jobs_whose_heartbeat_is_within_the_threshold(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None
    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        # heartbeat_at was just set to now() by claim(); well within threshold.
        _set(conn, job_id, error_class="TRANSIENT_NETWORK")
        count = queue.reap_stale(older_than_sec=300)
        assert count == 0
        row = _row(conn, job_id)
        assert row["state"] == "running"
        assert row["error_class"] == "TRANSIENT_NETWORK"


def test_reap_stale_dead_letters_a_job_that_already_used_its_max_attempts(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")  # kind max is 4
    assert job_id is not None
    _set(conn, job_id, attempts=3)
    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        assert job.attempts == 4
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET heartbeat_at = now() - interval '10 minutes' "
                "WHERE id = %s",
                (job_id,),
            )
        count = queue.reap_stale(older_than_sec=300)

    assert count == 1
    row = _row(conn, job_id)
    assert row["state"] == "dead"
    assert row["last_error"] == "worker lost: heartbeat stale"
    assert row["locked_by"] is None


def test_reap_stale_uses_settings_reap_after_sec_when_older_than_sec_omitted(
    conn: psycopg.Connection[Any],
) -> None:
    q = PostgresQueue(conn, settings=_settings(REAP_AFTER_SEC=300), rng=random.Random(1))
    job_id = q.enqueue("ingest", "v1")
    assert job_id is not None
    with q.claim(["ingest"], worker="w1") as job:
        assert job is not None
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET heartbeat_at = now() - interval '301 seconds' "
                "WHERE id = %s",
                (job_id,),
            )
        count = q.reap_stale()

    assert count == 1
    assert _row(conn, job_id)["state"] == "pending"


# ---------------------------------------------------------------------------
# reap_stale hardening (issue #35)
# ---------------------------------------------------------------------------


def _make_running(
    queue: PostgresQueue,
    conn: psycopg.Connection[Any],
    *,
    video_id: str = "v1",
    kind: str = "ingest",
    heartbeat_age_sec: int | None = 600,
    attempts: int = 1,
    locked_by: str = "w1",
) -> int:
    """A ``running`` job with the given heartbeat age, written directly."""
    job_id = queue.enqueue(kind, video_id)
    assert job_id is not None
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs SET state='running', locked_by=%(worker)s, locked_at=now(),
                   attempts=%(attempts)s,
                   heartbeat_at = CASE WHEN %(age)s::int IS NULL THEN NULL
                       ELSE now() - (%(age)s::int * interval '1 second') END
            WHERE id = %(id)s
            """,
            {
                "worker": locked_by,
                "attempts": attempts,
                "age": heartbeat_age_sec,
                "id": job_id,
            },
        )
    return job_id


def test_reap_stale_leaves_a_heartbeat_299s_old_alone_and_reaps_301s(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    fresh = _make_running(queue, conn, video_id="fresh", heartbeat_age_sec=299)
    stale = _make_running(queue, conn, video_id="stale", heartbeat_age_sec=301)

    count = queue.reap_stale(older_than_sec=300)

    assert count == 1
    assert _row(conn, fresh)["state"] == "running"
    assert _row(conn, stale)["state"] == "pending"


def test_reap_stale_treats_a_null_heartbeat_as_stale(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = _make_running(queue, conn, heartbeat_age_sec=None)

    count = queue.reap_stale(older_than_sec=300)

    assert count == 1
    assert _row(conn, job_id)["state"] == "pending"


@pytest.mark.parametrize("state", ["pending", "done", "dead"])
def test_reap_stale_never_touches_rows_that_are_not_running(
    queue: PostgresQueue, conn: psycopg.Connection[Any], state: str
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None
    _set(
        conn,
        job_id,
        state=state,
        heartbeat_at=datetime.now(UTC) - timedelta(hours=2),
        attempts=1,
        error_class="TRANSIENT_NETWORK",
    )
    before = _row(conn, job_id)

    count = queue.reap_stale(older_than_sec=300)

    assert count == 0
    assert _row(conn, job_id) == before


def test_reap_stale_requeue_keeps_attempts_and_records_why(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = _make_running(queue, conn, attempts=3)  # ingest max is 4
    _set(conn, job_id, error_class="TRANSIENT_NETWORK")

    queue.reap_stale(older_than_sec=300)

    row = _row(conn, job_id)
    assert row["state"] == "pending"
    assert row["attempts"] == 3
    assert row["locked_by"] is None
    assert row["locked_at"] is None
    assert row["last_error"] == "worker lost: heartbeat stale"
    assert row["finished_at"] is None
    assert row["error_class"] is None


@pytest.mark.parametrize(
    ("attempts", "expected"),
    [(3, "pending"), (4, "dead"), (5, "dead")],
)
def test_reap_stale_dead_letters_at_and_beyond_max_attempts(
    queue: PostgresQueue, conn: psycopg.Connection[Any], attempts: int, expected: str
) -> None:
    job_id = _make_running(queue, conn, attempts=attempts)
    _set(conn, job_id, error_class="TRANSIENT_NETWORK")

    queue.reap_stale(older_than_sec=300)

    row = _row(conn, job_id)
    assert row["state"] == expected
    assert row["last_error"] == "worker lost: heartbeat stale"
    assert row["locked_by"] is None
    assert row["error_class"] is None
    assert (row["finished_at"] is not None) == (expected == "dead")


# ---------------------------------------------------------------------------
# reap_stale clears error_class (issue #114)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "attempts", "expected"),
    [
        ("ingest", 1, "pending"),
        ("ingest", 4, "dead"),
        ("transcribe", 1, "pending"),
        ("transcribe", 2, "dead"),
    ],
)
def test_reap_stale_leaves_error_class_null_when_it_was_already_null(
    queue: PostgresQueue,
    conn: psycopg.Connection[Any],
    kind: str,
    attempts: int,
    expected: str,
) -> None:
    job_id = _make_running(queue, conn, kind=kind, attempts=attempts)
    assert _row(conn, job_id)["error_class"] is None

    count = queue.reap_stale(older_than_sec=300)

    assert count == 1
    row = _row(conn, job_id)
    assert row["state"] == expected
    assert row["error_class"] is None
    assert row["last_error"] == "worker lost: heartbeat stale"


@pytest.mark.parametrize(("kind", "attempts"), [("ingest", 4), ("transcribe", 2)])
def test_reap_stale_dead_letter_clears_a_seeded_class_at_each_kinds_max(
    queue: PostgresQueue, conn: psycopg.Connection[Any], kind: str, attempts: int
) -> None:
    job_id = _make_running(queue, conn, kind=kind, attempts=attempts)
    _set(conn, job_id, error_class="TRANSIENT_NETWORK")

    queue.reap_stale(older_than_sec=300)

    row = _row(conn, job_id)
    assert row["state"] == "dead"
    assert row["finished_at"] is not None
    assert row["error_class"] is None
    assert row["last_error"] == "worker lost: heartbeat stale"


def test_reap_stale_keeps_the_class_of_a_live_heartbeat_job_next_to_a_reaped_one(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    live = _make_running(queue, conn, video_id="live", heartbeat_age_sec=10)
    stale = _make_running(queue, conn, video_id="stale", heartbeat_age_sec=600)
    _set(conn, live, error_class="TOOL_FAILURE")
    _set(conn, stale, error_class="TOOL_FAILURE")

    count = queue.reap_stale(older_than_sec=300)

    assert count == 1
    assert _row(conn, live)["error_class"] == "TOOL_FAILURE"
    assert _row(conn, stale)["error_class"] is None


def test_a_job_reaped_to_pending_gets_its_class_back_when_it_later_fails(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = _make_running(queue, conn, attempts=1)
    _set(conn, job_id, error_class="TOOL_FAILURE")
    queue.reap_stale(older_than_sec=300)
    assert _row(conn, job_id)["error_class"] is None
    _set(conn, job_id, run_after=datetime.now(UTC) - timedelta(seconds=1))

    with queue.claim(["ingest"], worker="w2") as job:
        assert job is not None
        raise TransientNetworkError("dns lookup failed")

    row = _row(conn, job_id)
    assert row["state"] == "pending"
    assert row["error_class"] == "TRANSIENT_NETWORK"


def test_reaped_dead_job_is_selectable_by_null_class_and_not_a_fail_dead_one(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    # The predicates #44 (error_class=none) and #92 (not PERMANENT_SOURCE)
    # will use, asserted as SQL since neither is merged.
    reaped = _make_running(queue, conn, kind="transcribe", video_id="r", attempts=2)
    failed = queue.enqueue("transcribe", "f")
    assert failed is not None
    _set(conn, failed, state="dead", error_class="PERMANENT_SOURCE")

    queue.reap_stale(older_than_sec=300)

    with conn.cursor() as cur:
        cur.execute("SELECT id FROM jobs WHERE state = 'dead' AND error_class IS NULL")
        assert [r[0] for r in cur.fetchall()] == [reaped]
        cur.execute(
            "SELECT id FROM jobs WHERE state = 'dead' AND kind = 'transcribe'"
            " AND error_class IS DISTINCT FROM 'PERMANENT_SOURCE'"
        )
        assert [r[0] for r in cur.fetchall()] == [reaped]


@pytest.mark.parametrize(
    ("attempts", "expected"),
    [(1, "pending"), (2, "dead")],
)
def test_reap_stale_transcribe_uses_max_attempts_transcribe_setting(
    conn: psycopg.Connection[Any], attempts: int, expected: str
) -> None:
    q = PostgresQueue(
        conn, settings=_settings(MAX_ATTEMPTS_TRANSCRIBE=2), rng=random.Random(1)
    )
    job_id = _make_running(q, conn, kind="transcribe", attempts=attempts)

    q.reap_stale(older_than_sec=300)

    assert _row(conn, job_id)["state"] == expected


def test_reap_stale_logs_one_warning_per_reaped_job(
    queue: PostgresQueue,
    conn: psycopg.Connection[Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    requeued = _make_running(
        queue, conn, video_id="a", attempts=1, locked_by="worker-a"
    )
    dead = _make_running(
        queue, conn, video_id="b", kind="analyze", attempts=4, locked_by="worker-b"
    )
    _make_running(queue, conn, video_id="c", heartbeat_age_sec=1)

    with caplog.at_level(logging.DEBUG, logger="common.queue"):
        queue.reap_stale(older_than_sec=300)

    records = [r for r in caplog.records if r.getMessage() == "queue.job_reaped"]
    assert len(records) == 2
    assert all(r.levelno == logging.WARNING for r in records)
    by_id = {r.__dict__["job_id"]: r.__dict__ for r in records}
    assert by_id[requeued]["video_id"] == "a"
    assert by_id[requeued]["kind"] == "ingest"
    assert by_id[requeued]["locked_by"] == "worker-a"
    assert by_id[requeued]["attempts"] == 1
    assert by_id[requeued]["outcome"] == "pending"
    assert by_id[dead]["video_id"] == "b"
    assert by_id[dead]["kind"] == "analyze"
    assert by_id[dead]["locked_by"] == "worker-b"
    assert by_id[dead]["attempts"] == 4
    assert by_id[dead]["outcome"] == "dead"


def test_reap_stale_is_idempotent(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = _make_running(queue, conn)
    assert queue.reap_stale(older_than_sec=300) == 1
    after_first = _row(conn, job_id)

    assert queue.reap_stale(older_than_sec=300) == 0
    assert _row(conn, job_id) == after_first


@pytest.mark.parametrize("threshold", [0, -1, -300])
def test_reap_stale_rejects_a_non_positive_threshold_and_reaps_nothing(
    queue: PostgresQueue, conn: psycopg.Connection[Any], threshold: int
) -> None:
    job_id = _make_running(queue, conn, heartbeat_age_sec=1)

    with pytest.raises(ValueError, match="older_than_sec"):
        queue.reap_stale(older_than_sec=threshold)

    assert _row(conn, job_id)["state"] == "running"


def test_reap_stale_skips_a_row_another_connection_has_locked(
    queue: PostgresQueue, conn: psycopg.Connection[Any], head_dsn: str
) -> None:
    locked = _make_running(queue, conn, video_id="locked")
    free = _make_running(queue, conn, video_id="free")
    # A hang would fail as a lock timeout error instead of blocking the suite.
    conn.execute("SET lock_timeout = '3s'")

    with psycopg.connect(head_dsn) as other:
        other.execute("SELECT id FROM jobs WHERE id = %s FOR UPDATE", (locked,))

        count = queue.reap_stale(older_than_sec=300)

        assert count == 1
        assert _row(conn, free)["state"] == "pending"
        other.rollback()

    assert _row(conn, locked)["state"] == "running"


def test_reap_stale_does_not_reap_a_job_whose_heartbeat_commits_mid_pass(
    queue: PostgresQueue, conn: psycopg.Connection[Any], head_dsn: str
) -> None:
    job_id = _make_running(queue, conn, heartbeat_age_sec=600)
    conn.execute("SET lock_timeout = '3s'")

    with psycopg.connect(head_dsn) as other:
        # The heartbeat is in flight: its row lock is held, not yet committed.
        other.execute(
            "UPDATE jobs SET heartbeat_at = now() WHERE id = %s", (job_id,)
        )
        assert queue.reap_stale(older_than_sec=300) == 0
        other.commit()

    assert queue.reap_stale(older_than_sec=300) == 0
    row = _row(conn, job_id)
    assert row["state"] == "running"
    assert row["locked_by"] == "w1"


def test_reap_stale_is_one_transaction_and_a_failure_commits_nothing(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    good = _make_running(queue, conn, video_id="good")
    bad = _make_running(queue, conn, video_id="bad")
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE FUNCTION fail_reap() RETURNS trigger AS $$
            BEGIN
                IF NEW.state <> 'running' AND NEW.video_id = 'bad' THEN
                    RAISE EXCEPTION 'boom';
                END IF;
                RETURN NEW;
            END $$ LANGUAGE plpgsql
            """
        )
        cur.execute(
            "CREATE TRIGGER fail_reap BEFORE UPDATE ON jobs "
            "FOR EACH ROW EXECUTE FUNCTION fail_reap()"
        )

    with pytest.raises(psycopg.Error):
        queue.reap_stale(older_than_sec=300)

    assert _row(conn, good)["state"] == "running"
    assert _row(conn, bad)["state"] == "running"


# ---------------------------------------------------------------------------
# retry_dead (#44)
# ---------------------------------------------------------------------------


def _make_dead(
    queue: PostgresQueue,
    conn: psycopg.Connection[Any],
    *,
    video_id: str = "v1",
    kind: str = "ingest",
    dedupe_key: str = "default",
    attempts: int = 3,
    error_class: str = "TOOL_FAILURE",
    last_error: str = "boom",
    priority: int = 5,
) -> int:
    job_id = queue.enqueue(kind, video_id, dedupe_key=dedupe_key, priority=priority)
    assert job_id is not None
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs SET state='dead', attempts=%(attempts)s,
                   error_class=%(error_class)s, last_error=%(last_error)s,
                   finished_at=now(), locked_by='w1', locked_at=now(),
                   heartbeat_at=now()
            WHERE id = %(id)s
            """,
            {
                "id": job_id,
                "attempts": attempts,
                "error_class": error_class,
                "last_error": last_error,
            },
        )
    return job_id


def test_retry_dead_puts_the_job_back_to_pending_with_the_same_id(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = _make_dead(queue, conn, attempts=3)

    outcome = queue.retry_dead(job_id)

    assert outcome.status == "retried"
    assert outcome.job is not None
    assert outcome.job.id == job_id
    assert outcome.job.attempts == 0
    assert outcome.prev_attempts == 3
    row = _row(conn, job_id)
    assert row["state"] == "pending"
    assert row["attempts"] == 0
    assert row["finished_at"] is None
    assert row["locked_by"] is None
    assert row["locked_at"] is None
    assert row["heartbeat_at"] is None
    assert row["run_after"] <= datetime.now(UTC) + timedelta(seconds=5)


def test_retry_dead_keeps_priority_payload_dedupe_key_and_created_at(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = _make_dead(queue, conn, dedupe_key="k9", priority=7)
    before = _row(conn, job_id)

    queue.retry_dead(job_id)

    after = _row(conn, job_id)
    assert after["priority"] == before["priority"] == 7
    assert after["payload"] == before["payload"]
    assert after["dedupe_key"] == before["dedupe_key"] == "k9"
    assert after["created_at"] == before["created_at"]


def test_retry_dead_keeps_error_class_and_last_error(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = _make_dead(queue, conn, error_class="BUG", last_error="caf\u00e9 traceback")

    queue.retry_dead(job_id)

    row = _row(conn, job_id)
    assert row["error_class"] == "BUG"
    assert row["last_error"] == "caf\u00e9 traceback"


def test_retry_dead_job_is_claimable_at_once_and_gets_a_full_attempt_budget(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = _make_dead(queue, conn, kind="ingest", attempts=4)  # was already at kind max

    outcome = queue.retry_dead(job_id)
    assert outcome.status == "retried"

    with queue.claim(["ingest"], worker="w2") as job:
        assert job is not None
        assert job.id == job_id
        assert job.attempts == 1
        for _ in range(3):
            raise TransientNetworkError("still broken")
    # First of 4 fresh attempts: pending, not dead yet.
    assert _row(conn, job_id)["state"] == "pending"
    assert _row(conn, job_id)["attempts"] == 1


@pytest.mark.parametrize("error_class", ["PERMANENT_SOURCE", "BUG"])
def test_retry_dead_accepts_any_error_class(
    queue: PostgresQueue, conn: psycopg.Connection[Any], error_class: str
) -> None:
    job_id = _make_dead(queue, conn, error_class=error_class)

    outcome = queue.retry_dead(job_id)

    assert outcome.status == "retried"


def test_retry_dead_does_not_touch_videos_unavailable(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    conn.execute(
        "INSERT INTO videos (video_id, unavailable) VALUES (%s, %s)", ("v1", "private")
    )
    job_id = _make_dead(queue, conn, video_id="v1")

    queue.retry_dead(job_id)

    row = conn.execute(
        "SELECT unavailable FROM videos WHERE video_id = %s", ("v1",)
    ).fetchone()
    assert row == ("private",)


def test_retry_dead_not_found_for_a_missing_id(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    outcome = queue.retry_dead(999_999)

    assert outcome == RetryOutcome("not_found")


@pytest.mark.parametrize("state", ["pending", "running", "done"])
def test_retry_dead_refuses_a_non_dead_job_and_reports_its_state(
    queue: PostgresQueue, conn: psycopg.Connection[Any], state: str
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None
    _set(conn, job_id, state=state)

    outcome = queue.retry_dead(job_id)

    assert outcome == RetryOutcome("not_dead", state=state)
    assert _row(conn, job_id)["attempts"] == 0


def test_retry_dead_refuses_a_job_superseded_by_a_newer_active_job(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    dead_id = _make_dead(queue, conn, video_id="v1", kind="ingest", dedupe_key="default")
    newer_id = queue.enqueue("ingest", "v1", dedupe_key="default")
    assert newer_id is not None
    assert newer_id > dead_id

    outcome = queue.retry_dead(dead_id)

    assert outcome == RetryOutcome("superseded", newer_job_id=newer_id)
    assert _row(conn, dead_id)["state"] == "dead"


def test_retry_dead_refuses_a_job_superseded_by_later_redone_work(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    """E.g. the auto-caption fallback ingest (#92/#97): the later job is done."""
    dead_id = _make_dead(queue, conn, video_id="v1", kind="ingest", dedupe_key="default")
    newer_id = queue.enqueue("ingest", "v1", dedupe_key="default")
    assert newer_id is not None
    _set(conn, newer_id, state="done")

    outcome = queue.retry_dead(dead_id)

    assert outcome == RetryOutcome("superseded", newer_job_id=newer_id)


def test_retry_dead_allows_reviving_the_newest_job_for_a_key(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    """Only the newest dead job for a key is retryable; an older dead sibling is not."""
    older_dead = _make_dead(queue, conn, video_id="v1", kind="ingest", dedupe_key="default")
    newer_dead = _make_dead(queue, conn, video_id="v1", kind="ingest", dedupe_key="default")
    assert newer_dead > older_dead

    assert queue.retry_dead(older_dead) == RetryOutcome(
        "superseded", newer_job_id=newer_dead
    )
    outcome = queue.retry_dead(newer_dead)
    assert outcome.status == "retried"


def test_retry_dead_holds_the_row_lock_across_its_own_check_and_write(
    head_dsn: str,
) -> None:
    """One ``UPDATE ... WHERE ... RETURNING``, not a read then a write.

    A second ``retry_dead`` of the same job, started while the first is
    mid-transaction (not yet committed), must block on the row lock rather
    than running its own check against a state the first call might still
    change - proof there is no read-then-write gap for a race to slip
    through. (This complements the two already-committed-first-wins
    scenario covered by the barrier-based concurrency tests.)
    """
    seed_conn = psycopg.connect(head_dsn, autocommit=True)
    try:
        seed_queue = PostgresQueue(seed_conn, settings=_settings())
        job_id = _make_dead(seed_queue, seed_conn)
    finally:
        seed_conn.close()

    holder = psycopg.connect(head_dsn)  # autocommit=False: transaction stays open
    try:
        holder_queue = PostgresQueue(holder, settings=_settings())
        with holder.cursor() as cur:
            cur.execute("BEGIN")
        first_outcome = holder_queue.retry_dead(job_id)
        assert first_outcome.status == "retried"
        # The row is locked by the UPDATE above; the transaction is not
        # committed yet.

        with psycopg.connect(head_dsn, autocommit=True) as observer:
            second_conn = psycopg.connect(head_dsn, autocommit=True)
            try:
                second_queue = PostgresQueue(second_conn, settings=_settings())
                result: list[RetryOutcome] = []

                def attempt() -> None:
                    result.append(second_queue.retry_dead(job_id))

                t = threading.Thread(target=attempt)
                t.start()
                _wait_until_blocked_by(
                    observer, second_conn.info.backend_pid, holder.info.backend_pid
                )
                assert t.is_alive(), "the second retry did not block on the row lock"
                holder.commit()
                t.join(timeout=10)
                assert not t.is_alive()
            finally:
                second_conn.close()

        assert result[0].status == "not_dead"
        assert result[0].state == "pending"
    finally:
        holder.rollback()
        holder.close()


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


def test_retry_dead_concurrent_retries_give_exactly_one_200_and_one_409(
    head_dsn: str,
) -> None:
    seed_conn = psycopg.connect(head_dsn, autocommit=True)
    try:
        seed_queue = PostgresQueue(seed_conn, settings=_settings())
        job_id = _make_dead(seed_queue, seed_conn)
    finally:
        seed_conn.close()


    barrier = threading.Barrier(2)
    results: list[RetryOutcome] = [None, None]  # type: ignore[list-item]

    def worker(i: int) -> None:
        with psycopg.connect(head_dsn, autocommit=True) as c:
            q = PostgresQueue(c, settings=_settings())
            barrier.wait(timeout=10)
            results[i] = q.retry_dead(job_id)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert all(not t.is_alive() for t in threads)

    statuses = sorted(r.status for r in results)
    assert statuses == ["not_dead", "retried"]

    with psycopg.connect(head_dsn, autocommit=True) as c:
        row = _row(c, job_id)
    assert row["state"] == "pending"
    assert row["attempts"] == 0


def test_retry_dead_racing_an_enqueue_leaves_exactly_one_active_job(
    head_dsn: str,
) -> None:
    """A ``retry_dead`` whose ``NOT EXISTS`` check cannot see an in-flight
    concurrent enqueue (it is not yet committed, so a plain snapshot read
    does not see it) must still never produce two active rows for the same
    key: the forced interleaving below is the one a barrier only sometimes
    reproduces (testing-guidelines.md, "Flaky or racy behaviour") - here the
    competing INSERT is held open and uncommitted while ``retry_dead``'s
    UPDATE is confirmed blocked on it (via ``pg_blocking_pids``), so the
    UPDATE's own unique-index check, not the ``NOT EXISTS`` pre-check, is
    what has to catch the conflict and turn it into ``RetryOutcome
    "superseded"`` rather than a raised ``UniqueViolation`` or a second
    active row.
    """
    seed_conn = psycopg.connect(head_dsn, autocommit=True)
    try:
        seed_queue = PostgresQueue(seed_conn, settings=_settings())
        job_id = _make_dead(seed_queue, seed_conn, video_id="v1", dedupe_key="default")
    finally:
        seed_conn.close()

    holder = psycopg.connect(head_dsn)  # autocommit=False: transaction stays open
    try:
        # Not PostgresQueue.enqueue(): it wraps its INSERT in its own
        # conn.transaction() block, which commits on exit even though
        # holder.autocommit is False. A bare execute() leaves the INSERT in
        # the connection's still-open implicit transaction, uncommitted
        # until the explicit holder.commit() below - exactly the window a
        # concurrent snapshot read (retry_dead's own NOT EXISTS) cannot see.
        new_job_id = holder.execute(
            "INSERT INTO jobs (video_id, kind, dedupe_key, payload, priority, run_after)"
            " VALUES ('v1', 'ingest', 'default', '{}', 0, now()) RETURNING id"
        ).fetchone()
        assert new_job_id is not None

        with psycopg.connect(head_dsn, autocommit=True) as observer:
            racer_conn = psycopg.connect(head_dsn, autocommit=True)
            try:
                racer_queue = PostgresQueue(racer_conn, settings=_settings())
                outcome: list[RetryOutcome] = []

                def attempt() -> None:
                    outcome.append(racer_queue.retry_dead(job_id))

                t = threading.Thread(target=attempt)
                t.start()
                _wait_until_blocked_by(
                    observer, racer_conn.info.backend_pid, holder.info.backend_pid
                )
                assert t.is_alive(), "retry_dead did not block on the uncommitted insert"
                holder.commit()
                t.join(timeout=10)
                assert not t.is_alive()
            finally:
                racer_conn.close()

        assert outcome[0] == RetryOutcome("superseded", newer_job_id=new_job_id[0])
    finally:
        holder.rollback()
        holder.close()

    with psycopg.connect(head_dsn, autocommit=True) as c:
        active = c.execute(
            "SELECT count(*) FROM jobs WHERE video_id = 'v1' AND kind = 'ingest'"
            " AND dedupe_key = 'default' AND state IN ('pending', 'running')"
        ).fetchone()
    assert active == (1,)
    with psycopg.connect(head_dsn, autocommit=True) as c:
        dead_state = c.execute(
            "SELECT state FROM jobs WHERE id = %s", (job_id,)
        ).fetchone()
    assert dead_state == ("dead",)


# ---------------------------------------------------------------------------
# started_at (#60)
# ---------------------------------------------------------------------------


def _backdate_start(conn: psycopg.Connection[Any], job_id: int) -> datetime:
    """Move ``started_at`` into the past, so a later change is visible without sleeping."""
    past = datetime(2020, 1, 1, tzinfo=UTC)
    _set(conn, job_id, started_at=past)
    return past


def test_a_new_job_has_no_start_time_until_it_is_claimed(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None
    assert _row(conn, job_id)["started_at"] is None


def test_claim_sets_started_at_together_with_locked_at_and_exposes_it_on_the_job(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None

    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        assert job.started_at is not None
        assert job.started_at == job.locked_at

    assert _row(conn, job_id)["started_at"] is not None


def test_started_at_survives_done(queue: PostgresQueue, conn: psycopg.Connection[Any]) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None
    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        past = _backdate_start(conn, job_id)

    row = _row(conn, job_id)
    assert row["state"] == "done"
    assert row["started_at"] == past


def test_started_at_survives_dead(queue: PostgresQueue, conn: psycopg.Connection[Any]) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None
    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        past = _backdate_start(conn, job_id)
        raise PermanentSourceError("removed")

    row = _row(conn, job_id)
    assert row["state"] == "dead"
    assert row["started_at"] == past


def test_started_at_survives_retry_and_a_reclaim_moves_it_to_the_latest_claim(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None
    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        past = _backdate_start(conn, job_id)
        raise TransientNetworkError("dns")

    row = _row(conn, job_id)
    assert row["state"] == "pending"
    assert row["started_at"] == past

    _set(conn, job_id, run_after=datetime(2020, 1, 1, tzinfo=UTC))
    with queue.claim(["ingest"], worker="w2") as job:
        assert job is not None
        assert job.attempts == 2
        assert job.started_at is not None
        assert job.started_at > past


def test_started_at_survives_defer_and_cancellation(
    queue: PostgresQueue, conn: psycopg.Connection[Any]
) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None
    with queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        past = _backdate_start(conn, job_id)
        raise Defer(until=datetime.now(UTC) + timedelta(hours=1))
    assert _row(conn, job_id)["started_at"] == past

    _set(conn, job_id, run_after=datetime(2020, 1, 1, tzinfo=UTC))
    with pytest.raises(Cancelled), queue.claim(["ingest"], worker="w1") as job:
        assert job is not None
        past = _backdate_start(conn, job_id)
        raise Cancelled()
    assert _row(conn, job_id)["started_at"] == past


def test_started_at_survives_reap(queue: PostgresQueue, conn: psycopg.Connection[Any]) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None
    past = datetime(2020, 1, 1, tzinfo=UTC)
    _set(
        conn,
        job_id,
        state="running",
        locked_by="w1",
        locked_at=past,
        started_at=past,
        heartbeat_at=past,
        attempts=1,
    )

    assert queue.reap_stale(older_than_sec=300) == 1

    row = _row(conn, job_id)
    assert row["state"] == "pending"
    assert row["started_at"] == past


def test_started_at_survives_retry_dead(queue: PostgresQueue, conn: psycopg.Connection[Any]) -> None:
    job_id = queue.enqueue("ingest", "v1")
    assert job_id is not None
    past = datetime(2020, 1, 1, tzinfo=UTC)
    _set(conn, job_id, state="dead", started_at=past, finished_at=past)

    assert queue.retry_dead(job_id).status == "retried"

    assert _row(conn, job_id)["started_at"] == past
