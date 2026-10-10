"""``PostgresQueue.claim(dedupe_key=...)`` and the worker / analyzer wiring (issue #100)."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from pydantic import SecretStr

from common import queue as queue_module
from common.config import Settings
from common.queue import PostgresQueue
from tests.common.repo.conftest import conn, head_dsn

__all__ = ["conn", "head_dsn"]

pytestmark = pytest.mark.integration


def _queue(conn: psycopg.Connection[Any]) -> PostgresQueue:
    return PostgresQueue(conn, settings=Settings(DATABASE_URL=SecretStr("postgresql://u:p@h/db")))


def _job(
    conn: psycopg.Connection[Any],
    video_id: str,
    kind: str = "analyze",
    key: str = "default",
    priority: int = 0,
    run_after: str = "now()",
) -> int:
    row = conn.execute(
        f"INSERT INTO jobs (video_id, kind, dedupe_key, priority, run_after) "
        f"VALUES (%s, %s, %s, %s, {run_after}) RETURNING id",
        (video_id, kind, key, priority),
    ).fetchone()
    assert row is not None
    conn.commit()
    return int(row[0])


def _state(conn: psycopg.Connection[Any], job_id: int) -> tuple[str, int]:
    row = conn.execute("SELECT state, attempts FROM jobs WHERE id = %s", (job_id,)).fetchone()
    conn.commit()
    assert row is not None
    return (row[0], row[1])


def _claim(
    conn: psycopg.Connection[Any], kinds: list[str], **kwargs: Any
) -> tuple[int, str] | None:
    with _queue(conn).claim(kinds, worker="w", **kwargs) as job:
        return None if job is None else (job.id, job.dedupe_key)


def test_a_key_claims_only_the_job_with_exactly_that_key(conn: psycopg.Connection[Any]) -> None:
    v1 = _job(conn, "aaaaaaaaaaa", key="v1:ollama", priority=10)
    v2 = _job(conn, "bbbbbbbbbbb", key="v2:ollama")
    other = _job(conn, "ccccccccccc", key="v2:anthropic")

    assert _claim(conn, ["analyze"], dedupe_key="v2:ollama") == (v2, "v2:ollama")
    assert _claim(conn, ["analyze"], dedupe_key="v2:ollama") is None
    assert _state(conn, v1) == ("pending", 0)
    assert _state(conn, other) == ("pending", 0)


def test_without_a_key_the_highest_priority_job_is_claimed_as_before(
    conn: psycopg.Connection[Any],
) -> None:
    v1 = _job(conn, "aaaaaaaaaaa", key="v1:ollama", priority=10)
    _job(conn, "bbbbbbbbbbb", key="v2:ollama")
    _job(conn, "ccccccccccc", key="v2:anthropic")

    assert _claim(conn, ["analyze"]) == (v1, "v1:ollama")


def test_an_explicit_none_is_the_same_as_no_key(conn: psycopg.Connection[Any]) -> None:
    v1 = _job(conn, "aaaaaaaaaaa", key="v1:ollama", priority=10)
    _job(conn, "bbbbbbbbbbb", key="v2:ollama")

    assert _claim(conn, ["analyze"], dedupe_key=None) == (v1, "v1:ollama")


def test_the_key_filters_every_kind_in_the_list(conn: psycopg.Connection[Any]) -> None:
    ingest = _job(conn, "aaaaaaaaaaa", kind="ingest", priority=10)  # key "default"
    analyze = _job(conn, "bbbbbbbbbbb", key="v2:ollama")

    assert _claim(conn, ["ingest", "analyze"], dedupe_key="v2:ollama") == (analyze, "v2:ollama")
    assert _claim(conn, ["ingest", "analyze"], dedupe_key="v2:ollama") is None
    assert _state(conn, ingest) == ("pending", 0)


@pytest.mark.parametrize(
    "stored",
    ["v2_beta:ollama", "v2:ollama", "v2:ollama-large", "xv2:ollama", "V2:OLLAMA"],
)
@pytest.mark.parametrize("asked", ["", "v2", "v2_%", "v2%", "%:ollama", "v2:ollama%", "v_:ollama"])
def test_the_match_is_exact_not_a_prefix_or_a_pattern(
    conn: psycopg.Connection[Any], stored: str, asked: str
) -> None:
    job = _job(conn, "aaaaaaaaaaa", key=stored)

    assert _claim(conn, ["analyze"], dedupe_key=asked) is None
    assert _state(conn, job) == ("pending", 0)


def test_a_key_that_looks_like_sql_is_only_a_value(conn: psycopg.Connection[Any]) -> None:
    job = _job(conn, "aaaaaaaaaaa", key="v2:ollama")

    assert _claim(conn, ["analyze"], dedupe_key="x' OR '1'='1") is None
    assert _claim(conn, ["analyze"], dedupe_key="x'; DROP TABLE jobs; --") is None
    assert _state(conn, job) == ("pending", 0)


def test_among_jobs_with_the_key_the_highest_priority_goes_first(
    conn: psycopg.Connection[Any],
) -> None:
    low = _job(conn, "aaaaaaaaaaa", key="v2:ollama", priority=-5)
    high = _job(conn, "bbbbbbbbbbb", key="v2:ollama", priority=10)
    _job(conn, "ccccccccccc", key="v1:ollama", priority=100)

    assert _claim(conn, ["analyze"], dedupe_key="v2:ollama") == (high, "v2:ollama")
    assert _claim(conn, ["analyze"], dedupe_key="v2:ollama") == (low, "v2:ollama")


def test_the_key_does_not_override_run_after(conn: psycopg.Connection[Any]) -> None:
    _job(conn, "aaaaaaaaaaa", key="v2:ollama", run_after="now() + interval '1 hour'")

    assert _claim(conn, ["analyze"], dedupe_key="v2:ollama") is None


def test_only_the_keyed_statement_mentions_the_key() -> None:
    assert "dedupe_key" not in queue_module._CLAIM_SQL
    assert "AND dedupe_key = %(dedupe_key)s" in queue_module._CLAIM_SQL_KEYED
