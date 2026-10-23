"""The orphan pass end to end (issue #160): real Postgres and queue, the real
``AnthropicSummarizer`` on a ``FakeTransport``. No network, no real key."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest

from adapters.summarize.anthropic import AnthropicSummarizer
from common.errors import BugError
from common.queue import PostgresQueue
from common.repo.llm_batches import insert_checkpoint
from services.analyzer.orphans import ORPHAN_EXPIRED_AFTER, run_pass
from tests.adapters.summarize.test_anthropic import (
    API,
    BATCH_ID,
    FakeTransport,
    fixture_response,
    settings,
)

pytestmark = pytest.mark.integration

DEAD_VIDEO = "aaaaaaaaaaa"
PENDING_VIDEO = "bbbbbbbbbbb"
OTHER_BATCH = "msgbatch_PENDING000000000000"


def batch_rows(conn: psycopg.Connection[Any]) -> list[tuple[int, str]]:
    conn.rollback()
    return [
        (int(r[0]), str(r[1]))
        for r in conn.execute("SELECT job_id, batch_id FROM llm_batches ORDER BY job_id").fetchall()
    ]


def test_a_dead_lettered_jobs_batch_is_cancelled_and_a_pending_jobs_is_not(
    conn: psycopg.Connection[Any], head_dsn: str
) -> None:
    for video in (DEAD_VIDEO, PENDING_VIDEO):
        conn.execute("INSERT INTO videos (video_id) VALUES (%s)", (video,))
    queue = PostgresQueue(conn)
    dead_id = queue.enqueue("analyze", DEAD_VIDEO, dedupe_key="v1:anthropic")
    pending_id = queue.enqueue("analyze", PENDING_VIDEO, dedupe_key="v1:anthropic")
    assert dead_id is not None and pending_id is not None
    conn.commit()

    # Claim the first job, record its in-progress batch, and fail it for good.
    with queue.claim(["analyze"], worker="w1", dedupe_key="v1:anthropic") as job:
        assert job is not None and job.id == dead_id
        insert_checkpoint(conn, job.id, "roster", BATCH_ID, "sha")
        conn.commit()
        raise BugError("non-retryable")  # claim() dead-letters it through _fail
    insert_checkpoint(conn, pending_id, "roster", OTHER_BATCH, "sha")
    conn.commit()
    state = conn.execute("SELECT state FROM jobs WHERE id = %s", (dead_id,)).fetchone()
    assert state == ("dead",)
    conn.rollback()

    transport = FakeTransport(fixture_response("batch_in_progress.json"))
    canceller = AnthropicSummarizer(settings(), batch=True, transport=transport)

    counts = run_pass(
        lambda: psycopg.connect(head_dsn),
        canceller,
        limit=5,
        expired_after=ORPHAN_EXPIRED_AFTER,
        stop=lambda: False,
    )

    assert counts is not None and counts.cancelled == 1
    assert [(r.method, r.url) for r in transport.requests] == [
        ("POST", f"{API}/v1/messages/batches/{BATCH_ID}/cancel")
    ]
    assert batch_rows(conn) == [(pending_id, OTHER_BATCH)]
