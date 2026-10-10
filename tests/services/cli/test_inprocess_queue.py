"""The in-process ``JobQueue`` the CLI runs its jobs on (issue #31)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from common.errors import Defer
from common.queue import Job
from services.cli.inprocess import InProcessQueue

VID = "iG9CE55wbtY"


def claim_one(
    queue: InProcessQueue, kinds: tuple[str, ...] = ("ingest", "transcribe", "analyze")
) -> Job | None:
    with queue.claim(kinds, worker="cli") as job:
        return job


def test_enqueue_returns_distinct_ids_and_claim_hands_back_the_job_fields() -> None:
    queue = InProcessQueue()
    first = queue.enqueue("ingest", VID, priority=10, payload={"origin": "adhoc"})
    second = queue.enqueue("analyze", VID, dedupe_key="v1:ollama", priority=10)
    assert first is not None and second is not None and first != second

    job = claim_one(queue)
    assert job is not None
    assert (job.id, job.video_id, job.kind, job.priority) == (first, VID, "ingest", 10)
    assert job.dedupe_key == "default"
    assert job.payload == {"origin": "adhoc"}
    assert job.state == "running"
    assert job.attempts == 1


def test_jobs_run_in_fifo_order_whatever_their_priority() -> None:
    queue = InProcessQueue()
    queue.enqueue("ingest", VID, priority=0)
    queue.enqueue("transcribe", VID, priority=99)
    queue.enqueue("analyze", VID, dedupe_key="k", priority=10)
    kinds = []
    while (job := claim_one(queue)) is not None:
        kinds.append(job.kind)
    assert kinds == ["ingest", "transcribe", "analyze"]


def test_claim_returns_none_when_the_queue_is_empty() -> None:
    assert claim_one(InProcessQueue()) is None


def test_claim_only_hands_out_the_kinds_asked_for() -> None:
    queue = InProcessQueue()
    queue.enqueue("ingest", VID)
    assert claim_one(queue, ("analyze",)) is None
    job = claim_one(queue, ("ingest",))
    assert job is not None and job.kind == "ingest"


def test_a_claimed_job_is_not_handed_out_again_after_a_clean_exit() -> None:
    queue = InProcessQueue()
    queue.enqueue("ingest", VID)
    assert claim_one(queue) is not None
    assert claim_one(queue) is None


def test_a_duplicate_kind_and_dedupe_key_returns_none_while_the_first_is_pending() -> None:
    queue = InProcessQueue()
    assert queue.enqueue("analyze", VID, dedupe_key="v1:ollama") is not None
    assert queue.enqueue("analyze", VID, dedupe_key="v1:ollama") is None
    assert queue.enqueue("analyze", VID, dedupe_key="v1:anthropic") is not None
    assert queue.enqueue("ingest", VID, dedupe_key="v1:ollama") is not None


def test_a_duplicate_of_the_running_job_returns_none_like_postgres() -> None:
    queue = InProcessQueue()
    queue.enqueue("analyze", VID, dedupe_key="k")
    with queue.claim(["analyze"], worker="cli") as job:
        assert job is not None
        assert queue.enqueue("analyze", VID, dedupe_key="k") is None


def test_a_finished_job_no_longer_blocks_the_same_kind_and_key() -> None:
    queue = InProcessQueue()
    queue.enqueue("analyze", VID, dedupe_key="k")
    assert claim_one(queue) is not None
    assert queue.enqueue("analyze", VID, dedupe_key="k") is not None


def test_a_raising_handler_propagates_and_the_job_is_not_retried() -> None:
    queue = InProcessQueue()
    queue.enqueue("ingest", VID)
    with pytest.raises(RuntimeError, match="boom"), queue.claim(["ingest"], worker="cli"):
        raise RuntimeError("boom")
    assert claim_one(queue) is None


def test_defer_propagates_and_the_job_is_not_put_back() -> None:
    queue = InProcessQueue()
    queue.enqueue("ingest", VID)
    until = datetime(2026, 10, 1, tzinfo=UTC)
    with pytest.raises(Defer) as excinfo, queue.claim(["ingest"], worker="cli"):
        raise Defer(until)
    assert excinfo.value.until == until
    assert claim_one(queue) is None


def test_heartbeat_is_true_and_reap_stale_is_zero() -> None:
    queue = InProcessQueue()
    assert queue.heartbeat(1, "cli") is True
    assert queue.reap_stale() == 0
    assert queue.reap_stale(60) == 0


def test_pending_counts_jobs_not_yet_claimed() -> None:
    queue = InProcessQueue()
    assert queue.pending == 0
    queue.enqueue("ingest", VID)
    queue.enqueue("analyze", VID)
    assert queue.pending == 2
    claim_one(queue)
    assert queue.pending == 1


def test_a_dedupe_key_claims_only_a_job_with_exactly_that_key() -> None:
    queue = InProcessQueue()
    queue.enqueue("analyze", VID, dedupe_key="v1:ollama")
    second = queue.enqueue("analyze", "BZiu46G4ukc", dedupe_key="v2:ollama")

    with queue.claim(("analyze",), worker="cli", dedupe_key="v2:ollama") as job:
        assert job is not None and job.id == second
    assert claim_one(queue, ("analyze",)) is not None  # the v1 job was left pending
