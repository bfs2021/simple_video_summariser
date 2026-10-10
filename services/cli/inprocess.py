"""The in-process ``JobQueue`` behind ``ytdigest run`` (issue #31).

It keeps jobs in memory and never touches the ``jobs`` table, so a CLI run can
never claim, duplicate or strand a job that belongs to the running workers.
There is no retry: an exception raised inside ``claim`` goes straight back to
the caller, and the job is not put back.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from common.queue import Job


class InProcessQueue:
    """``JobQueue`` port over a list: FIFO, deduplicated on ``(kind, dedupe_key)``.

    Like ``PostgresQueue`` it treats a job as active while it is pending or
    running, so a finished job does not block the same kind and key again.
    Priority is stored on the job (so handlers pass it on) but does not
    reorder anything: the CLI runs one video, in the order jobs were enqueued.
    """

    def __init__(self) -> None:
        self._next_id = 1
        self._pending: list[Job] = []
        self._running: list[Job] = []

    @property
    def pending(self) -> int:
        return len(self._pending)

    def enqueue(
        self,
        kind: str,
        video_id: str,
        *,
        dedupe_key: str = "default",
        payload: dict[str, Any] | None = None,
        priority: int = 0,
        run_after: datetime | None = None,
    ) -> int | None:
        for active in (*self._pending, *self._running):
            if (active.kind, active.dedupe_key) == (kind, dedupe_key):
                return None
        now = datetime.now(UTC)
        job = Job(
            id=self._next_id,
            video_id=video_id,
            kind=kind,
            dedupe_key=dedupe_key,
            state="pending",
            priority=priority,
            payload=dict(payload) if payload is not None else {},
            attempts=0,
            last_error=None,
            error_class=None,
            run_after=run_after if run_after is not None else now,
            locked_by=None,
            locked_at=None,
            heartbeat_at=None,
            finished_at=None,
            created_at=now,
        )
        self._next_id += 1
        self._pending.append(job)
        return job.id

    @contextmanager
    def claim(
        self, kinds: Sequence[str], *, worker: str, dedupe_key: str | None = None
    ) -> Iterator[Job | None]:
        index = next(
            (
                i
                for i, job in enumerate(self._pending)
                if job.kind in kinds and (dedupe_key is None or job.dedupe_key == dedupe_key)
            ),
            None,
        )
        if index is None:
            yield None
            return
        now = datetime.now(UTC)
        job = replace(
            self._pending.pop(index),
            state="running",
            attempts=1,
            locked_by=worker,
            locked_at=now,
            heartbeat_at=now,
        )
        self._running.append(job)
        try:
            yield job
        finally:
            # Done or failed, the job leaves the queue: nothing is retried.
            self._running.remove(job)

    def heartbeat(self, job_id: int, worker: str) -> bool:
        return True

    def reap_stale(self, older_than_sec: int | None = None) -> int:
        return 0

    def leftover(self) -> list[tuple[str, str]]:
        """``(kind, video_id)`` of every job still pending, oldest first."""
        return [(job.kind, job.video_id) for job in self._pending]
