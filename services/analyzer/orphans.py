"""Cancel Anthropic batches left behind by jobs that are over (issue #160).

A job that dead-letters (or finishes without ``clear()`` running) can leave a
row in ``llm_batches`` whose batch keeps running, and billing, on Anthropic's
side. ``ANTHROPIC_API_KEY`` is on the analyzer only (architecture.md D6), so
the analyzer is the one process that can cancel it: a daemon thread runs one
pass at startup and then one every ``interval_sec``.

A pass lists up to ``limit`` rows of ``dead`` / ``done`` jobs and, for each:

* older than ``expired_after``: delete the row, no API call (Anthropic has
  expired the batch by then);
* otherwise ``cancel_orphan_batch``: ``cancelled`` and ``not_cancellable``
  delete the row, ``retry_later`` keeps it for the next pass.

Rows of ``pending`` and ``running`` jobs are never returned by the repo
(#89's resume depends on them). The pass uses the summarizer only through the
``OrphanBatchCanceller`` protocol and opens its own connection per pass.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol

import psycopg
import structlog

from common.models import OrphanBatchCanceller
from common.repo import llm_batches
from common.repo.llm_batches import OrphanBatch

_log = structlog.get_logger(__name__)

#: Production defaults; every one is injectable (``main(orphan_options=...)``).
ORPHAN_LIMIT = 20
ORPHAN_INTERVAL_SEC = 600
ORPHAN_EXPIRED_AFTER = timedelta(hours=24)

THREAD_NAME = "analyzer-orphans"


class OrphanStore(Protocol):
    def list(self, limit: int, expired_after: timedelta) -> Sequence[OrphanBatch]: ...

    def delete(self, row: OrphanBatch) -> bool: ...


class PostgresOrphanStore:
    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self._conn = conn

    def list(self, limit: int, expired_after: timedelta) -> Sequence[OrphanBatch]:
        return llm_batches.list_orphans(self._conn, limit=limit, expired_after=expired_after)

    def delete(self, row: OrphanBatch) -> bool:
        return llm_batches.delete_orphan(self._conn, row.job_id, row.step_key, row.batch_id)


@dataclass(slots=True)
class PassCounts:
    cancelled: int = 0
    not_cancellable: int = 0
    retry_later: int = 0
    expired: int = 0

    @property
    def total(self) -> int:
        return self.cancelled + self.not_cancellable + self.retry_later + self.expired


def handle_orphans(
    store: OrphanStore,
    canceller: OrphanBatchCanceller,
    *,
    limit: int,
    expired_after: timedelta,
    stop: Callable[[], bool],
) -> PassCounts:
    """One pass over the store. ``stop`` is checked before every row."""
    counts = PassCounts()
    for row in store.list(limit, expired_after):
        if stop():
            break
        ids = {"job_id": row.job_id, "step_key": row.step_key, "batch_id": row.batch_id}
        if row.expired:
            store.delete(row)
            counts.expired += 1
            _log.info("anthropic_orphan_batch_expired", **ids)
            continue
        outcome = canceller.cancel_orphan_batch(row.batch_id)
        if outcome == "retry_later":
            counts.retry_later += 1
            _log.warning("anthropic_orphan_batch_retry_later", **ids)
            continue
        store.delete(row)
        if outcome == "cancelled":
            counts.cancelled += 1
            _log.info("anthropic_orphan_batch_cancelled", **ids)
        else:
            counts.not_cancellable += 1
            _log.info("anthropic_orphan_batch_not_cancellable", **ids)
    if counts.total:
        _log.info(
            "anthropic_orphan_pass",
            cancelled=counts.cancelled,
            not_cancellable=counts.not_cancellable,
            retry_later=counts.retry_later,
            expired=counts.expired,
        )
    return counts


def run_pass(
    connect: Callable[[], Any],
    canceller: OrphanBatchCanceller,
    *,
    limit: int,
    expired_after: timedelta,
    stop: Callable[[], bool],
    store_factory: Callable[[Any], OrphanStore] = PostgresOrphanStore,
) -> PassCounts | None:
    """One pass on a connection opened for it and closed after it.

    A database (or any other) error is logged and ``None`` returned: the next
    pass opens a fresh connection. It never raises.
    """
    conn = None
    try:
        conn = connect()
        return handle_orphans(
            store_factory(conn), canceller, limit=limit, expired_after=expired_after, stop=stop
        )
    except Exception as exc:  # noqa: BLE001 - a failed pass must not end the thread
        _log.warning("analyzer.orphan_pass_failed", error_type=type(exc).__name__)
        return None
    finally:
        if conn is not None:
            with suppress(Exception):
                conn.close()


class _Waitable(Protocol):
    def is_set(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...


def run_loop(run_one: Callable[[], object], *, interval_sec: float, stop: _Waitable) -> None:
    """A pass at once, then one per ``interval_sec``; ``stop`` wakes the wait at once."""
    while not stop.is_set():
        try:
            run_one()
        except Exception as exc:  # noqa: BLE001 - belt and braces: the thread must survive
            _log.error("analyzer.orphan_pass_crashed", error_type=type(exc).__name__)
        if stop.wait(interval_sec):
            return


def start_orphan_thread(
    connect: Callable[[], Any],
    canceller: OrphanBatchCanceller,
    stop: threading.Event,
    *,
    interval_sec: float = ORPHAN_INTERVAL_SEC,
    limit: int = ORPHAN_LIMIT,
    expired_after: timedelta = ORPHAN_EXPIRED_AFTER,
    store_factory: Callable[[Any], OrphanStore] = PostgresOrphanStore,
) -> threading.Thread:
    """Start the daemon thread running ``run_loop`` over ``run_pass``."""

    def run_one() -> object:
        return run_pass(
            connect,
            canceller,
            limit=limit,
            expired_after=expired_after,
            stop=stop.is_set,
            store_factory=store_factory,
        )

    thread = threading.Thread(
        target=run_loop,
        args=(run_one,),
        kwargs={"interval_sec": interval_sec, "stop": stop},
        name=THREAD_NAME,
        daemon=True,
    )
    thread.start()
    return thread
