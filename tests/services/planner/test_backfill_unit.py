"""The ``channel_backfill`` per-run bound arithmetic, no Docker (issue #115)."""

from __future__ import annotations

import pytest

from adapters.youtube.catalog import MAX_CATALOG_LIMIT
from common.backfill import MAX_BACKFILL_REQUEST_LIMIT
from common.config import Settings
from services.planner.backfill import (
    BACKFILL_ENQUEUE_PER_RUN,
    BACKFILL_MAX_PENDING,
    BACKFILL_REQUEST_LIST_TIMEOUT_SEC,
    BACKFILL_REQUEST_MAX_ATTEMPTS,
    run_allowance,
)
from services.planner.main import LOCK_KEYS, build_registry


def test_the_named_constants_have_the_groomed_values() -> None:
    assert BACKFILL_ENQUEUE_PER_RUN == 500
    assert BACKFILL_MAX_PENDING == 1000
    assert BACKFILL_REQUEST_LIST_TIMEOUT_SEC == 300
    assert BACKFILL_REQUEST_MAX_ATTEMPTS == 3


def test_the_request_cap_equals_the_catalog_adapters_cap() -> None:
    assert MAX_BACKFILL_REQUEST_LIMIT == MAX_CATALOG_LIMIT == 5000


@pytest.mark.parametrize(
    ("pending", "expected"),
    [
        (0, 500),  # empty queue: the per-run cap
        (499, 500),  # budget 501 >= 500
        (500, 500),  # budget exactly 500
        (501, 499),  # one under the per-run cap
        (999, 1),
        (1000, 0),  # P exactly at BACKFILL_MAX_PENDING
        (1001, 0),  # P past it never goes negative
        (50_000, 0),
    ],
)
def test_run_allowance_is_the_smaller_of_the_run_cap_and_the_free_budget(
    pending: int, expected: int
) -> None:
    assert run_allowance(pending) == expected


def test_the_task_is_registered_with_its_own_lock_and_interval() -> None:
    settings = Settings(DATABASE_URL="postgresql://u:p@127.0.0.1:1/db", BACKFILL_INTERVAL_SEC=77)  # type: ignore[arg-type]

    tasks = {t.name: t for t in build_registry(settings, lambda: (_ for _ in ()).throw(AssertionError))}

    assert tasks["channel_backfill"].interval_sec == 77
    assert len(set(LOCK_KEYS.values())) == len(LOCK_KEYS)
    assert "channel_backfill" in LOCK_KEYS
    assert set(tasks) <= set(LOCK_KEYS)
