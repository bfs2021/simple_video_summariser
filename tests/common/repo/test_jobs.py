"""Tests for common/repo/jobs.py (issues #28, #39, #44)."""

from __future__ import annotations

import itertools

import psycopg
import pytest

from common.repo.jobs import (
    ActiveJob,
    JobRow,
    active_job,
    get_job,
    latest_job,
    latest_job_state,
    list_jobs,
    queue_depth,
)

pytestmark = pytest.mark.integration

VID = "abc12345678"

_key_counter = itertools.count()


def add_job(
    conn: psycopg.Connection,
    kind: str,
    state: str,
    *,
    key: str | None = None,
    video_id: str = VID,
    error_class: str | None = None,
    last_error: str | None = None,
    priority: int = 0,
) -> int:
    """Insert one ``jobs`` row.

    ``key`` defaults to a fresh value per call, so active (pending/running)
    jobs of the same video and kind never collide on ``jobs_active_uniq``
    unless a test asks for that collision explicitly.
    """
    if key is None:
        key = f"k{next(_key_counter)}"
    row = conn.execute(
        "INSERT INTO jobs (video_id, kind, dedupe_key, state, error_class, last_error, priority)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (video_id, kind, key, state, error_class, last_error, priority),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _ids(rows: list[JobRow]) -> list[int]:
    return [row.id for row in rows]


def test_latest_job_state_is_none_when_the_video_has_no_job_of_that_kind(
    conn: psycopg.Connection,
) -> None:
    add_job(conn, "analyze", "done")

    assert latest_job_state(conn, VID, "transcribe") is None
    assert latest_job_state(conn, "zzzzzzzzzzz", "analyze") is None


@pytest.mark.parametrize("state", ["pending", "running", "done", "dead"])
def test_latest_job_state_reports_each_state(conn: psycopg.Connection, state: str) -> None:
    add_job(conn, "transcribe", state)

    assert latest_job_state(conn, VID, "transcribe") == state


def test_a_pending_job_created_after_a_dead_one_is_the_latest(conn: psycopg.Connection) -> None:
    add_job(conn, "transcribe", "dead", key="a")
    add_job(conn, "transcribe", "pending", key="b")

    assert latest_job_state(conn, VID, "transcribe") == "pending"


def test_a_dead_job_created_after_a_done_one_is_the_latest(conn: psycopg.Connection) -> None:
    add_job(conn, "transcribe", "done", key="a")
    add_job(conn, "transcribe", "dead", key="b")

    assert latest_job_state(conn, VID, "transcribe") == "dead"


def test_ties_on_created_at_are_broken_by_id(conn: psycopg.Connection) -> None:
    # Both rows are inserted in this one transaction, so now() is identical.
    add_job(conn, "transcribe", "pending", key="a")
    add_job(conn, "transcribe", "dead", key="b")
    distinct = conn.execute("SELECT count(DISTINCT created_at) FROM jobs").fetchone()
    assert distinct == (1,)

    assert latest_job_state(conn, VID, "transcribe") == "dead"


def test_only_the_asked_kind_counts(conn: psycopg.Connection) -> None:
    add_job(conn, "transcribe", "dead")
    add_job(conn, "analyze", "pending")

    assert latest_job_state(conn, VID, "transcribe") == "dead"
    assert latest_job_state(conn, VID, "analyze") == "pending"


def test_queue_depth_of_an_empty_jobs_table_is_empty(conn: psycopg.Connection) -> None:
    assert queue_depth(conn) == {}


def test_queue_depth_counts_each_kind_and_state_group(conn: psycopg.Connection) -> None:
    add_job(conn, "ingest", "done", key="a")
    add_job(conn, "ingest", "done", key="b")
    add_job(conn, "ingest", "done", key="c")
    add_job(conn, "ingest", "pending", key="d")
    add_job(conn, "transcribe", "running", key="e")
    add_job(conn, "transcribe", "dead", key="f")
    add_job(conn, "transcribe", "dead", key="g")
    add_job(conn, "analyze", "pending", key="h")
    add_job(conn, "notify", "pending", key="i")
    add_job(conn, "analyze", "paused", key="j")

    assert queue_depth(conn) == {
        ("ingest", "done"): 3,
        ("ingest", "pending"): 1,
        ("transcribe", "running"): 1,
        ("transcribe", "dead"): 2,
        ("analyze", "pending"): 1,
        ("notify", "pending"): 1,
        ("analyze", "paused"): 1,
    }


def test_queue_depth_runs_inside_a_read_only_transaction(conn: psycopg.Connection) -> None:
    add_job(conn, "ingest", "pending")
    conn.commit()
    conn.read_only = True

    assert queue_depth(conn) == {("ingest", "pending"): 1}


def test_latest_job_is_none_without_a_job_of_that_kind(conn: psycopg.Connection) -> None:
    add_job(conn, "analyze", "done")

    assert latest_job(conn, VID, "ingest") is None


def test_latest_job_returns_the_newest_jobs_id_and_state(conn: psycopg.Connection) -> None:
    add_job(conn, "ingest", "dead")
    newest = add_job(conn, "ingest", "pending")
    add_job(conn, "analyze", "running")

    assert latest_job(conn, VID, "ingest") == (newest, "pending")


# ---------------------------------------------------------------------------
# list_jobs (#44)
# ---------------------------------------------------------------------------


def test_list_jobs_with_no_filters_orders_newest_first(conn: psycopg.Connection) -> None:
    first = add_job(conn, "ingest", "pending")
    second = add_job(conn, "transcribe", "dead")
    third = add_job(conn, "analyze", "done")

    rows = list_jobs(conn, limit=50)

    assert _ids(rows) == [third, second, first]


def test_list_jobs_default_page_size_is_50(conn: psycopg.Connection) -> None:
    for i in range(60):
        add_job(conn, "ingest", "pending", key=str(i))

    rows = list_jobs(conn, limit=50)

    assert len(rows) == 50


@pytest.mark.parametrize("field", ["state", "kind", "error_class"])
def test_list_jobs_filters_combine_with_and(conn: psycopg.Connection, field: str) -> None:
    match = add_job(conn, "transcribe", "dead", error_class="TOOL_FAILURE")
    add_job(conn, "ingest", "dead", error_class="TOOL_FAILURE")
    add_job(conn, "transcribe", "pending", error_class="TOOL_FAILURE")
    add_job(conn, "transcribe", "dead", error_class="BUG")

    rows = list_jobs(
        conn, state="dead", kind="transcribe", error_class="TOOL_FAILURE", limit=50
    )

    assert _ids(rows) == [match]


def test_list_jobs_error_class_none_matches_null_rows_only(conn: psycopg.Connection) -> None:
    reaped = add_job(conn, "transcribe", "dead", error_class=None)
    add_job(conn, "transcribe", "dead", error_class="BUG")

    rows = list_jobs(conn, error_class=None, limit=50)

    assert _ids(rows) == [reaped]


def test_list_jobs_without_error_class_filter_returns_both_null_and_set(
    conn: psycopg.Connection,
) -> None:
    a = add_job(conn, "transcribe", "dead", error_class=None)
    b = add_job(conn, "transcribe", "dead", error_class="BUG")

    rows = list_jobs(conn, limit=50)

    assert set(_ids(rows)) == {a, b}


def test_list_jobs_before_id_returns_only_older_rows(conn: psycopg.Connection) -> None:
    first = add_job(conn, "ingest", "pending")
    second = add_job(conn, "ingest", "pending")
    add_job(conn, "ingest", "pending")

    rows = list_jobs(conn, before_id=second + 1, limit=50)

    assert _ids(rows) == [second, first]


def test_list_jobs_pagination_boundaries(conn: psycopg.Connection) -> None:
    # list_jobs itself takes a literal limit (the route decides to ask for
    # limit + 1 to detect a next page; see test_routes_ops.py).
    assert list_jobs(conn, limit=5) == []

    ids = [add_job(conn, "ingest", "pending") for _ in range(5)]
    rows = list_jobs(conn, limit=5)
    assert _ids(rows) == list(reversed(ids))

    add_job(conn, "ingest", "pending")
    rows = list_jobs(conn, limit=5)
    assert len(rows) == 5  # the 6th row needs limit=6 to be seen
    rows_plus_one = list_jobs(conn, limit=6)
    assert len(rows_plus_one) == 6


def test_list_jobs_walking_pages_with_the_cursor_covers_every_row_once(
    conn: psycopg.Connection,
) -> None:
    ids = [add_job(conn, "ingest", "pending") for _ in range(7)]

    seen: list[int] = []
    before_id: int | None = None
    for _ in range(10):
        page = list_jobs(conn, before_id=before_id, limit=3)
        if not page:
            break
        seen.extend(_ids(page))
        before_id = page[-1].id

    assert seen == list(reversed(ids))


def test_list_jobs_new_rows_inserted_mid_walk_never_shift_older_pages(
    conn: psycopg.Connection,
) -> None:
    ids = [add_job(conn, "ingest", "pending") for _ in range(4)]

    first_page = list_jobs(conn, limit=2)
    assert _ids(first_page) == list(reversed(ids))[:2]

    add_job(conn, "ingest", "pending")  # a higher id, after the first page was read

    second_page = list_jobs(conn, before_id=first_page[-1].id, limit=2)
    assert _ids(second_page) == list(reversed(ids))[2:]


def test_list_jobs_item_shape_and_video_title_join(conn: psycopg.Connection) -> None:
    conn.execute(
        "INSERT INTO videos (video_id, title) VALUES (%s, %s)", (VID, "secret title")
    )
    job_id = add_job(conn, "transcribe", "dead", key="default", error_class="TOOL_FAILURE")

    rows = list_jobs(conn, limit=50)

    assert len(rows) == 1
    row = rows[0]
    assert row.id == job_id
    assert row.video_id == VID
    assert row.video_title == "secret title"
    assert row.kind == "transcribe"
    assert row.dedupe_key == "default"
    assert row.state == "dead"
    assert row.priority == 0
    assert row.attempts == 0
    assert row.error_class == "TOOL_FAILURE"
    assert row.last_error is None
    assert row.run_after is not None
    assert row.locked_by is None
    assert row.heartbeat_at is None
    assert row.finished_at is None
    assert row.created_at is not None
    assert not hasattr(row, "payload")


def test_list_jobs_video_title_is_null_without_a_videos_row_or_title(
    conn: psycopg.Connection,
) -> None:
    conn.execute("INSERT INTO videos (video_id) VALUES (%s)", ("notitle0001",))
    with_row_no_title = add_job(conn, "ingest", "pending", video_id="notitle0001")
    without_row = add_job(conn, "ingest", "pending", video_id="noviderow001")

    rows = {row.id: row for row in list_jobs(conn, limit=50)}

    assert rows[with_row_no_title].video_title is None
    assert rows[without_row].video_title is None
    assert rows[without_row].video_id == "noviderow001"


def test_list_jobs_last_error_round_trips_multiline_and_non_ascii(
    conn: psycopg.Connection,
) -> None:
    text = "Traceback (most recent call last):\n  line 2\nValueError: caf\u00e9 \U0001f600"
    job_id = add_job(conn, "transcribe", "dead", error_class="BUG", last_error=text)

    rows = list_jobs(conn, limit=50)

    assert next(r for r in rows if r.id == job_id).last_error == text


def test_list_jobs_runs_as_one_statement_with_bound_params_not_string_formatting(
    conn: psycopg.Connection,
) -> None:
    # A state value that would corrupt a string-formatted query is rejected
    # only because it never matches any row - never a syntax error.
    add_job(conn, "ingest", "pending")

    assert list_jobs(conn, state="pending' OR '1'='1", limit=50) == []


def test_list_jobs_runs_on_a_read_only_connection(conn: psycopg.Connection) -> None:
    add_job(conn, "ingest", "pending")
    conn.commit()
    conn.read_only = True

    assert len(list_jobs(conn, limit=50)) == 1


# ---------------------------------------------------------------------------
# get_job (#44)
# ---------------------------------------------------------------------------


def test_get_job_returns_none_for_a_missing_id(conn: psycopg.Connection) -> None:
    assert get_job(conn, 999_999) is None


def test_get_job_returns_the_same_shape_as_list_jobs(conn: psycopg.Connection) -> None:
    conn.execute("INSERT INTO videos (video_id, title) VALUES (%s, %s)", (VID, "t"))
    job_id = add_job(conn, "ingest", "dead", error_class="BUG")

    row = get_job(conn, job_id)

    assert row == list_jobs(conn, limit=50)[0]
    assert row is not None
    assert row.id == job_id
    assert row.video_title == "t"


# --- active_job (issue #101) ----------------------------------------------------


def _add_keyed(
    conn: psycopg.Connection,
    state: str,
    *,
    key: str = "v1:fake",
    kind: str = "analyze",
    video_id: str = VID,
    payload: str = "{}",
    priority: int = 0,
) -> int:
    row = conn.execute(
        "INSERT INTO jobs (video_id, kind, dedupe_key, state, payload, priority)"
        " VALUES (%s, %s, %s, %s, %s::jsonb, %s) RETURNING id",
        (video_id, kind, key, state, payload, priority),
    ).fetchone()
    assert row is not None
    return int(row[0])


@pytest.mark.parametrize("state", ["pending", "running"])
def test_active_job_returns_the_pending_or_running_job_at_the_key(
    conn: psycopg.Connection, state: str
) -> None:
    job_id = _add_keyed(conn, state, priority=-5)

    assert active_job(conn, VID, "analyze", "v1:fake") == ActiveJob(
        id=job_id, state=state, priority=-5, forced=False
    )


@pytest.mark.parametrize("state", ["done", "dead"])
def test_active_job_ignores_finished_jobs(conn: psycopg.Connection, state: str) -> None:
    _add_keyed(conn, state)

    assert active_job(conn, VID, "analyze", "v1:fake") is None


def test_active_job_ignores_other_keys_kinds_and_videos(conn: psycopg.Connection) -> None:
    _add_keyed(conn, "pending", key="v2:fake")
    _add_keyed(conn, "pending", kind="transcribe")
    _add_keyed(conn, "pending", video_id="zzzzzzzzzzz")

    assert active_job(conn, VID, "analyze", "v1:fake") is None


@pytest.mark.parametrize(
    ("payload", "forced"),
    [
        ('{"force": true}', True),
        ('{"force": false}', False),
        ('{"force": "true"}', False),
        ('{"force": 1}', False),
        ('{"other": 1}', False),
        ('{"force": true, "other": 1}', True),
    ],
)
def test_active_job_reports_forced_only_for_a_json_true(
    conn: psycopg.Connection, payload: str, forced: bool
) -> None:
    _add_keyed(conn, "pending", payload=payload)

    found = active_job(conn, VID, "analyze", "v1:fake")

    assert found is not None
    assert found.forced is forced
