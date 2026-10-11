"""``services.api.backfill.backfill`` against Postgres (issue #41).

Selection, the dry-run no-write guarantee, the real-run writes, idempotency,
the RTF estimate, the transaction-idle guarantee while the catalog runs, and
two races. Races are forced: one transaction is held open, the racer runs in
a thread, and the test waits (bounded, no sleep) until ``pg_blocking_pids``
shows it blocked by the holder.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import psycopg
import pytest
from psycopg.pq import TransactionStatus

from common.errors import PermanentSourceError
from common.models import CatalogEntry, ChannelCatalog, FeedEntry
from common.queue import PRIORITY_BACKFILL, PRIORITY_NORMAL, PostgresQueue
from common.repo.videos import insert_discovered_video
from services.api.backfill import (
    ESTIMATE_ASSUMES,
    BackfillResult,
    ChannelNotRegistered,
    backfill,
)

pytestmark = pytest.mark.integration

CHANNEL = "UCuAXFkgsw1L7xaCfnd5JJOw"


def _vid(i: int) -> str:
    return f"bf{i:09d}"


def _entries(n: int, *, duration: int | None = 60) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(_vid(i), f"title {i}", duration) for i in range(n))


class FakeCatalog:
    def __init__(
        self,
        entries: Sequence[CatalogEntry] = (),
        *,
        total_count: int | None = None,
        error: Exception | None = None,
        on_call: Callable[[], None] | None = None,
    ) -> None:
        self.entries = tuple(entries)
        self.total_count = total_count
        self.error = error
        self.on_call = on_call
        self.calls: list[tuple[str, int]] = []

    def list_uploads(self, channel_id: str, *, limit: int) -> ChannelCatalog:
        self.calls.append((channel_id, limit))
        if self.on_call is not None:
            self.on_call()
        if self.error is not None:
            raise self.error
        return ChannelCatalog(channel_id, self.entries[:limit], self.total_count)


@pytest.fixture
def db(head_dsn: str) -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(head_dsn, autocommit=True) as conn:
        yield conn


@pytest.fixture
def work(head_dsn: str) -> Iterator[psycopg.Connection[Any]]:
    """A connection configured like ``deps.get_conn``'s: not autocommit."""
    with psycopg.connect(head_dsn) as conn:
        yield conn


@pytest.fixture
def channel(db: psycopg.Connection[Any]) -> str:
    db.execute(
        "INSERT INTO channels (channel_id, active, monitor_from) VALUES (%s, true, now())",
        (CHANNEL,),
    )
    return CHANNEL


def _run(
    conn: psycopg.Connection[Any],
    catalog: FakeCatalog,
    *,
    limit: int = 50,
    dry_run: bool,
    channel_id: str = CHANNEL,
) -> BackfillResult:
    return backfill(
        conn, PostgresQueue(conn), catalog, channel_id, limit=limit, dry_run=dry_run
    )


def _counts(db: psycopg.Connection[Any]) -> tuple[int, int, int]:
    row = db.execute(
        "SELECT (SELECT count(*) FROM channels), (SELECT count(*) FROM videos),"
        " (SELECT count(*) FROM jobs)"
    ).fetchone()
    assert row is not None
    return (int(row[0]), int(row[1]), int(row[2]))


def _channel_row(db: psycopg.Connection[Any]) -> tuple[Any, ...] | None:
    return db.execute("SELECT * FROM channels WHERE channel_id = %s", (CHANNEL,)).fetchone()


def _jobs(db: psycopg.Connection[Any]) -> list[tuple[Any, ...]]:
    return db.execute(
        "SELECT video_id, kind, dedupe_key, priority, payload, state FROM jobs ORDER BY id"
    ).fetchall()


def _seed_known(db: psycopg.Connection[Any], video_id: str, origin: str) -> None:
    db.execute(
        "INSERT INTO videos (video_id, channel_id, origin) VALUES (%s, %s, %s)",
        (video_id, CHANNEL, origin),
    )


def _add_rtf(db: psycopg.Connection[Any], rtfs: Sequence[float]) -> None:
    for i, rtf in enumerate(rtfs):
        vid = f"rt{i:09d}"
        db.execute("INSERT INTO videos (video_id) VALUES (%s)", (vid,))
        db.execute(
            "INSERT INTO transcripts (video_id, source, segments, full_text, engine_meta)"
            " VALUES (%s, 'whisper', '[]', '', %s::jsonb)",
            (vid, json.dumps({"rtf": rtf})),
        )


def _wait_until_blocked_by(
    observer: psycopg.Connection[Any], waiter_pid: int, holder_pid: int
) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = observer.execute("SELECT pg_blocking_pids(%s)", (waiter_pid,)).fetchone()
        if row is not None and holder_pid in row[0]:
            return
    raise AssertionError(f"backend {waiter_pid} never blocked on {holder_pid}")


class _Call:
    def __init__(self, fn: Callable[[], BackfillResult]) -> None:
        self.result: BackfillResult | None = None
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._target, args=(fn,), daemon=True)

    def _target(self, fn: Callable[[], BackfillResult]) -> None:
        try:
            self.result = fn()
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            self.error = exc

    def start(self) -> _Call:
        self._thread.start()
        return self

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def finish(self) -> BackfillResult:
        self._thread.join(timeout=10)
        assert not self._thread.is_alive(), "backfill did not finish"
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


# --- channel check ----------------------------------------------------------


def test_an_unregistered_channel_raises_before_listing_and_writes_nothing(
    db: psycopg.Connection[Any], work: psycopg.Connection[Any]
) -> None:
    catalog = FakeCatalog(_entries(3))

    with pytest.raises(ChannelNotRegistered):
        _run(work, catalog, dry_run=False)

    assert catalog.calls == []
    assert _counts(db) == (0, 0, 0)
    assert work.info.transaction_status == TransactionStatus.IDLE


@pytest.mark.parametrize("dry_run", [True, False])
def test_an_inactive_channel_can_be_backfilled_and_no_channel_column_changes(
    db: psycopg.Connection[Any], work: psycopg.Connection[Any], dry_run: bool
) -> None:
    db.execute(
        "INSERT INTO channels (channel_id, active, last_poll_err) VALUES (%s, false, 'x')",
        (CHANNEL,),
    )
    before = _channel_row(db)

    result = _run(work, FakeCatalog(_entries(2)), dry_run=dry_run)

    assert result.new_videos == 2
    assert _channel_row(db) == before


def test_no_channel_column_changes_for_an_active_channel(
    db: psycopg.Connection[Any], work: psycopg.Connection[Any], channel: str
) -> None:
    db.execute("UPDATE channels SET last_polled = now() - interval '1 hour'")
    before = _channel_row(db)

    _run(work, FakeCatalog(_entries(2)), dry_run=False)

    assert _channel_row(db) == before


# --- selection --------------------------------------------------------------


def test_the_catalog_is_called_once_with_the_channel_and_limit(
    work: psycopg.Connection[Any], channel: str
) -> None:
    catalog = FakeCatalog(_entries(10))

    _run(work, catalog, limit=7, dry_run=True)

    assert catalog.calls == [(CHANNEL, 7)]


def test_no_transaction_is_open_while_the_catalog_runs(
    work: psycopg.Connection[Any], channel: str
) -> None:
    seen: list[TransactionStatus] = []
    catalog = FakeCatalog(
        _entries(2), on_call=lambda: seen.append(work.info.transaction_status)
    )

    _run(work, catalog, dry_run=False)

    assert seen == [TransactionStatus.IDLE]
    assert work.info.transaction_status == TransactionStatus.IDLE


def test_the_channel_check_is_committed_before_listing(
    db: psycopg.Connection[Any], work: psycopg.Connection[Any], channel: str
) -> None:
    locks: list[int] = []

    def count_locks() -> None:
        row = db.execute(
            "SELECT count(*) FROM pg_locks WHERE pid = %s", (work.info.backend_pid,)
        ).fetchone()
        assert row is not None
        locks.append(int(row[0]))

    _run(work, FakeCatalog(_entries(1), on_call=count_locks), dry_run=True)

    assert locks == [0]


@pytest.mark.parametrize("origin", ["adhoc", "rss", "backfill"])
@pytest.mark.parametrize("job_state", [None, "pending", "running", "done", "dead"])
def test_any_existing_video_row_is_known_and_never_reenqueued(
    db: psycopg.Connection[Any],
    work: psycopg.Connection[Any],
    channel: str,
    origin: str,
    job_state: str | None,
) -> None:
    _seed_known(db, _vid(1), origin)
    if job_state is not None:
        db.execute(
            "INSERT INTO jobs (kind, video_id, state) VALUES ('ingest', %s, %s)",
            (_vid(1), job_state),
        )
    jobs_before = _jobs(db)

    result = _run(work, FakeCatalog(_entries(3)), dry_run=False)

    assert result.already_known == 1
    assert result.video_ids == (_vid(0), _vid(2))
    assert [j for j in _jobs(db) if j[0] == _vid(1)] == jobs_before


def test_an_unavailable_video_is_known(
    db: psycopg.Connection[Any], work: psycopg.Connection[Any], channel: str
) -> None:
    db.execute(
        "INSERT INTO videos (video_id, unavailable, origin) VALUES (%s, 'removed', 'adhoc')",
        (_vid(0),),
    )

    result = _run(work, FakeCatalog(_entries(1)), dry_run=False)

    assert (result.already_known, result.new_videos, result.enqueued) == (1, 0, 0)


def test_new_videos_keep_catalog_order_and_entries_pass_through_unfiltered(
    work: psycopg.Connection[Any], channel: str
) -> None:
    entries = (
        CatalogEntry("zzzzzzzzzz1", "[Private video]", None),
        CatalogEntry("aaaaaaaaaa1", "#shorts", 15),
        CatalogEntry("mmmmmmmmmm1", "[Deleted video]", None),
    )

    result = _run(work, FakeCatalog(entries), dry_run=True)

    assert result.video_ids == ("zzzzzzzzzz1", "aaaaaaaaaa1", "mmmmmmmmmm1")


# --- dry run ----------------------------------------------------------------


def test_a_dry_run_reports_and_writes_nothing(
    db: psycopg.Connection[Any], work: psycopg.Connection[Any], channel: str
) -> None:
    _seed_known(db, _vid(0), "rss")
    _add_rtf(db, [0.5])
    before = _counts(db)

    result = _run(
        work, FakeCatalog(_entries(4, duration=100), total_count=1234), limit=10, dry_run=True
    )

    assert _counts(db) == before
    assert result.dry_run is True
    assert result.backfill_id is None
    assert (result.listed, result.total_count, result.already_known) == (4, 1234, 1)
    assert (result.new_videos, result.enqueued) == (3, 0)
    assert result.video_ids == (_vid(1), _vid(2), _vid(3))
    assert result.estimate.audio_sec == 300
    assert result.estimate.unknown_duration == 0
    assert result.estimate.rtf == 0.5
    assert result.estimate.rtf_samples == 1
    assert result.estimate.transcription_sec == 150
    assert result.estimate.assumes == ESTIMATE_ASSUMES == (
        "manual-subtitle videos skip speech-to-text; unchecked videos are assumed to need it"
    )


# --- real run ---------------------------------------------------------------


def test_a_real_run_writes_one_backfill_row_and_one_ingest_job_per_new_video(
    db: psycopg.Connection[Any], work: psycopg.Connection[Any], channel: str
) -> None:
    _seed_known(db, _vid(1), "rss")
    entries = (
        CatalogEntry(_vid(0), "first\x00", 61),
        CatalogEntry(_vid(1), "known", 5),
        CatalogEntry(_vid(2), None, None),
    )

    result = _run(work, FakeCatalog(entries), dry_run=False)

    assert result.enqueued == 2
    assert result.backfill_id is not None
    assert str(uuid.UUID(result.backfill_id)) == result.backfill_id
    rows = db.execute(
        "SELECT video_id, channel_id, title, duration_sec, origin FROM videos"
        " WHERE origin = 'backfill' ORDER BY video_id"
    ).fetchall()
    assert rows == [
        (_vid(0), CHANNEL, "first", 61, "backfill"),
        (_vid(2), CHANNEL, None, None, "backfill"),
    ]
    payload = {"origin": "backfill", "backfill_id": result.backfill_id}
    assert _jobs(db) == [
        (_vid(0), "ingest", "default", PRIORITY_BACKFILL, payload, "pending"),
        (_vid(2), "ingest", "default", PRIORITY_BACKFILL, payload, "pending"),
    ]
    assert PRIORITY_BACKFILL == -10


def test_repeating_a_real_run_creates_nothing_new(
    db: psycopg.Connection[Any], work: psycopg.Connection[Any], channel: str
) -> None:
    catalog = FakeCatalog(_entries(5))
    first = _run(work, catalog, dry_run=False)
    after_first = _counts(db)

    second = _run(work, catalog, dry_run=False)

    assert first.enqueued == 5
    assert _counts(db) == after_first
    assert (second.new_videos, second.enqueued, second.already_known) == (0, 0, second.listed)
    assert second.backfill_id is not None
    assert second.backfill_id != first.backfill_id


def test_a_larger_limit_reaches_further_back_and_skips_known_videos(
    db: psycopg.Connection[Any], work: psycopg.Connection[Any], channel: str
) -> None:
    catalog = FakeCatalog(_entries(8))
    _run(work, catalog, limit=3, dry_run=False)

    result = _run(work, catalog, limit=8, dry_run=False)

    assert (result.listed, result.already_known, result.enqueued) == (8, 3, 5)
    assert result.video_ids == tuple(_vid(i) for i in range(3, 8))


class _FailingQueue(PostgresQueue):
    def __init__(self, conn: psycopg.Connection[Any], fail_on: int) -> None:
        super().__init__(conn)
        self.calls = 0
        self.fail_on = fail_on

    def enqueue(self, *args: Any, **kwargs: Any) -> int | None:
        self.calls += 1
        if self.calls == self.fail_on:
            raise psycopg.OperationalError("connection lost")
        return super().enqueue(*args, **kwargs)


def test_a_database_error_leaves_each_video_complete_or_absent_and_a_repeat_fills_in(
    db: psycopg.Connection[Any], work: psycopg.Connection[Any], channel: str
) -> None:
    catalog = FakeCatalog(_entries(4))

    with pytest.raises(psycopg.OperationalError):
        backfill(work, _FailingQueue(work, 2), catalog, CHANNEL, limit=10, dry_run=False)

    assert db.execute("SELECT video_id FROM videos ORDER BY 1").fetchall() == [(_vid(0),)]
    assert [j[0] for j in _jobs(db)] == [_vid(0)]

    work.rollback()
    result = _run(work, catalog, limit=10, dry_run=False)

    assert result.enqueued == 3
    videos = {r[0] for r in db.execute("SELECT video_id FROM videos").fetchall()}
    jobs = [j[0] for j in _jobs(db)]
    assert videos == {_vid(i) for i in range(4)}
    assert sorted(jobs) == sorted(videos)


# --- estimate and boundaries ------------------------------------------------


def test_an_empty_catalog_reports_zeros(
    work: psycopg.Connection[Any], channel: str
) -> None:
    result = _run(work, FakeCatalog(()), dry_run=False)

    assert (result.listed, result.already_known, result.new_videos, result.enqueued) == (
        0, 0, 0, 0,
    )
    assert result.video_ids == ()
    assert result.estimate.audio_sec == 0
    assert result.estimate.unknown_duration == 0
    assert result.estimate.rtf is None
    assert result.estimate.rtf_samples == 0
    assert result.estimate.transcription_sec is None


def test_exactly_limit_entries_all_new(work: psycopg.Connection[Any], channel: str) -> None:
    result = _run(work, FakeCatalog(_entries(5)), limit=5, dry_run=True)

    assert (result.listed, result.new_videos, result.limit) == (5, 5, 5)


def test_every_entry_known(
    db: psycopg.Connection[Any], work: psycopg.Connection[Any], channel: str
) -> None:
    for i in range(3):
        _seed_known(db, _vid(i), "rss")

    result = _run(work, FakeCatalog(_entries(3)), dry_run=True)

    assert (result.already_known, result.new_videos, result.video_ids) == (3, 0, ())
    assert result.estimate.audio_sec == 0


def test_null_durations_are_counted_not_summed(
    db: psycopg.Connection[Any], work: psycopg.Connection[Any], channel: str
) -> None:
    _add_rtf(db, [0.1, 0.2, 0.4, 0.8])

    result = _run(work, FakeCatalog(_entries(3, duration=None)), dry_run=True)

    assert result.estimate.audio_sec == 0
    assert result.estimate.unknown_duration == 3
    assert result.estimate.rtf == pytest.approx(0.3)
    assert result.estimate.rtf_samples == 4
    assert result.estimate.transcription_sec == 0


def test_transcription_sec_is_rounded(
    db: psycopg.Connection[Any], work: psycopg.Connection[Any], channel: str
) -> None:
    _add_rtf(db, [0.333])

    result = _run(work, FakeCatalog(_entries(1, duration=100)), dry_run=True)

    assert result.estimate.transcription_sec == 33


# --- catalog failure ---------------------------------------------------------


@pytest.mark.parametrize("reason", ["removed", "private", "geoblocked", "agegated"])
def test_a_permanent_source_error_is_an_empty_listing(
    db: psycopg.Connection[Any], work: psycopg.Connection[Any], channel: str, reason: str
) -> None:
    before = (_counts(db), _channel_row(db))

    result = _run(
        work, FakeCatalog(error=PermanentSourceError(reason, "gone")), dry_run=False
    )

    assert (result.listed, result.total_count, result.new_videos, result.enqueued) == (
        0, None, 0, 0,
    )
    assert (_counts(db), _channel_row(db)) == before


# --- races ------------------------------------------------------------------


def test_two_concurrent_real_backfills_write_each_video_once(
    head_dsn: str, db: psycopg.Connection[Any], channel: str
) -> None:
    entries = _entries(4)
    paused = threading.Event()
    release = threading.Event()

    class PausingQueue(PostgresQueue):
        def enqueue(self, *args: Any, **kwargs: Any) -> int | None:
            job = super().enqueue(*args, **kwargs)
            if not paused.is_set():
                paused.set()
                assert release.wait(10), "never released"
            return job

    conn_a = psycopg.connect(head_dsn, application_name="bf-a")
    conn_b = psycopg.connect(head_dsn, application_name="bf-b")
    a = b = None
    try:
        a = _Call(
            lambda: backfill(
                conn_a, PausingQueue(conn_a), FakeCatalog(entries), CHANNEL,
                limit=10, dry_run=False,
            )
        ).start()
        assert paused.wait(10), "backfill A never reached its first enqueue"
        b = _Call(
            lambda: backfill(
                conn_b, PostgresQueue(conn_b), FakeCatalog(entries), CHANNEL,
                limit=10, dry_run=False,
            )
        ).start()
        _wait_until_blocked_by(db, conn_b.info.backend_pid, conn_a.info.backend_pid)
        assert b.is_alive()
        release.set()
        result_a, result_b = a.finish(), b.finish()
    finally:
        release.set()
        for call in (a, b):
            if call is not None:
                call._thread.join(timeout=10)
        conn_a.close()
        conn_b.close()

    assert result_a.enqueued + result_b.enqueued == 4
    rows = db.execute(
        "SELECT video_id, count(*) FROM videos GROUP BY video_id ORDER BY 1"
    ).fetchall()
    assert rows == [(e.video_id, 1) for e in entries]
    jobs = db.execute(
        "SELECT video_id, count(*) FROM jobs WHERE kind = 'ingest' GROUP BY 1 ORDER BY 1"
    ).fetchall()
    assert jobs == [(e.video_id, 1) for e in entries]


def test_a_backfill_racing_an_rss_insert_skips_the_video(
    head_dsn: str, db: psycopg.Connection[Any], channel: str
) -> None:
    from datetime import UTC, datetime

    holder = psycopg.connect(head_dsn)
    racer_conn = psycopg.connect(head_dsn)
    racer = None
    try:
        insert_discovered_video(
            holder,
            FeedEntry(_vid(0), CHANNEL, "from rss", datetime(2026, 1, 1, tzinfo=UTC)),
            origin="rss",
        )
        PostgresQueue(holder).enqueue("ingest", _vid(0), priority=PRIORITY_NORMAL)
        racer = _Call(
            lambda: backfill(
                racer_conn, PostgresQueue(racer_conn), FakeCatalog(_entries(1)),
                CHANNEL, limit=10, dry_run=False,
            )
        ).start()
        _wait_until_blocked_by(db, racer_conn.info.backend_pid, holder.info.backend_pid)
        assert racer.is_alive()
        holder.commit()
        result = racer.finish()
    finally:
        holder.rollback()
        holder.close()
        if racer is not None:
            racer._thread.join(timeout=10)
        racer_conn.close()

    assert (result.new_videos, result.enqueued) == (1, 0)
    assert db.execute("SELECT origin FROM videos").fetchall() == [("rss",)]
    assert [(j[0], j[3]) for j in _jobs(db)] == [(_vid(0), PRIORITY_NORMAL)]


@pytest.mark.parametrize("job_state", [None, "done", "dead"])
def test_a_row_another_writer_inserts_first_gets_no_backfill_job(
    head_dsn: str, db: psycopg.Connection[Any], channel: str, job_state: str | None
) -> None:
    holder = psycopg.connect(head_dsn)
    racer_conn = psycopg.connect(head_dsn)
    racer = None
    try:
        holder.execute(
            "INSERT INTO videos (video_id, channel_id, origin) VALUES (%s, %s, 'adhoc')",
            (_vid(0), CHANNEL),
        )
        if job_state is not None:
            holder.execute(
                "INSERT INTO jobs (kind, video_id, state) VALUES ('ingest', %s, %s)",
                (_vid(0), job_state),
            )
        racer = _Call(
            lambda: backfill(
                racer_conn, PostgresQueue(racer_conn), FakeCatalog(_entries(1)),
                CHANNEL, limit=10, dry_run=False,
            )
        ).start()
        _wait_until_blocked_by(db, racer_conn.info.backend_pid, holder.info.backend_pid)
        assert racer.is_alive()
        holder.commit()
        result = racer.finish()
    finally:
        holder.rollback()
        holder.close()
        if racer is not None:
            racer._thread.join(timeout=10)
        racer_conn.close()

    assert (result.new_videos, result.enqueued) == (1, 0)
    assert db.execute("SELECT origin FROM videos").fetchall() == [("adhoc",)]
    expected = [] if job_state is None else [(_vid(0), 0, job_state)]
    assert [(j[0], j[3], j[5]) for j in _jobs(db)] == expected


def test_a_permanent_source_error_is_logged_with_channel_and_reason(
    work: psycopg.Connection[Any], channel: str, caplog: pytest.LogCaptureFixture
) -> None:
    from structlog.testing import capture_logs

    with capture_logs() as logs:
        _run(work, FakeCatalog(error=PermanentSourceError("removed", "gone")), dry_run=True)

    [line] = [entry for entry in logs if entry["log_level"] == "info"]
    assert line["channel_id"] == CHANNEL
    assert line["reason"] == "removed"
    assert "removed" not in line["event"]
