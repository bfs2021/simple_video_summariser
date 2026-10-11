"""``backfill`` dry-run estimate on Postgres (issue #117).

The metadata source is a fake; nothing touches the network or yt-dlp.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from psycopg.pq import TransactionStatus

from common.models import CatalogEntry, ChannelCatalog
from common.queue import PostgresQueue
from services.api.backfill import BackfillResult, SubtitleProbe, backfill

pytestmark = pytest.mark.integration

CHANNEL = "UCuAXFkgsw1L7xaCfnd5JJOw"
ENTRIES = tuple(CatalogEntry(f"dr{i:09d}", f"secret {i}", 100) for i in range(4))


class Catalog:
    def list_uploads(self, channel_id: str, *, limit: int) -> ChannelCatalog:
        return ChannelCatalog(channel_id, ENTRIES[:limit], None)


class Meta:
    def __init__(self, conn: psycopg.Connection[Any] | None = None) -> None:
        self.conn = conn
        self.calls: list[str] = []
        self.statuses: list[TransactionStatus] = []
        self.manual = {ENTRIES[1].video_id}

    def factory(self, timeout: float) -> Meta:
        return self

    def fetch(self, video_id: str) -> Any:
        self.calls.append(video_id)
        if self.conn is not None:
            self.statuses.append(self.conn.info.transaction_status)
        return video_id

    def probe(self) -> SubtitleProbe:
        return SubtitleProbe(self.factory, lambda meta: str(meta) in self.manual)


@pytest.fixture
def db(head_dsn: str) -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(head_dsn, autocommit=True) as conn:
        conn.execute("INSERT INTO channels (channel_id) VALUES (%s)", (CHANNEL,))
        yield conn


def _run(
    conn: psycopg.Connection[Any], meta: Meta | None, *, dry_run: bool = True, **kw: Any
) -> BackfillResult:
    return backfill(
        conn,
        PostgresQueue(conn),
        Catalog(),
        CHANNEL,
        limit=10,
        dry_run=dry_run,
        probe=None if meta is None else meta.probe(),
        **kw,
    )


def _counts(db: psycopg.Connection[Any]) -> list[int]:
    row = db.execute(
        "SELECT (SELECT count(*) FROM channels), (SELECT count(*) FROM videos),"
        " (SELECT count(*) FROM jobs), (SELECT count(*) FROM transcripts),"
        " (SELECT count(*) FROM analyses)"
    ).fetchone()
    assert row is not None
    return [int(x) for x in row]


def test_a_dry_run_probes_classifies_and_reports_per_video(db: psycopg.Connection[Any]) -> None:
    meta = Meta()

    result = _run(db, meta)

    assert meta.calls == [e.video_id for e in ENTRIES]
    assert [v.video_id for v in result.videos] == [e.video_id for e in ENTRIES]
    assert [v.subtitles for v in result.videos] == ["transcribe", "manual", "transcribe", "transcribe"]
    assert (result.estimate.probed, result.estimate.manual_subtitles) == (4, 1)
    assert result.estimate.needs_transcription == 3
    assert result.estimate.unchecked == 0


def test_a_dry_run_writes_nothing(db: psycopg.Connection[Any]) -> None:
    before = _counts(db)

    _run(db, Meta())

    assert _counts(db) == before


def test_known_videos_are_never_probed(db: psycopg.Connection[Any]) -> None:
    db.execute("INSERT INTO videos (video_id) VALUES (%s)", (ENTRIES[0].video_id,))
    meta = Meta()

    result = _run(db, meta)

    assert ENTRIES[0].video_id not in meta.calls
    assert result.already_known == 1


def test_prefer_whisper_probes_nothing_and_everything_needs_transcription(
    db: psycopg.Connection[Any],
) -> None:
    meta = Meta()

    result = _run(db, meta, prefer_whisper=True)

    assert meta.calls == []
    assert [v.subtitles for v in result.videos] == ["transcribe"] * 4
    assert result.estimate.probed == 0
    assert result.estimate.needs_transcription == 4


def test_a_real_run_probes_nothing_reports_unchecked_and_enqueues_as_before(
    head_dsn: str, db: psycopg.Connection[Any]
) -> None:
    meta = Meta()

    result = _run(db, meta, dry_run=False, prefer_whisper=True)

    assert meta.calls == []
    assert [v.subtitles for v in result.videos] == ["unchecked"] * 4
    assert result.enqueued == 4
    with psycopg.connect(head_dsn) as fresh:  # a new connection: the rows are committed
        assert fresh.execute("SELECT count(*) FROM jobs WHERE kind = 'ingest'").fetchone() == (4,)
        assert fresh.execute("SELECT count(*) FROM videos").fetchone() == (4,)


def test_no_transaction_is_open_while_probing(head_dsn: str) -> None:
    with psycopg.connect(head_dsn) as work:  # not autocommit, like the API's connection
        work.execute("INSERT INTO channels (channel_id) VALUES (%s)", (CHANNEL,))
        work.commit()
        meta = Meta(work)

        _run(work, meta)

    assert meta.statuses == [TransactionStatus.IDLE] * 4


def test_the_estimate_takes_no_lock(head_dsn: str, db: psycopg.Connection[Any]) -> None:
    db.execute("INSERT INTO videos (video_id, duration_sec) VALUES ('held0000001', 10)")
    holder = psycopg.connect(head_dsn)  # holds row locks on everything it reads from
    try:
        holder.execute("SELECT * FROM videos FOR UPDATE")
        holder.execute("LOCK TABLE transcripts, analyses IN ROW EXCLUSIVE MODE")
        db.execute("SET lock_timeout = '2s'")

        result = _run(db, Meta())  # would raise LockNotAvailable if it waited
    finally:
        holder.rollback()
        holder.close()

    assert result.new_videos == 4


def test_the_llm_numbers_come_from_recorded_data_for_the_current_model(
    db: psycopg.Connection[Any],
) -> None:
    db.execute("INSERT INTO videos (video_id, duration_sec) VALUES ('hist0000001', 10)")
    row = db.execute(
        "INSERT INTO transcripts (video_id, source, segments, full_text)"
        " VALUES ('hist0000001', 'whisper', '[]', %s) RETURNING id",
        ("x" * 100,),
    ).fetchone()
    assert row is not None
    db.execute(
        "INSERT INTO analyses (video_id, transcript_id, chunk_strategy, model, prompt_version,"
        " tldr, cost_usd) VALUES ('hist0000001', %s, 's', 'claude-x', 'v1', 't', 0.01)"
        ", ('hist0000001', %s, 's', 'old-model', 'v1', 't', 5)",
        (row[0], row[0]),
    )

    anthropic = _run(db, Meta(), summarizer="anthropic", model="claude-x")
    ollama = _run(db, Meta(), summarizer="ollama", model="claude-x")

    llm = anthropic.estimate.llm
    assert (llm.chars_per_audio_sec, llm.chars_samples) == (10.0, 1)
    assert llm.cost_samples == 1
    assert llm.cost_per_char == pytest.approx(1e-4)
    assert anthropic.videos[0].est_input_tokens == 250  # 100 s * 10 chars/s / 4
    assert anthropic.videos[0].llm_cost_usd == pytest.approx(0.1)
    assert llm.cost_usd == pytest.approx(0.4)
    other = ollama.estimate.llm
    assert (other.cost_per_char, other.cost_samples, other.cost_usd) == (None, 0, None)
    assert other.est_input_tokens == 1000
