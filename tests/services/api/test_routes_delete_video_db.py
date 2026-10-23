from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from common.config import get_settings
from common.repo.deleted_videos import VideoRunning, delete_video
from services.api import routes_write
from services.api.main import create_app
from tests.services.api.conftest import LogSink

pytestmark = pytest.mark.integration

VID = "dQw4w9WgXcQ"
OTHER = "abcdefghijk"


@pytest.fixture
def dsn(head_dsn: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    monkeypatch.setenv("DATABASE_URL", head_dsn)
    monkeypatch.setenv("AUDIO_DIR", str(tmp_path))
    get_settings.cache_clear()
    return head_dsn


@pytest.fixture
def db(dsn: str) -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(dsn, autocommit=True) as conn:
        yield conn


@pytest.fixture
def api(dsn: str, logs: LogSink) -> Iterator[TestClient]:
    with TestClient(create_app()) as client:
        yield client


def seed_video(db: psycopg.Connection[Any], video_id: str = VID, *, state: str = "pending") -> None:
    db.execute("INSERT INTO channels (channel_id, title) VALUES ('UCkeep', 'Keep') ON CONFLICT DO NOTHING")
    db.execute(
        "INSERT INTO videos (video_id, channel_id, title, origin) VALUES (%s, 'UCkeep', 'Title', 'adhoc')",
        (video_id,),
    )
    db.execute("INSERT INTO jobs (video_id, kind, state) VALUES (%s, 'ingest', %s)", (video_id, state))


def test_delete_rejects_malformed_and_percent_encoded_ids_without_database_access(
    api: TestClient,
) -> None:
    for path in ("short", "%64Qw4w9WgXcQ"):
        response = api.delete(f"/videos/{path}")
        assert response.status_code == 422
        assert response.json() == {
            "detail": "video id must be 11 characters of A-Z, a-z, 0-9, '_' or '-'"
        }
        assert path not in response.text


def test_delete_unknown_and_repeated_delete_are_fixed_404s(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    missing = api.delete(f"/videos/{VID}")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "video not found"}
    assert db.execute("SELECT count(*) FROM deleted_videos").fetchone() == (0,)

    seed_video(db)
    assert api.delete(f"/videos/{VID}").status_code == 204
    repeated = api.delete(f"/videos/{VID}")
    assert repeated.status_code == 404
    assert repeated.json() == {"detail": "video not found"}
    assert db.execute("SELECT video_id FROM deleted_videos").fetchall() == [(VID,)]


def test_running_job_blocks_delete_without_changes(
    api: TestClient, db: psycopg.Connection[Any]
) -> None:
    seed_video(db, state="running")
    before = db.execute("SELECT * FROM videos WHERE video_id = %s", (VID,)).fetchone()

    response = api.delete(f"/videos/{VID}")

    assert response.status_code == 409
    assert response.json() == {"detail": "video is currently processing"}
    assert db.execute("SELECT * FROM videos WHERE video_id = %s", (VID,)).fetchone() == before
    assert db.execute("SELECT state FROM jobs WHERE video_id = %s", (VID,)).fetchall() == [("running",)]
    assert db.execute("SELECT count(*) FROM deleted_videos").fetchone() == (0,)


def test_delete_removes_the_tree_jobs_and_batches_but_preserves_other_rows(
    api: TestClient, db: psycopg.Connection[Any], tmp_path: Path
) -> None:
    seed_video(db)
    seed_video(db, OTHER)
    media = tmp_path / "dq" / "audio.opus"
    media.parent.mkdir()
    media.write_bytes(b"audio")
    db.execute(
        "INSERT INTO media (video_id, path, bytes, expires_at) VALUES (%s, 'dq/audio.opus', 5, now())",
        (VID,),
    )
    transcript_id = db.execute(
        "INSERT INTO transcripts (video_id, source, segments, full_text) VALUES (%s, 'whisper', '[]', 'x') RETURNING id",
        (VID,),
    ).fetchone()[0]
    db.execute(
        "INSERT INTO transcript_chunks (transcript_id, seq, start_sec, end_sec, text, chunk_strategy) VALUES (%s, 0, 0, 1, 'x', 'time:1:0')",
        (transcript_id,),
    )
    analysis_id = db.execute(
        "INSERT INTO analyses (video_id, transcript_id, chunk_strategy, model, prompt_version, tldr) VALUES (%s, %s, 'time:1:0', 'm', 'v1', 'x') RETURNING id",
        (VID, transcript_id),
    ).fetchone()[0]
    db.execute("INSERT INTO topics (analysis_id, seq, title) VALUES (%s, 0, 't')", (analysis_id,))
    db.execute("INSERT INTO claims (analysis_id, text) VALUES (%s, 'c')", (analysis_id,))
    db.execute("INSERT INTO quotes (analysis_id, text) VALUES (%s, 'q')", (analysis_id,))
    job_id = db.execute("SELECT id FROM jobs WHERE video_id = %s", (VID,)).fetchone()[0]
    db.execute(
        "INSERT INTO llm_batches (job_id, step_key, batch_id, request_sha256) VALUES (%s, 'x', 'b', 'h')",
        (job_id,),
    )
    channel_before = db.execute("SELECT * FROM channels WHERE channel_id = 'UCkeep'").fetchone()

    response = api.delete(f"/videos/{VID}")

    assert response.status_code == 204
    assert response.content == b""
    assert not media.exists()
    assert db.execute("SELECT video_id FROM deleted_videos").fetchall() == [(VID,)]
    for table in ("videos", "transcripts", "transcript_chunks", "analyses", "topics", "claims", "quotes", "media", "jobs", "llm_batches"):
        assert db.execute(f"SELECT count(*) FROM {table} WHERE " + ({
            "videos": "video_id = %s", "transcripts": "video_id = %s", "analyses": "video_id = %s", "media": "video_id = %s", "jobs": "video_id = %s"
        }.get(table, "1 = 1")), (VID,) if table in {"videos", "transcripts", "analyses", "media", "jobs"} else ()).fetchone() == (0,)
    assert db.execute("SELECT video_id FROM videos WHERE video_id = %s", (OTHER,)).fetchone() == (OTHER,)
    assert db.execute("SELECT * FROM channels WHERE channel_id = 'UCkeep'").fetchone() == channel_before


def test_database_failure_rolls_back_every_change_and_returns_write_503(
    api: TestClient,
    db: psycopg.Connection[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_video(db)
    before_video = db.execute("SELECT * FROM videos WHERE video_id = %s", (VID,)).fetchone()
    before_jobs = db.execute("SELECT * FROM jobs WHERE video_id = %s", (VID,)).fetchall()

    real_delete = routes_write.delete_video

    def fail(_conn: Any, _video_id: str) -> Any:
        def forced() -> None:
            raise psycopg.OperationalError("forced")

        return real_delete(_conn, _video_id, before_commit=forced)

    monkeypatch.setattr(routes_write, "delete_video", fail)

    response = api.delete(f"/videos/{VID}")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
    assert db.execute("SELECT * FROM videos WHERE video_id = %s", (VID,)).fetchone() == before_video
    assert db.execute("SELECT * FROM jobs WHERE video_id = %s", (VID,)).fetchall() == before_jobs
    assert db.execute("SELECT count(*) FROM deleted_videos").fetchone() == (0,)


def wait_until_blocked(db: psycopg.Connection[Any], app: str) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = db.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE application_name = %s AND wait_event_type = 'Lock'",
            (app,),
        ).fetchone()
        if row is not None and row[0] > 0:
            return
    raise AssertionError(f"{app} never blocked on a lock")


def test_delete_serializes_with_worker_claim_and_claim_wins(
    dsn: str, db: psycopg.Connection[Any]
) -> None:
    seed_video(db)
    result: dict[str, VideoRunning] = {}

    with psycopg.connect(dsn) as claiming:
        claimed = claiming.execute(
            """
            UPDATE jobs SET state = 'running', locked_by = 'worker'
            WHERE id = (SELECT id FROM jobs WHERE video_id = %s FOR UPDATE SKIP LOCKED)
            RETURNING id
            """,
            (VID,),
        ).fetchone()
        assert claimed is not None

        def delete() -> None:
            try:
                with psycopg.connect(dsn, application_name="delete-claim-race") as conn:
                    delete_video(conn, VID)
            except VideoRunning as exc:
                result["error"] = exc

        thread = threading.Thread(target=delete)
        thread.start()
        wait_until_blocked(db, "delete-claim-race")
        claiming.commit()
        thread.join(timeout=10)

    assert not thread.is_alive()
    assert isinstance(result["error"], routes_write.VideoRunning)
    assert db.execute("SELECT state FROM jobs WHERE video_id = %s", (VID,)).fetchone() == ("running",)
    assert db.execute("SELECT count(*) FROM deleted_videos").fetchone() == (0,)


def test_unsafe_or_failed_unlink_does_not_change_committed_204(
    api: TestClient,
    db: psycopg.Connection[Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_video(db)
    outside = tmp_path.parent / "outside.opus"
    outside.write_bytes(b"keep")
    db.execute(
        "INSERT INTO media (video_id, path, bytes, expires_at) VALUES (%s, '../outside.opus', 4, now())",
        (VID,),
    )

    response = api.delete(f"/videos/{VID}")

    assert response.status_code == 204
    assert outside.read_bytes() == b"keep"
    assert db.execute("SELECT video_id FROM deleted_videos").fetchall() == [(VID,)]
