"""Every planner task must leave its connection idle (issue #199). Needs Postgres.

The planner closes a task's connection without committing, so a write that is
still inside an open transaction when the task returns is silently lost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg
import pytest
from pydantic import SecretStr

from common.config import Settings
from services.planner.main import build_registry, run_on_fresh_connection

pytestmark = pytest.mark.integration


def test_a_task_that_leaves_a_transaction_open_fails_and_is_rolled_back(
    conn: psycopg.Connection[Any], head_dsn: str
) -> None:
    def forgetful(c: psycopg.Connection[Any]) -> int:
        c.execute("INSERT INTO videos (video_id) VALUES ('lost')")
        return 1

    with pytest.raises(RuntimeError, match="open transaction"):
        run_on_fresh_connection(lambda: psycopg.connect(head_dsn), forgetful)

    assert conn.execute("SELECT count(*) FROM videos").fetchone() == (0,)


def test_a_task_that_commits_passes_and_persists(
    conn: psycopg.Connection[Any], head_dsn: str
) -> None:
    def careful(c: psycopg.Connection[Any]) -> int:
        c.execute("INSERT INTO videos (video_id) VALUES ('kept')")
        c.commit()
        return 1

    assert run_on_fresh_connection(lambda: psycopg.connect(head_dsn), careful) == 1
    assert conn.execute("SELECT count(*) FROM videos").fetchone() == (1,)


def test_every_registered_task_returns_with_an_idle_connection(
    conn: psycopg.Connection[Any], head_dsn: str, tmp_path: Path
) -> None:
    # Empty database, so nothing needs the network. A task that opens a read
    # transaction and forgets to end it fails here, whatever it goes on to write.
    settings = Settings(DATABASE_URL=SecretStr(head_dsn), AUDIO_DIR=tmp_path)
    for task in build_registry(settings, lambda: psycopg.connect(head_dsn)):
        task.fn()
