"""``analysis_run_for``: the newest analysis of a video for one model and prompt (#62)."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest

from common.repo.analyses import analysis_run_for, latest_analysis_run
from tests.common.repo import read_seed as seed

pytestmark = pytest.mark.integration

VID = "ev000000001"


@pytest.fixture
def transcripts(conn: psycopg.Connection[Any]) -> tuple[int, int]:
    seed.video(conn, VID)
    return seed.transcript(conn, VID, "youtube_auto"), seed.transcript(conn, VID, "whisper")


def test_no_match_returns_none(conn: psycopg.Connection[Any], transcripts: tuple[int, int]) -> None:
    seed.analysis(conn, VID, transcripts[0], created=1, model="m1", prompt_version="v1")
    assert analysis_run_for(conn, VID, "m1", "v2") is None
    assert analysis_run_for(conn, VID, "m2", "v1") is None
    assert analysis_run_for(conn, "nope0000000", "m1", "v1") is None


def test_both_model_and_prompt_must_match(
    conn: psycopg.Connection[Any], transcripts: tuple[int, int]
) -> None:
    wanted = seed.analysis(conn, VID, transcripts[0], created=1, model="m1", prompt_version="v1")
    seed.analysis(conn, VID, transcripts[0], created=5, model="m1", prompt_version="v2")
    seed.analysis(conn, VID, transcripts[0], created=6, model="m2", prompt_version="v1")

    run = analysis_run_for(conn, VID, "m1", "v1")

    assert run is not None and run.analysis.id == wanted


def test_newest_wins_whatever_the_transcript_or_chunk_strategy(
    conn: psycopg.Connection[Any], transcripts: tuple[int, int]
) -> None:
    seed.analysis(conn, VID, transcripts[0], created=1, model="m1", prompt_version="v1")
    newest = seed.analysis(
        conn, VID, transcripts[1], created=2, model="m1", prompt_version="v1"
    )
    tie = seed.analysis(conn, VID, transcripts[0], created=2, model="m1", prompt_version="v1")

    run = analysis_run_for(conn, VID, "m1", "v1")

    assert run is not None
    assert run.analysis.id == tie > newest  # same created_at: the higher id wins
    assert run.transcript_source == "youtube_auto"


def test_returns_the_same_shape_as_latest_analysis_run(
    conn: psycopg.Connection[Any], transcripts: tuple[int, int]
) -> None:
    seed.analysis(
        conn, VID, transcripts[0], created=1, model="m1", prompt_version="v1",
        topics=[(1, "second"), (0, "first")],
        claims=[{"text": "late", "start_sec": 9.0}, {"text": "none", "start_sec": None},
                {"text": "early", "start_sec": 1.0}],
        quotes=[{"text": "q2", "start_sec": 3}, {"text": "q1", "start_sec": 2}],
        speaker_roster={"speakers": [{"name": "A", "role": "host"}]},
    )
    assert analysis_run_for(conn, VID, "m1", "v1") == latest_analysis_run(conn, VID)


def test_statement_count_is_fixed_whatever_the_number_of_children(
    conn: psycopg.Connection[Any], transcripts: tuple[int, int]
) -> None:
    seed.analysis(
        conn, VID, transcripts[0], created=1, model="m1", prompt_version="v1",
        topics=[(i, f"t{i}") for i in range(7)],
        claims=[{"text": f"c{i}", "start_sec": float(i)} for i in range(30)],
        quotes=[{"text": f"q{i}", "start_sec": float(i)} for i in range(12)],
    )
    spy = _CountingConnection(conn)

    run = analysis_run_for(spy, VID, "m1", "v1")  # type: ignore[arg-type]

    assert run is not None
    assert (len(run.analysis.topics), len(run.analysis.claims), len(run.analysis.quotes)) == (7, 30, 12)
    assert spy.statements == 4  # analysis row, topics, claims, quotes: nothing per child


class _CountingConnection:
    """Forwards ``execute`` to a real connection and counts the statements."""

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self._conn = conn
        self.statements = 0

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.statements += 1
        return self._conn.execute(*args, **kwargs)
