"""Tests for common/repo/analyses.py (issue #14, architecture.md §7.4)."""

from __future__ import annotations

import psycopg
import pytest

from common.models import Analysis, Claim, Quote, Topic
from common.repo.analyses import latest_analysis, save_analysis
from common.repo.transcripts import save_transcript

pytestmark = pytest.mark.integration


def make_analysis(video_id: str, transcript_id: int, **overrides: object) -> Analysis:
    defaults: dict[str, object] = {
        "video_id": video_id,
        "transcript_id": transcript_id,
        "chunk_strategy": "time:900:60",
        "model": "test-model",
        "prompt_version": "v1",
        "tldr": "tldr text",
        "topics": (Topic(seq=0, title="Topic A"),),
        "claims": (Claim(text="claim A", speaker="Alice"),),
        "quotes": (Quote(text="quote A", speaker="Alice"),),
    }
    defaults.update(overrides)
    return Analysis(**defaults)  # type: ignore[arg-type]


def test_save_analysis_inserts_analysis_and_children_and_returns_id(
    conn: psycopg.Connection, video_id: str
) -> None:
    transcript_id = save_transcript(conn, video_id, "whisper", "en", "none", (), None)
    analysis = make_analysis(video_id, transcript_id)

    analysis_id = save_analysis(conn, analysis)

    row = conn.execute(
        "SELECT tldr, model FROM analyses WHERE id = %s", (analysis_id,)
    ).fetchone()
    assert row == ("tldr text", "test-model")

    topics = conn.execute(
        "SELECT title FROM topics WHERE analysis_id = %s", (analysis_id,)
    ).fetchall()
    assert topics == [("Topic A",)]

    claims = conn.execute(
        "SELECT text, speaker FROM claims WHERE analysis_id = %s", (analysis_id,)
    ).fetchall()
    assert claims == [("claim A", "Alice")]

    quotes = conn.execute(
        "SELECT text FROM quotes WHERE analysis_id = %s", (analysis_id,)
    ).fetchall()
    assert quotes == [("quote A",)]


def test_save_analysis_does_not_commit(head_dsn: str) -> None:
    # A raw connection here, not the `conn` fixture, since this test needs
    # explicit control over rollback.
    with psycopg.connect(head_dsn) as raw_conn:
        raw_conn.execute("INSERT INTO videos (video_id) VALUES ('vid1')")
        transcript_row = raw_conn.execute(
            """
            INSERT INTO transcripts (video_id, source, segments, full_text)
            VALUES ('vid1', 'whisper', '[]', 'x') RETURNING id
            """
        ).fetchone()
        assert transcript_row is not None
        transcript_id = int(transcript_row[0])

        save_analysis(raw_conn, make_analysis("vid1", transcript_id))
        raw_conn.rollback()

    with psycopg.connect(head_dsn) as verify_conn:
        count = verify_conn.execute("SELECT count(*) FROM analyses").fetchone()
        assert count == (0,)


def test_latest_analysis_returns_none_when_no_analysis(
    conn: psycopg.Connection, video_id: str
) -> None:
    assert latest_analysis(conn, video_id) is None


def test_latest_analysis_returns_newest_with_children_ordered(
    conn: psycopg.Connection, video_id: str
) -> None:
    transcript_id = save_transcript(conn, video_id, "whisper", "en", "none", (), None)
    save_analysis(conn, make_analysis(video_id, transcript_id, tldr="old"))
    newest_id = save_analysis(
        conn,
        make_analysis(
            video_id,
            transcript_id,
            tldr="new",
            topics=(Topic(seq=1, title="B"), Topic(seq=0, title="A")),
            claims=(
                Claim(text="late", start_sec=10.0),
                Claim(text="early", start_sec=1.0),
            ),
        ),
    )

    result = latest_analysis(conn, video_id)

    assert result is not None
    assert result.id == newest_id
    assert result.tldr == "new"
    assert [t.title for t in result.topics] == ["A", "B"]
    assert [c.text for c in result.claims] == ["early", "late"]


def test_latest_analysis_breaks_created_at_ties_by_id_desc(
    conn: psycopg.Connection, video_id: str
) -> None:
    transcript_id = save_transcript(conn, video_id, "whisper", "en", "none", (), None)
    first_id = save_analysis(conn, make_analysis(video_id, transcript_id, tldr="first"))
    second_id = save_analysis(conn, make_analysis(video_id, transcript_id, tldr="second"))
    # force an identical created_at to exercise the tie-break
    conn.execute(
        "UPDATE analyses SET created_at = now() WHERE id IN (%s, %s)",
        (first_id, second_id),
    )

    result = latest_analysis(conn, video_id)

    assert result is not None
    assert result.id == max(first_id, second_id)


def test_save_analysis_strips_nul_from_all_text_and_json_fields(
    conn: psycopg.Connection, video_id: str
) -> None:
    transcript_id = save_transcript(conn, video_id, "whisper", "en", "none", (), None)
    analysis = make_analysis(
        video_id,
        transcript_id,
        tldr="bad\x00tldr",
        speaker_roster={"note": "bad\x00note"},
        topics=(Topic(seq=0, title="bad\x00title", summary="bad\x00summary"),),
        claims=(Claim(text="bad\x00claim"),),
        quotes=(Quote(text="bad\x00quote"),),
    )

    analysis_id = save_analysis(conn, analysis)

    row = conn.execute(
        "SELECT tldr, speaker_roster FROM analyses WHERE id = %s", (analysis_id,)
    ).fetchone()
    assert row == ("badtldr", {"note": "badnote"})

    topic_row = conn.execute(
        "SELECT title, summary FROM topics WHERE analysis_id = %s", (analysis_id,)
    ).fetchone()
    assert topic_row == ("badtitle", "badsummary")

    claim_row = conn.execute(
        "SELECT text FROM claims WHERE analysis_id = %s", (analysis_id,)
    ).fetchone()
    assert claim_row == ("badclaim",)

    quote_row = conn.execute(
        "SELECT text FROM quotes WHERE analysis_id = %s", (analysis_id,)
    ).fetchone()
    assert quote_row == ("badquote",)


def test_save_analysis_default_speaker_and_confidence_values(
    conn: psycopg.Connection, video_id: str
) -> None:
    transcript_id = save_transcript(conn, video_id, "whisper", "en", "none", (), None)
    analysis = make_analysis(
        video_id,
        transcript_id,
        claims=(Claim(text="unattributed"),),
        quotes=(Quote(text="unattributed"),),
    )

    analysis_id = save_analysis(conn, analysis)

    claim_row = conn.execute(
        "SELECT speaker, confidence FROM claims WHERE analysis_id = %s", (analysis_id,)
    ).fetchone()
    assert claim_row == ("unknown", None)


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
def test_save_analysis_defaults_token_counts_to_zero(
    conn: psycopg.Connection, video_id: str, field: str
) -> None:
    transcript_id = save_transcript(conn, video_id, "whisper", "en", "none", (), None)

    analysis_id = save_analysis(conn, make_analysis(video_id, transcript_id))

    row = conn.execute(
        f"SELECT {field} FROM analyses WHERE id = %s", (analysis_id,)
    ).fetchone()
    assert row == (0,)


# --- get_analysis (issue #109) ----------------------------------------------------


def _two_runs_for_a_and_one_for_b(conn: psycopg.Connection) -> tuple[int, int, int]:
    from common.models import Claim, Quote, Topic

    for vid in ("videoAAAAAA", "videoBBBBBB"):
        conn.execute("INSERT INTO videos (video_id) VALUES (%s)", (vid,))
    tid_a = save_transcript(conn, "videoAAAAAA", "whisper", "en", "none", (), None)
    tid_b = save_transcript(conn, "videoBBBBBB", "whisper", "en", "none", (), None)
    older = save_analysis(
        conn,
        make_analysis(
            "videoAAAAAA",
            tid_a,
            tldr="older",
            topics=(Topic(seq=1, title="old t1"), Topic(seq=0, title="old t0")),
            claims=(
                Claim(text="old late", start_sec=50),
                Claim(text="old none"),
                Claim(text="old tie 1", start_sec=10),
                Claim(text="old tie 2", start_sec=10),
            ),
            quotes=(Quote(text="old q late", start_sec=9), Quote(text="old q none")),
        ),
    )
    newer = save_analysis(
        conn,
        make_analysis(
            "videoAAAAAA",
            tid_a,
            tldr="newer",
            topics=(Topic(seq=0, title="new t0"),),
            claims=(Claim(text="new claim", start_sec=1),),
            quotes=(Quote(text="new quote", start_sec=1),),
        ),
    )
    other = save_analysis(conn, make_analysis("videoBBBBBB", tid_b, tldr="b run"))
    return older, newer, other


def test_get_analysis_returns_the_requested_run_with_only_its_own_children(
    conn: psycopg.Connection,
) -> None:
    from common.repo.analyses import get_analysis

    older, _newer, _other = _two_runs_for_a_and_one_for_b(conn)
    # Rewrite the first tied row so its physical position is after the second:
    # only an explicit ``id`` tiebreak keeps "old tie 1" first.
    # (Changing the indexed ``start_sec`` and back makes the update non-HOT.)
    conn.execute("UPDATE claims SET start_sec = 11 WHERE text = 'old tie 1'")
    conn.execute("UPDATE claims SET start_sec = 10 WHERE text = 'old tie 1'")

    result = get_analysis(conn, "videoAAAAAA", older)

    assert result is not None
    assert result.id == older
    assert result.tldr == "older"
    assert [t.title for t in result.topics] == ["old t0", "old t1"]
    assert [c.text for c in result.claims] == ["old tie 1", "old tie 2", "old late", "old none"]
    assert [q.text for q in result.quotes] == ["old q late", "old q none"]
    assert result.created_at is not None


def test_get_analysis_is_none_for_another_videos_id_and_for_an_unused_id(
    conn: psycopg.Connection,
) -> None:
    from common.repo.analyses import get_analysis

    _older, newer, other = _two_runs_for_a_and_one_for_b(conn)

    assert get_analysis(conn, "videoAAAAAA", other) is None
    assert get_analysis(conn, "videoBBBBBB", newer) is None
    assert get_analysis(conn, "videoAAAAAA", other + 10_000) is None
    assert get_analysis(conn, "unknownvide", newer) is None


def test_get_analysis_matches_latest_analysis_for_the_latest_run(
    conn: psycopg.Connection,
) -> None:
    from common.repo.analyses import get_analysis

    _older, newer, _other = _two_runs_for_a_and_one_for_b(conn)

    assert get_analysis(conn, "videoAAAAAA", newer) == latest_analysis(conn, "videoAAAAAA")


def test_get_analysis_does_not_commit(head_dsn: str) -> None:
    from common.repo.analyses import get_analysis

    with psycopg.connect(head_dsn) as conn:
        conn.execute("INSERT INTO videos (video_id) VALUES ('vidrollbk01')")
        tid = save_transcript(conn, "vidrollbk01", "whisper", "en", "none", (), None)
        aid = save_analysis(conn, make_analysis("vidrollbk01", tid))
        assert get_analysis(conn, "vidrollbk01", aid) is not None
        conn.rollback()
        assert get_analysis(conn, "vidrollbk01", aid) is None
