"""The two medians behind the backfill LLM estimate (issue #117)."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest

from common.repo.analyses import median_cost_per_char
from common.repo.transcripts import median_chars_per_audio_sec

pytestmark = pytest.mark.integration


def _video(conn: psycopg.Connection[Any], vid: str, duration: int | None) -> None:
    conn.execute("INSERT INTO videos (video_id, duration_sec) VALUES (%s, %s)", (vid, duration))


def _transcript(
    conn: psycopg.Connection[Any], vid: str, text: str, *, source: str = "whisper", age: int = 0
) -> int:
    row = conn.execute(
        "INSERT INTO transcripts (video_id, source, segments, full_text, created_at)"
        " VALUES (%s, %s, '[]', %s, now() - make_interval(secs => %s)) RETURNING id",
        (vid, source, text, age),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _analysis(
    conn: psycopg.Connection[Any],
    vid: str,
    transcript_id: int,
    cost: str | None,
    *,
    model: str = "m",
    age: int = 0,
) -> None:
    conn.execute(
        "INSERT INTO analyses (video_id, transcript_id, chunk_strategy, model, prompt_version,"
        " tldr, cost_usd, created_at)"
        " VALUES (%s, %s, 's', %s, 'v1', 't', %s, now() - make_interval(secs => %s))",
        (vid, transcript_id, model, cost, age),
    )


# --- chars per audio second ---------------------------------------------------------


def test_chars_per_sec_with_no_rows_is_none(conn: psycopg.Connection[Any]) -> None:
    assert median_chars_per_audio_sec(conn) == (None, 0)


def test_chars_per_sec_one_sample(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "c0000000001", 10)
    _transcript(conn, "c0000000001", "x" * 150)

    assert median_chars_per_audio_sec(conn) == (15.0, 1)


def test_chars_per_sec_even_count_is_the_mean_of_the_middle_two(
    conn: psycopg.Connection[Any],
) -> None:
    for i, chars in enumerate([10, 20, 30, 100]):
        _video(conn, f"c000000000{i}", 10)
        _transcript(conn, f"c000000000{i}", "x" * chars, age=i)

    assert median_chars_per_audio_sec(conn) == (2.5, 4)  # (2 + 3) / 2


def test_chars_per_sec_ignores_unusable_rows_and_any_source_counts(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "ok000000001", 10)
    _transcript(conn, "ok000000001", "x" * 50, source="manual_subs")
    _video(conn, "nodur000001", None)
    _transcript(conn, "nodur000001", "x" * 999)
    _video(conn, "zerodur0001", 0)
    _transcript(conn, "zerodur0001", "x" * 999)
    _video(conn, "negdur00001", -5)
    _transcript(conn, "negdur00001", "x" * 999)
    _video(conn, "empty000001", 10)
    _transcript(conn, "empty000001", "")

    assert median_chars_per_audio_sec(conn) == (5.0, 1)


def test_chars_per_sec_uses_only_the_newest_twenty(conn: psycopg.Connection[Any]) -> None:
    for i in range(25):
        _video(conn, f"n{i:010d}", 10)
        # the five oldest are wild outliers; the newest 20 all give 1.0
        _transcript(conn, f"n{i:010d}", "x" * (10 if i < 20 else 100000), age=100 - i if i < 20 else 1000 + i)

    value, samples = median_chars_per_audio_sec(conn)

    assert samples == 20
    assert value == 1.0


# --- cost per char -------------------------------------------------------------------


def test_cost_per_char_with_no_rows_is_none(conn: psycopg.Connection[Any]) -> None:
    assert median_cost_per_char(conn, "m") == (None, 0)


def test_cost_per_char_one_sample(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "a0000000001", 10)
    t = _transcript(conn, "a0000000001", "x" * 1000)
    _analysis(conn, "a0000000001", t, "0.01000")

    value, samples = median_cost_per_char(conn, "m")

    assert samples == 1
    assert value == pytest.approx(1e-5)


def test_cost_per_char_even_count_is_the_mean_of_the_middle_two(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "a0000000001", 10)
    t = _transcript(conn, "a0000000001", "x" * 1000)
    for i, cost in enumerate(["0.01", "0.02", "0.03", "0.50"]):
        _analysis(conn, "a0000000001", t, cost, age=i)

    value, samples = median_cost_per_char(conn, "m")

    assert samples == 4
    assert value == pytest.approx(2.5e-5)


def test_cost_per_char_ignores_null_zero_other_models_and_empty_transcripts(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "a0000000001", 10)
    good = _transcript(conn, "a0000000001", "x" * 1000)
    _analysis(conn, "a0000000001", good, "0.01")
    _analysis(conn, "a0000000001", good, None)
    _analysis(conn, "a0000000001", good, "0")
    _analysis(conn, "a0000000001", good, "0.99", model="other")
    _video(conn, "a0000000002", 10)
    empty = _transcript(conn, "a0000000002", "")
    _analysis(conn, "a0000000002", empty, "0.99")

    value, samples = median_cost_per_char(conn, "m")

    assert samples == 1
    assert value == pytest.approx(1e-5)


def test_cost_per_char_uses_only_the_newest_twenty(conn: psycopg.Connection[Any]) -> None:
    _video(conn, "a0000000001", 10)
    t = _transcript(conn, "a0000000001", "x" * 1000)
    for i in range(25):
        _analysis(conn, "a0000000001", t, "0.01" if i < 20 else "9.99", age=i if i < 20 else 1000 + i)

    value, samples = median_cost_per_char(conn, "m")

    assert samples == 20
    assert value == pytest.approx(1e-5)


# --- the 20-row window is taken before unusable rows are skipped (#117 QA) ------------


def test_chars_per_sec_empty_rows_do_not_shrink_the_newest_twenty(
    conn: psycopg.Connection[Any],
) -> None:
    # 5 empty transcripts are the newest; the SQL must skip them *before* LIMIT 20,
    # so the 20 usable ones that follow still all count
    for i in range(25):
        _video(conn, f"e{i:010d}", 10)
        _transcript(conn, f"e{i:010d}", "" if i < 5 else "x" * 100, age=i)

    assert median_chars_per_audio_sec(conn) == (10.0, 20)


def test_chars_per_sec_takes_the_newest_twenty_not_the_oldest(
    conn: psycopg.Connection[Any],
) -> None:
    # 30 rows so the oldest 20 (10 outliers + 10 normal) have a different median
    # from the newest 20 (all normal): ASC instead of DESC cannot give 1.0
    for i in range(30):
        _video(conn, f"d{i:010d}", 10)
        _transcript(conn, f"d{i:010d}", "x" * (10 if i < 20 else 1000), age=i)

    assert median_chars_per_audio_sec(conn) == (1.0, 20)


def test_cost_per_char_zero_cost_rows_do_not_shrink_the_newest_twenty(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "a0000000001", 10)
    t = _transcript(conn, "a0000000001", "x" * 1000)
    for i in range(25):
        _analysis(conn, "a0000000001", t, "0" if i < 5 else "0.01", age=i)

    value, samples = median_cost_per_char(conn, "m")

    assert samples == 20
    assert value == pytest.approx(1e-5)


def test_cost_per_char_takes_the_newest_twenty_not_the_oldest(
    conn: psycopg.Connection[Any],
) -> None:
    _video(conn, "a0000000001", 10)
    t = _transcript(conn, "a0000000001", "x" * 1000)
    for i in range(30):  # the oldest 20 have a different median from the newest 20
        _analysis(conn, "a0000000001", t, "0.01" if i < 20 else "5.00", age=i)

    value, samples = median_cost_per_char(conn, "m")

    assert samples == 20
    assert value == pytest.approx(1e-5)
