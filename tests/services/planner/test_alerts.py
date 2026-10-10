"""Dead-letter operator alerts (issue #67). Needs Postgres."""

from __future__ import annotations

import logging
from typing import Any

import psycopg
import pytest

from services.planner.alerts import send_dead_letter_alerts

pytestmark = pytest.mark.integration

HOUR = 3600


class Sink:
    def __init__(self, raises: Exception | None = None) -> None:
        self.sent: list[tuple[str, str]] = []
        self.raises = raises

    def send(self, title: str, message: str) -> None:
        if self.raises is not None:
            raise self.raises
        self.sent.append((title, message))


_n = 0


def _dead(
    conn: psycopg.Connection[Any],
    error_class: str,
    *,
    last_error: str = "Traceback\nToolFailureError: extractor broke",
    kind: str = "ingest",
) -> int:
    global _n
    _n += 1
    vid = f"al{_n:09d}"
    conn.execute("INSERT INTO videos (video_id) VALUES (%s)", (vid,))
    row = conn.execute(
        "INSERT INTO jobs (video_id, kind, state, error_class, last_error, attempts, "
        "finished_at) VALUES (%s, %s, 'dead', %s, %s, 3, now()) RETURNING id",
        (vid, kind, error_class, last_error),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _alerted_at(conn: psycopg.Connection[Any], job_id: int) -> Any:
    row = conn.execute("SELECT alerted_at FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert row is not None
    return row[0]


def _run(conn: psycopg.Connection[Any], sink: Sink | None, interval: int = HOUR) -> int:
    return send_dead_letter_alerts(conn, sink, min_interval_sec=interval)


def test_one_tool_failure_dead_letter_sends_one_alert_then_none(conn: psycopg.Connection[Any]) -> None:
    job = _dead(conn, "TOOL_FAILURE")
    sink = Sink()

    assert _run(conn, sink) == 1
    assert _run(conn, sink) == 0

    assert len(sink.sent) == 1
    title, message = sink.sent[0]
    assert "TOOL_FAILURE" in title + message
    assert str(job) in message
    assert "yt-dlp" in message
    assert _alerted_at(conn, job) is not None


def test_a_resource_dead_letter_alerts_with_a_resource_hint(conn: psycopg.Connection[Any]) -> None:
    _dead(conn, "RESOURCE", last_error="OSError: No space left on device")
    sink = Sink()

    assert _run(conn, sink) == 1

    title, message = sink.sent[0]
    assert "RESOURCE" in title + message
    assert "disk" in message.lower()
    assert "yt-dlp" not in message


def test_fifty_dead_letters_produce_one_alert_with_the_count(conn: psycopg.Connection[Any]) -> None:
    jobs = [_dead(conn, "TOOL_FAILURE") for _ in range(50)]
    sink = Sink()

    assert _run(conn, sink) == 1

    assert len(sink.sent) == 1
    assert "50" in sink.sent[0][0] + sink.sent[0][1]
    assert all(_alerted_at(conn, j) is not None for j in jobs)
    assert len(sink.sent[0][1]) < 2000  # a bounded sample, not 50 job lines


def test_a_new_dead_letter_inside_the_window_waits_then_alerts_once_with_its_count(
    conn: psycopg.Connection[Any],
) -> None:
    _dead(conn, "TOOL_FAILURE")
    sink = Sink()
    assert _run(conn, sink) == 1

    later = [_dead(conn, "TOOL_FAILURE") for _ in range(3)]
    assert _run(conn, sink) == 0
    assert all(_alerted_at(conn, j) is None for j in later)

    conn.execute("UPDATE jobs SET alerted_at = now() - interval '61 minutes' WHERE alerted_at IS NOT NULL")
    assert _run(conn, sink) == 1

    assert len(sink.sent) == 2
    assert "3" in sink.sent[1][0] + sink.sent[1][1]
    assert all(_alerted_at(conn, j) is not None for j in later)


def test_the_window_boundary_is_exclusive_of_exactly_the_interval(conn: psycopg.Connection[Any]) -> None:
    _dead(conn, "TOOL_FAILURE")
    sink = Sink()
    _run(conn, sink)
    new = _dead(conn, "TOOL_FAILURE")

    conn.execute("UPDATE jobs SET alerted_at = now() - interval '59 minutes' WHERE alerted_at IS NOT NULL")
    assert _run(conn, sink) == 0
    conn.execute("UPDATE jobs SET alerted_at = now() - interval '61 minutes' WHERE alerted_at IS NOT NULL")
    assert _run(conn, sink) == 1
    assert _alerted_at(conn, new) is not None


def test_each_class_has_its_own_window(conn: psycopg.Connection[Any]) -> None:
    _dead(conn, "TOOL_FAILURE")
    sink = Sink()
    _run(conn, sink)

    _dead(conn, "RESOURCE")

    assert _run(conn, sink) == 1
    assert len(sink.sent) == 2


@pytest.mark.parametrize(
    "cls",
    ["BUG", "PERMANENT_SOURCE", "RATE_LIMITED", "TRANSIENT_NETWORK", "LLM_INVALID_OUTPUT", "LLM_UNAVAILABLE"],
)
def test_other_classes_never_alert(conn: psycopg.Connection[Any], cls: str) -> None:
    job = _dead(conn, cls)
    sink = Sink()

    assert _run(conn, sink) == 0

    assert sink.sent == []
    assert _alerted_at(conn, job) is None


def test_only_dead_jobs_alert(conn: psycopg.Connection[Any]) -> None:
    job = _dead(conn, "TOOL_FAILURE")
    conn.execute("UPDATE jobs SET state = 'pending', finished_at = NULL WHERE id = %s", (job,))
    sink = Sink()

    assert _run(conn, sink) == 0


def test_no_channel_does_nothing_marks_nothing_and_is_not_an_error(
    conn: psycopg.Connection[Any], caplog: pytest.LogCaptureFixture
) -> None:
    job = _dead(conn, "TOOL_FAILURE")

    with caplog.at_level(logging.DEBUG):
        assert _run(conn, None) == 0

    assert _alerted_at(conn, job) is None
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_a_failing_channel_is_logged_and_leaves_the_job_alone(
    conn: psycopg.Connection[Any], caplog: pytest.LogCaptureFixture
) -> None:
    job = _dead(conn, "TOOL_FAILURE")
    before = conn.execute(
        "SELECT state, error_class, last_error, attempts, finished_at FROM jobs WHERE id = %s", (job,)
    ).fetchone()
    sink = Sink(raises=RuntimeError("channel down"))

    with caplog.at_level(logging.WARNING):
        assert _run(conn, sink) == 0

    assert any("channel down" in r.getMessage() or "channel down" in str(r.__dict__) for r in caplog.records)
    after = conn.execute(
        "SELECT state, error_class, last_error, attempts, finished_at FROM jobs WHERE id = %s", (job,)
    ).fetchone()
    assert after == before
    assert _alerted_at(conn, job) is None

    good = Sink()
    assert _run(conn, good) == 1  # retried on the next tick


def test_a_job_that_is_retried_and_dies_again_alerts_again(conn: psycopg.Connection[Any]) -> None:
    job = _dead(conn, "TOOL_FAILURE")
    sink = Sink()
    _run(conn, sink)

    conn.execute("UPDATE jobs SET alerted_at = now() - interval '2 hours'")
    conn.execute("UPDATE jobs SET finished_at = now() WHERE id = %s", (job,))

    assert _run(conn, sink) == 1


def test_secrets_in_last_error_do_not_reach_the_alert(conn: psycopg.Connection[Any]) -> None:
    _dead(
        conn,
        "TOOL_FAILURE",
        last_error=(
            "Traceback (most recent call last):\n  File x\n"
            "ToolFailureError: GET https://user:hunter2@host.test/v?sig=TOPSECRET&k=1 failed"
        ),
    )
    sink = Sink()

    _run(conn, sink)

    text = sink.sent[0][0] + sink.sent[0][1]
    assert "hunter2" not in text
    assert "TOPSECRET" not in text
    assert "ToolFailureError" in text  # the useful part survives


def test_a_very_long_last_error_line_is_truncated(conn: psycopg.Connection[Any]) -> None:
    _dead(conn, "TOOL_FAILURE", last_error="E: " + "x" * 5000)
    sink = Sink()

    _run(conn, sink)

    assert len(sink.sent[0][1]) < 1500


def test_a_dead_letter_announced_at_the_instant_it_finished_is_not_announced_again(
    conn: psycopg.Connection[Any],
) -> None:
    job = _dead(conn, "TOOL_FAILURE")
    conn.execute(
        "UPDATE jobs SET finished_at = now() - interval '2 hours', "
        "alerted_at = now() - interval '2 hours' WHERE id = %s",
        (job,),
    )

    assert _run(conn, Sink()) == 0


def test_an_alert_exactly_one_interval_old_no_longer_holds_back_the_next(
    conn: psycopg.Connection[Any],
) -> None:
    # now() is the transaction start, constant inside this test, so "exactly
    # min_interval_sec ago" is a precise boundary rather than a race.
    old = _dead(conn, "TOOL_FAILURE")
    conn.execute("UPDATE jobs SET alerted_at = now() - interval '3600 seconds' WHERE id = %s", (old,))
    conn.execute("UPDATE jobs SET finished_at = now() - interval '2 hours' WHERE id = %s", (old,))
    new = _dead(conn, "TOOL_FAILURE")
    sink = Sink()

    assert send_dead_letter_alerts(conn, sink, min_interval_sec=3600) == 1
    assert _alerted_at(conn, new) is not None


def test_the_interval_setting_is_what_sets_the_window(conn: psycopg.Connection[Any]) -> None:
    old = _dead(conn, "TOOL_FAILURE")
    conn.execute("UPDATE jobs SET alerted_at = now() - interval '30 seconds' WHERE id = %s", (old,))
    conn.execute("UPDATE jobs SET finished_at = now() - interval '2 hours' WHERE id = %s", (old,))
    _dead(conn, "TOOL_FAILURE")

    assert send_dead_letter_alerts(conn, Sink(), min_interval_sec=3600) == 0
    assert send_dead_letter_alerts(conn, Sink(), min_interval_sec=10) == 1


@pytest.mark.parametrize(("count", "more"), [(1, None), (3, None), (4, "1 more"), (5, "2 more")])
def test_the_sample_is_three_jobs_and_the_rest_are_counted(
    conn: psycopg.Connection[Any], count: int, more: str | None
) -> None:
    for _ in range(count):
        _dead(conn, "TOOL_FAILURE")
    sink = Sink()

    _run(conn, sink)

    message = sink.sent[0][1]
    assert message.count("\njob ") == min(count, 3)
    assert (more in message) if more else ("more" not in message)


@pytest.mark.parametrize(("length", "shown"), [(199, 199), (200, 200), (201, 200)])
def test_the_quoted_error_line_is_capped_at_200_chars_with_an_ellipsis(
    conn: psycopg.Connection[Any], length: int, shown: int
) -> None:
    _dead(conn, "TOOL_FAILURE", last_error="y" * length)
    sink = Sink()

    _run(conn, sink)

    quoted = sink.sent[0][1].splitlines()[-1].split(": ", 1)[1]
    assert len(quoted) == shown
    assert quoted.endswith("...") == (length > 200)


_TOKEN = "S3CRETTOKEN"


@pytest.mark.parametrize(
    "url",
    [
        f"https://127.0.0.1/my topic {_TOKEN}",
        f"https://127.0.0.1/my\ntopic{_TOKEN}",
        f"https://127.0.0.1:{_TOKEN}/x",
        f"https://user:p#{_TOKEN}@127.0.0.1/x",
        f"https://user:pa/{_TOKEN}@127.0.0.1/x",
    ],
    ids=["space", "newline", "port", "hash-password", "slash-password"],
)
def test_the_warning_record_never_carries_the_webhook_url(
    conn: psycopg.Connection[Any], caplog: pytest.LogCaptureFixture, url: str
) -> None:
    from adapters.alerts.webhook import WebhookAlerter

    _dead(conn, "TOOL_FAILURE")

    with caplog.at_level(logging.DEBUG):
        assert _run(conn, WebhookAlerter(url)) == 0  # type: ignore[arg-type]

    failed = [r for r in caplog.records if r.getMessage() == "planner.alert_failed"]
    assert failed
    for rec in caplog.records:
        assert _TOKEN not in repr(rec.__dict__)
        assert _TOKEN not in rec.getMessage()
        assert "user:" not in repr(rec.__dict__)
