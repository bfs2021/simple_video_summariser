"""Operator alerts for dead-lettered jobs (issue #67; architecture.md 8.4, 16.11).

``send_dead_letter_alerts`` is one pass, called by the planner loop (#38). It
looks for jobs that dead-lettered in an alerting error class
(``policy(cls).alert``: ``TOOL_FAILURE`` and ``RESOURCE``) and have not been
announced, and sends **one** message per class through the alert sink with the
count and a short sample.

Rate limit: a class that was announced within ``min_interval_sec`` is held
back; its new dead-letters wait (unmarked) and go out together, as one count,
once the window has passed. So a burst of 50 produces one alert, and a quiet
system alerts the first dead-letter at the next tick. ``jobs.alerted_at`` is
the only state, so the limit survives a planner restart.

Failure of the channel is logged and changes nothing: the job rows are never
touched except to stamp ``alerted_at`` after a successful send, and an unsent
alert is retried at the next tick.

Alert text is built from fields that already passed redaction at the source
(#63), and is scrubbed again here: only the last line of ``last_error`` is
quoted, URL credentials and query strings are removed, and it is truncated.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

import psycopg

from common.errors import ErrorClass, policy

_logger = logging.getLogger(__name__)

#: Newest dead-letters quoted in one alert.
SAMPLE_JOBS = 3
_MAX_ERROR_CHARS = 200

_HINTS = {
    ErrorClass.TOOL_FAILURE: (
        "A tool failed repeatedly; this usually means yt-dlp is out of date. "
        "Check for a newer release and bump the pin (architecture 16.11)."
    ),
    ErrorClass.RESOURCE: (
        "Out of a resource; check free disk space and memory. "
        "Retrying makes it worse, so the jobs were not retried."
    ),
}

_URL_REST = re.compile(r"(?<=://)\S+")
_URL_QUERY = re.compile(r"\?\S*")


def _strip_url_secrets(match: re.Match[str]) -> str:
    # Everything up to the LAST "@" of the token is userinfo, even if the
    # password holds "@", "/" or "#"; then drop any "#fragment".
    rest = match.group().rpartition("@")[2]
    return rest.partition("#")[0]


class AlertSink(Protocol):
    def send(self, title: str, message: str) -> None: ...


def alerting_classes() -> list[str]:
    return [str(c) for c in ErrorClass if policy(c).alert]


def _last_line(last_error: str | None) -> str:
    lines = [ln.strip() for ln in (last_error or "").splitlines() if ln.strip()]
    line = lines[-1] if lines else "(no error recorded)"
    line = _URL_QUERY.sub("?<redacted>", _URL_REST.sub(_strip_url_secrets, line))
    if len(line) > _MAX_ERROR_CHARS:
        line = line[: _MAX_ERROR_CHARS - 3] + "..."
    return line


def _message(cls: ErrorClass, rows: list[tuple[Any, ...]]) -> tuple[str, str]:
    """``rows``: (id, video_id, kind, last_error), newest first."""
    count = len(rows)
    noun = "dead-letter" if count == 1 else "dead-letters"
    title = f"{count} {cls} {noun}"
    lines = [f"{count} job(s) dead-lettered as {cls}.", _HINTS.get(cls, "")]
    for job_id, video_id, kind, last_error in rows[:SAMPLE_JOBS]:
        lines.append(f"job {job_id} ({kind}, video {video_id}): {_last_line(last_error)}")
    if count > SAMPLE_JOBS:
        lines.append(f"... and {count - SAMPLE_JOBS} more (Ops view lists them).")
    return title, "\n".join(ln for ln in lines if ln)


_PENDING_SQL = """
    SELECT id, video_id, kind, last_error FROM jobs
    WHERE state = 'dead' AND error_class = %(cls)s
      AND (alerted_at IS NULL OR alerted_at < finished_at)
    ORDER BY finished_at DESC, id DESC
"""

_RECENTLY_ALERTED_SQL = """
    SELECT EXISTS (
        SELECT 1 FROM jobs
        WHERE error_class = %(cls)s
          AND alerted_at > now() - %(sec)s * interval '1 second')
"""


def send_dead_letter_alerts(
    conn: psycopg.Connection[Any], sink: AlertSink | None, *, min_interval_sec: int
) -> int:
    """Send at most one alert per alerting class; return how many were sent.

    ``sink`` is ``None`` when no channel is configured: nothing is sent and
    nothing is marked, so a channel configured later still reports them.
    """
    if sink is None:
        return 0
    sent = 0
    for cls_name in alerting_classes():
        cls = ErrorClass(cls_name)
        with conn.transaction():
            held = conn.execute(
                _RECENTLY_ALERTED_SQL, {"cls": cls_name, "sec": min_interval_sec}
            ).fetchone()
            if held is not None and held[0]:
                continue
            rows = conn.execute(_PENDING_SQL, {"cls": cls_name}).fetchall()
        if not rows:
            continue
        title, message = _message(cls, rows)
        try:
            sink.send(title, message)
        except Exception as err:  # noqa: BLE001 - a broken channel must never break the planner
            _logger.warning(
                "planner.alert_failed",
                extra={"error_class": cls_name, "pending": len(rows), "error": str(err)},
            )
            continue
        with conn.transaction():
            conn.execute(
                "UPDATE jobs SET alerted_at = now() WHERE id = ANY(%(ids)s)",
                {"ids": [r[0] for r in rows]},
            )
        sent += 1
        _logger.info("planner.alert_sent", extra={"error_class": cls_name, "count": len(rows)})
    return sent
