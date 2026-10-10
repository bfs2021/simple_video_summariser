"""Map yt-dlp failures onto the error taxonomy (architecture.md 8.4).

The patterns below are checked against real yt-dlp stderr kept in
``tests/fixtures/ytdlp_errors/``. When yt-dlp changes its wording, capture the
new line there first, then adjust the pattern.
"""

from __future__ import annotations

import json
import re
from typing import Any

from common.errors import (
    JobError,
    PermanentSourceError,
    RateLimitedError,
    ResourceError,
    ToolFailureError,
    TransientNetworkError,
    UnavailableReason,
)

# How much of stderr a JobError message keeps; the rest is progress noise.
_MESSAGE_CHARS = 1000


class UpcomingVideoError(TransientNetworkError):
    """A scheduled live event or premiere that has not started yet.

    A TRANSIENT_NETWORK for callers that do not know better; the ingest
    handler turns it into a ``Defer`` (#65) so it costs no attempt.
    """


def _rx(*alternatives: str) -> re.Pattern[str]:
    return re.compile("|".join(alternatives), re.IGNORECASE)


# Order matters: the first match wins, checked in from_ytdlp in this order:
# _RESOURCE (a full disk/quota beats everything, even a rate-limit or removal
# message logged in the same run), _UPCOMING, _RATE_LIMITED (checked before
# _PERMANENT: current yt-dlp prefixes the rate-limit message with "Video
# unavailable.", which would otherwise match _PERMANENT's REMOVED pattern -
# issue #69), _PERMANENT, _FORBIDDEN, _TRANSIENT, then TOOL_FAILURE.
_UPCOMING = _rx(r"This live event will begin in", r"Premieres in")
_PERMANENT: list[tuple[re.Pattern[str], UnavailableReason]] = [
    (_rx(r"This video has been removed"), UnavailableReason.REMOVED),
    # Checked before "Video unavailable", which prefixes older geoblock lines.
    # "in your country" alone is not enough: removal notices say it too.
    (
        _rx(
            r"not available in your country",
            r"not made this video available in your country",
            r"not available from your location",
        ),
        UnavailableReason.GEOBLOCKED,
    ),
    # A channel's uploads playlist (UU...) that is missing. yt-dlp says the same
    # for a channel that does not exist and for a real one without uploads.
    (_rx(r"The playlist does not exist"), UnavailableReason.REMOVED),
    (_rx(r"Sign in to confirm your age"), UnavailableReason.AGEGATED),
    (_rx(r"Private video", r"members-only"), UnavailableReason.PRIVATE),
    (
        _rx(r"Video unavailable", r"This video is unavailable"),
        UnavailableReason.REMOVED,
    ),
]
# YouTube writes both the bot check and the rate-limit apostrophe as typographic
# ('). "rate-limited by YouTube" is yt-dlp's own suffix, appended for both the
# "The current session" and "Your account" variants (issue #69).
_RATE_LIMITED = _rx(
    r"HTTP Error 429",
    r"Sign in to confirm you['’]re not a bot",
    r"This content isn['’]t available, try again later",
    r"rate-limited by YouTube",
)
# 403 storms mean the extractor is broken (architecture.md 16.11), not the network.
_FORBIDDEN = _rx(r"HTTP Error 403")
_TRANSIENT = _rx(
    r"HTTP Error 5\d\d", r"timed out", r"Temporary failure in name resolution"
)
# Lines up with common.errors._RESOURCE_ERRNOS (ENOSPC, EDQUOT) and with
# adapters/youtube/audio.py's _RESOURCE_STDERR.
_RESOURCE = _rx(r"No space left on device", r"Disk quota exceeded")


def _error_part(stderr: str) -> str:
    """The text from the first ``ERROR:`` on, or all of it if there is none.

    Earlier WARNING lines describe retried hiccups, not the failure. ``ERROR:``
    is searched anywhere, since a progress line without a newline can precede it.
    """
    start = stderr.find("ERROR:")
    return stderr[start:] if start >= 0 else stderr


def from_ytdlp(stderr: str, returncode: int) -> JobError:
    """Classify a failed yt-dlp run from its stderr and exit code.

    An empty or unrecognized message is TOOL_FAILURE, never BUG: that class
    alerts, so someone looks at what yt-dlp said.
    """
    text = _error_part(stderr).strip()
    excerpt = text[:_MESSAGE_CHARS] or "<no stderr>"
    message = f"yt-dlp exited with {returncode}: {excerpt}"

    if _RESOURCE.search(text):
        return ResourceError(message)
    if _UPCOMING.search(text):
        return UpcomingVideoError(message)
    if _RATE_LIMITED.search(text):
        return RateLimitedError(message)
    for pattern, reason in _PERMANENT:
        if pattern.search(text):
            return PermanentSourceError(reason, message)
    if _FORBIDDEN.search(text):
        return ToolFailureError(message)
    if _TRANSIENT.search(text):
        return TransientNetworkError(message)
    # "Unable to extract", "ExtractorError", "Failed to parse JSON" and
    # anything unrecognized all land here.
    return ToolFailureError(message)


def decode_ytdlp_json(stdout: str) -> Any:
    """Parse yt-dlp's ``--dump-json`` output.

    Output that does not decode means yt-dlp itself misbehaved, so it is a
    TOOL_FAILURE rather than a raw ``JSONDecodeError`` (which would be a BUG).
    """
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ToolFailureError(f"yt-dlp output is not valid JSON: {exc}") from exc
