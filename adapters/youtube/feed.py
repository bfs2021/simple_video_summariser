"""YouTube implementation of the ``ChannelFeed`` port (architecture.md 3, C4).

``fetch`` reads a channel's public RSS feed
(``https://www.youtube.com/feeds/videos.xml?channel_id=<ID>``): no API key, no
quota, but only about the 15 latest uploads, so this is for monitoring, not
backfill. Entries come back unfiltered, in document order. Every failure is a
classified ``JobError`` (architecture.md 8.4), except a malformed channel ID,
which is the caller's bug and raises ``ValueError`` before any request.

The feed is untrusted input: the channel ID is validated before it reaches the
URL, the body is capped, DOCTYPE/ENTITY declarations are refused before any
entity expands, and each entry is checked on its own so one bad entry does not
cost the rest.
"""

from __future__ import annotations

import http.client
import logging
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit
from xml.parsers import expat

from common.errors import (
    RateLimitedError,
    ToolFailureError,
    TransientNetworkError,
)
from common.models import FeedEntry
from common.youtube_refs import is_video_id, validate_channel_id

_logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SEC = 30.0
MAX_BODY_BYTES = 1024 * 1024

_FEED_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={}"
_USER_AGENT = "simple-video-summary/1.0 (channel feed poller)"

_ATOM = "{http://www.w3.org/2005/Atom}"
_YT = "{http://www.youtube.com/xml/schemas/2015}"

_YOUTUBE_HOSTS = frozenset({"www.youtube.com", "youtube.com", "m.youtube.com"})

_SECONDS = re.compile(r"[0-9]+", re.ASCII)

# The opener takes a Request and a ``timeout`` keyword, like ``urlopen``.
Opener = Callable[..., Any]


class FeedNotFoundError(TransientNetworkError):
    """HTTP 404 from the feed endpoint.

    The endpoint is known to answer 404 intermittently for valid channels, so
    this is retried like any transient failure; the planner tells it apart by
    type and records it in ``last_poll_err``.
    """


class YouTubeFeed:
    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        opener: Opener = urllib.request.urlopen,
    ) -> None:
        self._timeout = timeout
        self._opener = opener

    def fetch(self, channel_id: str) -> list[FeedEntry]:
        validate_channel_id(channel_id)
        request = urllib.request.Request(
            _FEED_URL.format(channel_id),
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/atom+xml, application/xml;q=0.9",
            },
        )
        return parse_feed(self._read(request, channel_id), channel_id)

    def _read(self, request: urllib.request.Request, channel_id: str) -> bytes:
        try:
            response = self._opener(request, timeout=self._timeout)
        except urllib.error.HTTPError as exc:
            exc.close()
            raise _http_failure(exc.code, exc.headers, channel_id) from exc
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            # URLError (DNS, refused), TimeoutError and resets are all OSError.
            raise TransientNetworkError(
                f"feed request for {channel_id} failed: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            status = response.status
            if not 200 <= status < 300:
                raise _http_failure(status, response.headers, channel_id)
            try:
                body = response.read(MAX_BODY_BYTES + 1)
            except (OSError, http.client.HTTPException) as exc:
                raise TransientNetworkError(
                    f"reading feed for {channel_id} failed: {type(exc).__name__}: {exc}"
                ) from exc
        finally:
            response.close()
        if len(body) > MAX_BODY_BYTES:
            raise ToolFailureError(
                f"feed for {channel_id} is larger than {MAX_BODY_BYTES} bytes"
            )
        return bytes(body)


def _http_failure(status: int, headers: Any, channel_id: str) -> Exception:
    message = f"feed request for {channel_id} returned HTTP {status}"
    if status == 404:
        return FeedNotFoundError(message)
    if status == 429:
        return RateLimitedError(message, retry_after_sec=_retry_after(headers))
    if status >= 500:
        return TransientNetworkError(message)
    return ToolFailureError(message)


def _retry_after(headers: Any) -> int | None:
    """Integer seconds only; an HTTP-date, negative or junk value is ``None``."""
    value = headers.get("Retry-After") if headers is not None else None
    if not isinstance(value, str):
        return None
    value = value.strip()
    return int(value) if _SECONDS.fullmatch(value) else None


def _reject_declaration(*_args: object) -> None:
    raise ToolFailureError("feed contains a DOCTYPE or ENTITY declaration")


def _parse_document(data: bytes) -> ET.Element:
    # A pass with plain expat first: its handlers fire before any entity is
    # expanded, and it decodes UTF-16 and friends, which a byte search for
    # "<!DOCTYPE" would miss. The real feed never declares either.
    try:
        checker = expat.ParserCreate()
        checker.StartDoctypeDeclHandler = _reject_declaration
        checker.EntityDeclHandler = _reject_declaration
        checker.Parse(data, True)
        root = ET.fromstring(data)
    except ToolFailureError:
        raise
    except (
        expat.ExpatError,
        ET.ParseError,
        ValueError,
        LookupError,
        RecursionError,
    ) as exc:
        raise ToolFailureError(f"feed is not well-formed XML: {exc}") from exc
    return root


def parse_feed(data: bytes, channel_id: str) -> list[FeedEntry]:
    """Turn an Atom feed body into entries, newest first as served.

    Whole-feed problems raise ``ToolFailureError``. A bad entry is skipped with
    a WARNING naming the channel and the reason.
    """
    validate_channel_id(channel_id)
    root = _parse_document(data)
    if root.tag != f"{_ATOM}feed":
        raise ToolFailureError(f"feed for {channel_id} is not an Atom <feed>")

    entries: list[FeedEntry] = []
    seen: set[str] = set()
    for element in root.findall(f"{_ATOM}entry"):
        entry, problem = _parse_entry(element, channel_id)
        if entry is None:
            _logger.warning("skipping feed entry for %s: %s", channel_id, problem)
        elif entry.video_id not in seen:
            seen.add(entry.video_id)
            entries.append(entry)
    return entries


def _parse_entry(
    element: ET.Element, channel_id: str
) -> tuple[FeedEntry | None, str]:
    video_id = element.findtext(f"{_YT}videoId")
    if video_id is None or not is_video_id(video_id):
        return None, f"invalid video ID {_shown(video_id)}"

    # Entries carry the full "UC..." ID (only the feed-level element drops the
    # prefix, and that one is not read), as in the recorded feeds.
    entry_channel = (element.findtext(f"{_YT}channelId") or "").strip()
    if entry_channel != channel_id:
        return None, (
            f"video {video_id}: channelId mismatch "
            f"(feed says {_shown(entry_channel)})"
        )

    published_at = _parse_published(element.findtext(f"{_ATOM}published"))
    if published_at is None:
        return None, f"video {video_id}: missing or unparseable published time"

    title = (element.findtext(f"{_ATOM}title") or "").strip()
    return (
        FeedEntry(
            video_id, channel_id, title, published_at, is_short=_is_short_link(element, video_id)
        ),
        "",
    )


def _is_short_link(element: ET.Element, video_id: str) -> bool:
    """Whether the entry's ``<link rel="alternate">`` is ``/shorts/<video_id>`` (#80).

    Fails open: no such link, an unparseable ``href``, another host, another
    path or a different video ID all give ``False``, so an entry is never
    dropped for a link that is not clearly a Short's.
    """
    for link in element.findall(f"{_ATOM}link"):
        if link.get("rel") != "alternate":
            continue
        try:
            parts = urlsplit((link.get("href") or "").strip())
            host = parts.hostname
        except ValueError:
            continue
        if host in _YOUTUBE_HOSTS and parts.path == f"/shorts/{video_id}":
            return True
    return False


def _parse_published(text: str | None) -> datetime | None:
    """Aware UTC time, or ``None`` if absent, unparseable or without an offset."""
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.strip())
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(UTC)
    except (ValueError, OverflowError):
        return None


def _shown(value: str | None) -> str:
    """A bounded repr of untrusted text for log lines."""
    return repr((value or "")[:40])
