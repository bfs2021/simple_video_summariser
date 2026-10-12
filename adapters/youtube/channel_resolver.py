"""Resolve a YouTube ``@handle``, ``/c/`` or ``/user/`` name to its channel ID (#81).

The URL is built here from the parsed name, never from the user's text, and
yt-dlp is asked for one flat JSON object with no entries. The output is
untrusted: the channel ID must be a canonical ``UC...`` ID, and a missing or
disagreeing one is a ``ToolFailureError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from adapters.youtube.errors import decode_ytdlp_json
from adapters.youtube.ytdlp import ProcessRunner, run_process, run_ytdlp
from common.errors import ToolFailureError
from common.youtube_refs import is_channel_id, validate_channel_id

DEFAULT_TIMEOUT_SEC = 20.0
_TITLE_CHARS = 200
_FLAGS = ("--flat-playlist", "--dump-single-json", "--playlist-items", "0", "--no-warnings")
_PATH_PREFIX = {"handle": "@", "c": "c/", "user": "user/"}


@dataclass(frozen=True, slots=True)
class ResolvedChannel:
    channel_id: str
    title: str | None


def resolve_channel(
    kind: str,
    name: str,
    *,
    runner: ProcessRunner = run_process,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> ResolvedChannel:
    """The canonical channel ID and name for a ``handle``, ``c`` or ``user`` name.

    Raises ``PermanentSourceError`` for a name that does not exist,
    ``TransientNetworkError`` on timeout or a network error, ``RateLimitedError``
    when throttled and ``ToolFailureError`` for bad output.
    """
    if kind not in _PATH_PREFIX:
        raise ValueError(f"unknown channel name kind: {kind!r}")
    url = f"https://www.youtube.com/{_PATH_PREFIX[kind]}{quote(name, safe='')}"
    stdout = run_ytdlp([*_FLAGS, url], timeout=timeout, runner=runner)
    return _parse(decode_ytdlp_json(stdout))


def _parse(info: Any) -> ResolvedChannel:
    if not isinstance(info, dict):
        raise ToolFailureError("yt-dlp returned no channel object")
    channel_id = info.get("channel_id")
    fallback = info.get("id")
    if channel_id is None and is_channel_id(fallback):
        channel_id = fallback
    if isinstance(fallback, str) and is_channel_id(fallback) and fallback != channel_id:
        raise ToolFailureError("yt-dlp reported two different channel IDs")
    try:
        validated = validate_channel_id(channel_id)
    except ValueError:
        raise ToolFailureError("yt-dlp returned no valid channel ID") from None
    title = info.get("channel")
    if not isinstance(title, str):
        title = None
    else:
        title = title.strip()[:_TITLE_CHARS] or None
    return ResolvedChannel(channel_id=validated, title=title)
