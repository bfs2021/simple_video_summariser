"""YouTube implementation of the ``MetadataSource`` port (architecture.md 3).

``fetch`` runs ``yt-dlp --dump-json`` for one video and turns the result into
a ``VideoMeta``. Failures arrive already classified by the shared runner.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adapters.youtube.errors import UpcomingVideoError, decode_ytdlp_json
from adapters.youtube.ytdlp import ProcessRunner, run_process, run_ytdlp
from common.errors import ToolFailureError
from common.models import VideoMeta
from common.youtube_refs import VIDEO_ID_PATTERN

DEFAULT_TIMEOUT_SEC = 120.0

_DUMP_FLAGS = ("--dump-json", "--skip-download", "--no-playlist", "--no-warnings")
# yt-dlp refuses upcoming videos ("Premieres in 12 hours") unless told that
# having no formats is fine. It is only passed once the first run has said the
# video is upcoming: passed always, it would also turn private, removed and
# age-gated videos into successful dumps with no formats.
_UPCOMING_FLAGS = (*_DUMP_FLAGS, "--ignore-no-formats-error")

_VIDEO_ID = VIDEO_ID_PATTERN

# yt-dlp lists the live chat replay as a subtitle track; it is not a language.
_NOT_A_LANGUAGE = "live_chat"


def watch_url(video_id: str) -> str:
    """The full watch URL for ``video_id``; raises ``ValueError`` if malformed.

    yt-dlp always gets this URL, never a bare ID: an ID may start with ``-``
    and would then be read as an option.
    """
    if not isinstance(video_id, str) or not _VIDEO_ID.fullmatch(video_id):
        raise ValueError(f"not a YouTube video ID: {video_id!r}")
    return f"https://www.youtube.com/watch?v={video_id}"


class YouTubeMetadata:
    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        runner: ProcessRunner = run_process,
        cookie_file: Path | None = None,
    ) -> None:
        self._timeout = timeout
        self._runner = runner
        self._cookie_file = cookie_file

    def fetch(self, video_id: str) -> VideoMeta:
        url = watch_url(video_id)
        try:
            stdout = self._run([*_DUMP_FLAGS, url])
        except UpcomingVideoError:
            stdout = self._run([*_UPCOMING_FLAGS, url])
        return _parse(decode_ytdlp_json(stdout), video_id)

    def _run(self, args: list[str]) -> str:
        return run_ytdlp(
            args,
            timeout=self._timeout,
            runner=self._runner,
            cookie_file=self._cookie_file,
        )


def _parse(info: Any, video_id: str) -> VideoMeta:
    if not isinstance(info, dict):
        raise ToolFailureError(
            f"yt-dlp output is a JSON {type(info).__name__}, not an object"
        )
    if info.get("id") != video_id:
        raise ToolFailureError(
            f"yt-dlp returned video {info.get('id')!r}, asked for {video_id!r}"
        )
    return VideoMeta(
        video_id=video_id,
        channel_id=_required_str(info, "channel_id"),
        title=_required_str(info, "title"),
        description=_optional_str(info, "description") or "",
        duration_sec=_duration(info.get("duration")),
        published_at=_published_at(info),
        language=_optional_str(info, "language"),
        live_status=_optional_str(info, "live_status"),
        manual_subtitle_langs=_langs(info.get("subtitles")),
        auto_caption_langs=_langs(info.get("automatic_captions")),
        release_at=_from_timestamp(info.get("release_timestamp")),
    )


def _required_str(info: dict[str, Any], key: str) -> str:
    value = info.get(key)
    if not isinstance(value, str) or not value:
        raise ToolFailureError(f"yt-dlp output has no {key!r}")
    return value


def _optional_str(info: dict[str, Any], key: str) -> str | None:
    value = info.get(key)
    return value if isinstance(value, str) and value else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _duration(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def _from_timestamp(raw: Any) -> datetime | None:
    value = _number(raw)
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(value, UTC)
    except OverflowError, OSError, ValueError:
        return None


def _published_at(info: dict[str, Any]) -> datetime | None:
    for key in ("timestamp", "release_timestamp"):
        parsed = _from_timestamp(info.get(key))
        if parsed is not None:
            return parsed
    upload_date = info.get("upload_date")
    if isinstance(upload_date, str):
        try:
            return datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=UTC)
        except ValueError:
            pass
    return None


def _langs(tracks: Any) -> tuple[str, ...]:
    if not isinstance(tracks, dict):
        return ()
    return tuple(lang for lang in tracks if lang != _NOT_A_LANGUAGE)
