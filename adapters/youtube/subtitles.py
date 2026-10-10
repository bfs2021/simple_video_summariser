"""YouTube implementation of the ``SubtitleSource`` port (architecture.md 3).

``available`` reports which English subtitle tracks a video has using only the
``VideoMeta`` already fetched (#16) - it makes no process call. ``fetch``
downloads and parses one chosen track, running yt-dlp through the shared
runner (#16) into a fresh temporary directory, and applies the speaker-label
heuristics below to whatever ``parse_vtt`` (#17) returned.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from adapters.youtube.metadata import watch_url
from adapters.youtube.vtt import parse_vtt
from adapters.youtube.ytdlp import ProcessRunner, run_process, run_ytdlp
from common.errors import ToolFailureError
from common.models import Segment, VideoMeta

DEFAULT_TIMEOUT_SEC = 120.0

# English track preference: exact codes first, in this order, then any other
# `en-*` variant (deterministic, alphabetical - the issue does not rank those).
_PREFERRED_EXACT = ("en", "en-US", "en-GB")

# A cue starting with an all-caps label of at most 3 words, optionally after a
# bare speaker-change marker (`>>`), followed by a colon: `JANE DOE:`,
# `>> HOST:`. Each label word must be entirely uppercase (plus `'`, `.`, `-`)
# so lowercase-tailed words ("Note:"), mixed case ("Bob Smith:") and more than
# 3 words never match.
_LABEL = re.compile(
    r"^(?:>>[ \t]*)?([A-Z][A-Z'.\-]*(?:[ \t][A-Z][A-Z'.\-]*){0,2}):[ \t]*(.*)$",
    re.DOTALL,
)
# A bare speaker-change marker with no name attached.
_BARE_ARROW = re.compile(r"^>>[ \t]*(.*)$", re.DOTALL)


@dataclass(frozen=True, slots=True)
class SubtitleAvailability:
    """Which English subtitle track (if any) is available, per kind."""

    manual: str | None
    auto: str | None


class YouTubeSubtitles:
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

    def available(self, meta: VideoMeta) -> SubtitleAvailability:
        """Which English manual/auto track is preferred, from ``meta`` alone.

        A translated `en` auto track on a non-English video is not reported:
        it is a translation product, not the video's own English captioning.
        """
        manual = _preferred_lang(meta.manual_subtitle_langs)
        auto = _preferred_lang(meta.auto_caption_langs) if _is_english(meta.language) else None
        return SubtitleAvailability(manual=manual, auto=auto)

    def fetch(self, video_id: str, lang: str, kind: str) -> list[Segment]:
        """Download and parse one subtitle track into segments.

        Raises ``ToolFailureError`` if yt-dlp exits 0 but writes no matching
        `.vtt` file - silently returning `[]` there would let ingest (#28)
        fall through to hours of transcription for a track that ``available``
        already reported as present. A file that parses to zero segments is
        different: that legitimately means "no subtitles" and returns `[]`.
        """
        if kind not in ("manual", "auto"):
            raise ValueError(f"kind must be 'manual' or 'auto', got {kind!r}")
        url = watch_url(video_id)
        write_flag = "--write-subs" if kind == "manual" else "--write-auto-subs"

        with tempfile.TemporaryDirectory(prefix="ytdigest-subtitles-") as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            args = [
                "--skip-download",
                "--sub-format",
                "vtt",
                "--sub-langs",
                lang,
                write_flag,
                "--paths",
                str(tmpdir),
                url,
            ]
            run_ytdlp(
                args,
                timeout=self._timeout,
                runner=self._runner,
                cookie_file=self._cookie_file,
            )
            matches = sorted(tmpdir.glob(f"*.{lang}.vtt"))
            if not matches:
                raise ToolFailureError(
                    f"yt-dlp exited 0 but wrote no {lang!r} subtitle file for "
                    f"{video_id!r} (kind={kind!r})"
                )
            text = matches[0].read_text(encoding="utf-8")

        return _apply_speaker_labels(parse_vtt(text))


def speaker_source(segments: list[Segment]) -> str:
    """``subtitle_labels`` if any segment carries a speaker, else ``none``.

    Stored in ``transcripts.speaker_source`` (architecture.md 0, C3).
    """
    return "subtitle_labels" if any(segment.speaker for segment in segments) else "none"


def _is_english(language: str | None) -> bool:
    """Unknown language (``None``) is not treated as non-English."""
    return language is None or language == "en" or language.startswith("en-")


def _preferred_lang(langs: tuple[str, ...]) -> str | None:
    available = set(langs)
    for lang in _PREFERRED_EXACT:
        if lang in available:
            return lang
    others = sorted(
        lang for lang in available if lang.startswith("en-") and lang not in _PREFERRED_EXACT
    )
    return others[0] if others else None


def _apply_speaker_labels(segments: list[Segment]) -> list[Segment]:
    result: list[Segment] = []
    for segment in segments:
        if segment.speaker is not None:
            result.append(segment)  # <v Name> tags are kept as-is
            continue
        label_match = _LABEL.match(segment.text)
        if label_match is not None:
            label, rest = label_match.groups()
            result.append(replace(segment, speaker=label.title(), text=rest.strip()))
            continue
        arrow_match = _BARE_ARROW.match(segment.text)
        if arrow_match is not None:
            result.append(replace(segment, text=arrow_match.group(1).strip()))
            continue
        result.append(segment)
    return result
