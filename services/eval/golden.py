"""The golden-set file: one video per line (issue #62).

A pure parser (``parse_golden``) plus ``read_golden`` for the file or stdin.
Each line is an 11-character video ID or a YouTube URL, checked by
``common.youtube_refs.parse_video_ref`` - the same trust-boundary check the
CLI and the API use. Blank lines, lines starting with ``#`` and a trailing
`` # note`` are ignored. Error messages show the offending text sanitized and
capped, never raw.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from common.youtube_refs import parse_video_ref
from services.cli.output import sanitize_text

#: A golden set larger than this is almost certainly a mistake (and a very long run).
MAX_ENTRIES = 200
#: At most this many bad lines are listed, so a wrong file gives a short message.
MAX_LISTED_ERRORS = 10
#: Longest golden file read, in bytes: 200 URLs fit many times over.
MAX_FILE_BYTES = 1 << 20
_SHOWN_CHARS = 60
_NOTE = re.compile(r"\s+#.*$")


class GoldenError(ValueError):
    """The golden set is unusable. ``messages`` holds one fixed-format line per problem."""

    def __init__(self, messages: list[str]) -> None:
        super().__init__("; ".join(messages))
        self.messages = messages


@dataclass(frozen=True, slots=True)
class ParsedGolden:
    video_ids: tuple[str, ...]
    #: Duplicate-line warnings, for stderr.
    warnings: tuple[str, ...]


def _shown(text: str, limit: int = _SHOWN_CHARS) -> str:
    one_line = " ".join(sanitize_text(text).split())
    if len(one_line) > limit:
        one_line = one_line[: limit - 3] + "..."
    return repr(one_line)


def parse_golden(text: str) -> ParsedGolden:
    """The video IDs in ``text``, in order, each once.

    Raises ``GoldenError`` for an invalid line (naming its line number), an
    empty set or more than ``MAX_ENTRIES`` distinct videos.
    """
    seen: dict[str, int] = {}
    warnings: list[str] = []
    errors: list[str] = []
    bad_lines = 0
    for number, raw in enumerate(text.split("\n"), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = _NOTE.sub("", line).strip()
        try:
            video_id = parse_video_ref(line)
        except ValueError:
            bad_lines += 1
            if bad_lines <= MAX_LISTED_ERRORS:
                errors.append(f"line {number}: not a YouTube video ID or URL: {_shown(line)}")
            continue
        if video_id in seen:
            warnings.append(
                f"line {number}: duplicate of line {seen[video_id]} "
                f"(video {video_id}); kept once"
            )
            continue
        seen[video_id] = number
    if errors:
        if bad_lines > MAX_LISTED_ERRORS:
            errors.append(f"... and {bad_lines - MAX_LISTED_ERRORS} more invalid lines")
        raise GoldenError(errors)
    if not seen:
        raise GoldenError(["golden set is empty"])
    if len(seen) > MAX_ENTRIES:
        raise GoldenError([f"golden set has more than {MAX_ENTRIES} entries"])
    return ParsedGolden(tuple(seen), tuple(warnings))


def read_golden(source: str, *, stdin: Callable[[], str]) -> ParsedGolden:
    """Parse the file at ``source``, or standard input when ``source`` is ``-``."""
    if source == "-":
        try:
            text = stdin()
        except (OSError, UnicodeDecodeError):
            raise GoldenError(["cannot read the golden set from standard input"]) from None
    else:
        path = Path(source)
        try:
            with path.open("rb") as handle:
                data = handle.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise GoldenError([f"golden file is too large: {_shown(source, 300)}"])
            text = data.decode("utf-8")
        except GoldenError:
            raise
        except UnicodeDecodeError:
            raise GoldenError([f"golden file is not valid UTF-8: {_shown(source, 300)}"]) from None
        except OSError:
            raise GoldenError([f"cannot read golden file: {_shown(source, 300)}"]) from None
    return parse_golden(text.removeprefix("﻿"))
