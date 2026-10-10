"""Every requirement in the runtime requirements files is an exact pin (#61).

A range or an unpinned name lets one `pip install` today differ from the one
that passed CI yesterday. The checker is pure and tested on crafted input as
well as on the real files. Transitive dependencies are not covered: that is
#145.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENT_FILES = ("requirements.backend.txt", "requirements.whisper.txt")

# name, optional extras, then `==` and a plain version (no wildcard, no `===`).
_EXACT_PIN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    r"(?:\[[A-Za-z0-9._,\s-]+\])?"
    r"\s*==\s*[A-Za-z0-9][A-Za-z0-9._!+-]*"
)


def find_unpinned(filename: str, text: str) -> list[str]:
    """One message per requirement line that is not an exact `name==version` pin."""
    problems: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = re.sub(r"(^|\s)#.*$", "", line).strip()
        if not stripped:
            continue
        if _EXACT_PIN.fullmatch(stripped) is None:
            problems.append(f"{filename}:{number}: not an exact pin: {line.strip()!r}")
    return problems


@pytest.mark.parametrize(
    "text",
    [
        "name==1.2.3",
        "psycopg[binary]==3.3.6",
        "yt-dlp==2026.8.19  # pinned, see 16.11",
        "faster-whisper==1.2.1\t# tab comment",
        "",
        "\n\n",
        "# only a comment",
        "   # indented comment",
        "a==1\n\n# c\nb==2\n",
        "a==1\r\nb==2\r\n",
        "a==1.0.post1",
        "a==2!1.0+local.1",
    ],
)
def test_accepts_exact_pins_comments_and_blank_lines(text: str) -> None:
    assert find_unpinned("req.txt", text) == []


@pytest.mark.parametrize(
    "line",
    [
        "yt-dlp",
        "yt-dlp>=2026.8.19",
        "name~=1.2",
        "name<2",
        "name<=2",
        "name>1",
        "name!=1.0",
        "name>=1,<2",
        "name==1.*",
        "name===1.0",
        "name==",
        "name==1.0,!=1.1",
        "name @ https://example.com/name-1.0.tar.gz",
        "name@https://example.com/name-1.0.tar.gz",
        "https://example.com/name-1.0.tar.gz",
        "-e .",
        "-e git+https://example.com/x.git#egg=x",
        "-r other.txt",
        "-c constraints.txt",
        "--index-url https://example.com/simple",
        "--extra-index-url https://example.com/simple",
        "--hash=sha256:abc",
        "-f links",
        "name==1.0 ; python_version < '3.14'",
        "==1.0",
        # `#` starts a comment only after whitespace (pip), so this is a bad version.
        "name==1.0#not-a-comment",
    ],
)
def test_rejects_anything_that_is_not_an_exact_pin(line: str) -> None:
    problems = find_unpinned("req.txt", f"# header\n\nok==1.0\n{line}\n")
    assert len(problems) == 1
    # The message names the file, the 1-based line number and the offending line.
    assert problems[0].startswith("req.txt:4: ")
    assert line.strip() in problems[0]


def test_reports_every_offending_line() -> None:
    problems = find_unpinned("req.txt", "a\nb==1\nc>=2\n")
    assert [p.split(": ")[0] for p in problems] == ["req.txt:1", "req.txt:3"]


@pytest.mark.parametrize("filename", REQUIREMENT_FILES)
def test_real_requirements_file_is_exactly_pinned(filename: str) -> None:
    text = (ROOT / filename).read_text(encoding="utf-8")
    assert find_unpinned(filename, text) == []
    # Guard against a vacuous pass: the file must contain at least one pin.
    assert "==" in text
