"""The golden-set file parser (issue #62): a pure function, table-tested."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.eval.golden import (
    MAX_ENTRIES,
    MAX_FILE_BYTES,
    GoldenError,
    parse_golden,
    read_golden,
)

A = "dQw4w9WgXcQ"
B = "aaaaaaaaaaa"
DASH = "-wNyEUrxzFU"


def ids(text: str) -> tuple[str, ...]:
    return parse_golden(text).video_ids


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (A, (A,)),
        (f"{A}\n{B}\n", (A, B)),
        (f"{A}\n{B}", (A, B)),  # no final newline
        (f"{A}\r\n{B}\r\n", (A, B)),  # CRLF
        (f"\n\n  {A}  \n\n", (A,)),
        (f"# comment\n{A}\n   # indented comment\n", (A,)),
        (f"{A} # a note\n", (A,)),
        (f"{A}\t# tab note\n", (A,)),
        (f"{A} #nospace note\n", (A,)),
        (DASH, (DASH,)),  # a leading '-' is valid
        (f"https://www.youtube.com/watch?v={A}", (A,)),
        (f"youtu.be/{B}  # short link", (B,)),
        (f"https://youtu.be/{DASH}", (DASH,)),
        (f"https://www.youtube.com/watch?v={A}#t=10", (A,)),  # a fragment is not a note
        (f"{B}\n{A}\n{DASH}", (B, A, DASH)),  # file order is kept
    ],
)
def test_accepted_forms(text: str, expected: tuple[str, ...]) -> None:
    assert ids(text) == expected


def test_a_duplicate_is_kept_once_with_a_warning_naming_both_lines() -> None:
    parsed = parse_golden(f"{A}\n{B}\nhttps://youtu.be/{A}\n{A}\n")
    assert parsed.video_ids == (A, B)
    assert len(parsed.warnings) == 2
    assert "line 3" in parsed.warnings[0] and "line 1" in parsed.warnings[0]
    assert "line 4" in parsed.warnings[1]


@pytest.mark.parametrize("text", ["", "\n\n", "   \n", "# only a comment\n", "\r\n#x\r\n"])
def test_an_empty_set_is_an_error(text: str) -> None:
    with pytest.raises(GoldenError, match="golden set is empty"):
        parse_golden(text)


@pytest.mark.parametrize(
    "line",
    [
        "short",
        "a" * 12,
        "dQw4w9WgXc!",
        f"https://example.com/watch?v={A}",
        f"https://www.youtube.com/channel/UC{'a' * 22}",
        f"https://www.youtube.com/playlist?list=PL{'a' * 16}",
        "https://www.youtube.com/@name",
        f"javascript:alert({A})",
        f"file://{A}",
        "dQw4w9WgXcQ extra",
        "--exec",
        "\x1b[31mred",
    ],
)
def test_an_invalid_line_is_an_error_naming_its_line_number(line: str) -> None:
    with pytest.raises(GoldenError) as err:
        parse_golden(f"# header\n{A}\n{line}\n")
    assert err.value.messages[0].startswith("line 3: ")


def test_every_invalid_line_is_listed_up_to_a_cap() -> None:
    with pytest.raises(GoldenError) as err:
        parse_golden("bad1\n" + A + "\nbad3\n")
    assert [m.split(":")[0] for m in err.value.messages] == ["line 1", "line 3"]


def test_the_offending_text_is_sanitized_and_length_capped() -> None:
    line = "\x1b]0;owned\x07" + "x" * 500
    with pytest.raises(GoldenError) as err:
        parse_golden(line)
    message = err.value.messages[0]
    assert "\x1b" not in message and "\x07" not in message
    assert len(message) < 200


def test_entry_cap_boundary() -> None:
    def video(n: int) -> str:
        return f"{n:011d}"

    at_limit = "\n".join(video(n) for n in range(MAX_ENTRIES))
    assert len(ids(at_limit)) == MAX_ENTRIES
    with pytest.raises(GoldenError, match="more than 200"):
        parse_golden(at_limit + "\n" + video(MAX_ENTRIES))


def test_duplicates_do_not_count_towards_the_cap() -> None:
    text = "\n".join([A] * (MAX_ENTRIES + 5))
    assert ids(text) == (A,)


def test_other_control_separators_do_not_split_lines() -> None:
    # Only \n (and \r\n) end a line, so line numbers match what an editor shows.
    with pytest.raises(GoldenError) as err:
        parse_golden(f"{A}\x0c{B}\n{B}\n")
    assert err.value.messages[0].startswith("line 1: ")
    with pytest.raises(GoldenError):
        parse_golden(f"{A}\r{B}\n")


def test_file_size_cap_boundary(tmp_path: Path) -> None:
    header = f"{A}\n# "
    at_limit = tmp_path / "at.txt"
    at_limit.write_text(header + "x" * (MAX_FILE_BYTES - len(header)), encoding="utf-8")
    assert read_golden(str(at_limit), stdin=lambda: "").video_ids == (A,)
    over = tmp_path / "over.txt"
    over.write_text(header + "x" * (MAX_FILE_BYTES - len(header) + 1), encoding="utf-8")
    with pytest.raises(GoldenError, match="too large"):
        read_golden(str(over), stdin=lambda: "")


def test_read_golden_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(GoldenError, match="UTF-8"):
        read_golden(str(bad), stdin=lambda: "")
    with pytest.raises(GoldenError, match="missing.txt"):
        read_golden(str(tmp_path / "missing.txt"), stdin=lambda: "")
    assert read_golden("-", stdin=lambda: f"﻿{A}\n").video_ids == (A,)
