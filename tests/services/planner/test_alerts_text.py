"""Alert text scrubbing (issue #67): no database needed."""

from __future__ import annotations

import pytest

from services.planner.alerts import _last_line


@pytest.mark.parametrize(
    "line",
    [
        "GET https://user:pa@ss@host.test/x failed",
        "GET https://user:pa/ss@host.test/x failed",
        "GET https://user:p#TOKEN@host.test/x failed",
        "GET https://user:pa?ss@host.test/x failed",
        "GET (https://user:pa@ss/w@host.test/x) failed",
    ],
)
def test_userinfo_is_stripped_up_to_the_last_at_sign(line: str) -> None:
    out = _last_line(line)

    for leak in ("ss", "TOKEN", "pa", "user", "w@"):
        assert leak not in out.replace("host.test", "").replace("failed", "").replace("GET", "")
    assert "host.test" in out


def test_fragment_and_query_are_dropped() -> None:
    out = _last_line("see https://host.test/a#SECRETFRAG then https://h2.test/b?sig=SECRETQ ok")

    assert "SECRETFRAG" not in out
    assert "SECRETQ" not in out
    assert "host.test/a" in out


def test_a_line_without_urls_is_unchanged_and_hash_text_survives() -> None:
    assert _last_line("ToolFailureError: item #5 failed") == "ToolFailureError: item #5 failed"
