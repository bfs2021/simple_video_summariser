"""Scheduled yt-dlp update check (issue #68). No network: the fetch is injected."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ytdlp_update_check.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ytdlp-update-check.yml"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ytdlp_update_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load()

REQS = "fastapi==0.1\nyt-dlp==2026.8.19\npsycopg==3.2\n"


def _pypi(version: str) -> tuple[int, bytes]:
    return 200, json.dumps({"info": {"version": version}}).encode()


class Fetch:
    def __init__(self, response: tuple[int, bytes] | Exception) -> None:
        self.response = response
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float) -> tuple[int, bytes]:
        self.urls.append(url)
        assert timeout > 0
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _run(
    tmp_path: Path, reqs: str, fetch: Fetch, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    path = tmp_path / "requirements.backend.txt"
    path.write_text(reqs)
    code = check.main([str(path)], fetch=fetch)
    out = capsys.readouterr()
    assert path.read_text() == reqs  # the check reports; it never edits the pin
    return code, out.out, out.err


# -- the pin --------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    ["yt-dlp==2026.8.19", "yt-dlp == 2026.8.19", "YT_DLP==2026.8.19  # pinned", "yt-dlp[default]==2026.8.19",
     'yt-dlp==2026.8.19 ; python_version >= "3.9"', 'yt-dlp==2026.8.19;python_version>"3"'],
)
def test_pinned_version_is_read_from_the_requirements_line(line: str) -> None:
    assert check.pinned_version(f"a==1\n{line}\nb==2\n") == "2026.8.19"


@pytest.mark.parametrize("text", ["", "a==1\n", "yt-dlp>=2026.1.1\n", "# yt-dlp==2026.8.19\n", "yt-dlp-extra==1.0\n"])
def test_a_missing_or_unpinned_yt_dlp_is_an_error(text: str) -> None:
    with pytest.raises(check.PinNotFound):
        check.pinned_version(text)


def test_the_repository_requirements_file_pins_yt_dlp() -> None:
    text = (REPO_ROOT / "requirements.backend.txt").read_text()

    assert re.fullmatch(r"\d+(\.\d+)+", check.pinned_version(text))


# -- the comparison -------------------------------------------------------


def test_behind_names_both_versions_emits_an_annotation_and_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch = Fetch(_pypi("2026.9.30"))

    code, out, _ = _run(tmp_path, REQS, fetch, capsys)

    assert code == 1
    assert out.startswith("::error")  # a GitHub Actions annotation
    assert "2026.8.19" in out and "2026.9.30" in out
    assert "pinned" in out and "latest" in out
    assert fetch.urls == ["https://pypi.org/pypi/yt-dlp/json"]


def test_current_is_silent_and_exits_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run(tmp_path, REQS, Fetch(_pypi("2026.8.19")), capsys)

    assert (code, out, err) == (0, "", "")


def test_zero_padding_does_not_make_a_current_pin_look_behind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = _run(tmp_path, REQS, Fetch(_pypi("2026.08.19")), capsys)

    assert (code, out, err) == (0, "", "")


@pytest.mark.parametrize("latest", ["2026.8.20", "2026.8.19.1", "2026.10.1", "2027.1.1"])
def test_any_newer_release_is_behind(latest: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(tmp_path, REQS, Fetch(_pypi(latest)), capsys)

    assert code == 1 and latest in out


@pytest.mark.parametrize("latest", ["2026.8.18", "2026.7.30", "2025.12.31"])
def test_a_pin_ahead_of_pypi_is_not_noise(latest: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run(tmp_path, REQS, Fetch(_pypi(latest)), capsys)

    assert (code, out, err) == (0, "", "")


def test_versions_compare_numerically_not_as_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(tmp_path, "yt-dlp==2026.9.9\n", Fetch(_pypi("2026.10.1")), capsys)

    assert code == 1 and "2026.10.1" in out


# -- an outage fails quietly ---------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        TimeoutError("timed out"),
        ConnectionError("refused"),
        OSError("dns"),
        (503, b"upstream down"),
        (404, b""),
        (404, b'{"info": {"version": "2099.1.1"}}'),
        (200, b'{"info": {"version": "2099.1.1"}}' + b" " * (33 * 1024 * 1024)),
        (200, b"<html>maintenance</html>"),
        (200, b"[]"),
        (200, b"{}"),
        (200, b'{"info": {}}'),
        (200, b'{"info": {"version": 5}}'),
        (200, b'{"info": {"version": "not-a-version"}}'),
        (200, b'{"info": {"version": ""}}'),
    ],
)
def test_an_outage_or_junk_reply_exits_0_with_no_annotation(
    response: tuple[int, bytes] | Exception, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = _run(tmp_path, REQS, Fetch(response), capsys)

    assert code == 0
    assert out == ""
    assert "::error" not in err and "::warning" not in err
    assert err.count("\n") == 1  # one explanatory line for the log


def test_an_unparseable_pin_is_loud_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch = Fetch(_pypi("2026.9.30"))

    code, out, err = _run(tmp_path, "yt-dlp==abc\n", fetch, capsys)

    assert (code, out) == (2, "")
    assert "abc" in err and "Traceback" not in err
    assert fetch.urls == []


def test_a_missing_pin_is_loud(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fetch = Fetch(_pypi("2026.9.30"))

    code, _, err = _run(tmp_path, "fastapi==0.1\n", fetch, capsys)

    assert code == 2
    assert "yt-dlp" in err
    assert fetch.urls == []


# -- the scheduled workflow ----------------------------------------------


def _workflow() -> str:
    return WORKFLOW.read_text()


def test_workflow_runs_daily_and_by_hand_only() -> None:
    text = _workflow()

    assert re.search(r"^on:\s*$", text, re.MULTILINE)
    assert re.search(r"^\s+schedule:\s*\n\s+- cron: ['\"]\S+ \S+ \* \* \*['\"]", text, re.MULTILINE)  # daily
    assert "workflow_dispatch:" in text
    assert "pull_request" not in text and "push:" not in text


def test_workflow_is_read_only_secretless_and_time_boxed() -> None:
    text = _workflow()

    assert re.search(r"^permissions:\s*\n\s+contents: read\s*$", text, re.MULTILINE)
    assert "secrets." not in text
    assert re.search(r"timeout-minutes: \d+", text)
    assert "needs:" not in text  # it never gates anything


def test_workflow_uses_only_first_party_actions_pinned_to_a_sha() -> None:
    uses = re.findall(r"^\s*(?:-\s+)?uses:\s*(\S+)(.*)$", _workflow(), re.MULTILINE)

    assert uses
    for ref, rest in uses:
        assert re.fullmatch(r"actions/[\w-]+@[0-9a-f]{40}", ref), ref
        assert "#" in rest  # version tag in a trailing comment


def test_workflow_runs_the_script() -> None:
    assert "python3 scripts/ytdlp_update_check.py requirements.backend.txt" in _workflow()
