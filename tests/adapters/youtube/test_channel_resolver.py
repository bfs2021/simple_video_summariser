"""``resolve_channel``: @handle, /c/ and /user/ names to a channel ID (#81).

yt-dlp never runs: a ``FakeRunner`` answers from the recorded fixtures in
``tests/fixtures/ytdlp_channels/`` (see its README).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from adapters.youtube.channel_resolver import DEFAULT_TIMEOUT_SEC, resolve_channel
from common.errors import ToolFailureError, TransientNetworkError
from tests.adapters.youtube.fakes import FakeRunner, completed

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "ytdlp_channels"
YOUTUBE_ID = "UCBR8-60-B28hp2BmDPdntcQ"
UNAL_ID = "UC6FTzBckPgMYMNMwmwAhjqg"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _info(**changes: object) -> str:
    info = json.loads(_fixture("handle_youtube.json"))
    info.update(changes)
    return json.dumps(info)


@pytest.mark.parametrize(
    ("kind", "name", "fixture", "url"),
    [
        ("handle", "YouTube", "handle_youtube.json", "https://www.youtube.com/@YouTube"),
        ("c", "YouTube", "c_youtube.json", "https://www.youtube.com/c/YouTube"),
        ("user", "YouTube", "user_youtube.json", "https://www.youtube.com/user/YouTube"),
    ],
)
def test_the_three_forms_of_one_channel_resolve_to_the_same_id(
    kind: str, name: str, fixture: str, url: str
) -> None:
    runner = FakeRunner(completed(stdout=_fixture(fixture)))

    resolved = resolve_channel(kind, name, runner=runner)

    assert resolved.channel_id == YOUTUBE_ID
    assert resolved.title == "YouTube"
    [(argv, _)] = runner.calls
    assert argv[-1] == url


def test_a_non_ascii_handle_resolves_and_is_percent_encoded_in_the_url() -> None:
    runner = FakeRunner(completed(stdout=_fixture("handle_unicode.json")))

    resolved = resolve_channel("handle", "Ünal", runner=runner)

    assert resolved.channel_id == UNAL_ID
    assert resolved.title == "UNAL"
    [(argv, _)] = runner.calls
    assert argv[-1] == "https://www.youtube.com/@%C3%9Cnal"


def test_the_argv_is_flat_single_json_with_no_entries_and_the_url_last() -> None:
    runner = FakeRunner(completed(stdout=_fixture("handle_youtube.json")))

    resolve_channel("handle", "YouTube", runner=runner)

    [(argv, timeout)] = runner.calls
    assert argv[1:4] == ["-m", "yt_dlp", "--flat-playlist"]
    assert argv[4:] == [
        "--dump-single-json",
        "--playlist-items",
        "0",
        "--no-warnings",
        "https://www.youtube.com/@YouTube",
    ]
    assert timeout == DEFAULT_TIMEOUT_SEC


def test_the_default_timeout_is_at_most_twenty_seconds() -> None:
    assert DEFAULT_TIMEOUT_SEC <= 20


def test_a_name_containing_a_slash_or_percent_is_quoted_not_split() -> None:
    runner = FakeRunner(completed(stdout=_fixture("handle_youtube.json")))

    resolve_channel("c", "a%2Fb", runner=runner)

    [(argv, _)] = runner.calls
    assert argv[-1] == "https://www.youtube.com/c/a%252Fb"


def test_a_missing_channel_id_falls_back_to_id_when_id_is_a_channel_id() -> None:
    stdout = json.dumps({"id": YOUTUBE_ID, "channel": "Name"})

    resolved = resolve_channel("c", "x", runner=FakeRunner(completed(stdout=stdout)))

    assert resolved.channel_id == YOUTUBE_ID


@pytest.mark.parametrize(
    "stdout",
    [
        json.dumps({"id": "@YouTube", "channel": "x"}),
        json.dumps({"channel_id": "UCshort", "channel": "x"}),
        json.dumps({"channel_id": YOUTUBE_ID, "id": UNAL_ID}),
        json.dumps({"channel_id": 12}),
        json.dumps(["not", "an", "object"]),
        "not json",
    ],
)
def test_a_missing_malformed_or_disagreeing_id_is_a_tool_failure(stdout: str) -> None:
    with pytest.raises(ToolFailureError):
        resolve_channel("handle", "x", runner=FakeRunner(completed(stdout=stdout)))


def test_the_title_is_stripped_capped_and_none_when_empty() -> None:
    long = "  " + "t" * 300 + "  "
    capped = resolve_channel(
        "handle", "x", runner=FakeRunner(completed(stdout=_info(channel=long)))
    )
    empty = resolve_channel(
        "handle", "x", runner=FakeRunner(completed(stdout=_info(channel="   ")))
    )
    absent = resolve_channel(
        "handle", "x", runner=FakeRunner(completed(stdout=_info(channel=None)))
    )

    assert capped.title == "t" * 200
    assert empty.title is None
    assert absent.title is None


def test_a_timeout_is_transient_network() -> None:
    runner = FakeRunner(subprocess.TimeoutExpired(cmd="yt-dlp", timeout=20))

    with pytest.raises(TransientNetworkError):
        resolve_channel("handle", "x", runner=runner)


def test_a_missing_handle_maps_to_permanent_source_error() -> None:
    from common.errors import PermanentSourceError

    runner = FakeRunner(
        completed(
            stderr=(
                "ERROR: [youtube:tab] @zzqxnotarealhandle9981: Unable to download API page: "
                "HTTP Error 404: Not Found (caused by <HTTPError 404: Not Found>)"
            ),
            returncode=1,
        )
    )

    with pytest.raises(PermanentSourceError):
        resolve_channel("handle", "zzqxnotarealhandle9981", runner=runner)
