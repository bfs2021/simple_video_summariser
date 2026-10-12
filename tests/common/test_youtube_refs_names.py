"""``parse_channel_input``: @handle, /c/ and /user/ references (#81)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from common.youtube_refs import NameRef, parse_channel_input

CHANNEL = "UCBR8-60-B28hp2BmDPdntcQ"
_HANDLE_ALPHABET = st.characters(
    whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-."
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("@YouTube", NameRef("handle", "YouTube")),
        ("@Ünal", NameRef("handle", "Ünal")),
        ("@ドラえもん", NameRef("handle", "ドラえもん")),
        ("youtube.com/@YouTube/", NameRef("handle", "YouTube")),
        ("https://www.youtube.com/@YouTube/videos", NameRef("handle", "YouTube")),
        ("https://m.youtube.com/@YouTube/featured?x=1", NameRef("handle", "YouTube")),
        ("https://www.youtube.com/@YouTube#top", NameRef("handle", "YouTube")),
        ("https://www.youtube.com/@%C3%9Cnal", NameRef("handle", "Ünal")),
        ("http://YOUTUBE.com/c/SomeName", NameRef("c", "SomeName")),
        ("www.youtube.com/user/someone/streams", NameRef("user", "someone")),
        ("  @YouTube  ", NameRef("handle", "YouTube")),
        ("@" + "a" * 3, NameRef("handle", "aaa")),
        ("@" + "a" * 30, NameRef("handle", "a" * 30)),
        ("https://www.youtube.com/c/" + "x" * 100, NameRef("c", "x" * 100)),
        ("https://www.youtube.com/c/x", NameRef("c", "x")),
        (f"https://www.youtube.com/channel/{CHANNEL}", CHANNEL),
        (CHANNEL, CHANNEL),
    ],
)
def test_accepted_forms(text: str, expected: str | NameRef) -> None:
    assert parse_channel_input(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "@",
        "@ab",
        "@" + "a" * 31,
        "@a b",
        "@abc/def",
        "@abc%2Fdef",
        "@abc%3Bdef",
        "@abc?x",
        "@abc\x01",
        "@abc\n@def",
        "youtube",
        "YouTube",
        "youtube.com/@ab",
        "youtube.com/@ab/foo/bar",
        "youtube.com/c/",
        "youtube.com/c/" + "x" * 101,
        "youtube.com/user/a%",
        "youtube.com/@abc%",
        "youtube.com.evil.com/@abc",
        "https://youtube.com@evil.com/@abc",
        "ftp://youtube.com/@abc",
        "javascript:@abc",
        "-@abc",
        "--exec",
        "https://www.youtube.com/watch?v=aaaaaaaaaaa",
        "https://youtu.be/aaaaaaaaaaa",
        "https://www.youtube.com/playlist?list=PL1",
        "x" * 2049,
    ],
)
def test_rejected_forms(text: str) -> None:
    with pytest.raises(ValueError):
        parse_channel_input(text)


def test_input_of_exactly_2048_characters_is_still_parsed() -> None:
    prefix = "https://www.youtube.com/@abc?"
    text = prefix + "q" * (2048 - len(prefix))
    assert len(text) == 2048
    assert parse_channel_input(text) == NameRef("handle", "abc")


def test_input_of_2049_characters_is_rejected() -> None:
    prefix = "https://www.youtube.com/@abc?"
    with pytest.raises(ValueError, match="longer than 2048"):
        parse_channel_input(prefix + "q" * (2049 - len(prefix)))


def test_rejection_message_never_echoes_the_input() -> None:
    marker = "ZZMARKER42"
    with pytest.raises(ValueError) as info:
        parse_channel_input(f"youtube.com/@ab{marker}/x/y")
    assert marker not in str(info.value)


_SAFE = _HANDLE_ALPHABET


@given(st.text(alphabet=_SAFE, min_size=3, max_size=30))
def test_every_accepted_handle_form_parses_to_the_same_name(name: str) -> None:
    expected = NameRef("handle", name)
    for form in (
        f"@{name}",
        f"youtube.com/@{name}",
        f"https://www.youtube.com/@{name}/",
        f"https://m.youtube.com/@{name}/videos?x=1",
    ):
        assert parse_channel_input(form) == expected


@given(st.text(alphabet=_SAFE, min_size=1, max_size=100))
def test_every_accepted_legacy_form_parses_to_the_same_name(name: str) -> None:
    for kind in ("c", "user"):
        expected = NameRef(kind, name)
        for form in (
            f"https://www.youtube.com/{kind}/{name}",
            f"youtube.com/{kind}/{name}/",
            f"https://www.youtube.com/{kind}/{name}/shorts?x=y",
        ):
            assert parse_channel_input(form) == expected


@given(st.from_regex(r"UC[A-Za-z0-9_-]{22}", fullmatch=True))
def test_every_valid_channel_id_form_still_parses_to_that_id(channel_id: str) -> None:
    assert parse_channel_input(channel_id) == channel_id
    assert parse_channel_input(f"https://www.youtube.com/channel/{channel_id}") == channel_id
