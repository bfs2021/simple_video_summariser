"""Channel RSS feed adapter (issue #26).

No network: a ``FakeOpener`` stands in for ``urllib.request.urlopen`` and
answers with canned bytes and status codes, or raises real urllib errors.
The fixtures in fixtures/feed/ are described in their README.
"""

import email.message
import http.client
import io
import logging
import pathlib
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any, Self

import pytest
from hypothesis import given
from hypothesis import strategies as st

from adapters.youtube.feed import (
    FeedNotFoundError,
    YouTubeFeed,
    parse_feed,
)
from common.errors import (
    RateLimitedError,
    ToolFailureError,
    TransientNetworkError,
)
from common.models import ChannelFeed, FeedEntry
from common.youtube_refs import (
    is_channel_id,
    is_video_id,
    validate_channel_id,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "feed"
CHANNEL = "UCAuUUnT6oDeKwE6v1NGQxug"
URL = f"https://www.youtube.com/feeds/videos.xml?channel_id={CHANNEL}"
CAP = 1024 * 1024
LOGGER = "adapters.youtube.feed"


def entry_xml(
    video_id: str | None = "aBcD3fGh1Jk",
    *,
    channel: str | None = CHANNEL,
    title: str | None = "A title",
    published: str | None = "2026-08-04T15:00:04+00:00",
    links: tuple[str, ...] = (),
) -> str:
    parts = ["<entry>"]
    if video_id is not None:
        parts.append(f"<yt:videoId>{video_id}</yt:videoId>")
    if channel is not None:
        parts.append(f"<yt:channelId>{channel}</yt:channelId>")
    if title is not None:
        parts.append(f"<title>{title}</title>")
    if published is not None:
        parts.append(f"<published>{published}</published>")
    parts.extend(links)
    parts.append("<media:group><media:title>other</media:title></media:group>")
    parts.append("</entry>")
    return "".join(parts)


def feed_xml(*entries: str, header: str = "") -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015" '
        'xmlns:media="http://search.yahoo.com/mrss/" '
        'xmlns="http://www.w3.org/2005/Atom">'
        f"{header}<title>Channel</title>{''.join(entries)}</feed>"
    ).encode()


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def parse(data: bytes) -> list[FeedEntry]:
    return parse_feed(data, CHANNEL)


# --- ids -------------------------------------------------------------------


def test_channel_id_validator_accepts_canonical_ids() -> None:
    assert validate_channel_id(CHANNEL) == CHANNEL
    assert is_channel_id("UC" + "-_" * 11)


def test_video_id_check_accepts_leading_dash() -> None:
    assert is_video_id("-wNyEUrxzFU")
    assert not is_video_id("short")


# --- interface -------------------------------------------------------------


def test_feed_entry_is_frozen_and_slotted() -> None:
    e = FeedEntry("a" * 11, CHANNEL, "t", datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(AttributeError):
        e.title = "x"  # type: ignore[misc]
    assert not hasattr(e, "__dict__")


def test_youtube_feed_satisfies_channel_feed_port() -> None:
    feed: ChannelFeed = YouTubeFeed(opener=FakeOpener(ok(feed_xml())))
    assert feed.fetch(CHANNEL) == []


# --- fake opener -----------------------------------------------------------


class FakeResponse:
    def __init__(
        self, body: bytes = b"", status: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        self.body = body
        self.status = status
        self.headers = email.message.Message()
        for k, v in (headers or {}).items():
            self.headers[k] = v
        self.read_sizes: list[int | None] = []
        self.closed = False

    def read(self, size: int | None = None) -> bytes:
        self.read_sizes.append(size)
        if size is None:
            return self.body
        return self.body[:size]

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class FakeOpener:
    def __init__(self, *results: FakeResponse | BaseException) -> None:
        self.results = list(results)
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, request: Any, **kwargs: Any) -> FakeResponse:
        self.calls.append((request, kwargs))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def ok(body: bytes) -> FakeResponse:
    return FakeResponse(body)


def http_error(
    code: int, headers: dict[str, str] | None = None, body: bytes = b""
) -> urllib.error.HTTPError:
    hdrs = email.message.Message()
    for k, v in (headers or {}).items():
        hdrs[k] = v
    return urllib.error.HTTPError(URL, code, "msg", hdrs, io.BytesIO(body))


def fetch_with(*results: FakeResponse | BaseException) -> list[FeedEntry]:
    return YouTubeFeed(opener=FakeOpener(*results)).fetch(CHANNEL)


# --- channel id: untrusted input -------------------------------------------

BAD_CHANNEL_IDS: list[object] = [
    "UC" + "a" * 21,  # 23 chars
    "UC" + "a" * 23,  # 25 chars
    "uc" + "a" * 22,
    "@NASA",
    "https://www.youtube.com/channel/" + CHANNEL,
    "",
    None,
    123,
    b"UC" + b"a" * 22,
    CHANNEL[:-1] + "&",
    CHANNEL[:-1] + "#",
    CHANNEL[:-1] + "/",
    CHANNEL[:-1] + "%",
    CHANNEL[:-1] + " ",
    CHANNEL[:-1] + "\n",
    CHANNEL + "\n",
    CHANNEL[:-1] + "\t",
    CHANNEL[:-1] + "é",
]


@pytest.mark.parametrize("bad", BAD_CHANNEL_IDS, ids=repr)
def test_fetch_rejects_a_bad_channel_id_before_any_request(bad: Any) -> None:
    opener = FakeOpener()
    with pytest.raises(ValueError):
        YouTubeFeed(opener=opener).fetch(bad)
    assert opener.calls == []


@pytest.mark.parametrize("bad", BAD_CHANNEL_IDS, ids=repr)
def test_validate_channel_id_rejects(bad: Any) -> None:
    with pytest.raises(ValueError):
        validate_channel_id(bad)
    assert not is_channel_id(bad)


def test_request_url_is_exactly_the_feed_url() -> None:
    opener = FakeOpener(ok(feed_xml()))
    YouTubeFeed(opener=opener).fetch(CHANNEL)
    request, _ = opener.calls[0]
    assert request.full_url == URL


def test_dash_and_underscore_channel_ids_are_accepted() -> None:
    cid = "UC" + "-_" * 11
    opener = FakeOpener(ok(feed_xml()))
    YouTubeFeed(opener=opener).fetch(cid)
    assert opener.calls[0][0].full_url.endswith("channel_id=" + cid)


# --- fetching --------------------------------------------------------------


def test_request_uses_configured_timeout_and_a_user_agent() -> None:
    opener = FakeOpener(ok(feed_xml()))
    YouTubeFeed(timeout=7.5, opener=opener).fetch(CHANNEL)
    request, kwargs = opener.calls[0]
    assert kwargs["timeout"] == 7.5
    assert request.get_header("User-agent")
    assert "python-urllib" not in request.get_header("User-agent").lower()


def test_default_timeout_is_30_seconds() -> None:
    opener = FakeOpener(ok(feed_xml()))
    YouTubeFeed(opener=opener).fetch(CHANNEL)
    assert opener.calls[0][1]["timeout"] == 30.0


def test_default_opener_is_urlopen() -> None:
    assert YouTubeFeed()._opener is urllib.request.urlopen


def test_200_returns_entries() -> None:
    entries = fetch_with(ok(feed_xml(entry_xml())))
    assert [e.video_id for e in entries] == ["aBcD3fGh1Jk"]


def test_response_is_closed() -> None:
    resp = ok(feed_xml())
    fetch_with(resp)
    assert resp.closed


def test_body_exactly_at_the_cap_is_accepted() -> None:
    base = feed_xml(entry_xml())
    pad = b"<!-- " + b"x" * (CAP - len(base) - 9) + b" -->"
    body = base.replace(b"<title>Channel", pad + b"<title>Channel", 1)
    assert len(body) == CAP
    assert len(fetch_with(ok(body))) == 1


def test_body_one_byte_over_the_cap_is_rejected_without_reading_the_rest() -> None:
    resp = ok(b"x" * (CAP + 5000))
    with pytest.raises(ToolFailureError):
        fetch_with(resp)
    assert all(s is not None and s <= CAP + 1 for s in resp.read_sizes)


def test_404_raises_feed_not_found_which_is_transient() -> None:
    with pytest.raises(FeedNotFoundError) as info:
        fetch_with(http_error(404))
    assert isinstance(info.value, TransientNetworkError)
    with pytest.raises(FeedNotFoundError):
        fetch_with(FakeResponse(b"<html>", status=404))


def test_429_takes_retry_after_seconds() -> None:
    with pytest.raises(RateLimitedError) as info:
        fetch_with(http_error(429, {"Retry-After": "120"}))
    assert info.value.retry_after_sec == 120


@pytest.mark.parametrize(
    "value",
    [None, "-5", "abc", "Wed, 21 Oct 2026 07:28:00 GMT", "", "1.5", "١٢"],
)
def test_429_without_a_usable_retry_after_gives_none(value: str | None) -> None:
    headers = {} if value is None else {"Retry-After": value}
    with pytest.raises(RateLimitedError) as info:
        fetch_with(http_error(429, headers))
    assert info.value.retry_after_sec is None


def test_429_as_a_returned_response_is_rate_limited() -> None:
    with pytest.raises(RateLimitedError) as info:
        fetch_with(FakeResponse(status=429, headers={"Retry-After": "3"}))
    assert info.value.retry_after_sec == 3


@pytest.mark.parametrize("status", [500, 502, 503, 599])
def test_5xx_is_transient(status: int) -> None:
    with pytest.raises(TransientNetworkError) as info:
        fetch_with(http_error(status))
    assert not isinstance(info.value, FeedNotFoundError)


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timed out"),
        urllib.error.URLError("Temporary failure in name resolution"),
        urllib.error.URLError(ConnectionRefusedError("refused")),
        ConnectionResetError("reset"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_network_failures_are_transient(error: BaseException) -> None:
    with pytest.raises(TransientNetworkError):
        fetch_with(error)


def test_connection_reset_while_reading_the_body_is_transient() -> None:
    class Resetting(FakeResponse):
        def read(self, size: int | None = None) -> bytes:
            raise ConnectionResetError("reset by peer")

    with pytest.raises(TransientNetworkError):
        fetch_with(Resetting())


def test_incomplete_read_is_transient() -> None:
    class Short(FakeResponse):
        def read(self, size: int | None = None) -> bytes:
            raise http.client.IncompleteRead(b"<fe")

    with pytest.raises(TransientNetworkError):
        fetch_with(Short())


@pytest.mark.parametrize("status", [400, 403, 410])
def test_other_non_2xx_is_a_tool_failure_naming_the_status(status: int) -> None:
    with pytest.raises(ToolFailureError) as info:
        fetch_with(http_error(status))
    assert str(status) in str(info.value)


def test_error_messages_carry_at_most_200_bytes_of_the_body() -> None:
    secret = "SECRET-BODY-" + "z" * 5000
    cases: list[FakeResponse | BaseException] = [
        http_error(403, body=secret.encode()),
        http_error(404, body=secret.encode()),
        http_error(429, body=secret.encode()),
        http_error(503, body=secret.encode()),
        ok(secret.encode()),
    ]
    for case in cases:
        with pytest.raises((ToolFailureError, TransientNetworkError, RateLimitedError)) as info:
            fetch_with(case)
        message = str(info.value)
        assert "SECRET-BODY-" + "z" * 200 not in message
        assert len(message.encode()) < 1000


# --- parsing: fixtures -----------------------------------------------------


NASA = "UCLA_DiR1FfKNvjuUpBHmylQ"
VERGE = "UCddiUEpeqJcYeBxX1IVBKvQ"


def test_real_nasa_feed_maps_every_entry_in_document_order() -> None:
    entries = parse_feed(fixture("nasa.xml"), NASA)
    assert [e.video_id for e in entries] == [
        "j9epFget1W8", "v03RjDNwG1o", "aujP8wuMTMI", "cCpf0BOjlLE", "IwZVXmQdX1E",
        "90Kgw_SvK4w", "jHKf1eHp3eQ", "MVt2139voxk", "_oXt3-YDih4", "-HQOp1-LpU0",
        "6o3m9Bw67Os", "8NOLpgadWXc", "dnAJJwgpHs4", "9wq3VHsL_bE", "l5OJk1FuEKg",
    ]  # fmt: skip
    assert {e.channel_id for e in entries} == {NASA}
    assert entries[0] == FeedEntry(
        "j9epFget1W8",
        NASA,
        "Space Station Operations Update (Sept. 28, 2026)",
        datetime(2026, 9, 25, 22, 24, 7, tzinfo=UTC),
    )
    assert entries[-1].published_at == datetime(2026, 8, 29, 16, 34, 47, tzinfo=UTC)
    assert entries[2].title == (
        "NASA\u2019s SpaceX Crew-12 Pre-Departure News Conference (Sept. 16, 2026)"
    )
    assert all(e.published_at.tzinfo is UTC for e in entries)


def test_real_nasa_feed_has_ids_starting_with_underscore_and_dash() -> None:
    ids = {e.video_id for e in parse_feed(fixture("nasa.xml"), NASA)}
    assert {"_oXt3-YDih4", "-HQOp1-LpU0"} <= ids


def test_real_nasa_feed_keeps_a_typographic_apostrophe_in_titles() -> None:
    titles = [e.title for e in parse_feed(fixture("nasa.xml"), NASA)]
    assert any("NASA\u2019s SpaceX Crew-12" in t for t in titles)


def test_real_verge_feed_has_shorts_and_decodes_entities_in_titles() -> None:
    entries = parse_feed(fixture("the_verge.xml"), VERGE)
    assert len(entries) == 15
    assert entries[0] == FeedEntry(
        "9Y5BpmB8R8I",
        VERGE,
        "Would you want to know if Ghostface is at your door?",
        datetime(2026, 9, 26, 14, 0, 6, tzinfo=UTC),
        is_short=True,
    )
    by_id = {e.video_id: e for e in entries}
    assert by_id["ol9e_269eu4"].title == 'Mark Zuckerberg shows off "Muse Charm"'
    # Shorts are not filtered out: the feed marks them only by their link.
    assert b"/shorts/9Y5BpmB8R8I" in fixture("the_verge.xml")
    assert "9Y5BpmB8R8I" in by_id


def test_real_verge_feed_marks_exactly_the_shorts_by_their_link() -> None:
    entries = parse_feed(fixture("the_verge.xml"), VERGE)
    shorts = {e.video_id for e in entries if e.is_short}
    assert len(shorts) == 13
    linked = {m.decode() for m in re.findall(rb"/shorts/([A-Za-z0-9_-]{11})", fixture("the_verge.xml"))}
    assert shorts == linked
    assert len(entries) - len(shorts) == 2


def test_real_nasa_feed_has_no_shorts() -> None:
    assert not any(e.is_short for e in parse_feed(fixture("nasa.xml"), NASA))


def test_real_feed_is_rejected_for_a_different_requested_channel() -> None:
    assert parse_feed(fixture("nasa.xml"), VERGE) == []


def test_real_feed_via_fetch_with_a_fake_opener() -> None:
    feed = YouTubeFeed(opener=FakeOpener(ok(fixture("nasa.xml"))))
    assert len(feed.fetch(NASA)) == 15


# --- parsing: fields -------------------------------------------------------


@pytest.mark.parametrize(
    "published",
    [
        "2026-08-04T15:00:04+00:00",
        "2026-08-04T15:00:04Z",
        "2026-08-04T08:00:04-07:00",
        "2026-08-04T17:30:04+02:30",
    ],
)
def test_published_at_is_the_same_instant_in_utc(published: str) -> None:
    (e,) = parse(feed_xml(entry_xml(published=published)))
    assert e.published_at.utcoffset() == timedelta(0)
    assert e.published_at.tzinfo is UTC
    expected = datetime.fromisoformat(published)
    assert e.published_at == expected


def test_titles_keep_unicode_decode_entities_and_strip_whitespace() -> None:
    xml = feed_xml(
        entry_xml("aaaaaaaaaaa", title="  Привіт 你好 🎉 &amp; &lt;b&gt;\n "),
    )
    (e,) = parse(xml)
    assert e.title == "Привіт 你好 🎉 & <b>"


@pytest.mark.parametrize("title", [None, "", "   "])
def test_missing_or_empty_title_gives_empty_string(title: str | None) -> None:
    (e,) = parse(feed_xml(entry_xml(title=title)))
    assert e.title == ""


def test_feed_without_entries_returns_empty_list() -> None:
    assert parse(feed_xml()) == []


def test_feed_level_elements_are_not_entries() -> None:
    header = f"<yt:channelId>{CHANNEL}</yt:channelId><published>2006-01-01T00:00:00+00:00</published>"
    assert parse(feed_xml(header=header)) == []


def test_duplicate_video_id_is_returned_once_at_its_first_position() -> None:
    xml = feed_xml(
        entry_xml("aaaaaaaaaaa", title="first"),
        entry_xml("bbbbbbbbbbb"),
        entry_xml("aaaaaaaaaaa", title="second"),
    )
    entries = parse(xml)
    assert [(e.video_id, e.title) for e in entries] == [
        ("aaaaaaaaaaa", "first"),
        ("bbbbbbbbbbb", "A title"),
    ]


def test_order_is_document_order_not_date_order() -> None:
    xml = feed_xml(
        entry_xml("aaaaaaaaaaa", published="2020-01-01T00:00:00+00:00"),
        entry_xml("bbbbbbbbbbb", published="2026-01-01T00:00:00+00:00"),
    )
    assert [e.video_id for e in parse(xml)] == ["aaaaaaaaaaa", "bbbbbbbbbbb"]


# --- parsing: per-entry problems ------------------------------------------


@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        (entry_xml("short"), "video"),
        (entry_xml("aaaaaaaaaaaa"), "video"),
        (entry_xml("aaaaaaaaaa!"), "video"),
        (entry_xml(" aaaaaaaaaa"), "video"),
        (entry_xml(None), "video"),
        (entry_xml(published=None), "published"),
        (entry_xml(published="yesterday"), "published"),
        (entry_xml(published=""), "published"),
        (entry_xml(published="2026-08-04T15:00:04"), "published"),
        (entry_xml(published="2026-08-04"), "published"),
        (entry_xml(published="9999-12-31T23:59:59-23:59"), "published"),
        (entry_xml(channel="UC" + "b" * 22), "channel"),
        (entry_xml(channel=None), "channel"),
    ],
)
def test_a_bad_entry_is_skipped_with_a_warning_and_the_rest_survive(
    entry: str, reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    xml = feed_xml(entry_xml("aaaaaaaaaaa"), entry, entry_xml("bbbbbbbbbbb"))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        entries = parse(xml)
    assert [e.video_id for e in entries] == ["aaaaaaaaaaa", "bbbbbbbbbbb"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    text = warnings[0].getMessage()
    assert CHANNEL in text
    assert reason in text.lower()


def test_a_video_id_starting_with_dash_is_valid() -> None:
    (e,) = parse(feed_xml(entry_xml("-wNyEUrxzFU")))
    assert e.video_id == "-wNyEUrxzFU"


def test_entry_channel_id_without_the_uc_prefix_is_a_mismatch() -> None:
    # The real feed writes "UC..." on entries (only the feed-level element is
    # bare), so a bare entry ID is not accepted as the requested channel.
    assert parse(feed_xml(entry_xml(channel=CHANNEL[2:]))) == []


# --- parsing: whole-feed problems ------------------------------------------

WHOLE_FEED_BAD: list[bytes] = [
    b"",
    b"   ",
    b"<feed",
    b"not xml at all",
    b"<html><head><title>Before you continue</title></head><body>consent</body></html>",
    b"<!doctype html><html lang=en><meta charset=utf-8><title>captcha</title></html>",
    b'<?xml version="1.0"?><rss version="2.0"><channel/></rss>',
    b'<feed xmlns="http://example.com/other"></feed>',
    b"<feed></feed>",  # Atom name without the Atom namespace
    b"\xff\xfe\x00garbage",
    b'<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">\xff</feed>',
]


@pytest.mark.parametrize("data", WHOLE_FEED_BAD, ids=lambda d: repr(d[:25]))
def test_a_whole_feed_problem_raises_tool_failure(data: bytes) -> None:
    with pytest.raises(ToolFailureError):
        parse(data)


ATOM = 'xmlns="http://www.w3.org/2005/Atom"'


@pytest.mark.parametrize(
    "doc",
    [
        f'<!DOCTYPE feed [<!ENTITY a "b">]><feed {ATOM}><title>&a;</title></feed>',
        f"<!DOCTYPE feed><feed {ATOM}/>",
        f'<!DOCTYPE feed SYSTEM "http://evil.example/x.dtd"><feed {ATOM}/>',
        f'<!DOCTYPE feed [<!ENTITY x SYSTEM "file:///etc/passwd">]><feed {ATOM}/>',
        f'<?xml version="1.0"?>\n<!doctype feed><feed {ATOM}/>',
        f'<!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;">]><feed {ATOM}/>',
    ],
)
def test_doctype_and_entity_declarations_are_rejected(doc: str) -> None:
    with pytest.raises(ToolFailureError):
        parse(doc.encode())


def test_doctype_is_rejected_even_when_utf16_encoded() -> None:
    doc = f'<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE feed><feed {ATOM}/>'
    with pytest.raises(ToolFailureError):
        parse(doc.encode("utf-16"))


def test_the_word_doctype_in_a_title_is_not_a_declaration() -> None:
    (e,) = parse(feed_xml(entry_xml(title="How the &lt;!DOCTYPE&gt; tag works")))
    assert e.title == "How the <!DOCTYPE> tag works"


def test_parse_feed_rejects_a_bad_channel_id() -> None:
    with pytest.raises(ValueError):
        parse_feed(feed_xml(), "@NASA")


@given(st.binary(max_size=2000))
def test_arbitrary_bytes_give_valid_entries_or_tool_failure(data: bytes) -> None:
    try:
        result = parse(data)
    except ToolFailureError:
        return
    assert isinstance(result, list)
    for e in result:
        assert isinstance(e, FeedEntry)
        assert is_video_id(e.video_id)
        assert e.channel_id == CHANNEL
        assert e.published_at.utcoffset() == timedelta(0)


@given(
    st.lists(
        st.tuples(
            st.text(max_size=15),
            st.text(max_size=20),
            st.sampled_from(
                ["2026-08-04T15:00:04+00:00", "2026-08-04T15:00:04", "bad", ""]
            ),
        ),
        max_size=8,
    )
)
def test_structured_feeds_with_arbitrary_text_never_escape_tool_failure(
    items: list[tuple[str, str, str]],
) -> None:
    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;")

    entries = [
        entry_xml(esc(v), title=esc(t), published=p) for v, t, p in items
    ]
    try:
        result = parse(feed_xml(*entries))
    except ToolFailureError:
        return
    ids = [e.video_id for e in result]
    assert len(ids) == len(set(ids))
    assert all(is_video_id(i) for i in ids)


# --- shorts (#80) ------------------------------------------------------------


def _alt(href: str) -> str:
    return f'<link rel="alternate" href="{href}"/>'


VID = "aBcD3fGh1Jk"


@pytest.mark.parametrize(
    ("links", "expected"),
    [
        ((_alt(f"https://www.youtube.com/shorts/{VID}"),), True),
        ((_alt(f"https://youtube.com/shorts/{VID}"),), True),
        ((_alt(f"https://www.youtube.com/watch?v={VID}"),), False),
        ((), False),
        ((_alt(""),), False),
        (('<link rel="alternate"/>',), False),
        ((_alt("http://[bad"),), False),
        ((_alt("https://www.youtube.com/shorts/zzzzzzzzzzz"),), False),
        ((_alt(f"https://www.youtube.com/shorts/{VID}/extra"),), False),
        ((_alt(f"https://example.com/shorts/{VID}"),), False),
        ((f'<link rel="self" href="https://www.youtube.com/shorts/{VID}"/>',), False),
        (
            (
                _alt(f"https://www.youtube.com/watch?v={VID}"),
                _alt(f"https://www.youtube.com/shorts/{VID}"),
            ),
            True,
        ),
    ],
    ids=[
        "shorts_link",
        "bare_host",
        "watch_link",
        "no_link",
        "empty_href",
        "no_href",
        "unparseable_href",
        "other_video_id",
        "extra_path",
        "other_host",
        "not_alternate",
        "second_link_is_shorts",
    ],
)
def test_is_short_is_set_only_by_a_shorts_alternate_link_and_fails_open(
    links: tuple[str, ...], expected: bool
) -> None:
    entries = parse(feed_xml(entry_xml(VID, links=links)))

    assert [e.video_id for e in entries] == [VID]  # never dropped
    assert entries[0].is_short is expected


def test_the_title_does_not_make_an_entry_a_short() -> None:
    entries = parse(feed_xml(entry_xml(VID, title="#Shorts everything")))
    assert entries[0].is_short is False
