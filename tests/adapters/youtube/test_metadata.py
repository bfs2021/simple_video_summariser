"""YouTube metadata adapter (issue #16).

Fixtures are real, trimmed `yt-dlp --dump-json` output; see
fixtures/metadata/README.md. yt-dlp itself never runs: a ``FakeRunner`` stands
in for the process.
"""

import ast
import json
import pathlib
import re
import sys
from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

import adapters.youtube.metadata
import adapters.youtube.ytdlp
import common.models
from adapters.youtube.errors import UpcomingVideoError
from adapters.youtube.metadata import YouTubeMetadata
from common.errors import (
    PermanentSourceError,
    RateLimitedError,
    ToolFailureError,
)
from common.models import MetadataSource, VideoMeta
from tests.adapters.youtube.fakes import FakeRunner, completed

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "metadata"
DUMP_FLAGS = ["--dump-json", "--skip-download", "--no-playlist", "--no-warnings"]
YTDLP = [sys.executable, "-m", "yt_dlp"]


def fixture_text(name: str) -> str:
    return (FIXTURES / f"{name}.json").read_text(encoding="utf-8")


def fixture_dict(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(fixture_text(name))
    return data


def fetch_json(info: str | dict[str, Any], video_id: str | None = None) -> VideoMeta:
    stdout = info if isinstance(info, str) else json.dumps(info)
    if video_id is None:
        video_id = json.loads(stdout)["id"]
    source = YouTubeMetadata(runner=FakeRunner(completed(stdout=stdout)))
    return source.fetch(video_id)


def utc(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


# --- port ------------------------------------------------------------------


def test_youtube_metadata_is_a_metadata_source() -> None:
    source: MetadataSource = YouTubeMetadata(runner=FakeRunner())
    assert callable(source.fetch)


def test_video_meta_is_immutable() -> None:
    meta = fetch_json(fixture_text("no_captions"))
    with pytest.raises(AttributeError):
        meta.title = "changed"  # type: ignore[misc]


# --- input validation ------------------------------------------------------


@pytest.mark.parametrize(
    "video_id",
    [
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        "jNQXAC9IVR",  # 10 characters
        "jNQXAC9IVRww",  # 12 characters
        "jNQX C9IVRw",  # space
        "jNQXAC9;IVR",  # ;
        "jNQXAC9IVR\n",  # 11 characters with a trailing newline
        "\njNQXAC9IVR",
        "jNQXAC9IVRé",  # non-ASCII letter
        "jNQXAC9IVR.",
        "",
    ],
)
def test_a_malformed_video_id_is_rejected_before_any_process_starts(
    video_id: str,
) -> None:
    runner = FakeRunner()
    with pytest.raises(ValueError):
        YouTubeMetadata(runner=runner).fetch(video_id)
    assert runner.calls == []


_VALID = re.compile(r"[A-Za-z0-9_-]{11}")


@given(st.text(max_size=20).filter(lambda s: not _VALID.fullmatch(s)))
def test_any_string_that_is_not_an_11_char_id_is_rejected(video_id: str) -> None:
    runner = FakeRunner()
    with pytest.raises(ValueError):
        YouTubeMetadata(runner=runner).fetch(video_id)
    assert runner.calls == []


@given(st.from_regex(_VALID, fullmatch=True))
def test_every_11_char_id_reaches_yt_dlp_as_a_full_watch_url(video_id: str) -> None:
    info = fixture_dict("no_captions") | {"id": video_id}
    runner = FakeRunner(completed(stdout=json.dumps(info)))

    YouTubeMetadata(runner=runner).fetch(video_id)

    [(argv, _)] = runner.calls
    assert argv[-1] == f"https://www.youtube.com/watch?v={video_id}"
    assert video_id not in argv[:-1]


def test_an_id_starting_with_a_dash_is_passed_as_a_watch_url_never_bare() -> None:
    video_id = "-wNyEUrxzFU"
    info = fixture_dict("no_captions") | {"id": video_id}
    runner = FakeRunner(completed(stdout=json.dumps(info)))

    meta = YouTubeMetadata(runner=runner, timeout=42).fetch(video_id)

    assert meta.video_id == video_id
    assert runner.calls == [
        ([*YTDLP, *DUMP_FLAGS, "https://www.youtube.com/watch?v=-wNyEUrxzFU"], 42)
    ]


# --- recorded fixtures -----------------------------------------------------


def test_normal_video_with_manual_english_subtitles() -> None:
    meta = fetch_json(fixture_text("manual_en"))

    assert meta.video_id == "iG9CE55wbtY"
    assert meta.channel_id == "UCAuUUnT6oDeKwE6v1NGQxug"
    assert meta.title == "Do schools kill creativity? | Sir Ken Robinson | TED"
    assert meta.description.startswith("Visit http://TED.com")
    assert meta.duration_sec == 1203
    assert meta.published_at == datetime.fromtimestamp(1168146034, UTC)
    assert meta.language == "en"
    assert meta.live_status == "not_live"
    assert "en" in meta.manual_subtitle_langs
    assert len(meta.manual_subtitle_langs) == 64
    assert {"en", "en-orig"} <= set(meta.auto_caption_langs)


def test_auto_captions_only_video() -> None:
    meta = fetch_json(fixture_text("auto_only"))

    assert meta.video_id == "zjkBMFhNj_g"
    assert meta.manual_subtitle_langs == ()
    assert {"en-orig", "en"} <= set(meta.auto_caption_langs)
    assert len(meta.auto_caption_langs) == 157
    assert meta.language == "en"
    assert meta.duration_sec == 3588
    assert meta.live_status == "not_live"


def test_video_with_no_captions() -> None:
    meta = fetch_json(fixture_text("no_captions"))

    assert meta.video_id == "BZiu46G4ukc"
    assert meta.manual_subtitle_langs == ()
    assert meta.auto_caption_langs == ()
    assert meta.language is None
    assert meta.duration_sec == 3643
    assert meta.published_at == utc(2023, 8, 22, 22, 1, 28)


def test_short() -> None:
    meta = fetch_json(fixture_text("short"))

    assert meta.video_id == "VV_JW4iCni0"
    assert meta.channel_id == "UCLA_DiR1FfKNvjuUpBHmylQ"
    assert meta.title == "NASA Moon Base Update (Aug. 4, 2026)"
    assert meta.duration_sec == 32
    assert meta.language == "en-US"
    assert meta.live_status == "not_live"
    assert meta.manual_subtitle_langs == ("en",)
    assert "en" in meta.auto_caption_langs


# --- is_short (#80): yt-dlp's media_type, never the duration -----------------


def test_the_short_fixture_is_a_short() -> None:
    assert fetch_json(fixture_text("short")).is_short is True


@pytest.mark.parametrize(
    "name", ["manual_en", "auto_only", "no_captions", "non_english_translated_en", "upcoming_premiere"]
)
def test_every_other_fixture_is_not_a_short(name: str) -> None:
    assert fetch_json(fixture_text(name)).is_short is False


@pytest.mark.parametrize("media_type", [None, "", "livestream", "story", "SHORT"])
def test_a_missing_null_or_other_media_type_is_not_a_short(media_type: object) -> None:
    raw = fixture_dict("short") | {"media_type": media_type}
    assert fetch_json(raw).is_short is False
    raw.pop("media_type")
    assert fetch_json(raw).is_short is False


def test_a_32_second_video_labelled_video_is_not_a_short() -> None:
    raw = fixture_dict("short") | {"media_type": "video"}
    assert raw["duration"] == 32
    assert fetch_json(raw).is_short is False


def test_a_long_short_is_still_a_short() -> None:
    raw = fixture_dict("manual_en") | {"media_type": "short", "duration": 179}
    assert fetch_json(raw).is_short is True
    assert fetch_json(raw | {"duration": 600}).is_short is True


def test_upcoming_premiere() -> None:
    meta = fetch_json(fixture_text("upcoming_premiere"))

    assert meta.video_id == "WI4__7z_dJw"
    assert meta.live_status == "is_upcoming"
    assert meta.duration_sec is None
    assert meta.published_at == datetime.fromtimestamp(1790484723, UTC)
    assert meta.manual_subtitle_langs == ()
    assert meta.auto_caption_langs == ()


def test_release_at_is_the_scheduled_start_not_the_upload_time() -> None:
    meta = fetch_json(fixture_text("upcoming_premiere"))

    assert meta.release_at == datetime.fromtimestamp(1790585940, UTC)
    assert meta.release_at != meta.published_at


@pytest.mark.parametrize("value", [None, True, "soon", 1e30, -1e30])
def test_release_at_is_none_when_release_timestamp_is_unusable(value: object) -> None:
    raw = fixture_dict("upcoming_premiere")
    raw["release_timestamp"] = value

    assert fetch_json(raw).release_at is None


def test_release_at_is_none_when_release_timestamp_is_absent() -> None:
    raw = fixture_dict("upcoming_premiere")
    del raw["release_timestamp"]

    assert fetch_json(raw).release_at is None


def test_non_english_video_with_translated_english_auto_captions() -> None:
    raw = fixture_dict("non_english_translated_en")
    # The recording shows `en` is a machine translation of the Ukrainian track.
    assert "lang=uk&tlang=en" in raw["automatic_captions"]["en"][0]["url"]

    meta = fetch_json(raw)

    assert meta.video_id == "pnMM6MwpaTc"
    assert meta.language == "uk"
    assert meta.manual_subtitle_langs == ()
    assert "uk-orig" in meta.auto_caption_langs
    assert "en" in meta.auto_caption_langs


def test_live_chat_never_appears_in_manual_subtitle_langs() -> None:
    raw = fixture_dict("upcoming_premiere")
    assert "live_chat" in raw["subtitles"]  # the recording really has it

    meta = fetch_json(raw)

    assert "live_chat" not in meta.manual_subtitle_langs


def test_live_chat_is_dropped_but_other_manual_languages_are_kept() -> None:
    raw = fixture_dict("manual_en")
    raw["subtitles"] = {"live_chat": [{"ext": "json"}], **raw["subtitles"]}

    meta = fetch_json(raw)

    assert "live_chat" not in meta.manual_subtitle_langs
    assert len(meta.manual_subtitle_langs) == 64


# --- upcoming videos: yt-dlp refuses them without --ignore-no-formats-error ---

# live run on 2026-09-27, yt-dlp 2026.08.19, with DUMP_FLAGS and
# https://www.youtube.com/watch?v=WI4__7z_dJw (the recorded premiere), full stderr
PREMIERE_STDERR = "ERROR: [youtube] WI4__7z_dJw: Premieres in 12 hours\n"


def test_an_upcoming_video_is_fetched_again_ignoring_the_missing_formats() -> None:
    runner = FakeRunner(
        completed(stderr=PREMIERE_STDERR, returncode=1),
        completed(stdout=fixture_text("upcoming_premiere")),
    )

    meta = YouTubeMetadata(runner=runner, timeout=42).fetch("WI4__7z_dJw")

    assert meta.live_status == "is_upcoming"
    assert meta.duration_sec is None
    url = "https://www.youtube.com/watch?v=WI4__7z_dJw"
    assert runner.calls == [
        ([*YTDLP, *DUMP_FLAGS, url], 42),
        ([*YTDLP, *DUMP_FLAGS, "--ignore-no-formats-error", url], 42),
    ]


def test_a_failing_second_fetch_of_an_upcoming_video_raises_its_error() -> None:
    runner = FakeRunner(
        completed(stderr=PREMIERE_STDERR, returncode=1),
        completed(stderr=PREMIERE_STDERR, returncode=1),
    )
    with pytest.raises(UpcomingVideoError):
        YouTubeMetadata(runner=runner).fetch("WI4__7z_dJw")


# --- published_at ----------------------------------------------------------


def test_published_at_prefers_timestamp() -> None:
    raw = fixture_dict("upcoming_premiere")
    assert raw["timestamp"] != raw["release_timestamp"]
    assert fetch_json(raw).published_at == datetime.fromtimestamp(raw["timestamp"], UTC)


def test_published_at_falls_back_to_release_timestamp() -> None:
    raw = fixture_dict("upcoming_premiere")
    del raw["timestamp"]
    assert fetch_json(raw).published_at == datetime.fromtimestamp(1790585940, UTC)


def test_published_at_falls_back_to_upload_date_at_midnight_utc() -> None:
    raw = fixture_dict("manual_en") | {"timestamp": None, "release_timestamp": None}
    assert fetch_json(raw).published_at == utc(2007, 1, 7)


def test_an_out_of_range_timestamp_falls_through_to_the_next_source() -> None:
    raw = fixture_dict("upcoming_premiere") | {"timestamp": 10**20}
    assert fetch_json(raw).published_at == datetime.fromtimestamp(1790585940, UTC)


def test_published_at_is_none_when_nothing_is_reported() -> None:
    raw = fixture_dict("manual_en")
    for key in ("timestamp", "release_timestamp", "upload_date"):
        raw.pop(key)
    assert fetch_json(raw).published_at is None


@pytest.mark.parametrize("name", ["manual_en", "short", "upcoming_premiere"])
def test_published_at_is_timezone_aware_utc(name: str) -> None:
    published_at = fetch_json(fixture_text(name)).published_at
    assert published_at is not None
    assert published_at.utcoffset() is not None
    assert published_at.utcoffset().total_seconds() == 0  # type: ignore[union-attr]


# --- optional fields -------------------------------------------------------


def test_absent_optional_fields_become_none_or_empty() -> None:
    raw = fixture_dict("manual_en")
    for key in (
        "description",
        "duration",
        "language",
        "live_status",
        "subtitles",
        "automatic_captions",
    ):
        raw.pop(key)

    meta = fetch_json(raw)

    assert meta.description == ""
    assert meta.duration_sec is None
    assert meta.language is None
    assert meta.live_status is None
    assert meta.manual_subtitle_langs == ()
    assert meta.auto_caption_langs == ()


def test_a_fractional_duration_becomes_whole_seconds() -> None:
    raw = fixture_dict("short") | {"duration": 32.4}
    assert fetch_json(raw).duration_sec == 32


# --- failures --------------------------------------------------------------

# Real stderr lines, copied from tests/fixtures/ytdlp_errors/cases.toml (#15).
REMOVED_STDERR = (
    "WARNING: [youtube] No supported JavaScript runtime could be found.\n"
    "ERROR: [youtube] sJL6WA-aGkQ: Video unavailable\n"
)
PRIVATE_STDERR = "ERROR: [youtube] yZIXLfi8CZQ: Private video\n"
AGEGATED_STDERR = (
    "ERROR: [youtube] Tq92D6wQ1mg: Sign in to confirm your age. Use "
    "--cookies-from-browser or --cookies for the authentication. See  "
    "https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp  "
    "for how to manually pass cookies. Also see  "
    "https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies  "
    "for tips on effectively exporting YouTube cookies\n"
)
RATE_LIMITED_STDERR = (
    "ERROR: [generic] 429: Unable to download webpage: HTTP Error 429: Too Many "
    "Requests (caused by <HTTPError 429: Too Many Requests>)\n"
)


@pytest.mark.parametrize(
    ("video_id", "stderr", "reason"),
    [
        ("sJL6WA-aGkQ", REMOVED_STDERR, "removed"),
        ("yZIXLfi8CZQ", PRIVATE_STDERR, "private"),
        ("Tq92D6wQ1mg", AGEGATED_STDERR, "agegated"),
    ],
    ids=["removed", "private", "agegated"],
)
def test_an_unavailable_video_raises_permanent_source_with_its_reason(
    video_id: str, stderr: str, reason: str
) -> None:
    runner = FakeRunner(completed(stderr=stderr, returncode=1))
    with pytest.raises(PermanentSourceError) as info:
        YouTubeMetadata(runner=runner).fetch(video_id)
    assert info.value.reason == reason
    assert len(runner.calls) == 1


def test_a_429_raises_rate_limited() -> None:
    runner = FakeRunner(completed(stderr=RATE_LIMITED_STDERR, returncode=1))
    with pytest.raises(RateLimitedError):
        YouTubeMetadata(runner=runner).fetch("jNQXAC9IVRw")


@pytest.mark.parametrize(
    "stdout",
    [
        '{"id": "jNQXAC9IVRw", "title": "Me at the zoo", "formats": [{"format_id": "233", ',
        "",
        "not json",
    ],
    ids=["truncated", "empty", "garbage"],
)
def test_output_that_is_not_valid_json_raises_tool_failure(stdout: str) -> None:
    runner = FakeRunner(completed(stdout=stdout))
    with pytest.raises(ToolFailureError):
        YouTubeMetadata(runner=runner).fetch("jNQXAC9IVRw")


@pytest.mark.parametrize("stdout", ["[]", "null", '"jNQXAC9IVRw"', "42"])
def test_json_that_is_not_an_object_raises_tool_failure(stdout: str) -> None:
    runner = FakeRunner(completed(stdout=stdout))
    with pytest.raises(ToolFailureError):
        YouTubeMetadata(runner=runner).fetch("jNQXAC9IVRw")


@pytest.mark.parametrize("key", ["id", "channel_id", "title"])
def test_a_missing_required_field_raises_tool_failure(key: str) -> None:
    raw = fixture_dict("manual_en")
    del raw[key]
    runner = FakeRunner(completed(stdout=json.dumps(raw)))
    with pytest.raises(ToolFailureError):
        YouTubeMetadata(runner=runner).fetch("iG9CE55wbtY")


def test_output_for_a_different_video_raises_tool_failure() -> None:
    runner = FakeRunner(completed(stdout=fixture_text("manual_en")))
    with pytest.raises(ToolFailureError):
        YouTubeMetadata(runner=runner).fetch("jNQXAC9IVRw")


# --- import direction ------------------------------------------------------


def _top_level_imports(module: object) -> set[str]:
    tree = ast.parse(pathlib.Path(module.__file__).read_text())  # type: ignore[attr-defined]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize(
    "module", [adapters.youtube.metadata, adapters.youtube.ytdlp], ids=str
)
def test_youtube_adapters_import_no_service(module: object) -> None:
    assert "services" not in _top_level_imports(module)


def test_common_models_imports_no_adapter() -> None:
    assert not {"adapters", "services"} & _top_level_imports(common.models)
