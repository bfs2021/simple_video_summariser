"""``YTDLP_COOKIE_FILE`` in the transcriber (issue #63).

A bad path stops the service at startup, once, with a message naming the
variable. It must not become a failure of every job.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import services.transcriber.main as main_module
from common.config import get_settings
from services.transcriber.main import build_handlers, main
from tests.services.transcriber.fakes import make_settings

SECRET = "SENTINEL-cookie-value-123456"


@pytest.fixture(autouse=True)
def valid_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    monkeypatch.setenv("AUDIO_DIR", str(tmp_path / "audio"))
    monkeypatch.delenv("YTDLP_COOKIE_FILE", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _no_queue() -> Any:
    raise AssertionError("the worker must not start with a bad cookie file")


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_a_set_but_unusable_cookie_file_exits_2_naming_the_variable(
    kind: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "SENTINEL-nope.txt"
    if kind == "directory":
        target = tmp_path
    monkeypatch.setenv("YTDLP_COOKIE_FILE", str(target))

    rc = main([], handlers={}, queue_factory=_no_queue)

    captured = capsys.readouterr()
    assert rc == 2
    assert "YTDLP_COOKIE_FILE" in captured.err
    assert SECRET not in captured.err + captured.out


def test_the_default_adapters_receive_the_configured_cookie_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cookie = tmp_path / "cookies.txt"
    seen: dict[str, Path | None] = {}

    def recorder(name: str) -> Any:
        class Recorder:
            def __init__(self, *, cookie_file: Path | None = None, **_: Any) -> None:
                seen[name] = cookie_file

        return Recorder

    monkeypatch.setattr(main_module, "YouTubeAudio", recorder("audio"))
    monkeypatch.setattr(main_module, "YouTubeMetadata", recorder("metadata"))
    monkeypatch.setattr(main_module, "YouTubeSubtitles", recorder("subtitles"))

    build_handlers(
        make_settings(YTDLP_COOKIE_FILE=cookie),
        transcriber=None,  # type: ignore[arg-type]
    )

    assert seen == {"audio": cookie, "metadata": cookie, "subtitles": cookie}


def test_without_a_cookie_file_the_adapters_get_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Path | None] = []

    class Recorder:
        def __init__(self, *, cookie_file: Path | None = None, **_: Any) -> None:
            seen.append(cookie_file)

    monkeypatch.setattr(main_module, "YouTubeAudio", Recorder)
    monkeypatch.setattr(main_module, "YouTubeMetadata", Recorder)
    monkeypatch.setattr(main_module, "YouTubeSubtitles", Recorder)

    build_handlers(make_settings(), transcriber=None)  # type: ignore[arg-type]

    assert seen == [None, None, None]


