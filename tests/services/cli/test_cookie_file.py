"""``default_deps`` and ``YTDLP_COOKIE_FILE`` (issue #63)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.youtube.ytdlp import CookieFileError
from services.cli.main import default_deps
from tests.services.transcriber.fakes import make_settings


def test_default_deps_refuses_a_set_but_missing_cookie_file(tmp_path: Path) -> None:
    settings = make_settings(YTDLP_COOKIE_FILE=tmp_path / "missing.txt", SUMMARIZER="ollama")
    with pytest.raises(CookieFileError, match="YTDLP_COOKIE_FILE"):
        default_deps(settings)
