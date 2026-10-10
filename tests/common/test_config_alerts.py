"""Alert settings (issue #67)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from common.config import Settings

DB = "postgresql://u:p@h/db"
URL = "https://hooks.example.test/T000/s3cr3tT0ken"


def _settings(**values: Any) -> Settings:
    # ``Any`` lets plain strings stand in for the SecretStr fields.
    return Settings(**{"DATABASE_URL": DB, **values})


def test_defaults_mean_no_channel_hourly_cooldown_json() -> None:
    s = _settings()

    assert s.ALERT_WEBHOOK_URL is None
    assert s.ALERT_MIN_INTERVAL_SEC == 3600
    assert s.ALERT_WEBHOOK_FORMAT == "json"


def test_blank_webhook_url_means_unset() -> None:
    assert _settings(ALERT_WEBHOOK_URL="").ALERT_WEBHOOK_URL is None


def test_webhook_url_is_a_secret() -> None:
    s = _settings(ALERT_WEBHOOK_URL=URL)

    assert s.ALERT_WEBHOOK_URL is not None
    assert s.ALERT_WEBHOOK_URL.get_secret_value() == URL
    assert "s3cr3tT0ken" not in repr(s)
    assert "s3cr3tT0ken" not in str(s)


@pytest.mark.parametrize("bad", ["ftp://x/SECRETX", "not a url SECRETX", "http:///SECRETX", "ntfy.sh/SECRETX"])
def test_bad_webhook_url_is_rejected_without_echoing_it(bad: str) -> None:
    with pytest.raises(ValidationError) as info:
        _settings(ALERT_WEBHOOK_URL=bad)

    assert "ALERT_WEBHOOK_URL" in str(info.value)
    assert "SECRETX" not in str(info.value)


def test_interval_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _settings(ALERT_MIN_INTERVAL_SEC=0)
