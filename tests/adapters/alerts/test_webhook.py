"""Webhook alert channel (issue #67). No network: the POST is an injected seam."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from adapters.alerts.webhook import AlertDeliveryError, WebhookAlerter

URL = "https://hooks.example.test/T000/B000/s3cr3tT0ken"


class Recorder:
    def __init__(self, status: int = 200, raises: Exception | None = None) -> None:
        self.status = status
        self.raises = raises
        self.calls: list[tuple[str, bytes, Mapping[str, str], float]] = []

    def __call__(self, url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> int:
        self.calls.append((url, body, headers, timeout))
        if self.raises is not None:
            raise self.raises
        return self.status


def test_json_format_posts_a_slack_compatible_text_object() -> None:
    rec = Recorder()

    WebhookAlerter(URL, fmt="json", transport=rec).send("Title", "Body line")

    (url, body, headers, timeout) = rec.calls[0]
    assert url == URL
    assert json.loads(body) == {"text": "Title\nBody line"}
    assert headers["Content-Type"] == "application/json"
    assert 0 < timeout <= 30


def test_text_format_posts_the_message_with_a_title_header_for_ntfy() -> None:
    rec = Recorder()

    WebhookAlerter(URL, fmt="text", transport=rec).send("Title", "Body line")

    (_, body, headers, _) = rec.calls[0]
    assert body == b"Body line"
    assert headers["Title"] == "Title"
    assert headers["Content-Type"].startswith("text/plain")


@pytest.mark.parametrize("status", [199, 300, 404, 500])
def test_non_2xx_raises_without_the_url(status: int) -> None:
    alerter = WebhookAlerter(URL, transport=Recorder(status=status))

    with pytest.raises(AlertDeliveryError) as info:
        alerter.send("t", "m")

    assert str(status) in str(info.value)
    assert "s3cr3tT0ken" not in str(info.value)


@pytest.mark.parametrize("status", [200, 201, 204, 299])
def test_any_2xx_is_delivered(status: int) -> None:
    WebhookAlerter(URL, transport=Recorder(status=status)).send("t", "m")


def test_a_transport_error_is_wrapped_scrubbed_and_not_chained() -> None:
    boom = ConnectionError(f"cannot reach {URL} (refused)")
    alerter = WebhookAlerter(URL, transport=Recorder(raises=boom))

    with pytest.raises(AlertDeliveryError) as info:
        alerter.send("t", "m")

    assert "s3cr3tT0ken" not in str(info.value)
    assert "ConnectionError" in str(info.value)
    # `from None`: a traceback logged later must not print the original text.
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True


def test_repr_and_str_never_show_the_url() -> None:
    alerter = WebhookAlerter(URL, transport=Recorder())

    assert "s3cr3tT0ken" not in repr(alerter)
    assert "s3cr3tT0ken" not in str(alerter)


def test_unknown_format_is_rejected() -> None:
    with pytest.raises(ValueError):
        WebhookAlerter(URL, fmt="xml", transport=Recorder())  # type: ignore[arg-type]
