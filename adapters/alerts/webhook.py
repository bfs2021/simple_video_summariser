"""Webhook alert channel (issue #67; architecture.md 8.4, 16.11).

``WebhookAlerter.send`` POSTs one message to a configured URL: ntfy takes the
plain text body (``fmt="text"``, title in a ``Title`` header), a
Slack-compatible endpoint takes ``{"text": ...}`` (``fmt="json"``).

The URL is a secret. It is held only in a private attribute, left out of
``repr``, and every error this module raises is raised ``from None`` and is
built from safe parts only: exception *types* and, for a URL, the parsed
``scheme://host[:port]`` (``common.safe_url``). Exception text is never quoted,
because ``InvalidURL`` and friends echo fragments of the URL that no
whole-string scrub can match.
The POST goes through an injectable ``transport`` so tests make no network
call.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Literal

from common.safe_url import public_url

#: ``transport(url, body, headers, timeout_sec)`` returns the HTTP status.
#: HTTP error statuses are returned, not raised; any other failure raises.
type PostTransport = Callable[[str, bytes, Mapping[str, str], float], int]

DELIVERY_TIMEOUT_SEC = 10.0


class AlertDeliveryError(Exception):
    """The alert could not be delivered. The message never contains the URL."""


def urllib_post_transport(
    url: str, body: bytes, headers: Mapping[str, str], timeout: float
) -> int:
    request = urllib.request.Request(url, data=body, method="POST", headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as err:
        with err:
            return int(err.code)


class WebhookAlerter:
    def __init__(
        self,
        url: str,
        *,
        fmt: Literal["json", "text"] = "json",
        transport: PostTransport = urllib_post_transport,
    ) -> None:
        if fmt not in ("json", "text"):
            raise ValueError(f"unknown alert format {fmt!r}")
        self.__url = url
        self._fmt = fmt
        self._transport = transport

    def __repr__(self) -> str:
        return f"WebhookAlerter(fmt={self._fmt!r})"

    def send(self, title: str, message: str) -> None:
        if self._fmt == "json":
            body = json.dumps({"text": f"{title}\n{message}"}).encode()
            headers = {"Content-Type": "application/json"}
        else:
            body = message.encode()
            headers = {"Content-Type": "text/plain; charset=utf-8", "Title": title}
        try:
            status = self._transport(self.__url, body, headers, DELIVERY_TIMEOUT_SEC)
        except Exception as err:  # noqa: BLE001 - any transport failure is a delivery failure
            detail = type(err).__name__
            if isinstance(err, urllib.error.URLError) and isinstance(err.reason, BaseException):
                detail += f" ({type(err.reason).__name__})"
            where = public_url(self.__url, invalid="<invalid webhook URL>")
            raise AlertDeliveryError(f"alert not delivered to {where}: {detail}") from None
        if not 200 <= status < 300:
            raise AlertDeliveryError(f"alert not delivered: HTTP {status}")
