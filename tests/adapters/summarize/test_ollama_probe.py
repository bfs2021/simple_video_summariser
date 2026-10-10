"""The Ollama startup probe (issue #84): one ``GET /api/tags``, no network in tests."""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Self

import pytest

from adapters.summarize.ollama import (
    MAX_RESPONSE_BYTES,
    PROBE_DEADLINE_SEC,
    PROBE_TIMEOUT_SEC,
    HttpResponse,
    probe_ollama,
    public_host,
    urllib_get_transport,
)

HOST = "http://h:11434"
MODEL = "qwen3.5:4b"


class FakeGet:
    """A recording ``GetTransport``: returns ``result`` or raises it."""

    def __init__(self, result: HttpResponse | BaseException) -> None:
        self.result = result
        self.calls: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout: float) -> HttpResponse:
        self.calls.append((url, timeout))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def tags(*names: Any, key: str = "name") -> HttpResponse:
    models = [{key: n} if isinstance(n, str) else n for n in names]
    return HttpResponse(200, json.dumps({"models": models}).encode())


# -- request ------------------------------------------------------------------


@pytest.mark.parametrize("host", ["http://h:11434", "http://h:11434/", "http://h:11434///"])
def test_sends_exactly_one_get_to_api_tags_without_a_double_slash(host: str) -> None:
    transport = FakeGet(tags(MODEL))

    probe_ollama(host, MODEL, transport=transport)

    assert transport.calls == [("http://h:11434/api/tags", PROBE_TIMEOUT_SEC)]


def test_the_probe_timeout_is_a_fixed_five_seconds_not_the_generation_timeout() -> None:
    assert PROBE_TIMEOUT_SEC == 5


def test_a_transport_that_never_answers_is_given_at_most_the_probe_timeout() -> None:
    # The transport owns the wait; the probe must hand it the 5 s bound and report
    # the timeout it raises, without any clock or sleep of its own.
    def hangs(url: str, timeout: float) -> HttpResponse:
        assert timeout <= 5
        raise TimeoutError("timed out")

    result = probe_ollama(HOST, MODEL, transport=hangs)

    assert result.outcome == "unreachable"
    assert result.error == "TimeoutError: timed out"


# -- outcomes -----------------------------------------------------------------


def test_ready_when_the_model_is_listed() -> None:
    result = probe_ollama(HOST, MODEL, transport=FakeGet(tags("other:1b", MODEL)))

    assert (result.outcome, result.error) == ("ready", None)
    assert (result.host, result.model) == (HOST, MODEL)


def test_ready_matches_the_model_key_as_well_as_name() -> None:
    result = probe_ollama(HOST, MODEL, transport=FakeGet(tags(MODEL, key="model")))

    assert result.outcome == "ready"


@pytest.mark.parametrize(
    ("wanted", "listed", "outcome"),
    [
        ("qwen3.5", "qwen3.5:latest", "ready"),
        ("qwen3.5:latest", "qwen3.5", "ready"),
        ("qwen3.5:4b", "qwen3.5:4b", "ready"),
        ("qwen3.5:4b", "qwen3.5:8b", "model_missing"),
        ("qwen3.5:4b", "qwen3.5:4b-q8_0", "model_missing"),
        ("qwen3.5:4b", "qwen3.5", "model_missing"),
        ("qwen3.5", "qwen3.5:4b", "model_missing"),
        ("qwen3.5:4b", "qwen3.5:4B", "model_missing"),
        ("reg:5000/lib/m", "reg:5000/lib/m:latest", "ready"),
    ],
)
def test_matching_follows_the_implicit_latest_tag_and_is_otherwise_exact(
    wanted: str, listed: str, outcome: str
) -> None:
    result = probe_ollama(HOST, wanted, transport=FakeGet(tags(listed)))

    assert result.outcome == outcome


def test_model_missing_for_a_well_formed_body_without_it_including_an_empty_list() -> None:
    for response in (tags("other:1b"), tags()):
        result = probe_ollama(HOST, MODEL, transport=FakeGet(response))
        assert result.outcome == "model_missing"
        assert result.error is None


def test_malformed_entries_are_skipped_and_other_entries_can_still_match() -> None:
    body = {"models": ["str", 7, None, [], {"name": 5}, {"model": None}, {}, {"name": MODEL}]}

    result = probe_ollama(HOST, MODEL, transport=FakeGet(HttpResponse(200, json.dumps(body).encode())))

    assert result.outcome == "ready"


def test_only_malformed_entries_means_model_missing_not_an_error() -> None:
    body = {"models": [1, {"name": 3}]}

    result = probe_ollama(HOST, MODEL, transport=FakeGet(HttpResponse(200, json.dumps(body).encode())))

    assert result.outcome == "model_missing"


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionRefusedError("refused"),
        OSError("Name or service not known"),
        TimeoutError("timed out"),
        urllib.error.URLError("boom"),
        RuntimeError("anything"),
    ],
)
def test_any_transport_exception_is_unreachable_with_class_and_message(exc: Exception) -> None:
    result = probe_ollama(HOST, MODEL, transport=FakeGet(exc))

    assert result.outcome == "unreachable"
    assert result.error is not None
    assert result.error.startswith(f"{type(exc).__name__}: ")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (HttpResponse(500, b"server exploded"), "HTTP 500: server exploded"),
        (HttpResponse(404, b""), "HTTP 404"),
        (HttpResponse(301, b"moved"), "HTTP 301: moved"),
        (HttpResponse(200, b"<html>not json"), "not JSON"),
        (HttpResponse(200, b"\xff\xfe"), "not JSON"),
        (HttpResponse(200, b"[1, 2]"), "not a JSON object"),
        (HttpResponse(200, b"null"), "not a JSON object"),
        (HttpResponse(200, b"{}"), "no models list"),
        (HttpResponse(200, b'{"models": null}'), "no models list"),
        (HttpResponse(200, b'{"models": {"a": 1}}'), "no models list"),
        (HttpResponse(200, b"[" * 100000), "not JSON"),
    ],
)
def test_an_unexpected_answer_is_unreachable_with_a_reason(
    response: HttpResponse, expected: str
) -> None:
    result = probe_ollama(HOST, MODEL, transport=FakeGet(response))

    assert result.outcome == "unreachable"
    assert result.error is not None and expected in result.error


def test_the_quoted_body_is_cut_to_200_characters() -> None:
    result = probe_ollama(HOST, MODEL, transport=FakeGet(HttpResponse(502, b"x" * 5000)))

    assert result.error == "HTTP 502: " + "x" * 200


def test_a_response_larger_than_max_response_bytes_is_not_trusted() -> None:
    big = HttpResponse(200, b" " * (MAX_RESPONSE_BYTES + 1))

    result = probe_ollama(HOST, MODEL, transport=FakeGet(big))

    assert result.outcome == "unreachable"
    assert result.error is not None and str(MAX_RESPONSE_BYTES) in result.error


def test_a_malformed_host_is_unreachable_not_an_exception() -> None:
    def transport(url: str, timeout: float) -> HttpResponse:
        raise ValueError(f"bad url {url}")

    result = probe_ollama("http://[::1", MODEL, transport=transport)

    assert result.outcome == "unreachable"


# -- credentials --------------------------------------------------------------


def test_public_host_removes_userinfo_and_keeps_the_rest() -> None:
    assert public_host("http://user:s3cr3t@h:11434/base") == "http://h:11434"
    assert public_host("https://tok@[::1]:11434") == "https://[::1]:11434"
    assert public_host("http://h:11434") == "http://h:11434"
    assert public_host("http://h") == "http://h"
    assert public_host("HTTP://H:1/") == "http://h:1"


def test_result_and_error_never_contain_the_url_credentials() -> None:
    host = "http://user:s3cr3t-PW@h:11434"

    def transport(url: str, timeout: float) -> HttpResponse:
        raise ConnectionError(f"cannot reach {url}")

    result = probe_ollama(host, MODEL, transport=transport)

    assert result.host == "http://h:11434"
    assert result.error is not None
    assert "s3cr3t-PW" not in result.error and "user:" not in result.error
    assert "s3cr3t-PW" not in repr(result)


def test_a_response_body_that_echoes_the_credentials_is_scrubbed() -> None:
    host = "http://user:s3cr3t-PW@h:11434"

    result = probe_ollama(host, MODEL, transport=FakeGet(HttpResponse(500, b"bad s3cr3t-PW here")))

    assert result.error is not None and "s3cr3t-PW" not in result.error


SECRET = "S3CR3T-tok"

#: Every placement of a secret in a host string, including the ones RFC parsing
#: does not call userinfo (fragment, path, query, params) and a password with ``@``/``#``.
LEAK_HOSTS = [
    f"http://bob:{SECRET}@h:11434",
    f"http://bob:p@{SECRET}@h:11434",
    f"http://bob:{SECRET}%40x@h:11434",
    f"http://bob:{SECRET}%2Fx@h:11434",
    f"http://bob#{SECRET}@h:11434",
    f"http://bob/{SECRET}:x@h:11434",
    f"http://bob:pa#{SECRET}@h:11434",
    f"http://bob:pa#{SECRET}@h:11434/x",
    f"http://h:11434/{SECRET}",
    f"http://h:11434/base?token={SECRET}",
    f"http://h:11434/base;{SECRET}",
    f"http://h:11434/base;p={SECRET}?q=1#{SECRET}",
    f"http://h:11434?token={SECRET}",
    f"http://h:11434#{SECRET}",
    f"http://{SECRET}@h:11434",
    f"https://bob:{SECRET}@[::1]:11434",
    f"HTTP://bob:{SECRET}@h:11434",
    f"http://bob:{SECRET}@h:11434/api/tags",
    f"http://bob:pässwörd-{SECRET}@h:11434",
    f"{SECRET}@h:11434",
    f"bob:{SECRET}@h:11434",
]


@pytest.mark.parametrize("host", LEAK_HOSTS)
def test_the_shown_host_is_built_from_scheme_hostname_and_port_only(host: str) -> None:
    assert SECRET not in public_host(host)
    assert SECRET not in repr(probe_ollama(host, MODEL, transport=FakeGet(tags(MODEL))))


@pytest.mark.parametrize("host", LEAK_HOSTS)
def test_no_secret_reaches_host_or_error_whatever_the_transport_says(host: str) -> None:
    def echoes_url(url: str, timeout: float) -> HttpResponse:
        raise ConnectionError(f"cannot reach {url} (host {host!r})")

    for transport in (
        echoes_url,
        FakeGet(HttpResponse(502, f"bad gateway for {host}".encode())),
        FakeGet(HttpResponse(200, f"not json {host}".encode())),
        FakeGet(tags(MODEL)),
        FakeGet(tags()),
    ):
        result = probe_ollama(host, MODEL, transport=transport)
        assert SECRET not in repr(result), result
        assert SECRET not in result.host


def test_public_host_keeps_only_scheme_hostname_and_port() -> None:
    assert public_host("http://bob#x@h:11434") == "http://bob"
    assert public_host("http://h:11434/a/b?c=d;e#f") == "http://h:11434"
    assert public_host("not a url") == "<invalid OLLAMA_HOST>"
    assert public_host("http://[::1") == "<invalid OLLAMA_HOST>"
    assert public_host("ftp://h:1") == "<invalid OLLAMA_HOST>"


def test_a_password_containing_an_at_sign_is_cut_at_the_last_at_sign() -> None:
    # Pins the property the old ``rpartition("@")`` had: the host is what follows the LAST ``@``.
    assert public_host("http://bob:p@ss@h:11434") == "http://h:11434"
    result = probe_ollama(
        "http://bob:p@ss@h:11434", MODEL, transport=FakeGet(HttpResponse(500, b"p@ss bob"))
    )
    assert result.host == "http://h:11434"
    assert result.error is not None and "p@ss" not in result.error and "ss" not in result.error.replace(
        "HTTP 500", ""
    )


# -- the status that counts --------------------------------------------------------


@pytest.mark.parametrize("status", [201, 202, 204, 299])
def test_only_200_is_a_usable_answer_other_2xx_are_unreachable(status: int) -> None:
    result = probe_ollama(HOST, MODEL, transport=FakeGet(HttpResponse(status, tags(MODEL).body)))

    assert result.outcome == "unreachable"
    assert result.error is not None and result.error.startswith(f"HTTP {status}")


# -- total deadline ------------------------------------------------------------------


def test_the_total_deadline_is_a_named_constant_alongside_the_per_operation_timeout() -> None:
    assert PROBE_DEADLINE_SEC >= PROBE_TIMEOUT_SEC


def test_a_transport_that_drips_forever_is_cut_off_at_the_total_deadline() -> None:
    # A slow-loris server never makes a single socket operation time out, so the
    # per-operation timeout alone never fires. The transport here blocks until
    # released; the probe must give up on its own at the deadline.
    release = threading.Event()
    started = threading.Event()

    def drips(url: str, timeout: float) -> HttpResponse:
        started.set()
        release.wait(30)
        return tags(MODEL)

    t0 = time.monotonic()
    try:
        result = probe_ollama(HOST, MODEL, transport=drips, deadline_sec=0.05)
    finally:
        release.set()

    assert started.is_set()
    assert time.monotonic() - t0 < 5  # bounded by the deadline, not by the 30 s the fake would wait
    assert result.outcome == "unreachable"
    assert result.error is not None and "0.05" in result.error and "deadline" in result.error


def test_a_transport_that_answers_within_the_deadline_is_unaffected() -> None:
    assert probe_ollama(HOST, MODEL, transport=FakeGet(tags(MODEL)), deadline_sec=5).outcome == "ready"


def test_the_real_transport_stops_reading_a_dripping_body_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]

    class Drip:
        status = 200

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read1(self, n: int = -1) -> bytes:
            now[0] += 2.0  # each chunk takes 2 s, under the per-operation timeout
            return b"x"

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: Drip())
    monkeypatch.setattr("adapters.summarize.ollama.time.monotonic", lambda: now[0])

    with pytest.raises(TimeoutError):
        urllib_get_transport("http://h:11434/api/tags", 5.0)

    assert now[0] <= 5.0 + 2.0  # stopped after the first chunk past the deadline, not at MAX bytes


# -- the real transport, with urlopen replaced (no socket) -------------------------


class FakeUrlopenResponse(io.BytesIO):
    status = 200

    def __enter__(self) -> Self:
        return self


def test_urllib_get_transport_issues_a_get_with_the_timeout_and_a_bounded_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeUrlopenResponse:
        seen.update(url=request.full_url, method=request.get_method(), data=request.data, timeout=timeout)
        return FakeUrlopenResponse(b"x" * (MAX_RESPONSE_BYTES + 10))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    response = urllib_get_transport("http://h:11434/api/tags", 5.0)

    assert seen == {
        "url": "http://h:11434/api/tags",
        "method": "GET",
        "data": None,
        "timeout": 5.0,
    }
    assert response.status == 200
    assert len(response.body) == MAX_RESPONSE_BYTES + 1  # one past the limit, never the rest


def test_urllib_get_transport_returns_http_errors_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> Any:
        raise urllib.error.HTTPError(
            request.full_url, 503, "unavailable", {}, io.BytesIO(b"busy")  # type: ignore[arg-type]
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert urllib_get_transport("http://h/api/tags", 5.0) == HttpResponse(503, b"busy")
