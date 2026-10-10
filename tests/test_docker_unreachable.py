"""Clear failure when the Docker daemon is unreachable (issue #77). No Docker."""

from __future__ import annotations

import pytest
from docker.errors import DockerException  # type: ignore[import-untyped]

from tests import conftest

_DEFAULT_SOCKET = "unix:///var/run/docker.sock"


class _Calls:
    def __init__(self, exc: BaseException | None) -> None:
        self.exc = exc
        self.count = 0

    def __call__(self) -> str:
        self.count += 1
        if self.exc is not None:
            raise self.exc
        return "client"


def test_reachable_daemon_returns_the_factory_result() -> None:
    calls = _Calls(None)
    assert conftest._require_docker_daemon(calls, docker_host=None) == "client"
    assert calls.count == 1


@pytest.mark.parametrize(
    ("docker_host", "endpoint"),
    [
        ("tcp://127.0.0.1:1", "tcp://127.0.0.1:1"),
        (None, _DEFAULT_SOCKET),
        ("", _DEFAULT_SOCKET),
    ],
)
def test_unreachable_daemon_fails_with_one_line_actionable_message(
    docker_host: str | None, endpoint: str
) -> None:
    calls = _Calls(
        DockerException("Error while fetching server API version: boom\nsecond line")
    )
    with pytest.raises(pytest.fail.Exception) as info:
        conftest._require_docker_daemon(calls, docker_host=docker_host)
    message = str(info.value)
    assert "\n" not in message
    for needle in (
        "Docker Desktop",
        "WSL integration",
        'pytest -m "not integration"',
        endpoint,
    ):
        assert needle in message
    assert "DockerException" in message
    assert "Error while fetching server API version: boom" in message
    assert "second line" not in message
    assert info.value.pytrace is False
    assert calls.count == 1


@pytest.mark.parametrize("exc", [RuntimeError("pull failed"), ValueError("bad")])
def test_other_errors_propagate_unchanged_after_one_call(exc: Exception) -> None:
    calls = _Calls(exc)
    with pytest.raises(type(exc)) as info:
        conftest._require_docker_daemon(calls, docker_host=None)
    assert info.value is exc
    assert calls.count == 1
