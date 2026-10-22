"""Builds the backend image and checks its runtime (issue #55).

Marked `integration` (needs Docker Desktop) and `image` (a build downloads
packages from PyPI and Debian and takes minutes): deselected unless run with
`pytest -m image`. Static checks of the same files live in
`test_backend_image.py`. Every container, volume and compose project created
here is removed afterwards, even on failure.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.image]

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE = "ytdigest-backend:latest"
PASSWORD = "image-test-password"


def _docker(
    *args: str, timeout: int = 120, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def _run(
    *command: str, extra: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    return _docker("run", "--rm", *extra, IMAGE, *command)


def _ok(*command: str) -> str:
    result = _run(*command)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture(scope="module", autouse=True)
def image() -> str:
    result = _docker(
        "build", "-f", "ops/Dockerfile.backend", "-t", IMAGE, ".", timeout=1800
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return IMAGE


def test_runs_as_app_uid_10001() -> None:
    assert _ok("id", "-u") == "10001"
    assert _ok("id", "-un") == "app"


def test_python_is_the_venv_python_3_14() -> None:
    assert _ok("python", "--version").startswith("Python 3.14.")
    assert _ok("which", "python") == "/opt/venv/bin/python"
    assert _ok("printenv", "PATH").startswith("/opt/venv/bin")
    assert _ok("printenv", "PYTHONUNBUFFERED") == "1"
    assert _ok("printenv", "PYTHONPATH") == "/app"
    assert _ok("pwd") == "/app"


def test_app_directory_holds_only_the_application() -> None:
    entries = set(_ok("ls", "-A", "/app").split())
    assert entries == {"common", "adapters", "services", "migrations", "alembic.ini"}
    prompts = _ok("sh", "-c", "ls /app/adapters/summarize/prompts/v1/*.txt | wc -l")
    assert prompts == "7"


@pytest.mark.parametrize(
    "command",
    [
        ("ffmpeg", "-version"),
        ("ffprobe", "-version"),
        ("curl", "--version"),
        ("yt-dlp", "--version"),
        ("alembic", "--version"),
    ],
)
def test_runtime_tools_are_installed(command: tuple[str, ...]) -> None:
    assert _run(*command).returncode == 0


def test_yt_dlp_is_the_pinned_version() -> None:
    # requirements.backend.txt pins yt-dlp==2026.10.21; yt-dlp itself prints
    # the zero-padded spelling of the same version.
    assert _ok("yt-dlp", "--version") == "2026.10.21"


def test_runtime_stage_has_no_build_toolchain() -> None:
    assert _ok("sh", "-c", "command -v gcc cc make || true") == ""
    assert _run("dpkg", "-s", "build-essential").returncode != 0
    assert _ok("sh", "-c", "ls /var/lib/apt/lists | grep -v '^partial$' | wc -l") == "0"


@pytest.mark.parametrize(
    "module", ["faster_whisper", "pytest", "mypy", "testcontainers"]
)
def test_whisper_and_dev_packages_are_absent(module: str) -> None:
    result = _run("python", "-c", f"import {module}")
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr


def test_service_modules_import_without_database_or_network() -> None:
    result = _run(
        "python",
        "-c",
        "import services.planner.main, services.analyzer.main, services.transcriber.main",
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", ["services.planner.main", "services.cli"])
def test_service_help_exits_zero(module: str) -> None:
    assert _run("python", "-m", module, "--help").returncode == 0


def test_planner_without_database_url_fails_naming_it() -> None:
    result = _run("python", "-m", "services.planner.main", "--once")
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "DATABASE_URL" in output
    assert "Traceback" not in output


def test_data_audio_is_owned_by_app_and_writable_on_a_fresh_volume() -> None:
    assert _ok("stat", "-c", "%U:%G", "/data/audio") == "app:app"
    volume = f"ytdigest-test-audio-{uuid.uuid4().hex[:8]}"
    try:
        result = _run("touch", "/data/audio/x", extra=("-v", f"{volume}:/data/audio"))
        assert result.returncode == 0, result.stderr
    finally:
        _docker("volume", "rm", "-f", volume)


def test_tmp_is_writable_by_app() -> None:
    assert _run("touch", "/tmp/heartbeat").returncode == 0


def test_image_has_no_entrypoint_or_healthcheck_and_command_is_pid_1() -> None:
    config = json.loads(_docker("image", "inspect", IMAGE).stdout)[0]["Config"]
    assert not config.get("Entrypoint")
    assert not config.get("Healthcheck")
    assert config["User"] == "app" or config["User"] == "10001"
    assert _ok("python", "-c", "import os; print(os.getpid())") == "1"


# ---- compose `migrate` service against the compose Postgres ----


Compose = Callable[..., subprocess.CompletedProcess[str]]


@pytest.fixture
def compose(monkeypatch: pytest.MonkeyPatch) -> Iterator[Compose]:
    """`docker compose` bound to a throwaway project; torn down with its volumes."""
    project = f"ytdigest-imgtest-{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("POSTGRES_PASSWORD", PASSWORD)

    def run(*args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
        return _docker(
            "compose", "-p", project, "-f", "compose.yml", *args, timeout=timeout
        )

    try:
        yield run
    finally:
        run("down", "-v", "--remove-orphans")


def test_migrate_runs_twice_with_json_output_and_reaches_head(compose: Compose) -> None:
    up = compose("up", "-d", "--wait", "db")
    assert up.returncode == 0, up.stderr

    revisions = (REPO_ROOT / "migrations" / "versions").glob("0*.py")
    latest = max(p.stem.split("_")[0] for p in revisions)

    for _ in range(2):
        result = compose("run", "--rm", "migrate")
        assert result.returncode == 0, result.stderr
        for line in result.stdout.splitlines():
            if line.strip():
                assert isinstance(json.loads(line), dict), line

    current = compose("run", "--rm", "migrate", "alembic", "current")
    assert current.returncode == 0, current.stderr
    assert latest in current.stdout


def test_migrate_fails_in_bounded_time_when_db_is_down(compose: Compose) -> None:
    result = compose("run", "--rm", "--no-deps", "migrate", timeout=120)
    assert result.returncode != 0
