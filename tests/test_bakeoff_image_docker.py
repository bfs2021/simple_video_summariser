"""Builds the bake-off image and checks its runtime (issue #76, follow-up to #3).

Marked `integration` (needs Docker Desktop) and `image` (a build downloads
packages from PyPI and Debian and takes minutes): deselected unless run with
`pytest -m image`. Static checks of the same files live in
`test_bakeoff_image.py`. Nothing here contacts YouTube: the audio cache is
seeded with a clip made by the image's own ffmpeg under the harness's cache
key. Every image, container, volume and temporary context created here is
removed afterwards, even on failure.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.test_backend_image import REPO_ROOT

pytestmark = [pytest.mark.integration, pytest.mark.image]

RUN_ID = uuid.uuid4().hex[:8]
IMAGE = f"ytdigest-bakeoff-test:{RUN_ID}"
# A syntactically valid watch URL; with --duration 5 and the default --start
# 3:00 the harness's cache key is <video id>_<start s, 6 digits>_<duration, 5>.
URL = "https://www.youtube.com/watch?v=aaaaaaaaaaa"
CLIP_KEY = "aaaaaaaaaaa_000180_00005"
SEED_UID = os.getuid()


def _docker(
    *args: str, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _run(
    *command: str, extra: tuple[str, ...] = (), timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    return _docker("run", "--rm", *extra, IMAGE, *command, timeout=timeout)


def _sh(script: str) -> subprocess.CompletedProcess[str]:
    return _docker("run", "--rm", "--entrypoint", "sh", IMAGE, "-c", script)


def _ok(script: str) -> str:
    result = _sh(script)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture(scope="module", autouse=True)
def image() -> Iterator[str]:
    result = _docker(
        "build", "-f", "ops/Dockerfile.bakeoff", "-t", IMAGE, ".", timeout=1800
    )
    try:
        assert result.returncode == 0, result.stderr[-4000:]
        yield IMAGE
    finally:
        _docker("image", "rm", "-f", IMAGE)


@pytest.fixture
def volumes() -> Iterator[list[str]]:
    created: list[str] = []
    yield created
    for name in created:
        _docker("volume", "rm", "-f", name)


def _seeded_volume(volumes: list[str], uid: int) -> str:
    """A fresh cache volume holding a 5 s clip under the harness's cache key."""
    name = f"ytdigest-bakeoff-test-{uuid.uuid4().hex[:8]}"
    volumes.append(name)
    result = _docker(
        "run",
        "--rm",
        "--user",
        f"{uid}:{uid}",
        "-v",
        f"{name}:/cache",
        "--entrypoint",
        "ffmpeg",
        IMAGE,
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=5",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libopus",
        "-b:a",
        "16k",
        f"/cache/bakeoff/{CLIP_KEY}.opus",
    )
    assert result.returncode == 0, result.stderr
    return name


# ---- runtime contents ----


def test_default_user_is_non_root_app_10001() -> None:
    assert _ok("id -u") == "10001"
    assert _ok("id -un") == "app"


def test_python_3_14_and_the_pinned_yt_dlp() -> None:
    assert _ok("python --version").startswith("Python 3.14.")
    assert _ok("which python") == "/opt/venv/bin/python"
    assert _ok("python -m yt_dlp --version") == "2026.08.19"


@pytest.mark.parametrize("tool", ["ffmpeg", "ffprobe"])
def test_ffmpeg_tools_run(tool: str) -> None:
    assert _sh(f"{tool} -version").returncode == 0


def test_harness_interpreter_has_only_yt_dlp() -> None:
    frozen = _ok("pip freeze").splitlines()
    assert frozen == ["yt-dlp==2026.8.19"]
    result = _sh("/opt/venv/bin/python -c 'import faster_whisper'")
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr


def test_engine_venv_imports_all_engines_and_pip_check_passes() -> None:
    code = "import faster_whisper, ctranslate2, sherpa_onnx, numpy"
    assert _sh(f"/opt/bakeoff-venv/bin/python -c '{code}'").returncode == 0
    assert _sh("/opt/bakeoff-venv/bin/pip check").returncode == 0


def test_engine_venv_is_not_writable_by_the_default_user() -> None:
    assert _sh("touch /opt/bakeoff-venv/x").returncode != 0


def test_image_has_no_build_toolchain_and_no_apt_lists() -> None:
    assert _ok("command -v gcc cc make || true") == ""
    assert _ok("ls /var/lib/apt/lists | grep -v '^partial$' | wc -l") == "0"


def test_image_holds_only_bakeoff_py_from_the_repo() -> None:
    assert _ok("ls -A /app") == "bakeoff.py"
    leftovers = _ok(
        "ls -d /common /adapters /services /tests /_docs /app/.env* "
        "/requirements*.txt /pyproject.toml /app/subtitle_coverage.py 2>/dev/null || true"
    )
    assert leftovers == ""


def test_no_arguments_prints_usage_and_exits_2() -> None:
    result = _run()
    assert result.returncode == 2
    assert "usage: bakeoff.py" in result.stderr


def test_help_is_the_harnesss_own() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "Transcription engine bake-off" in result.stdout


def test_the_harness_is_pid_1_in_exec_form() -> None:
    inspected = _docker(
        "image", "inspect", "-f", "{{json .Config.Entrypoint}}|{{.Config.Healthcheck}}", IMAGE
    )
    assert inspected.stdout.strip() == '["python","/app/bakeoff.py"]|<nil>'


def test_hf_home_and_default_paths_point_into_the_image() -> None:
    env = dict(
        line.split("=", 1) for line in _ok("env").splitlines() if "=" in line
    )
    assert env["HF_HOME"].startswith("/cache/")
    assert env["BAKEOFF_ENGINES_VENV"] == "/opt/bakeoff-venv"
    assert env["BAKEOFF_CACHE_DIR"].startswith("/cache/")
    assert env["BAKEOFF_OUT_DIR"].startswith("/work/")


def test_cache_dirs_are_writable_by_any_uid(volumes: list[str]) -> None:
    name = f"ytdigest-bakeoff-test-{uuid.uuid4().hex[:8]}"
    volumes.append(name)
    result = _docker(
        "run", "--rm", "--user", "4321:4321", "-v", f"{name}:/cache",
        "--entrypoint", "sh", IMAGE, "-c",
        "touch /cache/bakeoff/a /cache/hf/b /cache/xdg/c",
    )
    assert result.returncode == 0, result.stderr


# ---- full runs on a seeded clip: no YouTube ----


def _bind(tmp_path: Path) -> tuple[Path, tuple[str, ...]]:
    work = tmp_path / "work"
    work.mkdir()
    return work, ("--user", f"{SEED_UID}:{SEED_UID}", "-v", f"{work}:/work")


def test_offline_run_with_an_empty_model_cache_reports_failed_engines(
    tmp_path: Path, volumes: list[str]
) -> None:
    cache = _seeded_volume(volumes, SEED_UID)
    work, user_and_work = _bind(tmp_path)
    result = _run(
        URL, "--duration", "5",
        "--engines", "whisper-large-v3-turbo,parakeet-tdt-0.6b-v3",
        extra=("--network", "none", "-v", f"{cache}:/cache", *user_and_work),
        timeout=600,
    )
    assert result.returncode == 1, result.stderr
    report = (work / "runs" / CLIP_KEY / "report.txt").read_text()
    assert "audio cache hit" in result.stderr
    assert re.search(r"## whisper-large-v3-turbo\nFAILED: \S", report)
    assert re.search(r"## parakeet-tdt-0\.6b-v3\nFAILED: \S", report)
    assert "container: yes" in report


def test_online_tiny_run_succeeds_and_output_belongs_to_the_invoking_uid(
    tmp_path: Path, volumes: list[str]
) -> None:
    cache = _seeded_volume(volumes, SEED_UID)
    work, user_and_work = _bind(tmp_path)
    result = _run(
        URL, "--duration", "5",
        "--engines", "whisper-large-v3-turbo", "--whisper-model-size", "tiny",
        extra=("-v", f"{cache}:/cache", *user_and_work),
        timeout=900,
    )
    assert result.returncode == 0, result.stderr
    run_dir = work / "runs" / CLIP_KEY
    report = (run_dir / "report.txt").read_text()
    assert re.search(r"^container: yes \(cgroup v2 cpu\.max: .+, memory\.max: .+\)$", report, re.MULTILINE)
    assert re.search(r"^RTF: \d", report, re.MULTILINE)
    assert re.search(r"^peak RSS: \d", report, re.MULTILINE)
    # Packaging only: a run never creates an engine venv or runs pip.
    assert "creating throwaway engine venv" not in result.stderr
    assert "pip" not in (result.stdout + result.stderr).lower().replace("pipe", "")
    paths = [work / "runs", run_dir, *run_dir.iterdir()]
    assert {p.stat().st_uid for p in paths} == {SEED_UID}
    # HF models landed on the volume, not in the container's layer.
    listing = _docker(
        "run", "--rm", "-v", f"{cache}:/cache", "--entrypoint", "ls", IMAGE, "/cache/hf/hub"
    )
    assert "faster-whisper-tiny" in listing.stdout


# ---- fail loudly: a bad engine install fails the build ----

BAD_SPECS = {
    "nonexistent-version": ("sherpa-onnx==1.12.40", "sherpa-onnx==0.0.0.dev0"),
    # Published only as a source distribution: needs a compiler, which the
    # image does not have and must not gain.
    "sdist-only": ("sherpa-onnx==1.12.40", "fasttext==0.9.3"),
}


@pytest.mark.parametrize("case", sorted(BAD_SPECS))
def test_a_bad_engine_pin_fails_the_build_with_pips_error(
    case: str, tmp_path: Path
) -> None:
    good, bad = BAD_SPECS[case]
    context = tmp_path / "context"
    (context / "ops").mkdir(parents=True)
    (context / "scripts" / "phase0").mkdir(parents=True)
    for name in ("Dockerfile.bakeoff", "Dockerfile.bakeoff.dockerignore"):
        shutil.copy(REPO_ROOT / "ops" / name, context / "ops")
    script = (REPO_ROOT / "scripts" / "phase0" / "bakeoff.py").read_text()
    assert good in script
    (context / "scripts" / "phase0" / "bakeoff.py").write_text(
        script.replace(good, bad)
    )
    tag = f"ytdigest-bakeoff-badpin-test:{uuid.uuid4().hex[:8]}"
    try:
        result = _docker(
            "build", "--progress=plain", "-f", "ops/Dockerfile.bakeoff", "-t", tag,
            str(context), timeout=900,
        )
        built = _docker("image", "inspect", tag)
    finally:
        _docker("image", "rm", "-f", tag)
    assert result.returncode != 0
    assert "No matching distribution found" in result.stderr
    assert "FAILED:" not in result.stderr
    assert built.returncode != 0
