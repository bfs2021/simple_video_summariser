"""Static checks of the bake-off image inputs (issue #76, follow-up to #3).

No Docker is invoked: these read `ops/Dockerfile.bakeoff`,
`ops/Dockerfile.bakeoff.dockerignore`, `compose.bakeoff.yml` and `.gitignore`
as text, so they run under `pytest -m "not integration"`.
`test_bakeoff_image_docker.py` builds the image and checks the runtime.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests.test_backend_image import REPO_ROOT, _copy_sources, _instructions
from tests.test_compose import _service_block, _top_level_section

DOCKERFILE = REPO_ROOT / "ops" / "Dockerfile.bakeoff"
DOCKERFILE_IGNORE = REPO_ROOT / "ops" / "Dockerfile.bakeoff.dockerignore"
COMPOSE_BAKEOFF_YML = REPO_ROOT / "compose.bakeoff.yml"
GITIGNORE = REPO_ROOT / ".gitignore"
REQUIREMENTS_BACKEND = REPO_ROOT / "requirements.backend.txt"
COMPOSE_YML = REPO_ROOT / "compose.yml"
COMPOSE_OVERRIDE_YML = REPO_ROOT / "compose.override.yml"
ROOT_DOCKERIGNORE = REPO_ROOT / ".dockerignore"


def _steps() -> list[tuple[str, str]]:
    return _instructions(DOCKERFILE)


def _code(path: Path) -> str:
    """File text with comment lines dropped."""
    return "\n".join(
        line for line in path.read_text().splitlines() if not line.lstrip().startswith("#")
    )


def _yt_dlp_pin() -> str:
    match = re.search(r"^yt-dlp==(\S+)$", REQUIREMENTS_BACKEND.read_text(), re.MULTILINE)
    assert match, "requirements.backend.txt has no yt-dlp pin"
    return match.group(1)


# ---- Dockerfile.bakeoff ----


def test_every_from_is_python_3_14_slim() -> None:
    froms = [args.split()[0] for name, args in _steps() if name == "FROM"]
    assert froms
    assert all(f == "python:3.14-slim" for f in froms)


def test_does_not_derive_from_the_backend_image() -> None:
    text = DOCKERFILE.read_text()
    assert "ytdigest-backend" not in text


def test_no_engine_version_pin_in_the_dockerfile() -> None:
    # The issue's own check: grep -E 'faster-whisper|sherpa-onnx|numpy'.
    assert not re.search(r"faster-whisper|sherpa-onnx|numpy", DOCKERFILE.read_text())


def test_yt_dlp_pin_matches_requirements_backend() -> None:
    pins = re.findall(r"yt-dlp==(\S+?)['\"]?(?:\s|$)", _code(DOCKERFILE))
    assert pins == [_yt_dlp_pin()]


def test_harness_interpreter_gets_only_yt_dlp() -> None:
    installs = [a for n, a in _steps() if n == "RUN" and "/opt/venv/bin/pip" in a]
    assert len(installs) == 1
    assert "--only-binary :all:" in installs[0]
    assert installs[0].split("install", 1)[1].split()[-1].strip("'\"").startswith("yt-dlp==")
    assert "-r " not in installs[0]


def test_only_apt_package_is_ffmpeg() -> None:
    runs = [a for n, a in _steps() if n == "RUN" and "apt-get install" in a]
    assert runs
    for run in runs:
        packages = run.split("apt-get install", 1)[1].split("&&")[0].split()
        packages = [p for p in packages if not p.startswith("-")]
        assert packages == ["ffmpeg"], packages
        assert "rm -rf /var/lib/apt/lists/*" in run
    assert len(runs) == 1


def test_no_copy_of_application_or_dependency_files() -> None:
    forbidden_dirs = ("common/", "adapters/", "services/", "tests/")
    forbidden_files = {
        "requirements.backend.txt",
        "requirements.whisper.txt",
        "pyproject.toml",
        ".env",
    }
    for name, args in _steps():
        if name in {"COPY", "ADD"}:
            for source in _copy_sources(args):
                assert not source.startswith(forbidden_dirs), source
                assert source not in forbidden_files, source
                assert source not in {".", "./"}, source


def test_only_bakeoff_py_is_copied_from_scripts() -> None:
    scripts_copies = [
        src
        for name, args in _steps()
        if name in {"COPY", "ADD"}
        for src in _copy_sources(args)
        if "scripts" in src or src.endswith("bakeoff.py")
    ]
    assert scripts_copies
    assert all(src.endswith("bakeoff.py") for src in scripts_copies)


def test_last_user_is_non_root_uid_10001() -> None:
    users = [args for name, args in _steps() if name == "USER"]
    assert users
    assert users[-1] not in {"root", "0"}
    text = DOCKERFILE.read_text()
    assert re.search(r"useradd\b[^\n]*-u 10001", text)


def test_no_healthcheck() -> None:
    assert all(name != "HEALTHCHECK" for name, _ in _steps())


def test_entrypoint_and_cmd_are_exec_form() -> None:
    entrypoints = [args for name, args in _steps() if name == "ENTRYPOINT"]
    assert entrypoints, "the harness must be the ENTRYPOINT so --help/usage work bare"
    for args in entrypoints:
        assert args.startswith("["), "shell-form ENTRYPOINT swallows SIGTERM"
    for name, args in _steps():
        if name == "CMD":
            assert args.startswith("[")


def test_engine_install_uses_the_scripts_own_install_mode() -> None:
    text = "\n".join(f"{n} {a}" for n, a in _steps())
    assert re.search(r"RUN \S*python /app/bakeoff\.py --install-engines /opt/bakeoff-venv", text)
    assert "--only-binary" in text  # wheels only; no compiler in the image
    assert "build-essential" not in text and "gcc" not in text


def test_hf_home_points_into_the_cache_volume() -> None:
    env = " ".join(a for n, a in _steps() if n == "ENV")
    assert re.search(r"\bHF_HOME=", env)
    assert "/home" not in re.search(r"HF_HOME=(\S+)", env).group(1)  # type: ignore[union-attr]


def test_env_overridable_default_paths_are_set() -> None:
    env = " ".join(a for n, a in _steps() if n == "ENV")
    assert "BAKEOFF_ENGINES_VENV=" in env
    assert "BAKEOFF_CACHE_DIR=" in env
    assert "BAKEOFF_OUT_DIR=" in env
    assert "BAKEOFF_ENGINES_PREINSTALLED=1" in env


# ---- ops/Dockerfile.bakeoff.dockerignore ----


def test_per_dockerfile_ignore_file_exists_and_narrows_scripts_back_open() -> None:
    assert DOCKERFILE_IGNORE.exists()
    lines = [
        line.strip()
        for line in DOCKERFILE_IGNORE.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines == ["*", "!scripts/phase0/bakeoff.py"]


def test_root_dockerignore_is_unchanged_and_still_excludes_scripts() -> None:
    lines = ROOT_DOCKERIGNORE.read_text().splitlines()
    assert "scripts/" in [line.strip() for line in lines]


# ---- compose.bakeoff.yml ----


def _bakeoff_text() -> str:
    return COMPOSE_BAKEOFF_YML.read_text()


def test_compose_bakeoff_has_its_own_project_name() -> None:
    assert "name: ytdigest-bakeoff" in _bakeoff_text()


def test_compose_bakeoff_has_exactly_one_service() -> None:
    services_text = _top_level_section(_bakeoff_text(), "services")
    headers = [
        line
        for line in services_text.splitlines()
        if line.startswith("  ")
        and not line.startswith("    ")
        and not line.lstrip().startswith("#")
    ]
    assert headers == ["  bakeoff:"]


def test_bakeoff_service_builds_the_right_dockerfile_and_tags_the_image() -> None:
    block = _service_block(_bakeoff_text(), "bakeoff")
    assert "build: {context: ., dockerfile: ops/Dockerfile.bakeoff}" in block
    assert "image: ytdigest-bakeoff:latest" in block


def test_bakeoff_service_has_no_ports() -> None:
    assert "ports:" not in _service_block(_bakeoff_text(), "bakeoff")


def test_bakeoff_service_is_not_on_an_internal_only_network() -> None:
    text = _bakeoff_text()
    assert "internal: true" not in text


def test_bakeoff_service_has_the_named_cache_volume() -> None:
    block = _service_block(_bakeoff_text(), "bakeoff")
    assert re.search(r"^\s+- bakeoff-cache:/cache$", block, re.MULTILINE)
    volumes_section = _top_level_section(_bakeoff_text(), "volumes")
    assert "bakeoff-cache" in volumes_section


def test_bakeoff_service_binds_the_data_directory_to_work() -> None:
    block = _service_block(_bakeoff_text(), "bakeoff")
    assert "./data/bakeoff:/work" in block


def test_no_deploy_resource_limits() -> None:
    code = _code(COMPOSE_BAKEOFF_YML)
    assert "deploy:" not in code
    assert "resources:" not in code
    assert "mem_limit" not in code and "cpus:" not in code


def test_header_comment_explains_report_reflects_added_limits() -> None:
    text = _bakeoff_text()
    header = "\n".join(
        line for line in text.splitlines() if line.lstrip().startswith("#")
    )
    assert "report header" in header.lower()
    assert "limit" in header.lower()


def test_compose_bakeoff_runs_non_root_with_configurable_uid_gid() -> None:
    block = _service_block(_bakeoff_text(), "bakeoff")
    assert "user:" in block
    assert re.search(r'user:\s*"?\$\{BAKEOFF_UID', block)


# ---- .gitignore ----


def test_gitignore_excludes_bakeoff_data_directory() -> None:
    lines = [line.strip() for line in GITIGNORE.read_text().splitlines()]
    assert "data/bakeoff/" in lines


# ---- the app stack does not know about the bake-off ----


@pytest.mark.parametrize("path", [COMPOSE_YML, COMPOSE_OVERRIDE_YML])
def test_app_compose_files_do_not_reference_the_bakeoff(path: Path) -> None:
    # compose.bakeoff.yml is used only with an explicit -f (issue #76).
    assert "bakeoff" not in path.read_text().lower()


# ---- scripts/phase0/bakeoff.py: the three permitted changes ----


@pytest.fixture
def bakeoff(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    for name in (
        "BAKEOFF_ENGINES_VENV",
        "BAKEOFF_CACHE_DIR",
        "BAKEOFF_OUT_DIR",
        "BAKEOFF_ENGINES_PREINSTALLED",
    ):
        monkeypatch.delenv(name, raising=False)
    path = REPO_ROOT / "scripts" / "phase0" / "bakeoff.py"
    spec = importlib.util.spec_from_file_location("bakeoff_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "bakeoff_under_test", module)
    spec.loader.exec_module(module)
    return module


def test_host_defaults_are_unchanged_without_the_environment(
    bakeoff: ModuleType,
) -> None:
    args = bakeoff.build_parser().parse_args(["https://www.youtube.com/watch?v=aaaaaaaaaaa"])
    assert args.engines_venv == Path.home() / ".cache" / "bakeoff-venv"
    assert args.cache_dir == Path.home() / ".cache" / "bakeoff-cache"
    assert args.out_dir is None


def test_environment_overrides_the_default_paths(
    bakeoff: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BAKEOFF_ENGINES_VENV", "/opt/v")
    monkeypatch.setenv("BAKEOFF_CACHE_DIR", "/c")
    args = bakeoff.build_parser().parse_args(
        ["https://www.youtube.com/watch?v=aaaaaaaaaaa", "--cache-dir", "/explicit"]
    )
    assert args.engines_venv == Path("/opt/v")
    assert args.cache_dir == Path("/explicit")  # a flag still wins


def test_header_says_container_no_on_the_host(
    bakeoff: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bakeoff, "in_container", lambda: False)
    assert bakeoff.container_line() == "container: no"


def test_header_gives_the_cgroup_limits_in_a_container(
    bakeoff: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cpu = tmp_path / "cpu.max"
    cpu.write_text("200000 100000\n")
    memory = tmp_path / "memory.max"
    memory.write_text("max\n")
    monkeypatch.setattr(bakeoff, "in_container", lambda: True)
    monkeypatch.setattr(bakeoff, "CGROUP_CPU_MAX", cpu)
    monkeypatch.setattr(bakeoff, "CGROUP_MEMORY_MAX", memory)
    assert bakeoff.container_line() == (
        "container: yes (cgroup v2 cpu.max: 200000 100000, memory.max: max)"
    )
    monkeypatch.setattr(bakeoff, "CGROUP_CPU_MAX", tmp_path / "missing")
    assert "cpu.max: unreadable" in bakeoff.container_line()
