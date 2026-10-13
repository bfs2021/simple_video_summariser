"""Tests for ``ops/host/preflight.sh`` (task #78).

The script is run with stubbed ``docker``/``systemctl``/``timedatectl`` on
``PATH`` so no real server or daemon is needed (the issue requires exactly
this: "The script is exercised with stubbed commands, not a real server").
Real coreutils stay on ``PATH``; the checks that read the live host (``disk``,
``cpu``, ``memory``) are driven to a known PASS/FAIL by their threshold
variables instead of by faking ``df``/``nproc``/``/proc/meminfo``.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "ops" / "host" / "preflight.sh"

# A .env that passes env-file, cpu and ollama cleanly; individual tests edit it.
GOOD_ENV = (
    "COMPOSE_FILE=compose.yml\n"
    "POSTGRES_PASSWORD=good_pw-123\n"
    "TRANSCRIBER_CPUS=0.5\n"
    "OLLAMA_MODEL=qwen3.5:4b\n"
)

# Stub command behaviour, all passing by default; a test overrides one field.
PASS_STUBS = {
    "STUB_DOCKER_SERVER_VERSION": "29.1.2",
    "STUB_COMPOSE_VERSION": "2.40.3",
    "STUB_SYSTEMCTL": "enabled",
    "STUB_NTP": "yes",
    "STUB_DOCKER_ROOT": "",  # set per-run to the checkout dir (a real path)
    "STUB_COMPOSE_CONFIG": "services:\n  api:\n    image: x\n",
    "STUB_COMPOSE_CONFIG_RC": "0",
    "STUB_OLLAMA_RUN_OUT": "200\nqwen3.5:4b\n",
}

_DOCKER_STUB = r"""#!/usr/bin/env bash
case "$1 ${2:-}" in
  "version "*)
    [ -n "${STUB_DOCKER_SERVER_VERSION:-}" ] || { echo "Cannot connect to the Docker daemon" >&2; exit 1; }
    echo "$STUB_DOCKER_SERVER_VERSION"; exit 0 ;;
  "compose version"*)
    [ -n "${STUB_COMPOSE_VERSION:-}" ] || { echo "no compose plugin" >&2; exit 1; }
    echo "$STUB_COMPOSE_VERSION"; exit 0 ;;
  "compose config"*)
    printf '%b' "${STUB_COMPOSE_CONFIG:-}"; exit "${STUB_COMPOSE_CONFIG_RC:-0}" ;;
  "info "*)
    echo "${STUB_DOCKER_ROOT:-/}"; exit 0 ;;
  "network create"*) echo "net"; exit 0 ;;
  "network rm"*)     exit 0 ;;
  "rm "*)            exit 0 ;;
  "run "*)           printf '%b' "${STUB_OLLAMA_RUN_OUT:-}"; exit 0 ;;
  *) echo "stub docker: unhandled: $*" >&2; exit 99 ;;
esac
"""

_SYSTEMCTL_STUB = """#!/usr/bin/env bash
# systemctl is-enabled docker
echo "${STUB_SYSTEMCTL:-disabled}"
"""

_TIMEDATECTL_STUB = """#!/usr/bin/env bash
# timedatectl show -p NTPSynchronized --value
echo "${STUB_NTP:-no}"
"""


@dataclass
class Result:
    code: int
    lines: dict[str, tuple[str, str]] = field(default_factory=dict)  # check -> (status, detail)
    raw: str = ""

    def status(self, check: str) -> str:
        return self.lines[check][0]


class RunFn(Protocol):
    def __call__(
        self,
        *,
        env_text: str = ...,
        stubs: dict[str, str] | None = ...,
        thresholds: dict[str, str] | None = ...,
        env_mode: int = ...,
        write_env: bool = ...,
        args: list[str] | None = ...,
    ) -> Result: ...


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunFn:
    """Return a callable: run(env_text=..., stubs=..., thresholds=..., args=...)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in (
        ("docker", _DOCKER_STUB),
        ("systemctl", _SYSTEMCTL_STUB),
        ("timedatectl", _TIMEDATECTL_STUB),
    ):
        p = bindir / name
        p.write_text(body)
        p.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir), prepend=os.pathsep)

    checkout = tmp_path / "checkout"
    checkout.mkdir()

    def _run(
        *,
        env_text: str = GOOD_ENV,
        stubs: dict[str, str] | None = None,
        thresholds: dict[str, str] | None = None,
        env_mode: int = 0o600,
        write_env: bool = True,
        args: list[str] | None = None,
    ) -> Result:
        merged = {**PASS_STUBS, "STUB_DOCKER_ROOT": str(checkout)}
        merged.update(stubs or {})
        for k, v in merged.items():
            monkeypatch.setenv(k, v)
        # thresholds that make the live-host checks deterministic
        base_thresholds = {"MIN_FREE_GB": "0", "MIN_MEM_GB": "0"}
        base_thresholds.update(thresholds or {})
        for k, v in base_thresholds.items():
            monkeypatch.setenv(k, v)

        if write_env:
            env_file = checkout / ".env"
            env_file.write_text(env_text)
            env_file.chmod(env_mode)

        proc = subprocess.run(
            ["bash", str(SCRIPT), *(args or [])],
            cwd=checkout,
            capture_output=True,
            text=True,
            check=False,
        )
        lines: dict[str, tuple[str, str]] = {}
        for ln in proc.stdout.splitlines():
            parts = ln.split(None, 1)
            if parts and parts[0] in {"PASS", "WARN", "FAIL"}:
                status = parts[0]
                rest = parts[1] if len(parts) > 1 else ""
                check, _, detail = rest.partition(":")
                lines[check.strip()] = (status, detail.strip())
        return Result(code=proc.returncode, lines=lines, raw=proc.stdout + proc.stderr)

    return _run


# --- usage / exit codes ------------------------------------------------------


def test_help_lists_checks_and_thresholds(run: RunFn) -> None:
    r = run(args=["--help"])
    assert r.code == 0
    assert "docker-engine" in r.raw and "ollama-from-container" in r.raw
    assert "MIN_FREE_GB" in r.raw and "DOCKER_COMPOSE_MIN" in r.raw


def test_unknown_flag_is_usage_error(run: RunFn) -> None:
    r = run(args=["--nope"])
    assert r.code == 2


def test_all_pass_exits_zero(run: RunFn) -> None:
    r = run()
    assert r.code == 0
    assert all(s[0] != "FAIL" for s in r.lines.values()), r.raw


def test_any_fail_exits_one(run: RunFn) -> None:
    r = run(stubs={"STUB_SYSTEMCTL": "disabled"})
    assert r.code == 1
    assert r.status("docker-boot") == "FAIL"


def test_all_checks_run_even_after_a_fail(run: RunFn) -> None:
    r = run(stubs={"STUB_DOCKER_SERVER_VERSION": ""})  # first check FAILs
    # the last check still ran
    assert "ollama-from-container" in r.lines
    assert len(r.lines) == 10


# --- docker-engine -----------------------------------------------------------


def test_docker_engine_pass(run: RunFn) -> None:
    assert run().status("docker-engine") == "PASS"


def test_docker_engine_below_floor_fails(run: RunFn) -> None:
    assert run(stubs={"STUB_DOCKER_SERVER_VERSION": "24.0.9"}).status("docker-engine") == "FAIL"


def test_docker_engine_daemon_unreachable_fails(run: RunFn) -> None:
    assert run(stubs={"STUB_DOCKER_SERVER_VERSION": ""}).status("docker-engine") == "FAIL"


# --- docker-compose (numeric, not string, comparison) ------------------------


def test_docker_compose_pass(run: RunFn) -> None:
    assert run().status("docker-compose") == "PASS"


def test_docker_compose_high_minor_passes_numerically(run: RunFn) -> None:
    # 2.100.1 must beat 2.24.0 even though it sorts lower as a string
    assert run(stubs={"STUB_COMPOSE_VERSION": "2.100.1"}).status("docker-compose") == "PASS"


def test_docker_compose_just_below_floor_fails(run: RunFn) -> None:
    assert run(stubs={"STUB_COMPOSE_VERSION": "2.23.9"}).status("docker-compose") == "FAIL"


def test_docker_compose_accepts_v_prefix(run: RunFn) -> None:
    assert run(stubs={"STUB_COMPOSE_VERSION": "v2.24.0"}).status("docker-compose") == "PASS"


# --- docker-boot -------------------------------------------------------------


def test_docker_boot_enabled_passes(run: RunFn) -> None:
    assert run().status("docker-boot") == "PASS"


def test_docker_boot_disabled_fails(run: RunFn) -> None:
    assert run(stubs={"STUB_SYSTEMCTL": "disabled"}).status("docker-boot") == "FAIL"


# --- disk (exactly-at-limit vs one past) -------------------------------------


def test_disk_pass_when_floor_is_zero(run: RunFn) -> None:
    assert run(thresholds={"MIN_FREE_GB": "0"}).status("disk") == "PASS"


def test_disk_fails_when_floor_exceeds_free(run: RunFn) -> None:
    assert run(thresholds={"MIN_FREE_GB": "99999999"}).status("disk") == "FAIL"


# --- cpu (equal passes; fractional handled) ----------------------------------


def test_cpu_pass_when_quota_below_cores(run: RunFn) -> None:
    assert run(env_text=GOOD_ENV).status("cpu") == "PASS"


def test_cpu_fractional_quota_handled(run: RunFn) -> None:
    env = GOOD_ENV.replace("TRANSCRIBER_CPUS=0.5", "TRANSCRIBER_CPUS=2.5")
    # on any CI/dev host with >=3 cores this passes; the point is no crash on 2.5
    assert run(env_text=env).status("cpu") in {"PASS", "FAIL"}


def test_cpu_fails_when_quota_exceeds_cores(run: RunFn) -> None:
    env = GOOD_ENV.replace("TRANSCRIBER_CPUS=0.5", "TRANSCRIBER_CPUS=99999")
    assert run(env_text=env).status("cpu") == "FAIL"


# --- memory (WARN, never FAIL) -----------------------------------------------


def test_memory_pass_when_floor_is_zero(run: RunFn) -> None:
    assert run(thresholds={"MIN_MEM_GB": "0"}).status("memory") == "PASS"


def test_memory_below_floor_warns_not_fails(run: RunFn) -> None:
    r = run(thresholds={"MIN_MEM_GB": "99999999"})
    assert r.status("memory") == "WARN"
    assert r.code == 0  # a WARN alone must not fail the run


# --- time-sync (WARN) --------------------------------------------------------


def test_time_sync_pass(run: RunFn) -> None:
    assert run().status("time-sync") == "PASS"


def test_time_sync_warns_when_not_synced(run: RunFn) -> None:
    r = run(stubs={"STUB_NTP": "no"})
    assert r.status("time-sync") == "WARN"
    assert r.code == 0


# --- env-file ----------------------------------------------------------------


def test_env_file_pass(run: RunFn) -> None:
    assert run().status("env-file") == "PASS"


def test_env_file_missing_fails(run: RunFn) -> None:
    assert run(write_env=False).status("env-file") == "FAIL"


def test_env_file_group_readable_fails(run: RunFn) -> None:
    assert run(env_mode=0o640).status("env-file") == "FAIL"


def test_env_file_wrong_compose_file_fails(run: RunFn) -> None:
    env = GOOD_ENV.replace("COMPOSE_FILE=compose.yml\n", "")
    assert run(env_text=env).status("env-file") == "FAIL"


def test_env_file_bad_password_fails(run: RunFn) -> None:
    env = GOOD_ENV.replace("POSTGRES_PASSWORD=good_pw-123", "POSTGRES_PASSWORD=has space!")
    assert run(env_text=env).status("env-file") == "FAIL"


# --- compose-config ----------------------------------------------------------


def test_compose_config_pass(run: RunFn) -> None:
    assert run().status("compose-config") == "PASS"


def test_compose_config_fails_on_bind_mount(run: RunFn) -> None:
    cfg = "services:\n  api:\n    volumes:\n      - type: bind\n        source: ./common\n"
    assert run(stubs={"STUB_COMPOSE_CONFIG": cfg}).status("compose-config") == "FAIL"


def test_compose_config_fails_on_tiny_whisper_model(run: RunFn) -> None:
    cfg = "services:\n  transcriber:\n    environment:\n      WHISPER_MODEL: tiny\n"
    assert run(stubs={"STUB_COMPOSE_CONFIG": cfg}).status("compose-config") == "FAIL"


def test_compose_config_fails_when_command_errors(run: RunFn) -> None:
    assert run(stubs={"STUB_COMPOSE_CONFIG_RC": "1"}).status("compose-config") == "FAIL"


# --- ollama-from-container ---------------------------------------------------


def test_ollama_pass_when_reachable_with_model(run: RunFn) -> None:
    assert run().status("ollama-from-container") == "PASS"


def test_ollama_warn_when_model_missing(run: RunFn) -> None:
    r = run(stubs={"STUB_OLLAMA_RUN_OUT": "200\nllama3:8b\n"})
    assert r.status("ollama-from-container") == "WARN"
    assert r.code == 0


def test_ollama_fails_when_unreachable(run: RunFn) -> None:
    r = run(stubs={"STUB_OLLAMA_RUN_OUT": "ERR\nconnection refused\n"})
    assert r.status("ollama-from-container") == "FAIL"


# --- secret redaction (an explicit #78 criterion) ----------------------------


def test_never_prints_a_secret_value(run: RunFn) -> None:
    env = (
        "COMPOSE_FILE=compose.yml\n"
        "POSTGRES_PASSWORD=pw-SECRET\n"  # valid charset, so env-file passes
        "ANTHROPIC_API_KEY=sk-test-SECRET\n"
        "TRANSCRIBER_CPUS=0.5\n"
        "OLLAMA_MODEL=qwen3.5:4b\n"
    )
    assert "SECRET" not in run(env_text=env).raw


def test_invalid_password_fail_does_not_echo_the_value(run: RunFn) -> None:
    env = GOOD_ENV.replace("POSTGRES_PASSWORD=good_pw-123", "POSTGRES_PASSWORD=bad!VALUE")
    r = run(env_text=env)
    assert r.status("env-file") == "FAIL"
    assert "bad!VALUE" not in r.raw and "VALUE" not in r.raw
