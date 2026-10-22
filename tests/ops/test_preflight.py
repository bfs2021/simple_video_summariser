from __future__ import annotations

import os
import select
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "ops" / "host" / "preflight.sh"
DROP_IN = ROOT / "ops" / "host" / "ollama.override.conf"
ENV_EXAMPLE = ROOT / ".env.example"

GOOD_ENV = (
    "COMPOSE_FILE=compose.yml\n"
    "POSTGRES_PASSWORD=good_pw-123\n"
    "TRANSCRIBER_CPUS=3.0\n"
    "OLLAMA_MODEL=qwen3.5:4b\n"
    "SUMMARIZER=ollama\n"
)

EGRESS_HOSTS = (
    "www.youtube.com",
    "redirector.googlevideo.com",
    "huggingface.co",
    "cdn-lfs.huggingface.co",
    "pypi.org",
    "files.pythonhosted.org",
    "deb.debian.org",
    "registry.npmjs.org",
    "registry-1.docker.io",
)

STUB_KEYS = (
    "STUB_COMPOSE_CONFIG",
    "STUB_COMPOSE_CONFIG_RC",
    "STUB_COMPOSE_VERSION",
    "STUB_CONTAINER_OUT",
    "STUB_CONTAINER_RC",
    "STUB_DF_FREE_KB",
    "STUB_DOCKER_GROUP",
    "STUB_DOCKER_LOG",
    "STUB_DOCKER_ROOT",
    "STUB_DOCKER_SERVER_VERSION",
    "STUB_HTTP_ERROR_HOST",
    "STUB_NETWORK_CREATE_RC",
    "STUB_NPROC",
    "STUB_NTP",
    "STUB_READY_FD",
    "STUB_RELEASE_FD",
    "STUB_SS",
    "STUB_SYSTEMCTL",
)

PASS_STUBS = {
    "STUB_DOCKER_SERVER_VERSION": "29.1.2",
    "STUB_COMPOSE_VERSION": "2.40.3",
    "STUB_SYSTEMCTL": "enabled",
    "STUB_DOCKER_ROOT": "/var/lib/docker",
    "STUB_COMPOSE_CONFIG": "services:\n  api:\n    image: x\n",
    "STUB_COMPOSE_CONFIG_RC": "0",
    "STUB_DF_FREE_KB": str(40 * 1024 * 1024),
    "STUB_NPROC": "3",
    "STUB_NTP": "yes",
    "STUB_SS": "LISTEN 0 4096 172.17.0.1:11434 0.0.0.0:*",
    "STUB_DOCKER_GROUP": "docker:x:999:deploy",
    "STUB_CONTAINER_OUT": "OLLAMA\t200\tqwen3.5:4b\n"
    + "".join(f"EGRESS\t{host}\tOK\n" for host in EGRESS_HOSTS),
}

_DOCKER_STUB = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$STUB_DOCKER_LOG"
case "$1 ${2:-}" in
  "version "*)
    [ -n "${STUB_DOCKER_SERVER_VERSION:-}" ] || exit 1
    echo "$STUB_DOCKER_SERVER_VERSION" ;;
  "compose version"*)
    [ -n "${STUB_COMPOSE_VERSION:-}" ] || exit 1
    echo "$STUB_COMPOSE_VERSION" ;;
  "compose config"*)
    printf '%b' "${STUB_COMPOSE_CONFIG:-}"
    exit "${STUB_COMPOSE_CONFIG_RC:-0}" ;;
  "info "*) echo "${STUB_DOCKER_ROOT:-}" ;;
  "network create"*) [ "${STUB_NETWORK_CREATE_RC:-0}" = 0 ] ;;
  "network rm"*) exit 0 ;;
  "rm "*) exit 0 ;;
  "run "*)
    if [ -n "${STUB_READY_FD:-}" ]; then
      printf 'ready\n' >&"$STUB_READY_FD"
      read -r _ <&"$STUB_RELEASE_FD"
    fi
    if [ -n "${STUB_HTTP_ERROR_HOST:-}" ]; then
      for arg in "$@"; do
        case "$arg" in PREFLIGHT_*=*) export "$arg" ;; esac
      done
      probe="${!#}"
      PREFLIGHT_PROBE="$probe" python -c 'import io,json,os,urllib.error,urllib.request
class Response:
 def __init__(self,url): self.status=200; self.url=url
 def __enter__(self): return self
 def __exit__(self,*args): return False
 def read(self,*args): return b"{}"
def urlopen(url,timeout):
 if os.environ["STUB_HTTP_ERROR_HOST"] in url:
  raise urllib.error.HTTPError(url,401,"unauthorized",{},io.BytesIO())
 response=Response(url)
 if "api/tags" in url:
  response.read=lambda *args: json.dumps({"models":[{"model":"qwen3.5:4b"}]}).encode()
 return response
urllib.request.urlopen=urlopen
exec(os.environ["PREFLIGHT_PROBE"])'
    else
      printf '%b' "${STUB_CONTAINER_OUT:-}"
    fi
    exit "${STUB_CONTAINER_RC:-0}" ;;
  *) exit 99 ;;
esac
"""

_STUBS = {
    "docker": _DOCKER_STUB,
    "systemctl": "#!/usr/bin/env bash\necho \"${STUB_SYSTEMCTL:-disabled}\"\n",
    "timedatectl": "#!/usr/bin/env bash\necho \"${STUB_NTP:-no}\"\n",
    "df": "#!/usr/bin/env bash\nprintf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\nprintf 'stub 1 1 %s 1%% /\\n' \"${STUB_DF_FREE_KB:-0}\"\n",
    "nproc": "#!/usr/bin/env bash\necho \"${STUB_NPROC:-0}\"\n",
    "ss": "#!/usr/bin/env bash\nprintf '%s\\n' \"${STUB_SS:-}\"\n",
    "getent": "#!/usr/bin/env bash\nprintf '%s\\n' \"${STUB_DOCKER_GROUP:-}\"\n",
}


@dataclass
class Result:
    code: int
    lines: dict[str, tuple[str, str]] = field(default_factory=dict)
    raw: str = ""
    docker_calls: list[str] = field(default_factory=list)

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
        interrupt: signal.Signals | None = ...,
    ) -> Result: ...


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunFn:
    for key in STUB_KEYS + (
        "ENV_FILE",
        "PREFLIGHT_MEMINFO",
        "MIN_FREE_GB",
        "MIN_MEM_GB",
        "MIN_DOCKER_VERSION",
        "MIN_COMPOSE_VERSION",
    ):
        monkeypatch.delenv(key, raising=False)

    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in _STUBS.items():
        path = bindir / name
        path.write_text(body)
        path.chmod(0o755)
    for name in ("awk", "bash", "cat", "cut", "grep", "sed", "stat", "tail", "tr"):
        source = shutil.which(name)
        assert source is not None
        (bindir / name).symlink_to(source)
    (bindir / "python").symlink_to(sys.executable)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setenv("ENV_FILE", ".env")

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    meminfo = tmp_path / "meminfo"
    docker_log = tmp_path / "docker.log"

    def _run(
        *,
        env_text: str = GOOD_ENV,
        stubs: dict[str, str] | None = None,
        thresholds: dict[str, str] | None = None,
        env_mode: int = 0o600,
        write_env: bool = True,
        args: list[str] | None = None,
        interrupt: signal.Signals | None = None,
    ) -> Result:
        for key in STUB_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("ENV_FILE", ".env")
        merged = {**PASS_STUBS, **(stubs or {})}
        for key, value in merged.items():
            monkeypatch.setenv(key, value)
        limits = {
            "MIN_FREE_GB": "40",
            "MIN_MEM_GB": "12",
            "MIN_DOCKER_VERSION": "25.0",
            "MIN_COMPOSE_VERSION": "2.24.0",
            **(thresholds or {}),
        }
        meminfo.write_text("MemTotal:       12582912 kB\n")
        monkeypatch.setenv("PREFLIGHT_MEMINFO", str(meminfo))
        for key, value in limits.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("STUB_DOCKER_LOG", str(docker_log))
        docker_log.write_text("")
        env_file = checkout / ".env"
        if write_env:
            env_file.write_text(env_text)
            env_file.chmod(env_mode)
        elif env_file.exists():
            env_file.unlink()

        command = [str(bindir / "bash"), str(SCRIPT), *(args or [])]
        if interrupt is None:
            completed = subprocess.run(
                command,
                cwd=checkout,
                capture_output=True,
                text=True,
                check=False,
            )
            return_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        else:
            ready_read, ready_write = os.pipe()
            release_read, release_write = os.pipe()
            monkeypatch.setenv("STUB_READY_FD", str(ready_write))
            monkeypatch.setenv("STUB_RELEASE_FD", str(release_read))
            proc = subprocess.Popen(
                command,
                cwd=checkout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                pass_fds=(ready_write, release_read),
            )
            os.close(ready_write)
            os.close(release_read)
            readable, _, _ = select.select([ready_read], [], [], 5)
            assert readable, "docker run did not reach the blocking stub"
            assert os.read(ready_read, 6) == b"ready\n"
            proc.send_signal(interrupt)
            os.close(release_write)
            stdout, stderr = proc.communicate(timeout=5)
            return_code = proc.returncode
            os.close(ready_read)

        lines: dict[str, tuple[str, str]] = {}
        for line in stdout.splitlines():
            parts = line.split(None, 1)
            if parts and parts[0] in {"PASS", "WARN", "FAIL"}:
                check, _, detail = (parts[1] if len(parts) > 1 else "").partition(":")
                lines[check.strip()] = (parts[0], detail.strip())
        return Result(
            code=return_code,
            lines=lines,
            raw=stdout + stderr,
            docker_calls=docker_log.read_text().splitlines(),
        )

    return _run


def test_help_lists_all_checks_and_external_inputs(run: RunFn) -> None:
    result = run(args=["--help"])
    assert result.code == 0
    for check in (
        "docker-engine",
        "docker-compose",
        "docker-boot",
        "disk",
        "cpu",
        "memory",
        "time-sync",
        "env-file",
        "compose-config",
        "ollama-from-container",
        "ollama-bind",
        "egress",
        "docker-group",
    ):
        assert check in result.raw
    for variable in (
        "MIN_FREE_GB",
        "MIN_MEM_GB",
        "MIN_DOCKER_VERSION",
        "MIN_COMPOSE_VERSION",
        "PREFLIGHT_MEMINFO",
    ):
        assert variable in result.raw


def test_unknown_flag_is_usage_error(run: RunFn) -> None:
    assert run(args=["--nope"]).code == 2


def test_each_run_overrides_ambient_env_file_and_stub_values(
    run: RunFn, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ambient = tmp_path / "ambient.env"
    ambient.write_text("COMPOSE_FILE=wrong.yml\nPOSTGRES_PASSWORD=ambient\n")
    monkeypatch.setenv("ENV_FILE", str(ambient))
    monkeypatch.setenv("STUB_SYSTEMCTL", "disabled")
    result = run()
    assert result.status("env-file") == "PASS"
    assert result.status("docker-boot") == "PASS"


def test_all_pass_exits_zero_and_reports_thirteen_checks(run: RunFn) -> None:
    result = run()
    assert result.code == 0
    assert len(result.lines) == 13
    assert all(status != "FAIL" for status, _ in result.lines.values())


def test_any_fail_exits_one(run: RunFn) -> None:
    result = run(stubs={"STUB_SYSTEMCTL": "disabled"})
    assert result.code == 1
    assert result.status("docker-boot") == "FAIL"


def test_all_checks_run_after_first_failure(run: RunFn) -> None:
    result = run(stubs={"STUB_DOCKER_SERVER_VERSION": ""})
    assert result.code == 1
    assert len(result.lines) == 13
    assert result.status("docker-group") == "PASS"


@pytest.mark.parametrize(
    ("stubs", "expected"),
    [({}, "PASS"), ({"STUB_DOCKER_SERVER_VERSION": "24.0.9"}, "FAIL"), ({"STUB_DOCKER_SERVER_VERSION": ""}, "FAIL")],
)
def test_docker_engine(run: RunFn, stubs: dict[str, str], expected: str) -> None:
    assert run(stubs=stubs).status("docker-engine") == expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [("2.24.0", "PASS"), ("2.100.1", "PASS"), ("2.23.9", "FAIL"), ("v2.24.0", "PASS")],
)
def test_docker_compose_versions(run: RunFn, version: str, expected: str) -> None:
    assert run(stubs={"STUB_COMPOSE_VERSION": version}).status("docker-compose") == expected


def test_version_floor_override_names(run: RunFn) -> None:
    result = run(
        stubs={"STUB_DOCKER_SERVER_VERSION": "25.0", "STUB_COMPOSE_VERSION": "2.24.0"},
        thresholds={"MIN_DOCKER_VERSION": "26.0", "MIN_COMPOSE_VERSION": "2.25.0"},
    )
    assert result.status("docker-engine") == "FAIL"
    assert result.status("docker-compose") == "FAIL"


@pytest.mark.parametrize(("enabled", "expected"), [("enabled", "PASS"), ("disabled", "FAIL")])
def test_docker_boot(run: RunFn, enabled: str, expected: str) -> None:
    assert run(stubs={"STUB_SYSTEMCTL": enabled}).status("docker-boot") == expected


@pytest.mark.parametrize(("free_gb", "expected"), [(40, "PASS"), (39, "FAIL")])
def test_disk_exact_boundary(run: RunFn, free_gb: int, expected: str) -> None:
    free_kb = free_gb * 1024 * 1024
    assert run(stubs={"STUB_DF_FREE_KB": str(free_kb)}).status("disk") == expected


@pytest.mark.parametrize(
    ("cores", "cpus", "expected"),
    [("3", "3.0", "PASS"), ("2", "2.5", "FAIL"), ("3", "2.5", "PASS")],
)
def test_cpu_boundaries(run: RunFn, cores: str, cpus: str, expected: str) -> None:
    env_text = GOOD_ENV.replace("TRANSCRIBER_CPUS=3.0", f"TRANSCRIBER_CPUS={cpus}")
    assert run(env_text=env_text, stubs={"STUB_NPROC": cores}).status("cpu") == expected


def test_memory_passes_at_exact_boundary(run: RunFn) -> None:
    assert run().status("memory") == "PASS"


def test_memory_below_floor_warns(run: RunFn, tmp_path: Path) -> None:
    meminfo = tmp_path / "low-meminfo"
    meminfo.write_text("MemTotal:       11534336 kB\n")
    result = run(thresholds={"PREFLIGHT_MEMINFO": str(meminfo)})
    assert result.status("memory") == "WARN"
    assert result.code == 0


@pytest.mark.parametrize(("synced", "expected"), [("yes", "PASS"), ("no", "WARN")])
def test_time_sync(run: RunFn, synced: str, expected: str) -> None:
    assert run(stubs={"STUB_NTP": synced}).status("time-sync") == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({}, "PASS"), ({"write_env": False}, "FAIL"), ({"env_mode": 0o640}, "FAIL")],
)
def test_env_file_presence_and_mode(run: RunFn, kwargs: dict[str, Any], expected: str) -> None:
    assert run(**kwargs).status("env-file") == expected


@pytest.mark.parametrize(
    "env_text",
    [
        GOOD_ENV.replace("COMPOSE_FILE=compose.yml\n", ""),
        GOOD_ENV.replace("POSTGRES_PASSWORD=good_pw-123", "POSTGRES_PASSWORD="),
        GOOD_ENV.replace("POSTGRES_PASSWORD=good_pw-123", "POSTGRES_PASSWORD=has space!"),
    ],
)
def test_env_file_rejects_invalid_settings(run: RunFn, env_text: str) -> None:
    assert run(env_text=env_text).status("env-file") == "FAIL"


@pytest.mark.parametrize(
    ("config", "return_code", "expected"),
    [
        ("services:\n  api:\n    image: x\n", "0", "PASS"),
        ("services:\n  api:\n    volumes:\n      - type: bind\n", "0", "FAIL"),
        ("services:\n  transcriber:\n    environment:\n      WHISPER_MODEL: tiny\n", "0", "FAIL"),
        ("", "1", "FAIL"),
    ],
)
def test_compose_config(run: RunFn, config: str, return_code: str, expected: str) -> None:
    result = run(stubs={"STUB_COMPOSE_CONFIG": config, "STUB_COMPOSE_CONFIG_RC": return_code})
    assert result.status("compose-config") == expected


@pytest.mark.parametrize(
    ("container_output", "expected"),
    [
        (PASS_STUBS["STUB_CONTAINER_OUT"], "PASS"),
        (PASS_STUBS["STUB_CONTAINER_OUT"].replace("qwen3.5:4b", "llama3:8b"), "WARN"),
        (PASS_STUBS["STUB_CONTAINER_OUT"].replace("OLLAMA\t200", "OLLAMA\tERR"), "FAIL"),
    ],
)
def test_ollama_from_container(run: RunFn, container_output: str, expected: str) -> None:
    assert run(stubs={"STUB_CONTAINER_OUT": container_output}).status("ollama-from-container") == expected


@pytest.mark.parametrize(
    ("listening", "expected"),
    [
        ("LISTEN 0 4096 172.17.0.1:11434 0.0.0.0:*", "PASS"),
        ("LISTEN 0 4096 0.0.0.0:11434 0.0.0.0:*", "WARN"),
        ("LISTEN 0 4096 [::]:11434 [::]:*", "WARN"),
        ("LISTEN 0 4096 *:11434 *:*", "WARN"),
        ("", "FAIL"),
    ],
)
def test_ollama_bind(run: RunFn, listening: str, expected: str) -> None:
    assert run(stubs={"STUB_SS": listening}).status("ollama-bind") == expected


def test_egress_passes_for_all_documented_destinations(run: RunFn) -> None:
    result = run()
    assert result.status("egress") == "PASS"
    for host in EGRESS_HOSTS:
        assert host in result.lines["egress"][1]


def test_googlevideo_probe_uses_stable_real_subdomain_in_container_invocation(run: RunFn) -> None:
    result = run()
    docker_run = next(call for call in result.docker_calls if call.startswith("run "))
    assert "redirector.googlevideo.com" in docker_run
    assert ".example.googlevideo.com" not in docker_run


@pytest.mark.parametrize("host", EGRESS_HOSTS)
def test_egress_failure_names_host(run: RunFn, host: str) -> None:
    output = PASS_STUBS["STUB_CONTAINER_OUT"].replace(f"EGRESS\t{host}\tOK", f"EGRESS\t{host}\tFAIL")
    result = run(stubs={"STUB_CONTAINER_OUT": output})
    assert result.status("egress") == "FAIL"
    assert host in result.lines["egress"][1]


def test_anthropic_egress_is_conditional(run: RunFn) -> None:
    anthropic_env = GOOD_ENV.replace("SUMMARIZER=ollama", "SUMMARIZER=anthropic")
    missing = run(env_text=anthropic_env)
    assert missing.status("egress") == "FAIL"
    assert "api.anthropic.com" in missing.lines["egress"][1]

    output = PASS_STUBS["STUB_CONTAINER_OUT"] + "EGRESS\tapi.anthropic.com\tOK\n"
    present = run(env_text=anthropic_env, stubs={"STUB_CONTAINER_OUT": output})
    assert present.status("egress") == "PASS"


def test_http_error_proves_egress_reachability(run: RunFn) -> None:
    env_text = GOOD_ENV.replace("SUMMARIZER=ollama", "SUMMARIZER=anthropic")
    result = run(env_text=env_text, stubs={"STUB_HTTP_ERROR_HOST": "api.anthropic.com"})
    assert result.status("egress") == "PASS", result.raw
    assert "api.anthropic.com" in result.lines["egress"][1]


def test_container_checks_use_one_network_and_one_container_without_pull(run: RunFn) -> None:
    result = run()
    assert sum(call.startswith("network create ") for call in result.docker_calls) == 1
    runs = [call for call in result.docker_calls if call.startswith("run ")]
    assert len(runs) == 1
    assert "--pull=never" in runs[0]
    assert "--network ytdigest-preflight-net-" in runs[0]


def test_docker_group_always_passes_and_lists_members(run: RunFn) -> None:
    result = run(stubs={"STUB_DOCKER_GROUP": "docker:x:999:alice,bob"})
    assert result.status("docker-group") == "PASS"
    assert "alice,bob" in result.lines["docker-group"][1]


def test_docker_group_missing_still_passes(run: RunFn) -> None:
    assert run(stubs={"STUB_DOCKER_GROUP": ""}).status("docker-group") == "PASS"


@pytest.mark.parametrize(
    "stubs",
    [{}, {"STUB_CONTAINER_OUT": "OLLAMA\tERR\t\n"}, {"STUB_NETWORK_CREATE_RC": "1"}],
)
def test_cleanup_records_container_and_network_removal(run: RunFn, stubs: dict[str, str]) -> None:
    calls = run(stubs=stubs).docker_calls
    assert any(call.startswith("rm -f ytdigest-preflight-") for call in calls)
    assert any(call.startswith("network rm ytdigest-preflight-net-") for call in calls)


@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM])
def test_cleanup_runs_when_externally_interrupted_during_blocked_docker_run(
    run: RunFn, number: signal.Signals
) -> None:
    result = run(interrupt=number)
    assert result.code == 128 + number
    assert any(call.startswith("rm -f ytdigest-preflight-") for call in result.docker_calls)
    assert any(call.startswith("network rm ytdigest-preflight-net-") for call in result.docker_calls)


def test_never_prints_secret_values(run: RunFn) -> None:
    env_text = GOOD_ENV.replace("good_pw-123", "pw-SECRET") + "ANTHROPIC_API_KEY=sk-test-SECRET\n"
    assert "SECRET" not in run(env_text=env_text).raw


def test_script_static_contract() -> None:
    text = SCRIPT.read_text()
    assert os.access(SCRIPT, os.X_OK)
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -u" in text


def test_ollama_drop_in_static_contract() -> None:
    text = DROP_IN.read_text()
    assert "[Service]" in text
    assert 'Environment="OLLAMA_HOST=' in text
    assert ":11434\"" in text


def test_env_example_does_not_set_compose_file() -> None:
    assignments = [line for line in ENV_EXAMPLE.read_text().splitlines() if not line.lstrip().startswith("#")]
    assert not any(line.startswith("COMPOSE_FILE=") for line in assignments)
