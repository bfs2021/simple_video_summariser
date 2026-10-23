"""`docker compose` behaviour that needs the real CLI (issue #7).

compose.yml/compose.override.yml *shape* is covered without Docker in
test_compose.py; these two checks need `docker compose` itself to resolve
`${POSTGRES_PASSWORD:?required}` and to confirm the dev overlay's published
port doesn't leak into an explicit `-f compose.yml` resolution. Both are
marked integration: Docker Desktop must be running.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_up_refuses_to_start_without_postgres_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    try:
        result = _run("up", "-d", "db")
        assert result.returncode != 0
        assert "POSTGRES_PASSWORD" in result.stderr
    finally:
        _run("down", "-v")


def test_explicit_compose_yml_config_has_no_published_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "integration-test-password")

    result = _run("-f", "compose.yml", "config")

    assert result.returncode == 0
    # Only `web` publishes a port (#58); `db`'s 5432 is dev-overlay only.
    assert result.stdout.count("published:") == 1
    assert "5432" not in result.stdout.replace("db:5432", "")
    # The dev-only `devhost` network (compose.override.yml, #7 QA fix) must
    # never reach the production resolution.
    assert "devhost" not in result.stdout


def test_default_config_attaches_db_to_the_dev_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Override merged in (the default, local behaviour): `db` picks up the
    # non-internal `devhost` network alongside `internal`, which is what
    # actually makes the `127.0.0.1:5432` publish reachable from the host -
    # `internal: true` alone blocks port forwarding regardless of the
    # `ports:` mapping (#7 QA FAIL root cause).
    monkeypatch.setenv("POSTGRES_PASSWORD", "integration-test-password")

    result = _run("config")

    assert result.returncode == 0
    assert "devhost" in result.stdout
    assert "127.0.0.1" in result.stdout


def _resolved(
    monkeypatch: pytest.MonkeyPatch, **env: str
) -> dict[str, dict[str, object]]:
    monkeypatch.setenv("POSTGRES_PASSWORD", "integration-test-password")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    result = _run("-f", "compose.yml", "config", "--format", "json")
    assert result.returncode == 0, result.stderr
    assert "warning" not in result.stderr.lower()  # no `$` interpolation
    services: dict[str, dict[str, object]] = json.loads(result.stdout)["services"]
    return services


def _env(services: dict[str, dict[str, object]], name: str) -> dict[str, str]:
    env = services[name]["environment"]
    assert isinstance(env, dict)
    return env


SHARED = ("PROMPT_VERSION", "SUMMARIZER", "CHUNK_SEC", "OVERLAP_SEC")


def test_shared_variables_are_equal_across_planner_analyzer_and_transcriber(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = _resolved(monkeypatch)
    for key in SHARED:
        values = {
            _env(services, n)[key] for n in ("planner", "analyzer", "transcriber")
        }
        assert len(values) == 1, key


def test_prompt_version_override_reaches_all_three_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = _resolved(monkeypatch, PROMPT_VERSION="v2")
    for name in ("planner", "analyzer", "transcriber"):
        assert _env(services, name)["PROMPT_VERSION"] == "v2"


def test_anthropic_api_key_only_on_the_analyzer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = _resolved(monkeypatch)
    assert "ANTHROPIC_API_KEY" in _env(services, "analyzer")
    for name in ("migrate", "api", "planner", "transcriber"):
        assert "ANTHROPIC_API_KEY" not in _env(services, name)


def test_web_is_the_only_published_port_and_on_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = _resolved(monkeypatch)
    for name, service in services.items():
        if name != "web":
            assert "ports" not in service, name
    ports = services["web"]["ports"]
    assert isinstance(ports, list) and len(ports) == 1
    assert ports[0]["host_ip"] == "127.0.0.1"
    assert str(ports[0]["published"]) == "8080"
    assert ports[0]["target"] == 80


def test_planner_has_one_replica_and_worker_healthcheck_keeps_its_dollars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = _resolved(monkeypatch)
    deploy = services["planner"]["deploy"]
    assert isinstance(deploy, dict) and deploy["replicas"] == 1
    health = services["planner"]["healthcheck"]
    assert isinstance(health, dict)
    # Verified against Docker Compose v2.40.3: `config --format json`
    # re-escapes every "$" as "$$" (the compose.yml escape for a literal "$").
    # The container still receives single dollars; the doubled form in the
    # resolved config is expected, not a bug. A single "$" in compose.yml
    # would be interpolated by Compose and fail this assertion.
    assert (
        "test $$(( $$(date +%s) - $$(stat -c %Y /tmp/heartbeat) )) -lt 180"
        in health["test"][1]
    )
    assert services["planner"]["healthcheck"] == services["analyzer"]["healthcheck"]
    assert services["planner"]["healthcheck"] == services["transcriber"]["healthcheck"]


def test_config_fails_naming_postgres_password_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    result = _run("-f", "compose.yml", "config")
    assert result.returncode != 0
    assert "POSTGRES_PASSWORD" in result.stderr


# --- development overlay (issue #59) ---------------------------------------
#
# Resolved-config checks: only the `docker compose` CLI, no build, no running
# stack. Hermetic: an empty --env-file (so the developer's .env is ignored)
# and COMPOSE_* / WHISPER_MODEL cleared from the environment.

Config = dict[str, dict[str, object]]
SOURCE_DIRS = ("common", "adapters", "services")
# The web files the dev overlay binds (read-only) into /home/node/app.
WEB_SOURCE_FILES = (
    "src", "index.html", "index.css", "package.json", "package-lock.json", "tsconfig.json",
    "vite.config.ts",
)  # fmt: skip
BACKEND_SERVICES = ("api", "planner", "analyzer", "transcriber")


def _hermetic(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for name in (
        "WHISPER_MODEL",
        "COMPOSE_FILE",
        "COMPOSE_PROFILES",
        "COMPOSE_PATH_SEPARATOR",
        "COMPOSE_PROJECT_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "integration-test-password")
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def _config(tmp_path: Path, *, prod: bool) -> tuple[Config, str]:
    env_file = tmp_path / "empty.env"
    env_file.write_text("")
    args = ["--env-file", str(env_file)]
    if prod:
        args += ["-f", "compose.yml"]
    result = _run(*args, "config", "--format", "json")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    services: Config = data["services"]
    return services, result.stdout


def _dev(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **env: str) -> Config:
    _hermetic(monkeypatch, **env)
    return _config(tmp_path, prod=False)[0]


def _binds(service: dict[str, object]) -> dict[str, tuple[str, bool]]:
    """target -> (source, read_only) for every bind mount of a service."""
    out: dict[str, tuple[str, bool]] = {}
    volumes = service.get("volumes", [])
    assert isinstance(volumes, list)
    for vol in volumes:
        if vol["type"] == "bind":
            out[vol["target"]] = (vol["source"], bool(vol.get("read_only")))
    return out


def _volumes(service: dict[str, object]) -> dict[str, str]:
    volumes = service.get("volumes", [])
    assert isinstance(volumes, list)
    return {v["target"]: v["type"] for v in volumes}


def _env_of(service: dict[str, object]) -> dict[str, str]:
    env = service["environment"]
    assert isinstance(env, dict)
    return env


@pytest.mark.parametrize("name", BACKEND_SERVICES)
def test_dev_config_mounts_backend_source_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    binds = _binds(_dev(monkeypatch, tmp_path)[name])
    assert binds == {f"/app/{d}": (str(REPO_ROOT / d), True) for d in SOURCE_DIRS}


def test_dev_config_migrate_mounts_migrations_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    services = _dev(monkeypatch, tmp_path)
    assert _binds(services["migrate"]) == {
        "/app/migrations": (str(REPO_ROOT / "migrations"), True)
    }
    assert services["migrate"]["command"] == ["alembic", "upgrade", "head"]


def test_dev_config_mounts_nothing_else_and_keeps_named_volumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    services = _dev(monkeypatch, tmp_path)
    sources = {src for svc in services.values() for src, _ in _binds(svc).values()}
    assert sources == {str(REPO_ROOT / d) for d in (*SOURCE_DIRS, "migrations")} | {
        str(REPO_ROOT / "web" / n) for n in WEB_SOURCE_FILES
    }
    assert _volumes(services["planner"])["/data/audio"] == "volume"
    assert _volumes(services["transcriber"])["/data/audio"] == "volume"
    assert _volumes(services["transcriber"])["/models"] == "volume"
    assert _volumes(services["db"]) == {"/var/lib/postgresql": "volume"}


def test_dev_config_api_runs_uvicorn_reload_workers_keep_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    services = _dev(monkeypatch, tmp_path)
    assert services["api"]["command"] == [
        "uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000",
        "--reload",
        "--reload-dir", "/app/common",
        "--reload-dir", "/app/adapters",
        "--reload-dir", "/app/services",
    ]  # fmt: skip
    assert services["planner"]["command"] == ["python", "-m", "services.planner.main"]
    assert services["analyzer"]["command"] == ["python", "-m", "services.analyzer.main"]
    assert services["transcriber"]["command"] == [
        "python", "-m", "services.transcriber.main"
    ]  # fmt: skip


WATCHED_WORKERS = ("planner", "analyzer", "transcriber")


def _watch_rules(service: dict[str, object]) -> list[dict[str, object]]:
    develop = service["develop"]
    assert isinstance(develop, dict)
    rules = develop["watch"]
    assert isinstance(rules, list)
    return rules


def test_dev_config_only_the_three_workers_have_a_watch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    services = _dev(monkeypatch, tmp_path)
    assert {n for n, svc in services.items() if "develop" in svc} == set(WATCHED_WORKERS)
    for name in ("api", "web", "migrate", "db"):
        assert "develop" not in services[name], name


@pytest.mark.parametrize("name", WATCHED_WORKERS)
def test_dev_config_worker_watch_restarts_on_its_own_sources_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    rules = _watch_rules(_dev(monkeypatch, tmp_path)[name])
    assert {r["action"] for r in rules} == {"restart"}
    assert sorted(str(r["path"]) for r in rules) == sorted(
        str(REPO_ROOT / p) for p in ("common", "adapters", f"services/{name}")
    )
    for rule in rules:
        assert set(rule["ignore"]) >= {"**/__pycache__/**", "**/*.pyc"}  # type: ignore[call-overload]


def test_dev_config_web_is_the_vite_dev_server_without_a_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    web = _dev(monkeypatch, tmp_path)["web"]
    assert web["image"] == "node:24-alpine"
    assert "build" not in web
    command = " ".join(web["command"])  # type: ignore[arg-type]
    assert "npm ci" in command and "npm run dev" in command
    assert "--host 0.0.0.0" in command and "--port 5173" in command
    assert "--strictPort" in command
    assert web["working_dir"] == "/home/node/app"
    assert _env_of(web)["API_PROXY_TARGET"] == "http://api:8000"
    assert list(web["networks"]) == ["edge"]  # type: ignore[call-overload]
    # Only source files, read-only; node_modules lives in the container layer,
    # so there is no volume at all (nothing nested in a bind source).
    assert _binds(web) == {
        f"/home/node/app/{n}": (str(REPO_ROOT / "web" / n), True)
        for n in WEB_SOURCE_FILES
    }
    assert _volumes(web) == {f"/home/node/app/{n}": "bind" for n in WEB_SOURCE_FILES}


def test_dev_config_publishes_exactly_db_5432_and_web_5173_on_loopback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    services = _dev(monkeypatch, tmp_path)
    published = {
        (name, str(p["published"]), p["target"], p.get("host_ip"))
        for name, svc in services.items()
        for p in svc.get("ports", [])  # type: ignore[attr-defined]
    }
    assert published == {
        ("db", "5432", 5432, "127.0.0.1"),
        ("web", "5173", 5173, "127.0.0.1"),
    }
    # Compose returns networks sorted alphabetically, not in declaration order.
    assert set(services["db"]["networks"]) == {"internal", "devhost"}  # type: ignore[call-overload]


def test_dev_config_whisper_model_defaults_to_tiny_and_follows_the_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unset = _dev(monkeypatch, tmp_path)
    assert _env_of(unset["transcriber"])["WHISPER_MODEL"] == "tiny"
    base = _dev(monkeypatch, tmp_path, WHISPER_MODEL="base")
    assert _env_of(base["transcriber"])["WHISPER_MODEL"] == "base"


def test_dev_config_adds_no_secrets_or_new_env_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dev = _dev(monkeypatch, tmp_path)
    _hermetic(monkeypatch)
    prod, _ = _config(tmp_path, prod=True)
    for name, svc in dev.items():
        added = set(_env_of(svc)) - set(prod[name].get("environment", {}))  # type: ignore[call-overload]
        assert added <= {"API_PROXY_TARGET"}, name
    assert "ANTHROPIC_API_KEY" not in _env_of(dev["transcriber"])
    assert _env_of(dev["transcriber"]).keys() == _env_of(prod["transcriber"]).keys()


def test_prod_config_has_none_of_the_dev_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _hermetic(monkeypatch)
    services, raw = _config(tmp_path, prod=True)
    for needle in (
        '"type": "bind"',
        "--reload",
        "node:24-alpine",
        "5173",
        "devhost",
        "API_PROXY_TARGET",
        "develop",
    ):
        assert needle not in raw, needle
    assert '"published": "5432"' not in raw
    for svc in services.values():
        assert not _binds(svc)
    assert _env_of(services["transcriber"])["WHISPER_MODEL"] == "large-v3"
    web = services["web"]
    assert web["image"] == "ytdigest-web:latest"
    assert "build" in web
    ports = web["ports"]
    assert isinstance(ports, list) and len(ports) == 1
    assert (ports[0]["host_ip"], str(ports[0]["published"]), ports[0]["target"]) == (
        "127.0.0.1", "8080", 80
    )  # fmt: skip
