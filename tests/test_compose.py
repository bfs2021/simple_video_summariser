"""compose.yml / compose.override.yml structure (issue #7, architecture.md 11.4).

No Docker is invoked here — these are plain-text checks of the checked-in
YAML (PyYAML isn't an approved dependency, so a small line-based scanner
stands in for a real parser; good enough for the flat, hand-written files
this repo commits). The whole file runs under `pytest -m "not integration"`
with Docker stopped. `test_compose_docker.py` covers the behaviour that
genuinely needs the `docker compose` CLI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_YML = REPO_ROOT / "compose.yml"
COMPOSE_OVERRIDE_YML = REPO_ROOT / "compose.override.yml"


def _service_block(compose_text: str, service: str) -> str:
    """The indented body of one top-level service.

    Line-based, not a real YAML parser: walks from the `  <service>:` header
    until a line that isn't blank and isn't indented at least 4 spaces
    (i.e. the next service, or a top-level key like `volumes:`).
    """
    lines = compose_text.splitlines()
    header = f"  {service}:"
    start = lines.index(header)
    end = start + 1
    while end < len(lines) and (lines[end] == "" or lines[end].startswith("    ")):
        end += 1
    return "\n".join(lines[start:end])


def _top_level_section(compose_text: str, key: str) -> str:
    """The body of a top-level section (e.g. `services:`), up to the next
    top-level key or end of file."""
    lines = compose_text.splitlines()
    start = lines.index(f"{key}:") + 1
    end = start
    while end < len(lines) and (lines[end] == "" or lines[end].startswith(" ")):
        end += 1
    return "\n".join(lines[start:end])


SERVICES = ["db", "migrate", "api", "planner", "analyzer", "transcriber", "web"]
LONG_RUNNING = ["db", "api", "planner", "analyzer", "transcriber", "web"]
WORKERS = ["planner", "analyzer", "transcriber"]


def _block(service: str) -> str:
    return _service_block(COMPOSE_YML.read_text(), service)


def _anchor_block(anchor_line: str) -> str:
    """The indented body under a top-level `x-...: &anchor` line."""
    lines = COMPOSE_YML.read_text().splitlines()
    start = lines.index(anchor_line) + 1
    end = start
    while end < len(lines) and lines[end].startswith("  "):
        end += 1
    return "\n".join(lines[start:end])


def _interpolated_vars(text: str) -> set[str]:
    return set(re.findall(r"\$\{([A-Z_]+)[:}?-]", text))


def test_compose_yml_defines_exactly_the_seven_stack_services() -> None:
    services_text = _top_level_section(COMPOSE_YML.read_text(), "services")
    service_headers = [
        line.strip()
        for line in services_text.splitlines()
        if line.startswith("  ")
        and not line.startswith("    ")
        and not line.lstrip().startswith("#")
    ]
    assert sorted(service_headers) == sorted(f"{name}:" for name in SERVICES)
    assert "metabase" not in COMPOSE_YML.read_text().lower()


def test_header_comment_lists_the_stack_and_production_command() -> None:
    header = COMPOSE_YML.read_text().split("x-backend-env", 1)[0]
    assert "added by their own tasks" not in header
    for name in SERVICES:
        assert f"`{name}`" in header
    assert "docker compose -f compose.yml up -d" in header


def test_backend_services_share_the_backend_image_and_only_migrate_builds_it() -> None:
    for name in ("migrate", "api", "planner", "analyzer"):
        assert "image: ytdigest-backend:latest" in _block(name)
    assert "dockerfile: ops/Dockerfile.backend" in _block("migrate")
    for name in ("api", "planner", "analyzer"):
        assert "build" not in _block(name)
        # A local-only tag: Compose must never try to pull it from a registry.
        assert "pull_policy: never" in _block(name)
    assert "ops/Dockerfile.backend" not in COMPOSE_YML.read_text().replace(
        "dockerfile: ops/Dockerfile.backend", "", 1
    )


def test_transcriber_and_web_build_their_own_images() -> None:
    transcriber = _block("transcriber")
    assert "dockerfile: ops/Dockerfile.whisper" in transcriber
    assert "image: ytdigest-whisper:latest" in transcriber
    web = _block("web")
    assert "dockerfile: ops/Dockerfile.frontend" in web
    assert "image: ytdigest-web:latest" in web


def test_whisper_build_takes_the_backend_image_from_the_migrate_build() -> None:
    # ops/Dockerfile.whisper is `FROM ytdigest-backend:latest`; naming that
    # image as an additional context from `migrate` makes Compose build
    # migrate first, so the base is never stale or missing.
    block = _block("transcriber")
    assert "additional_contexts:" in block
    # The key is the image name without its tag: the tagged key is ignored
    # by Compose 2.40 / BuildKit 0.26 (found on the first real build, #58).
    assert 'ytdigest-backend: "service:migrate"' in block
    code = [ln for ln in block.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("ytdigest-backend:latest" in ln for ln in code)
    assert (
        "FROM ytdigest-backend:latest"
        in (REPO_ROOT / "ops" / "Dockerfile.whisper").read_text()
    )


@pytest.mark.parametrize(
    ("service", "command"),
    [
        ("migrate", "alembic upgrade head"),
        ("api", "uvicorn services.api.main:app --host 0.0.0.0 --port 8000"),
        ("planner", "python -m services.planner.main"),
        ("analyzer", "python -m services.analyzer.main"),
        ("transcriber", "python -m services.transcriber.main"),
    ],
)
def test_service_command_is_a_plain_string(service: str, command: str) -> None:
    assert f"    command: {command}\n" in _block(service) + "\n"


def test_no_service_wraps_its_command_in_a_shell() -> None:
    text = COMPOSE_YML.read_text()
    for line in text.splitlines():
        if line.strip().startswith(("command:", "entrypoint:")):
            assert "sh -c" not in line
            assert "bash" not in line


def test_no_alembic_outside_the_migrate_service() -> None:
    for name in SERVICES:
        if name != "migrate":
            assert "alembic" not in _block(name)


def test_no_service_sets_container_name() -> None:
    assert "container_name" not in COMPOSE_YML.read_text()


def test_backend_env_anchor_holds_exactly_the_documented_keys() -> None:
    body = _anchor_block("x-backend-env: &backend-env")
    keys = [line.split(":", 1)[0].strip() for line in body.splitlines()]
    assert keys == [
        "DATABASE_URL",
        "SUMMARIZER",
        "ANTHROPIC_BATCH",
        "PROMPT_VERSION",
        "CHUNK_SEC",
        "OVERLAP_SEC",
        "LOG_FORMAT",
    ]
    assert "LOG_FORMAT: json" in body
    assert "ANTHROPIC_API_KEY" not in body


def test_migrate_uses_the_anchor_and_workers_merge_it() -> None:
    assert "environment: *backend-env" in _block("migrate")
    assert "environment: *backend-env" in _block("api")
    for name in WORKERS:
        assert "<<: *backend-env" in _block(name)


@pytest.mark.parametrize(
    ("service", "expected"),
    [
        (
            "planner",
            {
                "POLL_INTERVAL_SEC": "3600",
                "AUDIO_TTL_DAYS": "30",
                "AUDIO_MAX_GB": "20",
                "AUDIO_KEEP": "1",
            },
        ),
        (
            "analyzer",
            {
                "OLLAMA_HOST": "http://host.docker.internal:11434",
                "OLLAMA_MODEL": "qwen3.5:4b",
                "ANTHROPIC_MODEL": "claude-haiku-4-5",
            },
        ),
        (
            "transcriber",
            {
                "WHISPER_MODEL": "large-v3",
                "WHISPER_COMPUTE": "int8",
                "WHISPER_THREADS": "3",
                "PREFER_WHISPER": "0",
                "AUTO_CAPTION_FALLBACK": "1",
                "MAX_ATTEMPTS_TRANSCRIBE": "2",
                "AUDIO_KEEP": "1",
                "AUDIO_MAX_GB": "20",
            },
        ),
    ],
)
def test_service_specific_variables_have_defaults(
    service: str, expected: dict[str, str]
) -> None:
    block = _block(service)
    for key, default in expected.items():
        assert f"      {key}: ${{{key}:-{default}}}\n" in block + "\n"


def test_whisper_threads_default_is_three_and_documented_with_the_cpu_limit() -> None:
    block = _block("transcriber")
    assert "WHISPER_THREADS: ${WHISPER_THREADS:-3}" in block
    assert (
        "TRANSCRIBER_CPUS"
        in block.split("WHISPER_THREADS:")[0].rsplit("WHISPER_COMPUTE", 1)[1]
    )


def test_anthropic_api_key_is_on_the_analyzer_only() -> None:
    assert "ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}" in _block("analyzer")
    for name in ("db", "migrate", "api", "planner", "transcriber", "web"):
        assert "ANTHROPIC_API_KEY" not in _block(name)


def test_every_interpolated_variable_is_listed_in_env_example() -> None:
    env_keys = {
        line.split("=", 1)[0]
        for line in (REPO_ROOT / ".env.example").read_text().splitlines()
        if "=" in line and not line.startswith("#")
    }
    assert _interpolated_vars(COMPOSE_YML.read_text()) <= env_keys
    assert "TRANSCRIBER_CPUS" in env_keys


def test_env_example_defaults_match_compose_and_the_key_is_empty() -> None:
    text = (REPO_ROOT / ".env.example").read_text()
    assert "\nANTHROPIC_API_KEY=\n" in text
    assert "\nWHISPER_THREADS=3\n" in text
    assert "\nTRANSCRIBER_CPUS=3.0\n" in text
    assert "[A-Za-z0-9_-]" in text and "unencoded" in text
    assert "host's CPU" in text.replace("\n# ", " ")


def test_depends_on_conditions() -> None:
    for name in ("api", "planner", "analyzer", "transcriber"):
        assert "migrate: {condition: service_completed_successfully}" in _block(name)
    assert "depends_on: {api: {condition: service_healthy}}" in _block("web")
    migrate = _block("migrate")
    assert "depends_on: {db: {condition: service_healthy}}" in migrate
    assert 'restart: "no"' in migrate


def test_nothing_depends_on_migrate_except_through_completion() -> None:
    # migrate is a one-shot: it is never anyone's started/healthy dependency.
    text = COMPOSE_YML.read_text()
    assert text.count("migrate: {condition:") == 4
    assert text.count("condition: service_completed_successfully") == 4


@pytest.mark.parametrize(
    ("service", "networks"),
    [
        ("db", "[internal]"),
        ("migrate", "[internal]"),
        ("api", "[internal, edge]"),
        ("web", "[edge]"),
        ("planner", "[internal, egress]"),
        ("analyzer", "[internal, egress]"),
        ("transcriber", "[internal, egress]"),
    ],
)
def test_service_networks(service: str, networks: str) -> None:
    assert f"    networks: {networks}\n" in _block(service) + "\n"


def test_networks_section_internal_is_internal_and_egress_is_not() -> None:
    section = _top_level_section(COMPOSE_YML.read_text(), "networks")
    assert section.count("internal: true") == 1
    assert "  internal:\n    internal: true" in section
    assert "  egress: {}" in section
    assert "  edge: {}" in section
    assert "devhost" not in COMPOSE_YML.read_text()


def test_a_comment_explains_why_egress_exists() -> None:
    text = COMPOSE_YML.read_text()
    comment = text.split("networks:\n  internal:", 1)[0].rsplit("\n\n", 1)[1]
    assert "no route to the internet" in comment
    assert "egress" in comment


def test_ports_only_on_web_bound_to_loopback() -> None:
    for name in SERVICES:
        block = _block(name)
        if name == "web":
            assert 'ports: ["127.0.0.1:8080:80"]' in block
        else:
            assert "ports:" not in block
    assert COMPOSE_YML.read_text().count("ports:") == 1


def test_analyzer_maps_host_docker_internal() -> None:
    assert 'extra_hosts: ["host.docker.internal:host-gateway"]' in _block("analyzer")
    for name in ("planner", "transcriber"):
        assert "extra_hosts" not in _block(name)


def test_api_healthcheck_curls_healthz() -> None:
    block = _block("api")
    assert '["CMD", "curl", "-fsS", "http://localhost:8000/healthz"]' in block


def test_workers_share_one_heartbeat_healthcheck_with_escaped_dollars() -> None:
    text = COMPOSE_YML.read_text()
    assert text.count("&worker-health") == 1
    assert text.count("healthcheck: *worker-health") == 2
    planner = _block("planner")
    assert "healthcheck: &worker-health" in planner
    assert (
        '"test $$(( $$(date +%s) - $$(stat -c %Y /tmp/heartbeat) )) -lt 180"' in planner
    )
    # No single-dollar substitution left for Compose to interpolate.
    assert not re.search(r"(?<!\$)\$\(", planner)
    assert "start_period: 60s" in planner
    assert "start_interval: 5s" in planner


def test_db_keeps_pg_isready_and_web_has_no_healthcheck() -> None:
    assert "pg_isready" in _block("db")
    assert "healthcheck" not in _block("web")
    assert "healthcheck" not in _block("migrate")


def test_workers_have_a_30s_stop_grace_period_above_the_shutdown_grace() -> None:
    for name in WORKERS:
        assert "    stop_grace_period: 30s" in _block(name)
    text = COMPOSE_YML.read_text()
    assert "WORKER_SHUTDOWN_GRACE_SEC" in text
    assert "WORKER_SHUTDOWN_GRACE_SEC:" not in text


def test_planner_is_pinned_to_one_replica_with_a_comment() -> None:
    block = _block("planner")
    assert "deploy: {replicas: 1}" in block
    assert "architecture.md section 1" in block


def test_named_volumes_and_their_mounts() -> None:
    section = _top_level_section(COMPOSE_YML.read_text(), "volumes")
    assert [line.strip() for line in section.splitlines() if line.strip()] == [
        "pgdata:",
        "audio:",
        "whisper-models:",
    ]
    for name in SERVICES:
        block = _block(name)
        assert ("audio:/data/audio" in block) == (name in ("planner", "transcriber"))
        assert ("whisper-models:/models" in block) == (name == "transcriber")


def test_transcriber_has_cpu_and_memory_limits() -> None:
    block = _block("transcriber")
    assert 'limits: {cpus: "${TRANSCRIBER_CPUS:-3.0}", memory: 6G}' in block
    for name in SERVICES:
        if name != "transcriber":
            assert "resources" not in _block(name)


def test_every_service_logs_via_the_anchor_and_restarts_unless_stopped() -> None:
    for name in SERVICES:
        assert "logging: *logging" in _block(name)
    for name in LONG_RUNNING:
        assert "restart: unless-stopped" in _block(name)


def test_compose_yml_does_not_depend_on_the_override() -> None:
    code = [
        line
        for line in COMPOSE_YML.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    assert not any("override" in line or "devhost" in line for line in code)


def test_db_service_uses_postgres_18_alpine_at_least_18_6() -> None:
    block = _service_block(COMPOSE_YML.read_text(), "db")
    # >=18.6 per architecture.md 16.1: pinned to the exact tag, not `:18`.
    assert "image: postgres:18-alpine" in block


def test_db_service_has_a_named_pgdata_volume() -> None:
    block = _service_block(COMPOSE_YML.read_text(), "db")
    # postgres:18's image declares VOLUME /var/lib/postgresql (not .../data,
    # as older images and architecture.md's literal example do) — mounting
    # at .../data makes the entrypoint refuse to start (task #7 finding).
    assert "pgdata:/var/lib/postgresql" in block
    assert "pgdata:/var/lib/postgresql/data" not in block
    volumes_text = _top_level_section(COMPOSE_YML.read_text(), "volumes")
    assert "pgdata:" in volumes_text


def test_db_service_has_a_pg_isready_healthcheck() -> None:
    block = _service_block(COMPOSE_YML.read_text(), "db")
    assert "pg_isready" in block


def test_db_service_requires_postgres_password() -> None:
    block = _service_block(COMPOSE_YML.read_text(), "db")
    assert "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?required}" in block


def test_compose_yml_db_service_publishes_no_ports() -> None:
    block = _service_block(COMPOSE_YML.read_text(), "db")
    assert "ports:" not in block


def test_compose_override_publishes_db_on_loopback_only() -> None:
    block = _service_block(COMPOSE_OVERRIDE_YML.read_text(), "db")
    assert '"127.0.0.1:5432:5432"' in block


def test_compose_yml_db_service_is_only_on_the_internal_network() -> None:
    # architecture.md 11.4: no route off the compose bridge except through
    # explicitly attached networks. Production never adds a second one.
    block = _service_block(COMPOSE_YML.read_text(), "db")
    assert "networks: [internal]" in block


def test_compose_override_attaches_db_to_a_second_non_internal_network() -> None:
    # `internal: true` (compose.yml, 11.4) means Docker never forwards a
    # published port for that container, on any machine, regardless of the
    # port mapping (root cause of #7's QA FAIL). The dev-only fix: attach
    # `db` to an additional, ordinary bridge network here so the
    # `127.0.0.1:5432` publish above has a non-internal network to forward
    # through. `internal` must stay listed too - this adds a network, it
    # doesn't move `db` off the production one.
    override_text = COMPOSE_OVERRIDE_YML.read_text()
    block = _service_block(override_text, "db")
    assert "networks: [internal, devhost]" in block

    # The override's own top-level `networks:` section defines only the new
    # network, and it must not be `internal: true` itself.
    networks_text = _top_level_section(override_text, "networks")
    assert "devhost:" in networks_text
    assert "internal: true" not in networks_text


def test_env_example_is_committed_with_a_postgres_password_placeholder() -> None:
    text = (REPO_ROOT / ".env.example").read_text()
    assert "POSTGRES_PASSWORD=" in text


def test_dotenv_itself_is_gitignored() -> None:
    lines = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in lines


def test_alembic_ini_script_location_is_migrations() -> None:
    text = (REPO_ROOT / "alembic.ini").read_text()
    assert "script_location = migrations" in text


def test_alembic_ini_carries_no_database_url_or_credentials() -> None:
    text = (REPO_ROOT / "alembic.ini").read_text()
    assert "sqlalchemy.url" not in text
    assert "://" not in text


def test_alembic_ini_names_revisions_sequentially_via_file_template() -> None:
    text = (REPO_ROOT / "alembic.ini").read_text()
    assert "file_template = %%(rev)s_%%(slug)s" in text


# --- development overlay (issue #59) ---------------------------------------


def _override_block(service: str) -> str:
    return _service_block(COMPOSE_OVERRIDE_YML.read_text(), service)


def _code_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]


SOURCE_MOUNTS = [
    "- ./common:/app/common:ro",
    "- ./adapters:/app/adapters:ro",
    "- ./services:/app/services:ro",
]
BACKEND_SERVICES = ["api", "planner", "analyzer", "transcriber"]


@pytest.mark.parametrize("service", BACKEND_SERVICES)
def test_overlay_mounts_backend_source_read_only(service: str) -> None:
    block = _override_block(service)
    for mount in SOURCE_MOUNTS:
        assert mount in block, (service, mount)
    # Read-only is the point (uid 10001 would write __pycache__ otherwise).
    assert not any(
        ln.strip().startswith("- ./") and not ln.rstrip().endswith(":ro")
        for ln in block.splitlines()
    )


def test_overlay_adds_mounts_and_does_not_redeclare_the_named_volumes() -> None:
    # Compose merges `volumes` by mount target, so listing only the bind
    # mounts adds to `audio` / `whisper-models`; redeclaring them here would
    # be redundant and invites a typo that silently moves the target.
    text = COMPOSE_OVERRIDE_YML.read_text()
    for name in ("planner", "transcriber"):
        block = _override_block(name)
        assert "audio:" not in block and "whisper-models:" not in block
    # ... and compose.yml still has them where #58 put them.
    assert "audio:/data/audio" in _block("planner")
    assert "audio:/data/audio" in _block("transcriber")
    assert "whisper-models:/models" in _block("transcriber")
    assert "volumes: [" not in text.replace("volumes: [audio", "")


def test_overlay_migrate_mounts_migrations_read_only() -> None:
    block = _override_block("migrate")
    assert "- ./migrations:/app/migrations:ro" in block
    assert "command:" not in block  # still `alembic upgrade head`, on demand
    assert "restart" not in block and "depends_on" not in block


def test_overlay_mounts_nothing_else_from_the_repo() -> None:
    mounts = {
        ln.strip()[2:].split(":")[0]
        for ln in _code_lines(COMPOSE_OVERRIDE_YML.read_text())
        if ln.strip().startswith("- ./")
    }
    assert mounts == {
        "./common",
        "./adapters",
        "./services",
        "./migrations",
        "./web/src",
        "./web/index.html",
        "./web/package.json",
        "./web/package-lock.json",
        "./web/tsconfig.json",
        "./web/vite.config.ts",
    }
    for forbidden in ("tests", ".env", ".git", "_docs", "scripts"):
        assert f"./{forbidden}" not in mounts


def test_overlay_api_runs_uvicorn_reload_limited_to_the_mounted_dirs() -> None:
    block = _override_block("api")
    command = next(ln for ln in block.splitlines() if "command:" in ln)
    tokens = command.split()[1:]  # drop the "command:" key
    assert tokens[:7] == [
        "uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000",
        "--reload",
    ]  # fmt: skip
    # The bare `--reload` token, not a prefix: `--reload-dir` starts with it,
    # so a startswith check passes with reload switched off.
    assert tokens.count("--reload") == 1
    # Only the three; nothing like --reload-dir /app, which would watch
    # migrations, __pycache__ and the venv's parent.
    dirs = [tokens[i + 1] for i, t in enumerate(tokens) if t == "--reload-dir"]
    assert dirs == ["/app/common", "/app/adapters", "/app/services"]


def test_reload_appears_in_the_overlay_and_never_in_compose_yml() -> None:
    assert "--reload" not in COMPOSE_YML.read_text()
    assert "--reload" in COMPOSE_OVERRIDE_YML.read_text()
    for name in ("migrate", "planner", "analyzer", "transcriber", "web", "db"):
        assert "--reload" not in _override_block(name)


def test_overlay_workers_keep_their_commands() -> None:
    for name in WORKERS:
        assert "command:" not in _override_block(name)


def test_overlay_adds_no_extra_reloader_package() -> None:
    text = COMPOSE_OVERRIDE_YML.read_text()
    assert "watchfiles" not in text and "uvicorn[standard]" not in text


def test_overlay_web_runs_the_vite_dev_server_from_the_node_image() -> None:
    block = _override_block("web")
    assert "image: node:24-alpine" in block
    assert "working_dir: /home/node/app" in block
    # Check the command's own lines: a comment in the block mentions
    # --strictPort too, so a whole-block substring check cannot catch its removal.
    command = " ".join(
        ln.strip()
        for ln in _code_lines(block.split("command:", 1)[1].split("working_dir:")[0])
    )
    assert "npm ci" in command and "npm run dev" in command
    assert "--host 0.0.0.0" in command and "--port 5173" in command
    assert re.search(r"--strictPort(?=\s|'|$)", command)
    assert "API_PROXY_TARGET: http://api:8000" in block


def test_overlay_web_drops_the_nginx_build_with_reset() -> None:
    block = _override_block("web")
    assert "build: !reset null" in block
    assert "dockerfile" not in block
    # compose.yml itself keeps the build and its tag.
    assert "ops/Dockerfile.frontend" in _block("web")
    assert "image: ytdigest-web:latest" in _block("web")


def test_overlay_web_replaces_the_8080_publish_with_loopback_5173() -> None:
    block = _override_block("web")
    assert "ports: !override" in block
    assert '- "127.0.0.1:5173:5173"' in block
    assert "8080" not in block
    assert block.count("127.0.0.1:") == 1


def test_overlay_web_stays_off_the_internal_network() -> None:
    block = _override_block("web")
    assert "networks" not in block  # inherits [edge] from compose.yml
    assert "networks: [edge]" in _block("web")


def test_overlay_web_binds_only_source_files_so_node_modules_stays_in_the_container() -> (
    None
):
    block = _override_block("web")
    code = "\n".join(_code_lines(block))
    assert "/node_modules" not in code
    assert "./web:" not in code  # never the whole directory
    assert "web-node-modules" not in COMPOSE_OVERRIDE_YML.read_text()
    assert not _mounts(COMPOSE_OVERRIDE_YML.read_text(), "web", named_only=True)


def test_overlay_web_hands_its_workdir_to_node_and_drops_root() -> None:
    # The working_dir is created root-owned by the daemon; `npm ci` as `node`
    # would get EACCES, and running it as root would be needlessly privileged.
    block = _override_block("web")
    assert "chown node:node /home/node/app" in block
    assert "su node -c" in block


def _mounts(
    text: str, service: str, *, named_only: bool = False
) -> list[tuple[str, str]]:
    """(source, target) of every `- src:target[:opts]` volume entry of a service."""
    out: list[tuple[str, str]] = []
    in_volumes = False
    for ln in _code_lines(_service_block(text, service)):
        if re.match(r"^    volumes:", ln):
            in_volumes = True
            continue
        if in_volumes and re.match(r"^    \S", ln):
            in_volumes = False
        m = re.match(r"^      - (\S+?):(/[^:\s]*)", ln)
        if in_volumes and m:
            src, target = m.groups()
            if not named_only or not src.startswith((".", "/")):
                out.append((src, target))
    return out


def _nested_volume_mounts(compose_text: str, override_text: str) -> list[str]:
    """Named-volume targets that sit inside a host bind-mount's target path."""
    found: list[str] = []
    for service in re.findall(
        r"^  (\w+):\n", _top_level_section(override_text, "services"), re.MULTILINE
    ):
        mounts: list[tuple[str, str]] = []
        for text in (compose_text, override_text):
            try:
                mounts += _mounts(text, service)
            except StopIteration, ValueError, AssertionError:
                pass  # service absent from that file
        binds = [t for s, t in mounts if s.startswith((".", "/"))]
        for src, target in mounts:
            if src.startswith((".", "/")):
                continue
            for bind in binds:
                if target.startswith(bind.rstrip("/") + "/"):
                    found.append(f"{service}: {src} at {target} inside bind {bind}")
    return found


def test_no_named_volume_is_mounted_inside_a_host_bind_mount() -> None:
    # On a fresh checkout the daemon creates the nested mountpoint inside the
    # bind SOURCE as root, so the developer ends up with a root-owned directory
    # in the working tree (#59 QA: web/node_modules).
    assert (
        _nested_volume_mounts(COMPOSE_YML.read_text(), COMPOSE_OVERRIDE_YML.read_text())
        == []
    )


def test_the_nested_volume_check_catches_the_old_web_layout() -> None:
    old = (
        "services:\n  web:\n    volumes:\n"
        "      - ./web:/src\n      - web-node-modules:/src/node_modules\n"
    )
    assert _nested_volume_mounts("services:\n  web:\n    image: x\n", old) == [
        "web: web-node-modules at /src/node_modules inside bind /src"
    ]


def test_overlay_publishes_exactly_two_loopback_ports() -> None:
    text = COMPOSE_OVERRIDE_YML.read_text()
    ports = re.findall(r'^\s+- "([\d.:]+)"$', text, flags=re.MULTILINE)
    assert sorted(ports) == ["127.0.0.1:5173:5173", "127.0.0.1:5432:5432"]
    assert all(p.startswith("127.0.0.1:") for p in ports)


def test_overlay_db_publish_and_network_are_unchanged_from_7() -> None:
    block = _override_block("db")
    assert "networks: [internal, devhost]" in block
    assert '"127.0.0.1:5432:5432"' in block


def test_overlay_transcriber_whisper_model_defaults_to_tiny() -> None:
    assert "      WHISPER_MODEL: ${WHISPER_MODEL:-tiny}" in _override_block(
        "transcriber"
    )
    assert "tiny.en" not in "\n".join(_code_lines(COMPOSE_OVERRIDE_YML.read_text()))
    assert "WHISPER_MODEL: ${WHISPER_MODEL:-large-v3}" in _block("transcriber")


def test_overlay_only_new_environment_values_are_whisper_model_and_proxy() -> None:
    text = COMPOSE_OVERRIDE_YML.read_text()
    keys = re.findall(
        r"^\s+([A-Z][A-Z0-9_]+):", "\n".join(_code_lines(text)), re.MULTILINE
    )
    assert sorted(keys) == ["API_PROXY_TARGET", "WHISPER_MODEL"]
    for secret in ("ANTHROPIC", "SUMMARIZER", "PASSWORD", "POSTGRES"):
        assert secret not in "\n".join(_code_lines(text))


def test_overlay_header_says_dev_only_and_names_the_prod_command() -> None:
    header = COMPOSE_OVERRIDE_YML.read_text().split("\nservices:", 1)[0]
    assert "DEVELOPMENT ONLY" in header
    assert "automatically" in header
    assert "never used on the server" in header
    assert "docker compose -f compose.yml up -d" in header
    assert "seed of the dev overlay" not in header


def test_overlay_defines_only_the_services_it_merges_onto() -> None:
    lines = COMPOSE_OVERRIDE_YML.read_text().splitlines()
    names = set()
    for ln in lines[lines.index("services:") + 1 :]:
        if (
            ln.startswith("  ")
            and not ln.startswith("   ")
            and ln.strip().endswith(":")
        ):
            names.add(ln.strip()[:-1])
        elif ln and not ln.startswith(" "):
            break
    assert names == set(SERVICES)


def test_compose_yml_has_no_dev_only_content() -> None:
    code = "\n".join(_code_lines(COMPOSE_YML.read_text()))
    for needle in (
        "5173",
        "node:24-alpine",
        "API_PROXY_TARGET",
        "tiny",
        "devhost",
        "5432:",
        "./common",
        "./web",
        "!reset",
        "!override",
    ):
        assert needle not in code, needle
    assert not re.search(r"^\s*(include|extends):", code, re.MULTILINE)
    assert "override" not in code


def test_env_example_sets_no_compose_selection_variables() -> None:
    for line in (REPO_ROOT / ".env.example").read_text().splitlines():
        if line.startswith("#"):
            continue
        name = line.split("=", 1)[0].strip()
        assert name not in {
            "COMPOSE_FILE",
            "COMPOSE_PROFILES",
            "COMPOSE_PATH_SEPARATOR",
        }
        assert not name.startswith("COMPOSE_")


def test_transcriber_documents_an_optional_read_only_cookie_mount() -> None:
    # Issue #63: documented, but off by default.
    block = _block("transcriber")
    assert "YTDLP_COOKIE_FILE: /run/secrets/ytdlp-cookies.txt" in block
    assert "/run/secrets/ytdlp-cookies.txt:ro" in block
    active = [line for line in block.splitlines() if not line.lstrip().startswith("#")]
    assert not any("ytdlp-cookies" in line or "YTDLP_COOKIE" in line for line in active)
    for name in SERVICES:
        if name != "transcriber":
            assert "YTDLP_COOKIE_FILE" not in _block(name)
