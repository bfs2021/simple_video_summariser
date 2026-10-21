"""Builds the frontend image and checks it as a running container (issue #57).

Marked `integration` (needs Docker Desktop) and `image` (the build runs
`npm ci` against the registry and takes minutes): deselected unless run with
`pytest -m image`. Static checks of the same files live in
`test_frontend_image.py`.

The upstream is the stdlib stub `tests/fixtures/stub_api.py` in
`python:3.14-slim` on a per-test user-defined network under the alias `api`.
HTTP goes through `http.client`, which neither follows redirects nor merges
duplicate headers. Every container, network and tag created here is removed
afterwards, even on failure; waits poll with a deadline, never a fixed sleep.
"""

from __future__ import annotations

import gzip
import http.client
import json
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.image]

REPO_ROOT = Path(__file__).resolve().parent.parent
STUB = REPO_ROOT / "tests" / "fixtures" / "stub_api.py"
BASE = "nginx:1.27-alpine"
STUB_IMAGE = "python:3.14-slim"
CSP = (
    "default-src 'self'; frame-src https://www.youtube-nocookie.com; "
    "img-src 'self' https://i.ytimg.com data:; style-src 'self' 'unsafe-inline'; "
    "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)
RENDER_CSP = "default-src 'none'; style-src 'unsafe-inline'"
SECURITY = ("content-security-policy", "x-content-type-options", "referrer-policy")


def _docker(
    *args: str, timeout: int = 120, cwd: Path = REPO_ROOT
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _wait(
    condition: Callable[[], bool], *, deadline: float, interval: float = 0.2
) -> bool:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(interval)
    return condition()


@pytest.fixture(scope="module")
def tags() -> Iterator[dict[str, str]]:
    suffix = uuid.uuid4().hex[:8]
    names = {
        "web": f"ytdigest-web-test:{suffix}",
        "build": f"ytdigest-web-build-test:{suffix}",
    }
    try:
        yield names
    finally:
        for tag in names.values():
            _docker("image", "rm", "-f", tag)


@pytest.fixture(scope="module")
def image(tags: dict[str, str]) -> str:
    """The shipped image, built on a clean cache with the repo root as context."""
    result = _docker(
        "build",
        "--no-cache",
        "-f",
        "ops/Dockerfile.frontend",
        "-t",
        tags["web"],
        ".",
        timeout=1800,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return tags["web"]


@pytest.fixture(scope="module")
def build_image(tags: dict[str, str], image: str) -> str:
    result = _docker(
        "build",
        "-f",
        "ops/Dockerfile.frontend",
        "--target",
        "build",
        "-t",
        tags["build"],
        ".",
        timeout=1800,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return tags["build"]


def _sh(image: str, script: str) -> subprocess.CompletedProcess[str]:
    return _docker("run", "--rm", "--entrypoint", "sh", image, "-c", script)


def _files(image: str, root: str) -> list[str]:
    result = _sh(image, f"cd {root} && find . -type f | sort")
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


# ---- build and image contents ----


def test_build_stage_bundle_has_api_as_its_api_base(build_image: str) -> None:
    result = _sh(build_image, "cat /src/dist/assets/*.js")
    assert result.returncode == 0, result.stderr
    assert "`/api`" in result.stdout or '"/api"' in result.stdout


def test_build_output_has_hashed_assets_and_no_source_maps(build_image: str) -> None:
    files = _files(build_image, "/src/dist")
    assert "./index.html" in files
    assert any(re.fullmatch(r"\./assets/.+-[\w-]{8}\.js", f) for f in files)
    assert not [f for f in files if f.endswith(".map")]


def test_no_node_toolchain_in_the_shipped_image(image: str) -> None:
    assert _sh(image, "command -v node npm npx").stdout.strip() == ""
    assert _sh(image, "find / -xdev -name node_modules").stdout.strip() == ""
    assert _sh(image, "test ! -e /src").returncode == 0


def test_html_directory_is_exactly_the_build_output(
    image: str, build_image: str
) -> None:
    built = _files(build_image, "/src/dist")
    shipped = _files(image, "/usr/share/nginx/html")
    assert shipped == built
    assert "./50x.html" not in shipped
    assert not [f for f in shipped if f.endswith(".map")]


def test_conf_d_holds_only_default_conf_identical_to_ops_nginx_conf(
    image: str,
) -> None:
    assert _files(image, "/etc/nginx/conf.d") == ["./default.conf"]
    shipped = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "cat", image]
        + ["/etc/nginx/conf.d/default.conf"],
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert shipped.returncode == 0
    assert shipped.stdout == (REPO_ROOT / "ops" / "nginx.conf").read_bytes()


def test_config_is_valid_without_a_resolvable_api_host(image: str) -> None:
    result = _docker("run", "--rm", "--network", "none", image, "nginx", "-t")
    assert result.returncode == 0, result.stderr


def test_nginx_is_the_1_27_line(image: str) -> None:
    assert re.search(
        r"nginx/1\.27\.", _docker("run", "--rm", image, "nginx", "-v").stderr
    )


def test_image_is_at_most_10_mb_larger_than_the_base(image: str) -> None:
    pulled = _docker("pull", BASE, timeout=300)
    assert pulled.returncode == 0, pulled.stderr

    def size(tag: str) -> int:
        return int(_docker("image", "inspect", "-f", "{{.Size}}", tag).stdout)

    assert size(image) <= size(BASE) + 10 * 1024 * 1024


def test_image_has_no_healthcheck(image: str) -> None:
    config = json.loads(_docker("image", "inspect", image).stdout)[0]["Config"]
    assert not config.get("Healthcheck")


def test_docker_stop_exits_zero_within_the_default_grace_period(image: str) -> None:
    name = f"ytdigest-web-stop-{uuid.uuid4().hex[:8]}"
    try:
        run = _docker("run", "-d", "--name", name, "--network", "none", image)
        assert run.returncode == 0, run.stderr

        def up() -> bool:
            logs = _docker("logs", name)
            return "start worker processes" in logs.stdout + logs.stderr

        assert _wait(up, deadline=15)
        t0 = time.monotonic()
        assert _docker("stop", name, timeout=30).returncode == 0
        assert time.monotonic() - t0 < 10
        state = json.loads(_docker("inspect", name).stdout)[0]["State"]
        assert state["ExitCode"] == 0
    finally:
        _docker("rm", "-f", name)


def test_layer_order_keeps_npm_ci_cached_when_a_source_file_changes(
    tmp_path: Path,
) -> None:
    ctx = _scratch_context(tmp_path)
    tag = f"ytdigest-web-layer-test:{uuid.uuid4().hex[:8]}"
    cmd = ["build", "--progress=plain", "-f", "ops/Dockerfile.frontend", "-t", tag, "."]
    try:
        assert _docker(*cmd, cwd=ctx, timeout=1800).returncode == 0
        main = ctx / "web" / "src" / "main.tsx"
        main.write_text(main.read_text() + "\n// touched by the layer-order test\n")
        second = _docker(*cmd, cwd=ctx, timeout=1800)
        assert second.returncode == 0, second.stderr[-4000:]
        log = second.stderr
        step = re.search(r"#(\d+) \[build \d+/\d+\] RUN npm ci", log)
        assert step, log[-3000:]
        assert re.search(rf"#{step.group(1)} CACHED", log), "npm ci was re-run"
    finally:
        _docker("image", "rm", "-f", tag)


def test_lockfile_is_authoritative(tmp_path: Path) -> None:
    ctx = _scratch_context(tmp_path)
    package = ctx / "web" / "package.json"
    manifest = json.loads(package.read_text())
    # Add a dependency the lockfile has never heard of. Unlike bumping a pin to
    # a version that may not exist (npm then fails with ETARGET, a registry
    # lookup, proving nothing about the lockfile), this can only be caught by
    # npm ci's sync check, which runs before any registry access.
    name, version = "left-pad", "1.3.0"
    manifest["dependencies"][name] = version
    package.write_text(json.dumps(manifest, indent=2) + "\n")
    tag = f"ytdigest-web-lock-test:{uuid.uuid4().hex[:8]}"
    try:
        result = _docker(
            "build", "--progress=plain", "-f", "ops/Dockerfile.frontend", "-t", tag, ".",
            cwd=ctx, timeout=1800,
        )  # fmt: skip
        assert result.returncode != 0
        out = result.stdout + result.stderr
        # npm ci's sync check, text captured from a real run (npm 11.19.0 in
        # the image): "npm error code EUSAGE", then "`npm ci` can only install
        # packages when your package.json and package-lock.json or
        # npm-shrinkwrap.json are in sync ..." and "Missing: left-pad@1.3.0
        # from lock file". If npm rewords it, update these patterns.
        assert "ETARGET" not in out, out[-3000:]
        assert "npm error code EUSAGE" in out, out[-3000:]
        assert re.search(
            r"can only install packages when your package\.json and "
            r"package-lock\.json or npm-shrinkwrap\.json are in sync",
            out,
        ), out[-3000:]
        assert f"Missing: {name}@{version} from lock file" in out, out[-3000:]
    finally:
        _docker("image", "rm", "-f", tag)


def _scratch_context(tmp_path: Path) -> Path:
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    shutil.copytree(
        REPO_ROOT / "web",
        ctx / "web",
        ignore=shutil.ignore_patterns("node_modules", "dist", ".env*"),
    )
    shutil.copytree(REPO_ROOT / "ops", ctx / "ops")
    shutil.copy(REPO_ROOT / ".dockerignore", ctx / ".dockerignore")
    return ctx


# ---- a running container ----


@dataclass
class Response:
    status: int
    headers: list[tuple[str, str]]
    body: bytes

    def all(self, name: str) -> list[str]:
        return [v for k, v in self.headers if k.lower() == name.lower()]

    def one(self, name: str) -> str:
        values = self.all(name)
        assert len(values) == 1, f"{name}: {values}"
        return values[0]


@dataclass
class Stack:
    network: str
    image: str
    web: str
    port: int

    def get(
        self,
        path: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        timeout: float = 15,
    ) -> Response:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.putrequest(method, path, skip_accept_encoding=True)
            for key, value in (headers or {}).items():
                conn.putheader(key, value)
            if body is not None:
                conn.putheader("Content-Length", str(len(body)))
            conn.endheaders(body)
            resp = conn.getresponse()
            return Response(resp.status, resp.getheaders(), resp.read())
        finally:
            conn.close()

    def start_api(self) -> str:
        name = f"ytdigest-stub-{uuid.uuid4().hex[:8]}"
        created = _docker(
            "create", "--name", name, "--network", self.network,
            "--network-alias", "api", STUB_IMAGE, "python", "-u", "/stub_api.py",
        )  # fmt: skip
        assert created.returncode == 0, created.stderr
        assert _docker("cp", str(STUB), f"{name}:/stub_api.py").returncode == 0
        started = _docker("start", name)
        assert started.returncode == 0, started.stderr
        return name

    def start_address_holder(self) -> str:
        name = f"ytdigest-address-holder-{uuid.uuid4().hex[:8]}"
        started = _docker(
            "run", "-d", "--name", name, "--network", self.network,
            STUB_IMAGE, "sleep", "300",
        )  # fmt: skip
        assert started.returncode == 0, started.stderr
        return name

    def ip_of(self, container: str) -> str:
        out = _docker(
            "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            container,
        )  # fmt: skip
        return out.stdout.strip()

    def wait_for_api(self, *, deadline: float = 30) -> bool:
        def up() -> bool:
            try:
                return self.get("/api/healthz").status == 200
            except OSError:
                return False

        return _wait(up, deadline=deadline)

    def echo(self, path: str, **kwargs: object) -> tuple[Response, dict[str, object]]:
        resp = self.get(path, **kwargs)  # type: ignore[arg-type]
        return resp, json.loads(resp.body)

    def logs(self) -> str:
        out = _docker("logs", self.web)
        return out.stdout + out.stderr


@pytest.fixture
def make_stack(image: str) -> Iterator[Callable[[], Stack]]:
    networks: list[str] = []
    containers: list[str] = []

    def make() -> Stack:
        suffix = uuid.uuid4().hex[:8]
        network = f"ytdigest-web-net-{suffix}"
        assert _docker("network", "create", network).returncode == 0
        networks.append(network)
        web = f"ytdigest-web-{suffix}"
        containers.append(web)
        run = _docker(
            "run", "-d", "--name", web, "--network", network,
            "-p", "127.0.0.1::80", image,
        )  # fmt: skip
        assert run.returncode == 0, run.stderr
        port_out = _docker("port", web, "80/tcp").stdout.split()[0]
        stack = Stack(network, image, web, int(port_out.rsplit(":", 1)[1]))
        started = _wait(lambda: _serves(stack), deadline=20)
        assert started, stack.logs()
        return stack

    try:
        yield make
    finally:
        for container in reversed(containers):
            _docker("rm", "-f", container)
        # Remove the stub containers still attached to each network.
        for network in networks:
            members = _docker(
                "network", "inspect", "-f",
                "{{range .Containers}}{{.Name}} {{end}}", network,
            )  # fmt: skip
            for member in members.stdout.split():
                _docker("rm", "-f", member)
            _docker("network", "rm", network)


def _serves(stack: Stack) -> bool:
    try:
        return stack.get("/").status == 200
    except OSError:
        return False


@pytest.fixture
def stack(make_stack: Callable[[], Stack]) -> Stack:
    s = make_stack()
    s.start_api()
    assert s.wait_for_api(), s.logs()
    return s


def _index(stack: Stack) -> bytes:
    resp = stack.get("/index.html")
    assert resp.status == 200
    return resp.body


def _asset_paths(stack: Stack) -> list[str]:
    files = _files(stack.image, "/usr/share/nginx/html/assets")
    return ["/assets/" + f.removeprefix("./") for f in files]


def test_entrypoint_leaves_the_mounted_config_byte_identical(stack: Stack) -> None:
    out = subprocess.run(
        ["docker", "exec", stack.web, "cat", "/etc/nginx/conf.d/default.conf"],
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert out.stdout == (REPO_ROOT / "ops" / "nginx.conf").read_bytes()


# ---- SPA serving and caching ----


@pytest.mark.parametrize("path", ["/", "/index.html"])
def test_root_serves_the_built_index_html(stack: Stack, path: str) -> None:
    resp = stack.get(path)
    assert resp.status == 200
    assert resp.one("Content-Type").startswith("text/html")
    assert b'<div id="root">' in resp.body


@pytest.mark.parametrize(
    "path",
    [
        "/videos/-wNyEUrxzFU",
        "/search?q=a%20b",
        "/ops",
        "/some/unknown/path",
        "/apiary",
        "/api-docs",
    ],
)
def test_deep_links_and_api_lookalikes_serve_the_app(stack: Stack, path: str) -> None:
    resp = stack.get(path)
    assert resp.status == 200
    assert resp.body == _index(stack)


def test_html_is_no_cache_with_an_etag_that_revalidates(stack: Stack) -> None:
    for path in ("/", "/index.html", "/videos/abc12345678"):
        resp = stack.get(path)
        assert resp.all("Cache-Control") == ["no-cache"]
    etag = stack.get("/index.html").one("ETag")
    again = stack.get("/index.html", headers={"If-None-Match": etag})
    assert again.status == 304


def test_every_asset_is_immutable_with_the_right_type(stack: Stack) -> None:
    paths = _asset_paths(stack)
    assert paths
    for path in paths:
        resp = stack.get(path)
        assert resp.status == 200, path
        assert resp.all("Cache-Control") == ["public, max-age=31536000, immutable"]
        kind = resp.one("Content-Type").split(";")[0]
        if path.endswith(".js"):
            assert kind in {"text/javascript", "application/javascript"}
        elif path.endswith(".css"):
            assert kind == "text/css"


def test_missing_asset_is_a_real_404_that_is_not_cached(stack: Stack) -> None:
    resp = stack.get("/assets/does-not-exist-abc123.js")
    assert resp.status == 404
    assert b'<div id="root">' not in resp.body
    cache = " ".join(resp.all("Cache-Control"))
    assert "immutable" not in cache and "31536000" not in cache


def test_directory_redirect_is_relative(stack: Stack) -> None:
    resp = stack.get("/assets")
    assert resp.status == 301
    assert resp.one("Location").startswith("/")
    assert "://" not in resp.one("Location")


def test_server_header_and_error_pages_carry_no_version(stack: Stack) -> None:
    assert stack.get("/").one("Server") == "nginx"
    missing = stack.get("/assets/nope.js")
    assert missing.one("Server") == "nginx"
    assert b"nginx/1" not in missing.body


def test_gzip_for_large_text_assets_and_not_without_the_header(stack: Stack) -> None:
    big = [p for p in _asset_paths(stack) if p.endswith(".js")]
    assert big
    plain = stack.get(big[0])
    assert plain.all("Content-Encoding") == []
    assert len(plain.body) > 1024
    zipped = stack.get(big[0], headers={"Accept-Encoding": "gzip"})
    assert zipped.all("Content-Encoding") == ["gzip"]
    assert "Accept-Encoding" in zipped.one("Vary")
    assert gzip.decompress(zipped.body) == plain.body
    html = stack.get("/", headers={"Accept-Encoding": "gzip"})
    assert html.all("Content-Encoding") == ["gzip"]
    assert html.all("Vary")


# ---- security headers ----


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/index.html",
        "/videos/-wNyEUrxzFU",
        "/assets/does-not-exist-abc123.js",
        "/assets",
        "/api",
    ],
)
def test_security_headers_once_each_with_the_exact_csp(stack: Stack, path: str) -> None:
    resp = stack.get(path)
    assert resp.one("Content-Security-Policy") == CSP
    assert resp.one("X-Content-Type-Options") == "nosniff"
    assert resp.one("Referrer-Policy") == "no-referrer"
    assert resp.all("Strict-Transport-Security") == []


def test_security_headers_on_every_asset(stack: Stack) -> None:
    for path in _asset_paths(stack):
        resp = stack.get(path)
        assert resp.one("Content-Security-Policy") == CSP
        assert resp.one("X-Content-Type-Options") == "nosniff"
        assert resp.one("Referrer-Policy") == "no-referrer"


# ---- the /api/ reverse proxy ----


@pytest.mark.parametrize(
    ("client", "upstream"),
    [
        ("/api/healthz", "/healthz"),
        ("/api/", "/"),
        ("/api/videos/abc12345678/render", "/videos/abc12345678/render"),
        (
            "/api/search?q=caf%C3%A9%20x&limit=5&offset=0",
            "/search?q=caf%C3%A9%20x&limit=5&offset=0",
        ),
    ],
)
def test_prefix_is_stripped_and_the_query_arrives_byte_for_byte(
    stack: Stack, client: str, upstream: str
) -> None:
    resp, echoed = stack.echo(client)
    assert resp.status == 200
    assert echoed["path"] == upstream
    assert echoed["method"] == "GET"


def test_metrics_is_not_published_but_the_rest_of_the_api_is(stack: Stack) -> None:
    # #60 grooming: /api/metrics must be a 404 from nginx itself, never the
    # API's route and never the SPA; its neighbours still proxy normally.
    for path in (
        "/api/metrics",
        "/api/metrics/",
        "/api/metrics?x=1",
        "/api/metrics/sub",
    ):
        resp = stack.get(path)
        assert resp.status == 404
        assert b'<div id="root">' not in resp.body
        assert b"stub" not in resp.body and b'"method"' not in resp.body
        assert resp.one("Content-Security-Policy") == CSP
        assert resp.one("X-Content-Type-Options") == "nosniff"
        assert resp.one("Referrer-Policy") == "no-referrer"
    ok, echoed = stack.echo("/api/healthz")
    assert ok.status == 200
    assert echoed["path"] == "/healthz"


@pytest.mark.xfail(
    strict=True,
    reason="#186 (follow-up to #57): the prefix-strip map reads the raw $request_uri "
    "while location matching uses the normalized URI",
)
@pytest.mark.parametrize("client", ["/%61pi/healthz", "/x/../api/healthz"])
def test_non_canonical_api_prefixes_reach_the_same_upstream_path(
    stack: Stack, client: str
) -> None:
    _, echoed = stack.echo(client)
    assert echoed["path"] == "/healthz"


def test_api_without_slash_is_never_the_spa(stack: Stack) -> None:
    resp = stack.get("/api")
    assert resp.status == 301
    assert resp.one("Location") == "/api/"
    assert b'<div id="root">' not in resp.body


def test_post_body_and_content_type_reach_the_upstream(stack: Stack) -> None:
    payload = json.dumps({"url": "https://youtu.be/x", "n": 1}).encode()
    resp, echoed = stack.echo(
        "/api/videos",
        method="POST",
        headers={"Content-Type": "application/json"},
        body=payload,
    )
    assert resp.status == 200
    assert echoed["method"] == "POST"
    assert echoed["path"] == "/videos"
    assert echoed["body"].encode("latin-1") == payload  # type: ignore[attr-defined]
    headers = {k.lower(): v for k, v in echoed["headers"]}  # type: ignore[attr-defined]
    assert headers["content-type"] == "application/json"


@pytest.mark.parametrize("status", [404, 503])
def test_upstream_errors_pass_through_unchanged(stack: Stack, status: int) -> None:
    resp = stack.get(f"/api/status/{status}")
    assert resp.status == status
    assert json.loads(resp.body) == {"error": f"canned-{status}"}


def test_api_headers_pass_through_exactly_once_and_nothing_is_added(
    stack: Stack,
) -> None:
    render = stack.get("/api/videos/abc12345678/render")
    assert render.all("Content-Security-Policy") == [RENDER_CSP]
    assert render.all("X-Content-Type-Options") == ["nosniff"]
    assert render.all("Referrer-Policy") == ["no-referrer"]
    plain = stack.get("/api/healthz")
    for name in (*SECURITY, "cache-control", "strict-transport-security"):
        assert plain.all(name) == [], name
    assert plain.all("X-Request-Id")  # the upstream's own header passes through


def test_api_bodies_are_not_gzipped(stack: Stack) -> None:
    resp = stack.get("/api/healthz", headers={"Accept-Encoding": "gzip"})
    assert resp.all("Content-Encoding") == []
    assert json.loads(resp.body)["path"] == "/healthz"


def test_request_id_is_nginx_generated_fresh_and_logged(stack: Stack) -> None:
    seen = []
    for _ in range(2):
        _, echoed = stack.echo("/api/healthz", headers={"X-Request-Id": "evil"})
        headers = {k.lower(): v for k, v in echoed["headers"]}  # type: ignore[attr-defined]
        seen.append(headers["x-request-id"])
    assert all(re.fullmatch(r"[0-9a-f]{32}", rid) for rid in seen)
    assert seen[0] != seen[1]
    assert "evil" not in seen
    for rid in seen:
        assert _wait(_logged(stack, rid), deadline=10)


def _logged(stack: Stack, rid: str) -> Callable[[], bool]:
    return lambda: f"rid={rid}" in stack.logs()


# ---- upstream lifecycle ----


def test_spa_serves_and_api_fails_fast_while_the_upstream_is_missing(
    make_stack: Callable[[], Stack],
) -> None:
    stack = make_stack()
    assert stack.get("/").status == 200
    t0 = time.monotonic()
    resp = stack.get("/api/healthz", timeout=15)
    assert resp.status in (502, 504)
    assert time.monotonic() - t0 < 10
    assert b'<div id="root">' not in resp.body
    stack.start_api()
    assert stack.wait_for_api(), stack.logs()  # no restart of `web`


def test_nginx_follows_the_api_to_a_new_address(
    make_stack: Callable[[], Stack],
) -> None:
    stack = make_stack()
    first = stack.start_api()
    try:
        assert stack.wait_for_api(), stack.logs()
        old_ip = stack.ip_of(first)

        stopped = _docker("stop", first)
        assert stopped.returncode == 0, stopped.stderr

        def unavailable() -> bool:
            try:
                return stack.get("/api/healthz").status in (502, 504)
            except OSError:
                return False

        assert _wait(unavailable, deadline=30), stack.logs()
        holder = stack.start_address_holder()
        assert stack.ip_of(holder) == old_ip
        second = stack.start_api()
        new_ip = stack.ip_of(second)
        assert new_ip and new_ip != old_ip
        assert stack.wait_for_api(deadline=30), stack.logs()
    finally:
        _docker("rm", "-f", first)
