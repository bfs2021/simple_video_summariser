"""Static checks of the frontend image inputs (issue #57, architecture.md 9, 11.3).

No Docker, no Node and no nginx binary: these read `ops/Dockerfile.frontend`,
`ops/nginx.conf` and `.dockerignore` as text, so they run under
`pytest -m "not integration"`. `test_frontend_image_docker.py` builds the image
and checks the running container.
"""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

from tests.test_backend_image import (
    REPO_ROOT,
    _copy_sources,
    _dockerignore_lines,
    _instructions,
)

DOCKERFILE = REPO_ROOT / "ops" / "Dockerfile.frontend"
NGINX_CONF = REPO_ROOT / "ops" / "nginx.conf"

EXPECTED_CSP: dict[str, list[str]] = {
    "default-src": ["'self'"],
    "frame-src": ["https://www.youtube-nocookie.com"],
    "img-src": ["'self'", "https://i.ytimg.com", "data:"],
    "style-src": ["'self'", "'unsafe-inline'"],
    "object-src": ["'none'"],
    "base-uri": ["'self'"],
    "form-action": ["'self'"],
    "frame-ancestors": ["'none'"],
}
SECURITY_HEADERS = (
    "Content-Security-Policy",
    "X-Content-Type-Options",
    "Referrer-Policy",
)


def _steps() -> list[tuple[str, str]]:
    return _instructions(DOCKERFILE)


def _stages() -> list[list[tuple[str, str]]]:
    stages: list[list[tuple[str, str]]] = []
    for step in _steps():
        if step[0] == "FROM":
            stages.append([])
        stages[-1].append(step)
    return stages


# ---- nginx.conf parsing: just enough brace matching for this one file ----


def _conf_text() -> str:
    # Comments dropped; a `#` inside a quoted string is not used in this file.
    return "\n".join(
        line.split("#", 1)[0] for line in NGINX_CONF.read_text().splitlines()
    )


def _block(text: str, header: str) -> str:
    """Body of the first `<header> {` block, by brace counting."""
    match = re.search(re.escape(header) + r"\s*\{", text)
    assert match, f"no `{header}` block in nginx.conf"
    depth, start = 1, match.end()
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start:i]
    raise AssertionError(f"unbalanced braces after `{header}`")


def _server() -> str:
    return _block(_conf_text(), "server")


def _location(header: str) -> str:
    return _block(_server(), f"location {header}")


def _directives(body: str) -> list[str]:
    # Quote-aware: the CSP value itself contains semicolons.
    parts = re.findall(r'(?:"[^"]*"|[^;"])+', body)
    return [d.strip() for d in parts if d.strip()]


def _add_headers(body: str) -> list[tuple[str, str, bool]]:
    found = []
    for d in _directives(body):
        m = re.match(
            r'add_header\s+(\S+)\s+("(?:[^"]*)"|\S+)(\s+always)?$', d, re.DOTALL
        )
        if m:
            found.append((m.group(1), m.group(2).strip('"'), bool(m.group(3))))
    return found


def _parse_csp(value: str) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for part in value.split(";"):
        tokens = part.split()
        if tokens:
            assert tokens[0] not in parsed, f"duplicate directive {tokens[0]}"
            parsed[tokens[0]] = tokens[1:]
    return parsed


def _all_locations() -> list[str]:
    """Every `location` header in the server block, e.g. "^~ /api/metrics"."""
    found = [
        " ".join(m.group(1).split())
        for m in re.finditer(r"location\s+([^{]+?)\s*\{", _server())
    ]
    assert found, "no location blocks parsed from nginx.conf"
    return found


# Derived from nginx.conf, so a location added later is checked automatically:
# the proxy is the one with a proxy_pass, every other location nginx answers
# itself and must carry the security headers.
PROXY_LOCATIONS = [h for h in _all_locations() if "proxy_pass" in _location(h)]
OWN_LOCATIONS = [h for h in _all_locations() if h not in PROXY_LOCATIONS]


# ---- Dockerfile ----


def test_stages_are_node_24_alpine_build_then_unprivileged_nginx_1_30_alpine() -> None:
    froms = [args for name, args in _steps() if name == "FROM"]
    assert froms == [
        "node:24-alpine AS build",
        "nginx/nginx-unprivileged:1.30-alpine",
    ]


def test_build_installs_with_npm_ci_and_no_escape_hatches() -> None:
    build = _stages()[0]
    runs = [a for n, a in build if n == "RUN"]
    assert any(re.fullmatch(r"npm ci", r) for r in runs)
    text = " ".join(runs)
    for forbidden in ("npm install", "--force", "--legacy-peer-deps", "--omit=dev"):
        assert forbidden not in text
    assert "NODE_ENV" not in " ".join(a for n, a in build if n in {"ENV", "ARG"})
    assert "NODE_ENV" not in text


def test_manifests_are_copied_and_installed_before_the_rest_of_web() -> None:
    build = _stages()[0]
    copies = [(i, _copy_sources(a)) for i, (n, a) in enumerate(build) if n == "COPY"]
    manifest = next(
        i
        for i, srcs in copies
        if "web/package.json" in srcs and "web/package-lock.json" in srcs
    )
    rest = next(i for i, srcs in copies if "web/" in srcs)
    ci = next(i for i, (n, a) in enumerate(build) if n == "RUN" and a == "npm ci")
    assert manifest < ci < rest


def test_vite_api_base_arg_defaults_to_api_and_precedes_the_build() -> None:
    build = _stages()[0]
    arg = next(
        i for i, (n, a) in enumerate(build) if n == "ARG" and a == "VITE_API_BASE=/api"
    )
    run = next(
        i for i, (n, a) in enumerate(build) if n == "RUN" and a == "npm run build"
    )
    assert arg < run


def test_build_stage_needs_no_python_or_backend() -> None:
    text = " ".join(a for n, a in _stages()[0] if n in {"RUN", "COPY"})
    for word in ("pip", "python", "apk add", "requirements"):
        assert word not in text


def test_runtime_copies_only_the_build_output_and_nginx_conf() -> None:
    runtime = _stages()[1]
    copies = [a for n, a in runtime if n in {"COPY", "ADD"}]
    assert len(copies) == 2
    assert "--from=build" in copies[0]
    assert _copy_sources(copies[0]) == ["/src/dist"]
    assert copies[0].split()[-1] == "/usr/share/nginx/html"
    assert _copy_sources(copies[1]) == ["ops/nginx.conf"]
    assert copies[1].split()[-1] == "/etc/nginx/conf.d/default.conf"


def test_base_image_html_is_removed_before_the_build_output_is_copied() -> None:
    runtime = _stages()[1]
    rm = next(
        i
        for i, (n, a) in enumerate(runtime)
        if n == "RUN" and "rm -rf /usr/share/nginx/html/*" in a
    )
    copy = next(
        i for i, (n, a) in enumerate(runtime) if n == "COPY" and "--from=build" in a
    )
    assert rm < copy


def test_runtime_has_no_node_toolchain() -> None:
    text = " ".join(a for n, a in _stages()[1])
    for word in ("node", "npm", "npx", "node_modules"):
        assert word not in text


def test_image_has_no_healthcheck() -> None:
    assert all(name != "HEALTHCHECK" for name, _ in _steps())


def test_image_has_no_shell_form_entrypoint_or_cmd() -> None:
    # The base image's exec-form entrypoint keeps nginx as PID 1; nothing here
    # may replace it with a shell form that swallows the stop signal.
    for name, args in _steps():
        if name in {"ENTRYPOINT", "CMD"}:
            assert args.startswith("["), f"shell-form {name} swallows SIGTERM"


def test_runtime_stage_ends_as_a_numeric_non_root_user() -> None:
    # Root is allowed only for the build-time `rm -rf` of the base image's
    # pages; the user the container starts as must be numeric and non-zero.
    users = [args for name, args in _stages()[1] if name == "USER"]
    assert users, "the runtime stage must declare a USER"
    assert users[-1].isdigit() and int(users[-1]) != 0
    steps = _stages()[1]
    last_user = max(i for i, (n, _) in enumerate(steps) if n == "USER")
    assert all(n != "RUN" for n, _ in steps[last_user:])


def test_build_stage_declares_no_user() -> None:
    assert all(name != "USER" for name, _ in _stages()[0])


# ---- .dockerignore ----


def test_dockerignore_excludes_web_dist() -> None:
    assert "web/dist" in _dockerignore_lines()


def test_dockerignore_still_excludes_node_modules_and_env_files() -> None:
    lines = _dockerignore_lines()
    assert "**/node_modules" in lines
    assert "**/.env*" in lines


@pytest.mark.parametrize(
    "path",
    [
        "web",
        "web/package-lock.json",
        "web/openapi.json",
        "web/src/api/schema.d.ts",
        "web/index.html",
        "ops/nginx.conf",
    ],
)
def test_dockerignore_keeps_what_the_build_needs(path: str) -> None:
    for line in _dockerignore_lines():
        assert not line.startswith("!")
        if line.startswith("**/"):
            continue  # **/node_modules, **/.env*, **/*.md: none match these paths
        if "*" in line:
            assert not fnmatchcase(path, line), f"{line} excludes {path}"
        else:
            pattern = line.rstrip("/")
            assert path != pattern and not path.startswith(pattern + "/"), (
                f"{line} excludes {path}"
            )


def test_dockerignore_web_dist_is_appended_after_the_55_entries() -> None:
    lines = _dockerignore_lines()
    assert lines[-1] == "web/dist"
    assert lines.index(".git") < lines.index("web/dist")
    assert lines.index("*:Zone.Identifier") < lines.index("web/dist")


# ---- nginx.conf: serving and caching ----


def test_server_listens_on_8080_and_roots_at_the_html_directory() -> None:
    server = _server()
    assert "listen 8080;" in server
    assert not re.search(r"listen\s+(\S+:)?80\s*;", server)
    assert "root /usr/share/nginx/html;" in server


def test_spa_fallback_serves_index_html_for_unknown_paths() -> None:
    assert "try_files $uri /index.html;" in _location("/")


def test_html_is_never_cached_but_revalidated() -> None:
    headers = _add_headers(_location("/"))
    assert [v for k, v, _ in headers if k == "Cache-Control"] == ["no-cache"]


def test_assets_are_immutable_and_have_no_spa_fallback() -> None:
    body = _location("/assets/")
    assert [v for k, v, _ in _add_headers(body) if k == "Cache-Control"] == [
        "public, max-age=31536000, immutable"
    ]
    assert "try_files" not in body and "index.html" not in body


def test_asset_cache_control_is_not_sent_on_errors() -> None:
    # `always` would make a 404 immutable for a year.
    for k, _, always in _add_headers(_location("/assets/")):
        if k == "Cache-Control":
            assert not always


def test_no_expires_directive_that_would_add_a_second_cache_control() -> None:
    assert not re.search(r"\bexpires\b", _conf_text())


def test_assets_without_slash_redirects_relatively() -> None:
    assert "return 301 /assets/;" in _location("= /assets")


def test_absolute_redirect_and_server_tokens_are_off() -> None:
    server = _server()
    assert "absolute_redirect off;" in server
    assert "server_tokens off;" in server


def test_gzip_covers_text_types_and_varies() -> None:
    server = _server()
    assert "gzip on;" in server
    assert "gzip_vary on;" in server
    assert "gzip_proxied off;" in server or "gzip_proxied" not in server
    types = re.search(r"gzip_types\s+([^;]+);", server)
    assert types
    assert {
        "text/css",
        "application/json",
        "image/svg+xml",
    } <= set(types.group(1).split())
    assert {"text/javascript", "application/javascript"} <= set(types.group(1).split())


def test_apiary_and_api_docs_are_not_matched_by_the_proxy_location() -> None:
    # A prefix location ending in a slash does not match /apiary or /api-docs.
    assert re.search(r"location /api/ \{", _server())
    assert "location /api " not in _server()
    assert not re.search(r"location\s+(~\*?|\^~)?\s*/api[^/]", _server())


def test_api_without_slash_never_serves_the_spa() -> None:
    body = _location("= /api")
    assert "return 301 /api/" in body
    assert "index.html" not in body


# ---- nginx.conf: security headers ----


@pytest.mark.parametrize("header", OWN_LOCATIONS)
def test_every_own_location_sends_the_csp_exactly_once(header: str) -> None:
    csps = [
        v
        for k, v, _ in _add_headers(_location(header))
        if k == "Content-Security-Policy"
    ]
    assert len(csps) == 1
    assert _parse_csp(csps[0]) == EXPECTED_CSP


@pytest.mark.parametrize("header", OWN_LOCATIONS)
@pytest.mark.parametrize("name", SECURITY_HEADERS)
def test_security_headers_are_sent_once_and_with_always(header: str, name: str) -> None:
    found = [
        (v, always) for k, v, always in _add_headers(_location(header)) if k == name
    ]
    assert len(found) == 1
    assert found[0][1], f"{name} lacks `always` in location {header}"


def test_locations_are_derived_and_include_the_known_ones() -> None:
    assert PROXY_LOCATIONS == ["/api/"]
    assert {"/assets/", "= /assets", "/", "= /api", "^~ /api/metrics"} <= set(
        OWN_LOCATIONS
    )


def test_metrics_is_blocked_before_the_proxy_location() -> None:
    text = _server()
    assert text.index("location ^~ /api/metrics") < text.index("location /api/ ")
    assert "return 404;" in _location("^~ /api/metrics")
    assert "proxy_pass" not in _location("^~ /api/metrics")


def test_metrics_block_covers_the_whole_prefix_not_just_the_exact_path() -> None:
    # An exact `=` match is bypassable with /api/metrics/ or /api/metrics/x,
    # which fall through to the proxy. Simulate nginx's choice: an exact match
    # wins, else a ^~ prefix wins, else the longest prefix.
    locs = _all_locations()

    def chosen(uri: str) -> str:
        exact = [h for h in locs if h.startswith("= ") and h[2:] == uri]
        if exact:
            return exact[0]
        prefixes = [
            (h.removeprefix("^~ "), h)
            for h in locs
            if not h.startswith("= ") and uri.startswith(h.removeprefix("^~ "))
        ]
        return max(prefixes, key=lambda p: len(p[0]))[1]

    for uri in (
        "/api/metrics",
        "/api/metrics/",
        "/api/metrics/anything",
        "/api/metrics/a/b",
    ):
        assert chosen(uri) == "^~ /api/metrics", uri
    assert chosen("/api/healthz") == "/api/"
    assert "^~" in " ".join(locs)
    assert not [h for h in locs if h.startswith("~")], "regex locations need review"


def test_proxy_locations_carry_no_security_header() -> None:
    for header in PROXY_LOCATIONS:
        assert _add_headers(_location(header)) == []


def test_static_security_header_values() -> None:
    for header in OWN_LOCATIONS:
        values = {k: v for k, v, _ in _add_headers(_location(header))}
        assert values["X-Content-Type-Options"] == "nosniff"
        assert values["Referrer-Policy"] == "no-referrer"


def test_csp_names_no_other_origin_and_nothing_loosened() -> None:
    csp = next(
        v for k, v, _ in _add_headers(_location("/")) if k == "Content-Security-Policy"
    )
    parsed = _parse_csp(csp)
    assert "script-src" not in parsed
    assert "www.youtube.com" not in csp
    assert "unsafe-eval" not in csp
    assert "*" not in csp
    assert "http:" not in csp and not re.search(r"(?<![\w.])https:(?!//)", csp)
    assert csp.count("https://www.youtube-nocookie.com") == 1
    assert csp.count("https://i.ytimg.com") == 1
    assert "https://www.youtube-nocookie.com" in parsed["frame-src"]
    assert "https://i.ytimg.com" in parsed["img-src"]
    for directive, sources in parsed.items():
        if directive != "style-src":
            assert "'unsafe-inline'" not in sources
    origins = re.findall(r"https?://[^\s;]+", csp)
    assert sorted(origins) == [
        "https://i.ytimg.com",
        "https://www.youtube-nocookie.com",
    ]


def test_csp_keeps_the_four_spec_directives_verbatim() -> None:
    # Architecture 9/11.3 verbatim: these four directives, unchanged.
    parsed = _parse_csp(
        next(
            v
            for k, v, _ in _add_headers(_location("/"))
            if k == "Content-Security-Policy"
        )
    )
    assert parsed["default-src"] == ["'self'"]
    assert parsed["frame-src"] == ["https://www.youtube-nocookie.com"]
    assert parsed["img-src"] == ["'self'", "https://i.ytimg.com", "data:"]
    assert parsed["style-src"] == ["'self'", "'unsafe-inline'"]


def test_referrer_policy_is_not_loosened_anywhere() -> None:
    text = _conf_text()
    assert text.count("Referrer-Policy") == len(OWN_LOCATIONS)
    assert "strict-origin" not in text
    assert "origin-when-cross-origin" not in text


def test_no_hsts_header() -> None:
    assert "strict-transport-security" not in _conf_text().lower()


def test_no_security_header_at_server_level() -> None:
    # A server-level add_header is inherited by /api/ and would be added to
    # the API's own headers; every own location carries its own set instead.
    server = _server()
    top = server
    for loc in re.finditer(r"location\s+[^{]+\{", server):
        top = top.replace(_block(server, loc.group(0).rstrip("{").rstrip()), "")
    assert "add_header" not in top


# ---- nginx.conf: /api/ proxy ----


def test_api_location_has_no_add_header() -> None:
    assert "add_header" not in _location("/api/")


def test_api_location_is_a_lazy_variable_upstream_with_prefix_stripped() -> None:
    body = _location("/api/")
    assert "set $api_upstream api:8000;" in body
    assert "proxy_pass http://$api_upstream$api_path;" in body
    assert "resolver 127.0.0.11 valid=10s" in _server()
    assert "proxy_pass http://api:8000" not in _conf_text()


def test_api_path_is_the_raw_request_uri_without_the_prefix() -> None:
    text = _conf_text()
    map_body = _block(text, "map $request_uri $api_path")
    assert "default" in map_body and "/;" in map_body
    assert r"~^/+api(?<rest>/.*)$ $rest;" in map_body


def test_request_id_is_set_from_nginx_and_logged() -> None:
    assert "proxy_set_header X-Request-Id $request_id;" in _location("/api/")
    text = _conf_text()
    assert re.search(r"log_format\s+\w+\s[^;]*\$request_id", text, re.DOTALL)
    name = re.search(r"log_format\s+(\w+)", text)
    assert name
    assert f"access_log /var/log/nginx/access.log {name.group(1)};" in _server()


def test_proxy_read_timeout_is_120s_and_connect_is_bounded() -> None:
    body = _location("/api/")
    assert "proxy_read_timeout 120s;" in body
    assert re.search(r"proxy_connect_timeout\s+([0-9]+)s;", body)
    assert int(re.search(r"proxy_connect_timeout\s+([0-9]+)s;", body).group(1)) < 10  # type: ignore[union-attr]


def test_api_responses_are_not_intercepted_or_compressed() -> None:
    body = _location("/api/")
    assert "proxy_intercept_errors" not in _conf_text()
    assert "gzip off;" in body
    assert "gzip_proxied on" not in _conf_text()
    for forbidden in ("proxy_hide_header", "proxy_set_header Host", "more_set_headers"):
        assert forbidden not in body


def test_dockerfile_and_conf_files_exist_where_the_dockerfile_copies_them() -> None:
    assert Path(REPO_ROOT / "ops" / "nginx.conf").is_file()
    assert Path(REPO_ROOT / "web" / "package-lock.json").is_file()
    assert Path(REPO_ROOT / "web" / "openapi.json").is_file()
