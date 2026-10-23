"""Contract drift (issue #46): the committed web/openapi.json must match the backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from services.api import openapi_export

COMMITTED = Path(__file__).resolve().parents[3] / "web" / "openapi.json"
REGEN = "python -m services.api.openapi_export && (cd web && npm run gen:api)"
NON_JSON_ROUTES = {
    ("delete", "/videos/{video_id}"),
    ("get", "/videos/{video_id}/render"),
}
METHODS = {"get", "put", "post", "delete", "patch", "options", "head"}


def diff_paths(committed: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Human-readable added/removed/changed paths; never the whole document."""
    old, new = committed.get("paths", {}), current.get("paths", {})
    lines = [f"added path: {p}" for p in sorted(new.keys() - old.keys())]
    lines += [f"removed path: {p}" for p in sorted(old.keys() - new.keys())]
    lines += [f"changed path: {p}" for p in sorted(old.keys() & new.keys()) if old[p] != new[p]]
    if not lines and committed != current:
        lines.append("changed outside paths (components or info)")
    return lines


def drift_message(committed: dict[str, Any], current: dict[str, Any]) -> str:
    return (
        "web/openapi.json is out of date with the backend. Regenerate with:\n"
        f"  {REGEN}\n" + "\n".join(diff_paths(committed, current))
    )


def test_committed_schema_matches_the_backend_byte_for_byte() -> None:
    current = openapi_export.render(openapi_export.build_schema())
    committed = COMMITTED.read_bytes().decode("utf-8")
    if committed != current:
        raise AssertionError(drift_message(json.loads(committed), json.loads(current)))


def test_drift_message_names_command_and_lists_added_removed_changed() -> None:
    base = {"paths": {"/a": {"get": 1}, "/b": {"get": 1}, "/c": {"get": 1}}, "big": "X" * 500}
    cur = {"paths": {"/a": {"get": 1}, "/b": {"get": 2}, "/d": {"get": 1}}, "big": "X" * 500}
    message = drift_message(base, cur)
    assert "python -m services.api.openapi_export" in message
    assert "npm run gen:api" in message
    assert "added path: /d" in message
    assert "removed path: /c" in message
    assert "changed path: /b" in message
    assert "/a" not in message
    assert "X" * 50 not in message


def test_drift_message_flags_changes_outside_paths() -> None:
    message = drift_message({"paths": {}, "components": 1}, {"paths": {}, "components": 2})
    assert "outside paths" in message


def _success_codes(operation: dict[str, Any]) -> list[str]:
    return sorted(c for c in operation["responses"] if c.startswith("2"))


def test_every_json_operation_declares_a_success_response_schema() -> None:
    offenders: list[str] = []
    for path, item in openapi_export.build_schema()["paths"].items():
        for method, operation in item.items():
            if method not in METHODS or (method, path) in NON_JSON_ROUTES:
                continue
            codes = _success_codes(operation)
            if not codes:
                offenders.append(f"{method.upper()} {path} (no 2xx response)")
            for code in codes:
                schema = (
                    operation["responses"][code]
                    .get("content", {})
                    .get("application/json", {})
                    .get("schema")
                )
                if not schema or (schema.get("type") == "object" and not schema.get("properties")):
                    offenders.append(f"{method.upper()} {path} ({code})")
    assert offenders == []


def test_html_route_declares_text_html_not_json() -> None:
    paths = openapi_export.build_schema()["paths"]
    for method, path in {("get", "/videos/{video_id}/render")}:
        operation = paths[path][method]
        codes = _success_codes(operation)
        assert codes == ["200"]
        content = operation["responses"]["200"]["content"]
        assert list(content) == ["text/html"]
