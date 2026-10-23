"""Deselects ``@pytest.mark.whisper`` tests unless explicitly requested.

Registered in ``pyproject.toml``. The marker can't simply ride in
``addopts`` as ``-m "not whisper"``: pytest's ``-m`` option is single-valued,
so a bare ``pytest -m "not integration"`` would silently replace it rather
than combine with it, and whisper tests would run (and fail, since
faster-whisper is deliberately not installed on the host - AGENTS.md ->
Setup). Filtering here instead means "not whisper" is applied unconditionally
alongside whatever ``-m`` expression, if any, the caller passes.

This hook lives here rather than in the shared ``tests/conftest.py`` so it
stays inside issue #20's file list; it fires for the whole session regardless
(pytest loads every conftest.py on the path to a collected file, and
``pytest_collection_modifyitems`` implementations all run against the same
session-wide item list).
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    markexpr = config.getoption("markexpr") or ""
    if "whisper" in markexpr:
        return  # e.g. `-m whisper` or `-m "whisper and not slow"`: let it through

    keep: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        if "whisper" in item.keywords:
            deselected.append(item)
        else:
            keep.append(item)

    if deselected:
        items[:] = keep
        config.hook.pytest_deselected(items=deselected)
