"""Fixtures for analyzer tests (issue #30).

Reuses the migrated-database fixtures from ``tests/common/repo/conftest.py``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from adapters.summarize.ollama import HttpResponse
from services.analyzer import main as analyzer_main
from tests.common.repo.conftest import conn, head_dsn

__all__ = ["conn", "head_dsn"]


@pytest.fixture(autouse=True)
def _no_real_ollama_probe(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Startup runs an Ollama probe (#84); no analyzer test may reach the network.

    A test that cares about the probe passes ``probe_transport`` to ``main``.
    """

    def refused(url: str, timeout: float) -> HttpResponse:
        raise ConnectionRefusedError("no Ollama in tests")

    monkeypatch.setattr(analyzer_main, "urllib_get_transport", refused)
    yield
