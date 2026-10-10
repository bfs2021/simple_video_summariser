"""Fixtures for eval tests (issue #62): the migrated-database fixtures, and clean logging."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
import structlog

from tests.common.repo.conftest import conn, head_dsn

__all__ = ["conn", "head_dsn"]


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if type(handler).__name__ == "_Handler":
            root.removeHandler(handler)
            handler.close()
    structlog.reset_defaults()
