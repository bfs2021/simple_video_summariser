"""Shuffle the test order, reproducibly (issue #72).

On by default; ``--shuffle-seed=N`` replays an order and ``--no-shuffle``
turns it off. Order comes only from a dedicated ``random.Random(seed)``; the
global ``random`` state is never touched.

Tests stay grouped by their parents: packages, modules and classes are each
shuffled as a unit, and the tests inside them are shuffled among themselves.
A module's tests are therefore never interleaved with another module's, so
module-scoped fixtures (image builds, summarize module fixtures) are still
set up once per module.
"""

from __future__ import annotations

import random

import pytest
from _pytest import nodes

_SEED_KEY = pytest.StashKey[int | None]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("shuffle", "test order shuffling")
    group.addoption(
        "--shuffle-seed",
        action="store",
        type=int,
        default=None,
        metavar="N",
        help="Seed for the shuffled test order (replays a run). Default: random, printed.",
    )
    group.addoption(
        "--no-shuffle",
        action="store_true",
        default=False,
        help="Run tests in plain collection order.",
    )


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("no_shuffle"):
        config.stash[_SEED_KEY] = None
        return
    seed = config.getoption("shuffle_seed")
    if seed is None:
        seed = random.SystemRandom().randrange(2**32)
    config.stash[_SEED_KEY] = seed


def _shuffled(
    items: list[pytest.Item], rng: random.Random, depth: int
) -> list[pytest.Item]:
    groups: dict[nodes.Node, list[pytest.Item]] = {}
    singles: list[list[pytest.Item]] = []
    for item in items:
        chain = item.listchain()
        if depth >= len(chain) - 1:
            singles.append([item])
        else:
            groups.setdefault(chain[depth], []).append(item)
    units = [
        _shuffled(members, rng, depth + 1) for members in groups.values()
    ] + singles
    rng.shuffle(units)
    return [item for unit in units for item in unit]


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    # trylast: after the image/whisper deselection hooks. pytest's cache plugin
    # (--ff/--lf/--nf) reorders in a wrapper after all plain hooks, so it wins.
    seed = config.stash.get(_SEED_KEY, None)
    if seed is None:
        return
    items[:] = _shuffled(items, random.Random(seed), 0)


def pytest_report_header(config: pytest.Config) -> str:
    seed = config.stash.get(_SEED_KEY, None)
    return "shuffle: off" if seed is None else f"shuffle seed: {seed}"


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
) -> None:
    seed = config.stash.get(_SEED_KEY, None)
    if seed is not None:
        terminalreporter.write_line(f"shuffle seed: {seed}")
