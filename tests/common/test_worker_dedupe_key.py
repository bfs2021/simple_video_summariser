"""``Worker(dedupe_key=...)`` (issue #100). No database: a fake queue records the claim call."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from common.queue import Job
from common.worker import Worker


class KeywordlessQueue:
    """A queue whose ``claim`` predates the ``dedupe_key`` keyword (like many fakes)."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str]] = []

    @contextmanager
    def claim(self, kinds: Sequence[str], *, worker: str) -> Iterator[Job | None]:
        self.calls.append((tuple(kinds), worker))
        yield None


class KeyedQueue:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @contextmanager
    def claim(
        self, kinds: Sequence[str], *, worker: str, dedupe_key: str | None = None
    ) -> Iterator[Job | None]:
        self.calls.append({"kinds": tuple(kinds), "worker": worker, "dedupe_key": dedupe_key})
        yield None


def _worker(queue: Any, tmp_path: Path, **kwargs: Any) -> Worker:
    return Worker(
        "w1",
        ["analyze"],
        lambda job, ctx: None,
        queue,
        liveness_path=tmp_path / "hb",
        heartbeat_sec=60,
        shutdown_grace_sec=20,
        sleep=lambda _s: None,
        rng=random.Random(0),
        heartbeat_queue_factory=lambda: queue,
        reconnect=lambda: queue,
        install_signal_handlers=False,
        **kwargs,
    )


def test_without_a_key_claim_gets_no_dedupe_key_argument_at_all(tmp_path: Path) -> None:
    # A fake whose claim has no ``dedupe_key`` keyword would raise TypeError otherwise.
    queue = KeywordlessQueue()

    _worker(queue, tmp_path).run(max_iterations=2)

    assert queue.calls == [(("analyze",), "w1"), (("analyze",), "w1")]


def test_without_a_key_a_keyed_queue_sees_none(tmp_path: Path) -> None:
    queue = KeyedQueue()

    _worker(queue, tmp_path).run(max_iterations=1)

    assert [c["dedupe_key"] for c in queue.calls] == [None]


def test_with_a_key_every_claim_passes_it(tmp_path: Path) -> None:
    queue = KeyedQueue()

    _worker(queue, tmp_path, dedupe_key="v2:ollama").run(max_iterations=3)

    assert queue.calls == [
        {"kinds": ("analyze",), "worker": "w1", "dedupe_key": "v2:ollama"}
    ] * 3
