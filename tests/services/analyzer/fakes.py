"""Test doubles for the analyze handler tests (issue #30).

``FakeSummarizer`` satisfies the ``Summarizer`` port from ``common/models.py``
(``name``, ``model``, ``take_usage()``): canned results, recorded calls, and a
``fail_on`` table to raise a given error on a given call. No network, no sleep.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from common.models import Chunk, ChunkAnalysis, Roster, Usage, VideoMeta
from common.queue import Job

_NOW = datetime(2026, 9, 28, tzinfo=UTC)

# (input_tokens, output_tokens) a successful call adds, per call kind.
DEFAULT_TOKENS: dict[str, tuple[int, int]] = {
    "roster": (7, 3),
    "chunk": (100, 50),
    "reduce": (20, 10),
}


class FakeSummarizer:
    """Returns canned results and records every call as ``(kind, *args)``.

    ``fail_on`` maps ``"roster"``, ``"chunk:<seq>"`` or ``"reduce"`` to the
    exception raised by that call. A failed call adds ``usage_on_fail`` tokens
    (as a real adapter that spent tokens on an unusable answer would).
    ``on_call(kind)`` runs at the start of every call. ``costs`` maps a call kind
    (``"roster"``, ``"chunk"``, ``"reduce"``) to the ``cost_usd`` its successful
    call reports (``None`` = priced unknown); a kind not listed reports no call at
    all (``Usage.calls == 0``), like a summarizer that does not track calls.
    """

    def __init__(
        self,
        *,
        name: str = "ollama",
        model: str = "fake-model",
        roster: Roster | None = None,
        chunks: dict[int, ChunkAnalysis] | None = None,
        default_chunk: ChunkAnalysis | None = None,
        tldr: str = "the tldr",
        tokens: dict[str, tuple[int, int]] | None = None,
        fail_on: dict[str, BaseException] | None = None,
        usage_on_fail: tuple[int, int] = (0, 0),
        on_call: Callable[[str], None] | None = None,
        costs: dict[str, float | None] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.roster = roster if roster is not None else Roster(())
        self.chunks = chunks or {}
        self.default_chunk = default_chunk or ChunkAnalysis(topics=(), claims=(), quotes=())
        self.tldr = tldr
        self.tokens = {**DEFAULT_TOKENS, **(tokens or {})}
        self.fail_on = fail_on if fail_on is not None else {}
        self.usage_on_fail = usage_on_fail
        self.on_call = on_call
        self.costs = costs or {}
        self.calls: list[tuple[Any, ...]] = []
        self.take_usage_calls = 0
        self._pending = [0, 0]
        self._calls = 0
        self._cost: float | None = None
        self._cost_unknown = False

    @property
    def kinds(self) -> list[str]:
        return [str(call[0]) for call in self.calls]

    def _begin(self, kind: str, key: str) -> None:
        if self.on_call is not None:
            self.on_call(kind)
        failure = self.fail_on.get(key)
        if failure is not None:
            self._pending[0] += self.usage_on_fail[0]
            self._pending[1] += self.usage_on_fail[1]
            raise failure

    def _succeed(self, kind: str) -> None:
        tokens_in, tokens_out = self.tokens[kind]
        self._pending[0] += tokens_in
        self._pending[1] += tokens_out
        if kind in self.costs:
            self._calls += 1
            cost = self.costs[kind]
            if cost is None:
                self._cost_unknown = True
            else:
                self._cost = (self._cost or 0.0) + cost

    def derive_roster(self, meta: VideoMeta, opening: str) -> Roster:
        self.calls.append(("roster", meta, opening))
        self._begin("roster", "roster")
        self._succeed("roster")
        return self.roster

    def analyze_chunk(self, chunk: Chunk, roster: Roster, meta: VideoMeta) -> ChunkAnalysis:
        self.calls.append(("chunk", chunk, roster, meta))
        self._begin("chunk", f"chunk:{chunk.seq}")
        self._succeed("chunk")
        return self.chunks.get(chunk.seq, self.default_chunk)

    def reduce(self, partials: list[ChunkAnalysis], meta: VideoMeta) -> str:
        self.calls.append(("reduce", list(partials), meta))
        self._begin("reduce", "reduce")
        self._succeed("reduce")
        return self.tldr

    def take_usage(self) -> Usage:
        self.take_usage_calls += 1
        usage = Usage(
            input_tokens=self._pending[0],
            output_tokens=self._pending[1],
            calls=self._calls,
            cost_usd=None if self._cost_unknown else self._cost,
        )
        self._pending = [0, 0]
        self._calls = 0
        self._cost = None
        self._cost_unknown = False
        return usage


class FakeClock:
    """Returns ``start``, ``start + step``, ... one value per call."""

    def __init__(self, *, start: float = 0.0, step: float = 1.0) -> None:
        self._next = start
        self._step = step
        self.calls = 0

    def __call__(self) -> float:
        value = self._next
        self._next += self._step
        self.calls += 1
        return value


def make_job(
    video_id: str,
    *,
    dedupe_key: str = "v1:ollama",
    payload: dict[str, Any] | None = None,
) -> Job:
    return Job(
        id=1,
        video_id=video_id,
        kind="analyze",
        dedupe_key=dedupe_key,
        state="running",
        priority=0,
        payload=payload if payload is not None else {},
        attempts=1,
        last_error=None,
        error_class=None,
        run_after=_NOW,
        locked_by="w",
        locked_at=_NOW,
        heartbeat_at=_NOW,
        finished_at=None,
        created_at=_NOW,
    )
