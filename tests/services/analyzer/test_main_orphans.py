"""The orphan pass wired into the analyzer entrypoint (issue #160).

No Postgres, no network: the worker runs on a ``FakeDatabase``, the orphan
thread on a fake connection, store and canceller. The worker's injected
``sleep`` waits (bounded) for the condition under test instead of racing it.
"""

from __future__ import annotations

import os
import signal
import threading
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from common.models import OrphanOutcome, Summarizer
from common.repo.llm_batches import OrphanBatch
from services.analyzer import main as analyzer_main
from services.analyzer import orphans
from tests.services.analyzer.fake_db import FakeDatabase
from tests.services.analyzer.fakes import FakeSummarizer
from tests.services.analyzer.test_main import (
    _isolated_process_state,  # noqa: F401 - autouse fixture, reused
    make_settings,
    spy_handler_factory,
    worker_options,
)

DEADLINE_SEC = 5.0


class CancellingSummarizer(FakeSummarizer):
    """A fake that also implements ``OrphanBatchCanceller``."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cancelled: list[str] = []
        self.first_cancel = threading.Event()

    def cancel_orphan_batch(self, batch_id: str) -> OrphanOutcome:
        self.cancelled.append(batch_id)
        self.first_cancel.set()
        return "cancelled"


class OneRowStore:
    def __init__(self, *_: Any) -> None:
        pass

    def list(self, limit: int, expired_after: timedelta) -> Sequence[OrphanBatch]:
        return [OrphanBatch(7, "roster", "msgbatch_7", False)]

    def delete(self, row: OrphanBatch) -> bool:
        return True


class FakeOrphanConn:
    def close(self) -> None:
        pass


class Rig:
    def __init__(
        self, tmp_path: Path, summarizer: FakeSummarizer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.tmp_path = tmp_path
        self.summarizer = summarizer
        self.threads: list[threading.Thread] = []
        real_start = orphans.start_orphan_thread

        def recording_start(*args: Any, **kwargs: Any) -> threading.Thread:
            thread = real_start(*args, **kwargs)
            self.threads.append(thread)
            return thread

        monkeypatch.setattr(orphans, "start_orphan_thread", recording_start)

    def run(self, first_pass_ran: threading.Event, **settings: Any) -> int:
        def sleep(_seconds: float) -> None:
            # The worker idles here: hold it until the orphan pass has run, then stop it.
            assert first_pass_ran.wait(DEADLINE_SEC), "the first orphan pass never ran"
            os.kill(os.getpid(), signal.SIGTERM)

        return analyzer_main.main(
            settings=make_settings(**settings),
            connect=FakeDatabase().connect,
            summarizer_factory=lambda s: self.summarizer,
            handler_factory=spy_handler_factory()[0],
            worker_options={**worker_options(self.tmp_path), "sleep": sleep},
            orphan_options={
                "connect": FakeOrphanConn,
                "store_factory": OneRowStore,
                "interval_sec": 3600,
                "limit": 2,
            },
        )


def test_sigterm_ends_main_and_the_orphan_thread_after_its_first_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summarizer = CancellingSummarizer(name="anthropic")
    rig = Rig(tmp_path, summarizer, monkeypatch)

    code = rig.run(summarizer.first_cancel, SUMMARIZER="anthropic", ANTHROPIC_API_KEY="k")

    assert code == 0
    assert summarizer.cancelled == ["msgbatch_7"]  # the first pass ran at once, one cancel
    (thread,) = rig.threads
    thread.join(DEADLINE_SEC)
    assert not thread.is_alive(), f"orphan thread {thread.name!r} still alive {DEADLINE_SEC}s after shutdown"
    assert thread.daemon
    assert thread.name == orphans.THREAD_NAME


def test_anthropic_without_batch_mode_still_starts_the_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summarizer = CancellingSummarizer(name="anthropic")
    rig = Rig(tmp_path, summarizer, monkeypatch)

    rig.run(
        summarizer.first_cancel,
        SUMMARIZER="anthropic",
        ANTHROPIC_API_KEY="k",
        ANTHROPIC_BATCH=False,
    )

    assert summarizer.cancelled == ["msgbatch_7"]


def test_a_summarizer_without_the_protocol_starts_no_thread_and_makes_no_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_start(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an orphan thread must not start for this summarizer")

    monkeypatch.setattr(orphans, "start_orphan_thread", must_not_start)
    db = FakeDatabase()

    code = analyzer_main.main(
        settings=make_settings(),
        connect=db.connect,
        summarizer_factory=lambda s: FakeSummarizer(name="ollama"),
        handler_factory=spy_handler_factory()[0],
        worker_options=worker_options(tmp_path),
        max_iterations=2,
    )

    assert code == 0
    assert not [t for t in threading.enumerate() if t.name == orphans.THREAD_NAME]


def test_the_fake_summarizer_does_not_implement_the_protocol() -> None:
    from common.models import OrphanBatchCanceller

    summarizer: Summarizer = FakeSummarizer(name="ollama")
    assert not isinstance(summarizer, OrphanBatchCanceller)
