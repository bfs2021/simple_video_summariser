"""The analyzer claims by its own dedupe key (issue #100). No Postgres: ``FakeDatabase``."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.queue import Job
from services.analyzer import main as analyzer_main
from tests.services.analyzer.fake_db import CLAIM_MARKER, FakeDatabase
from tests.services.analyzer.fakes import FakeSummarizer, make_job
from tests.services.analyzer.test_main import (  # noqa: F401 - the autouse fixture
    _isolated_process_state,
    fake_summarizer_factory,
    log_lines,
    make_settings,
    spy_handler_factory,
    worker_options,
)


def _run(
    tmp_path: Path, *, name: str = "ollama", jobs: list[Job] | None = None, iterations: int = 1,
    **settings: str,
) -> tuple[FakeDatabase, list[Job]]:
    db = FakeDatabase(jobs=jobs)
    factory, seen = spy_handler_factory()
    analyzer_main.main(
        settings=make_settings(**settings),
        connect=db.connect,
        summarizer_factory=fake_summarizer_factory(name),
        handler_factory=factory,
        worker_options=worker_options(tmp_path),
        max_iterations=iterations,
    )
    return db, seen


def _claims(db: FakeDatabase) -> list[dict[str, object]]:
    return [p for conn in db.connections for p in conn.statements(CLAIM_MARKER)]


def test_every_claim_carries_the_key_built_from_prompt_version_and_summarizer(
    tmp_path: Path,
) -> None:
    db, _ = _run(tmp_path, name="fake", iterations=3, PROMPT_VERSION="v7")

    claims = _claims(db)
    assert len(claims) == 3
    assert [c["dedupe_key"] for c in claims] == ["v7:fake"] * 3
    assert all(c["kinds"] == ["analyze"] for c in claims)


def test_the_started_line_logs_the_dedupe_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(tmp_path, PROMPT_VERSION="v3")

    (line,) = [e for e in log_lines(capsys) if e["event"] == "analyzer.started"]
    assert line["dedupe_key"] == "v3:ollama"


def test_a_summarizer_name_that_cannot_form_a_key_is_a_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = FakeDatabase()

    code = analyzer_main.main(
        settings=make_settings(),
        connect=db.connect,
        summarizer_factory=lambda s: FakeSummarizer(name="a:b"),
        handler_factory=spy_handler_factory()[0],
        worker_options=worker_options(tmp_path),
        max_iterations=1,
    )

    assert code == analyzer_main.EXIT_CONFIG
    events = [e["event"] for e in log_lines(capsys)]
    assert "analyzer.invalid_config" in events and "analyzer.started" not in events
    assert _claims(db) == []


def test_the_lazy_queue_forwards_the_keyword_to_the_claim() -> None:
    db = FakeDatabase()
    lazy = analyzer_main._LazyQueue(db.connect, make_settings())

    with lazy.claim(["analyze"], worker="w", dedupe_key="v2:ollama") as job:
        assert job is None

    (params,) = _claims(db)
    assert params["dedupe_key"] == "v2:ollama"


def test_the_lazy_queue_claims_without_a_key_when_none_is_given() -> None:
    db = FakeDatabase()
    lazy = analyzer_main._LazyQueue(db.connect, make_settings())

    with lazy.claim(["analyze"], worker="w") as job:
        assert job is None

    (params,) = _claims(db)
    assert "dedupe_key" not in params


def test_the_guard_stays_as_defence_in_depth_and_its_docstring_says_when_it_fires() -> None:
    doc = analyzer_main.build_job_handler.__doc__ or ""

    assert "without the key" in doc
    assert make_job("abcdefghijk").dedupe_key  # the guard's tests live in test_main.py
