"""The ``eval`` command group of ``ytdigest`` (issue #62).

``ytdigest eval run`` and ``ytdigest eval report``; registered by
``services.cli.main``. The harness does not coordinate with a running analyzer
worker, ingests nothing and runs no migration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import psycopg
from pydantic import ValidationError

from adapters.summarize.factory import SummarizerConfigError, build_summarizer
from adapters.summarize.prompt_loader import load_prompt_set
from common.chunking import chunk_strategy
from common.config import Settings
from common.db import connect
from common.logging import configure_logging
from common.models import Summarizer
from common.repo.analyses import analysis_run_for
from common.repo.videos import get_video_meta
from services.analyzer.handler import make_analyze_handler
from services.cli import main as cli
from services.eval.golden import GoldenError, ParsedGolden, read_golden
from services.eval.report import RunLabel, VideoEntry, render_report
from services.eval.runner import VideoOutcome, apply_overrides, evaluate_video, tally

_RUN_HELP = (
    "Analyse every golden-set video with the chosen summarizer, model and prompt "
    "version and store the results as ordinary analyses rows. Videos already "
    "analysed with the same model, prompt version, transcript and chunk strategy "
    "are skipped, so a rerun resumes. The harness does not coordinate with a "
    "running analyzer worker: if both analyse the same video with the same model "
    "and prompt at the same moment, two rows may be written. With the Anthropic "
    "summarizer set ANTHROPIC_BATCH=false, or a golden-set run can take hours."
)
_REPORT_HELP = (
    "Write a Markdown report that puts two runs, A and B, side by side for every "
    "golden-set video. A run is the newest analysis with a given model and prompt "
    "version. No score is computed."
)


def add_eval_parser(commands: Any) -> None:
    """Register ``eval run`` and ``eval report`` on the top-level subparsers."""
    group = commands.add_parser(
        "eval",
        help="compare summarizer outputs on a golden set",
        description="Quality evaluation harness (manual comparison, no automated score).",
    )
    sub = group.add_subparsers(dest="eval_command", required=True, metavar="subcommand")

    run = sub.add_parser("run", help="analyse a golden set", description=_RUN_HELP)
    run.add_argument("--golden", required=True, metavar="PATH", help="golden-set file, or - for stdin")
    run.add_argument("--summarizer", choices=("ollama", "anthropic"), help="default: SUMMARIZER")
    run.add_argument("--model", help="model of the chosen summarizer; default: its *_MODEL setting")
    run.add_argument("--prompt-version", dest="prompt_version", help="default: PROMPT_VERSION")
    run.add_argument(
        "--dry-run", action="store_true", help="print what would happen; no LLM call, no writes"
    )

    report = sub.add_parser("report", help="compare two runs", description=_REPORT_HELP)
    report.add_argument("--golden", required=True, metavar="PATH", help="golden-set file, or - for stdin")
    report.add_argument("--a-model", required=True, dest="a_model")
    report.add_argument("--a-prompt", required=True, dest="a_prompt")
    report.add_argument("--b-model", required=True, dest="b_model")
    report.add_argument("--b-prompt", required=True, dest="b_prompt")
    report.add_argument("--out", metavar="PATH", help="output file; default or - is stdout")


def _golden(path: str) -> ParsedGolden | None:
    try:
        parsed = read_golden(path, stdin=sys.stdin.read)
    except GoldenError as exc:
        for message in exc.messages:
            cli._say(f"ytdigest eval: error: {message}")
        return None
    for warning in parsed.warnings:
        cli._say(f"warning: {warning}")
    return parsed


def _base_settings(deps: cli.Deps | None) -> Settings | None:
    return deps.settings if deps is not None else cli._settings_from_environment()


def _open(deps: cli.Deps | None) -> psycopg.Connection[Any] | None:
    """A connection to a migrated database, or ``None`` after saying why not."""
    opener = deps.connect if deps is not None else connect
    try:
        conn = opener()
    except (psycopg.Error, OSError):
        # No detail: a connection error can carry the DSN, and with it the password.
        cli._say("cannot connect to the database: check DATABASE_URL and that Postgres is running")
        return None
    try:
        missing = cli._missing_tables(conn)
    except psycopg.Error:
        cli._say("cannot query the database: check DATABASE_URL and that Postgres is running")
        conn.close()
        return None
    if missing:
        cli._schema_missing(missing)
        conn.close()
        return None
    return conn


def _settings_error(exc: ValidationError) -> int:
    fields = sorted({str(e["loc"][0]) for e in exc.errors() if e["loc"]})
    if fields == ["PROMPT_VERSION"]:
        cli._say("ytdigest eval: error: invalid --prompt-version (must match ^[a-z0-9_-]+$)")
        return cli._EXIT_USAGE
    # Field names only: the invalid value may be a DSN with a password.
    cli._say(f"invalid configuration: check {', '.join(fields) or 'the environment'}")
    return cli._EXIT_USAGE if fields and set(fields) <= {"SUMMARIZER", "OLLAMA_MODEL", "ANTHROPIC_MODEL"} else cli._EXIT_FAILED


def run_command(args: argparse.Namespace, deps: cli.Deps | None) -> int:
    golden = _golden(args.golden)
    if golden is None:
        return cli._EXIT_USAGE
    base = _base_settings(deps)
    if base is None:
        return cli._EXIT_FAILED
    try:
        settings = apply_overrides(
            base, summarizer=args.summarizer, model=args.model, prompt_version=args.prompt_version
        )
    except ValidationError as exc:
        return _settings_error(exc)
    try:
        load_prompt_set(settings.PROMPT_VERSION)
    except FileNotFoundError:
        cli._say(f"ytdigest eval: error: prompt version {settings.PROMPT_VERSION!r} not found")
        return cli._EXIT_USAGE
    except ValueError:
        cli._say(f"ytdigest eval: error: prompt version {settings.PROMPT_VERSION!r} is invalid")
        return cli._EXIT_USAGE
    configure_logging(settings, stream=sys.stderr)
    summarizer: Summarizer
    if deps is not None:
        summarizer = deps.summarizer
    else:
        try:
            summarizer = build_summarizer(settings)
        except SummarizerConfigError as exc:
            cli._say(f"invalid configuration: {cli._one_line(str(exc))}")
            return cli._EXIT_FAILED
    strategy = chunk_strategy(settings.CHUNK_SEC, settings.OVERLAP_SEC)
    cli._say(
        f"eval: summarizer={summarizer.name} model={cli._one_line(summarizer.model)} "
        f"prompt_version={settings.PROMPT_VERSION} chunk_strategy={strategy}"
    )

    conn = _open(deps)
    if conn is None:
        return cli._EXIT_FAILED
    try:
        handler = None
        if not args.dry_run:
            handler = make_analyze_handler(
                connect=deps.connect if deps is not None else connect,
                summarizer=summarizer,
                settings=settings,
            )
        outcomes: list[VideoOutcome] = []
        total = len(golden.video_ids)
        for number, video_id in enumerate(golden.video_ids, start=1):
            outcome = evaluate_video(
                conn, handler, settings, summarizer, video_id, dry_run=args.dry_run
            )
            outcomes.append(outcome)
            cli._say(f"[{number}/{total}] {video_id} {outcome.text}")
        cli._say(tally(outcomes, dry_run=args.dry_run))
        bad = any(o.status in ("missing", "failed") for o in outcomes)
        return cli._EXIT_FAILED if bad else 0
    finally:
        conn.close()


def report_command(args: argparse.Namespace, deps: cli.Deps | None) -> int:
    golden = _golden(args.golden)
    if golden is None:
        return cli._EXIT_USAGE
    label_a = RunLabel(args.a_model, args.a_prompt)
    label_b = RunLabel(args.b_model, args.b_prompt)
    if label_a == label_b:
        cli._say("ytdigest eval: error: runs A and B are the same model and prompt version")
        return cli._EXIT_USAGE
    if _base_settings(deps) is None:
        return cli._EXIT_FAILED
    conn = _open(deps)
    if conn is None:
        return cli._EXIT_FAILED
    try:
        entries = []
        for video_id in golden.video_ids:
            meta = get_video_meta(conn, video_id)
            run_a = analysis_run_for(conn, video_id, label_a.model, label_a.prompt_version)
            run_b = analysis_run_for(conn, video_id, label_b.model, label_b.prompt_version)
            conn.rollback()
            entries.append(VideoEntry(video_id, meta.title if meta else None, run_a, run_b))
    finally:
        conn.close()
    text = render_report(label_a, label_b, entries)
    if args.out in (None, "-"):
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0
    try:
        Path(args.out).write_text(text, encoding="utf-8")
    except OSError:
        cli._say("ytdigest eval: error: cannot write the output file")
        return cli._EXIT_FAILED
    return 0


def eval_main(args: argparse.Namespace, deps: cli.Deps | None) -> int:
    if args.eval_command == "run":
        return run_command(args, deps)
    return report_command(args, deps)
