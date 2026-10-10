"""Side-by-side Markdown report for two eval runs (issue #62).

Everything here is a pure function from ``AnalysisRun`` objects and video
metadata to a string, so it is unit-tested without a database. The report
computes no score, rank or winner: it only makes claim specificity and speaker
attribution easy to judge by eye, with every timestamp linking to that moment
on YouTube. The output is deterministic (no generation time, stable order).

Titles, TL;DRs, topics, claims, quotes, roster entries, and the model and
prompt version values are untrusted (YouTube, an LLM that read a transcript
anyone can write, or the database). ``md_text`` is the only way such text gets
into the output.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from typing import Any

from common.models import Analysis, AnalysisRun, Claim, Quote
from common.youtube_refs import is_video_id
from services.cli.output import format_timestamp, sanitize_text

WINDOW_SEC = 300
UNKNOWN = "unknown"

_MD_META = re.compile(r"([\\`*_\[\]|()])")
_AUTOLINK_SCHEME = re.compile(r"(?i)\b(https?|ftp)://")
_AUTOLINK_WWW = re.compile(r"(?i)\bwww\.")


@dataclass(frozen=True, slots=True)
class RunLabel:
    """Which runs a side of the report stands for."""

    model: str
    prompt_version: str

    def text(self) -> str:
        return f"{md_text(self.model)} / {md_text(self.prompt_version)}"


@dataclass(frozen=True, slots=True)
class VideoEntry:
    """One golden-set video: ``title`` is ``None`` when it is not in the database."""

    video_id: str
    title: str | None
    run_a: AnalysisRun | None
    run_b: AnalysisRun | None


@dataclass(frozen=True, slots=True)
class Window[T]:
    """Items of A and B in one 5-minute window; ``index`` is ``None`` for "No timestamp"."""

    index: int | None
    a: tuple[T, ...]
    b: tuple[T, ...]


# ------------------------------------------------------------------- escaping


def md_text(text: str) -> str:
    """``text`` as inert Markdown: one line, control characters gone, HTML and Markdown escaped."""
    out = " ".join(sanitize_text(text).split())
    out = out.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out = _MD_META.sub(r"\\\1", out)
    # Bare URLs would be auto-linked by GitHub-flavoured renderers.
    out = _AUTOLINK_SCHEME.sub(lambda m: f"{m.group(1)}\\:\\/\\/", out)
    out = _AUTOLINK_WWW.sub("www\\.", out)
    return re.sub(r"^#", r"\\#", out)


# ----------------------------------------------------------- windows and links


def _valid_start(start_sec: float | None) -> bool:
    return start_sec is not None and math.isfinite(start_sec) and start_sec >= 0


def timestamp_link(video_id: str, start_sec: float | None) -> str | None:
    """``watch?v=<id>&t=<n>s`` for a validated ID and a finite, non-negative start."""
    if not is_video_id(video_id) or start_sec is None or not _valid_start(start_sec):
        return None
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start_sec)}s"


def window_label(index: int) -> str:
    return f"{format_timestamp(index * WINDOW_SEC)}-{format_timestamp((index + 1) * WINDOW_SEC)}"


def group_windows[T: (Claim, Quote)](a: Sequence[T], b: Sequence[T]) -> tuple[Window[T], ...]:
    """Items of both sides by 5-minute window ``[k*300, (k+1)*300)``, only non-empty windows.

    Items without a usable ``start_sec`` form a final window with ``index=None``.
    Order inside a side is the input order.
    """
    groups: dict[int | None, tuple[list[T], list[T]]] = {}

    def add(items: Sequence[T], side: int) -> None:
        for item in items:
            start = item.start_sec
            key = math.floor(start / WINDOW_SEC) if start is not None and _valid_start(start) else None
            groups.setdefault(key, ([], []))[side].append(item)

    add(a, 0)
    add(b, 1)
    keys = sorted(k for k in groups if k is not None)
    if None in groups:
        keys.append(None)  # type: ignore[arg-type]
    return tuple(Window(k, tuple(groups[k][0]), tuple(groups[k][1])) for k in keys)


# ------------------------------------------------------------------ rendering


def _roster_lines(roster: object) -> list[str]:
    speakers: Any = roster.get("speakers") if isinstance(roster, dict) else None
    lines: list[str] = []
    if isinstance(speakers, list):
        for speaker in speakers:
            if isinstance(speaker, dict) and isinstance(speaker.get("name"), str):
                role = speaker.get("role")
                suffix = f" ({md_text(role)})" if isinstance(role, str) and role.strip() else ""
                lines.append(f"- {md_text(speaker['name'])}{suffix}")
    return lines or ["No roster"]


def _count_unknown(analysis: Analysis) -> int:
    return sum(1 for c in analysis.claims if c.speaker == UNKNOWN)


def _confidence_cell(analysis: Analysis) -> str:
    counts = [sum(1 for c in analysis.claims if c.confidence == level) for level in
              ("high", "medium", "low")]
    counts.append(len(analysis.claims) - sum(counts))
    return " / ".join(str(n) for n in counts)


def _created(analysis: Analysis) -> str:
    created = analysis.created_at
    return created.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S") if created else "-"


def _row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [_row(*header), _row(*("---" for _ in header)), *(_row(*r) for r in rows)]


def _tokens(analysis: Analysis) -> int:
    return analysis.input_tokens + analysis.output_tokens


def _summary_cells(run: AnalysisRun | None) -> list[str]:
    """claims, quotes, unknown n/m, tokens, duration - ``-`` for a missing run."""
    if run is None:
        return ["-"] * 5
    a = run.analysis
    return [
        str(len(a.claims)),
        str(len(a.quotes)),
        f"{_count_unknown(a)}/{len(a.claims)}",
        str(_tokens(a)),
        f"{a.duration_ms} ms" if a.duration_ms is not None else "-",
    ]


def _interleave(a: list[str], b: list[str]) -> list[str]:
    return [cell for pair in zip(a, b, strict=True) for cell in pair]


def _totals(runs: Sequence[AnalysisRun | None]) -> list[str]:
    present = [r.analysis for r in runs if r is not None]
    if not present:
        return ["-"] * 5
    claims = sum(len(x.claims) for x in present)
    durations = [x.duration_ms for x in present if x.duration_ms is not None]
    return [
        str(claims),
        str(sum(len(x.quotes) for x in present)),
        f"{sum(_count_unknown(x) for x in present)}/{claims}",
        str(sum(_tokens(x) for x in present)),
        f"{sum(durations)} ms" if durations else "-",
    ]


def _summary_table(entries: Sequence[VideoEntry]) -> list[str]:
    header = ["Video"]
    for name in ("claims", "quotes", "unknown claims", "tokens", "duration"):
        header += [f"A {name}", f"B {name}"]
    rows = []
    for e in entries:
        title = md_text(e.title) if e.title else md_text(e.video_id)
        rows.append([title, *_interleave(_summary_cells(e.run_a), _summary_cells(e.run_b))])
    rows.append(
        [
            "**Total**",
            *_interleave(_totals([e.run_a for e in entries]), _totals([e.run_b for e in entries])),
        ]
    )
    return _table(header, rows)


def _details_table(a: AnalysisRun | None, b: AnalysisRun | None) -> list[str]:
    def cells(run: AnalysisRun | None) -> list[str]:
        if run is None:
            return ["-"] * 10
        x = run.analysis
        return [
            md_text(run.transcript_source),
            md_text(x.chunk_strategy),
            _created(x),
            f"{x.input_tokens} / {x.output_tokens}",
            f"{x.duration_ms}" if x.duration_ms is not None else "not recorded",
            f"{x.cost_usd:.4f}" if x.cost_usd is not None else "not recorded",
            f"{len(x.topics)} / {len(x.claims)} / {len(x.quotes)}",
            f"{_count_unknown(x)} of {len(x.claims)}",
            _confidence_cell(x),
            str(x.speakers_coerced) if x.speakers_coerced is not None else "not recorded",
        ]

    names = [
        "Transcript source",
        "Chunk strategy",
        "Created (UTC)",
        "Tokens (in / out)",
        "Duration (ms)",
        "Cost (USD)",
        "Topics / claims / quotes",
        "Unattributed claims",
        "Claim confidence (high / medium / low / not given)",
        "Speakers coerced to unknown",
    ]
    ca, cb = cells(a), cells(b)
    return _table(["", "A", "B"], [[n, x, y] for n, x, y in zip(names, ca, cb, strict=True)])


def _item_line(video_id: str, side: str, item: Claim | Quote) -> str:
    link = timestamp_link(video_id, item.start_sec)
    stamp = f"[{format_timestamp(item.start_sec or 0)}]({link})" if link and item.start_sec is not None else None
    speaker = "Unattributed" if item.speaker == UNKNOWN else md_text(item.speaker)
    parts = [f"**{side}**"]
    if stamp:
        parts.append(stamp)
    parts.append(speaker)
    if isinstance(item, Claim):
        parts.append(md_text(item.confidence) if item.confidence else "confidence not given")
    parts.append(md_text(item.text))
    return "- " + " · ".join(parts)


def _windowed(
    video_id: str,
    heading: str,
    empty: str,
    a: AnalysisRun | None,
    b: AnalysisRun | None,
    pick: Any,
    noun: str,
) -> list[str]:
    windows = group_windows(
        pick(a.analysis) if a else (), pick(b.analysis) if b else ()
    )
    lines = [f"### {heading}", ""]
    if not windows:
        return [*lines, empty, ""]
    for window in windows:
        lines += [f"#### {'No timestamp' if window.index is None else window_label(window.index)}", ""]
        for side, run, items in (("A", a, window.a), ("B", b, window.b)):
            if run is None:
                continue
            if items:
                lines += [_item_line(video_id, side, item) for item in items]
            else:
                lines.append(f"- **{side}** no {noun} in this window")
        lines.append("")
    return lines


def _run_section(side: str, label: RunLabel, run: AnalysisRun | None, video_id: str) -> list[str]:
    if run is None:
        return [f"### Run {side}", "", f"No analysis for {label.text()}", ""]
    x = run.analysis
    lines = [f"### Run {side}", "", "Roster", "", *_roster_lines(x.speaker_roster), ""]
    lines += [f"**TL;DR {side}:** {md_text(x.tldr) or 'empty'}", "", "Topics", ""]
    if x.topics:
        for topic in x.topics:
            link = timestamp_link(video_id, topic.start_sec)
            stamp = (
                f"[{format_timestamp(topic.start_sec or 0)}]({link}) " if link else ""
            )
            lines.append(f"- {stamp}{md_text(topic.title)}")
    else:
        lines.append("No topics")
    return [*lines, ""]


def _video_section(entry: VideoEntry, a: RunLabel, b: RunLabel) -> list[str]:
    vid = md_text(entry.video_id)
    if entry.title is None and entry.run_a is None and entry.run_b is None:
        return [f"## {vid}: not in the database", ""]
    name = f"{md_text(entry.title)} ({vid})" if entry.title else vid
    if entry.run_a is None and entry.run_b is None:
        return [f"## {name}: no analysis for either run", ""]
    lines = [f"## {name}", ""]
    if is_video_id(entry.video_id):
        lines += [f"[Watch on YouTube](https://www.youtube.com/watch?v={entry.video_id})", ""]
    lines += _details_table(entry.run_a, entry.run_b) + [""]
    if entry.run_a is not None and entry.run_b is not None:
        x, y = entry.run_a.analysis, entry.run_b.analysis
        if x.transcript_id != y.transcript_id:
            lines += ["Notice: A and B used different transcripts.", ""]
        if x.chunk_strategy != y.chunk_strategy:
            lines += ["Notice: A and B used different chunk strategies.", ""]
    lines += _run_section("A", a, entry.run_a, entry.video_id)
    lines += _run_section("B", b, entry.run_b, entry.video_id)
    lines += _windowed(
        entry.video_id, "Claims", "No claims.", entry.run_a, entry.run_b, lambda r: r.claims, "claims"
    )
    lines += _windowed(
        entry.video_id, "Quotes", "No quotes.", entry.run_a, entry.run_b, lambda r: r.quotes, "quotes"
    )
    return lines


def render_report(a: RunLabel, b: RunLabel, entries: Sequence[VideoEntry]) -> str:
    """The whole report. ``entries`` are in golden-set order."""
    missing_a = sum(1 for e in entries if e.run_a is None)
    missing_b = sum(1 for e in entries if e.run_b is None)
    lines = [
        "# Summarizer comparison",
        "",
        f"- **A**: {a.text()}",
        f"- **B**: {b.text()}",
        f"- Videos in the golden set: {len(entries)}",
        f"- Missing A: {missing_a}",
        f"- Missing B: {missing_b}",
        "",
        "## Summary",
        "",
        *_summary_table(entries),
        "",
    ]
    for entry in entries:
        lines += _video_section(entry, a, b)
    return "\n".join(lines).rstrip("\n") + "\n"
