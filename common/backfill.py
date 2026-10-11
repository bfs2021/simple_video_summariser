"""Channel backfill (issues #41, #115; architecture.md 4, 8.3; D9b, C4).

Shared by ``POST /channels/{id}/backfill`` (``services.api``, synchronous) and
the planner's ``channel_backfill`` task (``services.planner.backfill``): moved
here from ``services/api/backfill.py`` so the planner need not import a
service. It depends only on the ``CatalogSource`` port, never on an adapter.

``backfill`` lists a channel's newest ``limit`` uploads through a
``CatalogSource`` and, unless it is a dry run, gives every video the system
does not know yet a ``videos`` row (``origin = 'backfill'``) and one
``ingest`` job at ``PRIORITY_BACKFILL``. It is plain Python with no HTTP, so
the route only maps its result and errors.

Transactions:

- The channel-existence check is committed before the listing starts, so a
  slow yt-dlp run holds no lock or snapshot.
- Each new video is one transaction: row, then job only if this call
  inserted the row. A crash or a database error leaves every video either
  complete or absent, and a repeat call fills in the rest. A concurrent
  writer (RSS, another backfill) that inserted the row first wins; its job
  and origin are left alone.

A ``PermanentSourceError`` from the catalog means "The playlist does not
exist", which yt-dlp prints for a channel with no uploads as well as for a
removed one (#27, #107), so it is reported as an empty listing.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

import psycopg
import structlog
from psycopg.pq import TransactionStatus

from common.errors import (
    PermanentSourceError,
    RateLimitedError,
    ToolFailureError,
    TransientNetworkError,
)
from common.models import (
    CatalogEntry,
    CatalogSource,
    ChannelCatalog,
    MetadataSource,
    VideoMeta,
)
from common.queue import PRIORITY_BACKFILL, JobQueue
from common.repo.analyses import median_cost_per_char
from common.repo.channels import channel_exists
from common.repo.transcripts import median_chars_per_audio_sec, median_whisper_rtf
from common.repo.videos import insert_backfill_video, known_video_ids

_log = structlog.get_logger(__name__)

MAX_BACKFILL_LIMIT = 500
#: ``POST /channels/{id}/backfill-requests`` (#115); equals the catalog adapter's
#: ``MAX_CATALOG_LIMIT`` (checked by a test: ``common/`` never imports adapters).
MAX_BACKFILL_REQUEST_LIMIT = 5000
ESTIMATE_ASSUMES = (
    "manual-subtitle videos skip speech-to-text; unchecked videos are assumed to need it"
)
LLM_BASIS = "median recorded analyses.cost_usd per transcript character"

#: Dry-run time budget (#117): measured from the start of ``backfill`` and
#: covering the listing. Each probe's yt-dlp timeout is
#: ``min(BACKFILL_PROBE_TIMEOUT_SEC, remaining / 2)`` because
#: ``YouTubeMetadata.fetch`` may run yt-dlp twice for an upcoming video.
BACKFILL_DRY_RUN_DEADLINE_SEC = 100
BACKFILL_PROBE_TIMEOUT_SEC = 10
BACKFILL_PROBE_CONCURRENCY = 4

SUBTITLES_MANUAL = "manual"
SUBTITLES_TRANSCRIBE = "transcribe"
SUBTITLES_UNCHECKED = "unchecked"


class ChannelNotRegistered(Exception):
    """The channel has no ``channels`` row; register it with ``POST /channels``."""


@dataclass(frozen=True, slots=True)
class SubtitleProbe:
    """What a dry run needs to check one video's subtitles (#117).

    ``metadata_factory(timeout)`` builds a ``MetadataSource`` whose yt-dlp
    runs are bounded by ``timeout``; ``has_manual(meta)`` is the pure
    ``YouTubeSubtitles.available(meta).manual is not None``.
    """

    metadata_factory: Callable[[float], MetadataSource]
    has_manual: Callable[[VideoMeta], bool]


@dataclass(frozen=True, slots=True)
class LlmEstimate:
    summarizer: str
    model: str
    chars_per_audio_sec: float | None
    chars_samples: int
    cost_per_char: float | None
    cost_samples: int
    est_input_tokens: int | None
    cost_usd: float | None
    basis: str = LLM_BASIS


def _no_llm_estimate() -> LlmEstimate:
    return LlmEstimate("ollama", "", None, 0, None, 0, None, None)


@dataclass(frozen=True, slots=True)
class VideoEstimate:
    video_id: str
    duration_sec: int | None
    subtitles: str
    transcription_sec: int | None
    est_input_tokens: int | None
    llm_cost_usd: float | None


@dataclass(frozen=True, slots=True)
class Estimate:
    audio_sec: int
    unknown_duration: int
    rtf: float | None
    rtf_samples: int
    transcription_sec: int | None
    assumes: str = ESTIMATE_ASSUMES
    probed: int = 0
    manual_subtitles: int = 0
    needs_transcription: int = 0
    unchecked: int = 0
    probe_failed: int = 0
    deadline_reached: bool = False
    transcribe_audio_sec: int = 0
    llm: LlmEstimate = field(default_factory=_no_llm_estimate)


@dataclass(frozen=True, slots=True)
class BackfillResult:
    channel_id: str
    dry_run: bool
    limit: int
    listed: int
    total_count: int | None
    already_known: int
    new_videos: int
    enqueued: int
    video_ids: tuple[str, ...]
    estimate: Estimate
    backfill_id: str | None
    videos: tuple[VideoEstimate, ...] = ()


def backfill(
    conn: psycopg.Connection[Any],
    queue: JobQueue,
    catalog: CatalogSource,
    channel_id: str,
    *,
    limit: int,
    dry_run: bool,
    probe: SubtitleProbe | None = None,
    prefer_whisper: bool = False,
    summarizer: str = "ollama",
    model: str = "",
    clock: Callable[[], float] = time.monotonic,
    backfill_id: str | None = None,
    max_enqueue: int | None = None,
) -> BackfillResult:
    """Select the channel's unknown uploads and, unless ``dry_run``, enqueue them.

    ``conn`` must have no open transaction; ``queue`` must write through it.
    Raises ``ChannelNotRegistered`` before listing if the channel has no row.
    Catalog errors other than ``PermanentSourceError`` propagate unchanged.

    A dry run also probes each new video's subtitles (newest first, bounded by
    ``BACKFILL_DRY_RUN_DEADLINE_SEC`` on ``clock``) when ``probe`` is given and
    ``prefer_whisper`` is false; the DB reads are one short transaction and no
    transaction is open while probes run. A real run probes nothing.

    A real run tags its jobs with ``backfill_id`` (a fresh UUID unless the
    caller passes one; the planner passes one stable id per request, #115).
    ``max_enqueue`` bounds how many of the new videos a real run writes (the
    planner's per-run slice); ``new_videos`` and ``video_ids`` still describe
    the whole listing, ``enqueued`` what was written. ``None`` writes them all.
    """
    started = clock()
    if conn.info.transaction_status != TransactionStatus.IDLE:
        raise RuntimeError("backfill expects a connection with no open transaction")
    with conn.transaction():
        registered = channel_exists(conn, channel_id)
    if not registered:
        raise ChannelNotRegistered(channel_id)

    listing = _list(catalog, channel_id, limit)

    with conn.transaction():
        known = known_video_ids(conn, [e.video_id for e in listing.entries])
        rtf, rtf_samples = median_whisper_rtf(conn)
        chars_per_sec, chars_samples = median_chars_per_audio_sec(conn)
        cost_per_char, cost_samples = (
            median_cost_per_char(conn, model) if summarizer == "anthropic" else (None, 0)
        )
    new = [e for e in listing.entries if e.video_id not in known]

    if dry_run and prefer_whisper:
        outcome = _ProbeOutcome({e.video_id: SUBTITLES_TRANSCRIBE for e in new})
    elif dry_run and probe is not None:
        outcome = _probe_videos(new, probe, clock, started, channel_id)
    else:
        outcome = _ProbeOutcome({})

    estimate, videos = _estimate(
        new,
        outcome,
        rtf=rtf,
        rtf_samples=rtf_samples,
        chars_per_sec=chars_per_sec,
        chars_samples=chars_samples,
        cost_per_char=cost_per_char,
        cost_samples=cost_samples,
        summarizer=summarizer,
        model=model,
    )

    if dry_run:
        backfill_id = None
    elif backfill_id is None:
        backfill_id = str(uuid.uuid4())
    enqueued = 0
    if backfill_id is not None:
        payload = {"origin": "backfill", "backfill_id": backfill_id}
        for entry in new if max_enqueue is None else new[: max(max_enqueue, 0)]:
            with conn.transaction():
                if insert_backfill_video(conn, entry, channel_id=channel_id):
                    job = queue.enqueue(
                        "ingest", entry.video_id, payload=payload, priority=PRIORITY_BACKFILL
                    )
                    if job is not None:
                        enqueued += 1

    return BackfillResult(
        channel_id=channel_id,
        dry_run=dry_run,
        limit=limit,
        listed=len(listing.entries),
        total_count=listing.total_count,
        already_known=len(listing.entries) - len(new),
        new_videos=len(new),
        enqueued=enqueued,
        video_ids=tuple(e.video_id for e in new),
        estimate=estimate,
        backfill_id=backfill_id,
        videos=videos,
    )


def _list(catalog: CatalogSource, channel_id: str, limit: int) -> ChannelCatalog:
    try:
        return catalog.list_uploads(channel_id, limit=limit)
    except PermanentSourceError as exc:
        _log.info(
            "backfill listing empty: playlist does not exist",
            channel_id=channel_id,
            reason=exc.reason.value,
        )
        return ChannelCatalog(channel_id, (), None)


@dataclass(slots=True)
class _ProbeOutcome:
    status: dict[str, str]
    failed: int = 0
    probed: int = 0
    deadline_reached: bool = False


def _probe_one(probe: SubtitleProbe, video_id: str, timeout: float) -> bool:
    return probe.has_manual(probe.metadata_factory(timeout).fetch(video_id))


def _probe_videos(
    new: Sequence[CatalogEntry],
    probe: SubtitleProbe,
    clock: Callable[[], float],
    started: float,
    channel_id: str,
) -> _ProbeOutcome:
    """Probe ``new`` newest-first until the deadline; see the module constants.

    No probe starts once ``clock() - started`` reaches the deadline. At most
    ``BACKFILL_PROBE_CONCURRENCY`` run at once. A per-video failure leaves the
    video unchecked; ``RateLimitedError`` stops new probes (running ones
    finish); any other exception propagates.
    """
    outcome = _ProbeOutcome({})
    pending = deque(new)
    inflight: dict[Any, str] = {}
    rate_limited = False
    with ThreadPoolExecutor(max_workers=BACKFILL_PROBE_CONCURRENCY) as pool:
        while inflight or (pending and not rate_limited):
            while pending and not rate_limited and len(inflight) < BACKFILL_PROBE_CONCURRENCY:
                remaining = BACKFILL_DRY_RUN_DEADLINE_SEC - (clock() - started)
                if remaining <= 0:
                    outcome.deadline_reached = True
                    pending.clear()
                    break
                entry = pending.popleft()
                timeout = min(BACKFILL_PROBE_TIMEOUT_SEC, remaining / 2)
                inflight[pool.submit(_probe_one, probe, entry.video_id, timeout)] = entry.video_id
            if not inflight:
                break
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                video_id = inflight.pop(future)
                try:
                    manual = future.result()
                except RateLimitedError:
                    rate_limited = True
                except (PermanentSourceError, TransientNetworkError, ToolFailureError, ValueError):
                    outcome.failed += 1
                except Exception:
                    _log.exception("backfill probe failed", channel_id=channel_id)
                    raise
                else:
                    outcome.probed += 1
                    outcome.status[video_id] = SUBTITLES_MANUAL if manual else SUBTITLES_TRANSCRIBE
    if rate_limited:
        _log.warning(
            "backfill.probe_rate_limited", channel_id=channel_id, probed=outcome.probed
        )
    return outcome


def _round_int(value: float) -> int | None:
    return round(value) if math.isfinite(value) else None


def _round_cost(value: float) -> float | None:
    return round(value, 5) if math.isfinite(value) else None


def _estimate(
    new: list[CatalogEntry],
    outcome: _ProbeOutcome,
    *,
    rtf: float | None,
    rtf_samples: int,
    chars_per_sec: float | None,
    chars_samples: int,
    cost_per_char: float | None,
    cost_samples: int,
    summarizer: str,
    model: str,
) -> tuple[Estimate, tuple[VideoEstimate, ...]]:
    """Per-video and total estimates. Pure: every float is checked finite."""
    videos: list[VideoEstimate] = []
    for entry in new:
        status = outcome.status.get(entry.video_id, SUBTITLES_UNCHECKED)
        duration = entry.duration_sec
        if status == SUBTITLES_MANUAL:
            transcription: int | None = 0
        elif rtf is None or duration is None:
            transcription = None
        else:
            transcription = _round_int(duration * rtf)
        chars = (
            None
            if chars_per_sec is None or duration is None
            else _round_int(duration * chars_per_sec)
        )
        cost = (
            None
            if chars is None or cost_per_char is None or summarizer != "anthropic"
            else _round_cost(chars * cost_per_char)
        )
        videos.append(
            VideoEstimate(
                video_id=entry.video_id,
                duration_sec=duration,
                subtitles=status,
                transcription_sec=transcription,
                est_input_tokens=None if chars is None else chars // 4,
                llm_cost_usd=cost,
            )
        )

    durations = [e.duration_sec for e in new if e.duration_sec is not None]
    transcribe_audio = sum(
        v.duration_sec
        for v in videos
        if v.subtitles != SUBTITLES_MANUAL and v.duration_sec is not None
    )
    manual = sum(v.subtitles == SUBTITLES_MANUAL for v in videos)
    transcribe = sum(v.subtitles == SUBTITLES_TRANSCRIBE for v in videos)
    tokens = [v.est_input_tokens for v in videos]
    costs = [v.llm_cost_usd for v in videos]
    total_tokens = None if any(t is None for t in tokens) else sum(t for t in tokens if t is not None)
    total_cost = (
        None
        if any(c is None for c in costs)
        else _round_cost(sum(c for c in costs if c is not None))
    )
    estimate = Estimate(
        audio_sec=sum(durations),
        unknown_duration=len(new) - len(durations),
        rtf=rtf,
        rtf_samples=rtf_samples,
        transcription_sec=None if rtf is None else _round_int(transcribe_audio * rtf),
        probed=outcome.probed,
        manual_subtitles=manual,
        needs_transcription=transcribe,
        unchecked=len(new) - manual - transcribe,
        probe_failed=outcome.failed,
        deadline_reached=outcome.deadline_reached,
        transcribe_audio_sec=transcribe_audio,
        llm=LlmEstimate(
            summarizer=summarizer,
            model=model,
            chars_per_audio_sec=chars_per_sec,
            chars_samples=chars_samples,
            cost_per_char=cost_per_char,
            cost_samples=cost_samples,
            est_input_tokens=total_tokens,
            cost_usd=total_cost,
        ),
    )
    return estimate, tuple(videos)
