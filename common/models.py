"""Domain dataclasses shared across services and adapters."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Segment:
    """A timed piece of transcript text; times are seconds from video start."""

    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass(frozen=True, slots=True)
class VideoMeta:
    """What the metadata source reports about one video."""

    video_id: str
    channel_id: str
    title: str
    description: str
    duration_sec: int | None
    published_at: datetime | None
    language: str | None
    live_status: str | None
    manual_subtitle_langs: tuple[str, ...]
    auto_caption_langs: tuple[str, ...]
    #: The scheduled start of a premiere or live stream (yt-dlp's
    #: ``release_timestamp``), aware UTC; ``None`` when unknown (#65).
    release_at: datetime | None = None
    #: yt-dlp's ``media_type == "short"``; never inferred from the duration (#80).
    is_short: bool = False


class MetadataSource(Protocol):
    def fetch(self, video_id: str) -> VideoMeta: ...


# FeedEntry and ChannelFeed belong to #26 (channel RSS adapter,
# adapters/youtube/feed.py). #34 (the planner) consumes the port; its
# grooming note calls the Protocol FeedSource, kept below as an alias.


@dataclass(frozen=True, slots=True)
class FeedEntry:
    """One video listed in a channel's RSS feed; ``published_at`` is aware UTC."""

    video_id: str
    channel_id: str
    title: str
    published_at: datetime
    #: The entry's alternate link has the path ``/shorts/<video_id>`` (#80).
    is_short: bool = False


class ChannelFeed(Protocol):
    def fetch(self, channel_id: str) -> list[FeedEntry]: ...


FeedSource = ChannelFeed


# CatalogEntry, ChannelCatalog and CatalogSource belong to #27 (channel catalog
# adapter, adapters/youtube/catalog.py). #41 (backfill) consumes the port.


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One upload in a channel listing; no per-video metadata is fetched."""

    video_id: str
    title: str | None
    duration_sec: int | None


@dataclass(frozen=True, slots=True)
class ChannelCatalog:
    """A channel's uploads, newest first; ``total_count`` is YouTube's own figure."""

    channel_id: str
    entries: tuple[CatalogEntry, ...]
    total_count: int | None


class CatalogSource(Protocol):
    def list_uploads(self, channel_id: str, *, limit: int) -> ChannelCatalog:
        """List a channel's uploads, newest first, or raise on failure.

        ``PermanentSourceError(REMOVED)`` means "nothing listable": the
        channel ID may never have existed, the channel may have been
        terminated, or it may simply have no uploads — including a channel
        whose uploads are all private, members-only, or Shorts, which is
        unrecorded but presumed to land here too. It is never proof that the
        channel was removed. Callers must treat it as an empty listing and
        must not persist it as channel or video state (architecture.md
        §8.3.1, #107).
        """
        ...


# Channel, Transcript, Chunk, Analysis, Topic, Claim and Quote, below,
# belong to #14 (the repository layer, common/repo/). They mirror the
# tables in architecture.md §6 one-for-one; common/repo/ functions build
# and return these instead of ever returning a raw row or tuple.


@dataclass(frozen=True, slots=True)
class Channel:
    """A monitored (or once-monitored) YouTube channel."""

    channel_id: str
    title: str | None
    active: bool
    monitor_from: datetime
    last_polled: datetime | None
    last_poll_err: str | None
    added_at: datetime
    poll_failing_since: datetime | None  # start of the current failed-poll run (#90)
    poll_failures: int  # failed polls in a row (#90)
    feed_gap_detected_at: datetime | None = None
    feed_gap_reason: str | None = None


@dataclass(frozen=True, slots=True)
class Transcript:
    """A saved transcript for one video, from one source (architecture.md §6)."""

    id: int
    video_id: str
    source: str  # youtube_manual | youtube_auto | whisper
    language: str | None
    speaker_source: str  # subtitle_labels | none  (C3)
    segments: tuple[Segment, ...]
    full_text: str
    engine_meta: dict[str, object] | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Chunk:
    """One window of a transcript under a given ``chunk_strategy`` (§7.3).

    ``id``, ``transcript_id`` and ``chunk_strategy`` default to ``None``
    because ``save_chunks`` (``common/repo/transcripts.py``) only needs
    ``seq``/``start_sec``/``end_sec``/``text`` from its caller - the
    transcript id and strategy are already that function's own arguments.
    ``get_chunks`` always returns all three set.
    """

    seq: int
    start_sec: float
    end_sec: float
    text: str
    chunk_strategy: str | None = None
    transcript_id: int | None = None
    id: int | None = None


@dataclass(frozen=True, slots=True)
class Topic:
    """One topic under an analysis, ordered by ``seq``."""

    seq: int
    title: str
    summary: str | None = None
    start_sec: float | None = None
    id: int | None = None


@dataclass(frozen=True, slots=True)
class Claim:
    """One claim under an analysis, ordered by ``start_sec``."""

    text: str
    speaker: str = "unknown"
    start_sec: float | None = None
    confidence: str | None = None  # high | medium | low
    source_chunk_seq: int | None = None
    id: int | None = None


@dataclass(frozen=True, slots=True)
class Quote:
    """One quote under an analysis."""

    text: str
    speaker: str = "unknown"
    start_sec: float | None = None
    source_chunk_seq: int | None = None
    id: int | None = None


# RosterSpeaker, Roster and ChunkAnalysis belong to #22 (LLM output schema,
# adapters/summarize/schema.py). They are what its parsers return, so they
# live here beside Topic/Claim/Quote, keeping pydantic types out of every
# public signature.


@dataclass(frozen=True, slots=True)
class RosterSpeaker:
    """One named speaker from the roster pass (architecture.md 8.1)."""

    name: str
    role: str  # host | guest | panelist | unknown


@dataclass(frozen=True, slots=True)
class Roster:
    """The speakers of one video; every later ``speaker`` is one of these or ``unknown``."""

    speakers: tuple[RosterSpeaker, ...]

    def to_json(self) -> dict[str, object]:
        """The value stored in ``analyses.speaker_roster``."""
        return {"speakers": [{"name": s.name, "role": s.role} for s in self.speakers]}


@dataclass(frozen=True, slots=True)
class ChunkAnalysis:
    """A validated per-chunk analysis plus what validation had to repair.

    ``speaker_coercions`` counts speakers replaced by ``unknown`` because they
    were not in the roster; ``start_sec_clamped`` counts timestamps moved into
    the chunk span; ``items_dropped`` counts items removed for being blank,
    over a length cap or past a list cap.
    """

    topics: tuple[Topic, ...]
    claims: tuple[Claim, ...]
    quotes: tuple[Quote, ...]
    speaker_coercions: int = 0
    start_sec_clamped: int = 0
    items_dropped: int = 0


@dataclass(frozen=True, slots=True)
class Analysis:
    """One LLM analysis pass over a video's transcript, with its children.

    ``id`` and ``created_at`` default to ``None``: they are unset until
    ``save_analysis`` (``common/repo/analyses.py``) persists the row.
    ``latest_analysis`` always returns both populated.
    """

    video_id: str
    transcript_id: int
    chunk_strategy: str
    model: str
    prompt_version: str
    tldr: str
    speaker_roster: dict[str, object] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    duration_ms: int | None = None
    #: Speakers the analyzer coerced to ``unknown`` (#60); ``None`` = not recorded.
    speakers_coerced: int | None = None
    topics: tuple[Topic, ...] = ()
    claims: tuple[Claim, ...] = ()
    quotes: tuple[Quote, ...] = ()
    id: int | None = None
    created_at: datetime | None = None


# AudioRef and AudioSource, below, belong to #19 (audio acquisition and
# normalization). #19's own file list names only adapters/youtube/audio.py
# and its tests, but the port and its return type live in common/models.py
# per architecture.md §3 - the same place #16 put MetadataSource beside
# VideoMeta. #20's Transcriber port also consumes AudioRef, so it cannot
# live in adapters/youtube/ (services/ -> adapters/ -> common/, AGENTS.md).


@dataclass(frozen=True, slots=True)
class AudioRef:
    """Where normalized audio ended up, and its real duration.

    ``rel_path`` is relative to ``AUDIO_DIR`` (#10's storage rule), e.g.
    ``"ab/abc123def45.opus"`` - never an absolute path. ``bytes`` is the
    size of the file on disk. ``duration_sec`` is measured by ``ffprobe``
    on the normalized audio itself (#19), never taken from video metadata.
    """

    rel_path: str
    bytes: int
    duration_sec: float


class AudioSource(Protocol):
    # returns 16 kHz mono opus, relative to `dest` (architecture.md 3, D6b)
    def fetch_normalized(self, video_id: str, dest: Path) -> AudioRef: ...


# TranscriptResult/Transcriber belong to #20 (faster-whisper adapter). Same
# reasoning as AudioRef/AudioSource above: the port and its return type live
# here per architecture.md 3, next to what they consume (AudioRef, Segment).


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    """What a ``Transcriber`` returns for one ``AudioRef``.

    ``segments`` are in time order, whitespace-trimmed, with empty segments
    already dropped. ``language`` is the language actually used for decoding
    - the caller's request if one was given, otherwise the detected one.
    ``engine_meta`` is a JSON-safe dict persisted verbatim to
    ``transcripts.engine_meta`` (architecture.md 7.2), e.g. engine name and
    version, model, compute_type, beam_size, vad, threads, audio_sec,
    load_sec, transcribe_sec and rtf.
    """

    segments: tuple[Segment, ...]
    language: str
    engine_meta: dict[str, Any]


class Transcriber(Protocol):
    def transcribe(
        self,
        audio: AudioRef,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> TranscriptResult: ...


# Usage and Summarizer belong to #24 (Ollama adapter, adapters/summarize/).
# It is built first, so it defines the port that #25 (Anthropic adapter) and
# #30 (analyze handler) share. Departs from architecture.md 3 in two additive
# ways: the ``model`` attribute and ``take_usage()``.


@dataclass(frozen=True, slots=True)
class Usage:
    """LLM usage since the previous take_usage(); every field is a total."""

    input_tokens: int = 0  # uncached input only
    output_tokens: int = 0
    cache_read_tokens: int = 0  # always 0 for Ollama
    cache_write_tokens: int = 0  # always 0 for Ollama
    calls: int = 0  # model requests that returned a response
    cost_usd: float | None = None  # None when no price is known (always for Ollama)


class Summarizer(Protocol):
    name: str  # the SUMMARIZER value: "ollama" | "anthropic" (analyze dedupe_key, C1)
    model: str  # the concrete model, recorded as analyses.model

    def derive_roster(self, meta: VideoMeta, opening: str) -> Roster: ...

    def analyze_chunk(
        self, chunk: Chunk, roster: Roster, meta: VideoMeta
    ) -> ChunkAnalysis: ...

    def reduce(self, partials: list[ChunkAnalysis], meta: VideoMeta) -> str: ...

    def take_usage(self) -> Usage: ...  # totals since the last call, then reset to zero


@runtime_checkable
class ChunkBatchSummarizer(Summarizer, Protocol):
    """A summarizer that can analyze every chunk of an episode in one call (#88).

    Optional: the analyze handler uses it when present and keeps the per-chunk
    loop otherwise, so Ollama and the fake do not implement it.
    """

    def analyze_chunks(
        self, chunks: Sequence[Chunk], roster: Roster, meta: VideoMeta
    ) -> list[ChunkAnalysis]:
        """One analysis per chunk, in input order; ``[]`` for no chunks."""
        ...


@dataclass(frozen=True, slots=True)
class BatchCheckpoint:
    """A Message Batch a job's step submitted, recorded before the first poll (#89)."""

    step_key: str
    batch_id: str
    request_sha256: str
    submitted_at: datetime


class JobScope(Protocol):
    """What a checkpointing summarizer needs from one analyze job (#89).

    The handler builds it from the job's own connection; the adapter never
    touches the database itself.
    """

    job_id: int

    def cancelled(self) -> bool: ...

    def now(self) -> datetime: ...

    def checkpoint(self, step_key: str) -> BatchCheckpoint | None: ...

    def record(self, step_key: str, batch_id: str, request_sha256: str) -> BatchCheckpoint:
        """Store a batch for ``step_key``; the stored row wins if there already is one."""
        ...

    def forget(self, step_key: str) -> None: ...


@runtime_checkable
class JobScopedSummarizer(Protocol):
    """A summarizer that checkpoints its batches per job (#89). Optional, like ``ChunkBatchSummarizer``."""

    def begin_job(self, scope: JobScope) -> None: ...

    def end_job(self) -> None: ...


#: What cancelling an orphaned batch came to (#160). ``cancelled`` and
#: ``not_cancellable`` both mean the row can go; ``retry_later`` keeps it.
type OrphanOutcome = Literal["cancelled", "not_cancellable", "retry_later"]


@runtime_checkable
class OrphanBatchCanceller(Protocol):
    """A summarizer that can cancel a batch no job is waiting on (#160). Optional.

    Must read and write no per-job state: it is called from another thread
    while a job runs. Never raises.
    """

    def cancel_orphan_batch(self, batch_id: str) -> OrphanOutcome: ...


# ActiveJob, JobFailure, VideoSummary, VideoPage, AnalysisRun, TranscriptInfo,
# IndexedSegment and TranscriptPage belong to #42 (read endpoints). They are
# what the common/repo/ read functions behind GET /videos... return.

VIDEO_STATUSES = ("done", "processing", "unavailable", "failed", "idle")


@dataclass(frozen=True, slots=True)
class ActiveJob:
    """A ``pending`` or ``running`` job of a video."""

    id: int
    kind: str
    state: str


@dataclass(frozen=True, slots=True)
class JobFailure:
    """A video's newest ``dead`` job. ``jobs.last_error`` is deliberately absent."""

    job_id: int
    kind: str
    error_class: str | None
    finished_at: datetime | None


@dataclass(frozen=True, slots=True)
class VideoSummary:
    """One ``videos`` row plus its derived processing status (#42)."""

    video_id: str
    title: str | None
    channel_id: str | None
    channel_title: str | None
    published_at: datetime | None
    duration_sec: int | None
    origin: str
    unavailable: str | None
    status: str  # one of VIDEO_STATUSES
    active_job: ActiveJob | None
    last_failure: JobFailure | None
    latest_analysis_at: datetime | None
    has_transcript: bool = False


@dataclass(frozen=True, slots=True)
class VideoPage:
    items: tuple[VideoSummary, ...]
    total: int


@dataclass(frozen=True, slots=True)
class ChannelListItem:
    """One row of ``GET /channels`` (#119): no poll fields, those are #158's."""

    channel_id: str
    title: str | None
    active: bool
    video_count: int


@dataclass(frozen=True, slots=True)
class ChannelPage:
    items: tuple[ChannelListItem, ...]
    total: int


@dataclass(frozen=True, slots=True)
class AnalysisRun:
    """An analysis plus the ``source`` of the transcript it was made from."""

    analysis: Analysis
    transcript_source: str


@dataclass(frozen=True, slots=True)
class TranscriptInfo:
    id: int
    source: str
    language: str | None
    speaker_source: str
    segment_count: int


@dataclass(frozen=True, slots=True)
class IndexedSegment:
    """A transcript segment and its absolute 0-based position."""

    index: int
    start: float
    end: float
    text: str
    speaker: str | None


@dataclass(frozen=True, slots=True)
class TranscriptPage:
    video_id: str
    info: TranscriptInfo
    segments: tuple[IndexedSegment, ...]
