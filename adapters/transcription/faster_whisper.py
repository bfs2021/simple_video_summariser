"""faster-whisper implementation of the ``Transcriber`` port (issue #20).

CPU only: int8 compute, beam size 5, VAD filtering (D3, architecture.md 7.2).
Importing this module never imports ``faster_whisper`` - only the first
``transcribe()`` call does, lazily, so an image without
``requirements.whisper.txt`` installed can still import it (task #56 builds
the transcriber image separately). Constructing a transcriber never loads
the model either; the first ``transcribe()`` call loads it once, and later
calls on the same instance reuse it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib import metadata
from time import perf_counter
from typing import Any

from common.config import get_settings
from common.errors import ToolFailureError, TransientNetworkError
from common.models import AudioRef, Segment, TranscriptResult

#: Decoding options fixed by D3 - not configurable per the issue's AC.
_BEAM_SIZE = 5
_VAD_FILTER = True

_REQUIREMENTS_FILE = "requirements.whisper.txt"

#: The codes Whisper accepts (faster-whisper rejects anything else, including
#: region-tagged ones like ``en-US``).
_WHISPER_LANGUAGES = frozenset(
    [
        "af",
        "am",
        "ar",
        "as",
        "az",
        "ba",
        "be",
        "bg",
        "bn",
        "bo",
        "br",
        "bs",
        "ca",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fo",
        "fr",
        "gl",
        "gu",
        "ha",
        "haw",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        "is",
        "it",
        "ja",
        "jw",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "la",
        "lb",
        "ln",
        "lo",
        "lt",
        "lv",
        "mg",
        "mi",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "mt",
        "my",
        "ne",
        "nl",
        "nn",
        "no",
        "oc",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sa",
        "sd",
        "si",
        "sk",
        "sl",
        "sn",
        "so",
        "sq",
        "sr",
        "su",
        "sv",
        "sw",
        "ta",
        "te",
        "tg",
        "th",
        "tk",
        "tl",
        "tr",
        "tt",
        "uk",
        "ur",
        "uz",
        "vi",
        "yi",
        "yo",
        "zh",
        "yue",
    ]
)

#: Old ISO 639-1 codes YouTube still emits, and Javanese, which Whisper spells "jw".
_LANGUAGE_ALIASES = {"iw": "he", "in": "id", "ji": "yi", "jv": "jw"}


def whisper_language(code: str | None) -> str | None:
    """Reduce a BCP-47 tag (``en-US``, ``pt_BR``, ``zh-Hans``) to a Whisper code.

    ``None`` for an empty or unknown code, which makes Whisper detect the
    language instead of failing the job.
    """
    if code is None:
        return None
    primary = code.strip().lower().replace("_", "-").split("-", 1)[0]
    primary = _LANGUAGE_ALIASES.get(primary, primary)
    return primary if primary in _WHISPER_LANGUAGES else None


class FasterWhisperTranscriber:
    """``Transcriber`` backed by faster-whisper on CPU (architecture.md 3, 7.2)."""

    def __init__(
        self,
        model: str | None = None,
        compute_type: str | None = None,
        threads: int | None = None,
    ) -> None:
        # Any explicit arg is used as-is; only a missing one reads settings,
        # so importing/constructing this class never forces get_settings()
        # to validate the environment (common/config.py's own contract).
        if model is None or compute_type is None or threads is None:
            settings = get_settings()
            model = settings.WHISPER_MODEL if model is None else model
            compute_type = (
                settings.WHISPER_COMPUTE if compute_type is None else compute_type
            )
            threads = settings.WHISPER_THREADS if threads is None else threads

        self._model_name = model
        self._compute_type = compute_type
        self._threads = threads
        self._loaded_model: Any = None
        self._engine_version: str = "unknown"

    def transcribe(
        self,
        audio: AudioRef,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> TranscriptResult:
        model, engine_version, load_sec = self._ensure_model()
        language = whisper_language(language)

        audio_path = get_settings().AUDIO_DIR / audio.rel_path

        start = perf_counter()
        try:
            raw_segments, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=_BEAM_SIZE,
                vad_filter=_VAD_FILTER,
            )
        except MemoryError:
            raise
        except Exception as exc:
            raise ToolFailureError(
                f"faster-whisper failed to transcribe {audio_path}: {exc}"
            ) from exc

        segments = list(_collect_segments(raw_segments, audio_path, on_progress))
        transcribe_sec = perf_counter() - start

        resolved_language = language if language is not None else info.language
        audio_sec = audio.duration_sec
        rtf = (transcribe_sec / audio_sec) if audio_sec else None

        engine_meta: dict[str, Any] = {
            "engine": "faster-whisper",
            "engine_version": engine_version,
            "model": self._model_name,
            "compute_type": self._compute_type,
            "beam_size": _BEAM_SIZE,
            "vad": _VAD_FILTER,
            "threads": self._threads,
            "audio_sec": audio_sec,
            "load_sec": load_sec,
            "transcribe_sec": transcribe_sec,
            "rtf": rtf,
        }

        return TranscriptResult(
            segments=tuple(segments),
            language=resolved_language,
            engine_meta=engine_meta,
        )

    def check_available(self) -> None:
        """Raise ``ToolFailureError`` if faster-whisper is not installed; loads nothing."""
        _import_faster_whisper()

    def _ensure_model(self) -> tuple[Any, str, float]:
        """Load the model on first use; reuse it after.

        Returns ``(model, engine_version, load_sec)``. ``load_sec`` is 0.0
        on every call after the first, since no loading happened.
        """
        if self._loaded_model is not None:
            return self._loaded_model, self._engine_version, 0.0

        faster_whisper = _import_faster_whisper()

        start = perf_counter()
        try:
            model = faster_whisper.WhisperModel(
                self._model_name,
                device="cpu",
                compute_type=self._compute_type,
                cpu_threads=self._threads,
            )
        except MemoryError:
            raise
        except OSError as exc:
            # Model download (huggingface_hub) failing offline surfaces as a
            # connection/socket error here; anything else at load time (a bad
            # model name, a corrupt cache entry) is left unclassified.
            raise TransientNetworkError(
                f"failed to load faster-whisper model {self._model_name!r}: {exc}"
            ) from exc
        load_sec = perf_counter() - start

        self._loaded_model = model
        self._engine_version = _engine_version(faster_whisper)
        return self._loaded_model, self._engine_version, load_sec


def _import_faster_whisper() -> Any:
    try:
        import faster_whisper  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ToolFailureError(
            f"faster-whisper is not installed; install {_REQUIREMENTS_FILE}"
        ) from exc
    return faster_whisper


def _collect_segments(
    raw_segments: Iterator[Any],
    audio_path: object,
    on_progress: Callable[[float], None] | None,
) -> Iterator[Segment]:
    """Turn faster-whisper's segment objects into ``Segment``s.

    ``on_progress`` is called after each decoded segment, outside of any
    error handling here, so an exception it raises (e.g. ``Cancelled``)
    stops iteration and propagates unchanged rather than being reclassified
    as a tool failure.
    """
    iterator = iter(raw_segments)
    while True:
        try:
            raw = next(iterator)
        except StopIteration:
            return
        except MemoryError:
            raise
        except Exception as exc:
            raise ToolFailureError(
                f"faster-whisper failed to transcribe {audio_path}: {exc}"
            ) from exc

        text = raw.text.strip()
        if text:
            yield Segment(start=raw.start, end=raw.end, text=text)
        if on_progress is not None:
            on_progress(raw.end)


def _engine_version(module: Any) -> str:
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        return metadata.version("faster-whisper")
    except metadata.PackageNotFoundError:
        return "unknown"
