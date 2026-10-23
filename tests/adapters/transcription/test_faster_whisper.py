"""Unit tests for the faster-whisper Transcriber adapter (issue #20).

No test here needs faster-whisper installed (AGENTS.md -> Setup): every test
injects a fake ``WhisperModel`` through ``sys.modules['faster_whisper']``,
which is exactly how the module imports it internally (a lazy, local
``import faster_whisper``). The one real smoke test lives in
``test_smoke.py``, marked ``@pytest.mark.whisper``.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest

from adapters.transcription.faster_whisper import FasterWhisperTranscriber
from common.config import get_settings
from common.errors import Cancelled, ToolFailureError, TransientNetworkError
from common.models import AudioRef, Transcriber, TranscriptResult

DB_URL = "postgresql://u:p@h/db"
MODULE_NAME = "adapters.transcription.faster_whisper"


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    # DATABASE_URL is required by Settings; every test that needs settings
    # defaults (AUDIO_DIR, WHISPER_*) gets a harmless one for free.
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("AUDIO_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --- fakes -------------------------------------------------------------


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str


@dataclass
class FakeInfo:
    language: str = "en"


def make_fake_model(
    *,
    segments: Iterable[FakeSegment] = (),
    language: str = "en",
    raise_on_init: BaseException | None = None,
    raise_on_transcribe: BaseException | None = None,
    segment_iterator: Callable[[], Iterator[FakeSegment]] | None = None,
) -> type:
    """A fresh fake ``WhisperModel`` class, one per test - no shared state.

    ``segment_iterator``, when given, replaces ``segments`` and lets a test
    raise partway through iteration (e.g. a decode error mid-stream) rather
    than from the initial ``transcribe()`` call.

    Returns a plain ``type`` (rather than a precise one) deliberately: it
    stands for a class the test file has no compile-time knowledge of,
    exactly like the real ``faster_whisper.WhisperModel`` it replaces.
    Callers assign it to an ``Any``-annotated variable before reaching for
    ``.instances``.
    """
    detected_language = language
    frozen_segments = tuple(segments)

    class _FakeWhisperModel:
        instances: ClassVar[list[_FakeWhisperModel]] = []

        def __init__(
            self,
            model_size_or_path: str,
            *,
            device: str,
            compute_type: str,
            cpu_threads: int,
        ) -> None:
            if raise_on_init is not None:
                raise raise_on_init
            self.model_size_or_path = model_size_or_path
            self.device = device
            self.compute_type = compute_type
            self.cpu_threads = cpu_threads
            self.transcribe_calls: list[dict[str, Any]] = []
            _FakeWhisperModel.instances.append(self)

        def transcribe(
            self,
            audio_path: str,
            *,
            language: str | None,
            beam_size: int,
            vad_filter: bool,
        ) -> tuple[Iterator[FakeSegment], FakeInfo]:
            self.transcribe_calls.append(
                {
                    "audio_path": audio_path,
                    "language": language,
                    "beam_size": beam_size,
                    "vad_filter": vad_filter,
                }
            )
            if raise_on_transcribe is not None:
                raise raise_on_transcribe
            iterator = (
                segment_iterator()
                if segment_iterator is not None
                else iter(frozen_segments)
            )
            return iterator, FakeInfo(language=language or detected_language)

    return _FakeWhisperModel


def install_fake_module(
    monkeypatch: pytest.MonkeyPatch, model_cls: type, *, version: str = "9.9.9-fake"
) -> None:
    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = model_cls  # type: ignore[attr-defined]
    fake_module.__version__ = version  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)


def make_clock(values: Iterable[float]) -> Callable[[], float]:
    iterator = iter(values)
    return lambda: next(iterator)


AUDIO = AudioRef(rel_path="ab/abc12345678.opus", bytes=1234, duration_sec=10.0)


# --- port and construction ----------------------------------------------


def test_faster_whisper_transcriber_satisfies_the_transcriber_port() -> None:
    transcriber: Transcriber = FasterWhisperTranscriber(
        model="tiny", compute_type="int8", threads=0
    )
    assert callable(transcriber.transcribe)


def test_transcript_result_is_immutable() -> None:
    result = TranscriptResult(segments=(), language="en", engine_meta={})
    with pytest.raises(AttributeError):
        result.language = "fr"  # type: ignore[misc]


def test_defaults_come_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHISPER_MODEL", "medium")
    monkeypatch.setenv("WHISPER_COMPUTE", "int16")
    monkeypatch.setenv("WHISPER_THREADS", "4")
    get_settings.cache_clear()

    fake_cls: Any = make_fake_model()
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber()

    result = transcriber.transcribe(AUDIO)

    [instance] = fake_cls.instances
    assert instance.model_size_or_path == "medium"
    assert instance.compute_type == "int16"
    assert instance.cpu_threads == 4
    assert result.engine_meta["model"] == "medium"
    assert result.engine_meta["compute_type"] == "int16"
    assert result.engine_meta["threads"] == 4


def test_explicit_constructor_args_override_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WHISPER_MODEL", "large-v3")
    get_settings.cache_clear()

    fake_cls: Any = make_fake_model()
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=2)

    transcriber.transcribe(AUDIO)

    [instance] = fake_cls.instances
    assert instance.model_size_or_path == "tiny"


def test_threads_zero_and_nonzero_pass_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls: Any = make_fake_model()
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    transcriber.transcribe(AUDIO)

    assert fake_cls.instances[0].cpu_threads == 0


def test_a_nonzero_thread_count_passes_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls: Any = make_fake_model()
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=6)

    transcriber.transcribe(AUDIO)

    assert fake_cls.instances[0].cpu_threads == 6


# --- lazy loading --------------------------------------------------------


def test_importing_the_module_and_constructing_do_not_need_faster_whisper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reload in place (same module object, same __dict__) rather than
    # delitem + fresh import_module: the latter would leave later tests'
    # already-bound `FasterWhisperTranscriber` name pointing at a stale
    # module object, silently unaffected by anything patched on the new one.
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    module = importlib.reload(sys.modules[MODULE_NAME])

    transcriber = module.FasterWhisperTranscriber(
        model="tiny", compute_type="int8", threads=0
    )

    assert transcriber is not None


def test_transcribe_raises_tool_failure_naming_the_requirements_file_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    with pytest.raises(ToolFailureError, match="requirements.whisper.txt"):
        transcriber.transcribe(AUDIO)


def test_constructing_does_not_load_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cls: Any = make_fake_model()
    install_fake_module(monkeypatch, fake_cls)

    FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    assert fake_cls.instances == []


def test_two_calls_on_the_same_instance_load_the_model_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls: Any = make_fake_model()
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    transcriber.transcribe(AUDIO)
    transcriber.transcribe(AUDIO)

    assert len(fake_cls.instances) == 1
    assert len(fake_cls.instances[0].transcribe_calls) == 2


def test_load_sec_is_zero_on_the_second_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cls: Any = make_fake_model()
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    first = transcriber.transcribe(AUDIO)
    second = transcriber.transcribe(AUDIO)

    assert first.engine_meta["load_sec"] >= 0
    assert second.engine_meta["load_sec"] == 0.0


# --- decoding options (D3) -----------------------------------------------


def test_decoding_uses_beam_size_5_and_vad_filter_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls: Any = make_fake_model()
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    transcriber.transcribe(AUDIO)

    [call] = fake_cls.instances[0].transcribe_calls
    assert call["beam_size"] == 5
    assert call["vad_filter"] is True


# --- results --------------------------------------------------------------


def test_segments_are_trimmed_and_empty_ones_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls: Any = make_fake_model(
        segments=[
            FakeSegment(start=0.0, end=1.0, text="  hello world  "),
            FakeSegment(start=1.0, end=1.2, text="   "),
            FakeSegment(start=1.2, end=2.0, text=""),
            FakeSegment(start=2.0, end=3.0, text="bye"),
        ]
    )
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    result = transcriber.transcribe(AUDIO)

    assert [s.text for s in result.segments] == ["hello world", "bye"]
    assert [(s.start, s.end) for s in result.segments] == [(0.0, 1.0), (2.0, 3.0)]


def test_language_defaults_to_the_detected_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls: Any = make_fake_model(language="es")
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    result = transcriber.transcribe(AUDIO)

    assert result.language == "es"
    assert fake_cls.instances[0].transcribe_calls[0]["language"] is None


@pytest.mark.parametrize(
    ("given", "passed"),
    [
        ("en-US", "en"),  # the YouTube spelling that broke job 7
        ("en_GB", "en"),
        ("pt-BR", "pt"),
        ("zh-Hans", "zh"),
        ("FR", "fr"),
        ("  de ", "de"),
        ("iw", "he"),  # old ISO 639-1 codes YouTube still emits
        ("jv", "jw"),  # Whisper spells Javanese "jw"
    ],
)
def test_region_and_legacy_language_codes_are_reduced_to_a_whisper_code(
    monkeypatch: pytest.MonkeyPatch, given: str, passed: str
) -> None:
    fake_cls: Any = make_fake_model(language="en")
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=1)

    result = transcriber.transcribe(AUDIO, language=given)

    assert fake_cls.instances[0].transcribe_calls[0]["language"] == passed
    assert result.language == passed


@pytest.mark.parametrize("given", ["", "   ", "xx", "und", "zxx", "-US", "klingon"])
def test_a_language_whisper_does_not_know_falls_back_to_detection(
    monkeypatch: pytest.MonkeyPatch, given: str
) -> None:
    fake_cls: Any = make_fake_model(language="es")
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=1)

    result = transcriber.transcribe(AUDIO, language=given)

    assert fake_cls.instances[0].transcribe_calls[0]["language"] is None
    assert result.language == "es"


def test_language_given_explicitly_is_used_and_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls: Any = make_fake_model(language="es")
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    result = transcriber.transcribe(AUDIO, language="fr")

    assert result.language == "fr"
    assert fake_cls.instances[0].transcribe_calls[0]["language"] == "fr"


def test_engine_meta_has_the_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cls: Any = make_fake_model()
    install_fake_module(monkeypatch, fake_cls, version="1.2.1")
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=3)

    result = transcriber.transcribe(AUDIO)
    meta = result.engine_meta

    assert meta["engine"] == "faster-whisper"
    assert meta["engine_version"] == "1.2.1"
    assert meta["model"] == "tiny"
    assert meta["compute_type"] == "int8"
    assert meta["beam_size"] == 5
    assert meta["vad"] is True
    assert meta["threads"] == 3
    assert meta["audio_sec"] == 10.0
    assert isinstance(meta["load_sec"], float)
    assert isinstance(meta["transcribe_sec"], float)
    assert meta["rtf"] == pytest.approx(meta["transcribe_sec"] / meta["audio_sec"])


def test_rtf_excludes_model_load_time(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cls: Any = make_fake_model()
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)
    clock = make_clock([100.0, 105.0, 200.0, 202.0])  # load: 5s: not part of rtf
    monkeypatch.setattr(f"{MODULE_NAME}.perf_counter", clock)

    result = transcriber.transcribe(AUDIO)  # audio_sec == 10.0

    assert result.engine_meta["load_sec"] == 5.0
    assert result.engine_meta["transcribe_sec"] == 2.0
    assert result.engine_meta["rtf"] == pytest.approx(0.2)


def test_silence_reduced_to_nothing_by_vad_returns_zero_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls: Any = make_fake_model(segments=[])
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    result = transcriber.transcribe(AUDIO)

    assert result.segments == ()


def test_zero_length_audio_gets_rtf_none_instead_of_a_division_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls: Any = make_fake_model()
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)
    zero_length = AudioRef(rel_path="ab/zero.opus", bytes=0, duration_sec=0.0)

    result = transcriber.transcribe(zero_length)

    assert result.engine_meta["audio_sec"] == 0.0
    assert result.engine_meta["rtf"] is None


def test_audio_path_is_resolved_under_audio_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_cls: Any = make_fake_model()
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    transcriber.transcribe(AUDIO)

    call = fake_cls.instances[0].transcribe_calls[0]
    assert call["audio_path"] == str(tmp_path / AUDIO.rel_path)


# --- progress and cancellation -------------------------------------------


def test_on_progress_is_called_once_per_decoded_segment_with_cumulative_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls: Any = make_fake_model(
        segments=[
            FakeSegment(start=0.0, end=1.0, text="a"),
            FakeSegment(start=1.0, end=2.5, text="b"),
            FakeSegment(start=2.5, end=4.0, text="c"),
        ]
    )
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)
    calls: list[float] = []

    transcriber.transcribe(AUDIO, on_progress=calls.append)

    assert calls == [1.0, 2.5, 4.0]


def test_an_exception_from_on_progress_stops_transcription_and_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls: Any = make_fake_model(
        segments=[
            FakeSegment(start=0.0, end=1.0, text="a"),
            FakeSegment(start=1.0, end=2.5, text="b"),
        ]
    )
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)
    calls: list[float] = []

    def on_progress(done_sec: float) -> None:
        calls.append(done_sec)
        raise Cancelled()

    with pytest.raises(Cancelled):
        transcriber.transcribe(AUDIO, on_progress=on_progress)

    assert calls == [1.0]  # stopped after the first segment, never reached the second


# --- errors ---------------------------------------------------------------


def test_a_network_failure_loading_the_model_raises_transient_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls: Any = make_fake_model(
        raise_on_init=ConnectionError("network unreachable")
    )
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    with pytest.raises(TransientNetworkError):
        transcriber.transcribe(AUDIO)


def test_corrupt_audio_raises_tool_failure_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls: Any = make_fake_model(
        raise_on_transcribe=RuntimeError("moov atom not found")
    )
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    with pytest.raises(ToolFailureError):
        transcriber.transcribe(AUDIO)


def test_a_decode_error_mid_iteration_raises_tool_failure_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_iterator() -> Iterator[FakeSegment]:
        yield FakeSegment(start=0.0, end=1.0, text="ok")
        raise RuntimeError("corrupt frame")

    fake_cls: Any = make_fake_model(segment_iterator=bad_iterator)
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    with pytest.raises(ToolFailureError):
        transcriber.transcribe(AUDIO)


def test_memory_error_from_model_construction_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls: Any = make_fake_model(raise_on_init=MemoryError())
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    with pytest.raises(MemoryError):
        transcriber.transcribe(AUDIO)


def test_memory_error_from_transcribe_call_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls: Any = make_fake_model(raise_on_transcribe=MemoryError())
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    with pytest.raises(MemoryError):
        transcriber.transcribe(AUDIO)


def test_memory_error_mid_iteration_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def bad_iterator() -> Iterator[FakeSegment]:
        yield FakeSegment(start=0.0, end=1.0, text="ok")
        raise MemoryError()

    fake_cls: Any = make_fake_model(segment_iterator=bad_iterator)
    install_fake_module(monkeypatch, fake_cls)
    transcriber = FasterWhisperTranscriber(model="tiny", compute_type="int8", threads=0)

    with pytest.raises(MemoryError):
        transcriber.transcribe(AUDIO)
