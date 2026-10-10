"""Phase 0: transcription engine bake-off harness (issue #3).

Throwaway measurement script. It imports nothing from ``common/``,
``adapters/`` or ``services/`` and uses only the stdlib in the *repo* venv.
The three engines under test (``faster-whisper large-v3``, ``faster-whisper
large-v3-turbo`` and ``parakeet-tdt-0.6b-v3`` via sherpa-onnx) are installed
into a separate, throwaway venv *outside* the repo (default
``~/.cache/bakeoff-venv``) the first time this script runs. They are never
added to any requirements file.

Parakeet: architecture.md 16.6 names ``parakeet.cpp`` as the intended C++
port. This host has no C/C++ toolchain at all (no gcc, cmake or make, and
installing one is a system change out of scope for a throwaway venv), so per
the issue's own fallback this harness uses a CPU ONNX runtime instead:
sherpa-onnx (ships cp314 manylinux wheels, no compiler needed), running
NVIDIA's parakeet-tdt-0.6b-v3 via its NeMo-transducer ONNX export
(``csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`` on Hugging Face).

Usage:
    python scripts/phase0/bakeoff.py <video-url> --start 10:00 --duration 300
    python scripts/phase0/bakeoff.py <video-url> --reference ref.txt --nouns nouns.txt
    python scripts/phase0/bakeoff.py <video-url> --engines whisper-large-v3-turbo \\
        --whisper-model-size tiny   # quick smoke test, not a real measurement

Each engine runs in its own subprocess under the throwaway venv's
interpreter, wrapped in a second subprocess (this same file, run under the
*repo* venv, hidden ``--measure-child`` mode) whose only child is that
engine process. That isolates
``resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`` per engine: since the
wrapper spawns exactly one child, the value read after it exits is that
engine's peak RSS alone, never mixed with another engine's.

RTF measured on this WSL dev machine does not count toward the bake-off
result (constraints in the issue). Only a run on the deploy host counts, and
the acceptance criteria require at least 3 real clips from the channel list,
with the report pasted as a comment on issue #3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

THIS_FILE = Path(__file__).resolve()

# Identical to adapters/youtube/audio.py's FFMPEG_AUDIO_FLAGS (#19), copied
# rather than imported (scripts/phase0/ imports nothing from adapters/):
# mono, 16 kHz, opus tuned for speech ("voip" application mode favours
# intelligibility over music fidelity).
FFMPEG_AUDIO_ARGS = ["-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "16k",
                      "-application", "voip"]
TARGET_SAMPLE_RATE = 16000

# architecture.md 11.4: WHISPER_THREADS default.
DEFAULT_THREADS = 4
# architecture.md 16.6: default start offset is not 0:00 - intros/music are
# not representative speech.
DEFAULT_START = "3:00"
DEFAULT_DURATION = 300

YTDLP_TIMEOUT_SEC = 900
FFMPEG_TIMEOUT_SEC = 300
FFPROBE_TIMEOUT_SEC = 30
ENGINE_TIMEOUT_SEC = 3600
DOWNLOAD_TIMEOUT_SEC = 1800

VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")

# Pinned exactly, like every other throwaway/runtime dependency in this repo
# (AGENTS.md -> Rules). These never move into requirements.backend.txt or
# requirements.whisper.txt: faster-whisper gets pinned there by #20, and
# Parakeet only if it wins the bake-off.
# av is faster-whisper's own dependency (declared only as ``av>=11``): av 19
# dropped the ``metadata_errors`` argument faster-whisper 1.2.1 passes to
# ``av.open``, so every transcription crashed. 18.1.0 is what #3's host venv
# resolved (issue #76).
FASTER_WHISPER_PIP_SPECS = ["faster-whisper==1.2.1", "av==18.1.0", "numpy==2.5.3"]
SHERPA_ONNX_PIP_SPECS = ["sherpa-onnx==1.12.40", "numpy==2.5.3"]

PARAKEET_MODEL_REPO = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
PARAKEET_MODEL_FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)
PARAKEET_FEATURE_DIM = 128

# Container support (issue #76). The image sets these; on a host with none of
# them set every default below is exactly what it was before.
ENV_ENGINES_VENV = "BAKEOFF_ENGINES_VENV"
ENV_CACHE_DIR = "BAKEOFF_CACHE_DIR"
ENV_OUT_DIR = "BAKEOFF_OUT_DIR"  # parent of the per-clip run directories
# Set in the image, where the engine venv is built once and is read-only: a
# run then never pip-installs (or upgrades) anything.
ENV_PREINSTALLED = "BAKEOFF_ENGINES_PREINSTALLED"
INSTALL_ENGINES_FLAG = "--install-engines"


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)  # noqa: TID251 - standalone script, not app code
    return Path(value) if value else default


# --- engine catalogue --------------------------------------------------------------


@dataclass(frozen=True)
class EngineSpec:
    key: str
    family: Literal["faster-whisper", "parakeet-sherpa"]
    model: str  # faster-whisper model id, or the HF repo id for parakeet


ENGINES: dict[str, EngineSpec] = {
    "whisper-large-v3": EngineSpec("whisper-large-v3", "faster-whisper", "large-v3"),
    "whisper-large-v3-turbo": EngineSpec(
        "whisper-large-v3-turbo", "faster-whisper", "large-v3-turbo"
    ),
    "parakeet-tdt-0.6b-v3": EngineSpec(
        "parakeet-tdt-0.6b-v3", "parakeet-sherpa", PARAKEET_MODEL_REPO
    ),
}
ENGINE_ORDER = (
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "parakeet-tdt-0.6b-v3",
)


@dataclass
class EngineResult:
    key: str
    ok: bool
    reason: str | None = None
    model_version: str | None = None
    load_time_sec: float | None = None
    transcribe_time_sec: float | None = None
    rtf: float | None = None
    peak_rss_mb: float | None = None
    transcript_path: Path | None = None
    wer: float | None = None
    noun_hits: int | None = None
    noun_total: int | None = None


# --- small shared helpers ----------------------------------------------------------


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


NOISE_MARKERS = ("No supported JavaScript runtime",)


def first_line(stderr: str) -> str:
    """First non-noise line of a failed subprocess's stderr, preferring ERROR:."""
    lines = [
        line.strip()
        for line in stderr.splitlines()
        if line.strip() and not any(marker in line for marker in NOISE_MARKERS)
    ]
    for line in lines:
        if line.startswith("ERROR:"):
            return line
    return lines[0] if lines else "(no stderr)"


def traceback_reason(stderr: str) -> str:
    """The informative line of a worker crash.

    Worker failures are Python tracebacks (engine_worker_main prints one on
    any exception); the useful message is the *last* line, not the first.
    """
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return lines[-1] if lines else "(no stderr)"


def parse_timecode(value: str) -> int:
    """Accept plain seconds, "MM:SS" or "H:MM:SS"."""
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"not a timecode: {value!r}")
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds


def format_timecode(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_video_id(url: str) -> str:
    """Best-effort 11-char video ID, for cache filenames/labels only.

    The full URL, never a reconstructed one, is what gets passed to yt-dlp
    (testing-guidelines.md: a bare ID can start with "-" and be read as an
    option).
    """
    for pattern in (r"[?&]v=([A-Za-z0-9_-]{11})", r"youtu\.be/([A-Za-z0-9_-]{11})",
                     r"/shorts/([A-Za-z0-9_-]{11})", r"/embed/([A-Za-z0-9_-]{11})"):
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    # Fall back to a short stable digest of the whole URL so caching still works.
    return "url" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


# --- system info ---------------------------------------------------------------


@dataclass(frozen=True)
class SystemInfo:
    cpu_model: str
    core_count: int
    ram_gb: float


def read_system_info() -> SystemInfo:
    cpu_model = "unknown"
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    ram_gb = 0.0
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                ram_gb = kb / (1024 * 1024)
                break
    except OSError:
        pass

    return SystemInfo(cpu_model=cpu_model, core_count=os.cpu_count() or 0, ram_gb=ram_gb)


CGROUP_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
CGROUP_MEMORY_MAX = Path("/sys/fs/cgroup/memory.max")


def in_container() -> bool:
    return Path("/.dockerenv").exists() or bool(
        os.environ.get("container")  # noqa: TID251 - set by podman and others
    )


def _read_cgroup(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip() or "unreadable"
    except OSError:
        return "unreadable"


def container_line() -> str:
    """The report header's container line.

    Inside a container /proc/meminfo and os.cpu_count() show the host or VM,
    not the limit, so the cgroup v2 limits are printed next to them ("max"
    means unlimited).
    """
    if not in_container():
        return "container: no"
    return (
        f"container: yes (cgroup v2 cpu.max: {_read_cgroup(CGROUP_CPU_MAX)}, "
        f"memory.max: {_read_cgroup(CGROUP_MEMORY_MAX)})"
    )


def tool_version(cmd: list[str]) -> str:
    try:
        proc = run(cmd, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"unavailable ({exc})"
    if proc.returncode != 0:
        return f"unavailable (exit {proc.returncode})"
    first = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    return first or "unknown"


# --- audio: download a section, normalize, cache ------------------------------------


def ensure_audio(video_url: str, start_sec: int, duration_sec: int,
                  cache_dir: Path) -> Path:
    """Download only [start, start+duration), normalize to 16k mono opus, cache it.

    Caching is keyed on (video id, start, duration) so repeated bake-off runs
    and every engine in a run share exactly the same bytes.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    video_id = extract_video_id(video_url)
    key = f"{video_id}_{start_sec:06d}_{duration_sec:05d}"
    normalized = cache_dir / f"{key}.opus"
    if normalized.exists() and normalized.stat().st_size > 0:
        log(f"audio cache hit: {normalized}")
        return normalized

    end_sec = start_sec + duration_sec
    section = f"*{format_timecode(start_sec)}-{format_timecode(end_sec)}"
    raw_template = cache_dir / f"{key}.raw.%(ext)s"
    # Full watch URL only (never a bare ID) is passed straight through.
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        "bestaudio/best",
        "--download-sections",
        section,
        "--no-playlist",
        "-o",
        str(raw_template),
        video_url,
    ]
    log(f"downloading section {format_timecode(start_sec)}-{format_timecode(end_sec)}")
    try:
        proc = run(cmd, timeout=YTDLP_TIMEOUT_SEC)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"yt-dlp timed out after {exc.timeout}s") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {first_line(proc.stderr)}")

    raw_matches = sorted(cache_dir.glob(f"{key}.raw.*"))
    if not raw_matches:
        raise RuntimeError("yt-dlp reported success but produced no output file")
    raw_path = raw_matches[0]

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_path),
        *FFMPEG_AUDIO_ARGS,
        str(normalized),
    ]
    try:
        proc = run(ffmpeg_cmd, timeout=FFMPEG_TIMEOUT_SEC)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timed out after {exc.timeout}s") from exc
    finally:
        raw_path.unlink(missing_ok=True)
    if proc.returncode != 0 or not normalized.exists():
        raise RuntimeError(f"ffmpeg normalization failed: {first_line(proc.stderr)}")
    return normalized


def audio_duration_sec(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = run(cmd, timeout=FFPROBE_TIMEOUT_SEC)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {first_line(proc.stderr)}")
    return float(proc.stdout.strip())


# --- throwaway engine venv -----------------------------------------------------


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def ensure_venv(venv_dir: Path) -> None:
    if venv_python(venv_dir).exists():
        return
    log(f"creating throwaway engine venv at {venv_dir}")
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    proc = run([sys.executable, "-m", "venv", str(venv_dir)], timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"venv creation failed: {first_line(proc.stderr)}")


def pip_install(venv_dir: Path, specs: list[str]) -> None:
    python = venv_python(venv_dir)
    proc = run(
        [str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        timeout=DOWNLOAD_TIMEOUT_SEC,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pip upgrade failed: {first_line(proc.stderr)}")
    proc = run(
        [str(python), "-m", "pip", "install", "--quiet", *specs],
        timeout=DOWNLOAD_TIMEOUT_SEC,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pip install failed: {first_line(proc.stderr)}")


# One family install failing (e.g. a wheel disappears for this Python minor)
# must not stop the other family's engines from running.
_family_install_cache: dict[str, str | None] = {}


def install_engines(venv_dir: Path) -> int:
    """Install-only mode (image build): both engine families, wheels only.

    The one place the pins above are used outside a normal run. pip's own
    output is inherited so a failure is visible, and a failure is returned as
    a non-zero exit code, never turned into a report line. `--only-binary`
    means a package without a wheel for this interpreter fails the install
    instead of needing a compiler.
    """
    specs = list(dict.fromkeys([*FASTER_WHISPER_PIP_SPECS, *SHERPA_ONNX_PIP_SPECS]))
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    for cmd in (
        [sys.executable, "-m", "venv", str(venv_dir)],
        [
            str(venv_python(venv_dir)), "-m", "pip", "install",
            "--only-binary", ":all:", *specs,
        ],
    ):
        log("+ " + " ".join(cmd))
        returncode = subprocess.run(cmd, check=False).returncode
        if returncode != 0:
            log(f"FATAL: {' '.join(cmd[:4])} exited {returncode}")
            return returncode
    return 0


def ensure_family_installed(venv_dir: Path, family: str) -> str | None:
    """Install the pip specs for `family` in the shared throwaway venv.

    Returns None on success, or a failure reason string (cached so a failing
    family is only retried once per process).
    """
    if family in _family_install_cache:
        return _family_install_cache[family]
    if os.environ.get(ENV_PREINSTALLED):  # noqa: TID251 - standalone script
        # Container: the engine venv was built into the image and is read-only.
        # Never create or pip-install anything at run time.
        if not venv_python(venv_dir).exists():
            return f"engine venv missing at {venv_dir}"
        return None
    specs = (
        FASTER_WHISPER_PIP_SPECS if family == "faster-whisper" else SHERPA_ONNX_PIP_SPECS
    )
    try:
        ensure_venv(venv_dir)
        pip_install(venv_dir, specs)
        reason = None
    except RuntimeError as exc:
        reason = str(exc)
    _family_install_cache[family] = reason
    return reason


# --- parakeet model download (plain urllib, no huggingface_hub dependency) ----------


def download_parakeet_model(model_dir: Path) -> Path:
    target = model_dir / PARAKEET_MODEL_REPO.split("/")[-1]
    target.mkdir(parents=True, exist_ok=True)
    for filename in PARAKEET_MODEL_FILES:
        dest = target / filename
        url = f"https://huggingface.co/{PARAKEET_MODEL_REPO}/resolve/main/{filename}"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        log(f"downloading parakeet model file {filename}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with (
                urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SEC) as resp,
                tmp.open("wb") as fh,
            ):
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
        except (urllib.error.URLError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"downloading {filename} failed: {exc}") from exc
        tmp.rename(dest)
    return target


# --- engine worker (runs INSIDE the throwaway venv's interpreter) ------------------


def decode_pcm_f32(audio_path: Path, sample_rate: int) -> Any:
    """Decode any ffmpeg-readable audio file to mono float32 PCM via ffmpeg.

    Used only by the parakeet/sherpa-onnx worker, which (unlike
    faster-whisper) needs raw samples rather than a file path. Returns a
    numpy array; typed Any here since numpy is not importable from the repo
    venv that mypy runs against.
    """
    import numpy as np  # type: ignore[import-not-found]

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_SEC, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg PCM decode failed: {first_line(proc.stderr.decode())}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def worker_run_faster_whisper(model: str, audio_path: Path, threads: int) -> dict[str, Any]:
    import faster_whisper  # type: ignore[import-not-found]

    WhisperModel = faster_whisper.WhisperModel
    fw_version = faster_whisper.version.__version__

    try:
        import ctranslate2  # type: ignore[import-not-found]

        ct2_version = ctranslate2.__version__
    except ImportError:
        ct2_version = "unknown"

    t0 = time.monotonic()
    model_obj = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)
    t1 = time.monotonic()
    segments, _info = model_obj.transcribe(str(audio_path), beam_size=5)
    text = "".join(segment.text for segment in segments)
    t2 = time.monotonic()

    return {
        "model_version": (
            f"faster-whisper {fw_version} (ctranslate2 {ct2_version}), "
            f"model={model}, compute=int8"
        ),
        "load_time_sec": t1 - t0,
        "transcribe_time_sec": t2 - t1,
        "transcript": text.strip(),
    }


def worker_run_parakeet(model_dir: Path, audio_path: Path, threads: int) -> dict[str, Any]:
    import sherpa_onnx  # type: ignore[import-not-found]

    model_files = download_parakeet_model(model_dir)

    t0 = time.monotonic()
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_files / "encoder.int8.onnx"),
        decoder=str(model_files / "decoder.int8.onnx"),
        joiner=str(model_files / "joiner.int8.onnx"),
        tokens=str(model_files / "tokens.txt"),
        model_type="nemo_transducer",
        feature_dim=PARAKEET_FEATURE_DIM,
        decoding_method="greedy_search",
        num_threads=threads,
        sample_rate=TARGET_SAMPLE_RATE,
        provider="cpu",
    )
    t1 = time.monotonic()
    samples = decode_pcm_f32(audio_path, TARGET_SAMPLE_RATE)
    stream = recognizer.create_stream()
    stream.accept_waveform(TARGET_SAMPLE_RATE, samples)
    recognizer.decode_stream(stream)
    text = stream.result.text
    t2 = time.monotonic()

    return {
        "model_version": (
            f"sherpa-onnx {sherpa_onnx.__version__}, model={PARAKEET_MODEL_REPO} "
            "(int8, NeMo transducer export)"
        ),
        "load_time_sec": t1 - t0,
        "transcribe_time_sec": t2 - t1,
        "transcript": text.strip(),
    }


def engine_worker_main(args: argparse.Namespace) -> int:
    spec = ENGINES[args.engine]
    try:
        if spec.family == "faster-whisper":
            model = args.whisper_model_size or spec.model
            result = worker_run_faster_whisper(model, Path(args.audio), args.threads)
        else:
            result = worker_run_parakeet(Path(args.model_dir), Path(args.audio), args.threads)
    except Exception:  # noqa: BLE001 - any engine failure must surface, not crash silently
        traceback.print_exc(file=sys.stderr)
        return 1
    result["threads"] = args.threads
    Path(args.result_out).write_text(json.dumps(result), encoding="utf-8")
    return 0


# --- RSS-isolating wrapper (runs under the REPO venv; stdlib only) -----------------


def measure_child_main(payload_json: str) -> int:
    payload = json.loads(payload_json)
    cmd: list[str] = payload["cmd"]
    timeout: int = payload["timeout"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        returncode = proc.returncode
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        returncode = -1
        stderr = f"timed out after {timeout}s"
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(json.dumps({
        "returncode": returncode,
        "stderr": stderr,
        "ru_maxrss_kb": usage.ru_maxrss,
    }))
    return 0


# --- orchestration -------------------------------------------------------------


def run_engine(
    spec: EngineSpec,
    venv_dir: Path,
    audio_path: Path,
    threads: int,
    audio_seconds: float,
    workdir: Path,
    model_dir: Path,
    whisper_model_size: str | None,
    timeout: int,
) -> EngineResult:
    install_failure = ensure_family_installed(venv_dir, spec.family)
    if install_failure is not None:
        return EngineResult(spec.key, ok=False, reason=f"install failed: {install_failure}")

    result_path = workdir / f"{spec.key}.result.json"
    result_path.unlink(missing_ok=True)
    worker_cmd = [
        str(venv_python(venv_dir)),
        str(THIS_FILE),
        "--engine-worker",
        spec.key,
        "--audio",
        str(audio_path),
        "--threads",
        str(threads),
        "--result-out",
        str(result_path),
        "--model-dir",
        str(model_dir),
    ]
    if whisper_model_size:
        worker_cmd += ["--whisper-model-size", whisper_model_size]

    wrapper_cmd = [
        sys.executable,
        str(THIS_FILE),
        "--measure-child",
        json.dumps({"cmd": worker_cmd, "timeout": timeout}),
    ]
    try:
        wrapper_proc = subprocess.run(
            wrapper_cmd, capture_output=True, text=True, timeout=timeout + 60, check=False
        )
    except subprocess.TimeoutExpired:
        return EngineResult(spec.key, ok=False, reason=f"wrapper timed out after {timeout + 60}s")

    try:
        wrapper_out = json.loads(wrapper_proc.stdout)
    except json.JSONDecodeError:
        detail = wrapper_proc.stderr.strip().splitlines()[-1] if wrapper_proc.stderr.strip() else ""
        return EngineResult(spec.key, ok=False, reason=f"measurement wrapper crashed: {detail}")

    if wrapper_out["returncode"] != 0 or not result_path.exists():
        reason = (
            traceback_reason(wrapper_out.get("stderr", ""))
            or f"exit {wrapper_out['returncode']}"
        )
        return EngineResult(spec.key, ok=False, reason=reason)

    data = json.loads(result_path.read_text(encoding="utf-8"))
    transcribe_time = float(data["transcribe_time_sec"])
    transcript_path = workdir / f"{spec.key}.transcript.txt"
    transcript_path.write_text(data["transcript"] + "\n", encoding="utf-8")

    return EngineResult(
        spec.key,
        ok=True,
        model_version=data["model_version"],
        load_time_sec=float(data["load_time_sec"]),
        transcribe_time_sec=transcribe_time,
        rtf=transcribe_time / audio_seconds if audio_seconds > 0 else float("nan"),
        peak_rss_mb=wrapper_out["ru_maxrss_kb"] / 1024,
        transcript_path=transcript_path,
    )


# --- scoring: WER and proper-noun accuracy ------------------------------------------


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_words(text: str) -> list[str]:
    stripped = _PUNCT_RE.sub("", text.lower())
    return stripped.split()


def edit_distance(a: list[str], b: list[str]) -> int:
    """Stdlib Levenshtein distance (word-level), per the issue's AC."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, wb in enumerate(b, start=1):
            cost = 0 if wa == wb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)
    if not ref_words:
        return float("nan")
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def noun_accuracy(transcript: str, nouns: list[str]) -> tuple[int, int]:
    hits = 0
    for noun in nouns:
        pattern = re.compile(r"\b" + re.escape(noun) + r"\b")
        if pattern.search(transcript):
            hits += 1
    return hits, len(nouns)


def read_nouns(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- report ----------------------------------------------------------------------


def format_report(
    video_url: str,
    start_sec: int,
    duration_sec: int,
    audio_seconds: float,
    threads: int,
    system: SystemInfo,
    yt_dlp_version: str,
    ffmpeg_version: str,
    results: list[EngineResult],
    reference_used: bool,
    nouns_used: int,
    container: str = "container: no",
) -> str:
    lines = [
        "# Transcription engine bake-off report",
        "",
        f"video: {video_url}",
        (
            f"section: {format_timecode(start_sec)} + {duration_sec}s "
            f"(audio duration measured: {audio_seconds:.1f}s)"
        ),
        f"threads: {threads}",
        f"CPU: {system.cpu_model} ({system.core_count} cores)",
        f"RAM: {system.ram_gb:.1f} GB",
        container,
        f"yt-dlp: {yt_dlp_version}",
        f"ffmpeg: {ffmpeg_version}",
        "",
    ]
    for r in results:
        lines.append(f"## {r.key}")
        if not r.ok:
            lines.append(f"FAILED: {r.reason}")
            lines.append("")
            continue
        lines.append(f"model: {r.model_version}")
        lines.append(f"load time: {r.load_time_sec:.2f}s")
        lines.append(f"transcribe time: {r.transcribe_time_sec:.2f}s")
        lines.append(f"RTF: {r.rtf:.3f}")
        lines.append(f"peak RSS: {r.peak_rss_mb:.1f} MB")
        if reference_used:
            wer_str = f"{r.wer:.3f}" if r.wer is not None else "n/a"
            lines.append(f"WER: {wer_str}")
        if nouns_used:
            lines.append(f"proper nouns spelled exactly: {r.noun_hits}/{r.noun_total}")
        lines.append(f"transcript: {r.transcript_path}")
        lines.append("")
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcription engine bake-off: faster-whisper large-v3, "
            "faster-whisper large-v3-turbo and parakeet-tdt-0.6b-v3 (via "
            "sherpa-onnx), measured on one downloaded-and-normalized clip."
        )
    )
    parser.add_argument("video_url", help="full YouTube watch URL")
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"section start, MM:SS or H:MM:SS (default: {DEFAULT_START}, not 0:00 - "
        "intros/music are not representative speech)",
    )
    parser.add_argument(
        "--duration", type=int, default=DEFAULT_DURATION,
        help=f"section length in seconds (default: {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_THREADS,
        help=f"thread count given to every engine (default: {DEFAULT_THREADS}, "
        "matching WHISPER_THREADS)",
    )
    parser.add_argument("--reference", type=Path, help="hand-corrected reference transcript")
    parser.add_argument("--nouns", type=Path, help="proper-noun list, one per line")
    parser.add_argument(
        "--engines", default=",".join(ENGINE_ORDER),
        help=f"comma-separated subset of {list(ENGINE_ORDER)}",
    )
    parser.add_argument(
        "--whisper-model-size",
        help="override the faster-whisper model id for BOTH whisper engines "
        "(e.g. 'tiny' for a quick smoke test). Not a real bake-off number.",
    )
    parser.add_argument(
        "--engines-venv", type=Path,
        default=env_path(ENV_ENGINES_VENV, Path.home() / ".cache" / "bakeoff-venv"),
        help="throwaway venv for the engines, outside the repo",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=env_path(ENV_CACHE_DIR, Path.home() / ".cache" / "bakeoff-cache"),
        help="audio and parakeet model cache",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        help="where transcripts/report go (default: <cache-dir>/runs/<clip key>; "
        f"${ENV_OUT_DIR} replaces <cache-dir>/runs)",
    )
    parser.add_argument(
        "--engine-timeout", type=int, default=ENGINE_TIMEOUT_SEC,
        help="per-engine subprocess timeout in seconds",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    # Hidden internal modes, dispatched before the public parser so they
    # don't show up in --help and don't have to share its argument shape.
    if argv and argv[0] == INSTALL_ENGINES_FLAG:
        if len(argv) != 2:
            print(f"usage: bakeoff.py {INSTALL_ENGINES_FLAG} <venv-dir>", file=sys.stderr)
            return 2
        return install_engines(Path(argv[1]))
    if argv and argv[0] == "--measure-child":
        return measure_child_main(argv[1])
    if argv and argv[0] == "--engine-worker":
        worker_parser = argparse.ArgumentParser()
        worker_parser.add_argument("--engine-worker", dest="engine", required=True)
        worker_parser.add_argument("--audio", required=True)
        worker_parser.add_argument("--threads", type=int, required=True)
        worker_parser.add_argument("--result-out", required=True)
        worker_parser.add_argument("--model-dir", required=True)
        worker_parser.add_argument("--whisper-model-size", default=None)
        worker_args = worker_parser.parse_args(argv)
        return engine_worker_main(worker_args)

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        start_sec = parse_timecode(args.start)
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.threads <= 0:
        parser.error("--threads must be positive")

    engine_keys = [k.strip() for k in args.engines.split(",") if k.strip()]
    unknown = [k for k in engine_keys if k not in ENGINES]
    if unknown:
        parser.error(f"unknown engine(s): {unknown}; choices are {list(ENGINE_ORDER)}")

    video_id = extract_video_id(args.video_url)
    clip_key = f"{video_id}_{start_sec:06d}_{args.duration:05d}"
    out_dir = args.out_dir or (
        env_path(ENV_OUT_DIR, args.cache_dir / "runs") / clip_key
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.cache_dir / "models"

    if args.whisper_model_size:
        log(
            f"NOTE: --whisper-model-size={args.whisper_model_size!r} overrides both "
            "whisper engines. This is a smoke test, not a bake-off measurement."
        )

    try:
        audio_path = ensure_audio(args.video_url, start_sec, args.duration, args.cache_dir)
        audio_seconds = audio_duration_sec(audio_path)
    except RuntimeError as exc:
        log(f"FATAL: could not prepare audio: {exc}")
        return 1

    reference = args.reference.read_text(encoding="utf-8") if args.reference else None
    nouns = read_nouns(args.nouns) if args.nouns else []

    results: list[EngineResult] = []
    for key in engine_keys:
        spec = ENGINES[key]
        log(f"running {key}")
        result = run_engine(
            spec,
            args.engines_venv,
            audio_path,
            args.threads,
            audio_seconds,
            out_dir,
            model_dir,
            args.whisper_model_size,
            args.engine_timeout,
        )
        if result.ok and result.transcript_path is not None:
            transcript = result.transcript_path.read_text(encoding="utf-8")
            if reference is not None:
                result.wer = word_error_rate(reference, transcript)
            if nouns:
                result.noun_hits, result.noun_total = noun_accuracy(transcript, nouns)
        results.append(result)
        status = "OK" if result.ok else f"FAILED: {result.reason}"
        log(f"  {key}: {status}")

    system = read_system_info()
    yt_dlp_version = tool_version([sys.executable, "-m", "yt_dlp", "--version"])
    ffmpeg_version = tool_version(["ffmpeg", "-version"])

    report = format_report(
        args.video_url,
        start_sec,
        args.duration,
        audio_seconds,
        args.threads,
        system,
        yt_dlp_version,
        ffmpeg_version,
        results,
        reference_used=reference is not None,
        nouns_used=len(nouns),
        container=container_line(),
    )
    report_path = out_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    log(f"report saved to {report_path}")

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
