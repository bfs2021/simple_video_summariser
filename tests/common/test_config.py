"""Tests for ``common.config`` (issue #5, architecture.md 10).

Every test builds ``Settings(...)`` directly or sets variables with
``monkeypatch``; the autouse fixture strips every settings variable first so
the developer's shell environment never leaks in.
"""

import json
import subprocess
import sys
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from common.config import Settings, get_settings

DB_URL = "postgresql://u:p@h/db"
DB_PASSWORD = "s3cr3t-db-pw"
DB_URL_WITH_SECRET = f"postgresql://ytdigest:{DB_PASSWORD}@db:5432/ytdigest"
API_KEY = "sk-ant-test-0123456789abcdef"

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    # Run from an empty directory so no real .env could ever be picked up.
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def build_settings(**values: Any) -> Settings:
    """``Settings(...)`` with plain values, as they would arrive from the env.

    The ``Any`` signature lets tests pass ``str`` where the field type is
    ``SecretStr`` or a ``Literal`` without a type-ignore on every call.
    """
    return Settings(**values)


def _errors_for(exc: ValidationError, field: str) -> list[str]:
    return [e["msg"] for e in exc.errors() if field in e["loc"]]


# --- import and caching -------------------------------------------------------


def test_importing_the_module_does_not_validate() -> None:
    # A fresh interpreter with an empty environment: import must not raise.
    result = subprocess.run(
        [sys.executable, "-c", "import common.config"],
        env={},
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_get_settings_builds_settings_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("CHUNK_SEC", "600")

    settings = get_settings()

    assert isinstance(settings, Settings)
    assert settings.DATABASE_URL.get_secret_value() == DB_URL
    assert settings.CHUNK_SEC == 600


def test_get_settings_returns_the_same_cached_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)

    first = get_settings()
    monkeypatch.setenv("CHUNK_SEC", "600")
    second = get_settings()

    assert second is first
    assert second.CHUNK_SEC == 900


def test_get_settings_raises_when_database_url_is_unset() -> None:
    with pytest.raises(ValidationError) as exc_info:
        get_settings()

    assert "DATABASE_URL" in str(exc_info.value)


# --- defaults -----------------------------------------------------------------


def test_defaults_match_the_architecture_table() -> None:
    s = build_settings(DATABASE_URL=DB_URL)

    assert s.DATABASE_URL.get_secret_value() == DB_URL
    assert s.SUMMARIZER == "ollama"
    assert s.OLLAMA_MODEL == "qwen3.5:4b"
    assert s.OLLAMA_HOST == "http://host.docker.internal:11434"
    assert s.OLLAMA_NUM_CTX == 8192
    assert s.OLLAMA_OUTPUT_RESERVE_TOKENS == 2048
    assert s.OLLAMA_TIMEOUT_SEC == 900
    assert s.ANTHROPIC_API_KEY is None
    assert s.ANTHROPIC_MODEL == "claude-haiku-4-5"
    assert s.ANTHROPIC_BATCH is True
    assert s.ANTHROPIC_MAX_TOKENS == 4096
    assert s.ANTHROPIC_BATCH_POLL_SEC == 30
    assert s.ANTHROPIC_BATCH_MAX_WAIT_SEC == 3600
    assert s.PROMPT_VERSION == "v1"
    assert s.CHUNK_SEC == 900
    assert s.OVERLAP_SEC == 60
    assert s.WHISPER_MODEL == "large-v3"
    assert s.WHISPER_COMPUTE == "int8"
    assert s.WHISPER_THREADS == 0
    assert s.PREFER_WHISPER is False
    assert s.AUTO_CAPTION_FALLBACK is True
    assert s.AUDIO_KEEP is True
    assert s.AUDIO_TTL_DAYS == 30
    assert s.AUDIO_MAX_GB == 20
    assert s.POLL_INTERVAL_SEC == 3600
    assert s.HEARTBEAT_SEC == 60
    assert s.REAP_AFTER_SEC == 300
    assert s.MAX_ATTEMPTS_TRANSCRIBE == 2
    assert s.LOG_LEVEL == "INFO"
    assert s.LOG_FORMAT == "json"
    # Not in the architecture table; added by #5 for later tasks.
    assert s.AUDIO_DIR == Path("/data/audio")
    assert s.YTDLP_COOKIE_FILE is None
    assert s.WORKER_SHUTDOWN_GRACE_SEC == 20


def test_every_architecture_variable_is_a_field_except_transcriber_cpus() -> None:
    expected = {
        "DATABASE_URL", "SUMMARIZER", "OLLAMA_MODEL", "OLLAMA_HOST",
        "OLLAMA_NUM_CTX", "OLLAMA_OUTPUT_RESERVE_TOKENS", "OLLAMA_TIMEOUT_SEC",
        "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_BATCH",
        "ANTHROPIC_MAX_TOKENS", "ANTHROPIC_BATCH_POLL_SEC",
        "ANTHROPIC_BATCH_MAX_WAIT_SEC",
        "PROMPT_VERSION", "CHUNK_SEC", "OVERLAP_SEC", "WHISPER_MODEL",
        "WHISPER_COMPUTE", "WHISPER_THREADS", "PREFER_WHISPER",
        "AUTO_CAPTION_FALLBACK", "AUDIO_KEEP", "AUDIO_TTL_DAYS",
        "AUDIO_MAX_GB", "POLL_INTERVAL_SEC", "HEARTBEAT_SEC",
        "REAP_AFTER_SEC", "MAX_ATTEMPTS_TRANSCRIBE", "LOG_LEVEL",
        "LOG_FORMAT", "AUDIO_DIR", "YTDLP_COOKIE_FILE",
        "WORKER_SHUTDOWN_GRACE_SEC", "REAP_INTERVAL_SEC",
        "INGEST_MAX_RUNTIME_SEC", "ANALYZE_MAX_RUNTIME_SEC", "TRANSCRIBE_MAX_RTF",
        "ALERT_WEBHOOK_URL", "ALERT_WEBHOOK_FORMAT", "ALERT_MIN_INTERVAL_SEC",
    }  # fmt: skip

    assert set(Settings.model_fields) == expected


# --- DATABASE_URL -------------------------------------------------------------


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_database_url_is_rejected_by_name(value: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        build_settings(DATABASE_URL=value)

    assert _errors_for(exc_info.value, "DATABASE_URL")


def test_empty_database_url_in_the_environment_is_rejected_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "")

    with pytest.raises(ValidationError) as exc_info:
        get_settings()

    assert _errors_for(exc_info.value, "DATABASE_URL")


def test_missing_database_url_is_rejected_by_name() -> None:
    with pytest.raises(ValidationError) as exc_info:
        build_settings()

    assert _errors_for(exc_info.value, "DATABASE_URL")


# --- SUMMARIZER ---------------------------------------------------------------


@pytest.mark.parametrize("value", ["ollama", "anthropic"])
def test_summarizer_accepts_the_two_backends(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("SUMMARIZER", value)

    assert get_settings().SUMMARIZER == value


def test_unknown_summarizer_is_rejected_listing_the_allowed_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("SUMMARIZER", "openai")

    with pytest.raises(ValidationError) as exc_info:
        get_settings()

    [msg] = _errors_for(exc_info.value, "SUMMARIZER")
    assert "'ollama'" in msg
    assert "'anthropic'" in msg


def test_anthropic_summarizer_loads_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # api and planner share SUMMARIZER but never receive the key (11.4).
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("SUMMARIZER", "anthropic")

    settings = get_settings()

    assert settings.SUMMARIZER == "anthropic"
    assert settings.ANTHROPIC_API_KEY is None


def test_empty_api_key_from_compose_counts_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Compose passes ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}, i.e. "".
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    assert get_settings().ANTHROPIC_API_KEY is None


def test_api_key_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)

    key = get_settings().ANTHROPIC_API_KEY
    assert key is not None
    assert key.get_secret_value() == API_KEY


# --- chunking -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("chunk", "overlap"),
    [
        ("0", "0"),  # zero-length chunk
        ("-900", "0"),  # negative chunk
        ("900", "-1"),  # negative overlap
        ("900", "900"),  # overlap exactly at the limit
        ("900", "901"),  # overlap past the limit
    ],
)
def test_invalid_chunk_window_is_rejected(
    monkeypatch: pytest.MonkeyPatch, chunk: str, overlap: str
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("CHUNK_SEC", chunk)
    monkeypatch.setenv("OVERLAP_SEC", overlap)

    with pytest.raises(ValidationError):
        get_settings()


def test_overlap_at_limit_error_names_both_variables() -> None:
    with pytest.raises(ValidationError) as exc_info:
        build_settings(DATABASE_URL=DB_URL, CHUNK_SEC=900, OVERLAP_SEC=900)

    message = str(exc_info.value)
    assert "OVERLAP_SEC" in message
    assert "CHUNK_SEC" in message


@pytest.mark.parametrize(("chunk", "overlap"), [(1, 0), (900, 899), (900, 0)])
def test_valid_chunk_window_edges_are_accepted(chunk: int, overlap: int) -> None:
    s = build_settings(DATABASE_URL=DB_URL, CHUNK_SEC=chunk, OVERLAP_SEC=overlap)

    assert (s.CHUNK_SEC, s.OVERLAP_SEC) == (chunk, overlap)


# --- OLLAMA_OUTPUT_RESERVE_TOKENS vs OLLAMA_NUM_CTX (#85) ---------------------


@pytest.mark.parametrize(("num_ctx", "reserve"), [(8192, 8192), (8192, 8193), (4096, 2048 * 2)])
def test_output_reserve_not_below_the_context_window_is_rejected_naming_both(
    num_ctx: int, reserve: int
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        build_settings(
            DATABASE_URL=DB_URL, OLLAMA_NUM_CTX=num_ctx, OLLAMA_OUTPUT_RESERVE_TOKENS=reserve
        )

    [message] = _errors_for(exc_info.value, "OLLAMA_OUTPUT_RESERVE_TOKENS")
    assert "OLLAMA_OUTPUT_RESERVE_TOKENS" in message
    assert "OLLAMA_NUM_CTX" in message


@pytest.mark.parametrize(("num_ctx", "reserve"), [(8192, 8191), (2049, 2048), (2, 1)])
def test_output_reserve_just_below_the_context_window_is_accepted(
    num_ctx: int, reserve: int
) -> None:
    s = build_settings(
        DATABASE_URL=DB_URL, OLLAMA_NUM_CTX=num_ctx, OLLAMA_OUTPUT_RESERVE_TOKENS=reserve
    )

    assert (s.OLLAMA_NUM_CTX, s.OLLAMA_OUTPUT_RESERVE_TOKENS) == (num_ctx, reserve)


@pytest.mark.parametrize("value", [0, -1])
def test_output_reserve_must_be_positive(value: int) -> None:
    with pytest.raises(ValidationError) as exc_info:
        build_settings(DATABASE_URL=DB_URL, OLLAMA_OUTPUT_RESERVE_TOKENS=value)

    assert _errors_for(exc_info.value, "OLLAMA_OUTPUT_RESERVE_TOKENS")


def test_output_reserve_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("OLLAMA_OUTPUT_RESERVE_TOKENS", "1000")

    assert get_settings().OLLAMA_OUTPUT_RESERVE_TOKENS == 1000


# --- REAP_AFTER_SEC vs HEARTBEAT_SEC -------------------------------------------


def test_reap_after_below_twice_the_heartbeat_is_rejected_naming_both() -> None:
    with pytest.raises(ValidationError) as exc_info:
        build_settings(DATABASE_URL=DB_URL, HEARTBEAT_SEC=60, REAP_AFTER_SEC=119)

    message = str(exc_info.value)
    assert "REAP_AFTER_SEC" in message
    assert "HEARTBEAT_SEC" in message


def test_reap_after_of_exactly_twice_the_heartbeat_is_accepted() -> None:
    s = build_settings(DATABASE_URL=DB_URL, HEARTBEAT_SEC=60, REAP_AFTER_SEC=120)

    assert (s.HEARTBEAT_SEC, s.REAP_AFTER_SEC) == (60, 120)


def test_default_reap_and_heartbeat_pass_the_rule() -> None:
    s = build_settings(DATABASE_URL=DB_URL)

    assert (s.HEARTBEAT_SEC, s.REAP_AFTER_SEC) == (60, 300)


# --- PROMPT_VERSION -----------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["../../etc", "", "V1", "v1/..", "..", ".", "v1\n", "v 1", "v1.2", "/abs"],
)
def test_prompt_version_rejects_anything_but_a_safe_directory_name(
    value: str,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        build_settings(DATABASE_URL=DB_URL, PROMPT_VERSION=value)

    assert _errors_for(exc_info.value, "PROMPT_VERSION")


@pytest.mark.parametrize("value", ["v1", "v2", "2026-09_draft", "a"])
def test_prompt_version_accepts_lowercase_digits_dash_underscore(
    value: str,
) -> None:
    assert (
        build_settings(DATABASE_URL=DB_URL, PROMPT_VERSION=value).PROMPT_VERSION
        == value
    )


# --- logging ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("debug", "DEBUG"), ("Info", "INFO"), ("WARNING", "WARNING"), ("error", "ERROR")],
)
def test_log_level_is_case_insensitive(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: str
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("LOG_LEVEL", value)

    assert get_settings().LOG_LEVEL == expected


@pytest.mark.parametrize("value", ["verbose", "", "trace"])
def test_unknown_log_level_is_rejected(value: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        build_settings(DATABASE_URL=DB_URL, LOG_LEVEL=value)

    assert _errors_for(exc_info.value, "LOG_LEVEL")


@pytest.mark.parametrize("value", ["json", "console"])
def test_log_format_accepts_json_and_console(value: str) -> None:
    assert build_settings(DATABASE_URL=DB_URL, LOG_FORMAT=value).LOG_FORMAT == value


@pytest.mark.parametrize("value", ["text", "", "logfmt"])
def test_log_format_rejects_anything_else(value: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        build_settings(DATABASE_URL=DB_URL, LOG_FORMAT=value)

    assert _errors_for(exc_info.value, "LOG_FORMAT")


# --- boolean flags ------------------------------------------------------------

BOOL_FLAGS = [
    "PREFER_WHISPER",
    "AUTO_CAPTION_FALLBACK",
    "AUDIO_KEEP",
    "ANTHROPIC_BATCH",
]


@pytest.mark.parametrize("flag", BOOL_FLAGS)
@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_boolean_flags_accept_zero_and_one(
    monkeypatch: pytest.MonkeyPatch, flag: str, raw: str, expected: bool
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv(flag, raw)

    assert getattr(get_settings(), flag) is expected


@pytest.mark.parametrize("flag", BOOL_FLAGS)
def test_boolean_flags_reject_garbage(
    monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv(flag, "2")

    with pytest.raises(ValidationError) as exc_info:
        get_settings()

    assert _errors_for(exc_info.value, flag)


# --- other numeric fields -----------------------------------------------------


@pytest.mark.parametrize("value", ["0", "-1"])
def test_worker_shutdown_grace_must_be_positive(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("WORKER_SHUTDOWN_GRACE_SEC", value)

    with pytest.raises(ValidationError) as exc_info:
        get_settings()

    assert _errors_for(exc_info.value, "WORKER_SHUTDOWN_GRACE_SEC")


def test_worker_shutdown_grace_of_one_second_is_accepted() -> None:
    s = build_settings(DATABASE_URL=DB_URL, WORKER_SHUTDOWN_GRACE_SEC=1)

    assert s.WORKER_SHUTDOWN_GRACE_SEC == 1


def test_reap_interval_defaults_to_one_minute() -> None:
    assert build_settings(DATABASE_URL=DB_URL).REAP_INTERVAL_SEC == 60


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5"])
def test_reap_interval_must_be_a_positive_int(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("REAP_INTERVAL_SEC", value)

    with pytest.raises(ValidationError) as exc_info:
        get_settings()

    assert _errors_for(exc_info.value, "REAP_INTERVAL_SEC")


def test_reap_interval_of_one_second_is_accepted() -> None:
    assert build_settings(DATABASE_URL=DB_URL, REAP_INTERVAL_SEC=1).REAP_INTERVAL_SEC == 1


def test_non_numeric_value_error_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("HEARTBEAT_SEC", "sixty")

    with pytest.raises(ValidationError) as exc_info:
        get_settings()

    assert _errors_for(exc_info.value, "HEARTBEAT_SEC")


# --- Anthropic ----------------------------------------------------------------

ANTHROPIC_NUMERIC = [
    "ANTHROPIC_MAX_TOKENS",
    "ANTHROPIC_BATCH_POLL_SEC",
    "ANTHROPIC_BATCH_MAX_WAIT_SEC",
]


@pytest.mark.parametrize("name", ANTHROPIC_NUMERIC)
@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_anthropic_numeric_settings_must_be_positive_integers(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError) as exc_info:
        get_settings()

    assert _errors_for(exc_info.value, name)


def test_anthropic_numeric_settings_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "1000")
    monkeypatch.setenv("ANTHROPIC_BATCH_POLL_SEC", "5")
    monkeypatch.setenv("ANTHROPIC_BATCH_MAX_WAIT_SEC", "60")

    s = get_settings()

    assert (s.ANTHROPIC_MAX_TOKENS, s.ANTHROPIC_BATCH_POLL_SEC, s.ANTHROPIC_BATCH_MAX_WAIT_SEC) == (
        1000,
        5,
        60,
    )


# --- Ollama -------------------------------------------------------------------


@pytest.mark.parametrize("name", ["OLLAMA_NUM_CTX", "OLLAMA_TIMEOUT_SEC"])
@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_ollama_numeric_settings_must_be_positive_integers(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError) as exc_info:
        get_settings()

    assert _errors_for(exc_info.value, name)


def test_ollama_numeric_settings_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SEC", "1")

    s = get_settings()

    assert (s.OLLAMA_NUM_CTX, s.OLLAMA_TIMEOUT_SEC) == (4096, 1)


@pytest.mark.parametrize(
    "value",
    [
        "localhost:11434",  # the form Ollama's own env var accepts
        "host.docker.internal:11434",
        "",
        "   ",
        "http://",
        "http:///api",
        "http://h:11434?x=1",
        "http://h:11434/?x=1",
        "ftp://h:11434",
        "http://h:notaport",
    ],
)
def test_invalid_ollama_host_is_rejected_naming_the_variable(value: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        build_settings(DATABASE_URL=DB_URL, OLLAMA_HOST=value)

    messages = _errors_for(exc_info.value, "OLLAMA_HOST")
    assert messages
    assert all("OLLAMA_HOST" in m for m in messages)


@pytest.mark.parametrize(
    "value",
    [
        "http://localhost:11434",
        "http://h:11434/",
        "https://ollama.example.com",
        "http://127.0.0.1:11434",
        "http://[::1]:11434",
    ],
)
def test_valid_ollama_host_is_accepted_unchanged(value: str) -> None:
    s = build_settings(DATABASE_URL=DB_URL, OLLAMA_HOST=value)

    assert s.OLLAMA_HOST == value


# --- paths --------------------------------------------------------------------


def test_audio_dir_and_cookie_file_are_read_as_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("AUDIO_DIR", "/srv/audio")
    monkeypatch.setenv("YTDLP_COOKIE_FILE", "/run/secrets/cookies.txt")

    s = get_settings()

    assert s.AUDIO_DIR == Path("/srv/audio")
    assert s.YTDLP_COOKIE_FILE == Path("/run/secrets/cookies.txt")


def test_empty_cookie_file_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("YTDLP_COOKIE_FILE", "")

    assert get_settings().YTDLP_COOKIE_FILE is None


# --- secrets ------------------------------------------------------------------


def test_repr_and_str_hide_the_api_key_and_database_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL_WITH_SECRET)
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)

    settings = get_settings()

    for rendered in (repr(settings), str(settings)):
        assert API_KEY not in rendered
        assert DB_PASSWORD not in rendered


# Letters and digits only, so JSON escaping cannot hide a match.
SECRET_PASSWORD = "Pw9xQ2mLk7TzR4vB"
SECRET_DSN = f"postgresql://u:{SECRET_PASSWORD}@h/db"
SECRET_KEY = "skant9Zq3Wm8Lp2Xc7Vb5Nr1"
PLACEHOLDER = "<redacted>"


def _leak_surfaces(exc: BaseException) -> dict[str, str]:
    assert isinstance(exc, ValidationError)
    return {
        "str": str(exc),
        "repr": repr(exc),
        "json": exc.json(),
        "errors": repr(exc.errors()),
        "traceback": "".join(traceback.format_exception(exc)),
    }


def _assert_no_secret_anywhere(exc: BaseException) -> None:
    for name, text in _leak_surfaces(exc).items():
        for secret in (SECRET_PASSWORD, SECRET_DSN, SECRET_KEY):
            assert secret not in text, f"{secret!r} leaked via {name}"


def _fail(monkeypatch: pytest.MonkeyPatch, **env: str) -> ValidationError:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET_KEY)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError) as exc_info:
        get_settings()
    return exc_info.value


def test_overlap_not_below_chunk_does_not_leak_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exc = _fail(monkeypatch, DATABASE_URL=SECRET_DSN, CHUNK_SEC="900", OVERLAP_SEC="900")

    _assert_no_secret_anywhere(exc)
    [error] = exc.errors()
    assert error["loc"] == ("OVERLAP_SEC",)
    assert "OVERLAP_SEC (900) must be less than CHUNK_SEC (900)" in error["msg"]


def test_reap_below_two_heartbeats_does_not_leak_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exc = _fail(
        monkeypatch, DATABASE_URL=SECRET_DSN, HEARTBEAT_SEC="60", REAP_AFTER_SEC="119"
    )

    _assert_no_secret_anywhere(exc)
    [error] = exc.errors()
    assert error["loc"] == ("REAP_AFTER_SEC",)
    assert "REAP_AFTER_SEC (119) must be at least 2 * HEARTBEAT_SEC (120)" in error["msg"]


def test_missing_database_url_does_not_leak_the_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exc = _fail(monkeypatch)

    _assert_no_secret_anywhere(exc)
    [error] = exc.errors()
    assert error["loc"] == ("DATABASE_URL",)
    assert error["type"] == "missing"


def test_two_errors_at_once_do_not_leak_and_are_both_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exc = _fail(
        monkeypatch,
        DATABASE_URL=SECRET_DSN,
        SUMMARIZER="openai",
        CHUNK_SEC="900",
        OVERLAP_SEC="900",
    )

    _assert_no_secret_anywhere(exc)
    by_loc = {e["loc"]: e for e in exc.errors()}
    assert set(by_loc) == {("SUMMARIZER",), ("OVERLAP_SEC",)}
    assert "ollama" in by_loc[("SUMMARIZER",)]["msg"]
    assert "anthropic" in by_loc[("SUMMARIZER",)]["msg"]
    assert by_loc[("SUMMARIZER",)]["type"] == "literal_error"


@pytest.mark.parametrize(
    "env",
    [
        {"DATABASE_URL": SECRET_DSN, "CHUNK_SEC": "900", "OVERLAP_SEC": "900"},
        {"DATABASE_URL": SECRET_DSN, "HEARTBEAT_SEC": "60", "REAP_AFTER_SEC": "119"},
        {"DATABASE_URL": SECRET_DSN, "SUMMARIZER": "openai"},
        {},
    ],
)
def test_every_error_input_is_the_same_fixed_placeholder(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    exc = _fail(monkeypatch, **env)

    assert exc.errors()
    assert [e["input"] for e in exc.errors()] == [PLACEHOLDER] * len(exc.errors())
    assert [e["input"] for e in json.loads(exc.json())] == [PLACEHOLDER] * len(exc.errors())


def test_raised_error_does_not_chain_to_the_unsanitised_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exc = _fail(monkeypatch, DATABASE_URL=SECRET_DSN, CHUNK_SEC="900", OVERLAP_SEC="900")

    seen: list[BaseException] = []
    link: BaseException | None = exc
    while link is not None and link not in seen:
        seen.append(link)
        link = link.__cause__ or link.__context__
    assert seen == [exc]


def test_failed_get_settings_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    _fail(monkeypatch, DATABASE_URL=SECRET_DSN, CHUNK_SEC="900", OVERLAP_SEC="900")

    monkeypatch.setenv("OVERLAP_SEC", "60")

    assert isinstance(get_settings(), Settings)


# --- sources ------------------------------------------------------------------


def test_dotenv_file_in_the_working_directory_is_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        "DATABASE_URL=postgresql://from:dotenv@h/db\nSUMMARIZER=anthropic\n"
    )

    with pytest.raises(ValidationError) as exc_info:
        get_settings()
    assert _errors_for(exc_info.value, "DATABASE_URL")

    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    settings = get_settings()

    assert settings.DATABASE_URL.get_secret_value() == DB_URL
    assert settings.SUMMARIZER == "ollama"


def test_lowercase_variable_names_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("summarizer", "anthropic")

    assert get_settings().SUMMARIZER == "ollama"


# --- lint rule: only common/config.py reads the environment -------------------

ENV_READS = [
    "import os\n\nprint(os.environ['X'])\n",
    "import os\n\nprint(os.getenv('X'))\n",
    "from os import environ\n\nprint(environ)\n",
    "from os import getenv\n\nprint(getenv('X'))\n",
    "import os\n\nprint(os.environb[b'X'])\n",
    "import os\n\nprint(os.getenvb(b'X'))\n",
    "import os\n\nos.putenv('X', '1')\n",
    "import os\n\nos.unsetenv('X')\n",
    "from os import putenv\n\nputenv('X', '1')\n",
]


def _ruff(
    source: str, as_path: str, *flags: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, "-m", "ruff", "check", "--no-cache", *flags,
            "--stdin-filename", as_path, "-",
        ],
        input=source,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )  # fmt: skip


@pytest.mark.parametrize("source", ENV_READS)
@pytest.mark.parametrize(
    "as_path",
    [
        "tests/common/test_probe.py",
        "services/api/main.py",
        "adapters/x.py",
        "common/y.py",
    ],
)
def test_ruff_flags_environment_reads_outside_config(source: str, as_path: str) -> None:
    result = _ruff(source, as_path, "--select", "TID251")

    assert result.returncode == 1, result.stdout + result.stderr
    assert "TID251" in result.stdout


@pytest.mark.parametrize("source", ENV_READS)
def test_ruff_allows_environment_reads_in_config(source: str) -> None:
    result = _ruff(source, "common/config.py", "--select", "TID251")

    assert result.returncode == 0, result.stdout + result.stderr


# The project's own config (no --select), so dropping TID251 from extend-select,
# deleting a banned-api entry or widening the per-file ignore fails here.

_OUTSIDE_CONFIG_PATHS = [
    "tests/common/test_probe.py",
    "services/api/main.py",
    "adapters/x.py",
    "common/y.py",
]


@pytest.mark.parametrize("source", ENV_READS)
@pytest.mark.parametrize("as_path", _OUTSIDE_CONFIG_PATHS)
def test_project_config_flags_environment_reads_outside_config(
    source: str, as_path: str
) -> None:
    result = _ruff(source, as_path)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "TID251" in result.stdout


@pytest.mark.parametrize("source", ENV_READS)
def test_project_config_allows_environment_reads_in_config(source: str) -> None:
    result = _ruff(source, "common/config.py")

    assert result.returncode == 0, result.stdout + result.stderr
