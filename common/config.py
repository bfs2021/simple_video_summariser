"""Runtime configuration (architecture.md 10).

This is the only module that reads environment variables; ruff's TID251 rule
bans ``os.environ`` and ``os.getenv`` everywhere else. Services call
``get_settings()`` once at startup, so a bad value fails fast with a message
naming the variable.

Importing the module validates nothing: tests, Alembic and tooling can import
it with an empty environment. Only ``get_settings()`` builds and checks the
object.

Values come from the process environment only. There is deliberately no
``.env`` or secrets-directory source, so a stray file in the working
directory cannot change behaviour. Secrets (``DATABASE_URL``,
``ANTHROPIC_API_KEY``) are ``SecretStr`` and never appear in ``repr``,
``str`` or validation errors.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _ollama_host_problem(value: str) -> str | None:
    """Why ``value`` is not a usable Ollama base URL, or ``None`` if it is."""
    try:
        parts = urlsplit(value)
        hostname = parts.hostname
        _ = parts.port  # raises ValueError for a malformed port
    except ValueError:
        return "not a valid URL"
    if parts.scheme not in ("http", "https"):
        return "missing http:// or https:// scheme"
    if not hostname:
        return "no host"
    if parts.query:
        return "has a query string"
    return None


class Settings(BaseSettings):
    """Every backend setting, named exactly as its environment variable."""

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=None,
        secrets_dir=None,
        extra="ignore",
        frozen=True,
        # Validation errors are logged at startup; never echo raw inputs,
        # which would include the API key and the database password.
        hide_input_in_errors=True,
    )

    DATABASE_URL: SecretStr

    # Summarizer. The API key is optional here: api and planner share
    # SUMMARIZER but never receive the key (11.4); the Anthropic adapter
    # checks for it.
    SUMMARIZER: Literal["ollama", "anthropic"] = "ollama"
    OLLAMA_MODEL: str = "qwen3.5:4b"
    OLLAMA_HOST: str = "http://host.docker.internal:11434"
    OLLAMA_NUM_CTX: PositiveInt = 8192
    # CPU inference on a 4B model is slow (architecture.md 16.7).
    OLLAMA_TIMEOUT_SEC: PositiveInt = 900
    ANTHROPIC_API_KEY: SecretStr | None = None
    ANTHROPIC_MODEL: str = "claude-haiku-4-5"
    ANTHROPIC_BATCH: bool = True
    ANTHROPIC_MAX_TOKENS: PositiveInt = 4096
    ANTHROPIC_BATCH_POLL_SEC: PositiveInt = 30
    # A batch usually ends within minutes; give up (and cancel) after an hour.
    ANTHROPIC_BATCH_MAX_WAIT_SEC: PositiveInt = 3600

    # Becomes a directory name under prompts/, so it must be a plain slug.
    PROMPT_VERSION: str = Field(default="v1", pattern=r"^[a-z0-9_-]+$")

    CHUNK_SEC: PositiveInt = 900
    OVERLAP_SEC: NonNegativeInt = 60

    WHISPER_MODEL: str = "large-v3"
    WHISPER_COMPUTE: str = "int8"
    WHISPER_THREADS: NonNegativeInt = 0  # 0 = auto
    PREFER_WHISPER: bool = False
    AUTO_CAPTION_FALLBACK: bool = True

    AUDIO_DIR: Path = Path("/data/audio")
    AUDIO_KEEP: bool = True
    AUDIO_TTL_DAYS: PositiveInt = 30
    AUDIO_MAX_GB: PositiveFloat = 20
    YTDLP_COOKIE_FILE: Path | None = None

    POLL_INTERVAL_SEC: PositiveInt = 3600
    HEARTBEAT_SEC: PositiveInt = 60
    REAP_AFTER_SEC: PositiveInt = 300
    # How often the planner looks for stale jobs (#38); much shorter than
    # POLL_INTERVAL_SEC so a crashed worker's job is not stranded for an hour.
    REAP_INTERVAL_SEC: PositiveInt = 60
    MAX_ATTEMPTS_TRANSCRIBE: PositiveInt = 2
    WORKER_SHUTDOWN_GRACE_SEC: PositiveInt = 20
    # Per-kind maximum job runtime (#66). A handler past it is treated as
    # hung and the worker exits. transcribe is audio duration x the RTF
    # ceiling (real-time factor: processing time / audio time).
    INGEST_MAX_RUNTIME_SEC: PositiveInt = 600
    ANALYZE_MAX_RUNTIME_SEC: PositiveInt = 3600
    TRANSCRIBE_MAX_RTF: PositiveFloat = 3.0

    LOG_LEVEL: LogLevel = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "json"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Constructor arguments and the process environment only.
        return (init_settings, env_settings)

    @field_validator("DATABASE_URL")
    @classmethod
    def _database_url_not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("ANTHROPIC_API_KEY", "YTDLP_COOKIE_FILE", mode="before")
    @classmethod
    def _empty_means_unset(cls, value: Any) -> Any:
        # Compose passes `${VAR:-}` through as an empty string.
        return None if value == "" else value

    @field_validator("OLLAMA_HOST")
    @classmethod
    def _ollama_host_is_http_url(cls, value: str) -> str:
        # Ollama's own OLLAMA_HOST accepts "localhost:11434"; we need a scheme.
        problem = _ollama_host_problem(value)
        if problem:
            raise ValueError(
                f"OLLAMA_HOST must be an http:// or https:// URL with a host "
                f"and no query string ({problem})"
            )
        return value

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def _upper_log_level(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value

    # Cross-field rules are field validators, not model validators, so the
    # error carries the field's own ``loc`` and is reported next to any other
    # field error (an "after" model validator is skipped when a field fails).
    # CHUNK_SEC and HEARTBEAT_SEC are declared first, so ``info.data`` has
    # them unless they were themselves invalid, which is reported separately.
    @field_validator("OVERLAP_SEC")
    @classmethod
    def _overlap_shorter_than_chunk(cls, value: int, info: ValidationInfo) -> int:
        chunk = info.data.get("CHUNK_SEC")
        if chunk is not None and value >= chunk:
            raise ValueError(
                f"OVERLAP_SEC ({value}) must be less than CHUNK_SEC ({chunk})"
            )
        return value

    @field_validator("REAP_AFTER_SEC")
    @classmethod
    def _reap_after_covers_two_heartbeats(cls, value: int, info: ValidationInfo) -> int:
        # A healthy job that misses a single heartbeat must not be reaped.
        heartbeat = info.data.get("HEARTBEAT_SEC")
        if heartbeat is not None and value < 2 * heartbeat:
            raise ValueError(
                f"REAP_AFTER_SEC ({value}) must be at least 2 * HEARTBEAT_SEC ({2 * heartbeat})"
            )
        return value


ERROR_INPUT_PLACEHOLDER = "<redacted>"


def _sanitised(exc: ValidationError) -> ValidationError:
    """Copy of ``exc`` whose every ``input`` is a fixed placeholder.

    ``hide_input_in_errors`` only hides inputs in ``str``/``repr``; for
    missing-field errors ``errors()`` and ``json()`` still carry the whole
    dict of settings values, DSN and API key included.
    """
    line_errors: list[Any] = [
        {
            "type": e["type"],
            "loc": e["loc"],
            "input": ERROR_INPUT_PLACEHOLDER,
            **({"ctx": e["ctx"]} if "ctx" in e else {}),
        }
        for e in exc.errors(include_url=False)
    ]
    return ValidationError.from_exception_data(exc.title, line_errors, hide_input=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build and validate settings from the environment, once per process.

    Raises ``pydantic.ValidationError`` naming each bad variable. Its inputs
    are replaced by a placeholder and it is raised outside the ``except``
    block, so neither it nor its chain can reach a secret. A failure is not
    cached.
    """
    try:
        return Settings()  # type: ignore[call-arg]  # DATABASE_URL comes from env
    except ValidationError as exc:
        sanitised = _sanitised(exc)
    raise sanitised
