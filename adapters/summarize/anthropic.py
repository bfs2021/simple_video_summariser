"""Anthropic summarizer adapter (architecture.md 3, 8, 16.8; task #25).

``AnthropicSummarizer`` implements the ``Summarizer`` port against the
Anthropic Messages API. By default each call goes through the Message
Batches API at half the price; ``batch=False`` sends a plain
``POST /v1/messages`` instead. Both modes send the same request parameters,
with the system prompt and the speaker roster as cacheable prefix blocks.

Every reply is untrusted: it goes through the #22 parsers, with the one
repair attempt from ``repair.call_with_repair``. Failures leave as
``common.errors`` classes and nothing else. The API key is read from
``Settings.ANTHROPIC_API_KEY`` only, sent only to the API host, and scrubbed
from every message and log record this module produces.

HTTP is the stdlib only. The round-trip is a ``Transport`` callable so tests
can replace it; ``urllib_transport`` is the real one.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any
from urllib.parse import urlsplit

from adapters.summarize.ollama import _partials_json
from adapters.summarize.prompt_loader import PromptSet, format_roster, load_prompt_set
from adapters.summarize.repair import Message, call_with_repair, cap_feedback
from adapters.summarize.schema import (
    MAX_ERROR_CHARS,
    MAX_FEEDBACK_CHARS,
    SchemaError,
    parse_chunk_analysis,
    parse_roster,
    parse_tldr,
)
from common.config import Settings
from common.errors import (
    BugError,
    JobError,
    LLMInvalidOutputError,
    LLMUnavailableError,
    RateLimitedError,
)
from common.models import Chunk, ChunkAnalysis, Roster, Usage, VideoMeta

_logger = logging.getLogger(__name__)

API_BASE = "https://api.anthropic.com"
API_VERSION = "2023-06-01"

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
#: One completion can take minutes (4096 output tokens); batch plumbing is quick.
SYNC_TIMEOUT_SEC = 300.0
PLUMBING_TIMEOUT_SEC = 60.0
MAX_POLL_FAILURES = 5
_MAX_DETAIL_CHARS = 200
_MAX_RETRY_AFTER_DIGITS = 9

_BATCH_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_ERROR_TYPES_BUG = frozenset(
    {
        "invalid_request_error",
        "authentication_error",
        "permission_error",
        "not_found_error",
        "request_too_large",
    }
)
_ERROR_TYPES_UNAVAILABLE = frozenset({"overloaded_error", "api_error", "timeout_error"})

# $ per million tokens: (input, output). Cache writes cost 1.25x input, cache
# reads 0.1x input, and the Batches API takes 0.5x of everything (16.8).
_PRICES: dict[str, tuple[Fraction, Fraction]] = {
    "claude-haiku-4-5": (Fraction(1), Fraction(5)),
    "claude-haiku-4-5-20251001": (Fraction(1), Fraction(5)),
}
_CACHE_WRITE_FACTOR = Fraction(5, 4)
_CACHE_READ_FACTOR = Fraction(1, 10)
_BATCH_FACTOR = Fraction(1, 2)
_MILLION = 1_000_000

_EPHEMERAL = {"type": "ephemeral"}


@dataclass(frozen=True, slots=True)
class Response:
    """What came back over HTTP: status, at most ``MAX_RESPONSE_BYTES + 1`` bytes,
    and the response headers with lower-cased names."""

    status: int
    body: bytes
    headers: Mapping[str, str]


#: ``transport(method, url, headers, body, timeout_sec)``. Raises ``OSError``
#: (or any exception) when there is no HTTP response; the adapter classifies it.
type Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], Response]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: it would carry the key to another host."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> Response:
    """Send one request. HTTP error statuses are returned, not raised."""
    request = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
    try:
        with _opener.open(request, timeout=timeout) as response:
            return Response(
                response.status,
                response.read(MAX_RESPONSE_BYTES + 1),
                {k.lower(): v for k, v in response.headers.items()},
            )
    except urllib.error.HTTPError as err:
        with err:
            return Response(
                err.code,
                err.read(MAX_RESPONSE_BYTES + 1),
                {k.lower(): v for k, v in err.headers.items()},
            )


def _retry_after(value: str | None) -> int | None:
    """An integer number of seconds, or None for anything else (HTTP dates too)."""
    if value is None:
        return None
    text = value.strip()
    if not re.fullmatch(rf"[0-9]{{1,{_MAX_RETRY_AFTER_DIGITS}}}", text):
        return None
    return int(text)


def _count(usage: Mapping[str, Any], field: str) -> int | None:
    value = usage.get(field)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


@dataclass(frozen=True, slots=True)
class _Job:
    """One chunk's request in a batch round: where it goes, what it sends, how it parses."""

    index: int
    system: list[dict[str, Any]]
    messages: list[Message]
    parse: Callable[[str], ChunkAnalysis]


def _reply_text(doc: Mapping[str, Any]) -> str:
    return "".join(
        block["text"]
        for block in doc["content"]
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def _stop_reason(doc: Mapping[str, Any]) -> str | None:
    stop = doc.get("stop_reason")
    return stop if isinstance(stop, str) else None


class AnthropicSummarizer:
    """``Summarizer`` over the Messages API. One instance per process."""

    name = "anthropic"

    def __init__(
        self,
        settings: Settings,
        *,
        batch: bool | None = None,
        transport: Transport = urllib_transport,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        secret = settings.ANTHROPIC_API_KEY
        key = secret.get_secret_value().strip() if secret is not None else ""
        if not key:
            raise BugError("ANTHROPIC_API_KEY is not set; the anthropic summarizer needs it")
        self._key = key
        self.model = settings.ANTHROPIC_MODEL
        self._max_tokens = settings.ANTHROPIC_MAX_TOKENS
        self._batch = settings.ANTHROPIC_BATCH if batch is None else batch
        self._poll_sec = settings.ANTHROPIC_BATCH_POLL_SEC
        self._max_wait_sec = settings.ANTHROPIC_BATCH_MAX_WAIT_SEC
        self._prompts: PromptSet = load_prompt_set(settings.PROMPT_VERSION)
        self._transport = transport
        self._clock = clock
        self._sleep = sleep
        self._host = urlsplit(API_BASE).hostname
        self._price = _PRICES.get(self.model)
        self._warned_price = False
        self._request_seq = 0
        self._reset_usage()

    def __repr__(self) -> str:
        return f"AnthropicSummarizer(model={self.model!r}, batch={self._batch})"

    # --- port ---------------------------------------------------------------

    def derive_roster(self, meta: VideoMeta, opening: str) -> Roster:
        prompt = self._prompts.render_roster(
            title=meta.title, description=meta.description, opening=opening
        )
        return self._run(
            "derive_roster", meta, [self._instructions(prompt.system)], prompt.user, parse_roster
        )

    def analyze_chunk(self, chunk: Chunk, roster: Roster, meta: VideoMeta) -> ChunkAnalysis:
        if not chunk.text.strip():
            return ChunkAnalysis(topics=(), claims=(), quotes=())
        system, user, parse = self._chunk_call(chunk, roster, meta)
        return self._run("analyze_chunk", meta, system, user, parse)

    def analyze_chunks(
        self, chunks: Sequence[Chunk], roster: Roster, meta: VideoMeta
    ) -> list[ChunkAnalysis]:
        if not self._batch:
            return [self.analyze_chunk(chunk, roster, meta) for chunk in chunks]
        results = [ChunkAnalysis(topics=(), claims=(), quotes=()) for _ in chunks]
        jobs: list[_Job] = []
        for index, chunk in enumerate(chunks):
            if not chunk.text.strip():
                continue
            system, user, parse = self._chunk_call(chunk, roster, meta)
            jobs.append(_Job(index, system, [{"role": "user", "content": user}], parse))
        if not jobs:
            return results

        hard: dict[int, JobError] = {}
        invalid: dict[int, tuple[str, str]] = {}
        parsed: dict[int, ChunkAnalysis] = {}
        first = self._batch_round("analyze_chunks", meta, jobs, repair=False)
        for job in jobs:
            outcome = first[job.index]
            if isinstance(outcome, JobError):
                hard[job.index] = outcome
                continue
            text, stop = _reply_text(outcome), _stop_reason(outcome)
            try:
                parsed[job.index] = self._checked(job.parse, text, stop)
            except SchemaError as err:
                invalid[job.index] = (text, str(err)[:MAX_FEEDBACK_CHARS])
            except LLMInvalidOutputError as err:
                hard[job.index] = err
        if hard:
            raise hard[min(hard)]

        if invalid:
            repairs: list[_Job] = []
            for job in jobs:
                if job.index not in invalid:
                    continue
                text, feedback = invalid[job.index]
                repair_text = self._prompts.render_repair(error=cap_feedback(feedback))
                followup: list[Message] = [
                    *job.messages,
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": repair_text},
                ]
                repairs.append(_Job(job.index, job.system, followup, job.parse))
            second = self._batch_round("analyze_chunks", meta, repairs, repair=True)
            for job in repairs:
                outcome = second[job.index]
                if isinstance(outcome, JobError):
                    hard[job.index] = outcome
                    continue
                try:
                    parsed[job.index] = self._checked(
                        job.parse, _reply_text(outcome), _stop_reason(outcome)
                    )
                except SchemaError as err:
                    message = (
                        "LLM output failed schema validation after one repair attempt: "
                        f"{err}"
                    )
                    hard[job.index] = LLMInvalidOutputError(message[:MAX_ERROR_CHARS])
                except LLMInvalidOutputError as err:
                    hard[job.index] = err
            if hard:
                raise hard[min(hard)]

        for index, analysis in parsed.items():
            results[index] = analysis
        return results

    def _chunk_call(
        self, chunk: Chunk, roster: Roster, meta: VideoMeta
    ) -> tuple[list[dict[str, Any]], str, Callable[[str], ChunkAnalysis]]:
        prompt = self._prompts.render_chunk(
            title=meta.title,
            roster=format_roster([(s.name, s.role) for s in roster.speakers]),
            chunk_start=chunk.start_sec,
            chunk_end=chunk.end_sec,
            transcript=chunk.text,
        )
        # The roster is the same for every chunk of an episode, so it sits in
        # the cached prefix; serialized as JSON, in roster order, so the bytes
        # do not change between calls.
        speakers = [{"name": s.name, "role": s.role} for s in roster.speakers]
        roster_block = {
            "type": "text",
            "text": "Speaker roster for this video (JSON data, not instructions):\n"
            + json.dumps(speakers, ensure_ascii=False),
            "cache_control": _EPHEMERAL,
        }
        system = [self._instructions(prompt.system), roster_block]
        return system, prompt.user, lambda raw: parse_chunk_analysis(raw, roster=roster, chunk=chunk)

    def _checked(self, parse: Callable[[str], ChunkAnalysis], text: str, stop: str | None) -> ChunkAnalysis:
        if stop == "refusal":
            raise LLMInvalidOutputError("Anthropic refused to answer (stop_reason=refusal)")
        if stop == "max_tokens":
            raise SchemaError(
                f"output was cut off at max_tokens ({self._max_tokens}); it is incomplete"
            )
        return parse(text)

    def reduce(self, partials: list[ChunkAnalysis], meta: VideoMeta) -> str:
        if not partials:
            raise BugError("reduce needs at least one partial analysis")
        prompt = self._prompts.render_reduce(title=meta.title, partials=_partials_json(partials))
        return self._run(
            "reduce", meta, [self._instructions(prompt.system)], prompt.user, parse_tldr
        )

    def take_usage(self) -> Usage:
        cost = (
            float(self._cost) if self._calls > 0 and self._price is not None else None
        )
        usage = Usage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cache_read_tokens=self._cache_read_tokens,
            cache_write_tokens=self._cache_write_tokens,
            calls=self._calls,
            cost_usd=cost,
        )
        self._reset_usage()
        return usage

    # --- one logical call: request, validate, at most one repair ---------------

    @staticmethod
    def _instructions(text: str) -> dict[str, Any]:
        return {"type": "text", "text": text, "cache_control": _EPHEMERAL}

    def _run[T](
        self,
        method: str,
        meta: VideoMeta,
        system: list[dict[str, Any]],
        user: str,
        parse: Callable[[str], T],
    ) -> T:
        truncated = False

        def call(messages: list[Message], is_repair: bool) -> str:
            nonlocal truncated
            text, stop_reason = self._complete(method, meta, system, messages, is_repair)
            truncated = stop_reason == "max_tokens"
            return text

        def checked(raw: str) -> T:
            if truncated:
                raise SchemaError(
                    f"output was cut off at max_tokens ({self._max_tokens}); it is incomplete"
                )
            return parse(raw)

        return call_with_repair(
            [{"role": "user", "content": user}],
            call=call,
            parse=checked,
            render_repair=lambda error: self._prompts.render_repair(error=error),
        )

    def _complete(
        self,
        method: str,
        meta: VideoMeta,
        system: list[dict[str, Any]],
        messages: list[Message],
        is_repair: bool,
    ) -> tuple[str, str | None]:
        """One request in the configured mode. Returns (text, stop_reason)."""
        params = self._params(system, messages)
        started = self._clock()
        doc = (
            self._batched(method, meta, params)
            if self._batch
            else self._sync(params)
        )
        self._account(doc)
        stop_reason = _stop_reason(doc)
        text = _reply_text(doc)
        usage = doc["usage"]
        _logger.info(
            "anthropic_call",
            extra={
                "method": method,
                "model": self.model,
                "batch": self._batch,
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "duration_ms": int((self._clock() - started) * 1000),
                "stop_reason": stop_reason,
                "repair": is_repair,
                "response_chars": len(text),
            },
        )
        if stop_reason == "refusal":
            raise LLMInvalidOutputError("Anthropic refused to answer (stop_reason=refusal)")
        return text, stop_reason

    # --- sync mode ---------------------------------------------------------

    def _sync(self, params: dict[str, Any]) -> dict[str, Any]:
        response = self._send("POST", "/v1/messages", self._dump(params), SYNC_TIMEOUT_SEC)
        self._check_status(response)
        return self._message(self._json(response, "messages response"))

    # --- batch mode --------------------------------------------------------

    def _batched(
        self, method: str, meta: VideoMeta, params: dict[str, Any]
    ) -> dict[str, Any]:
        custom_id = self._custom_id(method, meta)
        batch_id, entries = self._run_batch(method, [(custom_id, params)])
        outcome = self._outcome(batch_id, entries.get(custom_id))
        if isinstance(outcome, JobError):
            raise outcome
        return outcome

    def _run_batch(
        self, method: str, requests: Sequence[tuple[str, dict[str, Any]]]
    ) -> tuple[str, dict[str, dict[str, Any]]]:
        """Submit ``requests`` as one batch, wait for it, return its result lines by custom_id."""
        create = self._send(
            "POST",
            "/v1/messages/batches",
            self._dump(
                {"requests": [{"custom_id": cid, "params": p} for cid, p in requests]}
            ),
            PLUMBING_TIMEOUT_SEC,
        )
        self._check_status(create)
        batch_id = self._json(create, "batch create response").get("id")
        if not isinstance(batch_id, str) or not _BATCH_ID.fullmatch(batch_id):
            raise LLMUnavailableError("Anthropic batch create response has no usable id")
        _logger.info(
            "anthropic_batch_submitted",
            extra={"method": method, "batch_id": batch_id, "requests": len(requests)},
        )
        results_url = self._wait_for_batch(batch_id)
        return batch_id, self._batch_entries(batch_id, results_url)

    def _batch_round(
        self, method: str, meta: VideoMeta, jobs: Sequence[_Job], *, repair: bool
    ) -> dict[int, dict[str, Any] | JobError]:
        """One batch for ``jobs``; each job's message, or the error it ended with."""
        tag = "repair" if repair else "chunk"
        video = re.sub(r"[^A-Za-z0-9_-]", "_", meta.video_id)[:32] or "video"
        requests = [
            (f"{tag}-{job.index}-{video}"[:64], self._params(job.system, job.messages))
            for job in jobs
        ]
        batch_id, entries = self._run_batch(method, requests)
        outcomes: dict[int, dict[str, Any] | JobError] = {}
        for job, (custom_id, _) in zip(jobs, requests, strict=True):
            outcome = self._outcome(batch_id, entries.get(custom_id))
            if isinstance(outcome, dict):
                self._account(outcome)
                _logger.info(
                    "anthropic_call",
                    extra={
                        "method": method,
                        "model": self.model,
                        "batch": True,
                        "input_tokens": outcome["usage"]["input_tokens"],
                        "output_tokens": outcome["usage"]["output_tokens"],
                        "stop_reason": _stop_reason(outcome),
                        "repair": repair,
                        "response_chars": len(_reply_text(outcome)),
                    },
                )
            outcomes[job.index] = outcome
        return outcomes

    def _params(self, system: list[dict[str, Any]], messages: list[Message]) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "system": system,
            # The API rejects an empty assistant turn; a blank reply is
            # possible (that is what the repair is for).
            "messages": [
                {**m, "content": m["content"] or "(no output)"}
                if m["role"] == "assistant" and not m["content"].strip()
                else m
                for m in messages
            ],
        }

    def _custom_id(self, method: str, meta: VideoMeta) -> str:
        self._request_seq += 1
        video = re.sub(r"[^A-Za-z0-9_-]", "_", meta.video_id)[:32] or "video"
        return f"{self._request_seq}_{method}_{video}"[:64]

    def _wait_for_batch(self, batch_id: str) -> str:
        """Poll until the batch has ended; returns the results URL."""
        path = f"/v1/messages/batches/{batch_id}"
        started = self._clock()
        failures = 0
        while True:
            if self._clock() - started > self._max_wait_sec:
                self._cancel(batch_id)
                raise LLMUnavailableError(
                    f"Anthropic batch {batch_id} did not end within "
                    f"{self._max_wait_sec}s; cancel requested"
                )
            try:
                response = self._send("GET", path, None, PLUMBING_TIMEOUT_SEC)
                if response.status >= 500 or response.status == 429:
                    raise LLMUnavailableError(f"Anthropic poll returned HTTP {response.status}")
                self._check_status(response)
                doc = self._json(response, "batch poll response")
                status = doc.get("processing_status")
                if not isinstance(status, str):
                    raise LLMUnavailableError("Anthropic poll response has no processing_status")
            except LLMUnavailableError as err:
                failures += 1
                _logger.warning(
                    "anthropic_poll_failed",
                    extra={"batch_id": batch_id, "failures": failures, "error": str(err)},
                )
                if failures >= MAX_POLL_FAILURES:
                    raise LLMUnavailableError(
                        f"Anthropic batch {batch_id}: {failures} consecutive poll failures; "
                        f"last: {err}"
                    ) from None
            else:
                failures = 0
                if status == "ended":
                    url = doc.get("results_url")
                    return self._trusted_results_url(batch_id, url)
            self._sleep(self._poll_sec)

    def _trusted_results_url(self, batch_id: str, url: object) -> str:
        """The key goes to ``results_url`` only when it is https on the API host."""
        try:
            parts = urlsplit(url) if isinstance(url, str) else None
            trusted = (
                parts is not None
                and parts.scheme == "https"
                and parts.hostname == self._host
                and parts.port in (None, 443)
                and parts.username is None
                and parts.password is None
            )
        except ValueError:
            trusted = False
        if not trusted or not isinstance(url, str):
            raise LLMUnavailableError(
                f"Anthropic batch {batch_id} has no results_url on {self._host} over https"
            )
        return url

    def _cancel(self, batch_id: str) -> None:
        """Best effort: a failure is logged, never raised."""
        try:
            response = self._send(
                "POST", f"/v1/messages/batches/{batch_id}/cancel", b"{}", PLUMBING_TIMEOUT_SEC
            )
            if not 200 <= response.status < 300:
                raise self._http_error(response)
        except Exception as err:  # noqa: BLE001 - best effort, must not hide the timeout
            _logger.warning(
                "anthropic_batch_cancel_failed",
                extra={
                    "batch_id": batch_id,
                    "error": self._redact(f"{type(err).__name__}: {err}"),
                },
            )

    def _batch_entries(self, batch_id: str, url: str) -> dict[str, dict[str, Any]]:
        """The result lines by custom_id; the first line for an id wins."""
        response = self._send("GET", url, None, PLUMBING_TIMEOUT_SEC)
        self._check_status(response)
        entries: dict[str, dict[str, Any]] = {}
        for line in response.body.decode("utf-8", "replace").splitlines():
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except (ValueError, RecursionError):
                _logger.debug("anthropic_results_bad_line", extra={"batch_id": batch_id})
                continue
            if not isinstance(doc, dict) or not isinstance(doc.get("custom_id"), str):
                continue
            if doc["custom_id"] in entries:
                _logger.debug("anthropic_results_duplicate_line", extra={"batch_id": batch_id})
                continue
            entries[doc["custom_id"]] = doc
        return entries

    def _outcome(self, batch_id: str, entry: dict[str, Any] | None) -> dict[str, Any] | JobError:
        """A succeeded request's message, or the error it ended with."""
        if entry is None:
            return LLMUnavailableError(
                f"Anthropic batch {batch_id} results have no line for our request"
            )
        result = entry.get("result")
        kind = result.get("type") if isinstance(result, dict) else None
        if kind == "succeeded" and isinstance(result, dict):
            try:
                return self._message(result.get("message"))
            except LLMUnavailableError as err:
                return err
        if kind == "errored" and isinstance(result, dict):
            error_type, detail = self._error_fields(result.get("error"))
            return self._classify(None, error_type, detail, None)
        return LLMUnavailableError(
            f"Anthropic batch {batch_id} request ended as {self._detail(kind)}"
        )

    # --- HTTP --------------------------------------------------------------

    @staticmethod
    def _dump(payload: dict[str, Any]) -> bytes:
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _send(self, method: str, target: str, body: bytes | None, timeout: float) -> Response:
        url = target if target.startswith("https://") else API_BASE + target
        headers = {
            "x-api-key": self._key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        try:
            response = self._transport(method, url, headers, body, timeout)
        except Exception as err:  # noqa: BLE001 - any transport failure is classified below
            # ``from None``: the cause's text is not scrubbed and format_error()
            # prints chained exceptions.
            raise LLMUnavailableError(
                self._redact(
                    f"Anthropic {method} {urlsplit(url).path} failed: "
                    f"{type(err).__name__}: {err}"
                )
            ) from None
        if len(response.body) > MAX_RESPONSE_BYTES:
            raise LLMUnavailableError(f"Anthropic response is larger than {MAX_RESPONSE_BYTES} bytes")
        return response

    def _check_status(self, response: Response) -> None:
        if not 200 <= response.status < 300:
            raise self._http_error(response)

    def _http_error(self, response: Response) -> JobError:
        try:
            doc = json.loads(response.body.decode("utf-8", "replace"))
        except (ValueError, RecursionError):
            doc = None
        error_type, detail = self._error_fields(doc.get("error") if isinstance(doc, dict) else None)
        return self._classify(
            response.status,
            error_type,
            detail,
            _retry_after(response.headers.get("retry-after")),
        )

    def _error_fields(self, error: object) -> tuple[str | None, str]:
        """(type, message) from ``{"type": ..., "message": ...}``, also when nested
        one level deeper as in a batch ``errored`` result."""
        if isinstance(error, dict) and error.get("type") == "error":
            error = error.get("error")
        if not isinstance(error, dict):
            return None, ""
        kind = error.get("type")
        message = error.get("message")
        return (
            kind if isinstance(kind, str) else None,
            message if isinstance(message, str) else "",
        )

    def _classify(
        self, status: int | None, error_type: str | None, detail: str, retry_after: int | None
    ) -> JobError:
        text = (
            f"Anthropic API error: HTTP {status if status is not None else 'n/a'}, "
            f"type {self._detail(error_type)}: {self._detail(detail)}"
        )
        if status == 429 or error_type == "rate_limit_error":
            return RateLimitedError(text, retry_after_sec=retry_after)
        if (
            (status is not None and status >= 500)
            or error_type in _ERROR_TYPES_UNAVAILABLE
            or status == 408
        ):
            return LLMUnavailableError(text)
        if error_type in _ERROR_TYPES_BUG or (status is not None and 400 <= status < 500):
            return BugError(text)
        return LLMUnavailableError(text)

    def _json(self, response: Response, what: str) -> dict[str, Any]:
        try:
            doc = json.loads(response.body.decode("utf-8"))
        except (ValueError, RecursionError):
            raise LLMUnavailableError(f"Anthropic {what} is not JSON") from None
        if not isinstance(doc, dict):
            raise LLMUnavailableError(f"Anthropic {what} is not a JSON object")
        return doc

    def _message(self, doc: object) -> dict[str, Any]:
        """A message envelope with ``id``, ``content`` and ``usage``, or LLMUnavailableError."""
        if not isinstance(doc, dict):
            raise LLMUnavailableError("Anthropic message is not a JSON object")
        usage = doc.get("usage")
        if (
            not isinstance(doc.get("id"), str)
            or not isinstance(doc.get("content"), list)
            or not isinstance(usage, dict)
            or _count(usage, "input_tokens") is None
            or _count(usage, "output_tokens") is None
        ):
            raise LLMUnavailableError("Anthropic message lacks id, content or usage")
        return doc

    # --- text hygiene ------------------------------------------------------

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "***")

    def _detail(self, value: object) -> str:
        """Provider-supplied text, scrubbed and short enough for an error message."""
        return self._redact(str(value))[:_MAX_DETAIL_CHARS]

    # --- usage and cost ------------------------------------------------------

    def _reset_usage(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_read_tokens = 0
        self._cache_write_tokens = 0
        self._calls = 0
        self._cost = Fraction(0)

    def _account(self, doc: dict[str, Any]) -> None:
        usage = doc["usage"]
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        cache_write = _count(usage, "cache_creation_input_tokens") or 0
        cache_read = _count(usage, "cache_read_input_tokens") or 0
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._cache_write_tokens += cache_write
        self._cache_read_tokens += cache_read
        self._calls += 1
        if self._price is None:
            if not self._warned_price:
                self._warned_price = True
                _logger.warning(
                    "anthropic_price_unknown",
                    extra={"model": self.model, "detail": "cost_usd is None for this model"},
                )
            return
        price_in, price_out = self._price
        cost = (
            input_tokens * price_in
            + output_tokens * price_out
            + cache_write * price_in * _CACHE_WRITE_FACTOR
            + cache_read * price_in * _CACHE_READ_FACTOR
        ) / _MILLION
        self._cost += cost * _BATCH_FACTOR if self._batch else cost
