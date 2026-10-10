"""Ollama summarizer adapter (architecture.md 3, 8; task #24).

``OllamaSummarizer`` implements the ``Summarizer`` port against a host-run
Ollama's ``POST /api/chat``. Every reply is untrusted: it goes through the
#22 parsers, with the one repair attempt from ``repair.call_with_repair``.
Failures leave as ``common.errors`` classes and nothing else.

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
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal
from urllib.parse import unquote, urlsplit, urlunsplit

from adapters.summarize.prompt_loader import (
    PromptSet,
    RenderedPrompt,
    format_roster,
    load_prompt_set,
)
from adapters.summarize.repair import Message, call_with_repair
from adapters.summarize.schema import (
    MAX_CLAIMS,
    MAX_QUOTES,
    MAX_TOPICS,
    parse_chunk_analysis,
    parse_roster,
    parse_tldr,
)
from common.config import Settings, get_settings
from common.errors import BugError, LLMInvalidOutputError, LLMUnavailableError
from common.models import (
    Chunk,
    ChunkAnalysis,
    Roster,
    Usage,
    VideoMeta,
)

_logger = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_BODY_IN_MESSAGE_BYTES = 1024
_MAX_UNAVAILABLE_BODY_CHARS = 200


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """What came back over HTTP: the status and at most ``MAX_RESPONSE_BYTES + 1`` bytes."""

    status: int
    body: bytes


#: ``transport(url, request_body, timeout_sec)``. Raises ``OSError`` (or any
#: exception) when there is no HTTP response; the adapter classifies it.
type Transport = Callable[[str, bytes, float], HttpResponse]


def urllib_transport(url: str, body: bytes, timeout: float) -> HttpResponse:
    """POST ``body`` as JSON. HTTP error statuses are returned, not raised."""
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(response.status, response.read(MAX_RESPONSE_BYTES + 1))
    except urllib.error.HTTPError as err:
        with err:
            return HttpResponse(err.code, err.read(MAX_RESPONSE_BYTES + 1))


#: Fixed timeout of the startup probe (#84). Not ``OLLAMA_TIMEOUT_SEC``, which
#: is sized for generation (900 s); a startup check must not hold the worker up.
PROBE_TIMEOUT_SEC = 5.0

#: ``transport(url, timeout_sec)`` for the probe's GET. Raises on no response.
type GetTransport = Callable[[str, float], HttpResponse]


def urllib_get_transport(url: str, timeout: float) -> HttpResponse:
    """GET ``url``. HTTP error statuses are returned, not raised."""
    request = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(response.status, response.read(MAX_RESPONSE_BYTES + 1))
    except urllib.error.HTTPError as err:
        with err:
            return HttpResponse(err.code, err.read(MAX_RESPONSE_BYTES + 1))


@dataclass(frozen=True, slots=True)
class OllamaProbe:
    """What ``probe_ollama`` found. ``host`` has no credentials; ``error`` is set
    only for ``unreachable``."""

    outcome: Literal["ready", "model_missing", "unreachable"]
    host: str
    model: str
    error: str | None = None


def public_host(host: str) -> str:
    """``host`` without any ``user:password@`` userinfo, safe to log."""
    try:
        parts = urlsplit(host)
        if "@" not in parts.netloc:
            return host
        return urlunsplit(parts._replace(netloc=parts.netloc.rpartition("@")[2]))
    except ValueError:
        return "<invalid OLLAMA_HOST>"


def _secrets_of(host: str) -> list[str]:
    try:
        parts = urlsplit(host)
        userinfo = parts.netloc.rpartition("@")[0] if "@" in parts.netloc else ""
        pieces = [userinfo, parts.username or "", parts.password or ""]
    except ValueError:
        return []
    pieces += [unquote(piece) for piece in pieces]
    return sorted({piece for piece in pieces if piece}, key=len, reverse=True)


def _scrub(text: str, host: str) -> str:
    for secret in _secrets_of(host):
        text = text.replace(secret, "***")
    return text


def _with_implicit_tag(name: str) -> str:
    """``qwen3.5`` -> ``qwen3.5:latest``; a name with a tag is unchanged."""
    last_segment = name.rpartition("/")[2]
    return name if ":" in last_segment else f"{name}:latest"


def _lists_model(models: list[Any], model: str) -> bool:
    wanted = _with_implicit_tag(model)
    for entry in models:
        if not isinstance(entry, dict):
            continue
        for key in ("name", "model"):
            value = entry.get(key)
            if isinstance(value, str) and _with_implicit_tag(value) == wanted:
                return True
    return False


def probe_ollama(
    host: str, model: str, *, transport: GetTransport = urllib_get_transport
) -> OllamaProbe:
    """Ask Ollama once whether it is up and ``model`` is pulled (#84).

    One ``GET <host>/api/tags`` with ``PROBE_TIMEOUT_SEC``; nothing else is
    requested and nothing is raised: every failure becomes ``unreachable``.
    The reply is untrusted. Model names match exactly, except that a missing
    tag means ``latest``, as in Ollama.
    """
    shown = public_host(host)

    def unreachable(reason: str) -> OllamaProbe:
        return OllamaProbe("unreachable", shown, model, _scrub(reason, host))

    try:
        response = transport(host.rstrip("/") + "/api/tags", PROBE_TIMEOUT_SEC)
    except Exception as err:  # noqa: BLE001 - any transport failure means "not reachable"
        return unreachable(f"{type(err).__name__}: {err}")

    quoted = response.body[: _MAX_UNAVAILABLE_BODY_CHARS * 4].decode("utf-8", "replace")
    quoted = quoted[:_MAX_UNAVAILABLE_BODY_CHARS]
    if not 200 <= response.status < 300:
        return unreachable(f"HTTP {response.status}" + (f": {quoted}" if quoted else ""))
    if len(response.body) > MAX_RESPONSE_BYTES:
        return unreachable(f"response is larger than {MAX_RESPONSE_BYTES} bytes")
    try:
        doc = json.loads(response.body.decode("utf-8"))
    except (ValueError, RecursionError):
        return unreachable(f"response is not JSON: {quoted}")
    if not isinstance(doc, dict):
        return unreachable("response is not a JSON object")
    models = doc.get("models")
    if not isinstance(models, list):
        return unreachable("response has no models list")
    if _lists_model(models, model):
        return OllamaProbe("ready", shown, model)
    return OllamaProbe("model_missing", shown, model)


# JSON schemas sent as ``format`` (structured output). They mirror what the
# #22 parsers accept; the parsers stay the authority on validity.
_ROLES = ["host", "guest", "panelist", "unknown"]
_ROSTER_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string", "enum": _ROLES},
                },
                "required": ["name", "role"],
            },
        }
    },
    "required": ["speakers"],
}
_CHUNK_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "start_sec": {"type": "number"},
                },
                "required": ["title", "summary", "start_sec"],
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "speaker": {"type": "string"},
                    "start_sec": {"type": "number"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["text", "speaker", "start_sec", "confidence"],
            },
        },
        "quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "speaker": {"type": "string"},
                    "start_sec": {"type": "number"},
                },
                "required": ["text", "speaker", "start_sec"],
            },
        },
    },
    "required": ["topics", "claims", "quotes"],
}

_THINK_BLOCK = re.compile(r"\A\s*<think>.*?</think>", re.DOTALL)


def _strip_think(content: str) -> str:
    return _THINK_BLOCK.sub("", content, count=1)


def _count(value: object, field: str) -> int:
    """A usable token count, or 0 (Ollama omits the prompt count when cached)."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    _logger.debug("unusable_token_count", extra={"field": field, "type": type(value).__name__})
    return 0


def _partials_json(partials: list[ChunkAnalysis]) -> str:
    return json.dumps(
        [
            {
                "topics": [
                    {"title": t.title, "summary": t.summary, "start_sec": t.start_sec}
                    for t in p.topics
                ],
                "claims": [
                    {
                        "text": c.text,
                        "speaker": c.speaker,
                        "start_sec": c.start_sec,
                        "confidence": c.confidence,
                    }
                    for c in p.claims
                ],
                "quotes": [
                    {"text": q.text, "speaker": q.speaker, "start_sec": q.start_sec}
                    for q in p.quotes
                ],
            }
            for p in partials
        ],
        ensure_ascii=False,
        indent=1,
    )


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: ``ceil(utf-8 bytes / 3)`` (#85).

    Deliberately pessimistic for English and for non-Latin scripts, and needs
    no tokenizer library. The post-call overflow check still guards against it
    being too low.
    """
    return -(-len(text.encode("utf-8")) // 3)


_ELLIPSIS = "…"
_LINE_SECOND = re.compile(r"\[(\d+)\]")


def _cut_to_fit(text: str, fits: Callable[[str], bool]) -> str:
    """The longest prefix of ``text`` plus ``…`` for which ``fits`` holds.

    Whole characters only. ``""`` when not even a bare ``…`` fits. ``fits`` is
    monotone: a longer prefix never fits when a shorter one does not.
    """
    lo, hi = -1, len(text) - 1  # lo: largest known fitting prefix length
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(text[:mid] + _ELLIPSIS):
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + _ELLIPSIS if lo >= 0 else ""


def _merge_analyses(parts: list[ChunkAnalysis]) -> ChunkAnalysis:
    """Concatenate piece results in order, capped like a single analysis."""
    topics = [t for p in parts for t in p.topics]
    claims = [c for p in parts for c in p.claims]
    quotes = [q for p in parts for q in p.quotes]
    capped = sum(
        max(0, len(items) - cap)
        for items, cap in ((topics, MAX_TOPICS), (claims, MAX_CLAIMS), (quotes, MAX_QUOTES))
    )
    return ChunkAnalysis(
        topics=tuple(replace(t, seq=i) for i, t in enumerate(topics[:MAX_TOPICS])),
        claims=tuple(claims[:MAX_CLAIMS]),
        quotes=tuple(quotes[:MAX_QUOTES]),
        speaker_coercions=sum(p.speaker_coercions for p in parts),
        start_sec_clamped=sum(p.start_sec_clamped for p in parts),
        items_dropped=sum(p.items_dropped for p in parts) + capped,
    )


class OllamaSummarizer:
    """``Summarizer`` over Ollama's ``/api/chat``. One instance per process."""

    name = "ollama"

    def __init__(
        self,
        *,
        host: str,
        model: str,
        num_ctx: int,
        timeout_sec: float,
        output_reserve_tokens: int = 2048,
        prompts: PromptSet,
        transport: Transport = urllib_transport,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = model
        self._url = host.rstrip("/") + "/api/chat"
        self._num_ctx = num_ctx
        self._reserve = output_reserve_tokens
        self._budget = num_ctx - output_reserve_tokens
        self._timeout = timeout_sec
        self._prompts = prompts
        self._transport = transport
        self._clock = clock
        self._input_tokens = 0
        self._output_tokens = 0
        self._calls = 0

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, *, transport: Transport = urllib_transport
    ) -> OllamaSummarizer:
        s = settings or get_settings()
        return cls(
            host=s.OLLAMA_HOST,
            model=s.OLLAMA_MODEL,
            num_ctx=s.OLLAMA_NUM_CTX,
            output_reserve_tokens=s.OLLAMA_OUTPUT_RESERVE_TOKENS,
            timeout_sec=s.OLLAMA_TIMEOUT_SEC,
            prompts=load_prompt_set(s.PROMPT_VERSION),
            transport=transport,
        )

    # --- port ---------------------------------------------------------------

    def derive_roster(self, meta: VideoMeta, opening: str) -> Roster:
        description = meta.description

        def tokens(desc: str, open_: str) -> int:
            p = self._prompts.render_roster(title=meta.title, description=desc, opening=open_)
            return estimate_tokens(p.system) + estimate_tokens(p.user)

        budget = self._budget
        original = (len(description), len(opening))
        if tokens(description, opening) > budget:
            # Trim, never split: the description first, then the opening.
            if tokens("", opening) > budget:
                if tokens("", "") > budget:
                    raise LLMInvalidOutputError(
                        f"The roster prompt needs about {tokens('', '')} tokens even without "
                        f"the description and opening, more than the prompt budget of "
                        f"{budget} (OLLAMA_NUM_CTX={self._num_ctx} minus "
                        f"OLLAMA_OUTPUT_RESERVE_TOKENS={self._reserve}); raise OLLAMA_NUM_CTX"
                    )
                description = ""
                opening = _cut_to_fit(opening, lambda t: tokens("", t) <= budget)
            else:
                description = _cut_to_fit(description, lambda t: tokens(t, opening) <= budget)
            _logger.info(
                "ollama_roster_trimmed",
                extra={
                    "description_chars_original": original[0],
                    "description_chars_final": len(description),
                    "opening_chars_original": original[1],
                    "opening_chars_final": len(opening),
                },
            )
        prompt = self._prompts.render_roster(
            title=meta.title, description=description, opening=opening
        )
        return self._run(
            "derive_roster",
            prompt.system,
            prompt.user,
            format=_ROSTER_FORMAT,
            parse=parse_roster,
        )

    def analyze_chunk(self, chunk: Chunk, roster: Roster, meta: VideoMeta) -> ChunkAnalysis:
        if not chunk.text.strip():
            return ChunkAnalysis(topics=(), claims=(), quotes=())
        roster_text = format_roster([(s.name, s.role) for s in roster.speakers])

        def render(text: str, start: float, end: float) -> RenderedPrompt:
            return self._prompts.render_chunk(
                title=meta.title,
                roster=roster_text,
                chunk_start=start,
                chunk_end=end,
                transcript=text,
            )

        def parse(raw: str) -> ChunkAnalysis:
            # Every piece is validated against the whole chunk, not its own span.
            return parse_chunk_analysis(raw, roster=roster, chunk=chunk)

        whole = render(chunk.text, chunk.start_sec, chunk.end_sec)
        estimated = estimate_tokens(whole.system) + estimate_tokens(whole.user)
        if estimated <= self._budget:
            return self._run(
                "analyze_chunk", whole.system, whole.user, format=_CHUNK_FORMAT, parse=parse
            )

        pieces = self._split_chunk(chunk, render)
        _logger.info(
            "ollama_chunk_split",
            extra={
                "chunk_seq": chunk.seq,
                "pieces": len(pieces),
                "estimated_tokens": estimated,
                "budget": self._budget,
            },
        )
        results = [
            self._run("analyze_chunk", p.system, p.user, format=_CHUNK_FORMAT, parse=parse)
            for p in pieces
        ]
        return _merge_analyses(results)

    def _split_chunk(
        self, chunk: Chunk, render: Callable[[str, float, float], RenderedPrompt]
    ) -> list[RenderedPrompt]:
        """Cut ``chunk`` at line boundaries into the fewest prompts that fit.

        A piece shows its own span: from its first line's second to the next
        piece's first second (the chunk's ``end_sec`` for the last piece).
        """
        lines = chunk.text.split("\n")
        seconds: list[int] = []
        for line in lines:
            match = _LINE_SECOND.match(line)
            seconds.append(
                int(match.group(1)) if match else (seconds[-1] if seconds else int(chunk.start_sec))
            )
        n = len(lines)

        def prompt(i: int, j: int) -> RenderedPrompt:  # lines[i:j]
            end = seconds[j] if j < n else chunk.end_sec
            return render("\n".join(lines[i:j]), seconds[i], end)

        def tokens(p: RenderedPrompt) -> int:
            return estimate_tokens(p.system) + estimate_tokens(p.user)

        pieces: list[RenderedPrompt] = []
        i = 0
        while i < n:
            alone = tokens(prompt(i, i + 1))
            if alone > self._budget:
                raise LLMInvalidOutputError(
                    f"chunk {chunk.seq}: the line at second {seconds[i]} needs about "
                    f"{alone} tokens by itself, more than the prompt budget of "
                    f"{self._budget} (OLLAMA_NUM_CTX={self._num_ctx} minus "
                    f"OLLAMA_OUTPUT_RESERVE_TOKENS={self._reserve}); raise OLLAMA_NUM_CTX"
                )
            lo, hi = i + 1, n  # lo always fits; fitting is monotone in the piece length
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if tokens(prompt(i, mid)) <= self._budget:
                    lo = mid
                else:
                    hi = mid - 1
            pieces.append(prompt(i, lo))
            i = lo
        return pieces

    def reduce(self, partials: list[ChunkAnalysis], meta: VideoMeta) -> str:
        if not partials:
            raise BugError("reduce needs at least one partial analysis")
        prompt = self._prompts.render_reduce(title=meta.title, partials=_partials_json(partials))
        return self._run(
            "reduce", prompt.system, prompt.user, format=None, parse=parse_tldr
        )

    def take_usage(self) -> Usage:
        usage = Usage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            calls=self._calls,
        )
        self._input_tokens = self._output_tokens = self._calls = 0
        return usage

    # --- one logical call: request, validate, at most one repair ---------------

    def _run[T](
        self,
        method: str,
        system: str,
        user: str,
        *,
        format: dict[str, Any] | None,
        parse: Callable[[str], T],
    ) -> T:
        def call(messages: list[Message], is_repair: bool) -> str:
            return self._chat(method, messages, format, is_repair)

        return call_with_repair(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            call=call,
            parse=parse,
            render_repair=lambda error: self._prompts.render_repair(error=error),
        )

    # --- one HTTP call -----------------------------------------------------

    def _chat(
        self,
        method: str,
        messages: list[Message],
        format: dict[str, Any] | None,
        is_repair: bool,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"num_ctx": self._num_ctx},
        }
        if format is not None:
            payload["format"] = format
        request = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        started = self._clock()
        try:
            response = self._transport(self._url, request, self._timeout)
        except Exception as err:
            self._log(method, started, 0, 0, None, is_repair, len(request), 0, error=type(err).__name__)
            raise LLMUnavailableError(
                f"Ollama at {self._url} did not answer: {type(err).__name__}: {err}"
            ) from err
        self._calls += 1

        counts = (0, 0)
        done_reason: str | None = None
        content = ""
        try:
            self._check_status(response)
            if len(response.body) > MAX_RESPONSE_BYTES:
                raise LLMUnavailableError(
                    f"Ollama response is larger than {MAX_RESPONSE_BYTES} bytes"
                )
            doc = self._decode(response.body)
            message = doc.get("message")
            text = message.get("content") if isinstance(message, dict) else None
            if not isinstance(text, str):
                raise LLMUnavailableError("Ollama response has no message.content text")
            content = text
            counts = (
                _count(doc.get("prompt_eval_count"), "prompt_eval_count"),
                _count(doc.get("eval_count"), "eval_count"),
            )
            reason = doc.get("done_reason")
            done_reason = reason if isinstance(reason, str) else None
            self._input_tokens += counts[0]
            self._output_tokens += counts[1]
        finally:
            self._log(
                method, started, *counts, done_reason, is_repair, len(request), len(content)
            )

        if done_reason == "length" or sum(counts) >= self._num_ctx:
            raise LLMInvalidOutputError(
                f"Ollama ran out of context (done_reason={done_reason!r}, "
                f"{counts[0]} prompt + {counts[1]} output tokens); "
                f"raise OLLAMA_NUM_CTX (now {self._num_ctx}) or send less text"
            )
        return _strip_think(content)

    def _check_status(self, response: HttpResponse) -> None:
        status = response.status
        if 200 <= status < 300:
            return
        snippet = response.body[:MAX_BODY_IN_MESSAGE_BYTES].decode("utf-8", "replace")
        if status == 404:
            raise LLMUnavailableError(
                f"Ollama has no model {self.model!r} (HTTP 404); run: ollama pull {self.model}"
            )
        if 400 <= status < 500:
            raise BugError(f"Ollama rejected the request with HTTP {status}: {snippet}")
        raise LLMUnavailableError(
            f"Ollama returned HTTP {status}: {snippet[:_MAX_UNAVAILABLE_BODY_CHARS]}"
        )

    @staticmethod
    def _decode(body: bytes) -> dict[str, Any]:
        try:
            doc = json.loads(body.decode("utf-8"))
        except (ValueError, RecursionError):
            raise LLMUnavailableError("Ollama response is not JSON") from None
        if not isinstance(doc, dict):
            raise LLMUnavailableError("Ollama response is not a JSON object")
        if doc.get("error") is not None:
            raise LLMUnavailableError(f"Ollama reported an error: {str(doc['error'])[:_MAX_UNAVAILABLE_BODY_CHARS]}")
        return doc

    def _log(
        self,
        method: str,
        started: float,
        input_tokens: int,
        output_tokens: int,
        done_reason: str | None,
        is_repair: bool,
        request_chars: int,
        response_chars: int,
        error: str | None = None,
    ) -> None:
        _logger.info(
            "ollama_call",
            extra={
                "method": method,
                "model": self.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "duration_ms": int((self._clock() - started) * 1000),
                "done_reason": done_reason,
                "repair": is_repair,
                "request_chars": request_chars,
                "response_chars": response_chars,
                "error": error,
            },
        )
