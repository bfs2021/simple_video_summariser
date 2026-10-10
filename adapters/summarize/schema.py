"""Turn raw LLM text into typed, validated results (architecture.md 8.1, D4b, D8).

Everything the model writes is untrusted input. The parsers here guarantee:

- every ``speaker`` that comes out is a roster name or ``"unknown"``;
- every ``start_sec`` is ``None`` or inside the chunk span;
- anything that cannot be made well-formed raises ``SchemaError`` - never
  another exception type, and never a half-valid result.

``parse_with_repair`` wraps a parser with the single repair attempt from
8.1; the caller supplies the ``repair`` callable that asks the LLM again.
Pydantic is used for structural validation only; no pydantic type leaves
this module.
"""

import json
import logging
import math
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from common.errors import LLMInvalidOutputError
from common.models import (
    Chunk,
    ChunkAnalysis,
    Claim,
    Quote,
    Roster,
    RosterSpeaker,
    Topic,
)

_logger = logging.getLogger(__name__)

MAX_RAW_CHARS = 256 * 1024
MAX_SPEAKERS = 20
MAX_SPEAKER_NAME_CHARS = 100
MAX_TOPICS = 50
MAX_CLAIMS = 200
MAX_QUOTES = 200
MAX_TITLE_CHARS = 200
MAX_TEXT_CHARS = 2000
MAX_SUMMARY_CHARS = 2000
MAX_TLDR_CHARS = 2000
NONTRIVIAL_CHUNK_WORDS = 150

MAX_FEEDBACK_CHARS = 2000
MAX_ERROR_CHARS = 2000
_MAX_REPORTED_ERRORS = 20
_MAX_LOGGED_VALUE_CHARS = 100
# A stray "{" that does not start valid JSON costs one full decode attempt;
# bound them so hostile input cannot make extraction quadratic.
_MAX_DECODE_ATTEMPTS = 200

_ROLES = frozenset({"host", "guest", "panelist", "unknown"})
_CONFIDENCES = frozenset({"high", "medium", "low"})
UNKNOWN = "unknown"
HONORIFICS = frozenset({"dr", "prof", "professor", "mr", "mrs", "ms", "mx", "sir", "dame"})
NAME_SUFFIXES = frozenset({"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "phd", "md"})


class SchemaError(ValueError):
    """The LLM output cannot be turned into a well-formed result.

    The message names the failing field locations and never quotes the raw
    output, so it is safe to feed back to the model and to log.
    """


class _NonFiniteConstant(ValueError):
    pass


# --- pydantic models (structure only; private) ------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class _RosterDoc(_Model):
    speakers: list[Any]


class _AnalysisDoc(_Model):
    topics: list[Any]
    claims: list[Any]
    quotes: list[Any]


class _TopicItem(_Model):
    title: str
    summary: Any = None
    start_sec: Any = None


class _ClaimItem(_Model):
    text: str
    speaker: Any = None
    start_sec: Any = None
    confidence: Any = None


class _QuoteItem(_Model):
    text: str
    speaker: Any = None
    start_sec: Any = None


def _validate[M: BaseModel](model: type[M], obj: object, where: str = "") -> M:
    """Validate ``obj``; raise ``SchemaError`` naming each failing location."""
    try:
        return model.model_validate(obj)
    except ValidationError as err:
        problems = []
        for e in err.errors()[:_MAX_REPORTED_ERRORS]:
            loc = ".".join(str(p) for p in ((where,) if where else ()) + tuple(e["loc"]))
            problems.append(f"{loc or 'output'}: {e['msg']}")
        raise SchemaError("; ".join(problems)) from None


# --- JSON extraction --------------------------------------------------------


def _reject_constant(name: str) -> object:
    raise _NonFiniteConstant(name)


_decoder = json.JSONDecoder(parse_constant=_reject_constant)


def _skip_fence_opener(text: str) -> str:
    """Drop a leading ``` fence (with optional language tag) from ``text``."""
    if not text.startswith("```"):
        return text
    newline = text.find("\n")
    if newline != -1 and "`" not in text[3:newline]:
        return text[newline + 1 :].lstrip()
    return text[3:].lstrip()


def _extract_object(raw: str) -> dict[str, Any]:
    """Find the one top-level JSON object in ``raw``.

    Tolerates code fences (closed or not), prose around the object, a BOM
    and surrounding whitespace. Raises ``SchemaError`` if there is no
    object, more than one, or the JSON is invalid.
    """
    if len(raw) > MAX_RAW_CHARS:
        raise SchemaError(f"output is longer than {MAX_RAW_CHARS} characters")
    text = raw.lstrip("﻿").strip()
    if not text:
        raise SchemaError("output is empty")

    lead = _skip_fence_opener(text)
    if lead[:1] == "[":
        raise SchemaError("top-level JSON value must be an object, not an array")
    if lead[:1] != "{" and lead[:1] != "":
        try:
            first, _ = _decoder.raw_decode(lead)
        except _NonFiniteConstant:
            raise SchemaError("JSON contains NaN or Infinity") from None
        except RecursionError:
            raise SchemaError("JSON is nested too deeply") from None
        except ValueError:
            pass  # prose before the object; scan for it below
        else:
            raise SchemaError(f"top-level JSON value must be an object, not {type(first).__name__}")

    found: list[dict[str, Any]] = []
    failures = 0
    pos = text.find("{")
    while pos != -1:
        try:
            value, end = _decoder.raw_decode(text, pos)
        except _NonFiniteConstant:
            raise SchemaError("JSON contains NaN or Infinity") from None
        except RecursionError:
            raise SchemaError("JSON is nested too deeply") from None
        except ValueError:
            failures += 1
            if failures > _MAX_DECODE_ATTEMPTS:
                break
            pos = text.find("{", pos + 1)
            continue
        found.append(value)
        if len(found) > 1:
            raise SchemaError("output contains more than one JSON object; it is ambiguous")
        pos = text.find("{", end)

    if not found:
        raise SchemaError("no valid JSON object found in output")
    return found[0]


# --- small helpers ----------------------------------------------------------


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _name_key(value: str) -> str:
    return _normalize(value).casefold()


def _start_sec(value: object, where: str) -> float | None:
    """Parse a ``start_sec``; ``None`` stays ``None``. Raise on bool/garbage."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise SchemaError(f"{where}.start_sec: must be a number, not a boolean")
    number: float
    try:
        if isinstance(value, (int, float)):
            number = float(value)
        elif isinstance(value, str):
            number = float(value.strip())
        else:
            raise SchemaError(f"{where}.start_sec: must be a number")
    except (ValueError, OverflowError):
        raise SchemaError(f"{where}.start_sec: is not a finite number") from None
    if not math.isfinite(number):
        raise SchemaError(f"{where}.start_sec: is not a finite number")
    return number


def _clamp(number: float | None, chunk: Chunk) -> tuple[float | None, bool]:
    if number is None:
        return None, False
    if number < chunk.start_sec:
        return chunk.start_sec, True
    if number > chunk.end_sec:
        return chunk.end_sec, True
    return number, False


# --- roster -----------------------------------------------------------------


def parse_roster(raw: str) -> Roster:
    """Parse the roster pass output (``{"speakers": [...]}``)."""
    doc = _validate(_RosterDoc, _extract_object(raw))
    speakers: list[RosterSpeaker] = []
    seen: set[str] = set()
    for entry in doc.speakers:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        name = _normalize(name)
        key = name.casefold()
        if not name or len(name) > MAX_SPEAKER_NAME_CHARS or key == UNKNOWN or key in seen:
            continue
        seen.add(key)
        role = entry.get("role")
        role_key = role.strip().casefold() if isinstance(role, str) else ""
        speakers.append(RosterSpeaker(name, role_key if role_key in _ROLES else UNKNOWN))
    if len(speakers) > MAX_SPEAKERS:
        _logger.warning(
            "roster_truncated",
            extra={"speakers": len(speakers), "kept": MAX_SPEAKERS},
        )
        speakers = speakers[:MAX_SPEAKERS]
    return Roster(tuple(speakers))


# --- chunk analysis ---------------------------------------------------------


class _Counters:
    def __init__(self) -> None:
        self.coercions = 0
        self.clamped = 0
        self.dropped = 0


def _is_honorific(token: str) -> bool:
    """``dr`` and ``dr.`` count; ``dr..`` and ``d.r.`` do not."""
    return token in HONORIFICS or (token.endswith(".") and token[:-1] in HONORIFICS)


def _stripped_tokens(key: str) -> list[str]:
    """Tokens of a name key with all *leading* honorifics removed."""
    tokens = key.split()
    start = 0
    while start < len(tokens) and _is_honorific(tokens[start]):
        start += 1
    return tokens[start:]


def _last_name_key(stripped: list[str]) -> str | None:
    """The last stripped token, if it can stand for the entry on its own."""
    if len(stripped) < 2:
        return None
    last = stripped[-1]
    if last in NAME_SUFFIXES or _is_honorific(last) or len(last) < 2:
        return None
    return last if last.replace("-", "").replace("'", "").isalpha() else None


def _alias_index(roster: Roster) -> dict[str, str]:
    """Map each *unique* alias key to the one roster name it stands for.

    Built once per parse call. A key is dropped when two entries produce it
    (as full or last-name key), and a last-name key is also dropped when any
    other entry has that token anywhere in its stripped name (D4b: a wrong
    attribution is worse than none).
    """
    producers: dict[str, set[int]] = {}
    token_owners: dict[str, set[int]] = {}
    last_name_keys: set[str] = set()
    for i, speaker in enumerate(roster.speakers):
        stripped = _stripped_tokens(_name_key(speaker.name))
        for token in stripped:
            token_owners.setdefault(token, set()).add(i)
        if not stripped:
            continue
        producers.setdefault(" ".join(stripped), set()).add(i)
        last = _last_name_key(stripped)
        if last is not None:
            producers.setdefault(last, set()).add(i)
            last_name_keys.add(last)
    index: dict[str, str] = {}
    for key, owners in producers.items():
        if len(owners) != 1:
            continue
        if key in last_name_keys and len(token_owners[key]) != 1:
            continue
        index[key] = roster.speakers[next(iter(owners))].name
    return index


def _closed_speaker(
    value: object, names: dict[str, str], aliases: dict[str, str], counters: _Counters
) -> str:
    """Map ``value`` to a roster name or ``unknown``; count and log a coercion.

    Order: exact roster match (#22), then a unique alias (#83), else coerce.
    """
    if value is None:
        return UNKNOWN
    if isinstance(value, str):
        key = _name_key(value)
        if key in ("", UNKNOWN):
            return UNKNOWN
        if key in names:
            return names[key]
        candidate = " ".join(_stripped_tokens(key))
        if candidate and candidate != UNKNOWN and candidate in aliases:
            resolved = aliases[candidate]
            _logger.info(
                "speaker_aliased",
                extra={"speaker": value[:_MAX_LOGGED_VALUE_CHARS], "resolved": resolved},
            )
            return resolved
    counters.coercions += 1
    _logger.warning(
        "speaker_coerced",
        extra={"speaker": str(value)[:_MAX_LOGGED_VALUE_CHARS]},
    )
    return UNKNOWN


def _placed(value: object, where: str, chunk: Chunk, counters: _Counters) -> float | None:
    result, clamped = _clamp(_start_sec(value, where), chunk)
    counters.clamped += clamped
    return result


def parse_chunk_analysis(raw: str, *, roster: Roster, chunk: Chunk) -> ChunkAnalysis:
    """Parse and validate one chunk's analysis against ``roster`` and ``chunk``."""
    doc = _validate(_AnalysisDoc, _extract_object(raw))
    names = {_name_key(s.name): s.name for s in roster.speakers}
    aliases = _alias_index(roster)
    counters = _Counters()
    # Collect every failing item so the repair feedback can name them all.
    errors: list[str] = []

    def capped(items: list[Any], cap: int) -> list[Any]:
        counters.dropped += max(0, len(items) - cap)
        return items[:cap]

    topics: list[Topic] = []
    for i, item in enumerate(capped(doc.topics, MAX_TOPICS)):
        where = f"topics.{i}"
        try:
            t = _validate(_TopicItem, item, where)
            start = _placed(t.start_sec, where, chunk, counters)
        except SchemaError as err:
            errors.append(str(err))
            continue
        title = t.title.strip()
        summary = t.summary.strip() if isinstance(t.summary, str) else ""
        if not title or len(title) > MAX_TITLE_CHARS or len(summary) > MAX_SUMMARY_CHARS:
            counters.dropped += 1
            continue
        topics.append(Topic(seq=len(topics), title=title, summary=summary or None, start_sec=start))

    claims: list[Claim] = []
    for i, item in enumerate(capped(doc.claims, MAX_CLAIMS)):
        where = f"claims.{i}"
        try:
            c = _validate(_ClaimItem, item, where)
            start = _placed(c.start_sec, where, chunk, counters)
        except SchemaError as err:
            errors.append(str(err))
            continue
        text = c.text.strip()
        if not text or len(text) > MAX_TEXT_CHARS:
            counters.dropped += 1
            continue
        confidence = c.confidence.strip().casefold() if isinstance(c.confidence, str) else ""
        claims.append(
            Claim(
                text=text,
                speaker=_closed_speaker(c.speaker, names, aliases, counters),
                start_sec=start,
                confidence=confidence if confidence in _CONFIDENCES else None,
                source_chunk_seq=chunk.seq,
            )
        )

    quotes: list[Quote] = []
    for i, item in enumerate(capped(doc.quotes, MAX_QUOTES)):
        where = f"quotes.{i}"
        try:
            q = _validate(_QuoteItem, item, where)
            start = _placed(q.start_sec, where, chunk, counters)
        except SchemaError as err:
            errors.append(str(err))
            continue
        text = q.text.strip()
        if not text or len(text) > MAX_TEXT_CHARS:
            counters.dropped += 1
            continue
        quotes.append(
            Quote(
                text=text,
                speaker=_closed_speaker(q.speaker, names, aliases, counters),
                start_sec=start,
                source_chunk_seq=chunk.seq,
            )
        )

    if errors:
        raise SchemaError("; ".join(errors)[:MAX_FEEDBACK_CHARS])

    if not claims and len(chunk.text.split()) >= NONTRIVIAL_CHUNK_WORDS:
        _logger.warning("empty_claims", extra={"chunk_seq": chunk.seq})

    return ChunkAnalysis(
        topics=tuple(topics),
        claims=tuple(claims),
        quotes=tuple(quotes),
        speaker_coercions=counters.coercions,
        start_sec_clamped=counters.clamped,
        items_dropped=counters.dropped,
    )


# --- tl;dr ------------------------------------------------------------------


def parse_tldr(raw: str) -> str:
    """Return the TL;DR text with surrounding code fences and whitespace removed."""
    if len(raw) > MAX_RAW_CHARS:
        raise SchemaError(f"output is longer than {MAX_RAW_CHARS} characters")
    text = _skip_fence_opener(raw.lstrip("﻿").strip())
    text = text.removesuffix("```").strip()
    if not text:
        raise SchemaError("tldr is empty")
    if len(text) > MAX_TLDR_CHARS:
        raise SchemaError(f"tldr is longer than {MAX_TLDR_CHARS} characters")
    return text


# --- repair -----------------------------------------------------------------


def parse_with_repair[T](raw: str, parse: Callable[[str], T], repair: Callable[[str], str]) -> T:
    """Run ``parse(raw)``; on ``SchemaError`` ask ``repair`` once and parse again.

    ``repair`` receives feedback naming the failing fields and returns new raw
    output. Exceptions from ``repair`` propagate unchanged. A second
    ``SchemaError`` becomes ``LLMInvalidOutputError``.
    """
    try:
        return parse(raw)
    except SchemaError as first:
        feedback = str(first)[:MAX_FEEDBACK_CHARS]
    repaired = repair(feedback)
    try:
        return parse(repaired)
    except SchemaError as second:
        message = f"LLM output failed schema validation after one repair attempt: {second}"
        raise LLMInvalidOutputError(message[:MAX_ERROR_CHARS]) from second
